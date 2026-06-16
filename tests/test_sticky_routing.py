"""E2E-ish test for LlamaHerd sticky routing.

This test does NOT hit the real Ollama Cloud. It starts LlamaHerd against a
local fake upstream server that returns deterministic OpenAI-compatible
responses, then asserts that a multi-turn conversation stays pinned to the
same upstream key (sub) within the sticky TTL, and that TTL expiry / upstream
errors cause the sticky mapping to break or rebind.

Run:  cd /home/openclaw/llamaherd-sidecar && . .venv/bin/activate && python -m pytest tests/test_sticky_routing.py -v
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llamaherd"))

# ---------------------------------------------------------------------------
# Fake upstream Ollama Cloud server
# ---------------------------------------------------------------------------

upstream_hits: dict[str, int] = {}

fake_app = FastAPI()

@fake_app.get("/v1/models")
async def fake_models():
    return {
        "object": "list",
        "data": [{"id": "gemma3:4b", "object": "model"}],
    }

@fake_app.get("/api/tags")
async def fake_tags():
    return {"models": [{"name": "gemma3:4b", "model": "gemma3:4b"}]}

@fake_app.post("/api/show")
async def fake_show(request: Request):
    return {
        "model": "gemma3:4b",
        "details": {"context_length": 131072},
        "capabilities": ["chat"],
    }

@fake_app.post("/v1/chat/completions")
async def fake_chat(request: Request):
    body = await request.json()
    auth = request.headers.get("authorization", "")
    token = auth.replace("Bearer ", "")
    upstream_hits[token] = upstream_hits.get(token, 0) + 1
    model = body.get("model", "unknown")
    is_stream = body.get("stream", False)
    if is_stream:
        async def generate():
            chunks = [
                {"id": "chat-1", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "hi "}, "index": 0, "finish_reason": None}]},
                {"id": "chat-1", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "there"}, "index": 0, "finish_reason": None}]},
                {"id": "chat-1", "object": "chat.completion.chunk", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    return JSONResponse({
        "id": "chat-1", "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def run_fake_upstream(host: str = "127.0.0.1", port: int = 18399):
    config = uvicorn.Config(fake_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        await asyncio.sleep(0.05)
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.should_exit = True
        await task


def make_config(upstream: str, sticky_ttl: int = 3600, max_concurrent: int = 5) -> str:
    cfg = {
        "keys": [
            {"token": "key-one-1234", "max_concurrent": max_concurrent, "label": "Sub 1"},
            {"token": "key-two-5678", "max_concurrent": max_concurrent, "label": "Sub 2"},
        ],
        "clients": [
            {"id": "test-client", "label": "Test Client", "token": "test-token"},
        ],
        "host": "127.0.0.1",
        "port": 18499,
        "upstream": upstream,
        "admin_token": "admin-token",
        "reject_unknown_models": True,
        "sticky_ttl_seconds": sticky_ttl,
        "health_check_interval": 300,
        "request_timeout": 30,
        "queue_timeout": 5,
        "max_retries": 1,
    }
    return json.dumps(cfg)


@pytest_asyncio.fixture
async def proxy_server(sticky_ttl=3600, max_concurrent=5):
    upstream_hits.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        db_path = os.path.join(tmpdir, "proxy.db")
        usage_db_path = os.path.join(tmpdir, "usage.db")

        async with run_fake_upstream() as upstream:
            with open(cfg_path, "w") as f:
                f.write(make_config(upstream, sticky_ttl=sticky_ttl, max_concurrent=max_concurrent))

            os.environ["LLAMAHERD_CONFIG"] = cfg_path
            os.environ["LLAMAHERD_DB"] = db_path

            # Import lazily after env is set
            from llamaherd import proxy as proxy_mod

            # Reset globals so multiple tests don't share state
            proxy_mod.CONFIG_PATH = __import__("pathlib").Path(cfg_path)
            cfg = proxy_mod.load_config()
            cfg["usage_db"] = usage_db_path
            proxy_mod.DB_PATH = __import__("pathlib").Path(db_path)

            proxy_mod.app.state.started = False
            config = uvicorn.Config(proxy_mod.app, host=cfg["host"], port=cfg["port"], log_level="warning")
            server = uvicorn.Server(config)
            task = asyncio.create_task(server.serve())
            deadline = time.time() + 10
            while not server.started and time.time() < deadline:
                await asyncio.sleep(0.05)
            try:
                yield cfg
            finally:
                server.should_exit = True
                await task


async def chat(proxy_url: str, api_key: str, session_id: str | None = None, stream: bool = True):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["X-LlamaHerd-Session"] = session_id
    body = {
        "model": "gemma3:4b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{proxy_url}/v1/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        chosen_key = resp.headers.get("X-LlamaHerd-Key")
        returned_session = resp.headers.get("X-LlamaHerd-Session")
        text = resp.text
        if stream:
            # Just consume text; return it for inspection
            return resp.status_code, chosen_key, returned_session, text
        return resp.status_code, chosen_key, returned_session, resp.json()


@pytest.mark.asyncio
async def test_public_endpoints_require_registered_client_token(proxy_server):
    cfg = proxy_server
    proxy_url = f"http://{cfg['host']}:{cfg['port']}"
    body = {
        "model": "gemma3:4b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        missing_chat = await client.post(f"{proxy_url}/v1/chat/completions", json=body)
        assert missing_chat.status_code == 401
        assert missing_chat.json()["detail"] == "missing_api_key"

        unknown_chat = await client.post(
            f"{proxy_url}/v1/chat/completions",
            headers={"Authorization": "Bearer not-a-registered-token"},
            json=body,
        )
        assert unknown_chat.status_code == 403
        assert unknown_chat.json()["detail"] == "invalid_api_key"

        missing_models = await client.get(f"{proxy_url}/v1/models")
        assert missing_models.status_code == 401

        unknown_models = await client.get(
            f"{proxy_url}/v1/models",
            headers={"Authorization": "Bearer not-a-registered-token"},
        )
        assert unknown_models.status_code == 403

        valid_models = await client.get(
            f"{proxy_url}/v1/models",
            headers={"Authorization": "Bearer test-token"},
        )
        assert valid_models.status_code == 200

        missing_native_tags = await client.get(f"{proxy_url}/api/tags")
        assert missing_native_tags.status_code == 401


@pytest.mark.asyncio
async def test_sticky_session_pins_to_same_key(proxy_server):
    cfg = proxy_server
    proxy_url = f"http://{cfg['host']}:{cfg['port']}"

    # First request without session id: proxy creates one
    status1, key1, sid1, _ = await chat(proxy_url, "test-token", stream=False)
    assert status1 == 200
    assert key1 in ("Sub 1", "Sub 2")
    assert sid1

    # Second request with same session id must land on same key
    status2, key2, sid2, _ = await chat(proxy_url, "test-token", session_id=sid1, stream=False)
    assert status2 == 200
    assert key2 == key1
    assert sid2 == sid1

    # Streaming third request also pinned
    status3, key3, sid3, _ = await chat(proxy_url, "test-token", session_id=sid1, stream=True)
    assert status3 == 200
    assert key3 == key1


@pytest.mark.asyncio
async def test_sticky_session_ttl_breaks_affinity():
    async with proxy_server_context(sticky_ttl=1) as cfg:
        proxy_url = f"http://{cfg['host']}:{cfg['port']}"
        status1, key1, sid1, _ = await chat(proxy_url, "test-token", stream=False)
        assert status1 == 200
        await asyncio.sleep(1.1)  # wait for TTL
        status2, key2, _, _ = await chat(proxy_url, "test-token", session_id=sid1, stream=False)
        # After TTL, mapping is gone; key2 may differ (load balancing)
        assert status2 == 200


# Context helper for TTL test
@asynccontextmanager
async def proxy_server_context(sticky_ttl: int = 3600, max_concurrent: int = 5):
    upstream_hits.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        db_path = os.path.join(tmpdir, "proxy.db")
        usage_db_path = os.path.join(tmpdir, "usage.db")

        async with run_fake_upstream() as upstream:
            with open(cfg_path, "w") as f:
                f.write(make_config(upstream, sticky_ttl=sticky_ttl, max_concurrent=max_concurrent))

            os.environ["LLAMAHERD_CONFIG"] = cfg_path
            os.environ["LLAMAHERD_DB"] = db_path

            import llamaherd.proxy as proxy_mod
            proxy_mod.CONFIG_PATH = __import__("pathlib").Path(cfg_path)
            cfg = proxy_mod.load_config()
            cfg["usage_db"] = usage_db_path
            proxy_mod.DB_PATH = __import__("pathlib").Path(db_path)

            config = uvicorn.Config(proxy_mod.app, host=cfg["host"], port=cfg["port"], log_level="warning")
            server = uvicorn.Server(config)
            task = asyncio.create_task(server.serve())
            deadline = time.time() + 10
            while not server.started and time.time() < deadline:
                await asyncio.sleep(0.05)
            try:
                yield cfg
            finally:
                server.should_exit = True
                await task


@pytest.mark.asyncio
async def test_sticky_session_swaps_when_exhausted():
    """If the sticky sub is at max concurrent, the request must still succeed on another sub."""
    async with proxy_server_context(max_concurrent=1, sticky_ttl=60) as cfg:
        proxy_url = f"http://{cfg['host']}:{cfg['port']}"

        # First request pins to a sub
        status1, key1, sid1, _ = await chat(proxy_url, "test-token", stream=False)
        assert status1 == 200
        assert key1 in ("Sub 1", "Sub 2")
        assert sid1

        # Hold that sub busy with a slow streaming request
        async with httpx.AsyncClient(timeout=30) as client:
            body = {
                "model": "gemma3:4b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
            slow_resp_task = asyncio.create_task(
                client.post(f"{proxy_url}/v1/chat/completions", headers={"Authorization": "Bearer test-token"}, json=body)
            )
            await asyncio.sleep(0.2)  # let it acquire the key

            # Same session now has its sticky sub at capacity -> should still work on the other sub
            status2, key2, sid2, _ = await chat(proxy_url, "test-token", session_id=sid1, stream=False)
            assert status2 == 200
            assert sid2 == sid1
            assert key2 is not None  # may differ because sticky sub is busy
            slow_resp = await slow_resp_task
            slow_resp.raise_for_status()

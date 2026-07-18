import asyncio
import contextlib
import time

import pytest

from llamaherd import proxy
from llamaherd.fallback import FallbackProvider
from llamaherd.key_manager import KeyState


class _FakeManager:
    def __init__(self):
        self.releases = 0

    async def release(self, key, tokens_out=0):
        self.releases += 1

    async def mark_429(self, key):
        pass

    async def mark_402(self, key):
        pass


class _CancellableCloseResponse:
    status_code = 200

    def __init__(self):
        self.reading = asyncio.Event()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.closed = False

    async def aiter_lines(self):
        yield '{"message":{"content":"one"},"done":false}'
        self.reading.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.close_started.set()
        await self.allow_close.wait()
        self.closed = True


class _StreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        await self.response.aclose()


class _FakeClient:
    def __init__(self, response):
        self.response = response

    def stream(self, *args, **kwargs):
        return _StreamContext(self.response)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["openai", "native", "fallback", "bridge"])
async def test_stream_closes_upstream_despite_repeated_cancellation(monkeypatch, stream_kind):
    """A disconnect plus shutdown cancellation must not interrupt socket cleanup."""
    response = _CancellableCloseResponse()
    manager = _FakeManager()
    monkeypatch.setattr(proxy, "upstream_http_client", _FakeClient(response))
    monkeypatch.setattr(proxy, "upstream_url", "https://upstream.invalid/v1")
    monkeypatch.setattr(proxy, "manager", manager)
    monkeypatch.setattr(proxy, "sticky", None)
    monkeypatch.setattr(proxy, "_record_and_broadcast", lambda *args, **kwargs: None)

    key = KeyState(token="secret-key", label="test", max_concurrent=1)
    if stream_kind == "openai":
        streaming_response = await proxy._proxy_stream(
            "client", key, "/chat/completions", {}, b"{}", "model", time.time()
        )
    elif stream_kind == "native":
        streaming_response = await proxy._proxy_ndjson_stream(
            "client", key, "/chat", {}, b"{}", "model", time.time()
        )
    elif stream_kind == "fallback":
        fallback = FallbackProvider({"provider": "fallback-test"})
        streaming_response = await proxy._proxy_fallback_stream(
            "client", fallback, "https://fallback.invalid/chat/completions", {}, b"{}",
            "model", "mapped-model", time.time(), "fb:test",
        )
    else:
        streaming_response = await proxy._proxy_bridge_stream(
            "client", key, b"{}", "model", time.time()
        )
    iterator = streaming_response.body_iterator

    first = await anext(iterator)
    assert "one" in first

    consumer = asyncio.create_task(anext(iterator))
    await response.reading.wait()
    consumer.cancel()  # downstream disconnect
    await response.close_started.wait()
    consumer.cancel()  # concurrent shutdown/disconnect cancellation during aclose()
    response.allow_close.set()

    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
        await consumer
    await asyncio.sleep(0)

    assert response.closed, "the upstream response socket was left open"
    assert manager.releases == (0 if stream_kind == "fallback" else 1)

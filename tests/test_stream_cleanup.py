import asyncio
import time

import pytest

from llamaherd import proxy
from llamaherd.fallback import FallbackProvider
from llamaherd.key_manager import KeyState


class _FakeManager:
    def __init__(self):
        self.releases = 0
        self.release_started = asyncio.Event()
        self.allow_release = asyncio.Event()
        self.allow_release.set()

    async def release(self, key, tokens_out=0):
        self.release_started.set()
        await self.allow_release.wait()
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
        self.close_count = 0
        self.close_error = None

    async def aiter_lines(self):
        yield '{"message":{"content":"one"},"done":false}'
        self.reading.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.close_started.set()
        await self.allow_close.wait()
        self.closed = True
        self.close_count += 1
        if self.close_error:
            raise self.close_error


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
@pytest.mark.parametrize("attempt", range(10))
async def test_stream_closes_upstream_despite_repeated_cancellation(monkeypatch, stream_kind, attempt):
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
    await asyncio.wait_for(response.reading.wait(), 1)
    consumer.cancel()  # downstream disconnect
    await asyncio.wait_for(response.close_started.wait(), 1)
    consumer.cancel()  # concurrent shutdown/disconnect cancellation during aclose()
    response.allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, 1)
    await asyncio.sleep(0)

    assert response.closed, "the upstream response socket was left open"
    assert response.close_count == 1
    assert manager.releases == (0 if stream_kind == "fallback" else 1)
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("llamaherd-cleanup:")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["openai", "native", "fallback", "bridge"])
async def test_cleanup_failure_does_not_replace_cancellation(monkeypatch, stream_kind):
    """A close failure must be logged without turning cancellation into stream data."""
    response = _CancellableCloseResponse()
    response.close_error = RuntimeError("close failed")
    manager = _FakeManager()
    streaming_response = await _make_stream(monkeypatch, stream_kind, response, manager)
    iterator = streaming_response.body_iterator

    assert "one" in await anext(iterator)
    consumer = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(response.reading.wait(), 1)
    consumer.cancel()
    await asyncio.wait_for(response.close_started.wait(), 1)
    response.allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, 1)
    assert response.close_count == 1
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("llamaherd-cleanup:")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["openai", "native", "bridge"])
async def test_cancellation_during_release_does_not_strand_slot(monkeypatch, stream_kind):
    response = _CancellableCloseResponse()
    response.allow_close.set()
    manager = _FakeManager()
    manager.allow_release.clear()
    records = []
    streaming_response = await _make_stream(monkeypatch, stream_kind, response, manager, records)
    iterator = streaming_response.body_iterator

    assert "one" in await anext(iterator)
    consumer = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(response.reading.wait(), 1)
    consumer.cancel()
    await asyncio.wait_for(manager.release_started.wait(), 1)
    consumer.cancel()
    manager.allow_release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, 1)
    assert manager.releases == 1
    assert len(records) == 1
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("llamaherd-cleanup:")]


@pytest.mark.asyncio
async def test_cleanup_timeout_cancels_and_reaps_task():
    finished = asyncio.Event()

    async def stuck_cleanup():
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    with pytest.raises(TimeoutError, match="did not finish"):
        await proxy._await_cleanup(stuck_cleanup(), label="stuck test", timeout=0.01)
    assert finished.is_set()
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("llamaherd-cleanup:")]


async def _make_stream(monkeypatch, stream_kind, response, manager, records=None):
    records = records if records is not None else []
    monkeypatch.setattr(proxy, "upstream_http_client", _FakeClient(response))
    monkeypatch.setattr(proxy, "upstream_url", "https://upstream.invalid/v1")
    monkeypatch.setattr(proxy, "manager", manager)
    monkeypatch.setattr(proxy, "sticky", None)
    monkeypatch.setattr(proxy, "_record_and_broadcast", lambda *args, **kwargs: records.append((args, kwargs)))
    key = KeyState(token="secret-key", label="test", max_concurrent=1)
    if stream_kind == "openai":
        return await proxy._proxy_stream(
            "client", key, "/chat/completions", {}, b"{}", "model", time.time()
        )
    if stream_kind == "native":
        return await proxy._proxy_ndjson_stream(
            "client", key, "/chat", {}, b"{}", "model", time.time()
        )
    if stream_kind == "fallback":
        fallback = FallbackProvider({"provider": "fallback-test"})
        return await proxy._proxy_fallback_stream(
            "client", fallback, "https://fallback.invalid/chat/completions", {}, b"{}",
            "model", "mapped-model", time.time(), "fb:test",
        )
    return await proxy._proxy_bridge_stream(
        "client", key, b"{}", "model", time.time()
    )

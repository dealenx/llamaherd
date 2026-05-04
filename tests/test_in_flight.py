"""Unit tests for in-flight request tracking + request_start/request_end events."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from llamaherd import proxy


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Reset in-flight tracker and stub broadcaster between tests."""
    proxy._in_flight.clear()
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    monkeypatch.setattr(proxy, "usage_db", proxy.UsageDB(str(tmp_path / "usage.db")))
    yield
    proxy._in_flight.clear()


def test_new_request_id_is_unique_hex():
    a, b = proxy._new_request_id(), proxy._new_request_id()
    assert a != b
    int(a, 16)  # would raise if not hex
    assert len(a) == 16  # 8 bytes hex


def test_request_start_populates_tracker():
    rid = "deadbeefcafebabe"
    proxy._request_start(rid, "client-x", "glm-5.1", "Sub 1", "ollama-cloud")
    assert rid in proxy._in_flight
    assert proxy._in_flight[rid]["client_id"] == "client-x"
    assert proxy._in_flight[rid]["target_provider"] == "ollama-cloud"


def test_record_and_broadcast_pops_in_flight():
    rid = "feedfacefeedface"
    proxy._request_start(rid, "client-y", "glm-5.1", "Sub 2", "ollama-cloud")
    assert rid in proxy._in_flight
    proxy._record_and_broadcast(
        "client-y", "abcd1234", "glm-5.1", 10, 5, 100, 200,
        request_id=rid, provider="ollama-cloud",
    )
    assert rid not in proxy._in_flight


def test_admin_in_flight_endpoint_returns_active_requests():
    proxy._request_start("aaa1", "client-a", "model-1", "Sub 1", "ollama-cloud")
    proxy._request_start("bbb2", "client-b", "model-2", "fb:nvidia-build", "nvidia-build")

    client = TestClient(proxy.app)
    r = client.get("/admin/in-flight?token=test-token")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    ids = {row["request_id"] for row in body["in_flight"]}
    assert ids == {"aaa1", "bbb2"}
    for row in body["in_flight"]:
        assert "elapsed_ms" in row
        assert row["elapsed_ms"] >= 0


def test_admin_in_flight_requires_token():
    client = TestClient(proxy.app)
    r = client.get("/admin/in-flight")
    assert r.status_code == 401


def test_dashboard_renders_in_flight_panel():
    """The dashboard HTML should reference in-flight elements + request_start/end events."""
    html = proxy.DASHBOARD_HTML
    assert "inflight-list" in html
    assert "request_start" in html
    assert "request_end" in html
    assert "renderInFlight" in html
    assert "loadInFlightInitial" in html
    # CSS animation keyframes
    assert "inflight-pulse" in html


def test_request_end_event_payload(monkeypatch):
    """request_end should include request_id and provider derived from in-flight entry."""
    captured: list[tuple[str, dict]] = []

    async def fake_broadcast(event_type, data):
        captured.append((event_type, data))

    monkeypatch.setattr(proxy.broadcaster, "broadcast", fake_broadcast)

    async def runner():
        proxy._request_start("xx99", "c1", "m1", "key-A", "ollama-cloud")
        proxy._record_and_broadcast(
            "c1", "abcd", "m1", 7, 3, 250, 200,
            request_id="xx99", provider="ollama-cloud",
        )
        # Allow ensure_future-scheduled broadcasts to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(runner())

    types = [t for t, _ in captured]
    assert "request_start" in types
    assert "request_end" in types
    end_payload = next(d for t, d in captured if t == "request_end")
    assert end_payload["request_id"] == "xx99"
    assert end_payload["provider"] == "ollama-cloud"
    assert end_payload["status"] == 200

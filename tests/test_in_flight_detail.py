"""Tests for in-flight detail: live token counters + sanitized headers."""
import pytest
from fastapi.testclient import TestClient

from llamaherd import proxy


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    proxy._in_flight.clear()
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    monkeypatch.setattr(proxy, "usage_db", proxy.UsageDB(str(tmp_path / "usage.db")))
    yield
    proxy._in_flight.clear()


def test_request_start_initializes_token_counters():
    proxy._request_start("rid1", "client-x", "glm-5.1", "Sub 1", "ollama-cloud")
    entry = proxy._in_flight["rid1"]
    assert entry["tokens_in"] == 0
    assert entry["tokens_out"] == 0


def test_request_start_records_path_and_sanitized_headers():
    headers = {
        "Authorization": "Bearer SECRET-TOKEN-VALUE",
        "Cookie": "session=SECRET-COOKIE",
        "User-Agent": "openclaw-cli/1.2",
        "Content-Type": "application/json",
        "X-Api-Key": "AKIA-SHOULD-NOT-LEAK",
    }
    proxy._request_start(
        "rid2", "client-y", "kimi-k2.6", "Sub 2", "ollama-cloud",
        headers=headers, path="/v1/chat/completions",
    )
    entry = proxy._in_flight["rid2"]

    assert entry["path"] == "/v1/chat/completions"
    assert "headers" in entry
    h = entry["headers"]
    assert "User-Agent" in h
    assert h["User-Agent"] == "openclaw-cli/1.2"
    assert "Content-Type" in h
    # Sensitive keys must be stripped.
    assert "Authorization" not in h
    assert "Cookie" not in h
    assert "X-Api-Key" not in h
    # Make sure no value in any field equals the secret.
    for v in h.values():
        assert "SECRET-TOKEN-VALUE" not in v
        assert "SECRET-COOKIE" not in v
        assert "AKIA-SHOULD-NOT-LEAK" not in v


def test_sanitize_headers_truncates_long_values():
    long_val = "x" * 5000
    out = proxy._sanitize_headers({"X-Trace": long_val})
    assert len(out["X-Trace"]) <= 200


def test_update_in_flight_tokens_only_increases():
    proxy._request_start("rid3", "c", "m", "k", "ollama-cloud")
    proxy._update_in_flight_tokens("rid3", tokens_in=10, tokens_out=5)
    assert proxy._in_flight["rid3"]["tokens_in"] == 10
    assert proxy._in_flight["rid3"]["tokens_out"] == 5

    # Going down (e.g. retransmit) must not roll back the visible counter.
    proxy._update_in_flight_tokens("rid3", tokens_in=3, tokens_out=2)
    assert proxy._in_flight["rid3"]["tokens_in"] == 10
    assert proxy._in_flight["rid3"]["tokens_out"] == 5

    # Going up replaces.
    proxy._update_in_flight_tokens("rid3", tokens_in=20, tokens_out=8)
    assert proxy._in_flight["rid3"]["tokens_in"] == 20
    assert proxy._in_flight["rid3"]["tokens_out"] == 8


def test_update_in_flight_tokens_no_op_for_unknown_request():
    # Should not raise, should not create a phantom entry.
    proxy._update_in_flight_tokens("missing", tokens_in=1, tokens_out=1)
    assert "missing" not in proxy._in_flight


def test_admin_in_flight_returns_token_counters_and_headers():
    proxy._request_start(
        "rid4", "client-z", "deepseek-v3.2", "fb:nvidia-build", "nvidia-build",
        headers={"User-Agent": "ua/1.0", "Authorization": "Bearer x"},
        path="/v1/chat/completions",
    )
    proxy._update_in_flight_tokens("rid4", tokens_in=12, tokens_out=34)

    client = TestClient(proxy.app)
    r = client.get("/admin/in-flight?token=test-token")
    assert r.status_code == 200
    rows = r.json()["in_flight"]
    [row] = [x for x in rows if x["request_id"] == "rid4"]
    assert row["tokens_in"] == 12
    assert row["tokens_out"] == 34
    assert row["path"] == "/v1/chat/completions"
    assert row["headers"]["User-Agent"] == "ua/1.0"
    assert "Authorization" not in row["headers"]

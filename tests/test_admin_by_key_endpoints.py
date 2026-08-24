"""Tests for /admin/usage/by-key* endpoints."""
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from llamaherd import proxy


_NOW = time.time()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Reset global state and set up a temp usage DB with check_same_thread=False."""
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    db_path = str(tmp_path / "usage.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    db = proxy.UsageDB.__new__(proxy.UsageDB)
    db.db_path = db_path
    db._conn = conn
    # Re-create tables (the __new__ skip __init__)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            ts REAL NOT NULL,
            day TEXT NOT NULL,
            client_id TEXT NOT NULL,
            upstream_key TEXT NOT NULL,
            model TEXT NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            status INTEGER NOT NULL,
            session_id TEXT DEFAULT ''
        )
    """)
    for idx_cols in [
        "client_id, day",
        "model, day",
        "day",
        "client_id, model, day",
        "session_id",
    ]:
        idx_name = f"idx_usage_{'_'.join(idx_cols.replace(' ', '').split(','))}"
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON usage ({idx_cols})")
        except (sqlite3.OperationalError, RuntimeError):
            pass
    conn.commit()
    monkeypatch.setattr(proxy, "usage_db", db)

    class _FakeKey:
        label = "Pro Sub 1"
        plan = "Pro"
        account_email = "sub1@example.com"
        token = "abcdef1234567890"

    class _FakeManager:
        keys = [_FakeKey()]
        def key_by_token_prefix(self, prefix):
            if prefix == "abcdef12":
                return _FakeKey()
            return None

    monkeypatch.setattr(proxy, "manager", _FakeManager())
    monkeypatch.setattr(proxy, "_load_openrouter_pricing", lambda: {
        "glm-5.1": {"input_per_1m": 0.50, "output_per_1m": 1.00, "openrouter_id": "z-ai/glm-5.1"},
    })
    yield


def _seed(db):
    db._conn.executemany(
        "INSERT INTO usage (ts, day, client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (_NOW, "2026-07-20", "hermes", "abcdef12", "glm-5.1", 1_000_000, 500_000, 200, 200, ""),
            (_NOW, "2026-07-21", "hermes", "abcdef12", "glm-5.1", 500_000, 200_000, 250, 200, ""),
        ],
    )
    db._conn.commit()


def test_get_by_key_endpoint():
    _seed(proxy.usage_db)
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key?days=30")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["upstream_key"] == "abcdef12"
    assert data[0]["label"] == "Pro Sub 1"
    assert data[0]["plan"] == "Pro"
    assert data[0]["requests"] == 2


def test_get_by_key_costs_endpoint():
    _seed(proxy.usage_db)
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key-costs?days=30")
    assert r.status_code == 200
    data = r.json()
    assert "keys" in data
    assert len(data["keys"]) == 1
    key = data["keys"][0]
    assert key["upstream_key"] == "abcdef12"
    assert key["label"] == "Pro Sub 1"
    assert key["total_cost_usd"] == pytest.approx(1.45, abs=0.01)
    assert len(key["models"]) == 1
    assert key["models"][0]["model"] == "glm-5.1"


def test_get_by_key_detail_endpoint():
    _seed(proxy.usage_db)
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key/abcdef12?days=30")
    assert r.status_code == 200
    data = r.json()
    assert data["upstream_key"] == "abcdef12"
    assert data["label"] == "Pro Sub 1"
    assert data["plan"] == "Pro"
    assert data["account_email"] == "sub1@example.com"
    assert len(data["models"]) == 1
    assert data["models"][0]["model"] == "glm-5.1"
    assert len(data["clients"]) == 1
    assert len(data["daily"]) == 2
    assert data["totals"]["requests"] == 2


def test_by_key_endpoint_empty_db():
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key?days=30")
    assert r.status_code == 200
    assert r.json() == []


def test_by_key_with_date_params():
    _seed(proxy.usage_db)
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key?start_date=2026-07-20&end_date=2026-07-20")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["requests"] == 1
    assert data[0]["tokens_in"] == 1_000_000


def test_by_key_costs_with_key_filter():
    _seed(proxy.usage_db)
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key-costs?days=30&key=abcdef12")
    assert r.status_code == 200
    data = r.json()
    assert len(data["keys"]) == 1
    assert data["keys"][0]["upstream_key"] == "abcdef12"


def test_by_key_costs_with_key_filter_no_match():
    _seed(proxy.usage_db)
    client = TestClient(proxy.app, headers={"Authorization": "Bearer test-token"})
    r = client.get("/admin/usage/by-key-costs?days=30&key=nonexistent")
    assert r.status_code == 200
    data = r.json()
    assert data["keys"] == []


def test_unauthorized_access():
    client = TestClient(proxy.app)
    r = client.get("/admin/usage/by-key")
    assert r.status_code == 401
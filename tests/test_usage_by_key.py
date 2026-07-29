import time

import pytest

from llamaherd import proxy


def _seed_rows(db, rows):
    db._conn.executemany(
        "INSERT INTO usage (ts, day, client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db._conn.commit()


_NOW = time.time()


def test_by_upstream_key_basic(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 100, 50, 200, 200, ""),
        (_NOW, "2026-07-20", "openclaw", "key-aaaa", "glm-5.1", 200, 80, 250, 200, ""),
        (_NOW, "2026-07-21", "hermes", "key-bbbb", "gemma3:4b", 50, 20, 100, 200, ""),
    ])
    result = db.by_upstream_key(days=30)
    assert len(result) == 2
    assert result[0]["upstream_key"] == "key-aaaa"
    assert result[0]["requests"] == 2
    assert result[0]["tokens_in"] == 300
    assert result[0]["tokens_out"] == 130
    assert result[0]["tokens_total"] == 430
    assert result[1]["upstream_key"] == "key-bbbb"
    assert result[1]["requests"] == 1


def test_by_upstream_key_excludes_none(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 100, 50, 200, 200, ""),
        (_NOW, "2026-07-20", "hermes", "none", "glm-5.1", 200, 80, 250, -1, ""),
    ])
    result = db.by_upstream_key(days=30)
    assert len(result) == 1
    assert result[0]["upstream_key"] == "key-aaaa"


def test_by_upstream_key_excludes_failed_status(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 100, 50, 200, 200, ""),
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 200, 80, 250, -1, ""),
    ])
    result = db.by_upstream_key(days=30)
    assert len(result) == 1
    assert result[0]["requests"] == 1
    assert result[0]["tokens_in"] == 100


def test_by_upstream_key_date_filter(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-10", "hermes", "key-aaaa", "glm-5.1", 100, 50, 200, 200, ""),
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 200, 80, 250, 200, ""),
    ])
    result = db.by_upstream_key(start_date="2026-07-20", end_date="2026-07-20")
    assert len(result) == 1
    assert result[0]["tokens_in"] == 200


def test_upstream_key_costs_basic(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 1_000_000, 500_000, 200, 200, ""),
        (_NOW, "2026-07-20", "hermes", "key-bbbb", "glm-5.1", 500_000, 200_000, 250, 200, ""),
    ])
    pricing = {
        "glm-5.1": {"input_per_1m": 0.50, "output_per_1m": 1.00, "openrouter_id": "z-ai/glm-5.1"},
    }
    result = db.upstream_key_costs(pricing, days=30)
    assert result["total_cost_usd"] == pytest.approx(1.45, abs=0.01)
    keys = result["keys"]
    assert len(keys) == 2
    assert keys[0]["upstream_key"] == "key-aaaa"
    assert keys[0]["total_cost_usd"] == pytest.approx(1.0, abs=0.01)
    assert keys[0]["tokens_in"] == 1_000_000
    assert keys[0]["tokens_out"] == 500_000
    assert len(keys[0]["models"]) == 1
    assert keys[0]["models"][0]["model"] == "glm-5.1"
    assert keys[0]["models"][0]["input_per_1m"] == 0.50


def test_upstream_key_costs_unpriced(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "unknown-model", 100, 50, 200, 200, ""),
    ])
    result = db.upstream_key_costs({}, days=30)
    assert result["total_cost_usd"] == 0
    assert "unknown-model" in result["unpriced_models"]
    assert len(result["keys"]) == 1
    assert result["keys"][0]["total_cost_usd"] == 0


def test_upstream_key_costs_with_labels(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 100, 50, 200, 200, ""),
    ])
    pricing = {"glm-5.1": {"input_per_1m": 0.50, "output_per_1m": 1.00}}
    labels = {"key-aaaa": {"label": "Pro Sub 1", "plan": "Pro"}}
    result = db.upstream_key_costs(pricing, days=30, key_labels=labels)
    assert result["keys"][0]["label"] == "Pro Sub 1"
    assert result["keys"][0]["plan"] == "Pro"


def test_upstream_key_detail_basic(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    _seed_rows(db, [
        (_NOW, "2026-07-20", "hermes", "key-aaaa", "glm-5.1", 100, 50, 200, 200, ""),
        (_NOW, "2026-07-20", "openclaw", "key-aaaa", "gemma3:4b", 200, 80, 250, 200, ""),
        (_NOW, "2026-07-21", "hermes", "key-aaaa", "glm-5.1", 300, 100, 300, 429, ""),
    ])
    detail = db.upstream_key_detail("key-aaaa", days=30)
    assert detail["upstream_key"] == "key-aaaa"
    assert len(detail["models"]) == 2
    assert detail["models"][0]["model"] == "glm-5.1"
    assert detail["models"][0]["tokens_in"] == 400
    assert detail["models"][0]["tokens_out"] == 150
    assert len(detail["clients"]) == 2
    assert len(detail["daily"]) == 2
    assert detail["daily"][0]["day"] == "2026-07-21"
    status_map = {s["status"]: s["count"] for s in detail["status_counts"]}
    assert status_map.get(200) == 2
    assert status_map.get(429) == 1
    assert detail["totals"]["requests"] == 3
    assert detail["totals"]["tokens_in"] == 600
    assert detail["totals"]["tokens_out"] == 230


def test_upstream_key_detail_empty_key(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    detail = db.upstream_key_detail("key-zzzz", days=30)
    assert detail["upstream_key"] == "key-zzzz"
    assert detail["models"] == []
    assert detail["clients"] == []
    assert detail["daily"] == []
    assert detail["totals"]["requests"] == 0
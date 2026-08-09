import asyncio
import json
import shutil
import subprocess

import pytest

from llamaherd import proxy


def test_recent_calls_filters_by_client_and_model(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    rows = [
        (1000.0, "2026-05-04", "hermes", "key-a", "glm-5.1", 10, 5, 100, 200, "sess-a"),
        (1001.0, "2026-05-04", "openclaw", "key-a", "glm-5.1", 20, 6, 120, 200, "sess-b"),
        (1002.0, "2026-05-04", "hermes", "key-b", "gemma3:4b", 30, 7, 130, 200, ""),
        (1003.0, "2026-05-03", "hermes", "key-b", "glm-5.1", 40, 8, 140, 200, "sess-c"),
    ]
    db._conn.executemany("INSERT INTO usage (ts, day, client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db._conn.commit()

    hermes_glm = db.recent_calls(
        limit=10,
        start_date="2026-05-04",
        end_date="2026-05-04",
        client_id="hermes",
        model="glm-5.1",
    )
    assert len(hermes_glm) == 1
    assert hermes_glm[0]["client_id"] == "hermes"
    assert hermes_glm[0]["model"] == "glm-5.1"
    assert hermes_glm[0]["tokens_total"] == 15

    openclaw = db.recent_calls(limit=10, client_id="openclaw")
    assert [row["client_id"] for row in openclaw] == ["openclaw"]

    gemma = db.recent_calls(limit=10, model="gemma3:4b")
    assert [row["model"] for row in gemma] == ["gemma3:4b"]


def test_totals_include_real_latency_and_error_rate(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    rows = [
        (1000.0, "2026-05-04", "hermes", "key-a", "glm-5.1", 10, 5, 100, 200, "sess-a"),
        (1001.0, "2026-05-04", "hermes", "key-a", "glm-5.1", 20, 6, 200, 429, "sess-b"),
        (1002.0, "2026-05-05", "openclaw", "key-b", "gemma3:4b", 30, 7, 300, -1, "sess-c"),
    ]
    db._conn.executemany(
        "INSERT INTO usage (ts, day, client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db._conn.commit()

    totals = db.totals()
    assert totals == {
        "total_calls": 3,
        "total_tokens_in": 60,
        "total_tokens_out": 18,
        "total_tokens": 78,
        "avg_latency_ms": 200.0,
        "error_rate_pct": 66.7,
    }

    filtered = db.totals(start_date="2026-05-04", end_date="2026-05-04")
    assert filtered["avg_latency_ms"] == 150.0
    assert filtered["error_rate_pct"] == 50.0


def test_empty_totals_include_zero_operational_metrics(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))

    assert db.totals() == {
        "total_calls": 0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_tokens": 0,
        "avg_latency_ms": None,
        "error_rate_pct": None,
    }


def test_admin_totals_without_usage_db_keeps_unknown_rates_unknown(monkeypatch):
    monkeypatch.setattr(proxy, "usage_db", None)

    totals = asyncio.run(proxy.admin_totals())

    assert totals["total_calls"] == 0
    assert totals["avg_latency_ms"] is None
    assert totals["error_rate_pct"] is None


def test_dashboard_script_has_no_five_second_polling_and_valid_syntax(tmp_path):
    html = proxy.DASHBOARD_PATH.read_text()
    assert "EventSource" in html
    assert "/admin/events" in html
    assert "setInterval" in html  # allowed for relative refresh labels
    assert "5000" not in html
    assert "schedulePeriodRefresh" in html
    assert "period-select" in html
    assert "last_month" in html
    assert "function renderKpis(" in html
    assert "function accountHealth(" in html
    assert "function sortAccountsStable(" in html
    assert "sortAccountsByUrgency" not in html
    assert "account-health-summary" in html
    assert 'id="totals"' in html
    assert "recorded-call error rate" in html

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    start = html.index("<script>") + len("<script>")
    end = html.index("</script>", start)
    script = html[start:end]
    script_path = tmp_path / "dashboard.js"
    script_path.write_text(script)

    result = subprocess.run([node, "--check", str(script_path)], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr

    account_source = script[script.index("function accountHealth("):script.index("\n\nfunction renderKeyStatus(")]
    account_probe = account_source + """
const cases = {
  capacity: accountHealth({in_flight:10,max_concurrent:10,period_remaining_pct:5,session_usage_pct:-1,weekly_usage_pct:-1}),
  billing: accountHealth({in_flight:0,max_concurrent:10,period_remaining_pct:5,session_usage_pct:10,weekly_usage_pct:10}),
  unknown: accountHealth({in_flight:0,max_concurrent:10,period_remaining_pct:50,session_usage_pct:null,weekly_usage_pct:null}),
  healthy: accountHealth({in_flight:0,max_concurrent:10,period_remaining_pct:50,session_usage_pct:10,weekly_usage_pct:10})
};
const keys = [
  {label:'z-sub', exhausted:false, suspended:false, max_concurrent:15, in_flight:14, session_usage_pct:10, weekly_usage_pct:10, period_remaining_pct:50},
  {label:'a-sub', exhausted:true, suspended:false, max_concurrent:15, in_flight:0, session_usage_pct:10, weekly_usage_pct:10, period_remaining_pct:50},
  {label:'m-sub', exhausted:false, suspended:false, max_concurrent:15, in_flight:0, session_usage_pct:10, weekly_usage_pct:10, period_remaining_pct:50},
];
const orderBusy = sortAccountsStable(keys).map(k => k.label);
// Capacity boundary: z at max_concurrent would be "At capacity" attention —
// order must still be pure label, not attention-grouped.
keys[0].in_flight = 15;
keys[2].in_flight = 0;
const orderAtCap = sortAccountsStable(keys).map(k => k.label);
keys[0].in_flight = 0;
keys[2].in_flight = 7;
const orderIdle = sortAccountsStable(keys).map(k => k.label);
console.log(JSON.stringify({cases, orderBusy, orderAtCap, orderIdle}));
"""
    result = subprocess.run([node, "-e", account_probe], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    health = payload["cases"]
    assert {name: value["label"] for name, value in health.items()} == {
        "capacity": "At capacity",
        "billing": "Low billing",
        "unknown": "Telemetry unknown",
        "healthy": "Healthy",
    }
    assert all(health[name]["attention"] for name in ("capacity", "billing", "unknown"))
    assert not health["healthy"]["attention"]
    # Pure static label order — unaffected by exhausted/capacity/in_flight churn.
    assert payload["orderBusy"] == ["a-sub", "m-sub", "z-sub"]
    assert payload["orderAtCap"] == ["a-sub", "m-sub", "z-sub"]
    assert payload["orderIdle"] == ["a-sub", "m-sub", "z-sub"]
    assert payload["orderBusy"] == payload["orderIdle"] == payload["orderAtCap"]

    fmt_source = script[script.index("function fmt(n)"):script.index("\nfunction fmtTs(")]
    latency_source = script[script.index("function fmtLatency("):script.index("\n\nfunction pctBarWithElapsed(")]
    kpi_source = script[script.index("function renderKpis("):script.index("\n\nconst adminHeaders")]
    kpi_probe = fmt_source + latency_source + kpi_source + """
const elements = {};
const document = {getElementById: id => elements[id] ||= {textContent:''}};
function getDateRange() { return {label:'Today'}; }
renderKpis({total_calls:0,total_tokens_in:0,total_tokens_out:0,total_tokens:0,avg_latency_ms:null,error_rate_pct:null}, 'Empty');
console.log(JSON.stringify(Object.fromEntries(Object.entries(elements).map(([k,v]) => [k,v.textContent]))));
"""
    result = subprocess.run([node, "-e", kpi_probe], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    kpis = json.loads(result.stdout)
    assert kpis["kpi-total-calls"] == "0"
    assert kpis["kpi-latency"] == "—"
    assert kpis["kpi-error-rate"] == "—"

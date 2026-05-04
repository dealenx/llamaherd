import shutil
import subprocess

import pytest

from llamaherd import proxy


def test_recent_calls_filters_by_client_and_model(tmp_path):
    db = proxy.UsageDB(str(tmp_path / "usage.db"))
    rows = [
        (1000.0, "2026-05-04", "hermes", "key-a", "glm-5.1", 10, 5, 100, 200),
        (1001.0, "2026-05-04", "openclaw", "key-a", "glm-5.1", 20, 6, 120, 200),
        (1002.0, "2026-05-04", "hermes", "key-b", "gemma3:4b", 30, 7, 130, 200),
        (1003.0, "2026-05-03", "hermes", "key-b", "glm-5.1", 40, 8, 140, 200),
    ]
    db._conn.executemany("INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
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


def test_dashboard_script_has_no_five_second_polling_and_valid_syntax(tmp_path):
    html = proxy.DASHBOARD_HTML
    assert "EventSource" in html
    assert "/admin/events" in html
    assert "setInterval" in html  # allowed for relative refresh labels
    assert "5000" not in html
    assert "schedulePeriodRefresh" in html
    assert "period-select" in html
    assert "last_month" in html

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

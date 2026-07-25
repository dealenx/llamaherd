from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "llamaherd" / "static"


class _BrandingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.favicon = None
        self.mascot = None
        self.classes = set()
        self.id_counts = {}
        self.in_account_rail = False
        self.key_status_in_account_rail = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        self.classes.update(classes)
        if attrs.get("id"):
            element_id = attrs["id"]
            self.id_counts[element_id] = self.id_counts.get(element_id, 0) + 1
        if tag == "aside" and "account-health-rail" in classes:
            self.in_account_rail = True
        if attrs.get("id") == "key-status" and self.in_account_rail:
            self.key_status_in_account_rail = True
        if tag == "link" and attrs.get("rel") == "icon":
            self.favicon = attrs
        elif tag == "img" and "dash-logo" in classes:
            self.mascot = attrs

    def handle_endtag(self, tag):
        if tag == "aside" and self.in_account_rail:
            self.in_account_rail = False


def test_dashboard_uses_pixel_mascot_and_compact_operations_layout():
    html = (STATIC / "dashboard.html").read_text()
    parser = _BrandingParser()
    parser.feed(html)

    assert parser.favicon is not None
    assert parser.favicon.get("href") == "/static/llamaherd-logo.png"
    assert parser.favicon.get("type") == "image/png"
    assert parser.mascot is not None
    assert parser.mascot.get("src") == "/static/llamaherd-logo.png"
    assert parser.mascot.get("alt") == "LlamaHerd pixel llama mascot"
    assert "operations-header" in parser.classes
    assert "header-art" in parser.classes
    assert "kpi-strip" in parser.classes
    assert "dashboard-layout" in parser.classes
    assert "dashboard-main" in parser.classes
    assert "account-health-rail" in parser.classes
    assert parser.key_status_in_account_rail
    assert parser.id_counts["key-status"] == 1
    assert parser.id_counts["kpi-total-calls"] == 1
    assert parser.id_counts["kpi-tokens-in"] == 1
    assert parser.id_counts["kpi-tokens-out"] == 1
    assert parser.id_counts["kpi-total-tokens"] == 1
    assert parser.id_counts["kpi-in-flight"] == 1
    assert parser.id_counts["kpi-latency"] == 1
    assert 'class="tab" data-tab="subs">Accounts</' in html
    assert "url('/static/llamaherd-hero.png')" in html
    assert "max-width: 1920px" in html
    assert "#fallback-control { margin-left: 0 !important; width: 100%; flex-wrap: wrap; }" in html
    assert '.fallback-priority { width: 100%; min-width: 0; flex: 0 0 100%; }' in html
    assert '<label for="fb-priority"' in html
    assert "#fb-priority { min-width: 0; max-width: 100%; flex: 1; box-sizing: border-box; }" in html

    assert (STATIC / "llamaherd-logo.png").stat().st_size > 0
    assert (STATIC / "llamaherd-hero.png").stat().st_size > 0

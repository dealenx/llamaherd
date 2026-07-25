from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "llamaherd" / "static"


class _BrandingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.favicon = None
        self.mascot = None
        self.has_hero = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "link" and attrs.get("rel") == "icon":
            self.favicon = attrs
        elif tag == "img" and "dash-logo" in classes:
            self.mascot = attrs
        elif tag == "section" and "brand-hero" in classes:
            self.has_hero = True


def test_dashboard_uses_pixel_mascot_and_visible_hero():
    html = (STATIC / "dashboard.html").read_text()
    parser = _BrandingParser()
    parser.feed(html)

    assert parser.favicon is not None
    assert parser.favicon.get("href") == "/static/llamaherd-logo.png"
    assert parser.favicon.get("type") == "image/png"
    assert parser.mascot is not None
    assert parser.mascot.get("src") == "/static/llamaherd-logo.png"
    assert parser.mascot.get("alt") == "LlamaHerd pixel llama mascot"
    assert parser.has_hero
    assert "url('/static/llamaherd-hero.png')" in html

    assert (STATIC / "llamaherd-logo.png").stat().st_size > 0
    assert (STATIC / "llamaherd-hero.png").stat().st_size > 0

from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "llamaherd" / "static"


def test_dashboard_uses_pixel_mascot_and_visible_hero():
    html = (STATIC / "dashboard.html").read_text()
    soup = BeautifulSoup(html, "html.parser")

    favicon = soup.find("link", rel="icon")
    mascot = soup.select_one(".brand-hero .dash-logo")
    hero = soup.select_one("section.brand-hero")

    assert favicon is not None
    assert favicon.get("href") == "/static/llamaherd-logo.png"
    assert favicon.get("type") == "image/png"
    assert mascot is not None
    assert mascot.get("src") == "/static/llamaherd-logo.png"
    assert mascot.get("alt") == "LlamaHerd pixel llama mascot"
    assert hero is not None
    assert "url('/static/llamaherd-hero.png')" in html

    assert (STATIC / "llamaherd-logo.png").stat().st_size > 0
    assert (STATIC / "llamaherd-hero.png").stat().st_size > 0

import sys
from types import SimpleNamespace

from llamaherd.scraper import UsageScraper


class _FakeCookies:
    def set(self, *args, **kwargs):
        pass


class _FakeResponse:
    text = "Just a moment"

    def raise_for_status(self):
        pass


class _FakeCloudScraper:
    def __init__(self):
        self.cookies = _FakeCookies()
        self.close_count = 0

    def get(self, *args, **kwargs):
        return _FakeResponse()

    def close(self):
        self.close_count += 1


def test_scrape_usage_closes_cloudscraper_on_early_return(monkeypatch):
    cloud_scraper = _FakeCloudScraper()
    monkeypatch.setitem(
        sys.modules,
        "cloudscraper",
        SimpleNamespace(create_scraper=lambda: cloud_scraper),
    )
    monkeypatch.setitem(
        sys.modules,
        "bs4",
        SimpleNamespace(BeautifulSoup=lambda *args, **kwargs: None),
    )
    usage_scraper = UsageScraper([
        {"label": "sub-1", "cookies": {"secure_session": "cookie"}},
    ])
    key = SimpleNamespace(label="sub-1")

    assert usage_scraper.scrape_usage(key) is None
    assert cloud_scraper.close_count == 1

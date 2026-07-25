import asyncio
import sys
from types import SimpleNamespace

import pytest

from llamaherd import proxy
from llamaherd.scraper import ActivityUsageRefresher, UsageScraper


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


@pytest.mark.asyncio
async def test_activity_refresh_scrapes_only_active_account_once():
    keys = [
        SimpleNamespace(label="sub-1"),
        SimpleNamespace(label="sub-2"),
    ]
    calls = []
    updates = []

    scraper = UsageScraper([
        {"label": key.label, "cookies": {"secure_session": "cookie"}}
        for key in keys
    ])

    def scrape_usage(key):
        calls.append(key.label)
        return {
            "session_usage_pct": 12.5,
            "session_resets_at": "session-reset",
            "weekly_usage_pct": 34.5,
            "weekly_resets_at": "weekly-reset",
            "session_models": {"glm": {}},
            "weekly_models": {"glm": {}},
        }

    scraper.scrape_usage = scrape_usage

    async def on_update():
        updates.append(True)

    refresher = ActivityUsageRefresher(
        scraper,
        debounce_seconds=0,
        min_interval_seconds=0,
        on_update=on_update,
    )
    refresher.schedule(keys[0])
    refresher.schedule(keys[0])
    await asyncio.gather(*list(refresher._pending.values()))

    assert calls == ["sub-1"]
    assert updates == [True]
    assert keys[0].session_usage_pct == 12.5
    assert keys[0].weekly_usage_pct == 34.5
    assert not hasattr(keys[1], "session_usage_pct")
    assert refresher._pending == {}


@pytest.mark.asyncio
async def test_activity_refresh_ignores_account_without_cookie():
    scraper = UsageScraper([])
    scraper.scrape_usage = lambda key: pytest.fail("unexpected scrape")
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=0, min_interval_seconds=0)

    refresher.schedule(SimpleNamespace(label="sub-1"))
    await asyncio.sleep(0)

    assert refresher._pending == {}


@pytest.mark.asyncio
async def test_activity_refresh_close_cancels_delayed_tasks():
    scraper = UsageScraper([
        {"label": "sub-1", "cookies": {"secure_session": "cookie"}},
    ])
    scraper.scrape_usage = lambda key: pytest.fail("unexpected scrape")
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=60, min_interval_seconds=0)
    refresher.schedule(SimpleNamespace(label="sub-1"))

    await refresher.close()

    assert refresher._pending == {}
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("llamaherd-usage-refresh:")]


@pytest.mark.asyncio
async def test_completed_ollama_request_schedules_its_account_only(monkeypatch):
    active_key = SimpleNamespace(label="sub-1", token="abcdefgh-token")
    idle_key = SimpleNamespace(label="sub-2", token="ijklmnop-token")
    scheduled = []
    refresher = SimpleNamespace(schedule=lambda key: scheduled.append(key.label))
    manager = SimpleNamespace(keys=[active_key, idle_key], status=lambda: [])
    monkeypatch.setattr(proxy, "usage_db", None)
    monkeypatch.setattr(proxy, "usage_refresher", refresher)
    monkeypatch.setattr(proxy, "manager", manager)

    proxy._record_and_broadcast(
        "client", "abcdefgh", "glm", 1, 1, 10, 200,
        provider="ollama-cloud",
    )
    proxy._record_and_broadcast(
        "client", "fallback", "glm", 1, 1, 10, 200,
        provider="openrouter",
    )
    await asyncio.sleep(0)

    assert scheduled == ["sub-1"]

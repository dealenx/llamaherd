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
        SimpleNamespace(label="sub-1", token="token-1"),
        SimpleNamespace(label="sub-2", token="token-2"),
    ]
    calls = []
    updates = []

    scraper = UsageScraper([
        {"label": key.label, "cookies": {"secure_session": "cookie"}}
        for key in keys
    ])

    def scrape_usage(key, *, cookies=None):
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
    scraper.scrape_usage = lambda key, **kwargs: pytest.fail("unexpected scrape")
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=0, min_interval_seconds=0)

    refresher.schedule(SimpleNamespace(label="sub-1", token="token-1"))
    await asyncio.sleep(0)

    assert refresher._pending == {}


@pytest.mark.asyncio
async def test_activity_refresh_close_cancels_delayed_tasks():
    scraper = UsageScraper([
        {"label": "sub-1", "cookies": {"secure_session": "cookie"}},
    ])
    scraper.scrape_usage = lambda key, **kwargs: pytest.fail("unexpected scrape")
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=60, min_interval_seconds=0)
    refresher.schedule(SimpleNamespace(label="sub-1", token="token-1"))

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


@pytest.mark.asyncio
async def test_account_rename_keeps_identity_and_cooldown(monkeypatch):
    key = SimpleNamespace(label="old-label", token="stable-token")
    scraper = UsageScraper([
        {"label": "old-label", "cookies": {"secure_session": "cookie"}},
    ])
    calls = []
    delays = []

    def scrape_usage(key, *, cookies=None):
        assert cookies is not None
        calls.append((key.token, cookies["secure_session"]))
        return None

    async def record_sleep(delay):
        delays.append(delay)

    scraper.scrape_usage = scrape_usage
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=0, min_interval_seconds=300)
    refresher.mark_scraped([key])
    key.label = "new-label"
    scraper.cookie_map["new-label"] = scraper.cookie_map.pop("old-label")
    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    refresher.schedule(key)
    await asyncio.gather(*list(refresher._pending.values()))

    assert len(delays) == 1
    assert delays[0] > 299
    assert calls == [("stable-token", "cookie")]
    assert refresher._pending == {}


@pytest.mark.asyncio
async def test_account_rename_does_not_strand_pending_task():
    key = SimpleNamespace(label="old-label", token="stable-token")
    scraper = UsageScraper([
        {"label": "old-label", "cookies": {"secure_session": "cookie"}},
    ])
    scraper.scrape_usage = lambda key, **kwargs: None
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=0, min_interval_seconds=0)
    refresher.schedule(key)
    key.label = "new-label"
    await asyncio.gather(*list(refresher._pending.values()))

    assert refresher._pending == {}


@pytest.mark.asyncio
async def test_deleted_account_refresh_is_cancelled_before_label_reuse():
    old_key = SimpleNamespace(label="shared-label", token="old-token")
    scraper = UsageScraper([
        {"label": "shared-label", "cookies": {"secure_session": "old-cookie"}},
    ])
    calls = []
    scraper.scrape_usage = lambda key, **kwargs: calls.append(key.token)
    refresher = ActivityUsageRefresher(scraper, debounce_seconds=60, min_interval_seconds=0)
    refresher.schedule(old_key)

    refresher.cancel(old_key)
    scraper.cookie_map["shared-label"] = {"secure_session": "new-cookie"}
    await asyncio.gather(*list(refresher._pending.values()), return_exceptions=True)

    assert calls == []
    assert refresher._pending == {}

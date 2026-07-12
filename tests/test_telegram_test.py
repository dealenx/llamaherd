"""Tests for the /admin/telegram-test endpoint.

Reviewer #7 on PR #2: the endpoint existed but had no test coverage. These tests
cover the three relevant paths:
  1. Telegram disabled (env vars missing) — returns an error, no HTTP send.
  2. Telegram enabled, success path — sends and returns {"sent": true}.
  3. Telegram enabled, non-200 from Telegram API — returns {"sent": false}.
"""

import pytest
from fastapi.testclient import TestClient

from llamaherd import proxy


@pytest.fixture(autouse=True)
def _auth_state(monkeypatch):
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    proxy._admin_sessions.clear()
    yield
    proxy._admin_sessions.clear()


def _admin_headers():
    return {"Authorization": "Bearer test-token"}


def _install_notifier(monkeypatch, *, enabled: bool, send_return: bool = True,
                      post_status: int = 200):
    """Wire a controllable TelegramNotifier into proxy module state."""
    class FakeNotifier:
        def __init__(self):
            self.enabled = enabled
            self.chat_id = "12345" if enabled else None
            self.topic_id = None
            self.interval = 60
            self._client = object() if enabled else None

        async def send_usage_notification(self, _manager):
            if not enabled:
                return False
            return send_return

    fake = FakeNotifier()

    async def fake_send(text):
        # Mimic the real notifier.send: only used if enabled and a client exists.
        if not enabled:
            return False
        # Simulate a Telegram API call result.
        fake._last_status = post_status
        return post_status == 200

    fake.send = fake_send

    monkeypatch.setattr(proxy, "telegram_notifier", fake)
    return fake


def test_telegram_test_returns_error_when_notifier_not_initialized(monkeypatch):
    monkeypatch.setattr(proxy, "telegram_notifier", None)
    monkeypatch.setattr(proxy, "manager", object())
    client = TestClient(proxy.app)
    r = client.post("/admin/telegram-test", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "not initialized" in body["error"]


def test_telegram_test_returns_error_when_disabled_via_env(monkeypatch):
    _install_notifier(monkeypatch, enabled=False)
    monkeypatch.setattr(proxy, "manager", object())
    client = TestClient(proxy.app)
    r = client.post("/admin/telegram-test", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "not configured" in body["error"].lower()


def test_telegram_test_sends_when_enabled_and_returns_sent_true(monkeypatch):
    _install_notifier(monkeypatch, enabled=True, send_return=True)
    monkeypatch.setattr(proxy, "manager", object())
    client = TestClient(proxy.app)
    r = client.post("/admin/telegram-test", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] is True
    assert body["chat_id"] == "12345"
    assert body["topic_id"] is None
    assert body["interval"] == 60


def test_telegram_test_returns_sent_false_on_send_failure(monkeypatch):
    _install_notifier(monkeypatch, enabled=True, send_return=False)
    monkeypatch.setattr(proxy, "manager", object())
    client = TestClient(proxy.app)
    r = client.post("/admin/telegram-test", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] is False


def test_telegram_test_requires_admin_auth():
    """Unauthenticated POST must be rejected (defence-in-depth regression test)."""
    client = TestClient(proxy.app)
    r = client.post("/admin/telegram-test")
    assert r.status_code == 401

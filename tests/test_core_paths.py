import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llamaherd import cli, proxy
from llamaherd.db import ClientRegistry
from llamaherd.key_manager import KeyManager, StickySessionManager
from llamaherd.key_registry import KeyRegistry
from llamaherd.usage_db import UsageDB


def make_request(body=b"", token="client-token", path="/v1/chat/completions"):
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    headers = []
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    return Request(scope, receive)


def test_client_registry_crud_resolve_and_regenerate(tmp_path):
    registry = ClientRegistry(str(tmp_path / "clients.db"))
    created = registry.create("agent", "Agent", token="agent-token", rpm_limit=3)
    assert registry.resolve("agent-token")["id"] == "agent"
    assert registry.resolve("agent")["token"] == "agent-token"
    assert created["rpm_limit"] == 3

    updated = registry.update("agent", label="Agent 2", daily_request_limit=10)
    assert updated["label"] == "Agent 2"
    assert updated["daily_request_limit"] == 10
    regenerated = registry.regenerate_token("agent")
    assert regenerated["token"] != "agent-token"
    assert registry.resolve("agent-token") is None
    assert registry.delete("agent") is True
    assert registry.resolve(regenerated["token"]) is None


def test_key_registry_crud_and_cookies(tmp_path):
    registry = KeyRegistry(str(tmp_path / "keys.db"))
    registry.add("upstream-token", "Sub A", max_concurrent=4, cycle_day=7)
    assert registry.get_by_token("upstream-token")["label"] == "Sub A"
    updated = registry.update("upstream-token", label="Sub B", max_concurrent=8)
    assert updated["label"] == "Sub B"
    assert updated["max_concurrent"] == 8
    cookies = registry.update_cookies("upstream-token", {"secure_session": "cookie"})
    assert cookies["cookies"]["secure_session"] == "cookie"
    assert registry.remove("upstream-token") is True
    assert registry.get_by_token("upstream-token") is None


def test_resolve_client_accepts_registered_token_and_rejects_unknown(monkeypatch, tmp_path):
    registry = ClientRegistry(str(tmp_path / "clients.db"))
    registry.create("known", "Known", token="known-token")
    monkeypatch.setattr(proxy, "client_registry", registry)
    assert proxy._resolve_client(make_request(token="known-token"))["id"] == "known"
    with pytest.raises(HTTPException) as exc:
        proxy._resolve_client(make_request(token="unknown-token"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [proxy._proxy_request, proxy._proxy_ndjson_request])
async def test_proxy_handlers_reject_malformed_json(monkeypatch, tmp_path, handler):
    registry = ClientRegistry(str(tmp_path / "clients.db"))
    registry.create("client", "Client", token="client-token")
    monkeypatch.setattr(proxy, "client_registry", registry)
    response = await handler(make_request(body=b"{not-json"), "/chat")
    assert response.status_code == 400
    assert json.loads(response.body) == {"error": "Invalid JSON request body"}


@pytest.mark.asyncio
async def test_rate_limits_enforce_rpm_and_daily_requests(monkeypatch, tmp_path):
    proxy._rpm_tracker.clear()
    request = make_request()
    client = {"id": "limited", "rpm_limit": 1}
    assert await proxy._check_rate_limit(request, client) is None
    limited = await proxy._check_rate_limit(request, client)
    assert limited.status_code == 429
    assert json.loads(limited.body)["limit_type"] == "rpm"

    usage = UsageDB(str(tmp_path / "usage.db"))
    usage.record("daily", "key", "model", 1, 1, 1, 200)
    monkeypatch.setattr(proxy, "usage_db", usage)
    daily = await proxy._check_rate_limit(
        request,
        {"id": "daily", "daily_request_limit": 1},
    )
    assert daily.status_code == 429
    assert json.loads(daily.body)["limit_type"] == "daily_requests"


@pytest.mark.asyncio
async def test_proxy_request_retries_429_on_another_key(monkeypatch, tmp_path):
    client_registry = ClientRegistry(str(tmp_path / "clients.db"))
    client_registry.create("client", "Client", token="client-token")
    manager = KeyManager([
        {"token": "key-one", "label": "One"},
        {"token": "key-two", "label": "Two"},
    ])

    class FakeRegistry:
        models = {"test-model": ["key-one", "key-two"]}

        @staticmethod
        def get_preferred_key(model):
            return None

    class FakeUpstream:
        def __init__(self):
            self.tokens = []

        async def post(self, url, content, headers):
            self.tokens.append(headers["Authorization"])
            status = 429 if len(self.tokens) == 1 else 200
            return httpx.Response(status, json={"usage": {"prompt_tokens": 2, "completion_tokens": 3}})

    upstream = FakeUpstream()
    monkeypatch.setattr(proxy, "client_registry", client_registry)
    monkeypatch.setattr(proxy, "manager", manager)
    monkeypatch.setattr(proxy, "registry", FakeRegistry())
    monkeypatch.setattr(proxy, "sticky", StickySessionManager())
    monkeypatch.setattr(proxy, "usage_db", UsageDB(str(tmp_path / "usage.db")))
    monkeypatch.setattr(proxy, "fallback_provider", None)
    monkeypatch.setattr(proxy, "model_alias_manager", None)
    monkeypatch.setattr(proxy, "upstream_http_client", upstream)
    monkeypatch.setattr(proxy, "upstream_url", "https://upstream.test/v1")
    monkeypatch.setattr(proxy, "reject_unknown_models", False)
    monkeypatch.setattr(proxy, "max_retries", 2)

    body = json.dumps({"model": "test-model", "stream": False}).encode()
    response = await proxy._proxy_request(make_request(body=body), "/chat/completions")
    assert response.status_code == 200
    assert upstream.tokens == ["Bearer key-one", "Bearer key-two"]
    assert manager.keys[0].total_429s == 1


def test_cli_create_and_list_commands_are_structured(monkeypatch, capsys):
    calls = []

    def fake_api(args, method, path, json_body=None, params=None):
        calls.append((method, path, json_body))
        return {"id": "agent", "token": "agent-token"}

    monkeypatch.setattr(cli, "_api", fake_api)
    args = SimpleNamespace(
        client_id="agent",
        label="Agent",
        notes=None,
        api_token="agent-token",
        daily_token_limit=None,
        daily_request_limit=None,
        rpm_limit=5,
        format="json",
    )
    cli.cmd_clients_create(args)
    output = json.loads(capsys.readouterr().out)
    assert output["id"] == "agent"
    assert calls == [("POST", "/admin/clients", {
        "id": "agent", "label": "Agent", "token": "agent-token", "rpm_limit": 5,
    })]

    monkeypatch.setattr(cli, "_api", lambda *a, **k: [{"key_id": "stable", "label": "Sub"}])
    cli.cmd_keys_list(SimpleNamespace(format="json"))
    assert json.loads(capsys.readouterr().out)[0]["key_id"] == "stable"

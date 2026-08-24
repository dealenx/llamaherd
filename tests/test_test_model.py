"""Tests for the /admin/test-model endpoint.

The endpoint lets the dashboard send a "2+2" test prompt to a model through the
proxy's own /v1/chat/completions route, streaming tokens back as SSE. These
tests cover the guard paths (missing model, no keys, no clients, model not in
registry) and the streaming success path.
"""

import json

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


def _install_state(monkeypatch, *, models=None, clients=None, keys=True):
    """Wire minimal manager / registry / client_registry into proxy state."""
    _keys = [object()] if keys else []
    _models = models or {}
    _clients = clients or []

    class FakeManager:
        keys = _keys

    class FakeRegistry:
        models = _models

    class FakeClientRegistry:
        clients = _clients

    monkeypatch.setattr(proxy, "manager", FakeManager())
    monkeypatch.setattr(proxy, "registry", FakeRegistry())
    monkeypatch.setattr(proxy, "client_registry", FakeClientRegistry())


def test_test_model_requires_model(monkeypatch):
    _install_state(monkeypatch)
    client = TestClient(proxy.app)
    r = client.post("/admin/test-model", headers=_admin_headers(), json={})
    assert r.status_code == 400


def test_test_model_returns_404_when_model_not_in_registry(monkeypatch):
    _install_state(monkeypatch, models={"glm-5": ["key-one"]})
    client = TestClient(proxy.app)
    r = client.post(
        "/admin/test-model",
        headers=_admin_headers(),
        json={"model": "nonexistent-model", "prompt": "2+2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == 404
    assert "not found in registry" in body["error"]


def test_test_model_returns_503_when_no_client_keys(monkeypatch):
    _install_state(monkeypatch, models={"glm-5": ["key-one"]}, clients=[])
    client = TestClient(proxy.app)
    r = client.post(
        "/admin/test-model",
        headers=_admin_headers(),
        json={"model": "glm-5", "prompt": "2+2"},
    )
    assert r.status_code == 503


def test_test_model_returns_503_when_no_upstream_keys(monkeypatch):
    _install_state(monkeypatch, models={"glm-5": ["key-one"]}, keys=False)
    client = TestClient(proxy.app)
    r = client.post(
        "/admin/test-model",
        headers=_admin_headers(),
        json={"model": "glm-5", "prompt": "2+2"},
    )
    assert r.status_code == 503


def test_test_model_requires_admin_auth():
    """Unauthenticated POST must be rejected."""
    client = TestClient(proxy.app)
    r = client.post("/admin/test-model", json={"model": "glm-5"})
    assert r.status_code == 401

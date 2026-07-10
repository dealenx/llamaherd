import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from llamaherd import proxy


@pytest.fixture(autouse=True)
def _admin_auth_state(monkeypatch):
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    proxy._admin_sessions.clear()
    yield
    proxy._admin_sessions.clear()


def test_admin_api_requires_bearer_header():
    client = TestClient(proxy.app)
    assert client.get("/admin/in-flight?token=test-token").status_code == 401
    response = client.get(
        "/admin/in-flight",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200


def test_admin_session_is_short_lived_and_limited_purpose():
    client = TestClient(proxy.app)
    response = client.post(
        "/admin/session",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["expires_in"] == proxy.ADMIN_SESSION_TTL_SECONDS
    assert data["session_token"] != "test-token"
    proxy._verify_admin_session(data["session_token"])

    proxy._admin_sessions[data["session_token"]] = time.time() - 1
    with pytest.raises(HTTPException) as exc:
        proxy._verify_admin_session(data["session_token"])
    assert exc.value.status_code == 401

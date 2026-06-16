from fastapi.testclient import TestClient

from llamaherd import proxy


def test_healthz_is_unauthenticated():
    client = TestClient(proxy.app)

    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

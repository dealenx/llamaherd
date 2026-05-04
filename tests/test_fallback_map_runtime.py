"""Tests for runtime fallback model_map mutations (POST/DELETE /admin/fallback-map)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from llamaherd import proxy
from llamaherd.proxy import FallbackProvider


def _provider(tmp_path, **overrides):
    cfg = {
        "provider": "nvidia-build",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvapi-test",
        "default_model": "deepseek-ai/deepseek-v4-flash",
        "priority": "after",
        "model_map": {"glm-5.1": "z-ai/glm-5.1"},
        "metadata_cache_path": str(tmp_path / "nvidia_meta.json"),
    }
    cfg.update(overrides)
    return FallbackProvider(cfg)


def test_add_mapping_string_form(tmp_path):
    fp = _provider(tmp_path)
    entry = fp.add_mapping("qwen3.5:397b", "qwen/qwen3.5-397b-a17b")
    assert entry["nvidia_model"] == "qwen/qwen3.5-397b-a17b"
    assert entry["priority"] is None
    assert fp.resolve_model("qwen3.5:397b") == "qwen/qwen3.5-397b-a17b"


def test_add_mapping_with_per_model_priority(tmp_path):
    fp = _provider(tmp_path)
    entry = fp.add_mapping("qwen3.5:397b", "qwen/qwen3.5-397b-a17b", priority="before")
    assert entry["priority"] == "before"
    assert fp.priority_for("qwen3.5:397b") == "before"


def test_add_mapping_invalid_priority_falls_back_to_global(tmp_path):
    fp = _provider(tmp_path)
    entry = fp.add_mapping("qwen3.5:397b", "qwen/qwen3.5-397b-a17b", priority="garbage")
    assert entry["priority"] is None
    # priority_for inherits the global default when no per-model override.
    assert fp.priority_for("qwen3.5:397b") == fp.priority


def test_add_mapping_requires_both_names(tmp_path):
    fp = _provider(tmp_path)
    with pytest.raises(ValueError):
        fp.add_mapping("", "qwen/x")
    with pytest.raises(ValueError):
        fp.add_mapping("qwen3.5:397b", "")


def test_remove_mapping(tmp_path):
    fp = _provider(tmp_path)
    fp.add_mapping("foo", "bar/baz")
    assert fp.remove_mapping("foo") is True
    assert fp.resolve_model("foo") is None
    assert fp.remove_mapping("foo") is False


def test_admin_add_fallback_map_endpoint(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")

    captured: list[tuple[str, dict]] = []

    async def fake_broadcast(event_type, data):
        captured.append((event_type, data))

    monkeypatch.setattr(proxy.broadcaster, "broadcast", fake_broadcast)

    client = TestClient(proxy.app)
    r = client.post(
        "/admin/fallback-map?token=test-token",
        json={"ollama_name": "qwen3.5:397b", "nvidia_name": "qwen/qwen3.5-397b-a17b"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["added"]["ollama_name"] == "qwen3.5:397b"
    assert body["added"]["nvidia_model"] == "qwen/qwen3.5-397b-a17b"
    assert body["added"]["priority"] == "after"  # global default
    ids = {a["id"] for a in body["model_map"]}
    assert "qwen3.5:397b" in ids

    # SSE event was broadcast.
    types = [t for t, _ in captured]
    assert "fallback_map_update" in types
    payload = next(d for t, d in captured if t == "fallback_map_update")
    assert payload["action"] == "add"
    assert payload["ollama_name"] == "qwen3.5:397b"


def test_admin_add_fallback_map_with_priority(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")

    client = TestClient(proxy.app)
    r = client.post(
        "/admin/fallback-map?token=test-token",
        json={
            "ollama_name": "qwen3.5:397b",
            "nvidia_name": "qwen/qwen3.5-397b-a17b",
            "priority": "before",
        },
    )
    assert r.status_code == 200
    assert r.json()["added"]["priority"] == "before"
    assert fp.priority_for("qwen3.5:397b") == "before"


def test_admin_add_fallback_map_validation(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")

    client = TestClient(proxy.app)
    r = client.post(
        "/admin/fallback-map?token=test-token",
        json={"ollama_name": "qwen3.5:397b"},
    )
    assert r.status_code == 400


def test_admin_delete_fallback_map(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    fp.add_mapping("transient", "nv/transient")
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")

    captured: list[tuple[str, dict]] = []

    async def fake_broadcast(event_type, data):
        captured.append((event_type, data))

    monkeypatch.setattr(proxy.broadcaster, "broadcast", fake_broadcast)

    client = TestClient(proxy.app)
    r = client.delete("/admin/fallback-map?token=test-token&ollama_name=transient")
    assert r.status_code == 200
    body = r.json()
    assert body["removed"] == "transient"
    assert all(a["id"] != "transient" for a in body["model_map"])

    # 404 when the alias is gone.
    r = client.delete("/admin/fallback-map?token=test-token&ollama_name=transient")
    assert r.status_code == 404

    # SSE event.
    types = [t for t, _ in captured]
    assert "fallback_map_update" in types
    payload = next(d for t, d in captured if t == "fallback_map_update")
    assert payload["action"] == "remove"
    assert payload["ollama_name"] == "transient"


def test_admin_fallback_map_requires_token(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    client = TestClient(proxy.app)
    r = client.post("/admin/fallback-map", json={"ollama_name": "x", "nvidia_name": "y"})
    assert r.status_code == 401
    r = client.delete("/admin/fallback-map?ollama_name=x")
    assert r.status_code == 401


def test_admin_fallback_map_when_disabled(monkeypatch):
    """When the fallback isn't configured, mutation endpoints reject with 400."""
    monkeypatch.setattr(proxy, "fallback_provider", FallbackProvider({}))
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    client = TestClient(proxy.app)
    r = client.post(
        "/admin/fallback-map?token=test-token",
        json={"ollama_name": "x", "nvidia_name": "y"},
    )
    assert r.status_code == 400
    r = client.delete("/admin/fallback-map?token=test-token&ollama_name=x")
    assert r.status_code == 400

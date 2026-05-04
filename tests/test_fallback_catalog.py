"""Tests for the FallbackProvider model catalog + metadata cache."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from llamaherd.proxy import FallbackProvider
from llamaherd import proxy


def _provider(tmp_path, **overrides):
    cfg = {
        "provider": "nvidia-build",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvapi-test",
        "default_model": "deepseek-ai/deepseek-v4-flash",
        "priority": "after",
        "model_map": {
            "glm-5.1": "z-ai/glm-5.1",
            "minimax-m2.5": "minimaxai/minimax-m2.5",
        },
        "metadata_cache_path": str(tmp_path / "nvidia_meta.json"),
    }
    cfg.update(overrides)
    return FallbackProvider(cfg)


def test_catalog_marks_mapped_and_unmapped(tmp_path):
    fp = _provider(tmp_path)
    fp.discovered_models = [
        {"id": "z-ai/glm-5.1", "owned_by": "z-ai"},
        {"id": "z-ai/glm5", "owned_by": "z-ai"},
        {"id": "minimaxai/minimax-m2.5", "owned_by": "minimaxai"},
        {"id": "deepseek-ai/deepseek-v4-pro", "owned_by": "deepseek-ai"},
    ]
    catalog = fp.get_catalog()
    by_id = {row["id"]: row for row in catalog}

    assert by_id["z-ai/glm-5.1"]["is_mapped"] is True
    assert by_id["z-ai/glm-5.1"]["ollama_equivalent"] == "glm-5.1"
    assert by_id["z-ai/glm5"]["is_mapped"] is False
    assert by_id["z-ai/glm5"]["ollama_equivalent"] is None
    assert by_id["minimaxai/minimax-m2.5"]["is_mapped"] is True
    assert by_id["deepseek-ai/deepseek-v4-pro"]["is_mapped"] is False

    # Sorted by org then id.
    orgs = [row["org"] for row in catalog]
    assert orgs == sorted(orgs)


def test_catalog_includes_model_card_url_and_org(tmp_path):
    fp = _provider(tmp_path)
    fp.discovered_models = [{"id": "z-ai/glm-5.1", "owned_by": "z-ai"}]
    catalog = fp.get_catalog()
    assert catalog[0]["model_card_url"] == "https://build.nvidia.com/z-ai/glm-5.1"
    assert catalog[0]["org"] == "z-ai"


def test_catalog_uses_cached_metadata(tmp_path):
    fp = _provider(tmp_path)
    fp.discovered_models = [{"id": "z-ai/glm-5.1", "owned_by": "z-ai"}]
    fp.metadata_cache["z-ai/glm-5.1"] = {
        "fetched_at": time.time(),
        "description": "GLM 5.1 chat model",
        "context_length": 131072,
        "parameter_count": 32_000_000_000,
    }
    [row] = fp.get_catalog()
    assert row["description"] == "GLM 5.1 chat model"
    assert row["context_length"] == 131072
    assert row["parameter_count"] == 32_000_000_000


def test_metadata_cache_persistence(tmp_path):
    cache_path = tmp_path / "meta.json"
    fp = _provider(tmp_path, metadata_cache_path=str(cache_path))
    fp.metadata_cache["z-ai/glm-5.1"] = {"fetched_at": 100.0, "description": "test"}
    fp._save_metadata_cache()

    # New provider re-loads cache from disk.
    fp2 = _provider(tmp_path, metadata_cache_path=str(cache_path))
    assert fp2.metadata_cache["z-ai/glm-5.1"]["description"] == "test"


def test_docs_url_swaps_slash_to_dash():
    assert FallbackProvider._docs_url("z-ai/glm-5.1") == \
        "https://docs.api.nvidia.com/nim/reference/z-ai-glm-5.1"


def test_admin_fallback_catalog_endpoint(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    fp.discovered_models = [
        {"id": "z-ai/glm-5.1", "owned_by": "z-ai"},
        {"id": "z-ai/glm5", "owned_by": "z-ai"},
        {"id": "minimaxai/minimax-m2.5", "owned_by": "minimaxai"},
    ]
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")

    client = TestClient(proxy.app)
    r = client.get("/admin/fallback-catalog?token=test-token")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["count"] == 3
    assert body["by_org"]["z-ai"] == 2
    assert body["by_org"]["minimaxai"] == 1
    ids = {row["id"] for row in body["catalog"]}
    assert "z-ai/glm-5.1" in ids


def test_admin_fallback_catalog_when_disabled(monkeypatch):
    monkeypatch.setattr(proxy, "fallback_provider", FallbackProvider({}))
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    client = TestClient(proxy.app)
    r = client.get("/admin/fallback-catalog?token=test-token")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["catalog"] == []


def test_admin_fallback_catalog_requires_token(monkeypatch, tmp_path):
    fp = _provider(tmp_path)
    monkeypatch.setattr(proxy, "fallback_provider", fp)
    monkeypatch.setattr(proxy, "admin_token", "test-token")
    client = TestClient(proxy.app)
    r = client.get("/admin/fallback-catalog")
    assert r.status_code == 401


def test_refresh_metadata_skips_fresh_entries(tmp_path):
    fp = _provider(tmp_path, metadata_max_age=3600)
    fp.discovered_models = [{"id": "z-ai/glm-5.1"}, {"id": "z-ai/glm5"}]
    # glm-5.1 is "fresh" (just fetched), glm5 has no cache entry.
    fp.metadata_cache["z-ai/glm-5.1"] = {
        "fetched_at": time.time(),
        "description": "fresh",
    }

    fetched: list[str] = []

    async def fake_fetch(model_id, timeout=5.0):
        fetched.append(model_id)
        fp.metadata_cache[model_id] = {
            "fetched_at": time.time(),
            "description": f"meta for {model_id}",
        }
        return fp.metadata_cache[model_id]

    fp.fetch_model_metadata = fake_fetch  # type: ignore

    import asyncio as _asyncio
    updated = _asyncio.run(fp.refresh_metadata_cache(timeout=1.0))

    # Only the stale (missing) entry should be re-fetched.
    assert fetched == ["z-ai/glm5"]
    assert updated == 1

import asyncio

import pytest
from starlette.requests import Request

from llamaherd import proxy


class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, headers=None):
        assert url == "https://ollama.com/api/tags"
        assert headers == {"Authorization": "Bearer tok-a"}
        return FakeResponse(200, {
            "models": [
                {
                    "name": "model-a",
                    "model": "model-a",
                    "modified_at": "2026-01-02T00:00:00Z",
                    "size": 123,
                    "digest": "abc123",
                    "details": {
                        "family": "llama",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
        })

    async def post(self, url, headers=None, json=None):
        assert url == "https://ollama.com/api/show"
        assert headers == {"Authorization": "Bearer tok-a", "Content-Type": "application/json"}
        assert json == {"model": "model-a"}
        return FakeResponse(200, {
            "modified_at": "2026-01-03T00:00:00Z",
            "details": {
                "parent_model": "model-a",
                "format": "gguf",
                "family": "llama",
                "families": None,
                "parameter_size": "8000000000",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {
                "general.parameter_count": 8000000000,
                "llama.context_length": 131072,
            },
            "capabilities": ["completion", "tools", "vision"],
        })


class FakeManager:
    keys = [proxy.KeyState(token="tok-a", label="Sub A")]


@pytest.mark.asyncio
async def test_model_registry_enriches_metadata_from_native_ollama(monkeypatch):
    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeAsyncClient)
    registry = proxy.ModelRegistry(FakeManager(), "https://ollama.com/v1")

    await registry.refresh()

    assert list(registry.models) == ["model-a"]
    meta = registry.model_metadata["model-a"]
    assert meta["context_length"] == 131072
    assert meta["parameter_count"] == 8000000000
    assert meta["family"] == "llama"
    assert meta["quantization_level"] == "Q4_K_M"
    assert meta["capabilities"] == ["completion", "tools", "vision"]

    openai_entry = registry.get_models_response()["data"][0]
    assert openai_entry["id"] == "model-a"
    assert openai_entry["created"] == 1767398400
    assert openai_entry["context_length"] == 131072
    assert openai_entry["capabilities"] == ["completion", "tools", "vision"]
    assert openai_entry["family"] == "llama"


@pytest.mark.asyncio
async def test_routes_use_registry_metadata(monkeypatch):
    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeAsyncClient)
    registry = proxy.ModelRegistry(FakeManager(), "https://ollama.com/v1")
    await registry.refresh()
    monkeypatch.setattr(proxy, "registry", registry)
    monkeypatch.setattr(proxy, "_resolve_client", lambda request: {"id": "test-client"})
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    model = await proxy.get_model("model-a", request)
    assert model["context_length"] == 131072
    assert model["capabilities"] == ["completion", "tools", "vision"]

    tags = await proxy.api_tags(request)
    assert tags["models"][0]["model"] == "model-a"
    assert tags["models"][0]["modified_at"] == "2026-01-03T00:00:00Z"
    assert tags["models"][0]["details"]["context_length"] == 131072
    assert tags["models"][0]["details"]["family"] == "llama"

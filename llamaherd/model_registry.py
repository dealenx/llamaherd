import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

import httpx

from .key_manager import KeyManager

log = logging.getLogger("llamaherd")


MODEL_CONTEXT_LENGTHS: dict[str, int] = {
    "cogito-2.1:671b": 131072,
    "deepseek-v3.1:671b": 131072,
    "deepseek-v3.2": 131072,
    "devstral-2:123b": 131072,
    "devstral-small-2:24b": 131072,
    "gemini-3-flash-preview": 1048576,
    "gemma3:12b": 131072,
    "gemma3:27b": 131072,
    "gemma3:4b": 131072,
    "gemma4:31b": 262144,
    "glm-4.6": 131072,
    "glm-4.7": 131072,
    "glm-5": 202752,
    "glm-5.1": 202752,
    "gpt-oss:120b": 131072,
    "gpt-oss:20b": 131072,
    "kimi-k2-instruct": 262144,
    "kimi-k2-thinking": 262144,
    "kimi-k2.5": 262144,
    "kimi-k2.6": 262144,
    "kimi-k2:1t": 262144,
    "minimax-m2": 196608,
    "minimax-m2.1": 196608,
    "minimax-m2.5": 196608,
    "minimax-m2.7": 196608,
    "ministral-3:14b": 131072,
    "ministral-3:3b": 131072,
    "ministral-3:8b": 131072,
    "mistral-large-3:675b": 131072,
    "nemotron-3-nano:30b": 131072,
    "nemotron-3-super": 131072,
    "qwen3-coder-next": 131072,
    "qwen3-coder:480b": 262144,
    "qwen3-next:80b": 131072,
    "qwen3-vl:235b": 131072,
    "qwen3-vl:235b-instruct": 131072,
    "qwen3.5:397b": 262144,
    "rnj-1:8b": 131072,
}


def fmt_param_count(n: int | None) -> str:
    """Format a parameter count as a B/T-suffixed string.

    Uses T when n >= 1 trillion, otherwise B. Drops the decimal when the
    value rounds cleanly to an integer (e.g. 8_000_000_000 -> '8B').
    """
    if n is None:
        return ""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n >= 1_000_000_000_000:
        v = n / 1_000_000_000_000
        suffix = "T"
    else:
        v = n / 1_000_000_000
        suffix = "B"
    if abs(v - round(v)) < 0.05:
        return f"{round(v)}{suffix}"
    return f"{v:.1f}{suffix}"


class ModelRegistry:
    def __init__(self, manager: KeyManager, upstream: str,
                 pricing_sync: Callable[[], Awaitable[int]] | None = None,
                 event_broadcaster=None):
        self.manager = manager
        self.upstream = upstream
        self.models: dict[str, list[str]] = {}
        self.model_metadata: dict[str, dict] = {}
        self.last_refresh: float = 0
        self._refresh_task: asyncio.Task | None = None
        self._pricing_sync = pricing_sync
        self._event_broadcaster = event_broadcaster

    async def start(self, interval: int = 300):
        self._refresh_task = asyncio.create_task(self._refresh_loop(interval))

    async def stop(self):
        if self._refresh_task:
            self._refresh_task.cancel()

    async def _refresh_loop(self, interval: int):
        while True:
            try:
                await self.refresh()
            except Exception as e:
                log.error(f"Model refresh failed: {e}")
            await asyncio.sleep(interval)

    def _native_base(self) -> str:
        base = self.upstream.rstrip("/")
        if base.endswith("/v1"):
            return base[:-3] + "/api"
        return base + "/api"

    @staticmethod
    def _created_from_modified(modified_at: str | None, fallback: float) -> int:
        if modified_at:
            try:
                return int(datetime.fromisoformat(modified_at).timestamp())
            except Exception:
                pass
        return int(fallback or time.time())

    @staticmethod
    def _context_from_show(show_data: dict) -> int | None:
        info = show_data.get("model_info") or {}
        for key, value in info.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except Exception:
                    return None
        return None

    @staticmethod
    def _parameter_count(show_data: dict) -> int | None:
        info = show_data.get("model_info") or {}
        value = info.get("general.parameter_count")
        if value is None:
            value = (show_data.get("details") or {}).get("parameter_size")
        try:
            return int(value)
        except Exception:
            return None

    def _model_entry(self, model_id: str) -> dict:
        meta = self.model_metadata.get(model_id, {})
        context_length = meta.get("context_length") or MODEL_CONTEXT_LENGTHS.get(model_id)
        entry = {
            "id": model_id,
            "object": "model",
            "created": self._created_from_modified(meta.get("modified_at"), self.last_refresh),
            "owned_by": "ollama",
        }
        if context_length:
            entry["context_length"] = context_length
        for key in ("modified_at", "size", "digest", "capabilities", "family", "parameter_count", "quantization_level"):
            if meta.get(key) is not None:
                entry[key] = meta[key]
        if entry.get("parameter_count") is not None:
            entry["parameter_count"] = fmt_param_count(entry["parameter_count"])
        return entry

    async def refresh(self):
        old_models = set(self.models.keys()) if self.models else set()
        all_models: dict[str, list[str]] = {}
        metadata: dict[str, dict] = dict(self.model_metadata)
        native_base = self._native_base()
        async with httpx.AsyncClient(timeout=30) as client:
            for key in self.manager.keys:
                try:
                    resp = await client.get(
                        f"{native_base}/tags",
                        headers={"Authorization": f"Bearer {key.token}"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for m in data.get("models", []):
                            model_id = m.get("model") or m.get("name") or ""
                            if not model_id:
                                continue
                            all_models.setdefault(model_id, []).append(key.token)
                            current = metadata.setdefault(model_id, {})
                            current.update({
                                "id": model_id,
                                "name": m.get("name") or model_id,
                                "model": model_id,
                                "modified_at": m.get("modified_at"),
                                "size": m.get("size"),
                                "digest": m.get("digest"),
                                "details": m.get("details") or current.get("details") or {},
                            })
                    else:
                        log.warning(f"Model list failed for {key.label}: {resp.status_code}")
                except Exception as e:
                    log.warning(f"Model list error for {key.label}: {e}")

            # Enrich new/changed models from native /api/show. This exposes context length,
            # capabilities (vision/tools/thinking), family, parameter count, and quantization.
            for model_id, tokens in sorted(all_models.items()):
                current = metadata.get(model_id, {})
                needs_show = not current.get("context_length") or not current.get("capabilities")
                if not needs_show:
                    continue
                try:
                    resp = await client.post(
                        f"{native_base}/show",
                        headers={"Authorization": f"Bearer {tokens[0]}", "Content-Type": "application/json"},
                        json={"model": model_id},
                    )
                    if resp.status_code != 200:
                        log.debug(f"/api/show metadata failed for {model_id}: {resp.status_code}")
                        continue
                    show = resp.json()
                    details = show.get("details") or current.get("details") or {}
                    info = show.get("model_info") or {}
                    context_length = self._context_from_show(show) or MODEL_CONTEXT_LENGTHS.get(model_id)
                    metadata[model_id] = {
                        **current,
                        "details": details,
                        "model_info": info,
                        "capabilities": show.get("capabilities") or current.get("capabilities") or [],
                        "modified_at": show.get("modified_at") or current.get("modified_at"),
                        "context_length": context_length,
                        "parameter_count": self._parameter_count(show),
                        "family": details.get("family") or info.get("general.architecture"),
                        "quantization_level": details.get("quantization_level"),
                    }
                except Exception as e:
                    log.debug(f"/api/show metadata error for {model_id}: {e}")

        self.models = all_models
        self.model_metadata = {mid: metadata[mid] for mid in all_models if mid in metadata}
        self.last_refresh = time.time()
        new_models = set(all_models.keys()) - old_models
        if new_models:
            log.info(f"New models discovered: {sorted(new_models)}")
            # Trigger an immediate pricing sync so new models get OpenRouter
            # pricing data right away (instead of waiting up to 24h).
            if self._pricing_sync:
                try:
                    asyncio.create_task(self._pricing_sync())
                except Exception as e:
                    log.debug(f"Pricing sync trigger for new models failed: {e}")
        log.info(f"Model registry: {len(self.models)} models discovered across {len(self.manager.keys)} keys")
        # Broadcast model changes via SSE
        if self._event_broadcaster:
            try:
                await self._event_broadcaster.broadcast("models", {
                    "count": len(self.models),
                    "last_refresh": self.last_refresh,
                    "new_models": sorted(new_models) if new_models else [],
                })
            except Exception:
                pass  # Don't fail if broadcast has no subscribers

    def get_models_response(self) -> dict:
        return {
            "object": "list",
            "data": [self._model_entry(model_id) for model_id in sorted(self.models.keys())],
        }

    def get_preferred_key(self, model: str) -> str | None:
        """Return the preferred key token for a model.

        If the model exists on only one key, prefer that key.
        If the model exists on multiple keys, return None so acquire()
        picks the least-loaded key via its normal load-balancing sort.
        """
        matching = list(dict.fromkeys(self.models.get(model, [])))  # dedupe preserving order
        if len(matching) == 1:
            return matching[0]
        # Available on 0 or 2+ keys — let acquire() decide by load
        return None

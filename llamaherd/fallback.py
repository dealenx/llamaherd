import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger("llamaherd")


class ModelAliasManager:
    """Manage model aliases (e.g. glm-5.2-256k → glm-5.2 with ctx=262144).

    Config format (in config.yaml)::

        model_aliases:
          - alias: glm-5.2-256k
            upstream_model: glm-5.2
            context_length: 262144
          - alias: glm-5.2-128k
            upstream_model: glm-5.2
            context_length: 131072

    Aliases are purely client-facing: the proxy rewrites ``req_json["model"]``
    to ``upstream_model`` before forwarding to Ollama Cloud.  The alias name
    is kept for usage tracking and displayed in /v1/models with the overridden
    context_length.
    """

    def __init__(self, entries: list[dict] | None = None):
        self._aliases: dict[str, dict] = {}
        if entries:
            for e in entries:
                alias = (e.get("alias") or "").strip()
                upstream = (e.get("upstream_model") or "").strip()
                if not alias or not upstream:
                    continue
                self._aliases[alias] = {
                    "upstream_model": upstream,
                    "context_length": int(e["context_length"]) if e.get("context_length") else None,
                }

    def resolve(self, model: str) -> tuple[str, int | None]:
        """If *model* is an alias, return (upstream_model, context_length_override).

        If not an alias, return (model, None) unchanged.
        """
        entry = self._aliases.get(model)
        if entry:
            return entry["upstream_model"], entry["context_length"]
        return model, None

    def is_alias(self, model: str) -> bool:
        return model in self._aliases

    @property
    def aliases(self) -> dict[str, dict]:
        """Read-only view of all alias entries."""
        return dict(self._aliases)

    def alias_entries(self) -> list[dict]:
        """Return alias metadata for /v1/models and /api/tags."""
        out = []
        for alias, entry in sorted(self._aliases.items()):
            out.append({
                "alias": alias,
                "upstream_model": entry["upstream_model"],
                "context_length": entry["context_length"],
            })
        return out


# ---------------------------------------------------------------------------
# Fallback Provider — secondary upstream (e.g. NVIDIA Build) for unmapped or
# overflow traffic. Speaks OpenAI /v1/chat/completions.
# ---------------------------------------------------------------------------

VALID_FALLBACK_PRIORITIES = ("after", "before", "only")


class FallbackProvider:
    """Routes selected models to a secondary OpenAI-compatible upstream.

    Config shape (under top-level ``fallback:`` in config.yaml):

        fallback:
          provider: nvidia-build
          base_url: https://integrate.api.nvidia.com/v1
          api_key: nvapi-...
          default_model: deepseek-ai/deepseek-v4-flash
          priority: after        # after | before | only
          model_map:
            glm-5.1: z-ai/glm-5.1
            glm5:
              nvidia_model: z-ai/glm5
              priority: before
    """

    def __init__(self, config: dict | None):
        cfg = config or {}
        self.provider: str = cfg.get("provider", "fallback")
        self.base_url: str = (cfg.get("base_url") or "").rstrip("/")
        self.api_key: str = cfg.get("api_key", "") or ""
        self.default_model: str | None = cfg.get("default_model")
        self.priority: str = cfg.get("priority", "after")
        if self.priority not in VALID_FALLBACK_PRIORITIES:
            log.warning(f"Invalid fallback priority {self.priority!r}; defaulting to 'after'")
            self.priority = "after"
        self._model_map: dict[str, dict] = {}
        for alias, value in (cfg.get("model_map") or {}).items():
            if isinstance(value, str):
                self._model_map[alias] = {"nvidia_model": value, "priority": None}
            elif isinstance(value, dict):
                self._model_map[alias] = {
                    "nvidia_model": value.get("nvidia_model") or value.get("model"),
                    "priority": value.get("priority"),
                }
        # Models discovered from the fallback's /v1/models on startup.
        self.discovered_models: list[dict] = []
        self.enabled: bool = bool(self.base_url and self.api_key)
        # Metadata cache for the fallback model catalog (keyed by model id).
        cache_path = cfg.get("metadata_cache_path", "~/llamaherd/nvidia_model_cache.json")
        self.metadata_cache_path: Path = Path(cache_path).expanduser()
        self.metadata_cache: dict[str, dict] = {}
        self._load_metadata_cache()
        # Refresh metadata older than this (seconds) — default 7 days.
        self.metadata_max_age: float = float(cfg.get("metadata_max_age", 7 * 86400))

    @property
    def label(self) -> str:
        return self.provider

    def resolve_model(self, ollama_model: str) -> str | None:
        """Map an Ollama-style model name to the fallback's model name.

        Returns None when the model isn't in the explicit map.
        """
        entry = self._model_map.get(ollama_model)
        if not entry:
            return None
        return entry.get("nvidia_model")

    def priority_for(self, ollama_model: str) -> str:
        """Effective priority for ``ollama_model``: per-model override or global default."""
        entry = self._model_map.get(ollama_model) or {}
        per_model = entry.get("priority")
        if per_model in VALID_FALLBACK_PRIORITIES:
            return per_model
        return self.priority

    def should_try(self, priority: str, model_available_on_ollama: bool) -> bool:
        """Decide whether to try the fallback for a model.

        ``priority`` is the per-model effective priority. ``model_available_on_ollama``
        indicates whether the model exists on the Ollama Cloud registry.
        """
        if not self.enabled:
            return False
        if priority == "only":
            return True
        if priority == "before":
            return True
        # after: only fallback if Ollama doesn't have the model (or all keys exhausted —
        # the caller handles the exhaustion path separately).
        return not model_available_on_ollama

    def set_priority(self, priority: str) -> str:
        """Update the global priority at runtime. Returns the active value."""
        if priority in VALID_FALLBACK_PRIORITIES:
            self.priority = priority
        return self.priority

    async def discover_models(self, timeout: float = 5.0):
        """Query the fallback's /v1/models. Best-effort, doesn't block startup."""
        if not self.enabled:
            return
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code != 200:
                    log.warning(f"Fallback /models returned {resp.status_code}")
                    return
                data = resp.json().get("data") or []
                self.discovered_models = data
                log.info(f"Fallback {self.provider}: discovered {len(data)} models")
        except Exception as e:
            log.warning(f"Fallback model discovery failed: {e}")

    def model_aliases(self) -> list[dict]:
        """Return the configured aliases as model entries (for /v1/models, /admin/models)."""
        out = []
        for alias, entry in self._model_map.items():
            out.append({
                "id": alias,
                "nvidia_model": entry.get("nvidia_model"),
                "priority": entry.get("priority") or self.priority,
                "provider": self.provider,
            })
        return out

    # ----- Runtime model_map mutations (used by /admin/fallback-map) -----

    def add_mapping(self, ollama_name: str, nvidia_name: str,
                    priority: str | None = None) -> dict:
        """Add or update an in-memory mapping. Returns the stored entry."""
        if not ollama_name or not nvidia_name:
            raise ValueError("ollama_name and nvidia_name are required")
        entry = {
            "nvidia_model": nvidia_name,
            "priority": priority if priority in VALID_FALLBACK_PRIORITIES else None,
        }
        self._model_map[ollama_name] = entry
        return entry

    def remove_mapping(self, ollama_name: str) -> bool:
        """Remove an in-memory mapping. Returns True if it existed."""
        return self._model_map.pop(ollama_name, None) is not None

    # ----- Metadata cache (NVIDIA Build catalog) -----

    def _load_metadata_cache(self) -> None:
        """Load cached model metadata from disk (best-effort)."""
        try:
            if self.metadata_cache_path.exists():
                with open(self.metadata_cache_path) as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.metadata_cache = raw
        except Exception as e:
            log.warning(f"Failed to load fallback metadata cache: {e}")

    def _save_metadata_cache(self) -> None:
        """Persist metadata cache to disk (best-effort)."""
        try:
            self.metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_cache_path, "w") as f:
                json.dump(self.metadata_cache, f, indent=2, sort_keys=True)
        except Exception as e:
            log.warning(f"Failed to save fallback metadata cache: {e}")

    @staticmethod
    def _docs_url(model_id: str) -> str:
        """Convert a NVIDIA model id (org/name) to its docs API URL."""
        slug = model_id.replace("/", "-")
        return f"https://docs.api.nvidia.com/nim/reference/{slug}"

    @staticmethod
    def _model_card_url(model_id: str) -> str:
        return f"https://build.nvidia.com/{model_id}"

    async def fetch_model_metadata(self, model_id: str, timeout: float = 5.0) -> dict | None:
        """Fetch metadata for a single fallback model from the docs API.

        Best-effort: returns None on failure. Stores result in self.metadata_cache.
        """
        url = self._docs_url(model_id)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
                ct = (resp.headers.get("content-type") or "").lower()
                meta: dict = {"fetched_at": time.time(), "source": url}
                if "json" in ct:
                    body = resp.json()
                    meta["raw"] = body
                    if isinstance(body, dict):
                        for k in ("description", "context_length", "parameter_count",
                                   "summary", "tags", "modality"):
                            if k in body:
                                meta[k] = body[k]
                else:
                    meta["raw"] = resp.text[:4000]
                self.metadata_cache[model_id] = meta
                return meta
        except Exception:
            return None

    async def refresh_metadata_cache(self, timeout: float = 5.0,
                                      max_concurrency: int = 4) -> int:
        """Refresh metadata for discovered models that are missing or stale.

        Returns the number of metadata entries updated.
        """
        if not self.discovered_models:
            return 0
        now = time.time()
        targets: list[str] = []
        for m in self.discovered_models:
            mid = m.get("id") if isinstance(m, dict) else None
            if not mid:
                continue
            cached = self.metadata_cache.get(mid)
            if cached and (now - cached.get("fetched_at", 0)) < self.metadata_max_age:
                continue
            targets.append(mid)
        if not targets:
            return 0
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(mid: str):
            async with sem:
                await self.fetch_model_metadata(mid, timeout=timeout)

        await asyncio.gather(*(_one(mid) for mid in targets), return_exceptions=True)
        self._save_metadata_cache()
        return len(targets)

    def get_catalog(self) -> list[dict]:
        """Return the full discovered-model catalog enriched with cached metadata."""
        # Build reverse lookup: nvidia_model -> ollama alias
        reverse: dict[str, str] = {}
        for alias, entry in self._model_map.items():
            nv = entry.get("nvidia_model")
            if nv:
                reverse[nv] = alias
        out: list[dict] = []
        for m in self.discovered_models:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not mid:
                continue
            meta = self.metadata_cache.get(mid) or {}
            org = mid.split("/", 1)[0] if "/" in mid else (m.get("owned_by") or "")
            ollama_alias = reverse.get(mid)
            out.append({
                "id": mid,
                "owned_by": m.get("owned_by") or org,
                "org": org,
                "context_length": meta.get("context_length"),
                "parameter_count": meta.get("parameter_count"),
                "description": meta.get("description") or meta.get("summary"),
                "model_card_url": self._model_card_url(mid),
                "is_mapped": ollama_alias is not None,
                "ollama_equivalent": ollama_alias,
                "metadata_fetched_at": meta.get("fetched_at"),
            })
        out.sort(key=lambda r: (r.get("org") or "", r.get("id") or ""))
        return out

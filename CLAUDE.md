# LlamaHerd project context

LlamaHerd is a FastAPI Ollama Cloud multi-key proxy in `llamaherd/proxy.py`, with CLI in `llamaherd/cli.py` and README docs.

## Important constraints
- Do not print or expose admin tokens, client tokens, Ollama API keys, cookies, or config secrets.
- Do not overwrite production `proxy.db` or `config.yaml` with test fixtures.
- Production runs as user systemd service `ollama-cloud-proxy.service` on `0.0.0.0:8399`.
- Staging/sidecar runs as user systemd service `llamaherd-sidecar.service` on `0.0.0.0:8499`.
- Prefer tests that use temp files/databases and mocked upstreams; avoid real Ollama Cloud calls in automated tests.
- We are on the `feat/fallback-provider` branch. Commit changes here. Do not merge to main.

## Current feature work
- Dashboard uses SSE `/admin/events` for live calls/status/models.
- Dashboard has date period filtering.
- Model registry uses native `/api/tags` and `/api/show` metadata; `/v1/models`, `/api/tags`, and `/admin/models` expose model metadata.
- OpenClaw should use native Ollama API with no `/v1`; Hermes/OpenAI clients use `/v1`.

## Verification commands
- Syntax: `python3 -m compileall llamaherd`
- Smoke production: `python3 ~/.hermes/skills/devops/ollama-cloud-proxy/scripts/production_smoke_test.py`
- Unit/e2e tests: add pytest tests under `tests/` and run `python3 -m pytest`.

## Task: 4 Features to Implement

You are on branch `feat/fallback-provider`. Implement ALL FOUR features below. This is a large task — plan carefully, implement incrementally, run syntax checks and tests after each feature.

### Feature 1: Parameter Display — Billion (B) / Trillion (T) Format

**Problem:** Model parameter counts from `/api/show` come as raw integers (e.g. `8000000000`). Dashboard and CLI currently show raw numbers or nothing. 

**Requirements:**
- Format parameter counts as B (billion) or T (trillion). Examples: `8B`, `70B`, `1.7T`
- Only use T when >= 1 trillion (1,000,000,000,000)
- Show 1 decimal place when the number isn't a clean integer: `8.3B`, `1.7T`
- No decimal when it's clean: `8B`, `70B`
- Apply this formatting in: dashboard model grid/chips, CLI `models` table, `/admin/models` endpoint, `/v1/models` endpoint (`parameter_count` field)
- The `_parameter_count()` method in ModelRegistry returns raw ints — keep that, but add a formatting helper `fmt_param_count(n: Optional[int]) -> str` that's used at display/output boundaries

### Feature 2: Live Request Visibility — In-Flight Request Tracking

**Problem:** The dashboard only shows completed requests in the live feed. Users want to see requests as they arrive, while in-flight, and when they complete — with animation for in-flight requests showing client, model, and target.

**Requirements:**
- When a request arrives at the proxy, immediately broadcast an SSE `request_start` event with: `{request_id, client_id, model, target_key (label), target_provider ("ollama-cloud" or fallback name), timestamp}`
- When a request completes, broadcast an SSE `request_end` event with: `{request_id, client_id, model, upstream_key, tokens_in, tokens_out, latency_ms, status, provider}`
- Add an in-memory dict `_in_flight: dict[str, dict]` to track active requests (keyed by request_id). Clean up on completion.
- Add `/admin/in-flight` endpoint returning current in-flight requests
- **Dashboard changes:**
  - Add a new "In-Flight" panel ABOVE the live feed table, showing current in-flight requests
  - Each in-flight request shows: client → model, target key, elapsed time (live ticking counter)
  - Animate in-flight rows with a subtle pulsing border or background (CSS animation)
  - When a request completes: row animates out (brief flash green for 200, red for error) then removes
  - The existing live feed table below now gets entries on `request_end` (not `call` — rename the event type)
  - Keep backward compat: also emit `call` event on completion so existing consumers aren't broken

**Implementation notes:**
- Generate request_id with `secrets.token_hex(8)` at the start of each proxy handler
- In the proxy route functions (`openai_chat`, `api_chat` and their streaming variants), wrap the request lifecycle with start/end broadcasts
- For streaming responses, the request is "in flight" until the stream generator finishes
- The `_record_and_broadcast` function already exists — extend it to also emit `request_end`

### Feature 3: NVIDIA Build Fallback Provider

**Problem:** When a requested model isn't available on Ollama Cloud (or all Ollama Cloud keys are exhausted), the proxy should fall back to NVIDIA Build API rather than just failing.

**Requirements:**
- Add a `fallback:` section to `config.yaml` (and `config.example.yaml`):
```yaml
fallback:
  provider: nvidia-build
  base_url: https://integrate.api.nvidia.com/v1
  api_key: nvapi-...  # from env or config
  default_model: deepseek-ai/deepseek-v4-flash  # used when no model_map match
  priority: after      # "after" = try ollama first, "before" = try nvidia first, "only" = nvidia only for mapped models
  model_map:
    glm4.7: z-ai/glm4.7
    minimax-m2.5: minimaxai/minimax-m2.5
    minimax-m2.7: minimaxai/minimax-m2.7
    deepseek-v3.2: deepseek-ai/deepseek-v4-pro
    deepseek-v3.2-flash: deepseek-ai/deepseek-v4-flash
    qwen3-coder-480b: qwen/qwen3-coder-480b-a35b-instruct
    gemma4:31b: google/gemma-4-31b-it
    llama4-maverick: meta/llama-4-maverick-17b-128e-instruct
    mistral-large3: mistralai/mistral-large-3-675b-instruct-2512
    mistral-medium3.5: mistralai/mistral-medium-3.5-128b
    kimi-k2.6: moonshotai/kimi-k2.6
```
- Create a `FallbackProvider` class that handles routing to the fallback:
  - `__init__(config: dict)` — takes the fallback config section
  - `resolve_model(ollama_model: str) -> Optional[str]` — returns the mapped NVIDIA model name or None
  - `should_try(priority: str, model_available_on_ollama: bool) -> bool` — decides whether to use fallback based on priority setting
  - `enabled: bool` — whether fallback is configured
- **Routing logic changes in proxy handler:**
  - When `priority: after` (default): try Ollama Cloud first. If model not found on any Ollama Cloud key, OR all keys are exhausted/429'd, route to fallback provider
  - When `priority: before`: try fallback provider first, then Ollama Cloud
  - When `priority: only`: only use fallback for models that appear in model_map; Ollama Cloud for everything else
  - The fallback request uses the same OpenAI `/v1/chat/completions` format — just different base_url and auth headers
  - Broadcast `request_start` with `target_provider` set appropriately ("ollama-cloud" vs "nvidia-build")
- **Model registry integration:**
  - Merge fallback provider models into the available model list (tag them with provider)
  - `/v1/models` should include fallback models with a `provider` field
  - `/admin/models` should show which provider(s) each model is available on
- **Startup:** On startup, query the fallback provider's `/v1/models` to discover available models (with a timeout — don't block startup if it's slow)
- **Dashboard:** Show provider badge on in-flight requests and in live feed

### Feature 4: Provider Priority Setting

**Problem:** Users need to control which provider (Ollama Cloud vs NVIDIA Build) is tried first for a given model.

**Requirements:**
- The `priority` field in the `fallback:` config section controls global default behavior:
  - `after` (default) — Ollama Cloud first, fallback second
  - `before` — Fallback first, Ollama Cloud second  
  - `only` — Use fallback exclusively for mapped models
- Add per-model priority override in `model_map`:
```yaml
model_map:
  minimax-m2.7:
    nvidia_model: minimaxai/minimax-m2.7
    priority: before    # try nvidia first for this model specifically
```
  - When `model_map` value is a string, it's the NVIDIA model name (priority inherits global default)
  - When `model_map` value is a dict with `nvidia_model` and optional `priority`, use per-model priority
  - Do not map GLM-5/5.1 to NVIDIA Build unless the Build page marks them as `nim_type_preview` / Free Endpoint. `/v1/models` may list partner/download-only models that hang or are not usable through the free trial API.
- **Dashboard priority control:**
  - Add a small dropdown or toggle in the dashboard header to switch global priority (after/before/only)
  - This calls a new `/admin/fallback-priority` POST endpoint to change priority at runtime (no restart)
  - The runtime change is in-memory only — persists until restart. Log when it's changed.

## Implementation Order

1. Feature 1 (parameter formatting) — smallest, self-contained, do first
2. Feature 3 (fallback provider) — biggest feature, core routing logic
3. Feature 4 (priority) — builds on Feature 3's config/infrastructure
4. Feature 2 (live request visibility) — touches the same proxy routes as Feature 3, do last to avoid conflicts

After each feature, run `python3 -m compileall llamaherd` and `python3 -m pytest -q` to verify.

## Architecture Notes for Implementation

- `proxy.py` is ~2812 lines. The DASHBOARD_HTML string is ~42K chars embedded in it.
- Proxy route functions start around line 1300. The main non-streaming handler uses a retry loop with `manager.acquire()`.
- `_record_and_broadcast()` at line 1168 is the completion hook — calls `usage_db.record()` then broadcasts SSE.
- SSE broadcaster at line 1130 (`SSEBroadcaster`) uses asyncio.Queue fan-out.
- ModelRegistry at line 630 handles model discovery from Ollama Cloud.
- Config is loaded in `lifespan()` at line 1212.
- The fallback provider is a new OpenAI-compatible endpoint — same `/v1/chat/completions` format as Ollama Cloud, just different base_url and auth.
- For model_map, the string shorthand (value is just the nvidia model name) should still work for backward compat.
# LlamaHerd Dashboard Feature Spec

All work is in `llamaherd/proxy.py` (the `DASHBOARD_HTML` constant and backend endpoints).

## Feature 1: In-Flight Request Detail

### Backend changes
Add a new endpoint `GET /admin/in-flight?token=...` that returns the current `_in_flight` dict as JSON.
Each entry already has: `request_id`, `client_id`, `model`, `target_key`, `target_provider`, `timestamp`.
Add `tokens_in` and `tokens_out` fields that update as the stream progresses (use the streaming counters).

### Dashboard changes
- In-flight rows should show: model name, client, provider badge (OC/NV), tokens in/out (updating live), elapsed time
- Clicking an in-flight row should expand it to show: full request details (request_id, target key, provider, start time, headers if available)
- Use a CSS transition for the expand/collapse

## Feature 2: NVIDIA Model Catalog Browser

### Concept
The dashboard should show the full list of 138+ NVIDIA Build models discovered during `FallbackProvider.discover_models()`. Currently only the 14 mapped models are visible. We want users to see ALL available models with metadata, and be able to add them to the model map.

### Backend changes
1. `FallbackProvider.discover_models()` already stores the raw `/v1/models` response. Add a method or endpoint that returns the full catalog with enriched metadata.
2. Add `GET /admin/fallback-catalog?token=...` endpoint returning the full NVIDIA model list with: `id` (NVIDIA slug like `z-ai/glm-5.1`), `owned_by`, and any metadata we can scrape from NVIDIA's docs API at `https://docs.api.nvidia.com/nim/reference/{org-slug}` (where org-slug is the id with `/` replaced by `-`).
3. For metadata enrichment, store a local JSON cache file at `~/llamaherd/nvidia_model_cache.json`. On startup, try to fetch metadata for each model from the NVIDIA docs API (with a 5s timeout per model, don't block startup). Cache results. Re-fetch weekly. If the API is unreachable, use whatever we have cached.
4. Each model entry in the catalog should have: `id`, `owned_by`, `context_length` (from metadata if available), `parameter_count` (from metadata if available), `description` (short summary from metadata), `model_card_url` (link to `https://build.nvidia.com/{id}`), `is_mapped` (boolean — is it in the current model_map?), `ollama_equivalent` (the Ollama Cloud model name if it's mapped, null otherwise).

### Dashboard changes
1. Add a "Model Catalog" section to the dashboard (collapsible panel below the models grid).
2. Show a table/grid of all NVIDIA models with: name, org, parameter count, context length, mapped status (green checkmark or "Add" button).
3. Each model name links to its model card at `https://build.nvidia.com/{id}`.
4. The "Add" button next to unmapped models should call `POST /admin/fallback-map?token=...&ollama_name=X&nvidia_name=Y` to add a mapping at runtime (in-memory only, logs a warning to persist in config).
5. Show a search/filter input to narrow the list by name or org.
6. Group models by org (z-ai, deepseek-ai, minimaxai, etc.) with collapsible sections.

### Important
- The catalog endpoint MUST be admin-authenticated (requires token).
- The catalog data should be cached, not fetched on every dashboard load.
- Don't block the proxy startup if NVIDIA metadata fetch fails.
- Keep the dashboard responsive — use lazy loading for the catalog section.

## Feature 3: Model Map Runtime Updates

### Backend changes
Add `POST /admin/fallback-map?token=...` endpoint that accepts JSON body:
```json
{
  "ollama_name": "qwen3.5:397b",
  "nvidia_name": "qwen/qwen3.5-397b-a17b"
}
```
This adds the mapping to the in-memory `FallbackProvider.model_map` and broadcasts an SSE event `fallback_map_update` so the dashboard refreshes. Log a warning: "Runtime fallback map update for X -> Y; add to config.yaml to persist."

Also add `DELETE /admin/fallback-map?token=...` to remove a mapping at runtime.

## Implementation Order
1. Feature 2 backend (catalog endpoint + metadata cache)
2. Feature 3 backend (runtime map updates)
3. Feature 1 backend (in-flight detail endpoint)
4. Dashboard UI for all three features

## Testing
- `python3 -m compileall llamaherd` must pass
- `python3 -m pytest -q` must pass
- No hardcoded admin tokens or API keys in test files
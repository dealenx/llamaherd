# LlamaHerd Code Review

> Generated 2026-07-10 via full-codebase automated review.

## 1. Architecture Overview

LlamaHerd is an OpenAI-compatible FastAPI proxy that routes LLM requests across multiple Ollama Cloud API keys with intelligent load balancing, usage tracking, rate limiting, client management, and a live dashboard.

**File structure:**
- `llamaherd/proxy.py` — **6,784 lines**: the entire server (14 classes, 226 functions, ~1,340 lines of embedded dashboard HTML/CSS/JS)
- `llamaherd/cli.py` — 568 lines: argparse-based CLI client that talks to the proxy's admin API
- `llamaherd/__init__.py` — 7 lines: version/tagline
- `tests/` — 13 test files, ~1,200 lines total
- `openrouter_pricing.yaml` — 184 lines: model pricing reference data
- `Dockerfile`, `config.example.yaml`, `.env.example`, `pyproject.toml`

**Core classes (all in proxy.py):**

| Class | Lines | Responsibility |
|-------|-------|---------------|
| `_LibSQLHTTPCursor/Connection` | 124–299 | DB-API 2.0 shim over libSQL HTTP protocol |
| `ClientRegistry` | 369–524 | Maps consumer API tokens → client identities (SQLite-backed) |
| `KeyState` | 531–669 | Per-upstream-key state: in-flight, usage, exhaustion, billing cycle |
| `StickySessionManager` | 672–722 | Session→key affinity with TTL for cache reuse |
| `KeyRegistry` | 729–848 | Persists upstream keys + cookies to DB |
| `TelegramNotifier` | 855–1001 | Env-var-configured Telegram usage alerts |
| `KeyManager` | 1004–1227 | Core routing: acquire/release keys, 429/402 handling, weekly-aware selection |
| `UsageScraper` | 1233–1381 | Scrapes ollama.com/settings via browser cookies + cloudscraper |
| `ModelRegistry` | 1458–1645 | Discovers models from /api/tags, enriches via /api/show |
| `ModelAliasManager` | 1655–1714 | Client-facing model name aliases with context_length overrides |
| `FallbackProvider` | 1725–1987 | Secondary upstream (NVIDIA Build) routing with priority system |
| `UsageDB` | 1994–2400 | Token usage tracking with client attribution + OpenRouter cost calculation |
| `EventBroadcaster` | 2516–2552 | SSE fan-out for live dashboard updates |

**Request flow:** Client → `_resolve_client()` (auth) → `_check_rate_limit()` → `_proxy_request()` (acquire key, forward, retry on 429/402, fallback routing) → `_record_and_broadcast()` (DB + SSE)

---

## 2. Code Quality

**The monolith problem.** `proxy.py` at 6,784 lines is more than double the project's own stated threshold ("split when a file exceeds ~3000 lines" per `AGENTS.md` line 26). The file contains: DB connection management, data classes, routing logic, usage scraping, HTML scraping, model discovery, pricing sync, SSE broadcasting, admin CRUD, and ~1,340 lines of inline HTML/CSS/JS dashboard. This makes navigation, testing, and review difficult.

**Embedded dashboard.** Lines 5,369–6,705 are a complete single-page HTML application as a Python string literal (`DASHBOARD_HTML`). This is 20% of the file. It should be a separate `static/dashboard.html` served by FastAPI's `StaticFiles`.

**Global state sprawl.** Lines 2,492–2,507 declare 11 module-level globals (`manager`, `key_registry`, `registry`, `usage_db`, `client_registry`, `usage_scraper`, `telegram_notifier`, `fallback_provider`, `model_alias_manager`, `sticky`, `upstream_url`). The `lifespan` function (line 2,691) uses `global` to mutate 15+ of them. This makes the codebase very hard to test in isolation — tests must `monkeypatch` individual globals rather than inject dependencies.

**Complex functions.** `_proxy_request()` (lines 3,045–3,295) is 250 lines long with 4 levels of nesting, interleaved with alias resolution, fallback routing decisions, rate limiting, retry loops, and streaming/non-streaming branches. `_proxy_ndjson_request()` (lines 3,614–3,763) is a near-copy with slightly different response parsing — significant code duplication.

**Good aspects:** Naming is consistent and descriptive. Docstrings are present on most classes and public methods. Comments explain *why* decisions were made (e.g., the 429 cooldown rationale at line 1,183). The `Ellipsis` sentinel pattern in `ClientRegistry.update()` (line 479) correctly distinguishes "not provided" from "clear to null."

---

## 3. Security Issues

**Admin token in URL query params.** `_verify_admin()` (line 2,676) accepts the admin token via `?token=` query parameter (line 2,682). The dashboard JavaScript stores this in `localStorage` (line 5,744) and passes it on every API call via URL (line 5,762: `url+sep+'token='+encodeURIComponent(ADMIN_TOKEN)`). This leaks the admin token in:
- Browser history
- Server access logs (uvicorn logs query params)
- HTTP Referer headers
- Proxy logs

**Session cookie missing `Secure` flag.** Line 3,036:
```python
f"llamaherd-session={session_id}; Max-Age={ttl}; Path=/; HttpOnly; SameSite=Lax"
```
No `Secure` flag — the cookie will be sent over plaintext HTTP. If the proxy is deployed on `0.0.0.0` (as the config example suggests), this is a session hijacking risk.

**Browser cookies stored in plaintext DB.** `KeyRegistry` (lines 744–756) stores `secure_session`, `aid`, `cf_clearance`, and `stripe_mid` cookies in the `upstream_keys` table as plaintext TEXT columns. These are full authentication credentials for ollama.com accounts. No encryption at rest.

**Admin token prefix logged.** Line 2,709: `log.info(f"Admin authentication enabled (token: {admin_token[:8]}...)\")` — logs the first 8 characters of the admin token. Combined with knowing the token format, this reduces brute-force space.

**No CORS configuration.** The FastAPI app has no `CORSMiddleware`. If the dashboard is served from a different origin than the API, this would break. More importantly, there's no explicit restriction preventing cross-origin requests with the admin token.

**No rate limiting on admin endpoints.** The admin token check (line 2,686) uses `secrets.compare_digest` (good — constant-time comparison), but there's no rate limit or lockout on failed attempts. An attacker can brute-force the admin token as fast as they can send requests.

**cloudscraper dependency.** `UsageScraper` (line 1,269) uses `cloudscraper` to bypass Cloudflare protection on ollama.com. This is actively circumventing anti-bot protections, which could violate Ollama's Terms of Service. The dependency itself is also unmaintained.

**No input validation on admin API endpoints.** `admin_add_key` (line 5,197) accepts a `token` from the request body and directly creates a `KeyState` with it. No validation that it's a real Ollama token format. `admin_update_client` (line 5,324) passes arbitrary body fields to `client_registry.update()`.

**`executescript` naive semicolon splitting.** Line 2,17: `for stmt in [s.strip() for s in sql.split(";") if s.strip()]` — this would break on SQL containing semicolons inside string literals. Only used for internal migrations, so low risk, but still fragile.

---

## 4. Bug Risks

**`session_id` used before assignment.** Line 3,127:
```python
_record_and_broadcast(client_id, "none", model, 0, 0, 0, 404,
                      request_id=request_id, provider="proxy", session_id=session_id)
```
But `session_id` is assigned at line 3,145 (`session_id = _extract_session_id(request, req_json)`). At line 3,127, `session_id` is `None` (or from a previous iteration in the loop — but this is before the loop). The `session_id` argument silently gets `None` here.

**`json.loads` without error handling.** Line 3,059: `req_json = json.loads(body) if body else {}` — if the client sends malformed JSON, this raises an unhandled `json.JSONDecodeError` which becomes a 500 error with a stack trace, rather than a clean 400 response. Same issue at line 3,624 in `_proxy_ndjson_request`.

**Key index-based admin endpoints.** Lines 5,158–5,234: `PUT /admin/keys/{key_index}`, `DELETE /admin/keys/{key_index}` use array indices. If a key is added or deleted between a client listing keys and performing an operation, the wrong key is modified/deleted. The CLI's `keys` command (line 1,84) also displays indices. This is a race condition — should use token prefix or stable ID instead.

**SQLite single-connection concurrency.** `ClientRegistry`, `KeyRegistry`, and `UsageDB` each open a single `sqlite3.connect()` (line 3,37). In the async FastAPI context, multiple coroutines may call `execute()`/`commit()` concurrently. SQLite's default mode allows only one writer at a time. While asyncio is single-threaded, a `commit()` on one connection while another coroutine has an in-flight `execute()` on a different connection to the same file can still produce `database is locked` errors.

**`_in_flight` dict mutation without synchronization.** Lines 2,556–2,674: `_in_flight` is a plain `dict` modified by `_request_start` (line 2,583), `_record_and_broadcast` (line 2,651), and `_sweep_stale_inflight` (line 2,906). While Python dicts are atomic for single operations, the `_sweep_stale_inflight` iterates and pops in a loop (line 2,905) which could race with concurrent `_request_start` calls. In practice, asyncio's cooperative scheduling makes this safe, but it's fragile.

**Token estimation is crude.** Line 3,345: `tokens_out += max(1, len(content) // 4)` — estimates completion tokens as characters/4. This is used when the upstream doesn't return `usage` in the stream. The estimate can be off by 2-10x for non-English content or code. Usage tracking accuracy depends on this.

**Stale in-flight sweeper matches by label.** Line 2,919: `if k.label == target_key_label` — uses the key's label to match. Labels are user-configured and not guaranteed unique. If two keys have the same label, the sweeper releases the wrong key's in_flight counter.

**`TelegramNotifier._client` never closed in lifespan shutdown.** Line 873 creates an `httpx.AsyncClient`, but the lifespan shutdown (lines 2,812–2,819) never calls `telegram_notifier.close()`. This leaks the connection pool on graceful shutdown.

**`model_alias_manager` used in `openrouter_costs` before null check.** Line 2,286: `if model_alias_manager and model_alias_manager.is_alias(lookup_key)` — `model_alias_manager` is a module-level global that could be `None` during startup if costs are queried before lifespan completes. The check is present but the function is called from admin endpoints which are available during startup.

---

## 5. Performance

**New `httpx.AsyncClient` per request.** Lines 3,211, 3,309, 3,426, 3,461, 3,542, 4,015, 4,412 — every single proxy request creates a new `httpx.AsyncClient` with a new connection pool. Under load, this means:
- New TCP connection per request to the upstream
- New TLS handshake per request (for HTTPS upstreams)
- No HTTP/2 connection reuse
- No keep-alive benefits

A shared `httpx.AsyncClient` created in `lifespan()` and reused for all upstream calls would dramatically reduce latency and resource usage.

**`key_by_token` is O(n).** Lines 1,112–1,118: linear scan through `self.keys` to find a key by token. Called on every sticky rebind decision. For N keys this is fine (typically <10), but a dict lookup would be O(1).

**Daily limit check queries DB on every request.** Lines 2,453–2,456: `_check_rate_limit` runs `SELECT COUNT(*), COALESCE(SUM(...)) FROM usage WHERE client_id=? AND day=?` on every request where daily limits are configured. This is a synchronous SQLite query in the async request path.

**Dashboard HTML re-parsed every request.** `DASHBOARD_HTML` is a 1,340-line string literal. FastAPI/Starlette processes it through `HTMLResponse` on every `/dashboard` request. It should be a static file or at minimum cached.

**`_load_openrouter_pricing` loads entire YAML.** Line 4,553–4,569: loads the entire pricing YAML (184 lines) into memory on first access. Cached after that, but the cache never invalidates.

**Model refresh makes serial API calls.** Lines 1,543–1,603: `ModelRegistry.refresh()` iterates keys sequentially, calling `/api/tags` then `/api/show` for each. For N keys with M models, this is O(N + N*M) sequential HTTP calls. Could be parallelized with `asyncio.gather`.

---

## 6. Test Coverage

**13 test files, ~1,200 lines, covering:**

| Test file | Lines | What it covers |
|-----------|-------|----------------|
| `test_fallback_provider.py` | 137 | FallbackProvider routing decisions, priority, resolve_model |
| `test_fallback_catalog.py` | 159 | Catalog metadata, cache persistence |
| `test_fallback_map_runtime.py` | 183 | Runtime map mutations via admin API endpoints |
| `test_fmt_param_count.py` | 21 | Parameter count formatting (B/T) |
| `test_health.py` | 12 | /healthz unauthenticated response |
| `test_in_flight.py` | 107 | In-flight tracking, request_start/end, admin endpoint |
| `test_in_flight_detail.py` | 101 | Token counters, header sanitization |
| `test_key_cooldown.py` | 35 | 429 vs 402 cooldown durations |
| `test_model_registry_metadata.py` | 114 | Model discovery + /api/show enrichment (mocked upstream) |
| `test_native_bridge_conversion.py` | 86 | OpenAI→Ollama message/tool_call/reasoning conversion |
| `test_sticky_routing.py` | 327 | E2E sticky routing with fake upstream server |
| `test_usage_and_dashboard.py` | 60 | recent_calls filtering, dashboard HTML syntax validation |
| `test_weekly_preferred_routing.py` | 151 | Weekly-aware key selection, sticky rebind policy |

**Not tested at all:**
- **`_proxy_request()` — the core proxy logic**: key acquisition, upstream forwarding, 429/402 retry, response handling, error paths.
- **`_proxy_stream()` and `_proxy_ndjson_stream()`**: streaming response handling, token capture from SSE/NDJSON.
- **`_resolve_client()`**: client token authentication, rejection of unknown tokens.
- **`_verify_admin()`**: admin token verification.
- **`_check_rate_limit()` / `_check_rpm()`**: rate limiting enforcement.
- **`UsageDB`**: record, summary, daily_totals, by_client, by_model, openrouter_costs, totals — only recent_calls() is partially tested.
- **`ClientRegistry`**: create, update, delete, regenerate_token, resolve — all untested.
- **`KeyRegistry`**: add, update, update_cookies, remove — all untested.
- **`TelegramNotifier`**: message formatting, send — untested.
- **`UsageScraper`**: HTML parsing, cookie handling — untested.
- **`ModelAliasManager`**: resolve, is_alias — untested.
- **OpenRouter pricing sync** — untested.
- **libSQL HTTP client** — untested.
- **`EventBroadcaster`**: subscribe/unsubscribe/broadcast — untested.
- **All CLI commands** in `cli.py` — untested.
- **Dashboard endpoints** — mostly untested.
- **Session cookie generation** — untested.
- **Fallback routing paths** in `_proxy_request` — untested.

**Overall: ~20% coverage of core functionality.** The tests that exist are well-written and test important edge cases, but the most critical paths are uncovered.

---

## 7. Dependencies

| Dependency | Version | Assessment |
|-----------|---------|------------|
| `fastapi>=0.100` | OK | Appropriate, well-maintained |
| `uvicorn>=0.23` | OK | Standard ASGI server |
| `httpx>=0.24` | OK | Good async HTTP client, but **misused** (new client per request) |
| `pyyaml>=6.0` | OK | Standard YAML parser |
| `cloudscraper>=1.2` | Risk | Unmaintained, bundles requests+bs4, circumvents Cloudflare |
| `beautifulsoup4>=4.12` | Risk | Only used by UsageScraper. Heavy transitive dependency for a single scraper |
| `python-dotenv>=1.0` | OK | Standard .env loader |
| `libsql>=0.1.0` (optional) | Risk | Relatively new, no Windows wheels, limited ecosystem |
| `pytest>=7.0` (dev) | OK | Standard |
| `pytest-asyncio>=0.21` (dev) | OK | Standard |
| `ruff>=0.1` (dev) | OK | Good linter |

**Concern**: `cloudscraper` and `beautifulsoup4` are hard dependencies for a feature (usage scraping) that's optional. Should be optional dependencies.

---

## 8. DevOps

**Dockerfile issues:**
1. No HEALTHCHECK — `/healthz` endpoint exists but isn't referenced
2. Runs as root — no `USER` instruction
3. No `.dockerignore` — `.git/`, `tests/`, `*.db`, `__pycache__/` get copied into the image
4. `config.example.yaml` copied as `config.yaml` — every fresh container starts with the example config
5. No multi-stage build — build tools remain in final image
6. `openrouter_pricing.yaml` not copied — pricing sync will fail silently
7. No volume guidance — `usage.db` and `proxy.db` are written to the container filesystem and lost on redeploy

**Config structure:** `config.example.yaml` is excellent — thoroughly documented. `.env.example` is good. `.gitignore` properly excludes secrets.

**No CI/CD:** No GitHub Actions, no pre-commit hooks, no automated test running.

**No versioned releases / changelog:** Version is hardcoded in `__init__.py` as `1.0.0`. No `CHANGELOG.md`. No git tags.

---

## 9. Suggestions (Ranked by Priority)

### P0 — Critical

1. **Split `proxy.py` into modules.** Proposed: db.py, models.py, routing.py, admin.py, dashboard.py, pricing.py, notifier.py, scraper.py, proxy.py
2. **Extract dashboard HTML to a static file.** 1,340 lines of inline HTML/CSS/JS should be `static/dashboard.html`
3. **Use a shared `httpx.AsyncClient`.** Create one in `lifespan()` and reuse for all upstream calls
4. **Fix `session_id` used before assignment (line 3,127).**
5. **Add `Secure` flag to session cookie (line 3,036).**

### P1 — High

6. **Add JSON parse error handling (line 3,059).** Return 400, not 500
7. **Stop passing admin token in URL query params.** Use Authorization header
8. **Add brute-force protection on admin endpoints.**
9. **Add health check to Dockerfile.**
10. **Run Docker as non-root.**
11. **Add `.dockerignore`.**
12. **Copy `openrouter_pricing.yaml` in Dockerfile.**
13. **Add test coverage for `_proxy_request()`.**
14. **Replace index-based key endpoints with token-based.**

### P2 — Medium

15. **Make `cloudscraper` and `beautifulsoup4` optional dependencies.**
16. **Add in-memory daily usage counters.**
17. **Use dict-based key lookup.**
18. **Parallelize model refresh with `asyncio.gather`.**
19. **Add CORS middleware.**
20. **Close `TelegramNotifier._client` in lifespan shutdown.**
21. **Encrypt cookies at rest.**
22. **Add CI pipeline.**
23. **SQLite WAL mode + connection timeout.**

### P3 — Low

24. **Add `__all__` exports in `__init__.py`.**
25. **Add type hints throughout.**
26. **Add structured logging.**
27. **Add request ID to all log lines.**
28. **Document the API.**

---

## Summary

LlamaHerd is a well-conceived, feature-rich proxy with thoughtful routing logic (weekly-aware key selection, sticky sessions, native bridge for truncation fixes, fallback provider). The code is well-commented and the AGENTS.md/CLAUDE.md documentation shows maturity in project management.

**The biggest problems are:**
1. A monolithic 6,784-line file that violates the project's own splitting guidelines
2. Per-request HTTP client creation that kills performance under load
3. ~20% test coverage of core functionality
4. Security concerns around admin token handling and plaintext cookie storage

**The biggest wins would be:** splitting the monolith, using a shared httpx client, adding tests for `_proxy_request()`, and fixing the admin token query-param exposure.
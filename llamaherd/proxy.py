"""
LlamaHerd — One endpoint. Many llamas. Smarter routing.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI-compatible proxy that routes requests across multiple Ollama Cloud API keys.
- Auto-discovers models from /v1/models on each key
- Tracks concurrency per key, routes to least-loaded
- Queues overflow requests instead of 429ing the client
- Balances usage across keys, prefers freshest billing cycle
- Retries 429s from upstream on another key automatically
- Per-client API keys for attribution (gateway, cron, CLI, etc.)
- Full token usage logging: per client/model/day/upstream-key
- Dynamic client key management via admin API (no restart needed)
- Live dashboard with SSE and time-period filtering
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from uvicorn import Config, Server

from .db import ClientRegistry, _is_libsql_url
from .fallback import VALID_FALLBACK_PRIORITIES, FallbackProvider, ModelAliasManager
from .key_manager import KeyManager, KeyState, StickySessionManager
from .key_registry import KeyRegistry
from .model_registry import MODEL_CONTEXT_LENGTHS, ModelRegistry, fmt_param_count
from .notifier import TelegramNotifier
from .routing import (
    _convert_openai_to_ollama_body,
    _ollama_chunk_to_sse,
)
from .scraper import ActivityUsageRefresher, UsageScraper
from .usage_db import UsageDB

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Load .env from project root (if present) before reading any env vars.
# Search order: LLAMAHERD_ENV (explicit), CWD/.env, repo root/.env, package dir/.env
for _env_candidate in (
    os.environ.get("LLAMAHERD_ENV"),
    str(Path.cwd() / ".env"),
    str(Path(__file__).parent.parent / ".env"),
    str(Path(__file__).parent / ".env"),
):
    if _env_candidate and Path(_env_candidate).is_file():
        load_dotenv(_env_candidate, override=False)
        break

CONFIG_PATH = Path(os.environ.get("LLAMAHERD_CONFIG", str(Path(__file__).parent / "config.yaml")))

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llamaherd")

# ---------------------------------------------------------------------------
# Database connection — SQLite (file) or libSQL (http(s):// URL) dispatcher
# ---------------------------------------------------------------------------
# LlamaHerd keeps usage data in SQLite by default. For containerised
# deployments where the local filesystem is ephemeral, you can point
# usage_db / LLAMAHERD_DB at a libSQL-compatible URL (Turso, self-hosted
# sqld, etc.) and install the optional `libsql` package:
#
#     pip install 'llamaherd[libsql]'
#
# If the URL looks like a remote endpoint (http://, https://, libsql://),
# we use libsql.connect(); otherwise we fall back to the stdlib sqlite3
# module. The connection object returned exposes the same
# execute()/commit()/fetchall() surface either way.
#
# Some self-hosted sqld deployments use Basic auth (user:pass) instead of
# the standard Turso Bearer token. If `db_auth_user` is set (or the token
# contains a colon like "user:pass"), we fall back to a built-in HTTP
# client that speaks the libSQL Hrana-over-HTTP protocol directly with
# the correct Basic auth header. This avoids the native `libsql` package
# (which has no Windows wheels) and works on any platform with httpx.


# ---------------------------------------------------------------------------
# Usage Scraper (cookie-based ollama.com/settings)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model Discovery
# ---------------------------------------------------------------------------

# Known context lengths for Ollama Cloud models (tokens).
# Used to populate /v1/models metadata so clients (Hermes) can auto-detect
# context windows instead of falling back to 128K defaults.

# ---------------------------------------------------------------------------
# Model Alias Manager — presents Ollama Cloud models under alternate names
# with overridden context_length.  Requests for an alias are transparently
# rewritten to the upstream model before forwarding; usage is logged under
# the alias name so the dashboard attributes tokens correctly.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Usage DB — full token tracking with client attribution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rate Limiting — per-client daily tokens, daily requests, RPM
# ---------------------------------------------------------------------------

_rpm_tracker: dict[str, deque] = {}  # client_id -> deque of request timestamps
_rpm_lock = asyncio.Lock()


def _check_rpm(client_id: str, rpm_limit: int) -> bool:
    """Check if client is within RPM limit. Returns True if allowed, False if rate limited."""
    now = time.time()
    if client_id not in _rpm_tracker:
        _rpm_tracker[client_id] = deque()
    window = _rpm_tracker[client_id]
    # Prune entries older than 60s
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= rpm_limit:
        return False
    window.append(now)
    return True


async def _check_rate_limit(request: Request, client: dict) -> JSONResponse | None:
    """Check all rate limits for a client. Returns 429 JSONResponse if limited, None if OK."""
    client_id = client["id"]

    # RPM check (fast, in-memory)
    rpm_limit = client.get("rpm_limit")
    if rpm_limit is not None:
        async with _rpm_lock:
            if not _check_rpm(client_id, rpm_limit):
                window = _rpm_tracker.get(client_id, deque())
                reset_at = int(window[0] + 60) if window else int(time.time() + 60)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "detail": f"RPM limit of {rpm_limit} exceeded for client '{client_id}'",
                        "limit_type": "rpm",
                        "limit": rpm_limit,
                        "reset_at": reset_at,
                    },
                )

    # Daily limits check (queries DB)
    daily_token_limit = client.get("daily_token_limit")
    daily_request_limit = client.get("daily_request_limit")
    if daily_token_limit is not None or daily_request_limit is not None:
        today = datetime.now(UTC).date().isoformat()
        row = usage_db._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_in + tokens_out), 0) FROM usage WHERE client_id = ? AND day = ?",
            (client_id, today),
        ).fetchone()
        today_requests = row[0]
        today_tokens = row[1]

        if daily_request_limit is not None and today_requests >= daily_request_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Daily request limit of {daily_request_limit} exceeded for client '{client_id}'",
                    "limit_type": "daily_requests",
                    "limit": daily_request_limit,
                    "used": today_requests,
                    "reset_at": int((datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()),
                },
            )

        if daily_token_limit is not None and today_tokens >= daily_token_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Daily token limit of {daily_token_limit} exceeded for client '{client_id}'",
                    "limit_type": "daily_tokens",
                    "limit": daily_token_limit,
                    "used": today_tokens,
                    "reset_at": int((datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()),
                },
            )

    return None

# ---------------------------------------------------------------------------
# Proxy App — Lifespan
# ---------------------------------------------------------------------------

manager: KeyManager | None = None
key_registry: KeyRegistry | None = None
registry: ModelRegistry | None = None
usage_db: UsageDB | None = None
client_registry: ClientRegistry | None = None
usage_scraper: UsageScraper | None = None
usage_refresher: ActivityUsageRefresher | None = None
telegram_notifier: TelegramNotifier | None = None
upstream_http_client: httpx.AsyncClient | None = None
fallback_provider: FallbackProvider | None = None
model_alias_manager: ModelAliasManager | None = None
sticky: StickySessionManager | None = None
upstream_url: str = ""
retry_on_429: bool = True
max_retries: int = 2
queue_timeout: int = 60
request_timeout: int = 120
admin_token: str = ""
reject_unknown_models: bool = False  # reject models unknown to both Ollama and fallback model_map
_admin_sessions: dict[str, float] = {}
ADMIN_SESSION_TTL_SECONDS = 60

_DB_DSN = os.environ.get("LLAMAHERD_DB", str(Path(__file__).parent / "proxy.db"))
DB_PATH = Path(_DB_DSN) if not _is_libsql_url(_DB_DSN) else _DB_DSN

# ---------------------------------------------------------------------------
# SSE Event Broadcaster — pushes live updates to dashboard
# ---------------------------------------------------------------------------

class EventBroadcaster:
    """Fan-out event bus for SSE dashboard updates.
    
    Subscribers are asyncio.Queue instances — one per SSE connection.
    When an event is broadcast, it's put into every subscriber queue.
    Stale subscribers (disconnected) are cleaned up automatically.
    """
    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def broadcast(self, event_type: str, data: dict | list):
        payload = json.dumps({"type": event_type, "data": data})
        stale = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                stale.append(q)
        for q in stale:
            self._subscribers.remove(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

broadcaster = EventBroadcaster()

# In-flight request tracker — keyed by request_id, populated by _request_start
# and removed by _record_and_broadcast (which also fires request_end).
_in_flight: dict[str, dict] = {}


def _new_request_id() -> str:
    """Generate a short, unique request id used to correlate start/end events."""
    return secrets.token_hex(8)


def _request_start(request_id: str, client_id: str, model: str,
                    target_key: str, target_provider: str,
                    *, headers: dict | None = None,
                    path: str | None = None) -> None:
    """Register an in-flight request and broadcast a request_start SSE event."""
    entry: dict = {
        "request_id": request_id,
        "client_id": client_id,
        "model": model,
        "target_key": target_key,
        "target_provider": target_provider,
        "started_at": time.time(),
        "tokens_in": 0,
        "tokens_out": 0,
    }
    if path:
        entry["path"] = path
    if headers:
        entry["headers"] = _sanitize_headers(headers)
    _in_flight[request_id] = entry
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcaster.broadcast("request_start", entry))
    except RuntimeError:
        pass


# Header keys that must never appear in the in-flight detail payload.
_SENSITIVE_HEADER_KEYS = {
    "authorization", "cookie", "set-cookie", "proxy-authorization",
    "x-api-key", "api-key", "x-admin-token",
}


def _sanitize_headers(headers: dict) -> dict:
    """Strip auth/cookie headers; keep only those safe to display in the dashboard."""
    out: dict = {}
    for k, v in (headers or {}).items():
        if not isinstance(k, str):
            continue
        if k.lower() in _SENSITIVE_HEADER_KEYS:
            continue
        try:
            out[k] = str(v)[:200]
        except Exception:
            continue
    return out


def _update_in_flight_tokens(request_id: str | None,
                              tokens_in: int | None = None,
                              tokens_out: int | None = None) -> None:
    """Update the live token counters on an in-flight entry (no-op if missing)."""
    if not request_id:
        return
    entry = _in_flight.get(request_id)
    if not entry:
        return
    if tokens_in is not None and tokens_in > entry.get("tokens_in", 0):
        entry["tokens_in"] = tokens_in
    if tokens_out is not None and tokens_out > entry.get("tokens_out", 0):
        entry["tokens_out"] = tokens_out


def _record_and_broadcast(client_id: str, upstream_key: str, model: str,
                           tokens_in: int, tokens_out: int, latency_ms: int, status: int,
                           *, request_id: str | None = None,
                           provider: str | None = None,
                           session_id: str | None = None):
    """Record usage to DB and broadcast call + request_end events to SSE subscribers."""
    # Internal Ollama call sites provide the full token so activity refresh can
    # target the exact account even when tokens share a prefix. Redact it before
    # persistence, events, or logs.
    activity_key = manager.key_by_token(upstream_key) if provider == "ollama-cloud" and manager else None
    if provider == "ollama-cloud":
        upstream_key = upstream_key[:8]
    # End the live request first. Usage persistence is best-effort and must not
    # leave a completed request occupying the dashboard/in-flight registry.
    entry = _in_flight.pop(request_id, None) if request_id else None
    if usage_db:
        try:
            usage_db.record(client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status,
                            session_id=session_id or "")
        except Exception:
            log.exception(
                "Failed to persist usage for request %s (client=%s model=%s)",
                request_id or "unknown", client_id, model,
            )
    call_data = {
        "ts": time.time(),
        "client_id": client_id,
        "upstream_key": upstream_key,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "status": status,
        "session_id": session_id or "",
    }
    end_data: dict | None = None
    if request_id:
        end_data = {
            **call_data,
            "request_id": request_id,
            "provider": provider or (entry.get("target_provider") if entry else "ollama-cloud"),
        }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcaster.broadcast("call", call_data))
            if end_data is not None:
                asyncio.ensure_future(broadcaster.broadcast("request_end", end_data))
            if manager:
                asyncio.ensure_future(broadcaster.broadcast("status", {"keys": manager.status(), "upstream": upstream_url}))
        else:
            loop.run_until_complete(broadcaster.broadcast("call", call_data))
            if end_data is not None:
                loop.run_until_complete(broadcaster.broadcast("request_end", end_data))
            if manager:
                loop.run_until_complete(broadcaster.broadcast("status", {"keys": manager.status(), "upstream": upstream_url}))
    except RuntimeError:
        pass  # No event loop — skip broadcast

    if usage_refresher and activity_key is not None:
        usage_refresher.schedule(activity_key)


def _verify_admin(request: Request) -> None:
    """FastAPI dependency: require the admin token via a Bearer header."""
    global admin_token
    if not admin_token:
        raise HTTPException(status_code=500, detail="admin_token not configured")
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and secrets.compare_digest(auth[7:].strip(), admin_token):
        return
    raise HTTPException(status_code=401, detail="unauthorized")


def _verify_admin_session(session_token: str) -> None:
    now = time.time()
    expired = [token for token, expires_at in _admin_sessions.items() if expires_at <= now]
    for token in expired:
        _admin_sessions.pop(token, None)
    expires_at = _admin_sessions.get(session_token)
    if not expires_at or expires_at <= now:
        raise HTTPException(status_code=401, detail="invalid or expired admin session")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, key_registry, registry, usage_db, client_registry, usage_scraper, usage_refresher, telegram_notifier, upstream_http_client, fallback_provider, model_alias_manager, sticky
    global upstream_url, retry_on_429, max_retries, queue_timeout, request_timeout
    global admin_token, NATIVE_BRIDGE_MODELS, reject_unknown_models

    cfg = load_config()
    # Allow env var overrides (also applied in main(), but lifespan reloads
    # config independently — re-apply here so .env / Dokploy env vars win)
    if os.environ.get("LLAMAHERD_ADMIN_TOKEN"):
        cfg["admin_token"] = os.environ["LLAMAHERD_ADMIN_TOKEN"]
    if os.environ.get("LLAMAHERD_HOST"):
        cfg["host"] = os.environ["LLAMAHERD_HOST"]
    if os.environ.get("LLAMAHERD_PORT"):
        cfg["port"] = int(os.environ["LLAMAHERD_PORT"])
    admin_token = cfg.get("admin_token", "")
    if not admin_token:
        log.warning("admin_token not set in config — admin endpoints will be inaccessible")
    else:
        # Redact the token — never log even a prefix. Reviewer #1 on PR #2.
        log.info(f"Admin authentication enabled (token length: {len(admin_token)})")
    manager = KeyManager(cfg.get("keys", []))
    db_auth_token = cfg.get("db_auth_token") or os.environ.get("LLAMAHERD_DB_AUTH_TOKEN")
    db_auth_user = cfg.get("db_auth_user") or os.environ.get("LLAMAHERD_DB_AUTH_USER")
    # KeyRegistry: persist upstream keys + cookies to DB (survives restarts)
    key_registry = KeyRegistry(str(DB_PATH), seed_keys=cfg.get("keys"),
                               auth_token=db_auth_token, auth_user=db_auth_user)
    # Replace in-memory keys with DB-persisted keys
    db_keys = key_registry.all()
    if db_keys:
        manager = KeyManager(db_keys)
        log.info(f"Loaded {len(db_keys)} upstream keys from DB")
    client_registry = ClientRegistry(str(DB_PATH), seed_clients=cfg.get("clients"),
                                     auth_token=db_auth_token, auth_user=db_auth_user)
    sticky = StickySessionManager(ttl_seconds=cfg.get("sticky_ttl_seconds", 3600))
    upstream_url = cfg.get("upstream", "https://ollama.com/v1")
    retry_on_429 = cfg.get("retry_on_429", True)
    max_retries = cfg.get("max_retries", 2)
    queue_timeout = cfg.get("queue_timeout", 60)
    request_timeout = cfg.get("request_timeout", 120)
    upstream_http_client = httpx.AsyncClient(timeout=request_timeout)
    reject_unknown_models = cfg.get("reject_unknown_models", False)

    # Native bridge: models whose /v1 endpoint misreports truncation
    NATIVE_BRIDGE_MODELS = cfg.get("native_bridge_models", [])
    if NATIVE_BRIDGE_MODELS:
        log.info(f"Native bridge enabled for models: {NATIVE_BRIDGE_MODELS}")

    # usage_db resolution: LLAMAHERD_USAGE_DB env > config > LLAMAHERD_DB env
    # (shared DB) > default. The shared-DB fallback lets deployments use a
    # single SQLite/libSQL file for both client registry and usage tracking
    # when remote-DB support is added later.
    usage_dsn = (os.environ.get("LLAMAHERD_USAGE_DB")
                 or cfg.get("usage_db")
                 or os.environ.get("LLAMAHERD_DB")
                 or "~/ollama-cloud-proxy/usage.db")
    usage_db = UsageDB(usage_dsn, auth_token=db_auth_token, auth_user=db_auth_user)
    registry = ModelRegistry(
        manager,
        upstream_url,
        pricing_sync=_sync_pricing_from_openrouter,
        event_broadcaster=broadcaster,
    )
    await registry.start(cfg.get("health_check_interval", 300))
    await registry.refresh()
    # Initial subscription poll
    await manager.poll_subscriptions()
    # Start periodic sub polling (every 6 hours)
    sub_poll_interval = cfg.get("sub_poll_interval", 21600)
    sub_task = asyncio.create_task(_poll_subscriptions_loop(manager, sub_poll_interval))
    # Usage scraper (cookie-based ollama.com/settings)
    usage_scraper = UsageScraper(db_keys if db_keys else cfg["keys"])
    scrape_results = {}
    # Scrape usage on startup (in thread pool to not block)
    loop = asyncio.get_event_loop()
    try:
        scrape_results = await loop.run_in_executor(None, usage_scraper.scrape_all, manager.keys)
        if scrape_results:
            log.info(f"Initial usage scrape: {scrape_results}")
    except Exception as e:
        log.warning(f"Initial usage scrape failed: {e}")
    active_manager = manager

    async def broadcast_usage_update():
        await broadcaster.broadcast("status", {"keys": active_manager.status()})

    usage_refresher = ActivityUsageRefresher(
        usage_scraper,
        debounce_seconds=cfg.get("usage_activity_debounce", 30),
        min_interval_seconds=cfg.get("usage_activity_min_interval", 300),
        on_update=broadcast_usage_update,
    )
    usage_refresher.mark_scraped([key for key in manager.keys if key.label in scrape_results])
    # Telegram notifier (env-based, no UI)
    telegram_notifier = TelegramNotifier()
    telegram_task: asyncio.Task | None = None
    if telegram_notifier.enabled:
        log.info(f"Telegram notifications enabled (interval={telegram_notifier.interval}s, chat={telegram_notifier.chat_id})")
        telegram_task = asyncio.create_task(
            _telegram_notify_loop(telegram_notifier, manager, telegram_notifier.interval)
        )
    else:
        log.info("Telegram notifications not configured (set LLAMAHERD_TELEGRAM_TOKEN + LLAMAHERD_TELEGRAM_CHAT_ID to enable)")
    # Fallback provider (NVIDIA Build, etc.)
    fallback_provider = FallbackProvider(cfg.get("fallback") or {})
    # Model aliases (client-facing alternate names with context_length overrides)
    model_alias_manager = ModelAliasManager(cfg.get("model_aliases") or [])
    if model_alias_manager.aliases:
        log.info(f"Model aliases configured: {list(model_alias_manager.aliases.keys())}")
    fb_metadata_task: asyncio.Task | None = None
    if fallback_provider.enabled:
        # Best-effort discovery — don't block startup if it's slow.
        try:
            await asyncio.wait_for(fallback_provider.discover_models(timeout=5.0), timeout=6.0)
        except TimeoutError:
            log.warning("Fallback model discovery timed out")
        except Exception as e:
            log.warning(f"Fallback model discovery error: {e}")
        log.info(f"Fallback enabled: {fallback_provider.provider} ({len(fallback_provider._model_map)} mapped, priority={fallback_provider.priority})")
        # Kick off metadata enrichment in the background — never block startup.
        fb_metadata_task = asyncio.create_task(
            _refresh_fallback_metadata_loop(fallback_provider)
        )
    else:
        log.info("Fallback provider not configured")

    log.info(f"Proxy started: {len(manager.keys)} upstream keys ({len(usage_scraper.cookie_map)} with usage cookies), {len(registry.models)} models, {len(client_registry.clients)} clients")
    if reject_unknown_models:
        log.info("reject_unknown_models: enabled — unknown models will be rejected with 404")

    # Stale in-flight entry sweeper — removes zombies (0 tokens, stuck >10 min)
    sweep_task = asyncio.create_task(_sweep_stale_inflight(interval=300, max_age_seconds=600))

    # OpenRouter pricing sync: immediate fetch + periodic refresh every 24h
    pricing_sync_task = asyncio.create_task(_pricing_sync_loop(interval_hours=24.0))
    # Immediate sync on startup (don't block — best-effort)
    asyncio.create_task(_sync_pricing_from_openrouter())

    yield

    sub_task.cancel()
    if usage_refresher:
        await usage_refresher.close()
        usage_refresher = None
    if telegram_task is not None:
        telegram_task.cancel()
    sweep_task.cancel()
    pricing_sync_task.cancel()
    if fb_metadata_task is not None:
        fb_metadata_task.cancel()
    if registry:
        await registry.stop()
    if telegram_notifier:
        await telegram_notifier.close()
    if upstream_http_client:
        await upstream_http_client.aclose()
        upstream_http_client = None


async def _poll_subscriptions_loop(mgr: KeyManager, interval: int):
    """Periodically poll /api/me for each upstream key."""
    while True:
        await asyncio.sleep(interval)
        try:
            await mgr.poll_subscriptions()
            # Broadcast updated status after subscription poll
            await broadcaster.broadcast("status", {
                "keys": mgr.status(),
            })
        except Exception as e:
            log.error(f"Subscription poll loop error: {e}")


async def _refresh_fallback_metadata_loop(fp: 'FallbackProvider'):
    """Background task: enrich the fallback model catalog with docs metadata.

    Runs once shortly after startup, then weekly. Best-effort — all failures
    are swallowed so the proxy keeps running even if NVIDIA's docs API is down.
    """
    # Wait briefly so the rest of startup completes first.
    await asyncio.sleep(2.0)
    try:
        updated = await fp.refresh_metadata_cache(timeout=5.0)
        if updated:
            log.info(f"Fallback metadata cache: refreshed {updated} entries")
    except Exception as e:
        log.warning(f"Fallback metadata refresh error: {e}")
    # Weekly refresh loop.
    while True:
        try:
            await asyncio.sleep(7 * 86400)
            updated = await fp.refresh_metadata_cache(timeout=5.0)
            if updated:
                log.info(f"Fallback metadata cache: refreshed {updated} entries")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"Fallback metadata refresh error: {e}")


async def _telegram_notify_loop(notifier: TelegramNotifier, mgr: KeyManager, interval: int):
    """Periodically send usage notifications to Telegram."""
    while True:
        await asyncio.sleep(interval)
        try:
            ok = await notifier.send_usage_notification(mgr)
            if ok:
                log.info("Telegram notification sent")
            else:
                log.warning("Telegram notification failed")
        except Exception as e:
            log.error(f"Telegram notify loop error: {e}")


async def _sweep_stale_inflight(interval: int = 300, max_age_seconds: int = 600):
    """Periodically remove in-flight entries with 0 tokens that have been stuck too long.

    These are zombie entries from disconnected clients — the request never completed
    but the tracker entry was never removed. 5 minutes of 0 tokens = definitely stuck.
    Also releases the leaked KeyState.in_flight counter for each swept entry.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            now = time.time()
            stale_ids = [
                rid for rid, entry in _in_flight.items()
                if entry.get("tokens_in", 0) == 0
                and entry.get("tokens_out", 0) == 0
                and (now - entry.get("started_at", now)) > max_age_seconds
            ]
            if stale_ids:
                for rid in stale_ids:
                    entry = _in_flight.pop(rid, None)
                    if entry:
                        # Release the leaked KeyState.in_flight counter
                        target_key_label = entry.get("target_key")
                        if target_key_label:
                            for k in manager.keys:
                                if k.label == target_key_label and k.in_flight > 0:
                                    k.in_flight = max(0, k.in_flight - 1)
                                    log.info(f"Released leaked in_flight slot on {k.label} (now {k.in_flight})")
                                    break
                        log.warning(
                            f"Swept stale in-flight entry: {rid} model={entry.get('model')} "
                            f"client={entry.get('client_id')} "
                            f"age={int(now - entry.get('started_at', now))}s"
                        )
                log.info(f"Swept {len(stale_ids)} stale in-flight entries")
        except Exception as e:
            log.error(f"Stale in-flight sweep error: {e}")


app = FastAPI(title="Ollama Cloud Proxy", lifespan=lifespan)


_STREAM_CLEANUP_TIMEOUT_SECONDS = 30.0


async def _await_cleanup(
    awaitable,
    *,
    label: str,
    preserve_cancellation: bool = False,
    timeout: float = _STREAM_CLEANUP_TIMEOUT_SECONDS,
):
    """Run cleanup to completion despite repeated cancellation, with a deadline.

    Cancellation remains the primary outcome when cleanup itself fails.  A
    deadline prevents a broken close/release implementation from pinning an
    ASGI task forever, and the cleanup task is always reaped before returning.
    """
    cleanup = asyncio.create_task(awaitable, name=f"llamaherd-cleanup:{label}")
    cancelled = preserve_cancellation
    deadline = asyncio.get_running_loop().time() + timeout

    while not cleanup.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            cleanup.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(cleanup), timeout=1.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass
            if not cleanup.done():
                cleanup.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
            error = TimeoutError(f"{label} did not finish within {timeout:g}s")
            if cancelled:
                log.error("%s; preserving request cancellation", error)
                raise asyncio.CancelledError from error
            raise error
        try:
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=remaining)
        except asyncio.CancelledError:
            cancelled = True
        except TimeoutError:
            # Re-enter the loop so the deadline branch cancels and reaps the task.
            continue

    try:
        cleanup.result()
    except asyncio.CancelledError:
        if not cancelled:
            raise
    except Exception:
        if cancelled:
            log.exception("%s failed while preserving request cancellation", label)
        else:
            raise

    if cancelled:
        raise asyncio.CancelledError


class _OnceAsyncFinalizer:
    """Serialize and run a stream finalizer at most once."""

    def __init__(self, callback):
        self._callback = callback
        self._lock = asyncio.Lock()
        self.done = False

    async def __call__(self):
        async with self._lock:
            if self.done:
                return
            self.done = True
            await self._callback()


class _FinalizingStreamingResponse(StreamingResponse):
    """Finalize acquired resources even if body iteration never starts."""

    def __init__(self, *args, finalizer: _OnceAsyncFinalizer, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_finalizer = finalizer

    async def __call__(self, scope, receive, send):
        exit_exc = None
        try:
            return await super().__call__(scope, receive, send)
        except BaseException as exc:
            exit_exc = exc
            raise
        finally:
            try:
                await _await_cleanup(
                    self._stream_finalizer(),
                    label="response-level stream accounting",
                    preserve_cancellation=isinstance(exit_exc, asyncio.CancelledError),
                )
            except BaseException as cleanup_exc:
                if exit_exc is None:
                    raise
                if not isinstance(cleanup_exc, asyncio.CancelledError):
                    log.exception("Response-level stream finalization failed while preserving response exit")
            if exit_exc is not None:
                raise exit_exc.with_traceback(exit_exc.__traceback__)


@asynccontextmanager
async def _cancellation_safe_stream(client: httpx.AsyncClient, method: str, url: str, **kwargs):
    """Keep response cleanup alive when an ASGI stream is cancelled repeatedly."""
    stream_context = client.stream(method, url, **kwargs)
    response = await stream_context.__aenter__()
    exc_info = (None, None, None)
    try:
        yield response
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        try:
            await _await_cleanup(
                stream_context.__aexit__(*exc_info),
                label="upstream response close",
                preserve_cancellation=isinstance(exc_info[1], asyncio.CancelledError),
            )
        except BaseException as cleanup_exc:
            if exc_info[1] is None:
                raise
            if not isinstance(cleanup_exc, asyncio.CancelledError):
                log.exception("Upstream response close failed while preserving stream exit")
        if exc_info[1] is not None:
            raise exc_info[1].with_traceback(exc_info[2])


STATIC_DIR = Path(__file__).parent / "static"
DASHBOARD_PATH = STATIC_DIR / "dashboard.html"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_secure_cookie_context: ContextVar[bool] = ContextVar("secure_cookie", default=False)


@app.middleware("http")
async def session_cookie_security(request: Request, call_next):
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    token = _secure_cookie_context.set(request.url.scheme == "https" or forwarded_proto == "https")
    try:
        return await call_next(request)
    finally:
        _secure_cookie_context.reset(token)


@app.get("/healthz")
async def healthz():
    """Unauthenticated liveness endpoint for Docker/systemd health checks."""
    return {"status": "ok"}


def _resolve_client(request: Request) -> dict:
    """Extract Bearer token from request and resolve to client identity.

    Public model/proxy endpoints require a valid registered LlamaHerd client
    token. Admin endpoints use the separate admin token dependency.
    """
    auth = request.headers.get("authorization", "")
    if not auth:
        raise HTTPException(
            status_code=401,
            detail="missing_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="invalid_authorization_header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth[7:].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="missing_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if client_registry is None:
        raise HTTPException(status_code=503, detail="client_registry_not_ready")
    client = client_registry.resolve(token)
    if client is None:
        peer = request.client.host if request.client else "unknown"
        log.warning(
            "Rejected request with unknown client token: source=%s token_prefix=%s...",
            peer,
            token[:8],
        )
        raise HTTPException(status_code=403, detail="invalid_api_key")
    return client


def _extract_session_id(request: Request, body_json: dict | None = None) -> str | None:
    """Extract or generate a session id for sticky routing.

    This implements an **optimistic cache-affinity strategy**: LlamaHerd assumes
    Ollama Cloud keeps a per-subscriber KV/context cache for recent requests
    (especially important for 1M-context and long-context models). When a
    stable session identifier is unavailable from the client, we derive one
    from the leading conversation context so repeated turns in the same chat
    route to the same upstream subscription and maximize the chance of a warm
    cache hit.

    Priority:
    1. X-LlamaHerd-Session request header
    2. llamaherd-session cookie
    3. X-Conversation-ID header (common alias)
    4. Hash of the first system + user messages (fallback for OpenAI clients)
    5. None — caller will generate a random id
    """
    sid = request.headers.get("x-llamaherd-session") or request.headers.get("x-conversation-id")
    if sid:
        return sid.strip()
    try:
        cookie = request.cookies.get("llamaherd-session")
        if cookie:
            return cookie.strip()
    except Exception:
        pass

    # Fallback: derive a stable key from the leading conversation context.
    # We hash the first user message plus the first system message (if any).
    # This keeps turns from the same chat on the same sub for KV-cache reuse.
    if body_json and isinstance(body_json, dict):
        messages = body_json.get("messages") or body_json.get("prompt") or []
        if isinstance(messages, list):
            seed_parts = []
            for msg in messages[:3]:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content") or ""
                    if role in ("system", "user") and content:
                        seed_parts.append(f"{role}:{content[:120]}")
            if seed_parts:
                seed = "|".join(seed_parts)
                # Prefix with client id so different clients using identical prompts don't collide
                client_hint = request.headers.get("authorization", "")[:16]
                return "lh_ctx_" + hashlib.sha256(f"{client_hint}:{seed}".encode()).hexdigest()[:24]
    return None


def _session_cookie_for_response(session_id: str, ttl: int) -> str:
    """Build a Set-Cookie header for the sticky session."""
    cookie = (
        f"llamaherd-session={session_id}; "
        f"Max-Age={ttl}; "
        f"Path=/; "
        f"HttpOnly; "
        f"SameSite=Lax"
    )
    if _secure_cookie_context.get():
        cookie += "; Secure"
    return cookie


async def _proxy_request(request: Request, path: str) -> Response:
    """Core proxy logic — acquire key, forward, handle 429s, release."""
    global _LAST_DISCOVERY_REFRESH

    client = _resolve_client(request)
    client_id = client["id"]

    # Check per-client rate limits before doing any upstream work
    rate_limit_response = await _check_rate_limit(request, client)
    if rate_limit_response is not None:
        return rate_limit_response

    request_id = _new_request_id()
    body = await request.body()
    try:
        req_json = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON request body"})
    session_id = _extract_session_id(request, req_json)
    if not session_id:
        session_id = "lh_" + secrets.token_urlsafe(16)

    is_stream = req_json.get("stream", False)
    model = req_json.get("model", "unknown")

    # --- Model alias resolution ---
    # If the requested model is an alias (e.g. "glm-5.2-256k"), rewrite the
    # request body to use the upstream model name (e.g. "glm-5.2") before
    # forwarding to Ollama Cloud.  The alias name is kept in the local
    # `model` variable for usage tracking and logging.  `resolved_model`
    # is what we use for registry/fallback lookups.
    resolved_model = model
    if model_alias_manager and model_alias_manager.is_alias(model):
        resolved_model, _ctx_override = model_alias_manager.resolve(model)
        req_json["model"] = resolved_model
        body = json.dumps(req_json).encode()
        log.info(f"Alias: {model} → {resolved_model} (client={client_id})")

    # Inject stream_options.include_usage = True for streaming requests
    # so upstream returns the real token count in the final chunk
    if is_stream and "stream_options" not in req_json:
        req_json["stream_options"] = {"include_usage": True}
        body = json.dumps(req_json).encode()
    elif is_stream:
        # stream_options already present — make sure include_usage is True
        so = req_json.get("stream_options", {})
        if not so.get("include_usage", False):
            so["include_usage"] = True
            req_json["stream_options"] = so
            body = json.dumps(req_json).encode()

    # Fallback routing decision (priority before/only/after).
    fp = fallback_provider
    has_fallback = bool(fp and fp.enabled)
    # Strip :cloud suffix for registry lookup — Ollama Cloud may report
    # model names without :cloud, but clients request "model:cloud".
    model_base = resolved_model.replace(":cloud", "").replace(":cloud-", "-")
    ollama_has_model = bool(registry and (registry.models.get(resolved_model) or registry.models.get(model_base)))
    fp_mapped = bool(has_fallback and fp.resolve_model(resolved_model))
    fp_can_serve = bool(has_fallback and (fp.resolve_model(resolved_model) or fp.default_model))
    priority = fp.priority_for(resolved_model) if has_fallback else "after"

    # Reject unknown models: when reject_unknown_models is true, models
    # that aren't known to Ollama AND aren't in the fallback model_map
    # get a 404 instead of silently routing to the fallback default_model.
    # But first, try an immediate registry + pricing refresh to discover
    # newly-available models before rejecting.
    if reject_unknown_models and not ollama_has_model and not fp_mapped and registry:
        # Attempt an immediate discovery refresh for the unknown model.
        # This catches models that appeared on Ollama Cloud after our last
        # periodic refresh (every 5 min for registry, 24h for pricing).
        # Cooldown: don't refresh more than once per 60 seconds to avoid
        # latency spikes on burst requests for the same unknown model.
        now = time.time()
        if now - _LAST_DISCOVERY_REFRESH > 60.0:
            try:
                log.info(f"Unknown model '{resolved_model}' requested — triggering immediate registry refresh + pricing sync")
                _LAST_DISCOVERY_REFRESH = now
                await registry.refresh()
                await _sync_pricing_from_openrouter()
                # Re-check after refresh
                ollama_has_model = bool(registry.models.get(resolved_model) or registry.models.get(model_base))
                fp_mapped = bool(fp and fp.enabled and fp.resolve_model(resolved_model))
            except Exception as e:
                log.warning(f"Immediate discovery refresh failed for '{resolved_model}': {e}")

    if reject_unknown_models and not ollama_has_model and not fp_mapped:
        _record_and_broadcast(client_id, "none", model, 0, 0, 0, 404,
                              request_id=request_id, provider="proxy", session_id=session_id)
        log.warning(f"Rejected unknown model: {model} (client={client_id})")
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{model}' not found. Available models: /v1/models"},
        )

    # priority=only — fallback only for mapped models; Ollama for others.
    if has_fallback and priority == "only" and fp_mapped:
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id, session_id=session_id)
    # priority=before — try fallback first when a mapping exists.
    if has_fallback and priority == "before" and fp_mapped:
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id, session_id=session_id)
    # priority=after — model unknown to Ollama but fallback can serve it.
    if has_fallback and priority == "after" and not ollama_has_model and fp_can_serve:
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id, session_id=session_id)

    prefer_key = registry.get_preferred_key(resolved_model) if registry else None
    sticky_key = await sticky.get_preferred_key(session_id) if sticky else None
    log.info(f"Sticky routing for {client_id} / {model}: session_id={session_id[:16]}... sticky_key={sticky_key[:8]+'...' if sticky_key else None}")

    last_error = None
    start_emitted = False
    for attempt in range(max_retries + 1):
        key = None
        deadline = time.time() + queue_timeout
        while time.time() < deadline:
            key = await manager.acquire(prefer_key=prefer_key, sticky_key=sticky_key)
            if key:
                break
            await asyncio.sleep(0.5)

        if not key:
            # All keys at capacity — fall back if priority allows it.
            if has_fallback and fp_can_serve and priority in ("after", "before") and not ollama_has_model:
                log.warning(f"Ollama keys at capacity for {model}; falling back to {fp.label}")
                return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id, session_id=session_id)
            if has_fallback and fp_can_serve and priority in ("after", "before") and ollama_has_model:
                log.warning(
                    f"Ollama keys at capacity for known Ollama model {model}; "
                    "not falling back to secondary provider"
                )
            _record_and_broadcast(client_id, "none", model, 0, 0, 0, 503,
                                  request_id=request_id, provider="ollama-cloud", session_id=session_id)
            return JSONResponse(
                status_code=503,
                content={"error": "all keys at capacity, queue timeout exceeded"},
            )

        if not start_emitted:
            _request_start(
                request_id, client_id, model, key.label, "ollama-cloud",
                headers=dict(request.headers), path=path,
            )
            start_emitted = True

        # Pin this session to the chosen sub (new or refreshed TTL).
        # Do not rebind sticky onto a worse-weekly temporary alternate.
        if sticky and session_id and manager.should_rebind_sticky(sticky_key, key):
            await sticky.set_session(session_id, key.token)
            sticky_key = key.token
            # else: keep previous sticky mapping; this request is a one-shot spill

        try:
            start = time.time()
            headers = {
                "Authorization": f"Bearer {key.token}",
                "Content-Type": "application/json",
            }

            # Native bridge: re-route GLM models via /api/chat to get correct
            # done_reason: "length" instead of the buggy finish_reason: "stop"
            if is_stream and _should_bridge_to_native(model):
                bridge_body = _convert_openai_to_ollama_body(req_json)
                log.info(f"Bridge: {model} via native /api/chat (client={client_id})")
                return await _proxy_bridge_stream(client_id, key, bridge_body, model, start, request_id, session_id=session_id)

            if is_stream:
                return await _proxy_stream(client_id, key, path, headers, body, model, start, request_id, session_id=session_id)

            resp = await upstream_http_client.post(
                f"{upstream_url}{path}",
                content=body,
                headers=headers,
            )

            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code == 429:
                log.warning(f"429 from {key.label} for {model} (client={client_id}, attempt {attempt+1})")
                await manager.mark_429(key)
                await manager.release(key)
                if sticky and session_id:
                    await sticky.clear_session(session_id)
                _record_and_broadcast(client_id, key.token, model, 0, 0, elapsed_ms, 429, request_id=request_id, provider="ollama-cloud", session_id=session_id)
                prefer_key = None
                sticky_key = None
                continue

            if resp.status_code == 402:
                log.warning(f"402 from {key.label} for {model} (client={client_id})")
                await manager.mark_402(key)
                await manager.release(key)
                if sticky and session_id:
                    await sticky.clear_session(session_id)
                _record_and_broadcast(client_id, key.token, model, 0, 0, elapsed_ms, 402, request_id=request_id, provider="ollama-cloud", session_id=session_id)
                prefer_key = None
                sticky_key = None
                continue

            resp_data = resp.json() if resp.status_code == 200 else {}
            usage = resp_data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token, model, tokens_in, tokens_out, elapsed_ms,
                                  resp.status_code, request_id=request_id, provider="ollama-cloud", session_id=session_id)

            log.info(f"{client_id} -> {model} via {key.label}: {tokens_in}+{tokens_out}tok {elapsed_ms}ms")

            if resp.status_code >= 400:
                log.warning(f"{resp.status_code} from {key.label} for {model}: {resp.text[:200]}")

            resp_headers = dict(resp.headers)
            if sticky and session_id and resp.status_code == 200:
                resp_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
                resp_headers["X-LlamaHerd-Session"] = session_id
                resp_headers["X-LlamaHerd-Key"] = key.label
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key)
            if sticky and session_id:
                await sticky.clear_session(session_id)
            _record_and_broadcast(client_id, key.token, model, 0, 0, elapsed_ms, -1, request_id=request_id, provider="ollama-cloud", session_id=session_id)
            last_error = str(e)
            log.error(f"Proxy error for {model} (client={client_id}): {e}")
            prefer_key = None
            sticky_key = None
            continue

    # Ollama exhausted retries — try fallback as a last resort.
    if has_fallback and fp_can_serve and priority in ("after", "before") and not ollama_has_model:
        log.warning(f"Ollama exhausted retries for {model}; falling back to {fp.label}")
        # Pop the Ollama in-flight entry so the fallback emits a fresh start.
        _in_flight.pop(request_id, None)
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id, session_id=session_id)
    if has_fallback and fp_can_serve and priority in ("after", "before") and ollama_has_model:
        log.warning(
            f"Ollama exhausted retries for known Ollama model {model}; "
            "not falling back to secondary provider"
        )

    _record_and_broadcast(client_id, "none", model, 0, 0, 0, 502,
                          request_id=request_id, provider="ollama-cloud", session_id=session_id)
    return JSONResponse(
        status_code=502,
        content={"error": f"all retries exhausted: {last_error}"},
    )


async def _proxy_stream(client_id: str, key: KeyState, path: str,
                         headers: dict, body: bytes, model: str, start: float,
                         request_id: str | None = None,
                         session_id: str | None = None) -> StreamingResponse:

    tokens_out = 0
    tokens_in = 0
    usage_captured = False
    final_status = 200

    async def finalize():
        elapsed_ms = int((time.time() - start) * 1000)
        await manager.release(key, tokens_out)
        _record_and_broadcast(client_id, key.token, model, tokens_in, tokens_out, elapsed_ms,
                              final_status, request_id=request_id, provider="ollama-cloud", session_id=session_id)
        usage_src = "usage" if usage_captured else "estimate"
        status_suffix = "" if final_status == 200 else f" status={final_status}"
        log.info(f"{client_id} -> {model} via {key.label}: stream {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({usage_src}){status_suffix}")

    finalizer = _OnceAsyncFinalizer(finalize)

    async def generate():
        nonlocal tokens_out, tokens_in, usage_captured, final_status
        try:
            async with _cancellation_safe_stream(
                upstream_http_client, "POST", f"{upstream_url}{path}",
                content=body, headers=headers,
            ) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        final_status = 429
                        yield 'data: {"error": "429 from upstream"}\n\n'
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
                        final_status = 402
                        yield 'data: {"error": "402 from upstream"}\n\n'
                        return

                    async for line in resp.aiter_lines():
                        yield line + "\n\n" if line.startswith("data:") else line + "\n"
                        if line.startswith("data:"):
                            try:
                                payload_text = line[5:].strip()
                                if payload_text == "[DONE]":
                                    continue
                                chunk = json.loads(payload_text)
                                # Capture usage from final chunk (we requested include_usage)
                                chunk_usage = chunk.get("usage")
                                if chunk_usage and chunk_usage.get("total_tokens", 0) > 0:
                                    tokens_in = chunk_usage.get("prompt_tokens", 0)
                                    tokens_out = chunk_usage.get("completion_tokens", 0)
                                    usage_captured = True
                                    _update_in_flight_tokens(request_id, tokens_in, tokens_out)
                                else:
                                    # Live progress: estimate completion tokens from delta content.
                                    choices = chunk.get("choices") or []
                                    if choices:
                                        delta = choices[0].get("delta") or {}
                                        content = delta.get("content")
                                        if content:
                                            tokens_out += max(1, len(content) // 4)
                                            _update_in_flight_tokens(request_id, None, tokens_out)
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
        except Exception as e:
            final_status = -1
            error_type = type(e).__name__
            error_message = str(e) or repr(e)
            log.error(
                f"Stream error for {model} (client={client_id}): "
                f"{error_type}: {error_message}"
            )
            # The HTTP response has already started for SSE streams, so we cannot
            # change the status code. Emit a structured SSE error and record the
            # usage row as failed so dashboards/DB queries don't show fake 200s.
            error_payload = json.dumps({
                "error": "upstream_stream_error",
                "error_type": error_type,
                "detail": error_message,
            })
            yield f"data: {error_payload}\n\n"
        finally:
            await _await_cleanup(finalizer(), label="OpenAI stream accounting")

    stream_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    if sticky and session_id:
        stream_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
        stream_headers["X-LlamaHerd-Session"] = session_id
        stream_headers["X-LlamaHerd-Key"] = key.label
    return _FinalizingStreamingResponse(
        generate(), media_type="text/event-stream", headers=stream_headers, finalizer=finalizer,
    )


# ---------------------------------------------------------------------------
# Fallback routing (e.g. NVIDIA Build) — same OpenAI /v1 protocol
# ---------------------------------------------------------------------------

async def _route_to_fallback(client_id: str, fp: FallbackProvider, path: str,
                              body: bytes, req_json: dict, original_model: str,
                              is_stream: bool, request_id: str | None = None,
                              session_id: str | None = None) -> Response:
    """Forward an OpenAI-style request to the fallback provider.

    Rewrites the model name in the request body using fp.resolve_model().
    Returns a Response (or StreamingResponse for is_stream).
    """
    mapped = fp.resolve_model(original_model) or fp.default_model
    if not mapped:
        raise RuntimeError(f"no fallback model mapping for {original_model}")
    new_req = dict(req_json)
    new_req["model"] = mapped
    new_body = json.dumps(new_req).encode()
    headers = {
        "Authorization": f"Bearer {fp.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{fp.base_url}{path}"
    upstream_label = f"fb:{fp.label}"
    start = time.time()

    if request_id is None:
        request_id = _new_request_id()
    _request_start(
        request_id, client_id, original_model, upstream_label, fp.provider,
        path=path,
    )

    if is_stream:
        return await _proxy_fallback_stream(
            client_id, fp, url, headers, new_body, original_model, mapped, start, upstream_label,
            request_id, session_id=session_id,
        )

    resp = await upstream_http_client.post(url, content=new_body, headers=headers)
    elapsed_ms = int((time.time() - start) * 1000)
    tokens_in = tokens_out = 0
    if resp.status_code == 200:
        try:
            data = resp.json()
            usage = data.get("usage") or {}
            tokens_in = usage.get("prompt_tokens", 0) or 0
            tokens_out = usage.get("completion_tokens", 0) or 0
        except Exception:
            pass
    _record_and_broadcast(client_id, upstream_label, original_model,
                          tokens_in, tokens_out, elapsed_ms, resp.status_code,
                          request_id=request_id, provider=fp.provider, session_id=session_id)
    log.info(f"{client_id} -> {original_model} via {upstream_label}({mapped}): "
             f"{tokens_in}+{tokens_out}tok {elapsed_ms}ms")
    if resp.status_code >= 400:
        log.warning(f"{resp.status_code} from {fp.label} for {original_model}: {resp.text[:200]}")
    safe_headers = {k: v for k, v in resp.headers.items()
                    if k.lower() not in ("content-encoding", "content-length", "transfer-encoding", "connection")}
    return Response(content=resp.content, status_code=resp.status_code, headers=safe_headers)


async def _proxy_fallback_stream(client_id: str, fp: FallbackProvider, url: str,
                                  headers: dict, body: bytes, original_model: str,
                                  mapped: str, start: float, upstream_label: str,
                                  request_id: Optional[str] = None,
                                  session_id: Optional[str] = None) -> StreamingResponse:
    tokens_in = 0
    tokens_out = 0
    usage_captured = False
    status_code = 200

    async def finalize():
        elapsed_ms = int((time.time() - start) * 1000)
        _record_and_broadcast(client_id, upstream_label, original_model,
                              tokens_in, tokens_out, elapsed_ms, status_code,
                              request_id=request_id, provider=fp.provider, session_id=session_id)
        src = "usage" if usage_captured else "estimate"
        log.info(f"{client_id} -> {original_model} via {upstream_label}({mapped}): "
                 f"stream {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({src})")

    finalizer = _OnceAsyncFinalizer(finalize)
    async def generate():
        nonlocal tokens_in, tokens_out, usage_captured, status_code
        try:
            async with _cancellation_safe_stream(
                upstream_http_client, "POST", url, content=body, headers=headers,
            ) as resp:
                    status_code = resp.status_code
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        yield f'data: {err.decode(errors="replace")}\n\n'
                        return
                    async for line in resp.aiter_lines():
                        yield (line + "\n\n") if line.startswith("data:") else (line + "\n")
                        if line.startswith("data:"):
                            try:
                                payload_text = line[5:].strip()
                                if payload_text == "[DONE]":
                                    continue
                                chunk = json.loads(payload_text)
                                chunk_usage = chunk.get("usage")
                                if chunk_usage and chunk_usage.get("total_tokens", 0) > 0:
                                    tokens_in = chunk_usage.get("prompt_tokens", 0)
                                    tokens_out = chunk_usage.get("completion_tokens", 0)
                                    usage_captured = True
                                    _update_in_flight_tokens(request_id, tokens_in, tokens_out)
                                else:
                                    choices = chunk.get("choices") or []
                                    if choices:
                                        delta = choices[0].get("delta") or {}
                                        content = delta.get("content")
                                        if content:
                                            tokens_out += max(1, len(content) // 4)
                                            _update_in_flight_tokens(request_id, None, tokens_out)
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
        except Exception as e:
            log.error(f"Fallback stream error for {original_model}: {e}")
        finally:
            await _await_cleanup(finalizer(), label="fallback stream accounting")

    bridge_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    if sticky and session_id:
        bridge_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
        bridge_headers["X-LlamaHerd-Session"] = session_id
        bridge_headers["X-LlamaHerd-Key"] = upstream_label
    return _FinalizingStreamingResponse(
        generate(), media_type="text/event-stream", headers=bridge_headers, finalizer=finalizer,
    )


# ---------------------------------------------------------------------------
# Proxy — Native Ollama API (NDJSON streaming)
# ---------------------------------------------------------------------------

def _native_api_upstream() -> str:
    """Derive the native Ollama API upstream URL from the OpenAI upstream.

    If upstream_url is 'https://ollama.com/v1', native API is 'https://ollama.com/api'.
    """
    base = upstream_url.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3] + "/api"
    return base + "/api"


async def _proxy_ndjson_stream(client_id: str, key: 'KeyState', path: str,
                                headers: dict, body: bytes, model: str,
                                start: float, request_id: str | None = None,
                                session_id: str | None = None) -> StreamingResponse:
    """Stream NDJSON from the native Ollama API, capturing usage from the final chunk."""

    tokens_out = 0
    tokens_in = 0
    usage_captured = False
    final_status = 200
    api_upstream = _native_api_upstream()

    async def finalize():
        elapsed_ms = int((time.time() - start) * 1000)
        await manager.release(key, tokens_out)
        _record_and_broadcast(client_id, key.token, model, tokens_in, tokens_out, elapsed_ms,
                              final_status, request_id=request_id, provider="ollama-cloud", session_id=session_id)
        usage_src = "usage" if usage_captured else "estimate"
        log.info(f"{client_id} -> {model} via {key.label}: ndjson {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({usage_src})")

    finalizer = _OnceAsyncFinalizer(finalize)

    async def generate():
        nonlocal tokens_out, tokens_in, usage_captured, final_status
        try:
            async with _cancellation_safe_stream(
                upstream_http_client, "POST", f"{api_upstream}{path}",
                content=body, headers=headers,
            ) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        if sticky and session_id:
                            await sticky.clear_session(session_id)
                        final_status = 429
                        yield json.dumps({"error": "429 from upstream"}) + "\n"
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
                        if sticky and session_id:
                            await sticky.clear_session(session_id)
                        final_status = 402
                        yield json.dumps({"error": "402 from upstream"}) + "\n"
                        return
                    if resp.status_code >= 400:
                        # For non-2xx, read the body and yield as a single NDJSON line
                        final_status = resp.status_code
                        error_body = await resp.aread()
                        yield error_body.decode(errors="replace").strip() + "\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        # Yield the raw NDJSON line
                        yield line + "\n"
                        # Try to parse for usage capture
                        try:
                            chunk = json.loads(line)
                            if chunk.get("done", False):
                                # Final chunk — extract usage
                                pev = chunk.get("prompt_eval_count")
                                ev = chunk.get("eval_count")
                                if pev is not None:
                                    tokens_in = int(pev)
                                if ev is not None:
                                    tokens_out = int(ev)
                                if tokens_in > 0 or tokens_out > 0:
                                    usage_captured = True
                                    _update_in_flight_tokens(request_id, tokens_in, tokens_out)
                            else:
                                # Live progress: estimate completion tokens from chunk content.
                                msg = chunk.get("message") or {}
                                content = msg.get("content") or chunk.get("response") or ""
                                if content:
                                    tokens_out += max(1, len(content) // 4)
                                    _update_in_flight_tokens(request_id, None, tokens_out)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
        except Exception as e:
            if sticky and session_id:
                await sticky.clear_session(session_id)
            log.error(f"NDJSON stream error for {model} (client={client_id}): {e}")
        finally:
            await _await_cleanup(finalizer(), label="native stream accounting")

    ndjson_headers = {"Content-Type": "application/x-ndjson"}
    if sticky and session_id:
        ndjson_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
        ndjson_headers["X-LlamaHerd-Session"] = session_id
        ndjson_headers["X-LlamaHerd-Key"] = key.label
    return _FinalizingStreamingResponse(
        generate(), media_type="application/x-ndjson", headers=ndjson_headers, finalizer=finalizer,
    )


async def _proxy_ndjson_request(request: Request, path: str) -> Response:
    """Core proxy logic for native Ollama /api/* routes — acquire key, forward, handle 429s, release.

    Handles both streaming (NDJSON) and non-streaming (JSON) native Ollama API requests.
    """
    client = _resolve_client(request)
    client_id = client["id"]

    request_id = _new_request_id()
    body = await request.body()
    try:
        req_json = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON request body"})
    model = req_json.get("model", "unknown")
    is_stream = req_json.get("stream", False)

    # --- Model alias resolution ---
    # Same as /v1 path: rewrite req_json["model"] to the upstream model,
    # keep the alias name for usage tracking.
    resolved_model = model
    if model_alias_manager and model_alias_manager.is_alias(model):
        resolved_model, _ctx_override = model_alias_manager.resolve(model)
        req_json["model"] = resolved_model
        body = json.dumps(req_json).encode()
        log.info(f"Alias: {model} → {resolved_model} (client={client_id})")

    prefer_key = registry.get_preferred_key(resolved_model) if registry else None
    session_id = _extract_session_id(request, req_json)
    if not session_id:
        session_id = "lh_" + secrets.token_urlsafe(16)
    sticky_key = await sticky.get_preferred_key(session_id) if sticky else None
    log.info(f"Sticky routing for {client_id} / {model}: session_id={session_id[:16]}... sticky_key={sticky_key[:8]+'...' if sticky_key else None}")

    last_error = None
    start_emitted = False
    for attempt in range(max_retries + 1):
        key = None
        deadline = time.time() + queue_timeout
        while time.time() < deadline:
            key = await manager.acquire(prefer_key=prefer_key, sticky_key=sticky_key)
            if key:
                break
            await asyncio.sleep(0.5)

        if not key:
            _record_and_broadcast(client_id, "none", model, 0, 0, 0, 503,
                                  request_id=request_id, provider="ollama-cloud", session_id=session_id)
            return JSONResponse(
                status_code=503,
                content={"error": "all keys at capacity, queue timeout exceeded"},
            )

        if not start_emitted:
            _request_start(
                request_id, client_id, model, key.label, "ollama-cloud",
                headers=dict(request.headers), path=path,
            )
            start_emitted = True

        # Pin this native session to the chosen sub (new or refreshed TTL).
        # Do not rebind sticky onto a worse-weekly temporary alternate.
        if sticky and session_id and manager.should_rebind_sticky(sticky_key, key):
            await sticky.set_session(session_id, key.token)
            sticky_key = key.token
            # else: keep previous sticky mapping; this request is a one-shot spill

        try:
            start = time.time()
            headers = {
                "Authorization": f"Bearer {key.token}",
                "Content-Type": "application/json",
            }
            api_upstream = _native_api_upstream()

            if is_stream:
                return await _proxy_ndjson_stream(client_id, key, path, headers, body, model, start, request_id, session_id=session_id)

            # Non-streaming: regular JSON response
            resp = await upstream_http_client.post(
                f"{api_upstream}{path}",
                content=body,
                headers=headers,
            )

            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code == 429:
                log.warning(f"429 from {key.label} for {model} (client={client_id}, attempt {attempt+1})")
                await manager.mark_429(key)
                await manager.release(key)
                if sticky and session_id:
                    await sticky.clear_session(session_id)
                _record_and_broadcast(client_id, key.token, model, 0, 0, elapsed_ms, 429, request_id=request_id, provider="ollama-cloud", session_id=session_id)
                prefer_key = None
                sticky_key = None
                continue

            if resp.status_code == 402:
                log.warning(f"402 from {key.label} for {model} (client={client_id})")
                await manager.mark_402(key)
                await manager.release(key)
                if sticky and session_id:
                    await sticky.clear_session(session_id)
                _record_and_broadcast(client_id, key.token, model, 0, 0, elapsed_ms, 402, request_id=request_id, provider="ollama-cloud", session_id=session_id)
                prefer_key = None
                sticky_key = None
                continue

            # Extract usage from non-streaming response
            resp_data = resp.json() if resp.status_code == 200 else {}
            tokens_in = resp_data.get("prompt_eval_count", 0) or 0
            tokens_out = resp_data.get("eval_count", 0) or 0
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token, model, tokens_in, tokens_out, elapsed_ms,
                                  resp.status_code, request_id=request_id, provider="ollama-cloud", session_id=session_id)

            log.info(f"{client_id} -> {model} via {key.label}: {tokens_in}+{tokens_out}tok {elapsed_ms}ms (native)")

            if resp.status_code >= 400:
                log.warning(f"{resp.status_code} from {key.label} for {model}: {resp.text[:200]}")

            resp_headers = dict(resp.headers)
            if sticky and session_id and resp.status_code == 200:
                resp_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
                resp_headers["X-LlamaHerd-Session"] = session_id
                resp_headers["X-LlamaHerd-Key"] = key.label
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key)
            if sticky and session_id:
                await sticky.clear_session(session_id)
            _record_and_broadcast(client_id, key.token, model, 0, 0, elapsed_ms, -1, request_id=request_id, provider="ollama-cloud", session_id=session_id)
            last_error = str(e)
            log.error(f"Native proxy error for {model} (client={client_id}): {e}")
            prefer_key = None
            sticky_key = None
            continue

    _record_and_broadcast(client_id, "none", model, 0, 0, 0, 502,
                          request_id=request_id, provider="ollama-cloud", session_id=session_id)
    return JSONResponse(
        status_code=502,
        content={"error": f"all retries exhausted: {last_error}"},
    )


# ---------------------------------------------------------------------------
# Routes — OpenAI-compatible
# ---------------------------------------------------------------------------

@app.get("/v1/models")
@app.get("/v1/models/")
async def list_models(request: Request):
    _resolve_client(request)
    base = registry.get_models_response() if registry else {"object": "list", "data": []}
    data = list(base.get("data") or [])
    seen = {entry.get("id") for entry in data}
    # Tag Ollama-Cloud-discovered entries with provider for parity with fallback rows.
    for entry in data:
        entry.setdefault("provider", "ollama-cloud")
    if fallback_provider and fallback_provider.enabled:
        for alias in fallback_provider.model_aliases():
            if alias["id"] in seen:
                # Model exists on both — annotate the existing row instead of duplicating.
                for entry in data:
                    if entry.get("id") == alias["id"]:
                        entry["provider"] = f"ollama-cloud,{fallback_provider.provider}"
                        entry["fallback_model"] = alias["nvidia_model"]
                        break
                continue
            data.append({
                "id": alias["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": fallback_provider.provider,
                "provider": fallback_provider.provider,
                "fallback_model": alias["nvidia_model"],
            })
            seen.add(alias["id"])
    # Inject model aliases (client-facing alternate names with context_length overrides)
    if model_alias_manager:
        existing_ids = {m.get("id") for m in data}
        for ae in model_alias_manager.alias_entries():
            alias_name = ae["alias"]
            upstream = ae["upstream_model"]
            ctx = ae["context_length"]
            # Skip self-aliases when the upstream model is already listed —
            # patch the existing entry instead of duplicating.
            if alias_name == upstream and alias_name in existing_ids:
                for m in data:
                    if m.get("id") == alias_name:
                        if ctx:
                            m["context_length"] = ctx
                        m["aliased_model"] = upstream
                        break
                continue
            # Copy metadata from the upstream model if it exists
            upstream_meta = registry.model_metadata.get(upstream, {}) if registry else {}
            entry: dict = {
                "id": alias_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "provider": "ollama-cloud",
                "aliased_model": upstream,
            }
            if ctx:
                entry["context_length"] = ctx
            # Copy other metadata from upstream (capabilities, family, etc.)
            for key in ("capabilities", "family", "parameter_count", "quantization_level"):
                if upstream_meta.get(key) is not None:
                    entry[key] = upstream_meta[key]
            data.append(entry)
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, request: Request):
    _resolve_client(request)
    # Check model aliases first
    if model_alias_manager and model_alias_manager.is_alias(model_id):
        upstream, ctx_override = model_alias_manager.resolve(model_id)
        upstream_meta = registry.model_metadata.get(upstream, {}) if registry else {}
        entry: dict = {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama",
            "provider": "ollama-cloud",
            "aliased_model": upstream,
        }
        ctx = ctx_override or upstream_meta.get("context_length")
        if ctx:
            entry["context_length"] = ctx
        for key in ("capabilities", "family", "parameter_count", "quantization_level"):
            if upstream_meta.get(key) is not None:
                entry[key] = upstream_meta[key]
        return entry
    if registry and model_id in registry.models:
        entry = registry._model_entry(model_id)
        entry["provider"] = "ollama-cloud"
        if fallback_provider and fallback_provider.enabled and fallback_provider.resolve_model(model_id):
            entry["provider"] = f"ollama-cloud,{fallback_provider.provider}"
            entry["fallback_model"] = fallback_provider.resolve_model(model_id)
        return entry
    if fallback_provider and fallback_provider.enabled and fallback_provider.resolve_model(model_id):
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": fallback_provider.provider,
            "provider": fallback_provider.provider,
            "fallback_model": fallback_provider.resolve_model(model_id),
        }
    return JSONResponse(status_code=404, content={"error": f"model '{model_id}' not found"})


@app.post("/v1/chat/completions")
@app.post("/v1/chat/completions/")
async def chat_completions(request: Request):
    return await _proxy_request(request, "/chat/completions")


@app.post("/v1/completions")
@app.post("/v1/completions/")
async def completions(request: Request):
    return await _proxy_request(request, "/completions")


@app.post("/v1/embeddings")
@app.post("/v1/embeddings/")
async def embeddings(request: Request):
    return await _proxy_request(request, "/embeddings")


# ---------------------------------------------------------------------------
# Routes — Native Ollama API (/api/*)
# ---------------------------------------------------------------------------

@app.get("/api/tags")
async def api_tags(request: Request):
    """Return model list in Ollama native /api/tags format."""
    _resolve_client(request)
    if not registry:
        return {"models": []}
    models = []
    for model_id in sorted(registry.models.keys()):
        meta = registry.model_metadata.get(model_id, {})
        details = dict(meta.get("details") or {})
        context_length = meta.get("context_length") or MODEL_CONTEXT_LENGTHS.get(model_id)
        if context_length:
            details["context_length"] = context_length
        entry = {
            "name": model_id,
            "model": model_id,
            "modified_at": meta.get("modified_at") or (
                datetime.fromtimestamp(registry.last_refresh, tz=UTC).isoformat()
                if registry.last_refresh else ""
            ),
            "size": meta.get("size") or 0,
            "digest": meta.get("digest") or "",
            "details": details,
            "size_vram": meta.get("size_vram", 0),
        }
        models.append(entry)
    # Inject model aliases
    if model_alias_manager:
        existing_names = {m.get("name") for m in models}
        for ae in model_alias_manager.alias_entries():
            alias_name = ae["alias"]
            upstream = ae["upstream_model"]
            ctx = ae["context_length"]
            # Skip self-aliases when the upstream model is already listed —
            # patch the existing entry instead of duplicating.
            if alias_name == upstream and alias_name in existing_names:
                for m in models:
                    if m.get("name") == alias_name:
                        if ctx and isinstance(m.get("details"), dict):
                            m["details"]["context_length"] = ctx
                        break
                continue
            upstream_meta = registry.model_metadata.get(upstream, {})
            details = dict(upstream_meta.get("details") or {})
            if ctx:
                details["context_length"] = ctx
            models.append({
                "name": alias_name,
                "model": alias_name,
                "modified_at": upstream_meta.get("modified_at") or "",
                "size": upstream_meta.get("size") or 0,
                "digest": upstream_meta.get("digest") or "",
                "details": details,
                "size_vram": upstream_meta.get("size_vram", 0),
            })
    return {"models": models}


@app.post("/api/chat")
async def api_chat(request: Request):
    """Native Ollama /api/chat — NDJSON streaming with key rotation."""
    return await _proxy_ndjson_request(request, "/chat")


@app.post("/api/generate")
async def api_generate(request: Request):
    """Native Ollama /api/generate — NDJSON streaming with key rotation."""
    return await _proxy_ndjson_request(request, "/generate")


@app.post("/api/show")
async def api_show(request: Request):
    """Native Ollama /api/show — proxy to upstream with key rotation."""
    _resolve_client(request)
    body = await request.body()
    req_json = json.loads(body) if body else {}
    model = req_json.get("name", req_json.get("model", "unknown"))
    session_id = _extract_session_id(request, req_json)

    # --- Model alias resolution ---
    resolved_model = model
    if model_alias_manager and model_alias_manager.is_alias(model):
        resolved_model, _ctx_override = model_alias_manager.resolve(model)
        # /api/show uses "name" field, not "model"
        if "name" in req_json:
            req_json["name"] = resolved_model
        else:
            req_json["model"] = resolved_model
        body = json.dumps(req_json).encode()

    prefer_key = registry.get_preferred_key(resolved_model) if registry else None

    last_error = None
    for attempt in range(max_retries + 1):
        key = None
        deadline = time.time() + queue_timeout
        while time.time() < deadline:
            key = await manager.acquire(prefer_key=prefer_key)
            if key:
                break
            await asyncio.sleep(0.5)

        if not key:
            return JSONResponse(
                status_code=503,
                content={"error": "all keys at capacity"},
            )

        try:
            headers = {
                "Authorization": f"Bearer {key.token}",
                "Content-Type": "application/json",
            }
            api_upstream = _native_api_upstream()

            resp = await upstream_http_client.post(
                f"{api_upstream}/show",
                content=body,
                headers=headers,
            )

            if resp.status_code == 429:
                await manager.mark_429(key)
                await manager.release(key)
                log.warning(f"429 from {key.label} for /api/show model={model}")
                prefer_key = None
                continue

            if resp.status_code == 402:
                await manager.mark_402(key)
                await manager.release(key)
                log.warning(f"402 from {key.label} for /api/show model={model}")
                prefer_key = None
                continue

            await manager.release(key)
            resp_headers = dict(resp.headers)
            if sticky and session_id and resp.status_code == 200:
                resp_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
                resp_headers["X-LlamaHerd-Session"] = session_id
                resp_headers["X-LlamaHerd-Key"] = key.label

            # --- Alias context_length override ---
            # When the requested model is an alias (e.g. glm-5.2-256k), the
            # upstream /api/show response contains the PARENT model's
            # context_length (e.g. 1M for glm-5.2).  Clients like Hermes
            # use /api/show to discover the real context window, so we must
            # patch the response to reflect the alias's configured
            # context_length override (e.g. 262144).
            if resp.status_code == 200 and model_alias_manager and model_alias_manager.is_alias(model):
                _, ctx_override = model_alias_manager.resolve(model)
                if ctx_override:
                    try:
                        show_json = json.loads(resp.content)
                        info = show_json.get("model_info") or {}
                        patched = False
                        for k in list(info.keys()):
                            if k.endswith(".context_length"):
                                info[k] = ctx_override
                                patched = True
                        if not patched:
                            # No context_length key found — inject one using
                            # the family/architecture key if present.
                            arch = info.get("general.architecture")
                            if arch:
                                info[f"{arch}.context_length"] = ctx_override
                                patched = True
                        if patched:
                            show_json["model_info"] = info
                            # Also patch details.context_length if present
                            details = show_json.get("details") or {}
                            details["context_length"] = ctx_override
                            show_json["details"] = details
                            resp_headers["content-type"] = "application/json"
                            resp_headers.pop("content-length", None)
                            resp_headers.pop("Content-Length", None)
                            log.info(f"Alias /api/show: patched context_length to {ctx_override} for {model}")
                            return Response(
                                content=json.dumps(show_json).encode(),
                                status_code=resp.status_code,
                                headers=resp_headers,
                            )
                    except Exception as e:
                        log.warning(f"Alias /api/show patch failed for {model}: {e}")

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )

        except Exception as e:
            await manager.release(key)
            last_error = str(e)
            log.error(f"Native /api/show error for {model}: {e}")
            continue

    return JSONResponse(
        status_code=502,
        content={"error": f"all retries exhausted: {last_error}"},
    )


@app.get("/api/ps")
async def api_ps(request: Request):
    """Native Ollama /api/ps — return empty models list (we don't track running models)."""
    _resolve_client(request)
    return {"models": []}


# ---------------------------------------------------------------------------
# Native Bridge — Re-route /v1 requests via Ollama native /api to fix
# GLM truncation misreports (done_reason: "length" vs finish_reason: "stop")
# ---------------------------------------------------------------------------

# Models whose /v1/chat/completions endpoint misreports truncation as "stop".
# The Ollama native /api/chat endpoint correctly reports done_reason: "length".
NATIVE_BRIDGE_MODELS: list[str] = []  # populated from config in lifespan()


def _should_bridge_to_native(model: str) -> bool:
    """Check if a /v1 request should be internally routed through /api/chat."""
    if not NATIVE_BRIDGE_MODELS:
        return False
    model_lower = model.lower()
    # Strip :cloud suffix for comparison
    model_base = model_lower.replace(":cloud", "")
    for prefix in NATIVE_BRIDGE_MODELS:
        prefix = prefix.lower().strip()
        if prefix.endswith("*"):
            if model_base.startswith(prefix[:-1]):
                return True
        elif model_base == prefix:
            return True
    return False


async def _proxy_bridge_stream(client_id: str, key: 'KeyState', body: bytes,
                                model: str, start: float,
                                request_id: str | None = None,
                                session_id: str | None = None) -> StreamingResponse:
    """Bridge stream: receive NDJSON from /api/chat, emit SSE for /v1/chat/completions client.

    This is the core of the native bridge. It re-routes the upstream request
    from the Ollama /v1 endpoint to the native /api/chat endpoint, which
    correctly reports done_reason: "length" for truncated responses. The
    NDJSON response is converted chunk-by-chunk to SSE format so the client
    (Hermes) sees a standard OpenAI-compatible stream.
    """
    api_upstream = _native_api_upstream()
    chunk_id = f"chatcmpl-bridge-{uuid.uuid4().hex[:8]}"
    tokens_out = 0
    tokens_in = 0
    usage_captured = False
    bridge_reason = "stop"
    final_status = 200

    async def finalize():
        elapsed_ms = int((time.time() - start) * 1000)
        await manager.release(key, tokens_out)
        _record_and_broadcast(client_id, key.token, model, tokens_in, tokens_out, elapsed_ms,
                              final_status, request_id=request_id, provider="ollama-cloud", session_id=session_id)
        usage_src = "usage" if usage_captured else "estimate"
        log.info(f"{client_id} -> {model} via {key.label}: bridge {tokens_in}+{tokens_out}tok {elapsed_ms}ms done={bridge_reason} ({usage_src})")

    finalizer = _OnceAsyncFinalizer(finalize)

    async def generate():
        nonlocal tokens_out, tokens_in, usage_captured, bridge_reason, final_status
        try:
            async with _cancellation_safe_stream(
                upstream_http_client, "POST", f"{api_upstream}/chat",
                content=body, headers={
                    "Authorization": f"Bearer {key.token}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        final_status = 429
                        yield 'data: {"error": "429 from upstream"}\n\n'
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
                        final_status = 402
                        yield 'data: {"error": "402 from upstream"}\n\n'
                        return
                    if resp.status_code >= 400:
                        error_body = await resp.aread()
                        err = error_body.decode(errors="replace").strip()
                        final_status = resp.status_code
                        # Try to format as OpenAI error
                        yield f'data: {json.dumps({"error": {"message": err, "type": "upstream_error", "code": resp.status_code}})}\n\n'
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            ollama_chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Convert and emit SSE
                        sse_line = _ollama_chunk_to_sse(ollama_chunk, chunk_id, model)
                        if sse_line:
                            yield sse_line + "\n\n"

                        # Capture usage from final chunk
                        if ollama_chunk.get("done", False):
                            done_reason = ollama_chunk.get("done_reason", "stop")
                            bridge_reason = done_reason
                            pev = ollama_chunk.get("prompt_eval_count")
                            ev = ollama_chunk.get("eval_count")
                            if pev is not None:
                                tokens_in = int(pev)
                            if ev is not None:
                                tokens_out = int(ev)
                            if tokens_in > 0 or tokens_out > 0:
                                usage_captured = True
                                _update_in_flight_tokens(request_id, tokens_in, tokens_out)
                            # Emit [DONE] after final chunk
                            yield "data: [DONE]\n\n"
                        else:
                            # Live progress: estimate completion tokens from delta content.
                            msg = ollama_chunk.get("message") or {}
                            content = msg.get("content") or ""
                            if content:
                                tokens_out += max(1, len(content) // 4)
                                _update_in_flight_tokens(request_id, None, tokens_out)

        except Exception as e:
            log.error(f"Bridge stream error for {model} (client={client_id}): {e}")
        finally:
            await _await_cleanup(finalizer(), label="bridge stream accounting")

    bridge_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    if sticky and session_id:
        bridge_headers["Set-Cookie"] = _session_cookie_for_response(session_id, sticky.ttl)
        bridge_headers["X-LlamaHerd-Session"] = session_id
        bridge_headers["X-LlamaHerd-Key"] = key.label
    return _FinalizingStreamingResponse(
        generate(), media_type="text/event-stream", headers=bridge_headers, finalizer=finalizer,
    )


# ---------------------------------------------------------------------------
# Admin — Status & Usage
# ---------------------------------------------------------------------------

@app.get("/admin/status", dependencies=[Depends(_verify_admin)])
async def admin_status():
    return {
        "keys": manager.status() if manager else [],
        "models": len(registry.models) if registry else 0,
        "last_refresh": registry.last_refresh if registry else 0,
        "upstream": upstream_url,
        "clients": client_registry.clients if client_registry else [],
        "sticky_sessions": sticky.get_status() if sticky else {},
        "sticky_ttl_seconds": sticky.ttl if sticky else None,
    }


@app.get("/admin/usage", dependencies=[Depends(_verify_admin)])
async def admin_usage(hours: int = 24, client: str | None = None, model: str | None = None):
    if usage_db:
        return usage_db.summary(hours, client=client, model=model)
    return []


@app.get("/admin/usage/daily", dependencies=[Depends(_verify_admin)])
async def admin_usage_daily(days: int = 30, start_date: str | None = None, end_date: str | None = None):
    if usage_db:
        return usage_db.daily_totals(days, start_date=start_date, end_date=end_date)
    return []


@app.get("/admin/usage/by-client", dependencies=[Depends(_verify_admin)])
async def admin_usage_by_client(days: int = 30, start_date: str | None = None, end_date: str | None = None):
    if usage_db:
        return usage_db.by_client(days, start_date=start_date, end_date=end_date)
    return []


@app.get("/admin/usage/by-model", dependencies=[Depends(_verify_admin)])
async def admin_usage_by_model(days: int = 30, start_date: str | None = None, end_date: str | None = None):
    if usage_db:
        return usage_db.by_model(days, start_date=start_date, end_date=end_date)
    return []


# --- OpenRouter cost tracking ---

_OPENROUTER_PRICING: dict | None = None
_LAST_DISCOVERY_REFRESH: float = 0.0  # epoch seconds of last immediate discovery refresh
_PRICING_LAST_SYNC: float | None = None  # epoch seconds of last successful OpenRouter API sync

# Mapping from LlamaHerd model names to OpenRouter model IDs.
# Used to enrich local models with OpenRouter pricing when they aren't in the YAML already.
# This is a best-effort mapping; models not listed here need manual openrouter_id in the YAML.
_PRICING_MODEL_ALIASES: dict[str, str] = {
    # Populated dynamically from the YAML's openrouter_id fields.
    # Additional heuristics are applied in _sync_pricing_from_openrouter().
}


def _load_openrouter_pricing() -> dict:
    """Load OpenRouter pricing from YAML. Caches after first load."""
    global _OPENROUTER_PRICING
    if _OPENROUTER_PRICING is not None:
        return _OPENROUTER_PRICING
    pricing_path = CONFIG_PATH.parent / "openrouter_pricing.yaml"
    if pricing_path.exists():
        with open(pricing_path) as f:
            data = yaml.safe_load(f)
        _OPENROUTER_PRICING = data.get("models", {}) if data else {}
        log.info(f"Loaded OpenRouter pricing for {len(_OPENROUTER_PRICING)} models")
        # Build alias map from existing openrouter_id entries
        _rebuild_pricing_aliases()
    else:
        _OPENROUTER_PRICING = {}
        log.warning(f"No openrouter_pricing.yaml found at {pricing_path}")
    return _OPENROUTER_PRICING


def _rebuild_pricing_aliases():
    """Rebuild _PRICING_MODEL_ALIASES from current pricing data."""
    global _PRICING_MODEL_ALIASES
    aliases = {}
    for name, entry in _OPENROUTER_PRICING.items():
        or_id = entry.get("openrouter_id", "")
        if or_id:
            aliases[name] = or_id
    _PRICING_MODEL_ALIASES = aliases


def _save_pricing_yaml(pricing: dict):
    """Write pricing data back to YAML file, preserving header comments."""
    pricing_path = CONFIG_PATH.parent / "openrouter_pricing.yaml"
    header = [
        "# OpenRouter equivalent pricing for LlamaHerd models",
        "# All prices in USD per 1M tokens",
        "# Source: OpenRouter API (https://openrouter.ai/api/v1/models) + supplementary sources",
        f"# Auto-synced: {datetime.now(UTC).isoformat()}",
        "#",
        '# Strips :cloud suffix automatically — "glm-5.1:cloud" uses "glm-5.1" prices.',
        "",
    ]
    # Build ordered dict for YAML output
    output = {"models": pricing}
    yaml_content = yaml.dump(output, default_flow_style=False, allow_unicode=True, sort_keys=False)
    with open(pricing_path, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write(yaml_content)
    log.info(f"Saved pricing YAML with {len(pricing)} models to {pricing_path}")


async def _sync_pricing_from_openrouter() -> int:
    """Fetch current pricing from OpenRouter API and merge into local pricing data.

    - New models discovered on OpenRouter are added.
    - Existing model prices are updated to match OpenRouter's live rates.
    - Models not on OpenRouter (openrouter_id: "") keep their manual prices.
    - Models with an openrouter_id get their prices refreshed.
    - Returns the number of models updated/added.

    This runs on startup (immediate) and periodically every 24-48h.
    """
    global _OPENROUTER_PRICING, _PRICING_LAST_SYNC

    url = "https://openrouter.ai/api/v1/models"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning(f"OpenRouter pricing sync failed: {e}")
        return 0

    models_list = data.get("data", [])
    if not models_list:
        log.warning("OpenRouter pricing sync: empty model list received")
        return 0

    # Build a lookup: openrouter_id -> {prompt, completion} per-token pricing
    or_pricing: dict[str, dict] = {}
    for m in models_list:
        mid = m.get("id", "")
        p = m.get("pricing", {})
        if not p:
            continue
        prompt = p.get("prompt", "0")
        completion = p.get("completion", "0")
        # Skip free models (both prices are "0")
        try:
            p_val = float(prompt)
            c_val = float(completion)
        except (ValueError, TypeError):
            continue
        if p_val == 0 and c_val == 0:
            continue
        # Convert per-token to per-1M tokens
        or_pricing[mid] = {
            "input_per_1m": round(p_val * 1_000_000, 6),
            "output_per_1m": round(c_val * 1_000_000, 6),
        }

    # Make sure we have the current pricing loaded
    if _OPENROUTER_PRICING is None:
        _load_openrouter_pricing()
    assert _OPENROUTER_PRICING is not None  # guaranteed by _load_openrouter_pricing

    updated = 0
    added = 0

    # Phase 1: Update existing models that have an openrouter_id
    for name, entry in list(_OPENROUTER_PRICING.items()):
        or_id = entry.get("openrouter_id", "")
        if not or_id:
            continue  # Manual-only entry, skip
        if or_id in or_pricing:
            new_prices = or_pricing[or_id]
            old_in = entry.get("input_per_1m", 0)
            old_out = entry.get("output_per_1m", 0)
            new_in = new_prices["input_per_1m"]
            new_out = new_prices["output_per_1m"]
            if old_in != new_in or old_out != new_out:
                log.info(f"Pricing updated for {name} ({or_id}): "
                         f"${old_in}/${old_out} -> ${new_in}/${new_out} per 1M")
                entry["input_per_1m"] = new_in
                entry["output_per_1m"] = new_out
                updated += 1

    # Phase 2: Discover Ollama Cloud models that have usage but no pricing entry
    # Map Ollama model names to likely OpenRouter IDs using known patterns
    # Build reverse lookup: openrouter_id -> local name (for discovery)
    or_id_to_local = {}
    for name, entry in _OPENROUTER_PRICING.items():
        or_id = entry.get("openrouter_id", "")
        if or_id:
            or_id_to_local[or_id] = name

    # Also check registry models against OpenRouter
    if registry:
        for model_id in registry.models:
            # Strip :cloud suffix for lookup
            lookup_key = model_id.replace(":cloud", "").replace(":cloud-", "-")
            if lookup_key in _OPENROUTER_PRICING:
                continue  # Already have pricing

            # Try common naming patterns to find OpenRouter equivalent
            candidates = _guess_openrouter_id(lookup_key)
            for or_id in candidates:
                if or_id in or_pricing and or_id not in or_id_to_local:
                    prices = or_pricing[or_id]
                    _OPENROUTER_PRICING[lookup_key] = {
                        "openrouter_id": or_id,
                        "input_per_1m": prices["input_per_1m"],
                        "output_per_1m": prices["output_per_1m"],
                    }
                    or_id_to_local[or_id] = lookup_key
                    added += 1
                    log.info(f"New model discovered: {lookup_key} -> {or_id} "
                             f"(${prices['input_per_1m']}/${prices['output_per_1m']} per 1M)")
                    break

    # Persist to YAML
    if updated > 0 or added > 0:
        try:
            _save_pricing_yaml(_OPENROUTER_PRICING)
            _rebuild_pricing_aliases()
        except Exception as e:
            log.error(f"Failed to save pricing YAML: {e}")

    _PRICING_LAST_SYNC = time.time()
    log.info(f"OpenRouter pricing sync complete: {len(or_pricing)} models fetched, "
             f"{updated} updated, {added} added, {len(_OPENROUTER_PRICING)} total local entries")
    return updated + added


def _guess_openrouter_id(model_name: str) -> list[str]:
    """Given a LlamaHerd/Ollama model name, guess likely OpenRouter model IDs.

    Ollama and OpenRouter use different naming conventions. This function
    produces candidate OpenRouter IDs in priority order for matching.
    """
    candidates = []

    # Known provider prefixes on OpenRouter
    providers = {
        "glm": "z-ai",
        "gemma": "google",
        "deepseek": "deepseek",
        "kimi": "moonshotai",
        "qwen": "qwen",
        "mistral": "mistralai",
        "minimax": "minimax",
        "nemotron": "nvidia",
        "llama": "meta-llama",
        "devstral": "mistralai",
        "ministral": "mistralai",
        "rnj": "essentialai",
        "gpt-oss": "openai",
        "cogito": "deepcogito",
    }

    # Extract base family name
    name = model_name

    # Strip quantization/size suffix common in Ollama (e.g. :latest, :7b, :q4_0)
    base = name.split(":")[0] if ":" in name else name

    # Try direct match first — some models share names
    # e.g. "deepseek-v4-flash" -> "deepseek/deepseek-v4-flash"

    # Provider prefix mapping
    for prefix, or_provider in providers.items():
        if base.startswith(prefix):
            # e.g. "glm-5.1" -> ["z-ai/glm-5.1", "z-ai/glm5.1"]
            # e.g. "gemma3:12b" -> "gemma3" -> ["google/gemma-3-12b-it"]
            stem = base[len(prefix):]
            if stem.startswith(("-", "_")):
                stem = stem[1:]
            # Try provider/stem as-is
            candidates.append(f"{or_provider}/{base}")
            # Try with hyphens-to-dashes normalization
            candidates.append(f"{or_provider}/{prefix}-{stem}")
            # Gemma special: google/gemma-X-Yb-it
            if prefix == "gemma" and ":" in name:
                size_part = name.split(":")[1]
                # gemma3:12b -> google/gemma-3-12b-it
                major = base.replace("gemma", "")
                candidates.append(f"google/gemma-{major}-{size_part}-it")
            break
    else:
        # No known provider — try common patterns
        candidates.append(f"{base}/{base}")
        # Try the model name itself as a slug
        candidates.append(base)

    return candidates


async def _pricing_sync_loop(interval_hours: float = 24.0):
    """Periodically sync OpenRouter pricing every N hours (default 24h)."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        try:
            await _sync_pricing_from_openrouter()
        except Exception as e:
            log.error(f"Periodic pricing sync failed: {e}")


@app.get("/admin/usage/openrouter-costs", dependencies=[Depends(_verify_admin)])
async def admin_openrouter_costs(days: int = 30, start_date: str | None = None,
                                  end_date: str | None = None, client: str | None = None):
    """Calculate what usage WOULD have cost on OpenRouter (pay-per-token pricing reference)."""
    if not usage_db:
        return {"models": [], "total_cost_usd": 0, "unpriced_models": []}
    pricing = _load_openrouter_pricing()
    return usage_db.openrouter_costs(pricing, days, start_date=start_date,
                                     end_date=end_date, client_id=client,
                                     alias_manager=model_alias_manager)


@app.post("/admin/sync-pricing", dependencies=[Depends(_verify_admin)])
async def admin_sync_pricing():
    """Trigger an immediate sync of OpenRouter pricing data.

    Fetches current prices from https://openrouter.ai/api/v1/models,
    updates existing model prices, discovers new models, and persists
    changes to openrouter_pricing.yaml.
    """
    result = await _sync_pricing_from_openrouter()
    return {
        "sync_result": result,
        "last_sync": _PRICING_LAST_SYNC,
        "total_models": len(_OPENROUTER_PRICING) if _OPENROUTER_PRICING else 0,
    }


@app.get("/admin/pricing-status", dependencies=[Depends(_verify_admin)])
async def admin_pricing_status():
    """Show pricing data status: number of models, last sync time, unpriced models."""
    pricing = _load_openrouter_pricing() if _OPENROUTER_PRICING is None else _OPENROUTER_PRICING
    unpriced = [name for name, entry in pricing.items() if not entry.get("openrouter_id")]
    return {
        "total_models": len(pricing),
        "priced_models": len(pricing) - len(unpriced),
        "unpriced_models": unpriced,
        "last_sync": _PRICING_LAST_SYNC,
        "last_sync_iso": datetime.fromtimestamp(_PRICING_LAST_SYNC, tz=UTC).isoformat() if _PRICING_LAST_SYNC else None,
    }


@app.get("/admin/recent-calls", dependencies=[Depends(_verify_admin)])
async def admin_recent_calls(limit: int = 100, client: str | None = None, model: str | None = None,
                              start_date: str | None = None, end_date: str | None = None):
    """Return recent individual calls (not aggregates) for the live feed."""
    if not usage_db:
        return []
    return usage_db.recent_calls(limit, start_date=start_date, end_date=end_date,
                                 client_id=client, model=model)


@app.get("/admin/totals", dependencies=[Depends(_verify_admin)])
async def admin_totals(start_date: str | None = None, end_date: str | None = None):
    """Totals across all clients/models. Optionally filter by date range."""
    if usage_db:
        return usage_db.totals(start_date=start_date, end_date=end_date)
    return {
        "total_calls": 0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_tokens": 0,
        "avg_latency_ms": None,
        "error_rate_pct": None,
    }


@app.get("/admin/models", dependencies=[Depends(_verify_admin)])
async def admin_models():
    """List all discovered models with context lengths and availability."""
    fp = fallback_provider
    fp_enabled = bool(fp and fp.enabled)
    models_data: list[dict] = []
    seen_ids: set[str] = set()
    if registry:
        for model_id, keys in registry.models.items():
            meta = registry.model_metadata.get(model_id, {})
            param_count = meta.get("parameter_count")
            providers = ["ollama-cloud"]
            fb_mapped = fp.resolve_model(model_id) if fp_enabled else None
            if fb_mapped:
                providers.append(fp.provider)
            models_data.append({
                "id": model_id,
                "context_length": meta.get("context_length") or MODEL_CONTEXT_LENGTHS.get(model_id),
                "available_on": len(keys),
                "modified_at": meta.get("modified_at"),
                "size": meta.get("size"),
                "digest": meta.get("digest"),
                "capabilities": meta.get("capabilities") or [],
                "family": meta.get("family"),
                "parameter_count": param_count,
                "parameter_count_display": fmt_param_count(param_count),
                "quantization_level": meta.get("quantization_level"),
                "providers": providers,
                "fallback_model": fb_mapped,
                "priority": fp.priority_for(model_id) if fp_enabled else None,
            })
            seen_ids.add(model_id)
    if fp_enabled:
        for alias in fp.model_aliases():
            if alias["id"] in seen_ids:
                continue
            models_data.append({
                "id": alias["id"],
                "context_length": None,
                "available_on": 0,
                "modified_at": None,
                "size": None,
                "digest": None,
                "capabilities": [],
                "family": None,
                "parameter_count": None,
                "parameter_count_display": "",
                "quantization_level": None,
                "providers": [fp.provider],
                "fallback_model": alias["nvidia_model"],
                "priority": alias["priority"],
            })
    models_data.sort(key=lambda m: m["id"])
    return {
        "models": models_data,
        "count": len(models_data),
        "last_refresh": registry.last_refresh if registry else 0,
        "fallback": {
            "enabled": fp_enabled,
            "provider": fp.provider if fp_enabled else None,
            "priority": fp.priority if fp_enabled else None,
            "default_model": fp.default_model if fp_enabled else None,
            "discovered_count": len(fp.discovered_models) if fp_enabled else 0,
        },
    }


@app.get("/admin/in-flight", dependencies=[Depends(_verify_admin)])
async def admin_in_flight():
    """Return currently in-flight requests with elapsed time."""
    now = time.time()
    rows = []
    for entry in _in_flight.values():
        rows.append({
            **entry,
            "elapsed_ms": int((now - entry["started_at"]) * 1000),
        })
    rows.sort(key=lambda r: r["started_at"])
    return {"in_flight": rows, "count": len(rows)}


@app.get("/admin/fallback", dependencies=[Depends(_verify_admin)])
async def admin_fallback_status():
    """Inspect the fallback provider's runtime state."""
    fp = fallback_provider
    if not fp or not fp.enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "provider": fp.provider,
        "base_url": fp.base_url,
        "default_model": fp.default_model,
        "priority": fp.priority,
        "valid_priorities": list(VALID_FALLBACK_PRIORITIES),
        "model_map": fp.model_aliases(),
        "discovered_count": len(fp.discovered_models),
    }


@app.get("/admin/fallback-catalog", dependencies=[Depends(_verify_admin)])
async def admin_fallback_catalog():
    """Return the full discovered fallback model catalog with metadata."""
    fp = fallback_provider
    if not fp or not fp.enabled:
        return {"enabled": False, "catalog": [], "count": 0}
    catalog = fp.get_catalog()
    # Group counts per org for the dashboard badges.
    by_org: dict[str, int] = {}
    for entry in catalog:
        org = entry.get("org") or ""
        by_org[org] = by_org.get(org, 0) + 1
    return {
        "enabled": True,
        "provider": fp.provider,
        "catalog": catalog,
        "count": len(catalog),
        "by_org": by_org,
    }


@app.post("/admin/fallback-map", dependencies=[Depends(_verify_admin)])
async def admin_add_fallback_map(request: Request):
    """Add or update an in-memory fallback model_map entry.

    Body: {"ollama_name": "...", "nvidia_name": "...", "priority": "after|before|only"}
    The change is in-memory only; a warning is logged so it can be persisted to config.
    """
    fp = fallback_provider
    if not fp or not fp.enabled:
        raise HTTPException(status_code=400, detail="fallback provider not configured")
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}
    ollama_name = body.get("ollama_name") or request.query_params.get("ollama_name")
    nvidia_name = body.get("nvidia_name") or request.query_params.get("nvidia_name")
    priority = body.get("priority") or request.query_params.get("priority")
    if not ollama_name or not nvidia_name:
        raise HTTPException(status_code=400, detail="ollama_name and nvidia_name are required")
    try:
        entry = fp.add_mapping(ollama_name, nvidia_name, priority=priority)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log.warning(
        f"Runtime fallback map update for {ollama_name} -> {nvidia_name}; "
        f"add to config.yaml to persist."
    )
    payload = {
        "ollama_name": ollama_name,
        "nvidia_model": entry["nvidia_model"],
        "priority": entry.get("priority") or fp.priority,
    }
    await broadcaster.broadcast("fallback_map_update", {"action": "add", **payload})
    return {"added": payload, "model_map": fp.model_aliases()}


@app.delete("/admin/fallback-map", dependencies=[Depends(_verify_admin)])
async def admin_remove_fallback_map(request: Request):
    """Remove an in-memory fallback model_map entry.

    Accepts ``ollama_name`` via query param or JSON body.
    """
    fp = fallback_provider
    if not fp or not fp.enabled:
        raise HTTPException(status_code=400, detail="fallback provider not configured")
    ollama_name = request.query_params.get("ollama_name")
    if not ollama_name:
        try:
            body = await request.json()
        except Exception:
            body = None
        ollama_name = (body or {}).get("ollama_name") if isinstance(body, dict) else None
    if not ollama_name:
        raise HTTPException(status_code=400, detail="ollama_name is required")
    removed = fp.remove_mapping(ollama_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no mapping for {ollama_name}")
    log.warning(
        f"Runtime fallback map removal for {ollama_name}; "
        f"remove from config.yaml to persist."
    )
    await broadcaster.broadcast("fallback_map_update", {"action": "remove", "ollama_name": ollama_name})
    return {"removed": ollama_name, "model_map": fp.model_aliases()}


@app.post("/admin/fallback-priority", dependencies=[Depends(_verify_admin)])
async def admin_set_fallback_priority(request: Request):
    """Change the fallback provider's global priority at runtime (in-memory only).

    Body: {"priority": "after" | "before" | "only"}
    """
    fp = fallback_provider
    if not fp or not fp.enabled:
        raise HTTPException(status_code=400, detail="fallback provider not configured")
    body = await request.json()
    requested = (body or {}).get("priority")
    if requested not in VALID_FALLBACK_PRIORITIES:
        raise HTTPException(
            status_code=400,
            detail=f"priority must be one of {list(VALID_FALLBACK_PRIORITIES)}",
        )
    previous = fp.priority
    fp.set_priority(requested)
    log.info(f"Fallback priority changed at runtime: {previous} -> {fp.priority}")
    await broadcaster.broadcast("fallback_priority", {"priority": fp.priority, "previous": previous})
    return {"priority": fp.priority, "previous": previous}


@app.post("/admin/refresh", dependencies=[Depends(_verify_admin)])
async def admin_refresh():
    if registry:
        old_models = set(registry.models.keys())
        await registry.refresh()
        new_models = set(registry.models.keys()) - old_models
        # Broadcast model change via SSE
        await broadcaster.broadcast("models", {
            "count": len(registry.models),
            "last_refresh": registry.last_refresh,
            "new_models": sorted(new_models) if new_models else [],
        })
        return {"models": len(registry.models), "new": sorted(new_models)}
    return {"error": "registry not initialized"}


@app.post("/admin/reset-exhausted", dependencies=[Depends(_verify_admin)])
async def admin_reset_exhausted():
    if manager:
        for k in manager.keys:
            k.exhausted = False
            k.exhausted_until = 0
        return {"reset": len(manager.keys)}
    return {}


@app.post("/admin/poll-subscriptions", dependencies=[Depends(_verify_admin)])
async def admin_poll_subscriptions():
    """Manually trigger subscription status poll for all keys."""
    if manager:
        await manager.poll_subscriptions()
        return {"keys": manager.status()}
    return {"error": "manager not initialized"}


@app.post("/admin/scrape-usage", dependencies=[Depends(_verify_admin)])
async def admin_scrape_usage():
    """Manually trigger usage scraping from ollama.com/settings."""
    if usage_scraper and manager:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, usage_scraper.scrape_all, manager.keys)
        return {"keys": manager.status(), "scrape_results": result}
    return {"error": "usage_scraper or manager not initialized"}


@app.post("/admin/telegram-test", dependencies=[Depends(_verify_admin)])
async def admin_telegram_test():
    """Send a test Telegram notification with current key usage data."""
    if not telegram_notifier:
        return {"error": "Telegram notifier not initialized"}
    if not telegram_notifier.enabled:
        return {"error": "Telegram not configured. Set LLAMAHERD_TELEGRAM_TOKEN and LLAMAHERD_TELEGRAM_CHAT_ID env vars."}
    if not manager:
        return {"error": "Manager not initialized"}
    ok = await telegram_notifier.send_usage_notification(manager)
    return {"sent": ok, "chat_id": telegram_notifier.chat_id,
            "topic_id": telegram_notifier.topic_id or None,
            "interval": telegram_notifier.interval}


# ---------------------------------------------------------------------------
# Admin — Subscription (Upstream Key) Management
# ---------------------------------------------------------------------------

def _key_id(token: str) -> str:
    """Return a stable, non-secret identifier for an upstream token."""
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _find_key(key_id: str) -> KeyState | None:
    if not manager:
        return None
    return next((key for key in manager.keys if _key_id(key.token) == key_id), None)

@app.get("/admin/keys", dependencies=[Depends(_verify_admin)])
async def admin_list_keys():
    """List all upstream Ollama Cloud subscription keys (tokens masked)."""
    if not manager:
        return []
    result = []
    for i, k in enumerate(manager.keys):
        # Check if cookies exist for this key in usage_scraper
        has_cookies = False
        if usage_scraper and hasattr(usage_scraper, 'cookie_map'):
            has_cookies = k.label in usage_scraper.cookie_map and bool(usage_scraper.cookie_map[k.label].get('secure_session'))
        result.append({
            "label": k.label,
            "token_prefix": k.token[:8] + "...",
            "max_concurrent": k.max_concurrent,
            "cycle_day": k.cycle_day,
            "plan": k.plan,
            "suspended": k.suspended,
            "account_email": k.account_email if k.account_email else None,
            "has_cookies": has_cookies,
            "key_id": _key_id(k.token),
            "index": i,
        })
    return result


@app.put("/admin/keys/{key_id}", dependencies=[Depends(_verify_admin)])
async def admin_update_key(key_id: str, label: str | None = None, max_concurrent: int | None = None,
                           cycle_day: int | None = None):
    """Update a key's mutable fields (label, max_concurrent, cycle_day). Persists to DB."""
    k = _find_key(key_id)
    if not k:
        raise HTTPException(status_code=404, detail="key not found")
    label_changed = False
    old_label = k.label
    if label is not None:
        k.label = label
        label_changed = (label != old_label)
    if max_concurrent is not None:
        k.max_concurrent = max_concurrent
    if cycle_day is not None:
        k.cycle_day = cycle_day
    # If the label was renamed and the usage scraper has cookies stored under
    # the old label, migrate them so scraping keeps working. Reviewer #5 on PR #2.
    if label_changed and usage_scraper and hasattr(usage_scraper, "cookie_map"):
        if old_label in usage_scraper.cookie_map and old_label != label:
            usage_scraper.cookie_map[label] = usage_scraper.cookie_map.pop(old_label)
    # Persist to DB
    if key_registry:
        key_registry.update(k.token, label=label, max_concurrent=max_concurrent, cycle_day=cycle_day)
    return {"updated": key_id, "key_id": key_id, "label": k.label, "max_concurrent": k.max_concurrent, "cycle_day": k.cycle_day}


@app.put("/admin/keys/{key_id}/cookies", dependencies=[Depends(_verify_admin)])
async def admin_update_key_cookies(key_id: str, request: Request):
    """Update cookies for a specific key. Persists to DB and updates scraper."""
    k = _find_key(key_id)
    if not k:
        raise HTTPException(status_code=404, detail="key not found")
    body = await request.json()
    cookies = {}
    for cookie_field in ["secure_session", "aid", "cf_clearance", "stripe_mid"]:
        if cookie_field in body:
            cookies[cookie_field] = body[cookie_field]
    if usage_scraper and hasattr(usage_scraper, 'cookie_map') and cookies:
        usage_scraper.cookie_map[k.label] = cookies
    # Persist to DB
    if key_registry and cookies:
        key_registry.update_cookies(k.token, cookies)
    return {"updated": key_id, "key_id": key_id, "label": k.label, "cookies_set": list(cookies.keys())}


@app.post("/admin/keys", dependencies=[Depends(_verify_admin)])
async def admin_add_key(request: Request):
    """Add a new upstream key. Persists to DB."""
    if not manager:
        raise HTTPException(status_code=500, detail="manager not initialized")
    body = await request.json()
    token = body.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    label = body.get("label", f"Sub {len(manager.keys) + 1}")
    max_concurrent = body.get("max_concurrent", 15)
    cycle_day = body.get("cycle_day", 1)
    new_key = KeyState(token=token, max_concurrent=max_concurrent, cycle_day=cycle_day, label=label)
    manager.keys.append(new_key)
    cookies = body.get("cookies", {})
    if usage_scraper and cookies:
        usage_scraper.cookie_map[label] = cookies
    # Persist to DB
    if key_registry:
        key_registry.add(token, label, max_concurrent, cycle_day, cookies)
    return {"added": label, "key_id": _key_id(token),
            "note": "Key saved to DB and will persist across restarts"}


@app.delete("/admin/keys/{key_id}", dependencies=[Depends(_verify_admin)])
async def admin_delete_key(key_id: str):
    """Remove an upstream key. Persists to DB."""
    removed = _find_key(key_id)
    if not manager or not removed:
        raise HTTPException(status_code=404, detail="key not found")
    manager.keys.remove(removed)
    if usage_refresher:
        usage_refresher.cancel(removed)
    # Also clear cookies from the usage scraper so the deleted key stops
    # being scraped. Without this, an orphan entry sits in cookie_map
    # indefinitely and pollutes /admin/status output.
    if usage_scraper and hasattr(usage_scraper, 'cookie_map') and removed.label in usage_scraper.cookie_map:
        del usage_scraper.cookie_map[removed.label]
    # Persist deletion to DB so the key doesn't reappear on restart
    if key_registry:
        key_registry.remove(removed.token)
    return {"removed": removed.label, "key_id": key_id, "note": "Key removed from DB and will not reappear on restart"}


# ---------------------------------------------------------------------------
# Admin — Client Key Management
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SSE — Live event stream for dashboard
# ---------------------------------------------------------------------------

@app.post("/admin/session", dependencies=[Depends(_verify_admin)])
async def admin_create_session():
    """Create a short-lived token for authenticating an SSE connection."""
    session_token = secrets.token_urlsafe(32)
    _admin_sessions[session_token] = time.time() + ADMIN_SESSION_TTL_SECONDS
    return {"session_token": session_token, "expires_in": ADMIN_SESSION_TTL_SECONDS}


@app.get("/admin/events")
async def admin_events(request: Request, session_token: str = ""):
    """SSE endpoint authenticated by a short-lived, limited-purpose token."""
    _verify_admin_session(session_token)

    async def event_generator():
        q = broadcaster.subscribe()
        try:
            # Send initial status snapshot
            status_data = {
                "keys": manager.status() if manager else [],
                "models": len(registry.models) if registry else 0,
                "last_refresh": registry.last_refresh if registry else 0,
                "upstream": upstream_url,
                "clients": client_registry.clients if client_registry else [],
            }
            yield f"event: status\ndata: {json.dumps(status_data)}\n\n"
            yield f"event: models\ndata: {json.dumps({'count': len(registry.models) if registry else 0, 'last_refresh': registry.last_refresh if registry else 0, 'new_models': []})}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {payload}\n\n"
                except TimeoutError:
                    # Heartbeat keepalive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            broadcaster.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.get("/admin/clients", dependencies=[Depends(_verify_admin)])
async def admin_list_clients():
    """List all registered client keys."""
    if client_registry:
        return client_registry.clients
    return []


@app.post("/admin/clients", dependencies=[Depends(_verify_admin)])
async def admin_create_client(request: Request):
    """Create a new client key. Body: {"id": "my-app", "label": "My App", "notes": "optional", "daily_token_limit": 100000, "daily_request_limit": 500, "rpm_limit": 30}"""
    body = await request.json()
    client_id = body.get("id")
    label = body.get("label", client_id)
    notes = body.get("notes", "")
    custom_token = body.get("token")  # optional: provide your own token
    daily_token_limit = body.get("daily_token_limit")
    daily_request_limit = body.get("daily_request_limit")
    rpm_limit = body.get("rpm_limit")

    if not client_id:
        return JSONResponse(status_code=400, content={"error": "id is required"})
    if not client_id.replace("-", "").replace("_", "").isalnum():
        return JSONResponse(status_code=400,
                            content={"error": "id must be alphanumeric (dashes/underscores ok)"})

    try:
        result = client_registry.create(client_id, label, notes=notes, token=custom_token,
                                        daily_token_limit=daily_token_limit,
                                        daily_request_limit=daily_request_limit,
                                        rpm_limit=rpm_limit)
        log.info(f"Client created: {client_id} ({label})")
        return result
    except ValueError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})


@app.patch("/admin/clients/{client_id}", dependencies=[Depends(_verify_admin)])
async def admin_update_client(client_id: str, request: Request):
    """Update a client's label, notes, token, or rate limits. Body: {"label": "...", "notes": "...", "token": "***", "daily_token_limit": null, "daily_request_limit": 500, "rpm_limit": 30}"""
    body = await request.json()
    # Use Ellipsis sentinel: if key not in body, don't update; if null, clear the limit
    kwargs = {}
    for client_field in ("label", "notes", "token"):
        if client_field in body:
            kwargs[client_field] = body[client_field]
    for limit_field in ("daily_token_limit", "daily_request_limit", "rpm_limit"):
        if limit_field in body:
            kwargs[limit_field] = body[limit_field]  # None clears the limit
    try:
        result = client_registry.update(client_id, **kwargs)
        if result is None:
            return JSONResponse(status_code=404, content={"error": f"client '{client_id}' not found"})
        log.info(f"Client updated: {client_id}")
        return result
    except ValueError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})


@app.delete("/admin/clients/{client_id}", dependencies=[Depends(_verify_admin)])
async def admin_delete_client(client_id: str):
    """Delete a client key."""
    if client_registry.delete(client_id):
        log.info(f"Client deleted: {client_id}")
        return {"deleted": client_id}
    return JSONResponse(status_code=404, content={"error": f"client '{client_id}' not found"})


@app.post("/admin/clients/{client_id}/regenerate-token", dependencies=[Depends(_verify_admin)])
async def admin_regenerate_token(client_id: str):
    """Generate a new token for a client (e.g. if compromised)."""
    result = client_registry.regenerate_token(client_id)
    if result is None:
        return JSONResponse(status_code=404, content={"error": f"client '{client_id}' not found"})
    log.info(f"Token regenerated for client: {client_id}")
    return result


# ---------------------------------------------------------------------------
# Dashboard — Single-page HTML UI
# ---------------------------------------------------------------------------




@app.get("/admin/quota-cost", dependencies=[Depends(_verify_admin)])
async def admin_quota_cost():
    """Return per-model request counts and bar percentages from ollama.com/settings.

    Unlike the old algebraic solver, this uses the per-model usage segments
    Ollama already renders on the settings page. It gives you the actual model
    mix contributing to each key's session and weekly quota bars.
    """
    if not manager:
        return {"keys": [], "note": "Manager not initialized"}

    result_keys = []
    for key in manager.keys:
        result_keys.append({
            "label": key.label,
            "token_prefix": key.token[:8] + "...",
            "session_usage_pct": key.session_usage_pct,
            "session_resets_at": key.session_resets_at,
            "weekly_usage_pct": key.weekly_usage_pct,
            "weekly_resets_at": key.weekly_resets_at,
            "session_models": key.session_models,
            "weekly_models": key.weekly_models,
        })

    return {
        "keys": result_keys,
        "note": "Per-model request counts and bar percentages scraped from ollama.com/settings. "
                "Use this to see which models dominate each key's quota usage.",
    }


@app.get("/admin/quota-coefficients", dependencies=[Depends(_verify_admin)])
async def admin_quota_coefficients(days: int = 7):
    """Return implied Ollama quota cost coefficients per model.

    Coefficients are derived by dividing each model's scraped quota-share
    percentage by its actual token-share percentage, averaged across all
    configured keys. This lets you compare models by Ollama's internal
    quota cost, not just raw token volume.
    """
    if not usage_db:
        return {"models": [], "note": "Usage DB not available"}
    keys = manager.keys if manager else []
    return usage_db.quota_coefficients(keys, days=days)


@app.get("/dashboard")
async def dashboard():
    return FileResponse(DASHBOARD_PATH, media_type="text/html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Allow env var overrides from CLI
    if os.environ.get("LLAMAHERD_ADMIN_TOKEN"):
        cfg["admin_token"] = os.environ["LLAMAHERD_ADMIN_TOKEN"]
    if os.environ.get("LLAMAHERD_HOST"):
        cfg["host"] = os.environ["LLAMAHERD_HOST"]
    if os.environ.get("LLAMAHERD_PORT"):
        cfg["port"] = int(os.environ["LLAMAHERD_PORT"])

    host = cfg.get("host", "127.0.0.1")
    port = cfg.get("port", 8399)
    log.setLevel(logging.INFO)

    # Startup banner (stderr so it doesn't interfere with piped JSON)
    from . import __tagline__
    _banner = r"""
    __    __                      __  __              __
   / /   / /___ _____ ___  ____ _/ / / /__  _________/ /
  / /   / / __ `/ __ `__ \/ __ `/ /_/ / _ \/ ___/ __  /
 / /___/ / /_/ / / / / / / /_/ / __  /  __/ /  / /_/ /
/_____/_/\__,_/_/ /_/ /_/\__,_/_/ /_/\___/_/   \__,_/
""".strip("\n")
    print(f"\n{_banner}\n\n  {__tagline__}\n  http://{host}:{port}/dashboard\n", file=sys.stderr)

    config = Config(app, host=host, port=port, log_level="info")
    server = Server(config)
    log.info(f"LlamaHerd listening on {host}:{port}")
    server.run()


if __name__ == "__main__":
    main()

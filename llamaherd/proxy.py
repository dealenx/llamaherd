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
import json
import logging
import secrets
import sqlite3
import time
import uuid
import queue
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
import os
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from uvicorn import Config, Server

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
# Client Identity Registry (DB-backed, dynamic)
# ---------------------------------------------------------------------------

class ClientRegistry:
    """Maps consumer API keys to client identity for usage attribution.
    
    Backed by SQLite so keys survive restarts. Config.yaml seeds are only
    inserted on first run (if the DB is empty).
    """

    def __init__(self, db_path: str, seed_clients=None):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created REAL NOT NULL,
                notes TEXT DEFAULT '',
                daily_token_limit INTEGER DEFAULT NULL,
                daily_request_limit INTEGER DEFAULT NULL,
                rpm_limit INTEGER DEFAULT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_clients_token ON clients(token)")
        # Safe migration: add rate limit columns if they don't exist (existing DBs)
        for col, typ in [("daily_token_limit", "INTEGER"), ("daily_request_limit", "INTEGER"), ("rpm_limit", "INTEGER")]:
            try:
                self._conn.execute(f"ALTER TABLE clients ADD COLUMN {col} {typ} DEFAULT NULL")
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

        # In-memory cache — initialize BEFORE any _insert/_reload calls
        self._by_token: dict[str, dict] = {}
        self._by_id: dict[str, dict] = {}

        # Seed from config only if table is empty
        if seed_clients and self._count() == 0:
            for c in seed_clients:
                self._insert(c["id"], c.get("label", c["id"]), c["token"],
                             notes=c.get("notes", "seeded from config"),
                             daily_token_limit=c.get("daily_token_limit"),
                             daily_request_limit=c.get("daily_request_limit"),
                             rpm_limit=c.get("rpm_limit"))
            log.info(f"Seeded {len(seed_clients)} clients from config")
        else:
            self._reload()

    def _count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]

    def _reload(self):
        """Refresh in-memory cache from DB."""
        self._by_token.clear()
        self._by_id.clear()
        rows = self._conn.execute("SELECT id, label, token, created, notes, daily_token_limit, daily_request_limit, rpm_limit FROM clients").fetchall()
        for r in rows:
            entry = {"id": r[0], "label": r[1], "token": r[2], "created": r[3], "notes": r[4],
                     "daily_token_limit": r[5], "daily_request_limit": r[6], "rpm_limit": r[7]}
            self._by_token[r[2]] = entry
            self._by_id[r[0]] = entry

    def _insert(self, client_id: str, label: str, token: str, notes: str = "",
                daily_token_limit: int = None, daily_request_limit: int = None,
                rpm_limit: int = None) -> dict:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO clients (id, label, token, created, notes, daily_token_limit, daily_request_limit, rpm_limit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (client_id, label, token, now, notes, daily_token_limit, daily_request_limit, rpm_limit),
        )
        self._conn.commit()
        self._reload()
        return {"id": client_id, "label": label, "token": token, "created": now, "notes": notes,
                "daily_token_limit": daily_token_limit, "daily_request_limit": daily_request_limit,
                "rpm_limit": rpm_limit}

    def resolve(self, token: str) -> dict:
        """Resolve a Bearer token to a client identity. Unknown tokens still work."""
        if token in self._by_token:
            return self._by_token[token]
        # Check if it's an ID used as a token (convenience)
        if token in self._by_id:
            return self._by_id[token]
        return {"id": "unknown", "label": f"Unknown ({token[:8]}...)", "token": token}

    def create(self, client_id: str, label: str, notes: str = "",
               token: Optional[str] = None,
               daily_token_limit: int = None, daily_request_limit: int = None,
               rpm_limit: int = None) -> dict:
        """Create a new client. Auto-generates a token if not provided."""
        if client_id in self._by_id:
            raise ValueError(f"client id '{client_id}' already exists")
        if not token:
            token = f"ocp-{client_id}-{secrets.token_hex(8)}"
        if token in self._by_token:
            raise ValueError(f"token already in use by client '{self._by_token[token]['id']}'")
        return self._insert(client_id, label, token, notes=notes,
                            daily_token_limit=daily_token_limit,
                            daily_request_limit=daily_request_limit,
                            rpm_limit=rpm_limit)

    def update(self, client_id: str, label: Optional[str] = None,
               notes: Optional[str] = None, token: Optional[str] = None,
               daily_token_limit: Optional[int] = ...,
               daily_request_limit: Optional[int] = ...,
               rpm_limit: Optional[int] = ...) -> Optional[dict]:
        """Update an existing client's label, notes, token, or rate limits.
        Use ... (Ellipsis) as sentinel to distinguish None (clear limit) from 'not provided'."""
        if client_id not in self._by_id:
            return None
        existing = self._by_id[client_id]
        new_label = label if label is not None else existing["label"]
        new_notes = notes if notes is not None else existing["notes"]
        new_token = token if token is not None else existing["token"]
        new_dtl = daily_token_limit if daily_token_limit is not ... else existing.get("daily_token_limit")
        new_drl = daily_request_limit if daily_request_limit is not ... else existing.get("daily_request_limit")
        new_rpm = rpm_limit if rpm_limit is not ... else existing.get("rpm_limit")
        if new_token != existing["token"] and new_token in self._by_token:
            raise ValueError(f"token already in use by client '{self._by_token[new_token]['id']}'")
        self._conn.execute(
            "UPDATE clients SET label=?, notes=?, token=?, daily_token_limit=?, daily_request_limit=?, rpm_limit=? WHERE id=?",
            (new_label, new_notes, new_token, new_dtl, new_drl, new_rpm, client_id),
        )
        self._conn.commit()
        self._reload()
        return self._by_id.get(client_id)

    def delete(self, client_id: str) -> bool:
        """Delete a client by id. Returns True if deleted."""
        cur = self._conn.execute("DELETE FROM clients WHERE id=?", (client_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            self._reload()
        return deleted

    def regenerate_token(self, client_id: str) -> dict | None:
        """Generate a new token for a client. Returns updated client or None."""
        if client_id not in self._by_id:
            return None
        new_token = f"ocp-{client_id}-{secrets.token_hex(8)}"
        self._conn.execute("UPDATE clients SET token=? WHERE id=?", (new_token, client_id))
        self._conn.commit()
        self._reload()
        return self._by_id.get(client_id)

    @property
    def clients(self) -> list[dict]:
        return list(self._by_id.values())

# ---------------------------------------------------------------------------
# Key State (upstream Ollama Cloud subscriptions)
# ---------------------------------------------------------------------------

@dataclass
class KeyState:
    token: str
    max_concurrent: int = 15
    cycle_day: int = 1  # fallback if no subscription data
    label: str = ""
    in_flight: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    total_429s: int = 0
    last_429: float = 0.0
    exhausted: bool = False
    exhausted_until: float = 0.0
    # Populated by /api/me subscription poll
    plan: str = ""
    period_start: Optional[str] = None  # ISO timestamp
    period_end: Optional[str] = None      # ISO timestamp
    suspended: bool = False
    account_email: str = ""
    account_id: str = ""
    # Populated by cookie-based settings scrape
    session_usage_pct: float = -1.0  # -1 = unknown
    session_resets_at: Optional[str] = None
    weekly_usage_pct: float = -1.0
    weekly_resets_at: Optional[str] = None

    @property
    def available_slots(self) -> int:
        if self.exhausted and time.time() < self.exhausted_until:
            return 0
        if self.exhausted and time.time() >= self.exhausted_until:
            self.exhausted = False
        return max(0, self.max_concurrent - self.in_flight)

    def mark_exhausted(self, seconds: int = 3600):
        self.exhausted = True
        self.exhausted_until = time.time() + seconds
        self.last_429 = time.time()
        self.total_429s += 1

    @property
    def cycle_freshness(self) -> float:
        """0.0 = just reset, 1.0 = about to reset. Lower is fresher.
        
        Uses real subscription period from /api/me if available,
        falls back to cycle_day config otherwise.
        """
        if self.period_start and self.period_end:
            try:
                start = datetime.fromisoformat(self.period_start.replace("Z", "+00:00"))
                end = datetime.fromisoformat(self.period_end.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                total = (end - start).total_seconds()
                elapsed = (now - start).total_seconds()
                if total <= 0:
                    return 0.5  # edge case
                return max(0.0, min(1.0, elapsed / total))
            except (ValueError, TypeError):
                pass
        # Fallback: use cycle_day
        now = datetime.now(timezone.utc)
        day = now.day
        cycle = self.cycle_day
        if cycle <= day:
            days_into_cycle = day - cycle
        else:
            days_into_cycle = (30 - cycle) + day
        return days_into_cycle / 30.0

    @property
    def period_remaining_pct(self) -> float:
        """Percentage of billing period remaining (0-100)."""
        if self.period_start and self.period_end:
            try:
                start = datetime.fromisoformat(self.period_start.replace("Z", "+00:00"))
                end = datetime.fromisoformat(self.period_end.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                total = (end - start).total_seconds()
                remaining = (end - now).total_seconds()
                if total <= 0:
                    return 0.0
                return max(0.0, min(100.0, (remaining / total) * 100))
            except (ValueError, TypeError):
                pass
        # Fallback: use cycle_day
        now = datetime.now(timezone.utc)
        remaining_days = (self.cycle_day - now.day) % 30 or 30
        return round((remaining_days / 30) * 100, 1)

    def _elapsed_from_iso(self, iso_start: Optional[str], iso_end: Optional[str]) -> float:
        """Calculate elapsed percentage (0-100) between two ISO timestamps. Returns -1 if unknown."""
        if not iso_start or not iso_end:
            return -1.0
        try:
            start = datetime.fromisoformat(iso_start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(iso_end.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            total = (end - start).total_seconds()
            elapsed = (now - start).total_seconds()
            if total <= 0:
                return 100.0
            return round(max(0.0, min(100.0, (elapsed / total) * 100)), 1)
        except (ValueError, TypeError):
            return -1.0

    def _session_elapsed_pct(self) -> float:
        """Percentage of the current 5-hour session that has elapsed. -1 if unknown."""
        if self.session_resets_at:
            # session_resets_at is when the session ENDS (resets)
            # Session is 5 hours = 18000 seconds
            try:
                end = datetime.fromisoformat(self.session_resets_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                remaining = (end - now).total_seconds()
                total = 18000  # 5 hours
                elapsed = total - remaining
                if total <= 0:
                    return 100.0
                return round(max(0.0, min(100.0, (elapsed / total) * 100)), 1)
            except (ValueError, TypeError):
                return -1.0
        return -1.0

    def _weekly_elapsed_pct(self) -> float:
        """Percentage of the current weekly usage window that has elapsed. -1 if unknown."""
        if self.weekly_resets_at:
            try:
                end = datetime.fromisoformat(self.weekly_resets_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                remaining = (end - now).total_seconds()
                total = 7 * 86400  # 7 days
                elapsed = total - remaining
                if total <= 0:
                    return 100.0
                return round(max(0.0, min(100.0, (elapsed / total) * 100)), 1)
            except (ValueError, TypeError):
                return -1.0
        return -1.0


class KeyManager:
    def __init__(self, keys_config: list[dict]):
        self.keys: list[KeyState] = []
        self._lock = asyncio.Lock()
        for kc in keys_config:
            self.keys.append(KeyState(
                token=kc["token"],
                max_concurrent=kc.get("max_concurrent", 15),
                cycle_day=kc.get("cycle_day", 1),
                label=kc.get("label", ""),
            ))

    async def poll_subscriptions(self):
        """Poll /api/me for each key to get subscription status."""
        async with httpx.AsyncClient(timeout=15) as client:
            for key in self.keys:
                try:
                    resp = await client.post(
                        "https://ollama.com/api/me",
                        headers={"Authorization": f"Bearer {key.token}"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        key.plan = data.get("Plan", "")
                        key.account_email = data.get("Email", "")
                        key.account_id = data.get("ID", "")
                        key.suspended = data.get("SuspendedAt", {}).get("Valid", False)
                        
                        period_start = data.get("SubscriptionPeriodStart", {})
                        period_end = data.get("SubscriptionPeriodEnd", {})
                        if period_start.get("Valid"):
                            key.period_start = period_start.get("Time", "")
                        if period_end.get("Valid"):
                            key.period_end = period_end.get("Time", "")
                        
                        log.info(f"Sub poll for {key.label}: plan={key.plan} "
                                f"period={key.period_start[:10] if key.period_start else '?'} "
                                f"to {key.period_end[:10] if key.period_end else '?'} "
                                f"suspended={key.suspended}")
                    else:
                        log.warning(f"Sub poll failed for {key.label}: {resp.status_code}")
                except Exception as e:
                    log.warning(f"Sub poll error for {key.label}: {e}")

    async def acquire(self, prefer_key: Optional[str] = None) -> Optional[KeyState]:
        async with self._lock:
            if prefer_key:
                for k in self.keys:
                    if k.token == prefer_key and k.available_slots > 0:
                        k.in_flight += 1
                        return k

            candidates = [k for k in self.keys if k.available_slots > 0]
            if not candidates:
                return None

            # Least-connections with usage awareness:
            # Spread concurrent load first (in_flight), then prefer less-used keys
            candidates.sort(key=lambda k: (
                k.in_flight,
                k.weekly_usage_pct if k.weekly_usage_pct >= 0 else 999,
                k.session_usage_pct if k.session_usage_pct >= 0 else 999,
                k.cycle_freshness,
                k.total_tokens,
            ))
            best = candidates[0]
            best.in_flight += 1
            return best

    async def release(self, key: KeyState, tokens_used: int = 0):
        async with self._lock:
            key.in_flight = max(0, key.in_flight - 1)
            key.total_requests += 1
            key.total_tokens += tokens_used

    async def mark_429(self, key: KeyState):
        async with self._lock:
            # 429 from Ollama Cloud is a transient concurrency/rate backoff, not a
            # quota exhaustion signal.  A one-hour cooldown strands healthy keys
            # and sends known Ollama models to fallback long after the queue has
            # drained.  Keep this short so routing re-probes Ollama quickly.
            key.mark_exhausted(60)

    async def mark_402(self, key: KeyState):
        async with self._lock:
            key.mark_exhausted(86400)

    def status(self) -> list[dict]:
        now = time.time()
        return [{
            "label": k.label,
            "token_prefix": k.token[:8] + "...",
            "in_flight": k.in_flight,
            "available_slots": k.available_slots,
            "max_concurrent": k.max_concurrent,
            "total_requests": k.total_requests,
            "total_tokens": k.total_tokens,
            "total_429s": k.total_429s,
            "exhausted": k.exhausted,
            "cycle_freshness": round(k.cycle_freshness, 4),
            "period_remaining_pct": round(k.period_remaining_pct, 1),
            "plan": k.plan,
            "period_start": k.period_start,
            "period_end": k.period_end,
            "suspended": k.suspended,
            "account_email": k.account_email,
            "session_usage_pct": k.session_usage_pct,
            "session_resets_at": k.session_resets_at,
            "session_elapsed_pct": k._session_elapsed_pct(),
            "weekly_usage_pct": k.weekly_usage_pct,
            "weekly_resets_at": k.weekly_resets_at,
            "weekly_elapsed_pct": k._weekly_elapsed_pct(),
        } for k in self.keys]

    def key_by_token_prefix(self, prefix: str) -> Optional[KeyState]:
        """Look up a key by its token prefix (first 8 chars)."""
        for k in self.keys:
            if k.token[:8] == prefix[:8]:
                return k
        return None

# ---------------------------------------------------------------------------
# Usage Scraper (cookie-based ollama.com/settings)
# ---------------------------------------------------------------------------

class UsageScraper:
    """Scrapes session/weekly usage from ollama.com/settings using browser cookies.
    
    Requires __Secure-session, aid, and cf_clearance cookies per key.
    Falls back gracefully if cookies are missing or expired.
    """

    def __init__(self, keys_config: list[dict]):
        self.cookie_map: dict[str, dict] = {}  # label -> {secure_session, aid, cf_clearance}
        for kc in keys_config:
            label = kc.get("label", "")
            cookies = kc.get("cookies", {})
            if cookies and cookies.get("secure_session"):
                self.cookie_map[label] = {
                    "secure_session": cookies["secure_session"],
                    "aid": cookies.get("aid", ""),
                    "cf_clearance": cookies.get("cf_clearance", ""),
                    "stripe_mid": cookies.get("stripe_mid", ""),
                }

    def scrape_usage(self, key: KeyState) -> dict | None:
        """Scrape usage data for a single key. Returns dict or None."""
        if key.label not in self.cookie_map:
            return None
        cookies = self.cookie_map[key.label]
        if not cookies.get("secure_session"):
            return None

        try:
            import cloudscraper
            from bs4 import BeautifulSoup
        except ImportError:
            log.warning("cloudscraper or beautifulsoup4 not installed — usage scraping disabled")
            return None

        try:
            scraper = cloudscraper.create_scraper()
            scraper.cookies.set("__Secure-session", cookies["secure_session"], domain="ollama.com")
            if cookies.get("aid"):
                scraper.cookies.set("aid", cookies["aid"], domain="ollama.com")
            if cookies.get("cf_clearance"):
                scraper.cookies.set("cf_clearance", cookies["cf_clearance"], domain="ollama.com")
            if cookies.get("stripe_mid"):
                scraper.cookies.set("__stripe_mid", cookies["stripe_mid"], domain="ollama.com")

            resp = scraper.get(
                "https://ollama.com/settings",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=15,
            )
            resp.raise_for_status()

            if "Just a moment" in resp.text:
                log.warning(f"Usage scrape for {key.label}: Cloudflare challenge — need fresh cf_clearance cookie")
                return None
            if "Sign in" in resp.text[:3000]:
                log.warning(f"Usage scrape for {key.label}: auth required — __Secure-session cookie expired")
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            result = {"session_usage_pct": -1.0, "weekly_usage_pct": -1.0,
                      "session_resets_at": None, "weekly_resets_at": None}

            # Parse usage percentages
            for div in soup.find_all("div", class_="flex justify-between mb-2"):
                spans = div.find_all("span")
                if len(spans) >= 2:
                    label = spans[0].get_text(strip=True)
                    value = spans[1].get_text(strip=True)
                    pct_str = value.replace("% used", "").replace("%", "").strip()
                    try:
                        pct = float(pct_str)
                    except ValueError:
                        continue
                    if "Session" in label:
                        result["session_usage_pct"] = pct
                    elif "Weekly" in label:
                        result["weekly_usage_pct"] = pct

            # Parse reset times
            for i, div in enumerate(soup.find_all("div", class_="local-time")):
                iso_time = div.get("data-time", "")
                text = div.get_text(strip=True)
                if i == 0:
                    result["session_resets_at"] = iso_time or text
                elif i == 1:
                    result["weekly_resets_at"] = iso_time or text

            log.info(f"Usage scrape for {key.label}: "
                     f"session {result['session_usage_pct']}% "
                     f"weekly {result['weekly_usage_pct']}%")
            return result

        except Exception as e:
            log.warning(f"Usage scrape error for {key.label}: {e}")
            return None

    def scrape_all(self, keys: list[KeyState]) -> dict[str, dict]:
        """Scrape usage for all keys with cookies configured."""
        results = {}
        for key in keys:
            data = self.scrape_usage(key)
            if data:
                key.session_usage_pct = data["session_usage_pct"]
                key.session_resets_at = data["session_resets_at"]
                key.weekly_usage_pct = data["weekly_usage_pct"]
                key.weekly_resets_at = data["weekly_resets_at"]
                results[key.label] = data
        return results


# ---------------------------------------------------------------------------
# Model Discovery
# ---------------------------------------------------------------------------

# Known context lengths for Ollama Cloud models (tokens).
# Used to populate /v1/models metadata so clients (Hermes) can auto-detect
# context windows instead of falling back to 128K defaults.
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


def fmt_param_count(n: Optional[int]) -> str:
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
        return f"{int(round(v))}{suffix}"
    return f"{v:.1f}{suffix}"


class ModelRegistry:
    def __init__(self, manager: KeyManager, upstream: str):
        self.manager = manager
        self.upstream = upstream
        self.models: dict[str, list[str]] = {}
        self.model_metadata: dict[str, dict] = {}
        self.last_refresh: float = 0
        self._refresh_task: Optional[asyncio.Task] = None

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
    def _created_from_modified(modified_at: Optional[str], fallback: float) -> int:
        if modified_at:
            try:
                return int(datetime.fromisoformat(modified_at.replace("Z", "+00:00")).timestamp())
            except Exception:
                pass
        return int(fallback or time.time())

    @staticmethod
    def _context_from_show(show_data: dict) -> Optional[int]:
        info = show_data.get("model_info") or {}
        for key, value in info.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except Exception:
                    return None
        return None

    @staticmethod
    def _parameter_count(show_data: dict) -> Optional[int]:
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
        self.model_metadata = {mid: metadata[mid] for mid in all_models.keys() if mid in metadata}
        self.last_refresh = time.time()
        new_models = set(all_models.keys()) - old_models
        if new_models:
            log.info(f"New models discovered: {sorted(new_models)}")
        log.info(f"Model registry: {len(self.models)} models discovered across {len(self.manager.keys)} keys")
        # Broadcast model changes via SSE
        try:
            await broadcaster.broadcast("models", {
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

    def get_preferred_key(self, model: str) -> Optional[str]:
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

    def __init__(self, config: Optional[dict]):
        cfg = config or {}
        self.provider: str = cfg.get("provider", "fallback")
        self.base_url: str = (cfg.get("base_url") or "").rstrip("/")
        self.api_key: str = cfg.get("api_key", "") or ""
        self.default_model: Optional[str] = cfg.get("default_model")
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

    def resolve_model(self, ollama_model: str) -> Optional[str]:
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
                    priority: Optional[str] = None) -> dict:
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

    async def fetch_model_metadata(self, model_id: str, timeout: float = 5.0) -> Optional[dict]:
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


# ---------------------------------------------------------------------------
# Usage DB — full token tracking with client attribution
# ---------------------------------------------------------------------------

class UsageDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                ts REAL NOT NULL,
                day TEXT NOT NULL,
                client_id TEXT NOT NULL,
                upstream_key TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                status INTEGER NOT NULL,
                session_usage_pct REAL DEFAULT -1.0,
                weekly_usage_pct REAL DEFAULT -1.0
            )
            """)
        # Backward-compatible migration for existing DBs
        for col, default in [("session_usage_pct", -1.0), ("weekly_usage_pct", -1.0)]:
            try:
                self._conn.execute(f"ALTER TABLE usage ADD COLUMN {col} REAL DEFAULT {default}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        for idx_cols in [
            "client_id, day",
            "model, day",
            "day",
            "client_id, model, day",
        ]:
            idx_name = f"idx_usage_{'_'.join(idx_cols.replace(' ', '').split(','))}"
            try:
                self._conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON usage ({idx_cols})")
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def record(self, client_id: str, upstream_key: str, model: str,
               tokens_in: int, tokens_out: int, latency_ms: int, status: int,
               session_usage_pct: float = -1.0, weekly_usage_pct: float = -1.0):
        today = datetime.now(timezone.utc).date().isoformat()
        self._conn.execute(
            "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), today, client_id, upstream_key, model,
             tokens_in, tokens_out, latency_ms, status,
             session_usage_pct, weekly_usage_pct),
        )
        self._conn.commit()

    def summary(self, hours: int = 24, client: str = None, model: str = None) -> list[dict]:
        since = time.time() - hours * 3600
        query = """
            SELECT client_id, model, day,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total,
                   AVG(latency_ms) as avg_latency_ms
            FROM usage WHERE ts > ?
        """
        params: list = [since]
        if client:
            query += " AND client_id = ?"
            params.append(client)
        if model:
            query += " AND model = ?"
            params.append(model)
        query += " GROUP BY client_id, model, day ORDER BY day DESC, tokens_total DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [{
            "client_id": r[0],
            "model": r[1],
            "day": r[2],
            "requests": r[3],
            "tokens_in": r[4] or 0,
            "tokens_out": r[5] or 0,
            "tokens_total": r[6] or 0,
            "avg_latency_ms": round(r[7] or 0, 1),
        } for r in rows]

    def _date_range_where(self, days: int = None, start_date: str = None, end_date: str = None):
        """Build WHERE clause + params for date range filtering.
        Supports both `days` (relative) and start_date/end_date (absolute ISO date).
        Returns (where_clause, params).
        """
        if start_date and end_date:
            return "day >= ? AND day <= ?", [start_date, end_date]
        elif start_date:
            return "day >= ?", [start_date]
        elif end_date:
            return "day <= ?", [end_date]
        else:
            since = time.time() - (days or 30) * 86400
            return "ts > ?", [since]

    def daily_totals(self, days: int = 30, start_date: str = None, end_date: str = None) -> list[dict]:
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT day,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where}
            GROUP BY day ORDER BY day DESC
        """, params).fetchall()
        return [{
            "day": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in rows]

    def by_client(self, days: int = 30, start_date: str = None, end_date: str = None) -> list[dict]:
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT client_id,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where}
            GROUP BY client_id ORDER BY tokens_total DESC
        """, params).fetchall()
        return [{
            "client_id": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in rows]

    def by_model(self, days: int = 30, start_date: str = None, end_date: str = None) -> list[dict]:
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT model,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total,
                   AVG(latency_ms) as avg_latency_ms
            FROM usage WHERE {where}
            GROUP BY model ORDER BY tokens_total DESC
        """, params).fetchall()
        return [{
            "model": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
            "avg_latency_ms": round(r[5] or 0, 1),
        } for r in rows]

    def recent_calls(self, limit: int = 100, start_date: str = None, end_date: str = None,
                     client_id: str = None, model: str = None) -> list[dict]:
        where_parts = []
        params: list = []
        if start_date:
            where_parts.append("day >= ?")
            params.append(start_date)
        if end_date:
            where_parts.append("day <= ?")
            params.append(end_date)
        if client_id:
            where_parts.append("client_id = ?")
            params.append(client_id)
        if model:
            where_parts.append("model = ?")
            params.append(model)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        query = f"""
            SELECT ts, client_id, upstream_key, model,
                   tokens_in, tokens_out, latency_ms, status
            FROM usage WHERE {where}
            ORDER BY ts DESC LIMIT ?
        """
        params.append(min(limit, 500))
        rows = self._conn.execute(query, params).fetchall()
        return [{
            "ts": r[0],
            "time": datetime.fromtimestamp(r[0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "client_id": r[1],
            "upstream_key": r[2],
            "model": r[3],
            "tokens_in": r[4],
            "tokens_out": r[5],
            "tokens_total": r[4] + r[5],
            "latency_ms": r[6],
            "status": r[7],
        } for r in rows]

    def totals(self, start_date: str = None, end_date: str = None) -> dict:
        if start_date or end_date:
            where, params = self._date_range_where(None, start_date, end_date)
            row = self._conn.execute(f"""
                SELECT COUNT(*),
                       COALESCE(SUM(tokens_in), 0),
                       COALESCE(SUM(tokens_out), 0)
                FROM usage WHERE {where}
            """, params).fetchone()
        else:
            row = self._conn.execute("""
                SELECT COUNT(*),
                       COALESCE(SUM(tokens_in), 0),
                       COALESCE(SUM(tokens_out), 0)
                FROM usage
            """).fetchone()
        return {
            "total_calls": row[0],
            "total_tokens_in": row[1],
            "total_tokens_out": row[2],
            "total_tokens": row[1] + row[2],
        }

# ---------------------------------------------------------------------------
# Rate Limiting — per-client daily tokens, daily requests, RPM
# ---------------------------------------------------------------------------

from collections import deque
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


async def _check_rate_limit(request: Request, client: dict) -> Optional[JSONResponse]:
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
        today = datetime.now(timezone.utc).date().isoformat()
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
                    "reset_at": int((datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()),
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
                    "reset_at": int((datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()),
                },
            )

    return None

# ---------------------------------------------------------------------------
# Proxy App — Lifespan
# ---------------------------------------------------------------------------

manager: Optional[KeyManager] = None
registry: Optional[ModelRegistry] = None
usage_db: Optional[UsageDB] = None
client_registry: Optional[ClientRegistry] = None
usage_scraper: Optional[UsageScraper] = None
fallback_provider: Optional[FallbackProvider] = None
upstream_url: str = ""
retry_on_429: bool = True
max_retries: int = 2
queue_timeout: int = 60
request_timeout: int = 120
admin_token: str = ""
reject_unknown_models: bool = False  # reject models unknown to both Ollama and fallback model_map

DB_PATH = Path(__file__).parent / "proxy.db"

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
                    *, headers: Optional[dict] = None,
                    path: Optional[str] = None) -> None:
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


def _update_in_flight_tokens(request_id: Optional[str],
                              tokens_in: Optional[int] = None,
                              tokens_out: Optional[int] = None) -> None:
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
                           *, request_id: Optional[str] = None,
                           provider: Optional[str] = None):
    """Record usage to DB and broadcast call + request_end events to SSE subscribers."""
    # Look up the key's cached usage percentages for quota cost tracking
    s_pct = -1.0
    w_pct = -1.0
    if manager:
        key = manager.key_by_token_prefix(upstream_key)
        if key:
            s_pct = key.session_usage_pct if key.session_usage_pct >= 0 else -1.0
            w_pct = key.weekly_usage_pct if key.weekly_usage_pct >= 0 else -1.0
    usage_db.record(client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status,
                    session_usage_pct=s_pct, weekly_usage_pct=w_pct)
    call_data = {
        "ts": time.time(),
        "client_id": client_id,
        "upstream_key": upstream_key,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "status": status,
    }
    end_data: Optional[dict] = None
    if request_id:
        entry = _in_flight.pop(request_id, None)
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


def _verify_admin(request: Request) -> None:
    """FastAPI dependency: require admin_token via Bearer header or ?token= query param."""
    global admin_token
    if not admin_token:
        raise HTTPException(status_code=500, detail="admin_token not configured")
    # Check query param first (for browser/dashboard access)
    if request.query_params.get("token") == admin_token:
        return
    # Check Authorization header
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and secrets.compare_digest(auth[7:].strip(), admin_token):
        return
    raise HTTPException(status_code=401, detail="unauthorized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, registry, usage_db, client_registry, usage_scraper, fallback_provider
    global upstream_url, retry_on_429, max_retries, queue_timeout, request_timeout
    global admin_token, NATIVE_BRIDGE_MODELS, reject_unknown_models

    cfg = load_config()
    admin_token = cfg.get("admin_token", "")
    if not admin_token:
        log.warning("admin_token not set in config — admin endpoints will be inaccessible")
    else:
        log.info("Admin authentication enabled")
    manager = KeyManager(cfg["keys"])
    client_registry = ClientRegistry(str(DB_PATH), seed_clients=cfg.get("clients"))
    upstream_url = cfg.get("upstream", "https://ollama.com/v1")
    retry_on_429 = cfg.get("retry_on_429", True)
    max_retries = cfg.get("max_retries", 2)
    queue_timeout = cfg.get("queue_timeout", 60)
    request_timeout = cfg.get("request_timeout", 120)
    reject_unknown_models = cfg.get("reject_unknown_models", False)

    # Native bridge: models whose /v1 endpoint misreports truncation
    NATIVE_BRIDGE_MODELS = cfg.get("native_bridge_models", [])
    if NATIVE_BRIDGE_MODELS:
        log.info(f"Native bridge enabled for models: {NATIVE_BRIDGE_MODELS}")

    usage_db = UsageDB(cfg.get("usage_db", "~/ollama-cloud-proxy/usage.db"))
    registry = ModelRegistry(manager, upstream_url)
    await registry.start(cfg.get("health_check_interval", 300))
    await registry.refresh()
    # Initial subscription poll
    await manager.poll_subscriptions()
    # Start periodic sub polling (every 6 hours)
    sub_poll_interval = cfg.get("sub_poll_interval", 21600)
    sub_task = asyncio.create_task(_poll_subscriptions_loop(manager, sub_poll_interval))
    # Usage scraper (cookie-based ollama.com/settings)
    usage_scraper = UsageScraper(cfg["keys"])
    # Scrape usage on startup (in thread pool to not block)
    loop = asyncio.get_event_loop()
    try:
        scrape_results = await loop.run_in_executor(None, usage_scraper.scrape_all, manager.keys)
        if scrape_results:
            log.info(f"Initial usage scrape: {scrape_results}")
    except Exception as e:
        log.warning(f"Initial usage scrape failed: {e}")
    # Start periodic usage scraping (every 30 min)
    usage_scrape_interval = cfg.get("usage_scrape_interval", 1800)
    usage_task = asyncio.create_task(_scrape_usage_loop(usage_scraper, manager, usage_scrape_interval))
    # Fallback provider (NVIDIA Build, etc.)
    fallback_provider = FallbackProvider(cfg.get("fallback") or {})
    fb_metadata_task: Optional[asyncio.Task] = None
    if fallback_provider.enabled:
        # Best-effort discovery — don't block startup if it's slow.
        try:
            await asyncio.wait_for(fallback_provider.discover_models(timeout=5.0), timeout=6.0)
        except asyncio.TimeoutError:
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

    yield

    sub_task.cancel()
    usage_task.cancel()
    sweep_task.cancel()
    if fb_metadata_task is not None:
        fb_metadata_task.cancel()
    if registry:
        await registry.stop()


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


async def _scrape_usage_loop(scraper: UsageScraper, mgr: KeyManager, interval: int):
    """Periodically scrape ollama.com/settings for usage data."""
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(interval)
        try:
            result = await loop.run_in_executor(None, scraper.scrape_all, mgr.keys)
            if result:
                log.info(f"Usage scrape updated: {len(result)} keys")
                # Broadcast updated status after usage scrape
                await broadcaster.broadcast("status", {
                    "keys": mgr.status(),
                })
        except Exception as e:
            log.error(f"Usage scrape loop error: {e}")


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


def _resolve_client(request: Request) -> dict:
    """Extract Bearer token from request and resolve to client identity."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        return client_registry.resolve(token)
    return {"id": "anonymous", "label": "Anonymous", "token": ""}


async def _proxy_request(request: Request, path: str) -> Response:
    """Core proxy logic — acquire key, forward, handle 429s, release."""

    client = _resolve_client(request)
    client_id = client["id"]

    # Check per-client rate limits before doing any upstream work
    rate_limit_response = await _check_rate_limit(request, client)
    if rate_limit_response is not None:
        return rate_limit_response

    request_id = _new_request_id()
    body = await request.body()
    req_json = json.loads(body) if body else {}

    is_stream = req_json.get("stream", False)
    model = req_json.get("model", "unknown")

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
    ollama_has_model = bool(registry and registry.models.get(model))
    fp_mapped = bool(has_fallback and fp.resolve_model(model))
    fp_can_serve = bool(has_fallback and (fp.resolve_model(model) or fp.default_model))
    priority = fp.priority_for(model) if has_fallback else "after"

    # Reject unknown models: when reject_unknown_models is true, models
    # that aren't known to Ollama AND aren't in the fallback model_map
    # get a 404 instead of silently routing to the fallback default_model.
    if reject_unknown_models and not ollama_has_model and not fp_mapped:
        _record_and_broadcast(client_id, "none", model, 0, 0, 0, 404,
                              request_id=request_id, provider="proxy")
        log.warning(f"Rejected unknown model: {model} (client={client_id})")
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{model}' not found. Available models: /v1/models"},
        )

    # priority=only — fallback only for mapped models; Ollama for others.
    if has_fallback and priority == "only" and fp_mapped:
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id)
    # priority=before — try fallback first when a mapping exists.
    if has_fallback and priority == "before" and fp_mapped:
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id)
    # priority=after — model unknown to Ollama but fallback can serve it.
    if has_fallback and priority == "after" and not ollama_has_model and fp_can_serve:
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id)

    prefer_key = registry.get_preferred_key(model) if registry else None

    last_error = None
    start_emitted = False
    for attempt in range(max_retries + 1):
        key = None
        deadline = time.time() + queue_timeout
        while time.time() < deadline:
            key = await manager.acquire(prefer_key=prefer_key)
            if key:
                break
            await asyncio.sleep(0.5)

        if not key:
            # All keys at capacity — fall back if priority allows it.
            if has_fallback and fp_can_serve and priority in ("after", "before") and not ollama_has_model:
                log.warning(f"Ollama keys at capacity for {model}; falling back to {fp.label}")
                return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id)
            if has_fallback and fp_can_serve and priority in ("after", "before") and ollama_has_model:
                log.warning(
                    f"Ollama keys at capacity for known Ollama model {model}; "
                    "not falling back to secondary provider"
                )
            _record_and_broadcast(client_id, "none", model, 0, 0, 0, 503,
                                  request_id=request_id, provider="ollama-cloud")
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
                return await _proxy_bridge_stream(client_id, key, bridge_body, model, start, request_id)

            if is_stream:
                return await _proxy_stream(client_id, key, path, headers, body, model, start, request_id)

            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                resp = await client_http.post(
                    f"{upstream_url}{path}",
                    content=body,
                    headers=headers,
                )

            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code == 429:
                log.warning(f"429 from {key.label} for {model} (client={client_id}, attempt {attempt+1})")
                await manager.mark_429(key)
                await manager.release(key)
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 429)
                prefer_key = None
                continue

            if resp.status_code == 402:
                log.warning(f"402 from {key.label} for {model} (client={client_id})")
                await manager.mark_402(key)
                await manager.release(key)
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 402)
                prefer_key = None
                continue

            resp_data = resp.json() if resp.status_code == 200 else {}
            usage = resp_data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms,
                                  resp.status_code, request_id=request_id, provider="ollama-cloud")

            log.info(f"{client_id} -> {model} via {key.label}: {tokens_in}+{tokens_out}tok {elapsed_ms}ms")

            if resp.status_code >= 400:
                log.warning(f"{resp.status_code} from {key.label} for {model}: {resp.text[:200]}")

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key)
            _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, -1)
            last_error = str(e)
            log.error(f"Proxy error for {model} (client={client_id}): {e}")
            continue

    # Ollama exhausted retries — try fallback as a last resort.
    if has_fallback and fp_can_serve and priority in ("after", "before") and not ollama_has_model:
        log.warning(f"Ollama exhausted retries for {model}; falling back to {fp.label}")
        # Pop the Ollama in-flight entry so the fallback emits a fresh start.
        _in_flight.pop(request_id, None)
        return await _route_to_fallback(client_id, fp, path, body, req_json, model, is_stream, request_id)
    if has_fallback and fp_can_serve and priority in ("after", "before") and ollama_has_model:
        log.warning(
            f"Ollama exhausted retries for known Ollama model {model}; "
            "not falling back to secondary provider"
        )

    _record_and_broadcast(client_id, "none", model, 0, 0, 0, 502,
                          request_id=request_id, provider="ollama-cloud")
    return JSONResponse(
        status_code=502,
        content={"error": f"all retries exhausted: {last_error}"},
    )


async def _proxy_stream(client_id: str, key: KeyState, path: str,
                         headers: dict, body: bytes, model: str, start: float,
                         request_id: Optional[str] = None) -> StreamingResponse:

    async def generate():
        tokens_out = 0
        tokens_in = 0
        usage_captured = False
        final_status = 200
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                async with client_http.stream("POST", f"{upstream_url}{path}",
                                              content=body, headers=headers) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        final_status = 429
                        yield f'data: {{"error": "429 from upstream"}}\n\n'
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
                        final_status = 402
                        yield f'data: {{"error": "402 from upstream"}}\n\n'
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
            log.error(f"Stream error for {model} (client={client_id}): {e}")
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms,
                                  final_status, request_id=request_id, provider="ollama-cloud")
            usage_src = "usage" if usage_captured else "estimate"
            log.info(f"{client_id} -> {model} via {key.label}: stream {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({usage_src})")

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Fallback routing (e.g. NVIDIA Build) — same OpenAI /v1 protocol
# ---------------------------------------------------------------------------

async def _route_to_fallback(client_id: str, fp: FallbackProvider, path: str,
                              body: bytes, req_json: dict, original_model: str,
                              is_stream: bool, request_id: Optional[str] = None) -> Response:
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
            request_id,
        )

    async with httpx.AsyncClient(timeout=request_timeout) as ch:
        resp = await ch.post(url, content=new_body, headers=headers)
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
                          request_id=request_id, provider=fp.provider)
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
                                  request_id: Optional[str] = None) -> StreamingResponse:
    async def generate():
        tokens_in = 0
        tokens_out = 0
        usage_captured = False
        status_code = 200
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as ch:
                async with ch.stream("POST", url, content=body, headers=headers) as resp:
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
            elapsed_ms = int((time.time() - start) * 1000)
            _record_and_broadcast(client_id, upstream_label, original_model,
                                  tokens_in, tokens_out, elapsed_ms, status_code,
                                  request_id=request_id, provider=fp.provider)
            src = "usage" if usage_captured else "estimate"
            log.info(f"{client_id} -> {original_model} via {upstream_label}({mapped}): "
                     f"stream {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({src})")

    return StreamingResponse(generate(), media_type="text/event-stream")


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
                                start: float, request_id: Optional[str] = None) -> StreamingResponse:
    """Stream NDJSON from the native Ollama API, capturing usage from the final chunk."""

    async def generate():
        tokens_out = 0
        tokens_in = 0
        usage_captured = False
        final_status = 200
        api_upstream = _native_api_upstream()
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                async with client_http.stream("POST", f"{api_upstream}{path}",
                                              content=body, headers=headers) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        final_status = 429
                        yield json.dumps({"error": "429 from upstream"}) + "\n"
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
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
            log.error(f"NDJSON stream error for {model} (client={client_id}): {e}")
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms,
                                  final_status, request_id=request_id, provider="ollama-cloud")
            usage_src = "usage" if usage_captured else "estimate"
            log.info(f"{client_id} -> {model} via {key.label}: ndjson {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({usage_src})")

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def _proxy_ndjson_request(request: Request, path: str) -> Response:
    """Core proxy logic for native Ollama /api/* routes — acquire key, forward, handle 429s, release.

    Handles both streaming (NDJSON) and non-streaming (JSON) native Ollama API requests.
    """
    client = _resolve_client(request)
    client_id = client["id"]

    request_id = _new_request_id()
    body = await request.body()
    req_json = json.loads(body) if body else {}
    model = req_json.get("model", "unknown")
    is_stream = req_json.get("stream", False)

    prefer_key = registry.get_preferred_key(model) if registry else None

    last_error = None
    start_emitted = False
    for attempt in range(max_retries + 1):
        key = None
        deadline = time.time() + queue_timeout
        while time.time() < deadline:
            key = await manager.acquire(prefer_key=prefer_key)
            if key:
                break
            await asyncio.sleep(0.5)

        if not key:
            _record_and_broadcast(client_id, "none", model, 0, 0, 0, 503,
                                  request_id=request_id, provider="ollama-cloud")
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

        try:
            start = time.time()
            headers = {
                "Authorization": f"Bearer {key.token}",
                "Content-Type": "application/json",
            }
            api_upstream = _native_api_upstream()

            if is_stream:
                return await _proxy_ndjson_stream(client_id, key, path, headers, body, model, start, request_id)

            # Non-streaming: regular JSON response
            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                resp = await client_http.post(
                    f"{api_upstream}{path}",
                    content=body,
                    headers=headers,
                )

            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code == 429:
                log.warning(f"429 from {key.label} for {model} (client={client_id}, attempt {attempt+1})")
                await manager.mark_429(key)
                await manager.release(key)
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 429)
                prefer_key = None
                continue

            if resp.status_code == 402:
                log.warning(f"402 from {key.label} for {model} (client={client_id})")
                await manager.mark_402(key)
                await manager.release(key)
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 402)
                prefer_key = None
                continue

            # Extract usage from non-streaming response
            resp_data = resp.json() if resp.status_code == 200 else {}
            tokens_in = resp_data.get("prompt_eval_count", 0) or 0
            tokens_out = resp_data.get("eval_count", 0) or 0
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms,
                                  resp.status_code, request_id=request_id, provider="ollama-cloud")

            log.info(f"{client_id} -> {model} via {key.label}: {tokens_in}+{tokens_out}tok {elapsed_ms}ms (native)")

            if resp.status_code >= 400:
                log.warning(f"{resp.status_code} from {key.label} for {model}: {resp.text[:200]}")

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key)
            _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, -1)
            last_error = str(e)
            log.error(f"Native proxy error for {model} (client={client_id}): {e}")
            continue

    _record_and_broadcast(client_id, "none", model, 0, 0, 0, 502,
                          request_id=request_id, provider="ollama-cloud")
    return JSONResponse(
        status_code=502,
        content={"error": f"all retries exhausted: {last_error}"},
    )


# ---------------------------------------------------------------------------
# Routes — OpenAI-compatible
# ---------------------------------------------------------------------------

@app.get("/v1/models")
@app.get("/v1/models/")
async def list_models():
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
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
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
async def api_tags():
    """Return model list in Ollama native /api/tags format."""
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
                datetime.fromtimestamp(registry.last_refresh, tz=timezone.utc).isoformat()
                if registry.last_refresh else ""
            ),
            "size": meta.get("size") or 0,
            "digest": meta.get("digest") or "",
            "details": details,
            "size_vram": meta.get("size_vram", 0),
        }
        models.append(entry)
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
    client = _resolve_client(request)
    client_id = client["id"]
    body = await request.body()
    req_json = json.loads(body) if body else {}
    model = req_json.get("name", req_json.get("model", "unknown"))

    prefer_key = registry.get_preferred_key(model) if registry else None

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

            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                resp = await client_http.post(
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
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
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
async def api_ps():
    """Native Ollama /api/ps — return empty models list (we don't track running models)."""
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


def _parse_openai_tool_args(args: Any) -> Any:
    """Convert OpenAI tool call arguments to Ollama native shape.

    OpenAI stores function.arguments as a JSON string. Ollama native /api/chat
    expects the arguments value to be a JSON object. Passing the string through
    causes Ollama Cloud's parser to reject the body with:
      Value looks like object, but can't find closing '}' symbol
    """
    if isinstance(args, str):
        try:
            return json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            # Keep non-JSON strings as-is; better to preserve data than invent.
            return args
    return args


def _openai_tool_calls_to_ollama(tool_calls: Any) -> list[dict]:
    """Convert OpenAI assistant tool_calls to Ollama native tool_calls."""
    out: list[dict] = []
    if not isinstance(tool_calls, list):
        return out
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        native = {
            "function": {
                "name": fn.get("name", ""),
                "arguments": _parse_openai_tool_args(fn.get("arguments", {})),
            }
        }
        out.append(native)
    return out


def _convert_openai_messages_to_ollama(messages: Any) -> list[dict]:
    """Convert OpenAI chat messages to Ollama native /api/chat messages."""
    if not isinstance(messages, list):
        return []

    # Map OpenAI tool_call IDs to names so following role=tool messages can use
    # Ollama's native tool_name field instead of OpenAI's tool_call_id.
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            fn = call.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if call_id and name:
                tool_id_to_name[str(call_id)] = str(name)

    converted: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        out: dict = {}
        if role:
            out["role"] = role

        # Ollama expects content to be a string. OpenAI sometimes uses null for
        # assistant tool-call messages; normalize that to an empty string.
        if "content" in msg:
            out["content"] = "" if msg.get("content") is None else msg.get("content")

        if "images" in msg:
            out["images"] = msg["images"]
        if "name" in msg:
            out["name"] = msg["name"]

        if "tool_calls" in msg:
            native_calls = _openai_tool_calls_to_ollama(msg["tool_calls"])
            if native_calls:
                out["tool_calls"] = native_calls

        # OpenAI uses tool_call_id on tool-result messages. Ollama native uses
        # tool_name. Translate when possible, otherwise omit the unsupported ID.
        if "tool_name" in msg:
            out["tool_name"] = msg["tool_name"]
        elif "tool_call_id" in msg:
            name = tool_id_to_name.get(str(msg["tool_call_id"]))
            if name:
                out["tool_name"] = name

        # OpenAI/Ollama Cloud call this reasoning; Ollama native calls it thinking.
        if "thinking" in msg:
            out["thinking"] = msg["thinking"]
        elif "reasoning" in msg:
            out["thinking"] = msg["reasoning"]

        converted.append(out)
    return converted


def _convert_openai_to_ollama_body(req_json: dict) -> bytes:
    """Convert an OpenAI /v1/chat/completions request body to Ollama /api/chat format.

    Maps: messages, tools, max_tokens→options.num_predict, temperature→options.temperature,
    top_p→options.top_p, stream, model.  OpenAI-specific fields (stream_options, n, etc.)
    are dropped.
    """
    ollama: dict = {"model": req_json.get("model", "")}
    if "messages" in req_json:
        ollama["messages"] = _convert_openai_messages_to_ollama(req_json["messages"])
    if "stream" in req_json:
        ollama["stream"] = req_json["stream"]
    if "tools" in req_json:
        ollama["tools"] = req_json["tools"]

    # OpenAI-compatible reasoning controls → Ollama native thinking controls.
    # Ollama native accepts: think=true/false/"low"/"medium"/"high".
    if "think" in req_json:
        ollama["think"] = req_json["think"]
    elif "reasoning_effort" in req_json:
        effort = req_json.get("reasoning_effort")
        ollama["think"] = False if effort in (None, "none", "off", "false") else effort
    elif "reasoning" in req_json:
        reasoning = req_json.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort") or reasoning.get("level")
            if effort:
                ollama["think"] = False if effort in ("none", "off", "false") else effort
        elif isinstance(reasoning, (bool, str)):
            ollama["think"] = reasoning

    # Pack OpenAI kwargs into Ollama options dict
    options: dict = {}
    if "max_tokens" in req_json:
        options["num_predict"] = req_json["max_tokens"]
    if "temperature" in req_json:
        options["temperature"] = req_json["temperature"]
    if "top_p" in req_json:
        options["top_p"] = req_json["top_p"]
    if "frequency_penalty" in req_json:
        options["frequency_penalty"] = req_json["frequency_penalty"]
    if "presence_penalty" in req_json:
        options["presence_penalty"] = req_json["presence_penalty"]
    if "seed" in req_json:
        options["seed"] = req_json["seed"]
    if "stop" in req_json:
        # Ollama uses 'stop' directly at top level
        ollama["stop"] = req_json["stop"]
    if options:
        ollama["options"] = options

    return json.dumps(ollama).encode()


def _convert_ollama_tool_calls(ollama_tools: list[dict]) -> list[dict]:
    """Convert Ollama tool_calls format to OpenAI streaming format.

    Ollama: {"function": {"name": "x", "arguments": {dict}}}
    OpenAI: {"index": 0, "id": "call_xxx", "type": "function",
             "function": {"name": "x", "arguments": "{json_string}"}}
    """
    openai_tools = []
    for i, tc in enumerate(ollama_tools or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        # Ollama returns arguments as a dict; OpenAI wants a JSON string
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        openai_tools.append({
            "index": i,
            # Generate a deterministic-ish call ID from function name
            "id": f"call_{fn.get('name', 'unknown')}_{i}",
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "arguments": args_str,
            },
        })
    return openai_tools


def _ollama_chunk_to_sse(ollama_chunk: dict, chunk_id: str, model: str) -> str | None:
    """Convert a single Ollama /api/chat NDJSON chunk to an OpenAI SSE data line.

    Returns None for chunks that shouldn't be emitted (empty content, non-message chunks).
    Returns the SSE line WITHOUT the trailing \\n\\n (caller adds SSE framing).
    """
    # Only process chunks with a "message" field (content or tool_calls chunks)
    if "message" not in ollama_chunk and not ollama_chunk.get("done", False):
        return None

    done = ollama_chunk.get("done", False)
    message = ollama_chunk.get("message", {})

    # Build the OpenAI streaming chunk
    choices: list[dict] = []
    usage: dict | None = None

    if done:
        # Final chunk: emit finish_reason and usage
        done_reason = ollama_chunk.get("done_reason", "stop")
        # Map Ollama done_reason to OpenAI finish_reason
        finish_reason = "length" if done_reason == "length" else (
            "tool_calls" if message.get("tool_calls") else "stop"
        )
        delta: dict = {}
        # If the final chunk has tool_calls, include them
        if message.get("tool_calls"):
            delta["tool_calls"] = _convert_ollama_tool_calls(message["tool_calls"])
        choices.append({"index": 0, "delta": delta, "finish_reason": finish_reason})

        # Usage from final NDJSON chunk
        pev = ollama_chunk.get("prompt_eval_count")
        ev = ollama_chunk.get("eval_count")
        if pev is not None or ev is not None:
            usage = {
                "prompt_tokens": int(pev or 0),
                "completion_tokens": int(ev or 0),
                "total_tokens": int(pev or 0) + int(ev or 0),
            }
    else:
        # Content chunk
        content = message.get("content", "")
        delta: dict = {}
        # Reasoning/thinking chunk. Ollama native emits message.thinking;
        # OpenAI-compatible clients expect reasoning on the delta.
        thinking = message.get("thinking", "")
        if thinking:
            delta["reasoning"] = thinking
        if content:
            delta["content"] = content
        # Tool calls chunk
        if message.get("tool_calls"):
            delta["tool_calls"] = _convert_ollama_tool_calls(message["tool_calls"])
            delta["content"] = None  # OpenAI: content is null when tool_calls present
        if not delta:
            return None  # Empty delta, skip
        choices.append({"index": 0, "delta": delta, "finish_reason": None})

    chunk: dict = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage:
        chunk["usage"] = usage

    return f"data: {json.dumps(chunk)}"


async def _proxy_bridge_stream(client_id: str, key: 'KeyState', body: bytes,
                                model: str, start: float,
                                request_id: Optional[str] = None) -> StreamingResponse:
    """Bridge stream: receive NDJSON from /api/chat, emit SSE for /v1/chat/completions client.

    This is the core of the native bridge. It re-routes the upstream request
    from the Ollama /v1 endpoint to the native /api/chat endpoint, which
    correctly reports done_reason: "length" for truncated responses. The
    NDJSON response is converted chunk-by-chunk to SSE format so the client
    (Hermes) sees a standard OpenAI-compatible stream.
    """
    api_upstream = _native_api_upstream()
    chunk_id = f"chatcmpl-bridge-{uuid.uuid4().hex[:8]}"

    async def generate():
        tokens_out = 0
        tokens_in = 0
        usage_captured = False
        bridge_reason = "stop"  # default
        final_status = 200
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                async with client_http.stream("POST", f"{api_upstream}/chat",
                                              content=body, headers={
                                                  "Authorization": f"Bearer {key.token}",
                                                  "Content-Type": "application/json",
                                              }) as resp:
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
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms,
                                  final_status, request_id=request_id, provider="ollama-cloud")
            usage_src = "usage" if usage_captured else "estimate"
            log.info(f"{client_id} -> {model} via {key.label}: bridge {tokens_in}+{tokens_out}tok {elapsed_ms}ms done={bridge_reason} ({usage_src})")

    return StreamingResponse(generate(), media_type="text/event-stream")


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
    }


@app.get("/admin/usage", dependencies=[Depends(_verify_admin)])
async def admin_usage(hours: int = 24, client: str = None, model: str = None):
    if usage_db:
        return usage_db.summary(hours, client=client, model=model)
    return []


@app.get("/admin/usage/daily", dependencies=[Depends(_verify_admin)])
async def admin_usage_daily(days: int = 30, start_date: str = None, end_date: str = None):
    if usage_db:
        return usage_db.daily_totals(days, start_date=start_date, end_date=end_date)
    return []


@app.get("/admin/usage/by-client", dependencies=[Depends(_verify_admin)])
async def admin_usage_by_client(days: int = 30, start_date: str = None, end_date: str = None):
    if usage_db:
        return usage_db.by_client(days, start_date=start_date, end_date=end_date)
    return []


@app.get("/admin/usage/by-model", dependencies=[Depends(_verify_admin)])
async def admin_usage_by_model(days: int = 30, start_date: str = None, end_date: str = None):
    if usage_db:
        return usage_db.by_model(days, start_date=start_date, end_date=end_date)
    return []


@app.get("/admin/recent-calls", dependencies=[Depends(_verify_admin)])
async def admin_recent_calls(limit: int = 100, client: str = None, model: str = None,
                              start_date: str = None, end_date: str = None):
    """Return recent individual calls (not aggregates) for the live feed."""
    if not usage_db:
        return []
    return usage_db.recent_calls(limit, start_date=start_date, end_date=end_date,
                                 client_id=client, model=model)


@app.get("/admin/totals", dependencies=[Depends(_verify_admin)])
async def admin_totals(start_date: str = None, end_date: str = None):
    """Totals across all clients/models. Optionally filter by date range."""
    if usage_db:
        return usage_db.totals(start_date=start_date, end_date=end_date)
    return {"total_calls": 0, "total_tokens_in": 0, "total_tokens_out": 0, "total_tokens": 0}


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


# ---------------------------------------------------------------------------
# Admin — Subscription (Upstream Key) Management
# ---------------------------------------------------------------------------

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
            "index": i,
        })
    return result


@app.put("/admin/keys/{key_index}", dependencies=[Depends(_verify_admin)])
async def admin_update_key(key_index: int, label: str = None, max_concurrent: int = None,
                           cycle_day: int = None):
    """Update a key's mutable fields (label, max_concurrent, cycle_day). Requires restart to persist."""
    if not manager or key_index >= len(manager.keys):
        raise HTTPException(status_code=404, detail="key not found")
    k = manager.keys[key_index]
    if label is not None:
        k.label = label
    if max_concurrent is not None:
        k.max_concurrent = max_concurrent
    if cycle_day is not None:
        k.cycle_day = cycle_day
    return {"updated": key_index, "label": k.label, "max_concurrent": k.max_concurrent, "cycle_day": k.cycle_day}


@app.put("/admin/keys/{key_index}/cookies", dependencies=[Depends(_verify_admin)])
async def admin_update_key_cookies(key_index: int, request: Request):
    """Update cookies for a specific key. Cookies are used for usage scraping from ollama.com/settings."""
    if not manager or key_index >= len(manager.keys):
        raise HTTPException(status_code=404, detail="key not found")
    body = await request.json()
    k = manager.keys[key_index]
    # Update cookies in usage_scraper (keyed by label)
    cookies = {}
    for field in ["secure_session", "aid", "cf_clearance", "stripe_mid"]:
        if field in body:
            cookies[field] = body[field]
    if usage_scraper and hasattr(usage_scraper, 'cookie_map') and cookies:
        usage_scraper.cookie_map[k.label] = cookies
    return {"updated": key_index, "label": k.label, "cookies_set": list(cookies.keys())}


@app.post("/admin/keys", dependencies=[Depends(_verify_admin)])
async def admin_add_key(request: Request):
    """Add a new upstream key. Requires config.yaml update and restart to persist."""
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
    # Also update usage_scraper if cookies provided
    cookies = body.get("cookies", {})
    if usage_scraper and cookies:
        usage_scraper.cookie_map[label] = cookies
    return {"added": label, "key_index": len(manager.keys) - 1,
            "note": "Add this key to config.yaml and restart for persistence"}


@app.delete("/admin/keys/{key_index}", dependencies=[Depends(_verify_admin)])
async def admin_delete_key(key_index: int):
    """Remove an upstream key. Requires config.yaml update and restart to persist."""
    if not manager or key_index >= len(manager.keys):
        raise HTTPException(status_code=404, detail="key not found")
    removed = manager.keys.pop(key_index)
    return {"removed": removed.label, "note": "Remove this key from config.yaml and restart for persistence"}


# ---------------------------------------------------------------------------
# Admin — Client Key Management
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SSE — Live event stream for dashboard
# ---------------------------------------------------------------------------

@app.get("/admin/events")
async def admin_events(request: Request, token: str = None):
    """SSE endpoint for live dashboard updates. Authenticates via ?token= param."""
    global admin_token
    if not admin_token:
        raise HTTPException(status_code=500, detail="admin_token not configured")
    if not token or not secrets.compare_digest(token, admin_token):
        raise HTTPException(status_code=401, detail="unauthorized")

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
                except asyncio.TimeoutError:
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
    for field in ("label", "notes", "token"):
        if field in body:
            kwargs[field] = body[field]
    for field in ("daily_token_limit", "daily_request_limit", "rpm_limit"):
        if field in body:
            kwargs[field] = body[field]  # None clears the limit
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

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LlamaHerd — Ollama Cloud Router</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --dim: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --yellow: #d29922;
  --purple: #bc8cff; --orange: #f0883e;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; font-size: 14px; }
h1 { font-size: 22px; margin-bottom: 4px; }
.brandline { color: #f2d6a2; font-size: 13px; margin-bottom: 10px; }
.subtitle { color: var(--dim); font-size: 13px; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card .label { font-size: 12px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.card .value { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
.card .value.blue { color: var(--accent); }
.card .value.green { color: var(--green); }
.card .value.purple { color: var(--purple); }
.card .value.yellow { color: var(--yellow); }
.section { margin-bottom: 28px; }
.section h2 { font-size: 16px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
.section h3 { font-size: 14px; margin: 12px 0 6px; color: var(--accent); }
.badge { font-size: 11px; background: var(--border); padding: 2px 8px; border-radius: 10px; color: var(--dim); }
.badge.live { background: #1a3a1a; color: var(--green); }
.badge.new { background: #1a3a1a; color: var(--green); }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--dim); font-weight: 500; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.5px; padding: 8px 10px; border-bottom: 1px solid var(--border); }
td { padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:hover td { background: rgba(88,166,255,0.04); }
.bar { display: inline-block; height: 8px; border-radius: 4px; min-width: 2px; }
.bar.in { background: var(--accent); }
.bar.out { background: var(--purple); }
.bars { display: flex; gap: 2px; align-items: center; }
.key-status { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.key-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 14px; min-width: 260px; flex: 1; }
.key-card .key-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.key-card .key-label { font-weight: 600; font-size: 14px; }
.key-card .key-plan { font-size: 11px; background: var(--border); padding: 2px 8px; border-radius: 10px; }
.key-card .key-row { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; }
.key-card .key-row .kdim { color: var(--dim); }
.pct-bar-wrap { width: 100%; height: 6px; background: var(--border); border-radius: 3px; margin-top: 4px; overflow: hidden; position: relative; }
.pct-bar { height: 100%; border-radius: 3px; transition: width .3s, background .3s; }
.pct-elapsed { position: absolute; top: 0; bottom: 0; width: 2px; border-left: 2px dashed rgba(255,255,255,0.8); background: none; transition: left .3s; z-index: 1; }
.status-ok { color: var(--green); }
.status-err { color: var(--red); }
.status-warn { color: var(--yellow); }
.filters { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; }
.filters select, .filters input { background: var(--surface); color: var(--text); border: 1px solid var(--border);
       border-radius: 4px; padding: 4px 8px; font-size: 13px; }
.filters label { font-size: 12px; color: var(--dim); }
#feed-table { max-height: 500px; overflow-y: auto; }
.model-info { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; padding: 8px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; font-size: 12px; color: var(--dim); flex-wrap: wrap; }
.model-info .mi-val { color: var(--text); font-weight: 600; }
.model-info .new-models { color: var(--green); }
.model-info button, .btn { background: var(--border); color: var(--text); border: none; padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.model-info button:hover, .btn:hover { background: var(--accent); color: #fff; }
.model-info button:disabled, .btn:disabled { opacity: 0.5; cursor: default; }
.btn-danger { background: #3d1214; color: var(--red); }
.btn-danger:hover { background: var(--red); color: #fff; }
.btn-sm { padding: 2px 8px; font-size: 11px; }
.date-range { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.date-range select, .date-range input { background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; font-size: 13px; }
.date-range label { font-size: 12px; color: var(--dim); }
.sse-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.sse-dot.ok { background: var(--green); }
.sse-dot.off { background: var(--red); }
.tab-bar { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
.tab { padding: 8px 16px; font-size: 13px; color: var(--dim); cursor: pointer; border-bottom: 2px solid transparent; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.6); z-index: 100; display: flex; align-items: center; justify-content: center; }
.modal { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 24px; min-width: 400px; max-width: 600px; max-height: 80vh; overflow-y: auto; }
.modal h3 { margin-bottom: 16px; }
.modal label { display: block; font-size: 12px; color: var(--dim); margin: 8px 0 2px; }
.modal input, .modal select, .modal textarea { width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px; font-size: 13px; font-family: monospace; }
.modal textarea { min-height: 60px; }
.modal .modal-actions { margin-top: 16px; display: flex; gap: 8px; justify-content: flex-end; }
.models-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; }
.model-chip { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 12px; transition: border-color .2s; }
.model-chip:hover { border-color: var(--accent); }
.model-chip .mc-name { font-weight: 600; font-size: 13px; margin-bottom: 2px; word-break: break-all; }
.model-chip .mc-meta { color: var(--dim); font-size: 11px; }
.model-chip .mc-ctx { color: var(--purple); }
.model-chip .mc-keys { color: var(--green); }
.model-chip .mc-usage { margin-top: 4px; }
.model-chip .mc-usage .bar { height: 4px; }
.search-input { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; font-size: 13px; width: 200px; }
.key-mgmt { margin-top: 12px; }
.key-mgmt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 10px; }
.key-mgmt-card .km-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.key-mgmt-card .km-label { font-weight: 600; }
.key-mgmt-card .km-row { font-size: 12px; color: var(--dim); padding: 2px 0; display: flex; justify-content: space-between; }
.key-mgmt-card .km-row span:last-child { color: var(--text); }
.cookie-bad { color: var(--red); }
.cookie-ok { color: var(--green); }
/* In-flight panel */
@keyframes inflight-pulse {
  0% { box-shadow: 0 0 0 0 rgba(88,166,255,0.45); border-color: var(--accent); }
  70% { box-shadow: 0 0 0 6px rgba(88,166,255,0); border-color: var(--accent); }
  100% { box-shadow: 0 0 0 0 rgba(88,166,255,0); border-color: var(--accent); }
}
@keyframes flash-green { 0% { background: rgba(63,185,80,0.35); } 100% { background: transparent; } }
@keyframes flash-red   { 0% { background: rgba(248,81,73,0.35); } 100% { background: transparent; } }
.inflight-item { border: 1px solid var(--border); border-radius: 6px; margin-bottom: 6px;
  background: var(--surface); animation: inflight-pulse 1.6s ease-out infinite; overflow: hidden; }
.inflight-item.ending-ok  { animation: flash-green 0.4s ease-out forwards; }
.inflight-item.ending-err { animation: flash-red 0.4s ease-out forwards; }
.inflight-row { display: grid;
  grid-template-columns: minmax(140px,1fr) minmax(200px,2fr) 110px 110px 90px 18px;
  gap: 10px; align-items: center; padding: 8px 12px; font-size: 13px; cursor: pointer;
  user-select: none; }
.inflight-row .if-client { color: var(--text); }
.inflight-row .if-model  { color: var(--accent); font-family: monospace; }
.inflight-row .if-target { color: var(--dim); font-size: 12px; overflow: hidden; text-overflow: ellipsis; }
.inflight-row .if-tokens { color: var(--purple); font-size: 12px; font-variant-numeric: tabular-nums; }
.inflight-row .if-tokens .if-tin  { color: var(--green); }
.inflight-row .if-tokens .if-tout { color: var(--purple); }
.inflight-row .if-elapsed { font-variant-numeric: tabular-nums; color: var(--yellow); text-align: right; }
.inflight-row .if-caret { color: var(--dim); transition: transform .2s ease; text-align: center; font-size: 10px; }
.inflight-item.expanded .if-caret { transform: rotate(90deg); }
.inflight-details { max-height: 0; overflow: hidden; transition: max-height 0.25s ease-out;
  padding: 0 12px; border-top: 0 solid var(--border); font-size: 12px; }
.inflight-item.expanded .inflight-details { max-height: 380px; padding: 8px 12px; border-top: 1px solid var(--border); overflow-y: auto; }
.inflight-details .ifd-row { display: grid; grid-template-columns: 130px 1fr; gap: 8px; padding: 2px 0; }
.inflight-details .ifd-row .ifd-k { color: var(--dim); }
.inflight-details .ifd-row .ifd-v { color: var(--text); font-family: monospace; word-break: break-all; }
.inflight-details .ifd-headers { margin-top: 4px; padding-top: 4px; border-top: 1px dashed var(--border); }
.inflight-details .ifd-headers .ifd-k { font-size: 11px; }
.inflight-details .ifd-headers .ifd-v { font-size: 11px; color: var(--dim); }
.provider-badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; margin-left: 6px; vertical-align: middle; }
.provider-badge.oc { background: #173a23; color: #6fdc8c; }
.provider-badge.nv { background: #14274d; color: #76b6ff; }
.provider-badge.both { background: #2a2a4d; color: #c8b6ff; }
.provider-badge.unknown { background: var(--border); color: var(--dim); }
.fb-map-toggle { font-size: 11px; color: var(--accent); cursor: pointer; background: none; border: none; padding: 0 4px; }
.fb-map-panel { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-top: 6px; max-height: 240px; overflow-y: auto; font-size: 12px; }
.fb-map-panel table { width: 100%; }
.fb-map-panel td { padding: 3px 8px; border: none; }
.fb-map-panel td.fb-pri { color: var(--dim); font-size: 11px; }
.fb-map-panel td.fb-actions { width: 28px; text-align: right; }
.fb-map-panel button.fb-rm { background: transparent; color: var(--red); border: none; cursor: pointer; font-size: 14px; padding: 0 6px; }
.fb-map-panel button.fb-rm:hover { background: rgba(248,81,73,0.15); border-radius: 3px; }
.fb-map-panel .fb-add-row { display: flex; gap: 6px; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px dashed var(--border); flex-wrap: wrap; }
.fb-map-panel .fb-add-row input { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 4px 6px; font-size: 12px; font-family: monospace; min-width: 140px; flex: 1; }
#inflight-empty { color: var(--dim); font-size: 12px; padding: 8px 12px; }
/* Model Catalog */
.catalog-section { margin-top: 24px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
.catalog-header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; cursor: pointer; user-select: none; flex-wrap: wrap; }
.catalog-header:hover { background: rgba(88,166,255,0.05); }
.catalog-header .ch-title { font-weight: 600; font-size: 14px; flex: 1; }
.catalog-header .ch-caret { color: var(--dim); font-size: 11px; transition: transform .2s ease; }
.catalog-section.open .ch-caret { transform: rotate(90deg); }
.catalog-body { display: none; padding: 0 14px 14px; }
.catalog-section.open .catalog-body { display: block; }
.catalog-controls { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
.catalog-controls .search-input { flex: 1; min-width: 220px; }
.catalog-org { margin-bottom: 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); }
.catalog-org-header { display: flex; align-items: center; gap: 8px; padding: 6px 10px; cursor: pointer; user-select: none; font-size: 12px; }
.catalog-org-header:hover { background: rgba(88,166,255,0.04); }
.catalog-org-header .co-name { font-weight: 600; color: var(--accent); }
.catalog-org-header .co-count { color: var(--dim); font-size: 11px; }
.catalog-org-header .co-caret { color: var(--dim); font-size: 10px; transition: transform .2s ease; margin-left: auto; }
.catalog-org.open .co-caret { transform: rotate(90deg); }
.catalog-org-body { display: none; padding: 0 8px 8px; }
.catalog-org.open .catalog-org-body { display: block; }
.catalog-row { display: grid; grid-template-columns: 2fr 70px 80px 1.5fr 100px;
  gap: 8px; align-items: center; padding: 6px 8px; border-top: 1px solid var(--border); font-size: 12px; overflow: hidden; }
.catalog-row:first-child { border-top: none; }
.catalog-row .cr-name { font-family: monospace; word-break: break-all; }
.catalog-row .cr-name a { color: var(--text); text-decoration: none; }
.catalog-row .cr-name a:hover { color: var(--accent); text-decoration: underline; }
.catalog-row .cr-meta { color: var(--dim); font-variant-numeric: tabular-nums; }
.catalog-row .cr-mapped { color: var(--green); font-size: 11px; }
.catalog-row .cr-mapped .cr-alias { color: var(--text); font-family: monospace; }
.catalog-row .cr-action { text-align: right; }
.catalog-empty { color: var(--dim); font-size: 12px; padding: 8px 0; }
</style>
</head>
<body>
<h1>🦙 LlamaHerd</h1>
<p class="brandline">One endpoint. Many llamas. Smarter routing.</p>
<p class="subtitle">
  <span id="upstream-url"></span>
  <span class="sse-dot" id="sse-dot"></span>
  <span id="sse-label" style="font-size:11px">connecting...</span>
  <span class="date-range">
    <label>Period:</label>
    <select id="period-select">
      <option value="today" selected>Today</option>
      <option value="yesterday">Yesterday</option>
      <option value="7d">Last 7 days</option>
      <option value="this_week">This Week</option>
      <option value="this_month">This Month</option>
      <option value="last_month">Last Month</option>
      <option value="custom">Custom</option>
    </select>
    <span id="custom-dates" style="display:none">
      <input type="date" id="date-start">
      <input type="date" id="date-end">
    </span>
  </span>
</p>

<div class="model-info" id="model-info">
  <span>Models: <span class="mi-val" id="mi-count">-</span></span>
  <span>Refreshed: <span class="mi-val" id="mi-refresh">-</span></span>
  <span id="mi-new" class="new-models" style="display:none"></span>
  <button id="btn-refresh-models">🔄 Refresh</button>
  <span id="fallback-control" style="display:none; margin-left:auto; align-items:center; gap:6px;">
    <span class="badge" id="fb-provider-badge">fallback</span>
    <span style="font-size:11px; color:var(--dim)" id="fb-model-count">0 mapped / 0 discovered</span>
    <button type="button" class="fb-map-toggle" id="fb-map-toggle">show map</button>
    <label style="font-size:11px; color:var(--dim)">Priority:</label>
    <select id="fb-priority" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px;font-size:12px">
      <option value="after">after (Ollama first)</option>
      <option value="before">before (Fallback first)</option>
      <option value="only">only (Fallback only)</option>
    </select>
  </span>
</div>
<div id="fb-map-panel" class="fb-map-panel" style="display:none"></div>

<div class="section" id="inflight-section">
  <h2>In-Flight Requests <span class="badge live" id="inflight-count">0</span></h2>
  <div id="inflight-list"><div id="inflight-empty">No requests in flight</div></div>
</div>

<div class="tab-bar">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="models">Models</div>
  <div class="tab" data-tab="subs">Subscriptions</div>
  <div class="tab" data-tab="quota">Quota Cost</div>
</div>

<div class="tab-panel active" id="panel-overview">
  <div id="key-status" class="key-status"></div>
  <div id="totals" class="grid"></div>

  <div class="section">
    <h2>Per-Client Totals <span class="badge" id="badge-client">this month</span></h2>
    <div style="overflow-x:auto"><table id="client-table"><thead><tr>
      <th>Client</th><th>Calls</th><th>Tokens In</th><th>Tokens Out</th><th>Total Tokens</th><th>Breakdown</th>
    </tr></thead><tbody></tbody></table></div>
  </div>

  <div class="section">
    <h2>Per-Model Totals <span class="badge" id="badge-model">this month</span></h2>
    <div style="overflow-x:auto"><table id="model-table"><thead><tr>
      <th>Model</th><th>Calls</th><th>Tokens In</th><th>Tokens Out</th><th>Total Tokens</th><th>Avg Latency</th>
    </tr></thead><tbody></tbody></table></div>
  </div>

  <div class="section">
    <h2>Daily Breakdown <span class="badge" id="badge-daily">this month</span></h2>
    <div style="overflow-x:auto"><table id="daily-table"><thead><tr>
      <th>Day</th><th>Calls</th><th>Tokens In</th><th>Tokens Out</th><th>Total Tokens</th>
    </tr></thead><tbody></tbody></table></div>
  </div>

  <div class="section">
    <h2>Live Call Feed <span class="badge live" id="feed-count">0</span></h2>
    <div class="filters">
      <label>Client:</label><select id="filter-client"><option value="">All</option></select>
      <label>Model:</label><select id="filter-model"><option value="">All</option></select>
      <label>Limit:</label><select id="feed-limit">
        <option value="50">50</option><option value="100" selected>100</option><option value="250">250</option><option value="500">500</option>
      </select>
    </div>
    <div id="feed-table"><table id="feed-tbl"><thead><tr>
      <th>Time</th><th>Client</th><th>Model</th><th>Key</th><th>In</th><th>Out</th><th>Latency</th><th>Status</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
</div>

<div class="tab-panel" id="panel-models">
  <div style="margin-bottom:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap">
    <input type="text" class="search-input" id="model-search" placeholder="Search models...">
    <select id="model-sort" style="background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 8px;font-size:13px">
      <option value="name">Sort: Name</option>
      <option value="ctx">Sort: Context Length</option>
      <option value="keys">Sort: Availability</option>
      <option value="updated">Sort: Updated</option>
      <option value="params">Sort: Parameters</option>
      <option value="family">Sort: Family</option>
    </select>
  </div>
  <div class="models-grid" id="models-grid"></div>

  <div class="catalog-section" id="catalog-section">
    <div class="catalog-header" id="catalog-header">
      <span class="ch-caret">▶</span>
      <span class="ch-title">NVIDIA Build Catalog</span>
      <span class="badge" id="catalog-count">0</span>
      <span style="font-size:11px; color:var(--dim)" id="catalog-summary"></span>
    </div>
    <div class="catalog-body">
      <div class="catalog-controls">
        <input type="text" class="search-input" id="catalog-search" placeholder="Filter by name or org...">
        <button class="btn btn-sm" id="catalog-refresh">🔄 Refresh</button>
        <button class="btn btn-sm" id="catalog-add-mapping">+ Add Mapping</button>
      </div>
      <div id="catalog-list"><div class="catalog-empty">Open this panel to load the catalog.</div></div>
    </div>
  </div>
</div>

<div class="tab-panel" id="panel-subs">
  <div style="margin-bottom:12px; display:flex; gap:8px; align-items:center">
    <button class="btn" id="btn-add-key">+ Add Subscription</button>
    <span style="font-size:11px;color:var(--dim)">Changes take effect immediately but require config.yaml update + restart to persist</span>
  </div>
  <div id="subs-list"></div>
  <h3 style="margin-top:24px;border-top:1px solid var(--border);padding-top:16px">Client Keys</h3>
  <div style="margin-bottom:12px; display:flex; gap:8px; align-items:center">
    <button class="btn" id="btn-add-client">+ Add Client</button>
    <span style="font-size:11px;color:var(--dim)">Client keys attribute usage and enforce rate limits</span>
  </div>
  <div id="client-list"></div>
</div>

<div class="tab-panel" id="panel-quota">
  <div style="margin-bottom:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap">
    <h2 style="margin:0">Quota Cost Algebra</h2>
    <span class="badge" id="quota-badge">0</span>
    <button class="btn btn-sm" id="quota-refresh">🔄 Refresh</button>
    <span style="font-size:11px;color:var(--dim)" id="quota-subtitle">Separate input/output coefficients solved via least-squares from usage deltas.</span>
  </div>
  <div style="margin-bottom:8px; font-size:12px; color:var(--dim)" id="quota-solve-status"></div>
  <div style="overflow-x:auto"><table id="quota-table"><thead><tr>
    <th>Model</th><th>Confidence</th><th>Intervals</th><th>Session %</th><th>Tokens In</th><th>Tokens Out</th><th>% of Quota</th><th>Cost / 1K In</th><th>Cost / 1K Out</th><th>In/Out Ratio</th>
  </tr></thead><tbody></tbody></table></div>
  <div id="quota-note" style="font-size:11px;color:var(--dim);margin-top:8px"></div>
</div>

<div id="modal-root"></div>

<script>
const API = (function() { const p = window.location.pathname; return p.includes('/ocp') ? '/ocp' : ''; })();
const ADMIN_TOKEN = (function() {
  const p = new URLSearchParams(window.location.search);
  const t = p.get('token');
  if (t) { localStorage.setItem('ocp_admin_token', t); return t; }
  return localStorage.getItem('ocp_admin_token') || '';
})();
if (!ADMIN_TOKEN) { document.body.innerHTML = '<div style="color:var(--red);text-align:center;padding:3em"><h2>Authentication Required</h2><p>Provide admin token: <code>/dashboard?token=YOUR_TOKEN</code></p></div>'; }

function fmt(n) { if (n >= 1e6) return (n/1e6).toFixed(2)+'M'; if (n >= 1e3) return (n/1e3).toFixed(1)+'K'; return String(n); }
function fmtTs(ts) { const d = new Date(ts*1000); return d.toLocaleDateString('en-CA')+' '+d.toLocaleTimeString('en-GB'); }
function fmtCtx(n) { if (!n) return '-'; if (n >= 1048576) return (n/1048576).toFixed(0)+'M'; if (n >= 1024) return (n/1024).toFixed(0)+'K'; return n; }
function relTime(ts) { if (!ts) return '-'; const diff=(Date.now()/1000)-ts; if(diff<60)return Math.round(diff)+'s ago'; if(diff<3600)return Math.round(diff/60)+'m ago'; if(diff<86400)return Math.round(diff/3600)+'h ago'; return Math.round(diff/86400)+'d ago'; }

function pctBarWithElapsed(pct, elapsedPct, defaultColor) {
  if (pct == null || pct < 0) return '<div class="pct-bar-wrap"><div class="pct-bar" style="width:0%;background:var(--border)"></div></div>';
  let c = defaultColor;
  if (elapsedPct > 0 && pct > 0) { c = pct <= elapsedPct ? 'var(--green)' : pct < elapsedPct*2 ? 'var(--yellow)' : 'var(--red)'; }
  const el = (elapsedPct > 0 && elapsedPct < 100) ? `<div class="pct-elapsed" style="left:${Math.min(elapsedPct,98)}%"></div>` : '';
  return `<div class="pct-bar-wrap"><div class="pct-bar" style="width:${Math.min(pct,100)}%;background:${c}"></div>${el}</div>`;
}
function pctBar(pct, color) { if (pct == null) return ''; return `<div class="pct-bar-wrap"><div class="pct-bar" style="width:${Math.min(pct,100)}%;background:${color}"></div></div>`; }

async function loadJSON(url) { const sep = url.includes('?')?'&':'?'; const r = await fetch(API+url+sep+'token='+encodeURIComponent(ADMIN_TOKEN)); return r.json(); }
async function postJSON(url, body, method) { return fetch(API+url+'?token='+encodeURIComponent(ADMIN_TOKEN), { method: method||'POST', headers:{'Content-Type':'application/json'}, body: body?JSON.stringify(body):undefined }).then(r=>r.json()); }

// --- Date Range ---
function getDateRange() {
  const sel = document.getElementById('period-select').value;
  const today = new Date(); const iso = d => d.toISOString().slice(0,10);
  const monday = d => { const day=d.getDay(); const diff=d.getDate()-day+(day===0?-6:1); return new Date(d.setDate(diff)); };
  switch(sel) {
    case 'today': return {start:iso(today), end:iso(today), label:'Today'};
    case 'yesterday': { const y=new Date(today); y.setDate(y.getDate()-1); return {start:iso(y),end:iso(y),label:'Yesterday'}; }
    case '7d': { const s=new Date(today); s.setDate(s.getDate()-6); return {start:iso(s),end:iso(today),label:'7d'}; }
    case 'this_week': { const m=monday(new Date(today)); return {start:iso(m),end:iso(today),label:'This week'}; }
    case 'this_month': return {start:iso(new Date(today.getFullYear(),today.getMonth(),1)),end:iso(today),label:'This month'};
    case 'last_month': { const lm=new Date(today.getFullYear(),today.getMonth()-1,1); const le=new Date(today.getFullYear(),today.getMonth(),0); return {start:iso(lm),end:iso(le),label:'Last month'}; }
    case 'custom': return {start:document.getElementById('date-start').value,end:document.getElementById('date-end').value,label:'Custom'};
    default: return {start:iso(new Date(today.getFullYear(),today.getMonth(),1)),end:iso(today),label:'This month'};
  }
}
function updateBadges(l) { document.getElementById('badge-client').textContent=l; document.getElementById('badge-model').textContent=l; document.getElementById('badge-daily').textContent=l; }

// --- Tabs ---
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', function() {
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(x=>x.classList.remove('active'));
  this.classList.add('active');
  document.getElementById('panel-'+this.dataset.tab).classList.add('active');
  if (this.dataset.tab === 'models') loadModelsPanel();
  if (this.dataset.tab === 'subs') { loadSubsPanel(); loadClients(); }
}));

// --- SSE ---
let eventSource = null, reconnectDelay = 1000, feedCalls = [];

function connectSSE() {
  const url = `${window.location.protocol}//${window.location.host}${API}/admin/events?token=${encodeURIComponent(ADMIN_TOKEN)}`;
  eventSource = new EventSource(url);
  eventSource.onopen = () => { reconnectDelay=1000; document.getElementById('sse-dot').className='sse-dot ok'; document.getElementById('sse-label').textContent='live'; };
  eventSource.onerror = () => { document.getElementById('sse-dot').className='sse-dot off'; document.getElementById('sse-label').textContent='reconnecting...'; eventSource.close(); setTimeout(connectSSE,reconnectDelay); reconnectDelay=Math.min(reconnectDelay*2,30000); };
  eventSource.addEventListener('status', e => { const d=JSON.parse(e.data); renderKeyStatus(d.keys||[]); document.getElementById('upstream-url').textContent=d.upstream||''; });
  eventSource.addEventListener('models', e => updateModelInfo(JSON.parse(e.data)));
  eventSource.onmessage = e => {
    try {
      const m=JSON.parse(e.data);
      if(m.type==='call') { if(callInCurrentRange(m.data)) schedulePeriodRefresh(); }
      else if(m.type==='request_start') { addInFlight(m.data); }
      else if(m.type==='request_end') { addCallToFeed(m.data); removeInFlight(m.data); }
      else if(m.type==='status') { const d=m.data||{}; renderKeyStatus(d.keys||[]); if(d.upstream) document.getElementById('upstream-url').textContent=d.upstream; }
      else if(m.type==='models') updateModelInfo(m.data||{});
      else if(m.type==='fallback_priority') { const sel=document.getElementById('fb-priority'); if(m.data && m.data.priority) sel.value = m.data.priority; }
      else if(m.type==='fallback_map_update') { loadFallbackStatus(); if (catalogLoaded) loadCatalog(true); }
    } catch(err){}
  };
}

// --- In-Flight Panel ---
const inFlight = {};  // request_id -> { client_id, model, target_key, target_provider, started_at }
function fmtElapsed(ms) {
  if (ms < 1000) return ms + 'ms';
  if (ms < 60000) return (ms/1000).toFixed(1) + 's';
  const m = Math.floor(ms/60000), s = Math.floor((ms%60000)/1000);
  return m + 'm' + s + 's';
}
function providerBadge(p) {
  if (!p) return '<span class="provider-badge unknown" title="unknown provider">?</span>';
  // Allow combined "ollama-cloud,nvidia-build" tagging from /admin/models or end events.
  const parts = String(p).split(',').map(s => s.trim()).filter(Boolean);
  const hasOC = parts.some(x => x === 'ollama-cloud');
  const hasFB = parts.some(x => x && x !== 'ollama-cloud');
  if (hasOC && hasFB) return `<span class="provider-badge both" title="${p}">OC+NV</span>`;
  if (hasOC) return `<span class="provider-badge oc" title="ollama-cloud">OC</span>`;
  return `<span class="provider-badge nv" title="${p}">NV</span>`;
}
const expandedInFlight = new Set();  // request_ids that are currently expanded
function escAttr(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;'); }
function escHtml(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function renderInFlightDetails(r) {
  const startedIso = r.started_at ? new Date(r.started_at * 1000).toISOString() : '-';
  const headers = r.headers || {};
  const headerRows = Object.keys(headers).sort().map(k =>
    `<div class="ifd-row"><span class="ifd-k">${escHtml(k)}</span><span class="ifd-v">${escHtml(headers[k])}</span></div>`
  ).join('');
  return `<div class="inflight-details">
    <div class="ifd-row"><span class="ifd-k">request_id</span><span class="ifd-v">${escHtml(r.request_id)}</span></div>
    <div class="ifd-row"><span class="ifd-k">target key</span><span class="ifd-v">${escHtml(r.target_key || '-')}</span></div>
    <div class="ifd-row"><span class="ifd-k">provider</span><span class="ifd-v">${escHtml(r.target_provider || '-')}</span></div>
    <div class="ifd-row"><span class="ifd-k">model</span><span class="ifd-v">${escHtml(r.model || '-')}</span></div>
    <div class="ifd-row"><span class="ifd-k">client</span><span class="ifd-v">${escHtml(r.client_id || '-')}</span></div>
    <div class="ifd-row"><span class="ifd-k">started_at</span><span class="ifd-v">${escHtml(startedIso)}</span></div>
    ${r.path ? `<div class="ifd-row"><span class="ifd-k">path</span><span class="ifd-v">${escHtml(r.path)}</span></div>` : ''}
    ${headerRows ? `<div class="ifd-headers">${headerRows}</div>` : '<div class="ifd-row"><span class="ifd-k">headers</span><span class="ifd-v" style="color:var(--dim)">(none captured)</span></div>'}
  </div>`;
}
function renderInFlight() {
  const list = document.getElementById('inflight-list');
  const ids = Object.keys(inFlight);
  document.getElementById('inflight-count').textContent = ids.length;
  if (!ids.length) { list.innerHTML = '<div id="inflight-empty">No requests in flight</div>'; return; }
  const now = Date.now() / 1000;
  list.innerHTML = ids.map(id => {
    const r = inFlight[id];
    const elapsed = Math.max(0, Math.round((now - r.started_at) * 1000));
    const expanded = expandedInFlight.has(id) ? ' expanded' : '';
    const ending = r._ending ? ' ending-' + (r._ending === 'ok' ? 'ok' : 'err') : '';
    const tin = r.tokens_in || 0, tout = r.tokens_out || 0;
    return `<div class="inflight-item${expanded}${ending}" id="if-${escAttr(id)}" data-rid="${escAttr(id)}">
      <div class="inflight-row" onclick="toggleInFlight('${escAttr(id)}')">
        <div class="if-client">${escHtml(r.client_id)}</div>
        <div><span class="if-model">${escHtml(r.model)}</span>${providerBadge(r.target_provider)}</div>
        <div class="if-target" title="${escAttr(r.target_key || '')}">${escHtml(r.target_key || '-')}</div>
        <div class="if-tokens"><span class="if-tin" data-tin>${fmt(tin)}</span> in / <span class="if-tout" data-tout>${fmt(tout)}</span> out</div>
        <div class="if-elapsed" data-started="${r.started_at}">${fmtElapsed(elapsed)}</div>
        <div class="if-caret">▶</div>
      </div>
      ${renderInFlightDetails(r)}
    </div>`;
  }).join('');
}
function toggleInFlight(id) {
  if (expandedInFlight.has(id)) expandedInFlight.delete(id); else expandedInFlight.add(id);
  const item = document.getElementById('if-' + id);
  if (item) item.classList.toggle('expanded');
}
function addInFlight(d) {
  if (!d || !d.request_id) return;
  inFlight[d.request_id] = d;
  renderInFlight();
}
function removeInFlight(d) {
  if (!d || !d.request_id) return;
  const cur = inFlight[d.request_id];
  if (!cur) return;
  cur._ending = (d.status >= 200 && d.status < 300) ? 'ok' : 'err';
  // Reflect final tokens on close
  if (d.tokens_in != null) cur.tokens_in = d.tokens_in;
  if (d.tokens_out != null) cur.tokens_out = d.tokens_out;
  const item = document.getElementById('if-' + d.request_id);
  if (item) item.classList.add('ending-' + cur._ending);
  setTimeout(() => { delete inFlight[d.request_id]; expandedInFlight.delete(d.request_id); renderInFlight(); }, 350);
}
// Tick elapsed counters every 250ms without re-rendering rows.
setInterval(() => {
  const now = Date.now() / 1000;
  document.querySelectorAll('.if-elapsed').forEach(el => {
    const started = parseFloat(el.dataset.started);
    if (!started) return;
    const ms = Math.max(0, Math.round((now - started) * 1000));
    el.textContent = fmtElapsed(ms);
  });
}, 250);
// Poll /admin/in-flight while there are active requests so token counters update.
setInterval(async () => {
  if (!Object.keys(inFlight).length) return;
  try {
    const r = await loadJSON('/admin/in-flight');
    (r.in_flight || []).forEach(e => {
      const cur = inFlight[e.request_id];
      if (!cur) return;
      if (e.tokens_in != null && e.tokens_in > (cur.tokens_in || 0)) {
        cur.tokens_in = e.tokens_in;
        const tin = document.querySelector('#if-' + e.request_id + ' [data-tin]');
        if (tin) tin.textContent = fmt(e.tokens_in);
      }
      if (e.tokens_out != null && e.tokens_out > (cur.tokens_out || 0)) {
        cur.tokens_out = e.tokens_out;
        const tout = document.querySelector('#if-' + e.request_id + ' [data-tout]');
        if (tout) tout.textContent = fmt(e.tokens_out);
      }
      if (e.headers && !cur.headers) cur.headers = e.headers;
      if (e.path && !cur.path) cur.path = e.path;
    });
  } catch (e) { /* ignore */ }
}, 2000);

function callInCurrentRange(c) {
  const r=getDateRange(); if(!r.start||!r.end||!c.ts) return false;
  const day=new Date(c.ts*1000).toISOString().slice(0,10);
  return day>=r.start && day<=r.end;
}
let periodRefreshTimer=null;
function schedulePeriodRefresh() {
  if(periodRefreshTimer) return;
  periodRefreshTimer=setTimeout(async()=>{ periodRefreshTimer=null; await fetchPeriodData({preserveFeed:true}); }, 1000);
}

// --- Model Info Bar ---
let modelState = {count:0,last_refresh:0,new_models:[]};
function updateModelInfo(d) { modelState=d; document.getElementById('mi-count').textContent=d.count; document.getElementById('mi-refresh').textContent=relTime(d.last_refresh); const n=document.getElementById('mi-new'); if(d.new_models&&d.new_models.length){n.textContent='✨ New: '+d.new_models.join(', ');n.style.display='';}else{n.style.display='none';} }
document.getElementById('btn-refresh-models').addEventListener('click', async function(){ this.disabled=true;this.textContent='Refreshing...'; try{await postJSON('/admin/refresh');}catch(e){} this.disabled=false;this.textContent='🔄 Refresh'; });
setInterval(()=>{ if(modelState.last_refresh) document.getElementById('mi-refresh').textContent=relTime(modelState.last_refresh); },30000);

// --- Fallback Priority Control ---
let fbState = null;
async function loadFallbackStatus() {
  try {
    const fb = await loadJSON('/admin/fallback');
    const ctl = document.getElementById('fallback-control');
    const panel = document.getElementById('fb-map-panel');
    if (!fb || !fb.enabled) { ctl.style.display = 'none'; panel.style.display = 'none'; return; }
    fbState = fb;
    ctl.style.display = 'inline-flex';
    document.getElementById('fb-provider-badge').textContent = fb.provider;
    const mapped = (fb.model_map || []).length;
    const discovered = fb.discovered_count || 0;
    document.getElementById('fb-model-count').textContent = `${mapped} mapped / ${discovered} discovered`;
    const sel = document.getElementById('fb-priority');
    sel.value = fb.priority;
    renderFallbackMap();
  } catch (e) { /* fallback not configured */ }
}
function renderFallbackMap() {
  if (!fbState) return;
  const panel = document.getElementById('fb-map-panel');
  const addRow = `<div class="fb-add-row">
    <input id="fb-add-ollama" placeholder="ollama name (e.g. qwen3-coder)">
    <input id="fb-add-nvidia" placeholder="nvidia model (e.g. qwen/qwen3-coder-480b-a35b-instruct)">
    <button class="btn btn-sm" onclick="fbMapAddInline()">+ Add</button>
  </div>`;
  const rows = (fbState.model_map || []).map(a => {
    const nvidiaName = a.nvidia_model || a.fallback_model || '-';
    return `<tr>
      <td><span class="if-model">${escHtml(a.id)}</span></td>
      <td>→</td>
      <td>${escHtml(nvidiaName)}</td>
      <td class="fb-pri">${escHtml(a.priority || '')}</td>
      <td class="fb-actions"><button class="fb-rm" title="Remove mapping" onclick="fbMapRemove('${escAttr(a.id)}')">×</button></td>
    </tr>`;
  }).join('');
  panel.innerHTML = addRow + (rows
    ? `<table>${rows}</table>`
    : '<div style="color:var(--dim)">No model_map entries configured.</div>');
}
async function fbMapAddInline() {
  const ollama = (document.getElementById('fb-add-ollama').value || '').trim();
  const nvidia = (document.getElementById('fb-add-nvidia').value || '').trim();
  if (!ollama || !nvidia) { alert('Both ollama and nvidia names are required'); return; }
  try {
    await postJSON('/admin/fallback-map', { ollama_name: ollama, nvidia_name: nvidia });
    await loadFallbackStatus();
    if (catalogLoaded) await loadCatalog(true);
  } catch (e) { alert('Failed to add mapping: ' + e.message); }
}
document.getElementById('fb-map-toggle').addEventListener('click', function() {
  const panel = document.getElementById('fb-map-panel');
  const showing = panel.style.display !== 'none';
  panel.style.display = showing ? 'none' : 'block';
  this.textContent = showing ? 'show map' : 'hide map';
});
document.getElementById('fb-priority').addEventListener('change', async function(){
  const requested = this.value;
  try {
    const r = await postJSON('/admin/fallback-priority', { priority: requested });
    if (r && r.priority) this.value = r.priority;
  } catch (e) { alert('Failed to set priority: ' + e.message); }
});

// --- Key Status ---
function renderKeyStatus(keys) {
  document.getElementById('key-status').innerHTML = (keys||[]).map(k => {
    const slotPct=Math.round((k.in_flight/k.max_concurrent)*100), periodPct=Math.round((1-k.period_remaining_pct/100)*100);
    const sPct=k.session_usage_pct||0, wPct=k.weekly_usage_pct||0;
    const sEl=(k.session_elapsed_pct!=null&&k.session_elapsed_pct>=0)?k.session_elapsed_pct:-1;
    const wEl=(k.weekly_elapsed_pct!=null&&k.weekly_elapsed_pct>=0)?k.weekly_elapsed_pct:-1;
    const cls=k.exhausted?'status-err':k.suspended?'status-warn':'status-ok';
    return `<div class="key-card">
      <div class="key-header"><span class="key-label ${cls}">${k.label}</span><span class="key-plan">${k.plan||'?'}</span></div>
      <div class="key-row"><span class="kdim">Slots</span><span>${k.in_flight}/${k.max_concurrent}</span></div>${pctBar(slotPct,'var(--accent)')}
      <div class="key-row"><span class="kdim">Session</span><span>${sPct<0?'?':sPct.toFixed(1)}%${sEl>=0?' ('+sEl.toFixed(0)+'% elapsed)':''}</span></div>${pctBarWithElapsed(sPct,sEl,'var(--yellow)')}
      <div class="key-row"><span class="kdim">Weekly</span><span>${wPct<0?'?':wPct.toFixed(1)}%${wEl>=0?' ('+wEl.toFixed(0)+'% elapsed)':''}</span></div>${pctBarWithElapsed(wPct,wEl,'var(--purple)')}
      <div class="key-row"><span class="kdim">Billing</span><span>${k.period_remaining_pct?.toFixed(0)}% left</span></div>${pctBar(periodPct,'var(--green)')}
      <div class="key-row"><span class="kdim">Requests</span><span>${k.total_requests}</span></div>
      <div class="key-row"><span class="kdim">429s</span><span>${k.total_429s}</span></div>
    </div>`;
  }).join('');
}

// --- Call Feed ---
function addCallToFeed(c) { const cf=document.getElementById('filter-client').value,mf=document.getElementById('filter-model').value; let ok=true; if(cf&&c.client_id!==cf)ok=false; if(mf&&c.model!==mf)ok=false; if(ok){feedCalls.unshift(c); const lim=parseInt(document.getElementById('feed-limit').value); if(feedCalls.length>lim)feedCalls.length=lim; renderFeed(feedCalls);} }
function renderFeed(calls) { document.getElementById('feed-count').textContent=calls.length; document.querySelector('#feed-tbl tbody').innerHTML=calls.map(c=>{const sc=c.status===200?'status-ok':c.status>=400?'status-err':'status-warn'; const pb=c.provider?providerBadge(c.provider):''; return `<tr><td>${fmtTs(c.ts)}</td><td>${c.client_id}</td><td>${c.model}${pb}</td><td>${(c.upstream_key||'').slice(0,8)}</td><td>${fmt(c.tokens_in)}</td><td>${fmt(c.tokens_out)}</td><td>${c.latency_ms}ms</td><td class="${sc}">${c.status}</td></tr>`;}).join(''); }
let prevClients=[],prevModels=[];
function populateFilters(clients,models) { const sc=document.getElementById('filter-client'),sm=document.getElementById('filter-model'); const cc=sc.value,cm=sm.value; const cl=(clients||[]).map(c=>c.id||c),ml=(models||[]).map(m=>m.model||m); if(JSON.stringify(cl)!==JSON.stringify(prevClients)){sc.innerHTML='<option value="">All</option>'+cl.map(c=>`<option value="${c}">${c}</option>`).join('');sc.value=cc;prevClients=cl;} if(JSON.stringify(ml)!==JSON.stringify(prevModels)){sm.innerHTML='<option value="">All</option>'+ml.map(m=>`<option value="${m}">${m}</option>`).join('');sm.value=cm;prevModels=ml;} }

// --- Models Panel ---
let allModels = [], modelUsage = {};
async function loadModelsPanel() {
  const [models, usage] = await Promise.all([loadJSON('/admin/models'), loadJSON('/admin/usage/by-model?days=7')]);
  allModels = models.models || [];
  modelUsage = {};
  (usage||[]).forEach(u => { modelUsage[u.model] = u; });
  renderModelsGrid();
}
function renderModelsGrid() {
  const search = document.getElementById('model-search').value.toLowerCase();
  const sort = document.getElementById('model-sort').value;
  let filtered = allModels.filter(m => !search || m.id.toLowerCase().includes(search));
  if (sort === 'ctx') filtered.sort((a,b) => (b.context_length||0)-(a.context_length||0));
  else if (sort === 'keys') filtered.sort((a,b) => (b.available_on||0)-(a.available_on||0));
  else if (sort === 'updated') filtered.sort((a,b) => Date.parse(b.modified_at||0)-Date.parse(a.modified_at||0));
  else if (sort === 'params') filtered.sort((a,b) => (b.parameter_count||0)-(a.parameter_count||0));
  else if (sort === 'family') filtered.sort((a,b) => String(a.family||'').localeCompare(String(b.family||'')) || a.id.localeCompare(b.id));
  else filtered.sort((a,b) => a.id.localeCompare(b.id));
  document.getElementById('models-grid').innerHTML = filtered.map(m => {
    const u = modelUsage[m.id];
    const ctxStr = m.context_length ? `<span class="mc-ctx">${fmtCtx(m.context_length)} ctx</span>` : '';
    const keyStr = `<span class="mc-keys">${m.available_on||0} key${m.available_on!==1?'s':''}</span>`;
    const caps = (m.capabilities||[]).map(c => `<span class="badge">${c}</span>`).join(' ');
    const params = m.parameter_count_display ? `${m.parameter_count_display} params` : (m.parameter_count ? `${fmt(m.parameter_count)} params` : '');
    const updated = m.modified_at ? `updated ${m.modified_at.slice(0,10)}` : '';
    const metaBits = [ctxStr, keyStr, m.family||'', params, updated].filter(Boolean).join(' · ');
    const usageLine = u ? `<div class="mc-usage">${fmt(u.requests)} calls · ${fmt(u.tokens_total)} tokens · ${Math.round(u.avg_latency_ms||0)}ms</div>` : '<div class="mc-usage" style="color:var(--dim)">No usage (7d)</div>';
    const provBadge = providerBadge((m.providers||[]).join(','));
    return `<div class="model-chip"><div class="mc-name">${m.id}${provBadge}</div><div class="mc-meta">${metaBits}</div><div class="mc-meta">${caps}</div>${usageLine}</div>`;
  }).join('');
}
document.getElementById('model-search').addEventListener('input', renderModelsGrid);
document.getElementById('model-sort').addEventListener('change', renderModelsGrid);

// --- NVIDIA Build Catalog ---
let catalogData = null;          // { catalog: [...], by_org: {...}, count, enabled }
let catalogLoaded = false;
const catalogOpenOrgs = new Set();

function fmtParamCount(n) {
  if (!n || n <= 0) return '';
  if (n >= 1e12) { const v = n / 1e12; return Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v) + 'T' : v.toFixed(1) + 'T'; }
  const v = n / 1e9; return Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v) + 'B' : v.toFixed(1) + 'B';
}

async function loadCatalog(force) {
  if (catalogLoaded && !force) return;
  const list = document.getElementById('catalog-list');
  list.innerHTML = '<div class="catalog-empty">Loading catalog...</div>';
  try {
    const r = await loadJSON('/admin/fallback-catalog');
    catalogData = r;
    catalogLoaded = true;
    if (!r || !r.enabled) {
      list.innerHTML = '<div class="catalog-empty">Fallback provider is not configured.</div>';
      document.getElementById('catalog-count').textContent = '0';
      return;
    }
    renderCatalog();
  } catch (e) {
    list.innerHTML = '<div class="catalog-empty">Failed to load catalog.</div>';
  }
}

function renderCatalog() {
  if (!catalogData || !catalogData.enabled) return;
  const all = catalogData.catalog || [];
  const q = (document.getElementById('catalog-search').value || '').toLowerCase().trim();
  const filtered = q
    ? all.filter(m => (m.id || '').toLowerCase().includes(q) || (m.org || '').toLowerCase().includes(q) || (m.owned_by || '').toLowerCase().includes(q))
    : all;
  const mappedCount = all.filter(m => m.is_mapped).length;
  document.getElementById('catalog-count').textContent = String(all.length);
  document.getElementById('catalog-summary').textContent = `${mappedCount} mapped · ${filtered.length}${q ? ' filtered' : ''} of ${all.length} total`;
  const byOrg = {};
  filtered.forEach(m => { const o = m.org || m.owned_by || 'unknown'; (byOrg[o] = byOrg[o] || []).push(m); });
  const orgs = Object.keys(byOrg).sort((a, b) => a.localeCompare(b));
  if (!orgs.length) {
    document.getElementById('catalog-list').innerHTML = '<div class="catalog-empty">No models match the filter.</div>';
    return;
  }
  // If a search query is active, auto-open all matching orgs.
  if (q) orgs.forEach(o => catalogOpenOrgs.add(o));
  document.getElementById('catalog-list').innerHTML = orgs.map(org => {
    const rows = byOrg[org].slice().sort((a, b) => (a.id || '').localeCompare(b.id || ''));
    const open = catalogOpenOrgs.has(org) ? ' open' : '';
    const orgMapped = rows.filter(r => r.is_mapped).length;
    const inner = rows.map(m => {
      const params = fmtParamCount(m.parameter_count);
      const ctx = m.context_length ? fmtCtx(m.context_length) : '';
      const action = m.is_mapped
        ? `<span class="cr-mapped">✓ <span class="cr-alias" title="ollama alias">${escHtml(m.ollama_equivalent || '')}</span></span>`
        : `<button class="btn btn-sm" onclick="catalogAdd('${escAttr(m.id)}')">+ Add</button>`;
      const url = m.model_card_url || ('https://build.nvidia.com/' + m.id);
      return `<div class="catalog-row" data-mid="${escAttr(m.id)}">
        <div class="cr-name"><a href="${escAttr(url)}" target="_blank" rel="noopener" title="Open model card">${escHtml(m.id)}</a></div>
        <div class="cr-meta">${escHtml(params || '-')}</div>
        <div class="cr-meta">${escHtml(ctx || '-')}</div>
        <div class="cr-meta" title="${escAttr(m.description || '')}">${escHtml(m.description ? (m.description.length > 80 ? m.description.slice(0, 77) + '...' : m.description) : '')}</div>
        <div class="cr-action">${action}</div>
      </div>`;
    }).join('');
    return `<div class="catalog-org${open}" data-org="${escAttr(org)}">
      <div class="catalog-org-header" onclick="catalogToggleOrg('${escAttr(org)}')">
        <span class="co-caret">▶</span>
        <span class="co-name">${escHtml(org)}</span>
        <span class="co-count">${rows.length} model${rows.length !== 1 ? 's' : ''} · ${orgMapped} mapped</span>
      </div>
      <div class="catalog-org-body">${inner}</div>
    </div>`;
  }).join('');
}
function catalogToggleOrg(org) {
  if (catalogOpenOrgs.has(org)) catalogOpenOrgs.delete(org); else catalogOpenOrgs.add(org);
  const el = document.querySelector(`.catalog-org[data-org="${CSS.escape(org)}"]`);
  if (el) el.classList.toggle('open');
}
function catalogAdd(nvidiaId) {
  showModal(`<h3>Add Fallback Mapping</h3>
    <p style="font-size:11px;color:var(--dim);margin-bottom:8px">Maps an Ollama model name to the NVIDIA Build model. Runtime-only — add to <code>config.yaml</code> to persist.</p>
    <label>Ollama Model Name</label>
    <input id="m-fb-ollama" placeholder="e.g. ${escAttr(nvidiaId.split('/').pop() || 'qwen3-coder')}" autofocus>
    <label>NVIDIA Model</label>
    <input id="m-fb-nvidia" value="${escAttr(nvidiaId)}" readonly>
    <label>Priority <span style="color:var(--dim)">(optional)</span></label>
    <select id="m-fb-priority">
      <option value="">inherit (default)</option>
      <option value="after">after — Ollama first</option>
      <option value="before">before — Fallback first</option>
      <option value="only">only — Fallback only</option>
    </select>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn" onclick="submitFbMapAdd()">Add</button>
    </div>`);
}
async function submitFbMapAdd() {
  const ollama = document.getElementById('m-fb-ollama').value.trim();
  const nvidia = document.getElementById('m-fb-nvidia').value.trim();
  const priority = document.getElementById('m-fb-priority').value;
  if (!ollama || !nvidia) { alert('Both ollama_name and nvidia_name are required'); return; }
  const body = { ollama_name: ollama, nvidia_name: nvidia };
  if (priority) body.priority = priority;
  try {
    await postJSON('/admin/fallback-map', body);
    closeModal();
    await loadFallbackStatus();
    await loadCatalog(true);
  } catch (e) { alert('Failed to add mapping: ' + e.message); }
}
async function fbMapRemove(ollamaName) {
  if (!confirm(`Remove fallback mapping for "${ollamaName}"? Runtime-only — also remove from config.yaml to persist.`)) return;
  try {
    await postJSON('/admin/fallback-map?ollama_name=' + encodeURIComponent(ollamaName), null, 'DELETE');
    await loadFallbackStatus();
    if (catalogLoaded) await loadCatalog(true);
  } catch (e) { alert('Failed to remove mapping: ' + e.message); }
}
document.getElementById('catalog-header').addEventListener('click', function() {
  const sec = document.getElementById('catalog-section');
  const open = sec.classList.toggle('open');
  if (open) loadCatalog(false);
});
document.getElementById('catalog-search').addEventListener('input', () => { if (catalogData) renderCatalog(); });
document.getElementById('catalog-refresh').addEventListener('click', () => loadCatalog(true));
document.getElementById('catalog-add-mapping').addEventListener('click', () => {
  showModal(`<h3>Add Fallback Mapping</h3>
    <p style="font-size:11px;color:var(--dim);margin-bottom:8px">Maps an Ollama model name to a NVIDIA Build model. Runtime-only — add to <code>config.yaml</code> to persist.</p>
    <label>Ollama Model Name</label><input id="m-fb-ollama" placeholder="e.g. qwen3-coder" autofocus>
    <label>NVIDIA Model</label><input id="m-fb-nvidia" placeholder="e.g. qwen/qwen3-coder-480b-a35b-instruct">
    <label>Priority <span style="color:var(--dim)">(optional)</span></label>
    <select id="m-fb-priority">
      <option value="">inherit (default)</option>
      <option value="after">after — Ollama first</option>
      <option value="before">before — Fallback first</option>
      <option value="only">only — Fallback only</option>
    </select>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn" onclick="submitFbMapAdd()">Add</button>
    </div>`);
});

// --- Subscriptions Panel ---
async function loadSubsPanel() {
  const keys = await loadJSON('/admin/keys');
  document.getElementById('subs-list').innerHTML = keys.map((k, i) => {
    const statusCls = k.suspended ? 'status-err' : 'status-ok';
    return `<div class="key-mgmt-card">
      <div class="km-header">
        <span class="km-label ${statusCls}">${k.label}</span>
        <div style="display:flex;gap:4px">
          <button class="btn btn-sm" onclick="editKey(${i})">Edit</button>
          <button class="btn btn-sm" onclick="editCookies(${i})">🍪 Cookies</button>
          <button class="btn btn-sm btn-danger" onclick="deleteKey(${i},'${k.label}')">Remove</button>
        </div>
      </div>
      <div class="km-row"><span>Token</span><span>${k.token_prefix}</span></div>
      <div class="km-row"><span>Plan</span><span>${k.plan||'?'}</span></div>
      <div class="km-row"><span>Max Concurrent</span><span>${k.max_concurrent}</span></div>
      <div class="km-row"><span>Cycle Day</span><span>${k.cycle_day}</span></div>
      <div class="km-row"><span>Suspended</span><span class="${k.suspended?'status-err':'status-ok'}">${k.suspended?'Yes':'No'}</span></div>
      ${k.account_email ? `<div class="km-row"><span>Email</span><span>${k.account_email}</span></div>` : ''}
      <div class="km-row"><span>Cookies</span><span class="${k.has_cookies?'cookie-ok':'cookie-bad'}">${k.has_cookies?'Configured':'Not set'}</span></div>
    </div>`;
  }).join('');
}

// --- Modals ---
function showModal(html) { document.getElementById('modal-root').innerHTML = `<div class="modal-overlay" onclick="if(event.target===this)closeModal()"><div class="modal">${html}</div></div>`; }
function closeModal() { document.getElementById('modal-root').innerHTML = ''; }

function editKey(idx) {
  showModal(`<h3>Edit Subscription</h3>
    <label>Label</label><input id="m-label" placeholder="Subscription label">
    <label>Max Concurrent</label><input id="m-concurrent" type="number" value="15" min="1" max="50">
    <label>Cycle Day (billing reset, 1-28)</label><input id="m-cycle" type="number" value="1" min="1" max="28">
    <div class="modal-actions"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn" onclick="saveKey(${idx})">Save</button></div>`);
}
async function saveKey(idx) {
  const label=document.getElementById('m-label').value, mc=parseInt(document.getElementById('m-concurrent').value), cd=parseInt(document.getElementById('m-cycle').value);
  await postJSON(`/admin/keys/${idx}?label=${encodeURIComponent(label)}&max_concurrent=${mc}&cycle_day=${cd}`, null, 'PUT');
  closeModal(); loadSubsPanel();
}

function editCookies(idx) {
  showModal(`<h3>Edit Cookies for Key ${idx}</h3>
    <p style="font-size:11px;color:var(--dim);margin-bottom:12px">Extract from browser DevTools → Application → Cookies → ollama.com</p>
    <label>__Secure-session <span style="color:var(--red)">*</span></label><textarea id="m-ss" placeholder="Required — unique per account"></textarea>
    <label>aid</label><input id="m-aid" placeholder="Account ID (shared across accounts)">
    <label>cf_clearance</label><input id="m-cf" placeholder="Cloudflare bypass (optional)">
    <label>__stripe_mid</label><input id="m-stripe" placeholder="Stripe session (optional)">
    <div class="modal-actions"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn" onclick="saveCookies(${idx})">Save</button></div>`);
}
async function saveCookies(idx) {
  const body = {};
  const ss=document.getElementById('m-ss').value.trim(); if(ss) body.secure_session=ss;
  const aid=document.getElementById('m-aid').value.trim(); if(aid) body.aid=aid;
  const cf=document.getElementById('m-cf').value.trim(); if(cf) body.cf_clearance=cf;
  const stripe=document.getElementById('m-stripe').value.trim(); if(stripe) body.stripe_mid=stripe;
  await postJSON(`/admin/keys/${idx}/cookies`, body, 'PUT');
  closeModal(); loadSubsPanel();
}

async function deleteKey(idx, label) {
  if (!confirm(`Remove "${label}"? This takes effect immediately but you should also remove it from config.yaml.`)) return;
  await postJSON(`/admin/keys/${idx}`, null, 'DELETE');
  loadSubsPanel();
}

// --- Client Key Management ---
async function loadClients() {
  const clients = await loadJSON('/admin/clients');
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById('client-list').innerHTML = (clients || []).map(c => {
    const dtl = c.daily_token_limit != null ? c.daily_token_limit : '∞';
    const drl = c.daily_request_limit != null ? c.daily_request_limit : '∞';
    const rpm = c.rpm_limit != null ? c.rpm_limit : '∞';
    const tok = c.token ? c.token.slice(0, 8) + '...' + c.token.slice(-4) : '—';
    return `<div class="key-mgmt-card">
      <div class="km-header">
        <span class="km-label">${c.label || c.id}</span>
        <div style="display:flex;gap:4px">
          <button class="btn btn-sm" onclick="editClient('${c.id}')">Edit</button>
          <button class="btn btn-sm" onclick="regenClient('${c.id}')">🔄 Token</button>
          <button class="btn btn-sm btn-danger" onclick="deleteClient('${c.id}','${c.label || c.id}')">Remove</button>
        </div>
      </div>
      <div class="km-row"><span>ID</span><span>${c.id}</span></div>
      <div class="km-row"><span>Token</span><span style="font-family:monospace; font-size:11px">${tok}</span></div>
      <div class="km-row"><span>Daily Token Limit</span><span>${dtl}</span></div>
      <div class="km-row"><span>Daily Request Limit</span><span>${drl}</span></div>
      <div class="km-row"><span>RPM Limit</span><span>${rpm}</span></div>
      ${c.notes ? `<div class="km-row"><span>Notes</span><span style="font-size:11px">${c.notes}</span></div>` : ''}
    </div>`;
  }).join('') || '<div style="color:var(--dim);font-size:12px;padding:8px">No client keys yet</div>';
}

function editClient(id) {
  // Fetch current values to pre-fill
  loadJSON('/admin/clients').then(clients => {
    const c = (clients || []).find(x => x.id === id);
    if (!c) return;
    showModal(`<h3>Edit Client: ${id}</h3>
      <label>Label</label><input id="m-cl-label" value="${c.label || ''}">
      <label>Notes</label><input id="m-cl-notes" value="${c.notes || ''}">
      <label>Daily Token Limit <span style="color:var(--dim)">(null = unlimited)</span></label><input id="m-cl-dtl" type="number" placeholder="unlimited" value="${c.daily_token_limit ?? ''}">
      <label>Daily Request Limit</label><input id="m-cl-drl" type="number" placeholder="unlimited" value="${c.daily_request_limit ?? ''}">
      <label>RPM Limit</label><input id="m-cl-rpm" type="number" placeholder="unlimited" value="${c.rpm_limit ?? ''}">
      <div class="modal-actions"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn" onclick="saveClient('${id}')">Save</button></div>`);
  });
}
async function saveClient(id) {
  const body = {};
  const label = document.getElementById('m-cl-label').value;
  if (label) body.label = label;
  const notes = document.getElementById('m-cl-notes').value;
  if (notes) body.notes = notes;
  const dtl = document.getElementById('m-cl-dtl').value;
  body.daily_token_limit = dtl === '' ? null : parseInt(dtl);
  const drl = document.getElementById('m-cl-drl').value;
  body.daily_request_limit = drl === '' ? null : parseInt(drl);
  const rpm = document.getElementById('m-cl-rpm').value;
  body.rpm_limit = rpm === '' ? null : parseInt(rpm);
  await postJSON(`/admin/clients/${id}`, body, 'PATCH');
  closeModal(); loadClients();
}
async function regenClient(id) {
  if (!confirm(`Regenerate token for "${id}"? The old token will stop working immediately.`)) return;
  const r = await postJSON(`/admin/clients/${id}/regenerate-token`, null, 'POST');
  closeModal(); loadClients();
  alert(`New token: ${r.token}`);
}
async function deleteClient(id, label) {
  if (!confirm(`Remove client "${label}"? The token will stop working immediately.`)) return;
  await postJSON(`/admin/clients/${id}`, null, 'DELETE');
  loadClients();
}

document.getElementById('btn-add-key').addEventListener('click', () => {
  showModal(`<h3>Add Subscription</h3>
    <p style="font-size:11px;color:var(--dim);margin-bottom:12px">Add an Ollama Cloud API key. Get it from ollama.com/settings/keys</p>
    <label>API Key <span style="color:var(--red)">*</span></label><textarea id="m-token" placeholder="Required — paste the full API key"></textarea>
    <label>Label</label><input id="m-label" placeholder="e.g. Sub 3">
    <label>Max Concurrent</label><input id="m-concurrent" type="number" value="15" min="1" max="50">
    <label>Cycle Day (billing reset, 1-28)</label><input id="m-cycle" type="number" value="1" min="1" max="28">
    <h3 style="margin-top:16px">Cookies (optional)</h3>
    <p style="font-size:11px;color:var(--dim);margin-bottom:8px">For usage tracking from ollama.com/settings</p>
    <label>__Secure-session</label><textarea id="m-ss" placeholder="Per-account session cookie"></textarea>
    <label>aid</label><input id="m-aid" placeholder="Account ID">
    <label>cf_clearance</label><input id="m-cf" placeholder="Cloudflare bypass">
    <label>__stripe_mid</label><input id="m-stripe" placeholder="Stripe session">
    <div class="modal-actions"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn" onclick="addKey()">Add</button></div>`);
});
async function addKey() {
  const token=document.getElementById('m-token').value.trim();
  if(!token){alert('API key is required');return;}
  const body={token, label:document.getElementById('m-label').value||undefined, max_concurrent:parseInt(document.getElementById('m-concurrent').value)||15, cycle_day:parseInt(document.getElementById('m-cycle').value)||1, cookies:{}};
  const ss=document.getElementById('m-ss').value.trim(); if(ss) body.cookies.secure_session=ss;
  const aid=document.getElementById('m-aid').value.trim(); if(aid) body.cookies.aid=aid;
  const cf=document.getElementById('m-cf').value.trim(); if(cf) body.cookies.cf_clearance=cf;
  const stripe=document.getElementById('m-stripe').value.trim(); if(stripe) body.cookies.stripe_mid=stripe;
  const r=await postJSON('/admin/keys', body);
  closeModal(); loadSubsPanel();
  alert(r.note||'Key added');
}

// --- Add Client ---
document.getElementById('btn-add-client').addEventListener('click', () => {
  showModal(`<h3>Add Client Key</h3>
    <p style="font-size:11px;color:var(--dim);margin-bottom:12px">Create a client key for usage attribution and rate limiting</p>
    <label>Client ID <span style="color:var(--red)">*</span></label><input id="m-cl-id" placeholder="e.g. my-app (alphanumeric, dashes ok)">
    <label>Label</label><input id="m-cl-label" placeholder="e.g. My Application">
    <label>Notes</label><input id="m-cl-notes" placeholder="Optional notes">
    <h4 style="margin-top:12px">Rate Limits <span style="color:var(--dim);font-weight:normal">(leave empty for unlimited)</span></h4>
    <label>Daily Token Limit</label><input id="m-cl-dtl" type="number" placeholder="unlimited">
    <label>Daily Request Limit</label><input id="m-cl-drl" type="number" placeholder="unlimited">
    <label>RPM Limit</label><input id="m-cl-rpm" type="number" placeholder="unlimited">
    <div class="modal-actions"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn" onclick="addClient()">Create</button></div>`);
});

async function addClient() {
  const id = document.getElementById('m-cl-id').value.trim();
  if (!id) { alert('Client ID is required'); return; }
  const body = { id, label: document.getElementById('m-cl-label').value || id };
  const notes = document.getElementById('m-cl-notes').value.trim();
  if (notes) body.notes = notes;
  const dtl = document.getElementById('m-cl-dtl').value;
  if (dtl) body.daily_token_limit = parseInt(dtl);
  const drl = document.getElementById('m-cl-drl').value;
  if (drl) body.daily_request_limit = parseInt(drl);
  const rpm = document.getElementById('m-cl-rpm').value;
  if (rpm) body.rpm_limit = parseInt(rpm);
  try {
    const r = await postJSON('/admin/clients', body);
    closeModal(); loadClients();
    alert(`Client created! Token: ${r.token}`);
  } catch (e) {
    // postJSON already shows error
  }
}

// --- Data Fetch ---
async function fetchPeriodData(opts) {
  opts = opts || {};
  const range=getDateRange(); if(!range.start||!range.end)return; updateBadges(range.label);
  const fetches=[
    loadJSON(`/admin/totals?start_date=${range.start}&end_date=${range.end}`),
    loadJSON(`/admin/usage/by-client?start_date=${range.start}&end_date=${range.end}`),
    loadJSON(`/admin/usage/by-model?start_date=${range.start}&end_date=${range.end}`),
    loadJSON(`/admin/usage/daily?start_date=${range.start}&end_date=${range.end}`),
  ];
  if(!opts.preserveFeed) fetches.push(fetchRecentCalls());
  const [totals,clients,models,daily,calls]=await Promise.all(fetches);
  const t=totals||{};
  document.getElementById('totals').innerHTML=`
    <div class="card"><div class="label">Total Calls</div><div class="value blue">${fmt(t.total_calls||0)}</div></div>
    <div class="card"><div class="label">Tokens In</div><div class="value green">${fmt(t.total_tokens_in||0)}</div></div>
    <div class="card"><div class="label">Tokens Out</div><div class="value purple">${fmt(t.total_tokens_out||0)}</div></div>
    <div class="card"><div class="label">Total Tokens</div><div class="value yellow">${fmt(t.total_tokens||0)}</div></div>`;
  const maxTok=Math.max(...(clients||[]).map(c=>c.tokens_total||0),1);
  document.querySelector('#client-table tbody').innerHTML=(clients||[]).map(c=>{const inW=Math.round((c.tokens_in/maxTok)*80),outW=Math.round((c.tokens_out/maxTok)*80); return `<tr><td>${c.client_id}</td><td>${fmt(c.requests)}</td><td>${fmt(c.tokens_in)}</td><td>${fmt(c.tokens_out)}</td><td>${fmt(c.tokens_total)}</td><td><div class="bars"><div class="bar in" style="width:${inW}px"></div><div class="bar out" style="width:${outW}px"></div></div></td></tr>`;}).join('');
  document.querySelector('#model-table tbody').innerHTML=(models||[]).map(m=>`<tr><td>${m.model}</td><td>${fmt(m.requests)}</td><td>${fmt(m.tokens_in)}</td><td>${fmt(m.tokens_out)}</td><td>${fmt(m.tokens_total)}</td><td>${Math.round(m.avg_latency_ms||0)}ms</td></tr>`).join('');
  document.querySelector('#daily-table tbody').innerHTML=(daily||[]).map(d=>`<tr><td>${d.day}</td><td>${fmt(d.requests)}</td><td>${fmt(d.tokens_in)}</td><td>${fmt(d.tokens_out)}</td><td>${fmt(d.tokens_total)}</td></tr>`).join('');
  if(!opts.preserveFeed) { feedCalls=calls||[]; renderFeed(feedCalls); }
  populateFilters(null,models);
}
async function fetchRecentCalls() { const r=getDateRange(); const c=document.getElementById('filter-client').value,m=document.getElementById('filter-model').value,l=document.getElementById('feed-limit').value; let u=`/admin/recent-calls?limit=${l}&start_date=${r.start}&end_date=${r.end}`; if(c)u+=`&client=${encodeURIComponent(c)}`; if(m)u+=`&model=${encodeURIComponent(m)}`; return loadJSON(u); }

// --- Event Listeners ---
document.getElementById('period-select').addEventListener('change', function(){ document.getElementById('custom-dates').style.display=this.value==='custom'?'':'none'; fetchPeriodData(); });
document.getElementById('date-start').addEventListener('change',()=>fetchPeriodData());
document.getElementById('date-end').addEventListener('change',()=>fetchPeriodData());
document.getElementById('filter-client').addEventListener('change',()=>fetchPeriodData());
document.getElementById('filter-model').addEventListener('change',()=>fetchPeriodData());
document.getElementById('feed-limit').addEventListener('change',()=>fetchPeriodData());

async function loadInFlightInitial() {
  try {
    const r = await loadJSON('/admin/in-flight');
    (r.in_flight || []).forEach(e => { inFlight[e.request_id] = e; });
    renderInFlight();
  } catch (e) { /* ignore */ }
}

// --- Init ---
fetchPeriodData();
loadFallbackStatus();
loadInFlightInitial();
connectSSE();

// --- Quota Cost ---
async function loadQuotaCost() {
  try {
    const data = await loadJSON('/admin/quota-cost');
    const tbody = document.querySelector('#quota-table tbody');
    const badge = document.getElementById('quota-badge');
    const note = document.getElementById('quota-note');
    const status = document.getElementById('quota-solve-status');
    if (!data.models || data.models.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" style="color:var(--dim);text-align:center;padding:2em">' +
        (data.note || 'No quota cost data yet.') + '</td></tr>';
      badge.textContent = '0';
      note.textContent = data.note || '';
      status.textContent = '';
      return;
    }
    badge.textContent = data.models.length;
    note.textContent = data.note || '';
    status.textContent = `${data.solved_coefficients}/${data.total_models} models solved from ${data.total_intervals} intervals | Total quota delta: ${data.total_quota_pct}%`;
    tbody.innerHTML = data.models.map(m => {
      const confColor = m.confidence === 'solved' ? 'var(--green)' : 'var(--yellow)';
      const split = m.coefficients_split;
      // For combined coefficients, show unified cost; for split, show in/out separately
      let costIn, costOut, ratio;
      if (split) {
        costIn = m.quota_pct_per_1k_input !== undefined ? m.quota_pct_per_1k_input.toFixed(6) : '—';
        costOut = m.quota_pct_per_1k_output !== undefined ? m.quota_pct_per_1k_output.toFixed(6) : '—';
        ratio = m.ratio_in_out !== undefined ? m.ratio_in_out.toFixed(2) + '×' : '—';
      } else {
        costIn = m.quota_pct_per_1k_tokens ? m.quota_pct_per_1k_tokens.toFixed(6) + ' (combined)' : '—';
        costOut = '—';
        ratio = '=';
      }
      return '<tr>' +
        '<td style="font-family:monospace">' + escHtml(m.model) + '</td>' +
        '<td style="color:' + confColor + ';font-weight:600">' + m.confidence + '</td>' +
        '<td>' + m.intervals + '</td>' +
        '<td>' + m.total_session_pct_consumed.toFixed(3) + '%</td>' +
        '<td>' + fmt(m.tokens_in) + '</td>' +
        '<td>' + fmt(m.tokens_out) + '</td>' +
        '<td>' + m.pct_of_total_quota.toFixed(1) + '%</td>' +
        '<td style="font-family:monospace">' + costIn + '</td>' +
        '<td style="font-family:monospace">' + costOut + '</td>' +
        '<td style="font-family:monospace">' + ratio + '</td>' +
        '</tr>';
    }).join('');
  } catch (e) { /* ignore */ }
}
document.getElementById('quota-refresh').addEventListener('click', loadQuotaCost);
// Load quota data when tab is clicked
document.querySelectorAll('.tab').forEach(t => {
  if (t.dataset.tab === 'quota') t.addEventListener('click', loadQuotaCost);
});
</script>
</body>
</html>"""


@app.get("/admin/quota-cost", dependencies=[Depends(_verify_admin)])
async def admin_quota_cost():
    """Infer per-model input/output quota cost coefficients via algebraic solve.
    
    Each scraper interval (where session_usage_pct jumps) gives us one equation:
        delta_pct = Σ(coeff_in_model × total_tokens_in_model + coeff_out_model × total_tokens_out_model)
    
    Ollama Cloud likely charges differently for input vs output tokens, so we
    track separate coefficients. Single-model intervals with 2+ observations at
    different in/out ratios can solve for both coefficients directly via 2 equations.
    Multi-model intervals solve when only one model's coefficients are unknown.
    """
    if not usage_db or not usage_db._conn:
        return {"models": [], "note": "No usage DB configured"}
    
    rows = usage_db._conn.execute("""
        SELECT upstream_key, model, tokens_in, tokens_out, session_usage_pct, ts
        FROM usage 
        WHERE session_usage_pct >= 0 AND status > 0 AND tokens_in + tokens_out > 0
        ORDER BY upstream_key, ts ASC
    """).fetchall()
    
    if not rows:
        return {"models": [], "note": "No usage data with quota snapshots yet."}
    
    # Group calls into epochs per key: consecutive calls sharing same session_usage_pct
    key_epochs: dict[str, list[dict]] = {}
    
    for row in rows:
        key_prefix, model, t_in, t_out, s_pct, ts = row
        if key_prefix not in key_epochs:
            key_epochs[key_prefix] = []
        epochs = key_epochs[key_prefix]
        if not epochs or epochs[-1]["pct"] != s_pct:
            epochs.append({"pct": s_pct, "calls": []})
        epochs[-1]["calls"].append({"model": model, "t_in": t_in, "t_out": t_out})
    
    # Build intervals: each is one equation delta_pct = Σ(c_in × t_in + c_out × t_out)
    intervals: list[dict] = []
    
    for key_prefix, epochs in key_epochs.items():
        for i in range(1, len(epochs)):
            delta_pct = epochs[i]["pct"] - epochs[i - 1]["pct"]
            if delta_pct <= 0:
                continue
            prev_calls = epochs[i - 1]["calls"]
            tokens_by_model: dict[str, dict] = {}
            for call in prev_calls:
                m = call["model"]
                if m not in tokens_by_model:
                    tokens_by_model[m] = {"in": 0, "out": 0}
                tokens_by_model[m]["in"] += call["t_in"]
                tokens_by_model[m]["out"] += call["t_out"]
            intervals.append({
                "delta_pct": delta_pct,
                "tokens_by_model": tokens_by_model,
                "key": key_prefix,
            })
    
    if not intervals:
        return {"models": [], "intervals": 0, "note": "No usage transitions detected yet."}
    
    all_models = sorted(set(m for i in intervals for m in i["tokens_by_model"]))
    
    # --- Algebraic solver with separate input/output coefficients ---
    # Each model has two unknowns: coeff_in and coeff_out
    # Each interval gives one equation:
    #   delta_pct = Σ(coeff_in_m × t_in_m + coeff_out_m × t_out_m)
    # We need N equations to solve for N unknowns.
    
    # For single-model intervals: 1 eq, 2 unknowns → not directly solvable
    # But multiple single-model intervals with different in/out ratios give 2+ equations
    # for the same model, which IS solvable (2 equations, 2 unknowns).
    
    # known_coeffs: model -> {in: float, out: float}
    known_coeffs: dict[str, dict] = {}
    
    # Collect single-model intervals per model for 2-variable solve
    single_model_intervals: dict[str, list[dict]] = {}
    for interval in intervals:
        models = list(interval["tokens_by_model"].keys())
        if len(models) == 1:
            model = models[0]
            if model not in single_model_intervals:
                single_model_intervals[model] = []
            single_model_intervals[model].append({
                "delta_pct": interval["delta_pct"],
                "t_in": interval["tokens_by_model"][model]["in"],
                "t_out": interval["tokens_by_model"][model]["out"],
            })
    
    # Pass 1: Solve single-model intervals via least-squares (2+ observations)
    # For model M with observations: delta_i = c_in × t_in_i + c_out × t_out_i
    # This is a standard 2-variable linear regression.
    # When in/out ratio is nearly constant (ill-conditioned), fall back to
    # combined coefficient: delta = c × (t_in + t_out)
    for model, obs in single_model_intervals.items():
        if len(obs) < 2:
            continue  # Need at least 2 observations with different ratios
        # Solve via normal equations: A^T A x = A^T b
        n = len(obs)
        sum_tt = sum(o["t_in"] * o["t_in"] for o in obs)
        sum_tu = sum(o["t_in"] * o["t_out"] for o in obs)
        sum_uu = sum(o["t_out"] * o["t_out"] for o in obs)
        sum_td = sum(o["t_in"] * o["delta_pct"] for o in obs)
        sum_ud = sum(o["t_out"] * o["delta_pct"] for o in obs)
        det = sum_tt * sum_uu - sum_tu * sum_tu
        
        # Check condition number: if matrix is ill-conditioned, the in/out
        # ratio is too similar across observations to distinguish coefficients.
        # Fall back to combined coefficient (treat all tokens equally).
        total_in = sum(o["t_in"] for o in obs)
        total_out = sum(o["t_out"] for o in obs)
        ratio_variety = (max(o["t_in"] / max(o["t_out"], 1) for o in obs) / 
                        max(min(o["t_in"] / max(o["t_out"], 1) for o in obs), 0.001))
        min_det = (sum_tt + sum_uu) * 1e-10  # Scale-relative threshold
        
        if abs(det) < min_det or ratio_variety < 10:
            # Ill-conditioned: can't distinguish input vs output cost.
            # Fall back to combined coefficient: delta = c × total_tokens
            total_tokens = sum(o["t_in"] + o["t_out"] for o in obs)
            total_delta = sum(o["delta_pct"] for o in obs)
            if total_tokens > 0:
                c_combined = total_delta / total_tokens
                if c_combined >= 0:
                    known_coeffs[model] = {"in": c_combined, "out": c_combined}
            continue
        
        c_in = (sum_uu * sum_td - sum_tu * sum_ud) / det
        c_out = (sum_tt * sum_ud - sum_tu * sum_td) / det
        # Sanity: coefficients should be non-negative (quota cost can't be negative)
        if c_in >= 0 and c_out >= 0:
            known_coeffs[model] = {"in": c_in, "out": c_out}
    
    # Pass 2: Solve multi-model intervals iteratively
    # If all models but one have known coefficients, we have 1 equation with 2 unknowns
    # → can't solve. But if one unknown model has known coefficients from pass 1
    # AND appears in a multi-model interval, we can extract the remaining model.
    # For the case of 1 unknown model in an interval:
    #   delta - Σ(known) = c_in × t_in + c_out × t_out  (1 eq, 2 unknowns)
    # We can only solve if we make an assumption. We'll use weighted average of known
    # models' in/out ratio as a fallback, or skip if impossible.
    # Instead, for 1 unknown model, if we have multiple such intervals, we get
    # multiple equations and can least-squares solve for that model's c_in/c_out.
    
    changed = True
    max_passes = 10
    while changed and max_passes > 0:
        changed = False
        max_passes -= 1
        
        # Collect equations for unknown models
        unknown_obs: dict[str, list[dict]] = {}
        for interval in intervals:
            unknown_models = [m for m in interval["tokens_by_model"] if m not in known_coeffs]
            known_models = [m for m in interval["tokens_by_model"] if m in known_coeffs]
            
            if len(unknown_models) != 1:
                continue  # Can't solve with 0 or 2+ unknowns in this pass
            
            unknown = unknown_models[0]
            # Compute residual: delta_pct minus all known models' contributions
            known_contrib = sum(
                known_coeffs[m]["in"] * interval["tokens_by_model"][m]["in"] +
                known_coeffs[m]["out"] * interval["tokens_by_model"][m]["out"]
                for m in known_models
            )
            residual = interval["delta_pct"] - known_contrib
            if residual <= 0:
                continue  # Sanity check
            
            if unknown not in unknown_obs:
                unknown_obs[unknown] = []
            unknown_obs[unknown].append({
                "delta_pct": residual,
                "t_in": interval["tokens_by_model"][unknown]["in"],
                "t_out": interval["tokens_by_model"][unknown]["out"],
            })
        
        # Try to solve each unknown model via least-squares
        for model, obs in unknown_obs.items():
            if model in known_coeffs:
                continue
            if len(obs) < 2:
                continue  # Need 2+ equations for 2 unknowns
            sum_tt = sum(o["t_in"] * o["t_in"] for o in obs)
            sum_tu = sum(o["t_in"] * o["t_out"] for o in obs)
            sum_uu = sum(o["t_out"] * o["t_out"] for o in obs)
            sum_td = sum(o["t_in"] * o["delta_pct"] for o in obs)
            sum_ud = sum(o["t_out"] * o["delta_pct"] for o in obs)
            det = sum_tt * sum_uu - sum_tu * sum_tu
            if abs(det) < 1e-20:
                continue
            c_in = (sum_uu * sum_td - sum_tu * sum_ud) / det
            c_out = (sum_tt * sum_ud - sum_tu * sum_td) / det
            if c_in >= 0 and c_out >= 0:
                known_coeffs[model] = {"in": c_in, "out": c_out}
                changed = True
    
    # Build result
    model_stats: dict[str, dict] = {}
    for interval in intervals:
        for model, tok in interval["tokens_by_model"].items():
            if model not in model_stats:
                model_stats[model] = {
                    "total_pct": 0.0, "intervals": 0,
                    "tokens_in": 0, "tokens_out": 0,
                }
            # Attribute: use known coefficients if available
            if model in known_coeffs:
                attributed_pct = (known_coeffs[model]["in"] * tok["in"] +
                                  known_coeffs[model]["out"] * tok["out"])
            else:
                # Unknown: distribute remaining delta proportionally
                known_pct_in_interval = sum(
                    known_coeffs[m]["in"] * interval["tokens_by_model"][m]["in"] +
                    known_coeffs[m]["out"] * interval["tokens_by_model"][m]["out"]
                    for m in interval["tokens_by_model"] if m in known_coeffs
                )
                remaining_pct = max(0, interval["delta_pct"] - known_pct_in_interval)
                unknown_total = sum(
                    interval["tokens_by_model"][m]["in"] + interval["tokens_by_model"][m]["out"]
                    for m in interval["tokens_by_model"] if m not in known_coeffs
                )
                if unknown_total > 0:
                    attributed_pct = remaining_pct * ((tok["in"] + tok["out"]) / unknown_total)
                else:
                    attributed_pct = 0
            model_stats[model]["total_pct"] += attributed_pct
            model_stats[model]["intervals"] += 1
            model_stats[model]["tokens_in"] += tok["in"]
            model_stats[model]["tokens_out"] += tok["out"]
    
    all_interval_pct = sum(i["delta_pct"] for i in intervals)
    solved_count = len(known_coeffs)
    total_models = len(all_models)
    
    result = []
    for model in sorted(model_stats.keys(), key=lambda m: -model_stats[m]["total_pct"]):
        ms = model_stats[model]
        total_tok = ms["tokens_in"] + ms["tokens_out"]
        coeff = known_coeffs.get(model)
        
        entry = {
            "model": model,
            "confidence": "solved" if coeff else "estimated",
            "intervals": ms["intervals"],
            "total_session_pct_consumed": round(ms["total_pct"], 3),
            "tokens_in": ms["tokens_in"],
            "tokens_out": ms["tokens_out"],
            "tokens_total": total_tok,
            "pct_of_total_quota": round(ms["total_pct"] / all_interval_pct * 100, 1) if all_interval_pct > 0 else 0,
        }
        
        if coeff:
            entry["coeff_in"] = round(coeff["in"], 10)
            entry["coeff_out"] = round(coeff["out"], 10)
            entry["quota_pct_per_1k_input"] = round(coeff["in"] * 1000, 6)
            entry["quota_pct_per_1k_output"] = round(coeff["out"] * 1000, 6)
            # Mark whether coefficients were actually separated or combined (equal)
            entry["coefficients_split"] = abs(coeff["in"] - coeff["out"]) > coeff["in"] * 0.01
            if not entry["coefficients_split"]:
                # Combined: just show the unified cost per 1K tokens
                entry["quota_pct_per_1k_tokens"] = round(coeff["in"] * 1000, 6)
            if coeff["out"] > 0:
                entry["ratio_in_out"] = round(coeff["in"] / coeff["out"], 2)
        
        result.append(entry)
    
    return {
        "models": result,
        "total_intervals": len(intervals),
        "total_quota_pct": round(all_interval_pct, 3),
        "solved_coefficients": solved_count,
        "total_models": total_models,
        "note": f"Algebraic solve with separate input/output coefficients: {solved_count}/{total_models} models solved from {len(intervals)} intervals. "
                f"Each model has 2 unknowns (coeff_in, coeff_out); solved via least-squares when 2+ observations exist."
    }


@app.get("/dashboard", dependencies=[Depends(_verify_admin)])
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


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

    config = Config(app, host=host, port=port, log_level="info")
    server = Server(config)
    log.info(f"Starting LlamaHerd on {host}:{port} — One endpoint. Many llamas. Smarter routing.")
    server.run()


if __name__ == "__main__":
    main()
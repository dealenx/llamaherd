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
import queue
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
import os
from pathlib import Path
from typing import Optional

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

            candidates.sort(key=lambda k: (
                k.weekly_usage_pct if k.weekly_usage_pct >= 0 else 999,
                k.session_usage_pct if k.session_usage_pct >= 0 else 999,
                k.cycle_freshness,
                k.in_flight,
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
            key.mark_exhausted(3600)

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
                status INTEGER NOT NULL
            )
        """)
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
               tokens_in: int, tokens_out: int, latency_ms: int, status: int):
        today = datetime.now(timezone.utc).date().isoformat()
        self._conn.execute(
            "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), today, client_id, upstream_key, model,
             tokens_in, tokens_out, latency_ms, status),
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
upstream_url: str = ""
retry_on_429: bool = True
max_retries: int = 2
queue_timeout: int = 60
request_timeout: int = 120
admin_token: str = ""

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


def _record_and_broadcast(client_id: str, upstream_key: str, model: str,
                           tokens_in: int, tokens_out: int, latency_ms: int, status: int):
    """Record usage to DB and broadcast the event to SSE subscribers."""
    usage_db.record(client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status)
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
    # Use asyncio.run_coroutine_threadsafe? No — we're in async context normally.
    # But _record_and_broadcast is called from sync proxy code, so we schedule the broadcast.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcaster.broadcast("call", call_data))
            if manager:
                asyncio.ensure_future(broadcaster.broadcast("status", {"keys": manager.status(), "upstream": upstream_url}))
        else:
            loop.run_until_complete(broadcaster.broadcast("call", call_data))
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
    global manager, registry, usage_db, client_registry, usage_scraper
    global upstream_url, retry_on_429, max_retries, queue_timeout, request_timeout
    global admin_token

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
    log.info(f"Proxy started: {len(manager.keys)} upstream keys ({len(usage_scraper.cookie_map)} with usage cookies), {len(registry.models)} models, {len(client_registry.clients)} clients")

    yield

    sub_task.cancel()
    usage_task.cancel()
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
            _record_and_broadcast(client_id, "none", model, 0, 0, 0, 503)
            return JSONResponse(
                status_code=503,
                content={"error": "all keys at capacity, queue timeout exceeded"},
            )

        try:
            start = time.time()
            headers = {
                "Authorization": f"Bearer {key.token}",
                "Content-Type": "application/json",
            }

            if is_stream:
                return await _proxy_stream(client_id, key, path, headers, body, model, start)

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
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 429)
                prefer_key = None
                continue

            if resp.status_code == 402:
                log.warning(f"402 from {key.label} for {model} (client={client_id})")
                await manager.mark_402(key)
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 402)
                prefer_key = None
                continue

            resp_data = resp.json() if resp.status_code == 200 else {}
            usage = resp_data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms, resp.status_code)

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

    _record_and_broadcast(client_id, "none", model, 0, 0, 0, 502)
    return JSONResponse(
        status_code=502,
        content={"error": f"all retries exhausted: {last_error}"},
    )


async def _proxy_stream(client_id: str, key: KeyState, path: str,
                         headers: dict, body: bytes, model: str, start: float) -> StreamingResponse:

    async def generate():
        tokens_out = 0
        tokens_in = 0
        usage_captured = False
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                async with client_http.stream("POST", f"{upstream_url}{path}",
                                              content=body, headers=headers) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        _record_and_broadcast(client_id, key.token[:8], model, 0, 0, 0, 429)
                        yield f'data: {{"error": "429 from upstream"}}\n\n'
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
                        _record_and_broadcast(client_id, key.token[:8], model, 0, 0, 0, 402)
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
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
        except Exception as e:
            log.error(f"Stream error for {model} (client={client_id}): {e}")
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms, 200)
            usage_src = "usage" if usage_captured else "estimate"
            log.info(f"{client_id} -> {model} via {key.label}: stream {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({usage_src})")

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
                                start: float) -> StreamingResponse:
    """Stream NDJSON from the native Ollama API, capturing usage from the final chunk."""

    async def generate():
        tokens_out = 0
        tokens_in = 0
        usage_captured = False
        api_upstream = _native_api_upstream()
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client_http:
                async with client_http.stream("POST", f"{api_upstream}{path}",
                                              content=body, headers=headers) as resp:
                    if resp.status_code == 429:
                        await manager.mark_429(key)
                        _record_and_broadcast(client_id, key.token[:8], model, 0, 0, 0, 429)
                        # Yield an NDJSON error line
                        yield json.dumps({"error": "429 from upstream"}) + "\n"
                        return
                    if resp.status_code == 402:
                        await manager.mark_402(key)
                        _record_and_broadcast(client_id, key.token[:8], model, 0, 0, 0, 402)
                        yield json.dumps({"error": "402 from upstream"}) + "\n"
                        return
                    if resp.status_code >= 400:
                        # For non-2xx, read the body and yield as a single NDJSON line
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
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
        except Exception as e:
            log.error(f"NDJSON stream error for {model} (client={client_id}): {e}")
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms, 200)
            usage_src = "usage" if usage_captured else "estimate"
            log.info(f"{client_id} -> {model} via {key.label}: ndjson {tokens_in}+{tokens_out}tok {elapsed_ms}ms ({usage_src})")

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def _proxy_ndjson_request(request: Request, path: str) -> Response:
    """Core proxy logic for native Ollama /api/* routes — acquire key, forward, handle 429s, release.

    Handles both streaming (NDJSON) and non-streaming (JSON) native Ollama API requests.
    """
    client = _resolve_client(request)
    client_id = client["id"]

    body = await request.body()
    req_json = json.loads(body) if body else {}
    model = req_json.get("model", "unknown")
    is_stream = req_json.get("stream", False)

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
            _record_and_broadcast(client_id, "none", model, 0, 0, 0, 503)
            return JSONResponse(
                status_code=503,
                content={"error": "all keys at capacity, queue timeout exceeded"},
            )

        try:
            start = time.time()
            headers = {
                "Authorization": f"Bearer {key.token}",
                "Content-Type": "application/json",
            }
            api_upstream = _native_api_upstream()

            if is_stream:
                return await _proxy_ndjson_stream(client_id, key, path, headers, body, model, start)

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
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 429)
                prefer_key = None
                continue

            if resp.status_code == 402:
                log.warning(f"402 from {key.label} for {model} (client={client_id})")
                await manager.mark_402(key)
                _record_and_broadcast(client_id, key.token[:8], model, 0, 0, elapsed_ms, 402)
                prefer_key = None
                continue

            # Extract usage from non-streaming response
            resp_data = resp.json() if resp.status_code == 200 else {}
            tokens_in = resp_data.get("prompt_eval_count", 0) or 0
            tokens_out = resp_data.get("eval_count", 0) or 0
            await manager.release(key, tokens_out)
            _record_and_broadcast(client_id, key.token[:8], model, tokens_in, tokens_out, elapsed_ms, resp.status_code)

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

    _record_and_broadcast(client_id, "none", model, 0, 0, 0, 502)
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
    if registry:
        return registry.get_models_response()
    return {"object": "list", "data": []}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if registry and model_id in registry.models:
        return registry._model_entry(model_id)
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
                log.warning(f"429 from {key.label} for /api/show model={model}")
                prefer_key = None
                continue

            if resp.status_code == 402:
                await manager.mark_402(key)
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
    if not registry:
        return {"models": [], "count": 0, "last_refresh": 0}
    models_data = []
    for model_id, keys in registry.models.items():
        meta = registry.model_metadata.get(model_id, {})
        models_data.append({
            "id": model_id,
            "context_length": meta.get("context_length") or MODEL_CONTEXT_LENGTHS.get(model_id),
            "available_on": len(keys),
            "modified_at": meta.get("modified_at"),
            "size": meta.get("size"),
            "digest": meta.get("digest"),
            "capabilities": meta.get("capabilities") or [],
            "family": meta.get("family"),
            "parameter_count": meta.get("parameter_count"),
            "quantization_level": meta.get("quantization_level"),
        })
    models_data.sort(key=lambda m: m["id"])
    return {
        "models": models_data,
        "count": len(registry.models),
        "last_refresh": registry.last_refresh,
    }


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
      <option value="today">Today</option>
      <option value="yesterday">Yesterday</option>
      <option value="7d">Last 7 days</option>
      <option value="this_week">This Week</option>
      <option value="this_month" selected>This Month</option>
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
</div>

<div class="tab-bar">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="models">Models</div>
  <div class="tab" data-tab="subs">Subscriptions</div>
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
      if(m.type==='call') { addCallToFeed(m.data); if(callInCurrentRange(m.data)) schedulePeriodRefresh(); }
      else if(m.type==='status') { const d=m.data||{}; renderKeyStatus(d.keys||[]); if(d.upstream) document.getElementById('upstream-url').textContent=d.upstream; }
      else if(m.type==='models') updateModelInfo(m.data||{});
    } catch(err){}
  };
}

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
function renderFeed(calls) { document.getElementById('feed-count').textContent=calls.length; document.querySelector('#feed-tbl tbody').innerHTML=calls.map(c=>{const sc=c.status===200?'status-ok':c.status>=400?'status-err':'status-warn'; return `<tr><td>${fmtTs(c.ts)}</td><td>${c.client_id}</td><td>${c.model}</td><td>${(c.upstream_key||'').slice(0,8)}</td><td>${fmt(c.tokens_in)}</td><td>${fmt(c.tokens_out)}</td><td>${c.latency_ms}ms</td><td class="${sc}">${c.status}</td></tr>`;}).join(''); }
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
    const params = m.parameter_count ? `${fmt(m.parameter_count)} params` : '';
    const updated = m.modified_at ? `updated ${m.modified_at.slice(0,10)}` : '';
    const metaBits = [ctxStr, keyStr, m.family||'', params, updated].filter(Boolean).join(' · ');
    const usageLine = u ? `<div class="mc-usage">${fmt(u.requests)} calls · ${fmt(u.tokens_total)} tokens · ${Math.round(u.avg_latency_ms||0)}ms</div>` : '<div class="mc-usage" style="color:var(--dim)">No usage (7d)</div>';
    return `<div class="model-chip"><div class="mc-name">${m.id}</div><div class="mc-meta">${metaBits}</div><div class="mc-meta">${caps}</div>${usageLine}</div>`;
  }).join('');
}
document.getElementById('model-search').addEventListener('input', renderModelsGrid);
document.getElementById('model-sort').addEventListener('change', renderModelsGrid);

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

// --- Init ---
fetchPeriodData();
connectSSE();
</script>
</body>
</html>"""




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
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

import httpx

log = logging.getLogger("llamaherd")


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
    period_start: str | None = None  # ISO timestamp
    period_end: str | None = None      # ISO timestamp
    suspended: bool = False
    account_email: str = ""
    account_id: str = ""
    # Populated by cookie-based settings scrape
    session_usage_pct: float = -1.0  # -1 = unknown
    session_resets_at: str | None = None
    weekly_usage_pct: float = -1.0
    weekly_resets_at: str | None = None
    session_models: dict = field(default_factory=dict)
    weekly_models: dict = field(default_factory=dict)

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
                start = datetime.fromisoformat(self.period_start)
                end = datetime.fromisoformat(self.period_end)
                now = datetime.now(UTC)
                total = (end - start).total_seconds()
                elapsed = (now - start).total_seconds()
                if total <= 0:
                    return 0.5  # edge case
                return max(0.0, min(1.0, elapsed / total))
            except (ValueError, TypeError):
                pass
        # Fallback: use cycle_day
        now = datetime.now(UTC)
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
                start = datetime.fromisoformat(self.period_start)
                end = datetime.fromisoformat(self.period_end)
                now = datetime.now(UTC)
                total = (end - start).total_seconds()
                remaining = (end - now).total_seconds()
                if total <= 0:
                    return 0.0
                return max(0.0, min(100.0, (remaining / total) * 100))
            except (ValueError, TypeError):
                pass
        # Fallback: use cycle_day
        now = datetime.now(UTC)
        remaining_days = (self.cycle_day - now.day) % 30 or 30
        return round((remaining_days / 30) * 100, 1)

    def _elapsed_from_iso(self, iso_start: str | None, iso_end: str | None) -> float:
        """Calculate elapsed percentage (0-100) between two ISO timestamps. Returns -1 if unknown."""
        if not iso_start or not iso_end:
            return -1.0
        try:
            start = datetime.fromisoformat(iso_start)
            end = datetime.fromisoformat(iso_end)
            now = datetime.now(UTC)
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
                end = datetime.fromisoformat(self.session_resets_at)
                now = datetime.now(UTC)
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
                end = datetime.fromisoformat(self.weekly_resets_at)
                now = datetime.now(UTC)
                remaining = (end - now).total_seconds()
                total = 7 * 86400  # 7 days
                elapsed = total - remaining
                if total <= 0:
                    return 100.0
                return round(max(0.0, min(100.0, (elapsed / total) * 100)), 1)
            except (ValueError, TypeError):
                return -1.0
        return -1.0


class StickySessionManager:
    """Manages sticky session -> upstream key mappings with TTL for cache affinity.

    Follows industry standards for session stickiness (e.g. nginx/ELB cookie
    expiration defaults around 20min-1h). Default 3600s (1h) balances cache
    reuse against staleness risk. Sessions auto-expire; on upstream errors
    (429/402) the caller should clear to allow rebalancing.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._sessions: dict[str, dict] = {}  # session_id -> {"key_token": str, "expires_at": float}
        self._lock = asyncio.Lock()

    async def get_preferred_key(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry and time.time() < entry["expires_at"]:
                return entry["key_token"]
            if entry:
                self._sessions.pop(session_id, None)
            return None

    async def set_session(self, session_id: str, key_token: str) -> None:
        async with self._lock:
            self._sessions[session_id] = {
                "key_token": key_token,
                "expires_at": time.time() + self.ttl,
            }

    async def clear_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        async with self._lock:
            self._sessions.pop(session_id, None)

    def get_status(self) -> dict:
        """Return active sticky sessions for admin/debug (sanitized)."""
        now = time.time()
        active = {}
        for sid, entry in list(self._sessions.items()):
            if entry["expires_at"] > now:
                active[sid] = {
                    "key_prefix": entry["key_token"][:8] + "...",
                    "expires_in_sec": int(entry["expires_at"] - now),
                }
            else:
                self._sessions.pop(sid, None)
        return active


# ---------------------------------------------------------------------------
# Upstream Key Registry — persists Ollama Cloud keys + cookies to DB
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Telegram Notifier — sends usage alerts to Telegram chat/group/topic
# ---------------------------------------------------------------------------

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

    # Routing thresholds for weekly-aware selection. Tuned for the common case
    # where one or more subs are already over weekly pace and a fresher sub
    # should absorb traffic until its own capacity is full.
    WEEKLY_HARD_LIMIT = 99.0          # near-hard weekly cap — last resort only
    WEEKLY_NEAR_BEST_PP = 20.0        # prefer keys within this many pp of the best weekly
    WEEKLY_LOW_ABSOLUTE = 40.0        # always treat absolute weekly under this as preferred
    WEEKLY_PACE_FACTOR = 1.15         # under-pace if weekly <= elapsed * factor
    STICKY_REBIND_WEEKLY_MARGIN = 5.0 # do not re-pin sticky onto a much worse weekly key

    @staticmethod
    def _weekly_pct(k: "KeyState") -> float:
        """Normalized weekly usage; unknown (-1) sorts last as 999."""
        return k.weekly_usage_pct if k.weekly_usage_pct >= 0 else 999.0

    @staticmethod
    def _session_pct(k: "KeyState") -> float:
        return k.session_usage_pct if k.session_usage_pct >= 0 else 999.0

    def _is_preferred_weekly(self, k: "KeyState", min_weekly: float) -> bool:
        """Whether *k* belongs in the preferred (fresh-weekly) pool.

        Prefer keys that are under weekly pace, close to the best available
        weekly usage, or still have large absolute weekly headroom. This keeps
        traffic on a fresh Sub N while older subs are already over weekly.
        """
        w = self._weekly_pct(k)
        if w >= self.WEEKLY_HARD_LIMIT:
            return False
        if w <= min_weekly + self.WEEKLY_NEAR_BEST_PP:
            return True
        if w < self.WEEKLY_LOW_ABSOLUTE:
            return True
        elapsed = k._weekly_elapsed_pct()
        return bool(elapsed >= 0 and w <= max(elapsed * self.WEEKLY_PACE_FACTOR, elapsed + 5.0))

    def _select_from_candidates(self, candidates: list["KeyState"]) -> "KeyState":
        """Pick best key from available candidates.

        Policy (Jul 2026 — balance under-weekly subs before burning over-weekly):
        1. Soft-exclude keys at/near weekly hard limit unless nothing else.
        2. Build a preferred pool of under-weekly / near-best-weekly keys.
        3. Load-balance by in_flight *inside* that pool so multi-thread capacity
           still spreads when multiple fresh keys exist.
        4. Spill to over-weekly keys only when the preferred pool has no slots.
        """
        usable = [k for k in candidates if self._weekly_pct(k) < self.WEEKLY_HARD_LIMIT]
        if not usable:
            usable = list(candidates)

        min_weekly = min(self._weekly_pct(k) for k in usable)
        preferred = [k for k in usable if self._is_preferred_weekly(k, min_weekly)]
        pool = preferred if preferred else usable

        pool.sort(key=lambda k: (
            k.in_flight,
            self._weekly_pct(k),
            self._session_pct(k),
            k.cycle_freshness,
            k.total_tokens,
        ))
        return pool[0]

    def key_by_token(self, token: str | None) -> Optional["KeyState"]:
        if not token:
            return None
        for k in self.keys:
            if k.token == token:
                return k
        return None

    def should_rebind_sticky(self, prev_token: str | None, new_key: "KeyState") -> bool:
        """Whether sticky mapping should move from *prev_token* to *new_key*.

        Temporary spills (sticky sub at max concurrent / short 429 cooldown)
        must NOT permanently re-pin the session onto an over-weekly sub.
        Keep the original sticky when it still has weekly headroom and is only
        temporarily unavailable; rebind only when the previous key is gone,
        hard-exhausted, near weekly cap, or the new key is not worse on weekly.
        """
        if not prev_token:
            return True
        if new_key.token == prev_token:
            return True  # refresh TTL on same key

        prev = self.key_by_token(prev_token)
        if prev is None:
            return True
        if prev.suspended:
            return True
        # Long exhaustion (402-style) or near weekly hard limit → rebind OK
        if prev.exhausted and (prev.exhausted_until - time.time()) > 120:
            return True
        if self._weekly_pct(prev) >= self.WEEKLY_HARD_LIMIT:
            return True
        # Temporary alternate onto a worse-weekly key: keep original sticky
        return not self._weekly_pct(new_key) > self._weekly_pct(prev) + self.STICKY_REBIND_WEEKLY_MARGIN

    async def acquire(self, prefer_key: str | None = None, sticky_key: str | None = None) -> KeyState | None:
        async with self._lock:
            # Sticky key takes precedence for cache affinity (even if higher load)
            if sticky_key:
                for k in self.keys:
                    if k.token == sticky_key and not k.suspended and k.available_slots > 0:
                        k.in_flight += 1
                        return k
                # Sticky key exhausted or unavailable — fall through to free select.
                # Caller must use should_rebind_sticky() so temporary spills do not
                # permanently re-pin onto an over-weekly sub.

            if prefer_key:
                for k in self.keys:
                    if k.token == prefer_key and not k.suspended and k.available_slots > 0:
                        k.in_flight += 1
                        return k

            candidates = [k for k in self.keys if not k.suspended and k.available_slots > 0]
            if not candidates:
                return None

            best = self._select_from_candidates(candidates)
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
            "session_models": k.session_models,
            "weekly_models": k.weekly_models,
        } for k in self.keys]

    def key_by_token_prefix(self, prefix: str) -> KeyState | None:
        """Look up a key by its token prefix (first 8 chars)."""
        for k in self.keys:
            if k.token[:8] == prefix[:8]:
                return k
        return None

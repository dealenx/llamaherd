"""Unit tests for weekly-preferred key selection and sticky rebind policy."""
from __future__ import annotations

import asyncio
import time
from datetime import UTC

import pytest

from llamaherd.proxy import KeyManager, KeyState


def _mgr(keys: list[dict]) -> KeyManager:
    return KeyManager(keys)


def _set_usage(k: KeyState, *, weekly: float, session: float = 0.0, weekly_elapsed: float = 50.0):
    k.weekly_usage_pct = weekly
    k.session_usage_pct = session
    # Fake weekly_resets_at so _weekly_elapsed_pct returns ~weekly_elapsed
    # total window = 7d; remaining = total * (1 - elapsed/100)
    total = 7 * 86400
    remaining = total * (1.0 - weekly_elapsed / 100.0)
    # session_resets_at / weekly_resets_at are ISO end times
    end = time.time() + remaining
    from datetime import datetime

    k.weekly_resets_at = datetime.fromtimestamp(end, tz=UTC).isoformat().replace("+00:00", "Z")
    # session elapsed unused for preferred pool primarily; set a mid session
    sess_total = 18000
    sess_remaining = sess_total * 0.5
    k.session_resets_at = (
        datetime.fromtimestamp(time.time() + sess_remaining, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


@pytest.mark.asyncio
async def test_prefers_fresh_weekly_even_when_busy():
    """Sub 4 under weekly must win free-select even if it has in_flight > 0."""
    m = _mgr(
        [
            {"token": "sub1tokenxxxxxxxx", "label": "Sub 1", "max_concurrent": 15},
            {"token": "sub4tokenxxxxxxxx", "label": "Sub 4", "max_concurrent": 15},
        ]
    )
    s1, s4 = m.keys
    _set_usage(s1, weekly=68.0, session=10.0, weekly_elapsed=45.0)
    _set_usage(s4, weekly=18.7, session=48.0, weekly_elapsed=45.0)
    s4.in_flight = 3  # busy
    s1.in_flight = 0  # idle — old policy would pick this

    picked = await m.acquire()
    assert picked is s4
    assert s4.in_flight == 4


@pytest.mark.asyncio
async def test_load_balances_within_fresh_pool():
    """Two under-weekly keys should load-balance by in_flight."""
    m = _mgr(
        [
            {"token": "freshaaaaaaaaaaaa", "label": "A", "max_concurrent": 15},
            {"token": "freshbbbbbbbbbbbb", "label": "B", "max_concurrent": 15},
            {"token": "oldcccccccccccccccc", "label": "Old", "max_concurrent": 15},
        ]
    )
    a, b, old = m.keys
    _set_usage(a, weekly=10.0, weekly_elapsed=40.0)
    _set_usage(b, weekly=12.0, weekly_elapsed=40.0)
    _set_usage(old, weekly=70.0, weekly_elapsed=40.0)
    a.in_flight = 2
    b.in_flight = 0

    picked = await m.acquire()
    assert picked is b  # fewer in_flight inside preferred pool


@pytest.mark.asyncio
async def test_spills_to_over_weekly_when_fresh_full():
    """When preferred key is at max concurrent, spill to over-weekly for capacity."""
    m = _mgr(
        [
            {"token": "sub1tokenxxxxxxxx", "label": "Sub 1", "max_concurrent": 2},
            {"token": "sub4tokenxxxxxxxx", "label": "Sub 4", "max_concurrent": 2},
        ]
    )
    s1, s4 = m.keys
    _set_usage(s1, weekly=68.0, weekly_elapsed=45.0)
    _set_usage(s4, weekly=18.7, weekly_elapsed=45.0)
    s4.in_flight = 2  # full → available_slots = 0

    picked = await m.acquire()
    assert picked is s1


@pytest.mark.asyncio
async def test_sticky_honoured_when_available():
    m = _mgr(
        [
            {"token": "sub1tokenxxxxxxxx", "label": "Sub 1", "max_concurrent": 15},
            {"token": "sub4tokenxxxxxxxx", "label": "Sub 4", "max_concurrent": 15},
        ]
    )
    s1, s4 = m.keys
    _set_usage(s1, weekly=10.0, weekly_elapsed=45.0)  # better weekly than sticky
    _set_usage(s4, weekly=30.0, weekly_elapsed=45.0)

    picked = await m.acquire(sticky_key=s4.token)
    assert picked is s4


def test_should_not_rebind_sticky_to_worse_weekly():
    m = _mgr(
        [
            {"token": "sub1tokenxxxxxxxx", "label": "Sub 1", "max_concurrent": 15},
            {"token": "sub4tokenxxxxxxxx", "label": "Sub 4", "max_concurrent": 15},
        ]
    )
    s1, s4 = m.keys
    _set_usage(s1, weekly=68.0, weekly_elapsed=45.0)
    _set_usage(s4, weekly=18.7, weekly_elapsed=45.0)

    assert m.should_rebind_sticky(None, s4) is True
    assert m.should_rebind_sticky(s4.token, s4) is True  # refresh
    assert m.should_rebind_sticky(s4.token, s1) is False  # temporary spill
    # Hard-exhaust s4 → rebind allowed
    s4.mark_exhausted(86400)
    assert m.should_rebind_sticky(s4.token, s1) is True


@pytest.mark.asyncio
async def test_suspended_keys_skipped():
    m = _mgr(
        [
            {"token": "sub1tokenxxxxxxxx", "label": "Sub 1", "max_concurrent": 15},
            {"token": "sub4tokenxxxxxxxx", "label": "Sub 4", "max_concurrent": 15},
        ]
    )
    s1, s4 = m.keys
    _set_usage(s1, weekly=20.0, weekly_elapsed=45.0)
    _set_usage(s4, weekly=10.0, weekly_elapsed=45.0)
    s4.suspended = True

    picked = await m.acquire()
    assert picked is s1


if __name__ == "__main__":
    asyncio.run(test_prefers_fresh_weekly_even_when_busy())
    print("ok")

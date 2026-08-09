"""Regression tests for upstream-key transient cooldown behavior."""

import time

import pytest

from llamaherd.key_manager import KeyState
from llamaherd.proxy import KeyManager


@pytest.mark.asyncio
async def test_429_cooldown_is_short_transient_backoff():
    manager = KeyManager([
        {"token": "tok-1", "label": "key 1", "max_concurrent": 1},
    ])
    key = manager.keys[0]

    await manager.mark_429(key)

    assert key.exhausted is True
    remaining = key.exhausted_until - time.time()
    assert 0 < remaining <= 65


@pytest.mark.asyncio
async def test_402_does_not_cool_down_key():
    """402 is model/plan entitlement (extra-credit models), not key death."""
    manager = KeyManager([
        {"token": "tok-1", "label": "key 1", "max_concurrent": 1},
    ])
    key = manager.keys[0]

    await manager.mark_402(key)

    assert key.exhausted is False
    assert key.exhausted_until == 0.0
    assert key.total_402s == 1
    assert key.available_slots == 1
    # A second 402 still must not park the key.
    await manager.mark_402(key)
    assert key.exhausted is False
    assert key.total_402s == 2


def test_mark_exhausted_caps_at_five_minutes():
    """Nothing may park a key for hours/days — clamp absurd cooldowns."""
    key = KeyState(token="tok-1", label="key 1", max_concurrent=1)
    key.mark_exhausted(86400)
    remaining = key.exhausted_until - time.time()
    assert 0 < remaining <= KeyState.MAX_COOLDOWN_SECONDS + 1
    assert key.exhausted is True

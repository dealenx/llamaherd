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


@pytest.mark.asyncio
async def test_acquire_exclude_keys_skips_402_key_but_keeps_it_available():
    """Request-local exclusion: do not re-pick a 402 key on the same request."""
    manager = KeyManager([
        {"token": "tok-a", "label": "sub-a", "max_concurrent": 15},
        {"token": "tok-b", "label": "sub-b", "max_concurrent": 15},
    ])
    a, b = manager.keys

    # Simulate: first attempt got a; 402 → exclude a for remainder of request.
    first = await manager.acquire()
    assert first is a or first is b
    await manager.release(first)
    await manager.mark_402(first)

    # Without exclude, routing can pick the same key again (still healthy).
    again = await manager.acquire()
    assert again is not None
    await manager.release(again)

    # With exclude, must get the other key.
    other = await manager.acquire(exclude_keys={first.token})
    assert other is not None
    assert other.token != first.token
    await manager.release(other)

    # Excluded key is still globally available to other requests.
    same = await manager.acquire(exclude_keys={other.token})
    assert same is not None
    assert same.token == first.token
    await manager.release(same)

    # Sticky/prefer cannot override exclude.
    sticky_pick = await manager.acquire(sticky_key=first.token, exclude_keys={first.token})
    assert sticky_pick is not None
    assert sticky_pick.token != first.token
    await manager.release(sticky_pick)


def test_mark_exhausted_caps_at_five_minutes():
    """Nothing may park a key for hours/days — clamp absurd cooldowns."""
    key = KeyState(token="tok-1", label="key 1", max_concurrent=1)
    key.mark_exhausted(86400)
    remaining = key.exhausted_until - time.time()
    assert 0 < remaining <= KeyState.MAX_COOLDOWN_SECONDS + 1
    assert key.exhausted is True

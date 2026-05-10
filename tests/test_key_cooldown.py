"""Regression tests for upstream-key transient cooldown behavior."""

import time

import pytest

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
async def test_402_cooldown_remains_long_quota_exhaustion():
    manager = KeyManager([
        {"token": "tok-1", "label": "key 1", "max_concurrent": 1},
    ])
    key = manager.keys[0]

    await manager.mark_402(key)

    assert key.exhausted is True
    remaining = key.exhausted_until - time.time()
    assert remaining > 23 * 3600

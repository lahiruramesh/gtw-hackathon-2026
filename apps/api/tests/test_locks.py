"""Redis locks: one holder at a time, renewed while held, freed within one TTL when the holder dies."""

from __future__ import annotations

import asyncio

from skf_api.context import AppContext
from skf_api.core.locks import is_locked, try_lock


async def test_lock_is_renewed_while_held_and_released_after(ctx: AppContext) -> None:
    async with try_lock(ctx.redis, "stage:x", ttl_seconds=0.3) as acquired:
        assert acquired
        await asyncio.sleep(1.0)  # three TTLs of work
        assert await is_locked(ctx.redis, "stage:x")
        async with try_lock(ctx.redis, "stage:x", ttl_seconds=0.3) as second:
            assert not second
    assert not await is_locked(ctx.redis, "stage:x")


async def test_a_dead_holder_frees_the_lock_within_its_ttl(ctx: AppContext) -> None:
    holder = ctx.redis.lock("lock:stage:y", timeout=0.3, blocking=False)
    assert await holder.acquire()  # acquired, then the process "dies" (no renewal, no release)
    assert await is_locked(ctx.redis, "stage:y")
    await asyncio.sleep(0.5)
    async with try_lock(ctx.redis, "stage:y", ttl_seconds=0.3) as acquired:
        assert acquired

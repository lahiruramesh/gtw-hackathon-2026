"""Exclusive locks in Redis (one holder across API and worker processes).

A held lock is renewed in the background, so its TTL only has to cover a holder that died: work of any
length keeps the lock, and a crashed holder frees it within one TTL.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockError


def _key(name: str) -> str:
    return f"lock:{name}"


@asynccontextmanager
async def try_lock(redis: Redis, name: str, *, ttl_seconds: float) -> AsyncIterator[bool]:
    """Yields False immediately if someone else holds `name`."""
    lock = redis.lock(_key(name), timeout=ttl_seconds, blocking=False)
    if not await lock.acquire():
        yield False
        return
    renewal = asyncio.create_task(_keep_alive(lock, ttl_seconds / 3))
    try:
        yield True
    finally:
        renewal.cancel()
        with suppress(asyncio.CancelledError):
            await renewal
        # LockError: it expired while held (e.g. Redis was unreachable); the work it guarded is idempotent.
        with suppress(LockError):
            await lock.release()


async def is_locked(redis: Redis, name: str) -> bool:
    return bool(await redis.exists(_key(name)))


async def _keep_alive(lock: Lock, every_seconds: float) -> None:
    while True:
        await asyncio.sleep(every_seconds)
        try:
            await lock.reacquire()
        except LockError:
            return  # lost it; the holder finishes, and its work is idempotent

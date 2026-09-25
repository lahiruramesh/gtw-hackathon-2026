"""Live run events over Redis pub/sub, channel `run:<run_id>` (SPEC §9.3).

A published message is `{"type": ..., "items": [...]}`: the log sink batches many lines into one message,
and subscribers fan it out into one SSE event per item.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
import orjson
from redis.asyncio import Redis
from redis.asyncio.client import PubSub

EventType = Literal["log", "metric", "stage", "run"]


@dataclass(frozen=True)
class RunEvent:
    type: EventType
    data: dict[str, Any]


def channel(run_id: uuid.UUID | str) -> str:
    return f"run:{run_id}"


class EventBus:
    def __init__(self, redis: Redis):
        self._redis = redis

    async def publish(self, run_id: uuid.UUID | str, type: EventType, items: list[dict[str, Any]]) -> None:
        if items:
            payload = orjson.dumps({"type": type, "items": items}, default=str)
            await self._redis.publish(channel(run_id), payload)

    @asynccontextmanager
    async def subscription(self, run_id: uuid.UUID | str) -> AsyncIterator[AsyncIterator[RunEvent]]:
        """Subscribed on enter, so callers can replay history afterwards without missing live events."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel(run_id))
        try:
            yield self._iterate(pubsub)
        finally:
            # Shielded: an SSE client disconnecting cancels the stream, which must not cancel this cleanup.
            with anyio.CancelScope(shield=True):
                await pubsub.unsubscribe()
                await pubsub.aclose()

    @staticmethod
    async def _iterate(pubsub: PubSub) -> AsyncIterator[RunEvent]:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            decoded = orjson.loads(message["data"])
            for item in decoded["items"]:
                yield RunEvent(type=decoded["type"], data=item)

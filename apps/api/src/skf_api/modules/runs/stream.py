"""SSE for one run (SPEC §9.3): replay stored log lines, then live events from Redis.

The Redis subscription is opened *before* the replay query, so a line written in between is delivered
live rather than lost; live log lines at or below the last replayed id are dropped as duplicates. The replay
is read in a detached (cancellation-shielded) session before anything is yielded, so a client that
disconnects at any point never leaves a database connection checked out.
FastAPI's EventSourceResponse adds the `: ping` keep-alive every 15 s. The stream opens with a `: connected`
comment because uvicorn sends the response headers only with the first body chunk: without it a client would
wait for the first event or ping before its EventSource reports the connection as open.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi.sse import ServerSentEvent

from skf_api.context import AppContext
from skf_api.modules.runs import logs

REPLAY_LIMIT = 2000


async def run_events(ctx: AppContext, run_id: uuid.UUID, after_log_id: int) -> AsyncIterator[ServerSentEvent]:
    async with ctx.events.subscription(run_id) as live:
        replay = await ctx.db.detached_read(
            lambda session: logs.page(session, run_id, after_id=after_log_id, limit=REPLAY_LIMIT)
        )
        yield ServerSentEvent(comment="connected")
        last_log_id = after_log_id
        for line in replay.items:
            last_log_id = line.id
            yield ServerSentEvent(event="log", id=str(line.id), data=line.model_dump(mode="json"))
        async for event in live:
            if event.type == "log":
                log_id = int(event.data["id"])
                if log_id <= last_log_id:
                    continue
                last_log_id = log_id
                yield ServerSentEvent(event="log", id=str(log_id), data=event.data)
            else:
                yield ServerSentEvent(event=event.type, data=event.data)

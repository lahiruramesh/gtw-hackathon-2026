"""Background job queue used by the API and the orchestrator. arq in production; tests record and drain."""

from __future__ import annotations

import uuid
from typing import Protocol

from arq.connections import ArqRedis

ADVANCE_RUN = "advance_run"
EXECUTE_STAGE = "execute_stage"
CANCEL_RUN = "cancel_run"


class JobQueue(Protocol):
    async def advance_run(self, run_id: uuid.UUID) -> None: ...
    async def execute_stage(self, stage_id: uuid.UUID, *, defer_seconds: float = 0) -> None: ...
    async def cancel_run(self, run_id: uuid.UUID) -> None: ...


class ArqJobQueue:
    # No `_job_id` de-duplication: every task is idempotent under row locks, and a de-duplicated enqueue
    # issued while the same job is still running would be silently dropped.
    def __init__(self, redis: ArqRedis):
        self._redis = redis

    async def advance_run(self, run_id: uuid.UUID) -> None:
        await self._redis.enqueue_job(ADVANCE_RUN, str(run_id))

    async def execute_stage(self, stage_id: uuid.UUID, *, defer_seconds: float = 0) -> None:
        await self._redis.enqueue_job(EXECUTE_STAGE, str(stage_id), _defer_by=defer_seconds or None)

    async def cancel_run(self, run_id: uuid.UUID) -> None:
        await self._redis.enqueue_job(CANCEL_RUN, str(run_id))

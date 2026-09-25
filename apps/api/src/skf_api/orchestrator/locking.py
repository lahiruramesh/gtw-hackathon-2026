"""Locks shared by the orchestrator's tasks: row locks that re-read the row, and the per-stage Redis lock.

Sessions are created with `expire_on_commit=False`, so a plain `SELECT ... FOR UPDATE` of a row the session
already loaded hands back the identity-mapped object with its old attribute values. Every lock taken after
a slow backend call must see what a concurrent cancel committed meanwhile, hence `populate_existing`.
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.locks import is_locked
from skf_api.modules.runs.models import Run, Stage

# Held (and renewed) while execute_stage submits or the reconciler polls/collects a stage.
STAGE_LOCK_TTL_S = 60


def stage_lock(stage_id: uuid.UUID) -> str:
    return f"stage:{stage_id}"


async def submit_in_flight(redis: Redis, stage_id: uuid.UUID) -> bool:
    """A provisioning stage without an external_ref is still being submitted while its lock is held."""
    return await is_locked(redis, stage_lock(stage_id))


def _fresh(stmt: sa.Select) -> sa.Select:
    return stmt.with_for_update().execution_options(populate_existing=True)


async def lock_run(session: AsyncSession, run_id: uuid.UUID) -> Run | None:
    return await session.scalar(_fresh(sa.select(Run).where(Run.id == run_id)))


async def lock_stage(session: AsyncSession, stage_id: uuid.UUID) -> Stage | None:
    return await session.scalar(_fresh(sa.select(Stage).where(Stage.id == stage_id)))


async def lock_stages(session: AsyncSession, run_id: uuid.UUID) -> list[Stage]:
    stmt = sa.select(Stage).where(Stage.run_id == run_id).order_by(Stage.position, Stage.attempt)
    return list((await session.scalars(_fresh(stmt))).all())

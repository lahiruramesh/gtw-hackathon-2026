"""reconcile() (arq cron, every 20 s, SPEC §6): drive every non-terminal stage toward a terminal state.

Polls backend.status(), pulls logs when the job cannot phone home, accounts GPU time and cost, collects
outputs on success, and repairs what a worker restart can leave behind (queued stages whose job was lost,
provisioning stages whose submit died, runs whose advance_run was lost). Each stage is handled under its
Redis lock, so overlapping ticks and concurrent execute_stage calls never act on the same stage twice.
cancel_run does not take that lock (a cancel must not wait for a slow poll), so every transition re-locks the
stage row after the backend calls and re-checks the stage's status and its run's cancel_requested.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import timedelta

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendError, ComputeBackend, ExternalRef, JobState, JobStatus
from skf_api.context import AppContext
from skf_api.core.db import utcnow
from skf_api.core.locks import try_lock
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.ingest.service import IngestService, Line
from skf_api.modules.runs import transitions
from skf_api.modules.runs.models import ACTIVE_STAGE_STATUSES, Run, RunStatus, Stage, StageStatus
from skf_api.modules.runs.notify import publish_stages
from skf_api.modules.skills.service import get_skill
from skf_api.orchestrator.collector import CollectError, collect_outputs
from skf_api.orchestrator.executor import (
    DEFAULT_TIMEOUT_MINUTES,
    StageSetupError,
    execution_marker,
    phones_home,
    stage_definition,
)
from skf_api.orchestrator.locking import STAGE_LOCK_TTL_S, lock_stage, stage_lock
from skf_api.orchestrator.state_machine import cancel_retry_marker, release_if_idle
from skf_api.orchestrator.workdir import discard_stage_dir, reset_dir, stage_dir
from skf_api.skills_registry.manifest import StageDef

log = logging.getLogger(__name__)

HEARTBEAT_STALE = timedelta(minutes=10)
TIMEOUT_GRACE = timedelta(minutes=30)
REQUEUE_AFTER_S = 30
PARALLEL_STAGES = 8
MAX_LOG_PULLS_PER_TICK = 20
RUNNING_STATUSES = (StageStatus.PROVISIONING, StageStatus.RUNNING)
POLLED_STATUSES = (*RUNNING_STATUSES, StageStatus.COLLECTING)
EXECUTING_RUN_STATUSES = (RunStatus.QUEUED, RunStatus.RUNNING)


async def reconcile(ctx: AppContext) -> None:
    async with ctx.db.session() as session:
        stage_ids = list(
            await session.scalars(
                sa.select(Stage.id).where(Stage.status.in_(ACTIVE_STAGE_STATUSES)).order_by(Stage.started_at)
            )
        )
    limit = asyncio.Semaphore(PARALLEL_STAGES)

    async def one(stage_id: uuid.UUID) -> None:
        async with limit:
            try:
                await reconcile_stage(ctx, stage_id)
            except Exception:
                log.exception("reconcile failed", extra={"stage_id": str(stage_id)})

    await asyncio.gather(*(one(i) for i in stage_ids))
    await sweep_runs(ctx)


async def reconcile_stage(ctx: AppContext, stage_id: uuid.UUID) -> None:
    async with ctx.db.session() as session:
        stage = await session.get(Stage, stage_id)
        if stage is None or not stage.status.active:
            return
        if stage.status is StageStatus.QUEUED:
            # A lost execute_stage job (worker restart, Redis flush) is re-enqueued; while the marker is set
            # a job is pending, waiting for a slot, or backing off after a retryable submit error.
            if await ctx.redis.set(execution_marker(stage.id), 1, nx=True, ex=REQUEUE_AFTER_S):
                await ctx.jobs.execute_stage(stage.id)
            return
    async with try_lock(ctx.redis, stage_lock(stage_id), ttl_seconds=STAGE_LOCK_TTL_S) as acquired:
        if acquired:
            async with ctx.db.session() as session:
                await _reconcile_locked(ctx, session, stage_id)


async def _reconcile_locked(ctx: AppContext, session: AsyncSession, stage_id: uuid.UUID) -> None:
    stage = await session.get(Stage, stage_id)
    if stage is None or stage.status not in POLLED_STATUSES:
        return
    run = await session.get(Run, stage.run_id)
    target = await session.get(ComputeTarget, stage.compute_target_id) if stage.compute_target_id else None
    if run is None or target is None:
        await _finish_failed(ctx, session, stage.id, "The stage's compute target no longer exists")
        return
    if run.cancel_requested and not run.status.finished:
        await request_cancel(ctx, run.id)
        return
    if stage.external_ref is None:
        # We hold the stage lock, so no submit is in flight: the one that claimed this stage died.
        locked = await _relock(session, stage_id, StageStatus.PROVISIONING)
        if locked is None:
            return
        locked.status = StageStatus.QUEUED
        locked.message = "Re-queued: the worker stopped while submitting"
        await session.commit()
        await ctx.jobs.execute_stage(locked.id)
        await publish_stages(session, ctx.events, [locked])
        return

    _, manifest = await get_skill(session, run.skill_id)
    try:
        stage_def = stage_definition(
            manifest.stage(transitions.side_stage_def_id(stage.key) if stage.is_side_stage else stage.key),
            stage,
        )
    except StageSetupError as exc:
        await _finish_failed(ctx, session, stage.id, str(exc))
        return
    backend = ctx.backends.for_target(target)
    ref = ExternalRef.from_json(stage.external_ref)
    ingest = IngestService(ctx.redis, ctx.events, ctx.settings.log_max_lines_per_run)

    if stage.status is StageStatus.COLLECTING:
        await _collect(ctx, session, ingest, backend, run, stage, target, stage_def, ref)
        return

    try:
        status = await backend.status(ref)
    except BackendError as exc:
        locked = await _relock(session, stage_id, *RUNNING_STATUSES)
        if locked is not None:
            locked.message = f"Status check failed: {exc}"
            await session.commit()
        return
    if status.ref is not None:
        ref = status.ref
    if not phones_home(ctx, target):
        await _pull_logs(session, ingest, backend, stage, ref)

    def record(locked: Stage) -> None:
        if status.ref is not None:
            locked.external_ref = ref.to_json()
        _account(locked, target, status)

    if status.state in (JobState.FAILED, JobState.CANCELLED):
        reason = status.message or (
            "Cancelled outside the studio" if status.state is JobState.CANCELLED else "The job failed"
        )
        await _finish_failed(ctx, session, stage.id, reason, update=record)
        return
    timeout = timedelta(minutes=stage_def.timeout_minutes or DEFAULT_TIMEOUT_MINUTES)
    if stage.started_at and utcnow() - stage.started_at > timeout + TIMEOUT_GRACE:
        await backend.cancel(ref)
        await _finish_failed(
            ctx, session, stage.id, f"Timed out after {timeout.total_seconds() / 60:.0f} min", update=record
        )
        return

    # The backend calls above can take a while (SSH round trips); a cancel may have landed meanwhile.
    locked = await _relock(session, stage_id, *RUNNING_STATUSES)
    if locked is None:
        return
    record(locked)
    if status.state is JobState.SUCCEEDED:
        locked.status = StageStatus.COLLECTING
        locked.message = "Collecting outputs"
        await session.commit()
        await publish_stages(session, ctx.events, [locked])
        await _collect(ctx, session, ingest, backend, run, locked, target, stage_def, ref)
        return
    if status.state is JobState.RUNNING and locked.status is StageStatus.PROVISIONING:
        locked.status = StageStatus.RUNNING
    heartbeat = locked.last_heartbeat_at or locked.started_at
    if (
        phones_home(ctx, target)
        and locked.status is StageStatus.RUNNING
        and heartbeat
        and utcnow() - heartbeat > HEARTBEAT_STALE
    ):
        minutes = (utcnow() - heartbeat).total_seconds() / 60
        locked.message = f"Stalled: no heartbeat for {minutes:.0f} min (backend reports {status.state.value})"
    elif status.message:
        locked.message = status.message
    changed = bool(session.dirty)
    await session.commit()
    if changed:
        await publish_stages(session, ctx.events, [locked])


async def _relock(session: AsyncSession, stage_id: uuid.UUID, *expected: StageStatus) -> Stage | None:
    """The stage row, locked and re-read, if it is still in one of `expected` and its run is not being
    cancelled. Otherwise commits what the tick already did (e.g. the log cursor) and returns None."""
    stage = await lock_stage(session, stage_id)
    if stage is not None and stage.status in expected:
        cancelling = await session.scalar(sa.select(Run.cancel_requested).where(Run.id == stage.run_id))
        if not cancelling:
            return stage
    await session.commit()
    return None


async def request_cancel(ctx: AppContext, run_id: uuid.UUID) -> None:
    """Re-enqueue cancel_run for a run whose cancel is unfinished (lost job, or a teardown retry is due)."""
    if await ctx.redis.set(cancel_retry_marker(run_id), 1, nx=True, ex=REQUEUE_AFTER_S):
        await ctx.jobs.cancel_run(run_id)


def _account(stage: Stage, target: ComputeTarget, status: JobStatus) -> None:
    if status.gpu_seconds is not None:
        stage.gpu_seconds = status.gpu_seconds
        stage.cost = round(status.gpu_seconds / 3600.0 * target.cost_per_gpu_hour, 4)
    if status.progress is not None:
        stage.progress = status.progress


async def _pull_logs(
    session: AsyncSession, ingest: IngestService, backend: ComputeBackend, stage: Stage, ref: ExternalRef
) -> None:
    """Drain the job's log through fetch_logs (bounded per tick; the rest comes on the next tick)."""
    for _ in range(MAX_LOG_PULLS_PER_TICK):
        batch = await backend.fetch_logs(ref, stage.log_cursor)
        if batch.lines:
            await ingest.append_logs(session, stage, [Line(text=t) for t in batch.lines])
        advanced = batch.cursor != stage.log_cursor
        stage.log_cursor = batch.cursor
        if not batch.lines or not advanced:
            return


async def _collect(
    ctx: AppContext,
    session: AsyncSession,
    ingest: IngestService,
    backend: ComputeBackend,
    run: Run,
    stage: Stage,
    target: ComputeTarget,
    stage_def: StageDef,
    ref: ExternalRef,
) -> None:
    stage_id = stage.id
    out_dir = reset_dir(stage_dir(ctx, stage) / "out")
    try:
        await backend.collect(ref, out_dir)
        if not phones_home(ctx, target):
            await _pull_logs(session, ingest, backend, stage, ref)
        count = await collect_outputs(ctx, session, ingest, run, stage, stage_def, out_dir)
    except (BackendError, CollectError) as exc:
        await session.rollback()
        await _finish_failed(ctx, session, stage_id, f"Collecting outputs failed: {exc}")
        return
    # A run cancelled while its outputs were being collected keeps them, but the stage stays cancelled.
    locked = await _relock(session, stage_id, StageStatus.COLLECTING)
    if locked is None:
        return
    transitions.finish_stage(locked, StageStatus.SUCCEEDED)
    locked.message = f"{count} output files stored"
    await session.commit()
    await publish_stages(session, ctx.events, [locked])
    await _after_finish(ctx, session, locked)


async def _finish_failed(
    ctx: AppContext,
    session: AsyncSession,
    stage_id: uuid.UUID,
    error: str,
    *,
    update: Callable[[Stage], None] | None = None,
) -> None:
    locked = await lock_stage(session, stage_id)
    if locked is None or locked.status.terminal:
        await session.commit()
        return
    if update is not None:
        update(locked)
    transitions.finish_stage(locked, StageStatus.FAILED, error=error)
    await session.commit()
    await publish_stages(session, ctx.events, [locked])
    await _after_finish(ctx, session, locked)


async def _after_finish(ctx: AppContext, session: AsyncSession, stage: Stage) -> None:
    # advance_run first, so a worker restart during the slow release (SSH, stop_instances) cannot lose it.
    if not stage.is_side_stage:
        await ctx.jobs.advance_run(stage.run_id)
    await discard_stage_dir(ctx, stage)
    await release_if_idle(ctx, session, stage.compute_target_id)


async def sweep_runs(ctx: AppContext) -> None:
    """Re-enqueue advance_run for executing runs with no active main stage. Their advance_run was lost (worker
    restart between a stage finishing and the enqueue, or Redis unreachable when the API enqueued it), and
    nothing else would move them on. advance_run is idempotent, so an extra call for a moving run is harmless.
    """
    active_main_stage = sa.exists().where(
        Stage.run_id == Run.id, Stage.status.in_(ACTIVE_STAGE_STATUSES), Stage.checkpoint_id.is_(None)
    )
    async with ctx.db.session() as session:
        run_ids = list(
            await session.scalars(
                sa.select(Run.id).where(Run.status.in_(EXECUTING_RUN_STATUSES), ~active_main_stage)
            )
        )
    for run_id in run_ids:
        await ctx.jobs.advance_run(run_id)

"""Run-level transitions (SPEC §6): start the next stage, evaluate the gate inline, fail, cancel.

Every function locks the run row (`SELECT ... FOR UPDATE`), re-reads state and decides from it, so running
one twice, or concurrently, or after a worker restart, converges to the same result.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendError, ExternalRef
from skf_api.context import AppContext
from skf_api.core.db import utcnow
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.gates.service import evaluate_gate
from skf_api.modules.runs import transitions
from skf_api.modules.runs.models import (
    ACTIVE_STAGE_STATUSES,
    OCCUPYING_STAGE_STATUSES,
    RunStatus,
    Stage,
    StageKind,
    StageStatus,
)
from skf_api.modules.runs.notify import publish_run, publish_stages
from skf_api.modules.runs.views import main_stages
from skf_api.modules.skills.service import get_skill
from skf_api.orchestrator.locking import lock_run, lock_stages, submit_in_flight
from skf_api.orchestrator.workdir import discard_stage_dir

log = logging.getLogger(__name__)

MAX_CANCEL_RETRIES = 10
CANCEL_RETRY_BASE_S = 30
CANCEL_RETRY_MAX_S = 600


async def advance_run(ctx: AppContext, run_id: uuid.UUID) -> None:
    to_execute: uuid.UUID | None = None
    async with ctx.db.session() as session:
        run = await lock_run(session, run_id)
        if run is None or not run.status.executing:
            return
        if run.cancel_requested:
            await session.rollback()
            await ctx.jobs.cancel_run(run_id)
            return
        stages = main_stages(await lock_stages(session, run.id))
        changed: list[Stage] = []
        failed = next((s for s in stages if s.status in (StageStatus.FAILED, StageStatus.CANCELLED)), None)
        if failed is not None:
            changed = [s for s in stages if s.status is StageStatus.PENDING]
            transitions.fail_run(
                run, changed, failed.error or f"Stage '{failed.title}' {failed.status.value}"
            )
        elif not any(s.status in ACTIVE_STAGE_STATUSES for s in stages):
            nxt = next((s for s in stages if s.status is StageStatus.PENDING), None)
            if nxt is None:
                log.warning("run has no pending stage but is still executing", extra={"run_id": str(run.id)})
                return
            changed = [nxt]
            run.status = RunStatus.RUNNING
            run.started_at = run.started_at or utcnow()
            if nxt.kind is StageKind.GATE:
                _, manifest = await get_skill(session, run.skill_id)
                nxt.started_at = utcnow()
                decision = await evaluate_gate(session, run, manifest)
                passed = sum(c["passed"] for c in decision.criteria)
                nxt.message = f"{decision.verdict.value}: {passed}/{len(decision.criteria)} criteria met"
                transitions.finish_stage(nxt, StageStatus.SUCCEEDED)
            else:
                nxt.status = StageStatus.QUEUED
                to_execute = nxt.id
        elif run.status is RunStatus.QUEUED:
            run.status = RunStatus.RUNNING
            run.started_at = run.started_at or utcnow()
        else:
            return
        await session.commit()
        if to_execute is not None:
            await ctx.jobs.execute_stage(to_execute)
        await publish_stages(session, ctx.events, changed)
        await publish_run(session, ctx.events, run)


async def cancel_run(ctx: AppContext, run_id: uuid.UUID) -> None:
    """Tear down remote jobs first (idempotent backend.cancel), then mark stages and the run cancelled.

    A stage is marked cancelled only once nothing can still be running for it: its backend.cancel succeeded
    (or failed for good), or it never reached a backend. Until then it stays active with the run's
    cancel_requested set: a submit still in flight records its job and enqueues cancel_run again, and after a
    retryable teardown failure the reconciler enqueues cancel_run again once the backoff marker expires.
    """
    async with ctx.db.session() as session:
        run = await lock_run(session, run_id)
        if run is None or run.status.finished:
            return
        run.cancel_requested = True
        remote = [
            (s.id, s.compute_target_id, ExternalRef.from_json(s.external_ref))
            for s in await lock_stages(session, run.id)
            if s.status in OCCUPYING_STAGE_STATUSES and s.external_ref and s.compute_target_id
        ]
        await session.commit()

    teardown = _Teardown()
    for stage_id, target_id, ref in remote:
        error = await _tear_down(ctx, run_id, target_id, ref)
        if error is None:
            teardown.stopped[stage_id] = ref.job_id
        else:
            teardown.failed[stage_id] = error
    backoff = 0
    if teardown.failed:
        attempt = await _count_cancel_attempt(ctx, run_id)
        if attempt > MAX_CANCEL_RETRIES:
            errors = "; ".join(teardown.failed.values())
            log.error("giving up stopping remote jobs: %s", errors, extra={"run_id": str(run_id)})
            teardown.abandoned, teardown.failed = teardown.failed, {}
        backoff = min(CANCEL_RETRY_BASE_S * 2 ** (attempt - 1), CANCEL_RETRY_MAX_S)

    async with ctx.db.session() as session:
        run = await lock_run(session, run_id)
        if run is None or run.status.finished:
            return
        stages = await lock_stages(session, run.id)
        waiting = [s for s in stages if await teardown.pending(ctx, s)]
        for stage in waiting:
            stage.message = (
                f"Stopping the job failed: {teardown.failed[stage.id]}; retrying"
                if stage.id in teardown.failed
                else "Cancelling once the job is submitted"
            )
        done = [s for s in stages if s not in waiting and not s.status.terminal]
        if waiting:
            transitions.cancel_stages(done)
        else:
            transitions.cancel_run(run, stages)
        for stage in done:
            if stage.id in teardown.abandoned:
                stage.error = f"The job could not be stopped: {teardown.abandoned[stage.id]}"
        await session.commit()
        if teardown.failed:
            await ctx.redis.set(cancel_retry_marker(run_id), 1, ex=backoff)
        await publish_stages(session, ctx.events, [*waiting, *done])
        await publish_run(session, ctx.events, run)
        for stage in done:
            await discard_stage_dir(ctx, stage)
        for target_id in {s.compute_target_id for s in done if s.status is StageStatus.CANCELLED}:
            await release_if_idle(ctx, session, target_id)


@dataclass
class _Teardown:
    """Outcome of one cancel_run's backend.cancel calls, by stage id."""

    stopped: dict[uuid.UUID, str] = field(default_factory=dict)  # the job id that was stopped
    failed: dict[uuid.UUID, str] = field(default_factory=dict)  # retryable error: try again later
    abandoned: dict[uuid.UUID, str] = field(default_factory=dict)  # error after the last retry

    async def pending(self, ctx: AppContext, stage: Stage) -> bool:
        """Whether the stage may still have a job that is not torn down (so it must stay active)."""
        if stage.status not in OCCUPYING_STAGE_STATUSES or stage.id in self.abandoned:
            return False
        if stage.id in self.failed:
            return True
        if stage.external_ref is None:
            return await submit_in_flight(ctx.redis, stage.id)
        # A job recorded after the teardown pass (a submit that just finished) is not stopped yet.
        return self.stopped.get(stage.id) != ExternalRef.from_json(stage.external_ref).job_id


def cancel_retry_marker(run_id: uuid.UUID) -> str:
    """While set, the reconciler does not enqueue cancel_run for this run (teardown retry backoff)."""
    return f"cancel-pending:{run_id}"


async def _count_cancel_attempt(ctx: AppContext, run_id: uuid.UUID) -> int:
    key = f"cancel-retries:{run_id}"
    attempt = int(await ctx.redis.incr(key))
    await ctx.redis.expire(key, 24 * 3600)
    return attempt


async def _tear_down(
    ctx: AppContext, run_id: uuid.UUID, target_id: uuid.UUID, ref: ExternalRef
) -> str | None:
    """backend.cancel(ref); returns the error if it failed in a way worth retrying."""
    async with ctx.db.session() as session:
        target = await session.get(ComputeTarget, target_id)
    if target is None:
        return None
    extra = {"run_id": str(run_id), "job_id": ref.job_id}
    try:
        await ctx.backends.for_target(target).cancel(ref)
    except BackendError as exc:
        if exc.retryable:
            log.warning("backend cancel failed, will retry: %s", exc, extra=extra)
            return str(exc)
        log.error("backend cancel failed: %s", exc, extra=extra)
    except Exception:
        log.exception("backend cancel failed", extra=extra)
    return None


async def release_if_idle(ctx: AppContext, session: AsyncSession, target_id: uuid.UUID | None) -> None:
    """backend.release() once nothing is running or waiting on the target (e.g. stop the AWS box)."""
    if target_id is None:
        return
    busy = await session.scalar(
        sa.select(sa.func.count())
        .select_from(Stage)
        .where(
            Stage.compute_target_id == target_id,
            Stage.status.in_((*OCCUPYING_STAGE_STATUSES, StageStatus.QUEUED)),
        )
    )
    target = await session.get(ComputeTarget, target_id)
    if busy or target is None:
        return
    try:
        await ctx.backends.for_target(target).release()
    except Exception:
        log.exception("backend release failed", extra={"target_id": str(target_id)})

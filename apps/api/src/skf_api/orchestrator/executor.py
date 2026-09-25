"""execute_stage (SPEC §6): claim a queued stage, build its JobSpec, stage the inputs, submit to the backend.

The claim (queued -> provisioning) commits before the slow submit, so no DB transaction stays open while an
instance boots. A Redis lock held for the whole call tells the reconciler (and cancel_run) that a
provisioning stage without an external_ref is still being submitted; if the lock is free, the submit died
and the stage is re-queued.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendError, BackendKind, ExternalRef, IngestConfig, InputFile, JobSpec
from skf_api.context import AppContext
from skf_api.core.db import utcnow
from skf_api.core.locks import try_lock
from skf_api.modules.artifacts.models import Artifact, artifact_key
from skf_api.modules.artifacts.service import CHECKPOINT_KINDS
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.runs import transitions
from skf_api.modules.runs.models import OCCUPYING_STAGE_STATUSES, Run, Stage, StageStatus
from skf_api.modules.runs.notify import publish_stages
from skf_api.modules.skills.service import get_skill
from skf_api.orchestrator.locking import STAGE_LOCK_TTL_S, lock_stage, stage_lock
from skf_api.orchestrator.state_machine import release_if_idle
from skf_api.orchestrator.workdir import discard_stage_dir, stage_dir
from skf_api.skills_registry.manifest import StageDef
from skf_api.skills_registry.params import checkpoint_params
from skf_api.skills_registry.render import (
    ArtifactInput,
    CheckpointFile,
    RenderError,
    StageInputRef,
    StageOutputInput,
    render_stage,
)

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MINUTES = 24 * 60
BUSY_RETRY_S = 30
MAX_SUBMIT_RETRIES = 8
# Files a pipeline loader reads next to a checkpoint (obs sizes, env config).
CHECKPOINT_SIBLINGS = ("params.pkl", "config.json")


class StageSetupError(Exception):
    """The stage cannot be started as configured (missing input, unrenderable command)."""


def execution_marker(stage_id: uuid.UUID) -> str:
    """While set, the reconciler does not re-enqueue execute_stage for this queued stage."""
    return f"exec-pending:{stage_id}"


def phones_home(ctx: AppContext, target: ComputeTarget) -> bool:
    return target.kind is BackendKind.LOCAL_CPU or ctx.settings.public_ingest_url is not None


def stage_definition(manifest_stage: StageDef | None, stage: Stage) -> StageDef:
    if manifest_stage is None:
        raise StageSetupError(f"Stage '{stage.key}' is no longer in the skill manifest")
    return manifest_stage


async def execute_stage(ctx: AppContext, stage_id: uuid.UUID) -> None:
    async with try_lock(ctx.redis, stage_lock(stage_id), ttl_seconds=STAGE_LOCK_TTL_S) as acquired:
        if acquired:
            await _execute_locked(ctx, stage_id)


async def _execute_locked(ctx: AppContext, stage_id: uuid.UUID) -> None:
    async with ctx.db.session() as session:
        claimed = await _claim(ctx, session, stage_id)
        if claimed is None:
            return
        stage, run, target = claimed
        try:
            spec = await build_job_spec(ctx, session, run, stage, target)
            ref = await ctx.backends.for_target(target).submit(spec)
        except (StageSetupError, RenderError) as exc:
            await _fail(ctx, session, stage_id, str(exc))
            return
        except BackendError as exc:
            await _submit_failed(ctx, session, stage_id, exc)
            return
        except Exception as exc:
            log.exception("submit failed", extra={"stage_id": str(stage_id)})
            await _fail(
                ctx, session, stage_id, f"Internal error while starting the job: {type(exc).__name__}"
            )
            return
        await _record_submission(ctx, session, stage_id, target, ref)


async def _claim(
    ctx: AppContext, session: AsyncSession, stage_id: uuid.UUID
) -> tuple[Stage, Run, ComputeTarget] | None:
    stage = await lock_stage(session, stage_id)
    if stage is None or stage.status is not StageStatus.QUEUED:
        return None
    run = await session.get(Run, stage.run_id)
    if run is None or run.cancel_requested:
        return None
    target = (
        await session.scalar(
            sa.select(ComputeTarget).where(ComputeTarget.id == stage.compute_target_id).with_for_update()
        )
        if stage.compute_target_id
        else None
    )
    if target is None or not target.enabled:
        await _fail(ctx, session, stage.id, "The stage's compute target is missing or disabled")
        return None
    # The target row lock serialises this count across workers claiming stages on the same target.
    busy = await session.scalar(
        sa.select(sa.func.count())
        .select_from(Stage)
        .where(Stage.compute_target_id == target.id, Stage.status.in_(OCCUPYING_STAGE_STATUSES))
    )
    if (busy or 0) >= target.max_concurrent:
        message = f"Waiting for a free slot on {target.name}"
        if stage.message != message:
            stage.message = message
            await session.commit()
            await publish_stages(session, ctx.events, [stage])
        else:
            await session.rollback()
        await ctx.redis.set(execution_marker(stage.id), 1, ex=BUSY_RETRY_S)
        return None
    stage.status = StageStatus.PROVISIONING
    stage.message = f"Submitting to {target.name}"
    stage.started_at = utcnow()
    stage.error = None
    await session.commit()
    await publish_stages(session, ctx.events, [stage])
    return stage, run, target


async def _record_submission(
    ctx: AppContext, session: AsyncSession, stage_id: uuid.UUID, target: ComputeTarget, ref: ExternalRef
) -> None:
    stage = await lock_stage(session, stage_id)
    if stage is None or stage.status is not StageStatus.PROVISIONING:
        # Finished while submitting (cancelled or failed elsewhere): the job just started must not outlive it.
        await session.commit()  # releases the row lock; a rollback would expire `target`
        await _abandon(ctx, session, target, ref)
        return
    stage.external_ref = ref.to_json()
    stage.message = "Submitted"
    # Read, not locked: cancel_run locks the run before its stages, so locking the run here could deadlock.
    # Holding the stage lock is enough; a cancel that committed before it is visible to this read.
    cancel_requested = await session.scalar(sa.select(Run.cancel_requested).where(Run.id == stage.run_id))
    await session.commit()
    await ctx.redis.delete(f"submit-retries:{stage.id}")
    await publish_stages(session, ctx.events, [stage])
    if cancel_requested:
        # cancel_run left this stage to us while the submit was in flight; now it has a job to tear down.
        await ctx.jobs.cancel_run(stage.run_id)


async def _abandon(ctx: AppContext, session: AsyncSession, target: ComputeTarget, ref: ExternalRef) -> None:
    try:
        await ctx.backends.for_target(target).cancel(ref)
    except Exception:
        log.exception("cancelling an orphaned job failed", extra={"job_id": ref.job_id})
    await release_if_idle(ctx, session, target.id)


async def _submit_failed(
    ctx: AppContext, session: AsyncSession, stage_id: uuid.UUID, exc: BackendError
) -> None:
    retries = int(await ctx.redis.incr(f"submit-retries:{stage_id}"))
    await ctx.redis.expire(f"submit-retries:{stage_id}", 24 * 3600)
    if not exc.retryable or retries > MAX_SUBMIT_RETRIES:
        await _fail(ctx, session, stage_id, str(exc))
        return
    delay = min(60 * 2 ** (retries - 1), 900)
    await session.rollback()
    locked = await lock_stage(session, stage_id)
    if locked is None or locked.status is not StageStatus.PROVISIONING:
        return
    locked.status = StageStatus.QUEUED
    locked.message = f"{exc} (retry {retries}/{MAX_SUBMIT_RETRIES} in {delay // 60} min)"
    await session.commit()
    await ctx.redis.set(execution_marker(stage_id), 1, ex=delay)
    await publish_stages(session, ctx.events, [locked])


async def _fail(ctx: AppContext, session: AsyncSession, stage_id: uuid.UUID, error: str) -> None:
    await session.rollback()
    locked = await lock_stage(session, stage_id)
    if locked is None or locked.status.terminal:
        return
    transitions.finish_stage(locked, StageStatus.FAILED, error=error)
    await session.commit()
    await publish_stages(session, ctx.events, [locked])
    if not locked.is_side_stage:
        await ctx.jobs.advance_run(locked.run_id)
    await discard_stage_dir(ctx, locked)
    # submit may have started the target (booted a box) before it failed.
    await release_if_idle(ctx, session, locked.compute_target_id)


# ---------------------------------------------------------------------------------------------- JobSpec


async def build_job_spec(
    ctx: AppContext, session: AsyncSession, run: Run, stage: Stage, target: ComputeTarget
) -> JobSpec:
    _, manifest = await get_skill(session, run.skill_id)
    evaluate_checkpoint = None
    if stage.checkpoint_id is not None:
        stage_def = stage_definition(manifest.stage(transitions.side_stage_def_id(stage.key)), stage)
        ckpt = await session.get(Artifact, stage.checkpoint_id)
        if ckpt is None:
            raise StageSetupError("The checkpoint to evaluate no longer exists")
        evaluate_checkpoint = CheckpointFile(ckpt.id, PurePosixPath(ckpt.name).name)
    else:
        stage_def = stage_definition(manifest.stage(stage.key), stage)

    warm_starts = {}
    for param, artifact_id in checkpoint_params(manifest, run.params).items():
        artifact = await session.get(Artifact, artifact_id)
        if artifact is None:
            raise StageSetupError(f"Warm-start checkpoint for '{param}' no longer exists")
        warm_starts[param] = CheckpointFile(artifact.id, PurePosixPath(artifact.name).name)

    rendered = render_stage(manifest, stage_def, run.params, warm_starts, evaluate_checkpoint)
    inputs = await stage_inputs(ctx, session, run, rendered.inputs, stage_dir(ctx, stage) / "in")
    timeout = stage_def.timeout_minutes or DEFAULT_TIMEOUT_MINUTES
    ingest = None
    url = (
        ctx.settings.internal_ingest_url
        if target.kind is BackendKind.LOCAL_CPU
        else ctx.settings.public_ingest_url
    )
    if url:
        expires = int(time.time() + timedelta(minutes=timeout, hours=1).total_seconds())
        ingest = IngestConfig(url=url, token=ctx.ingest_tokens.mint(stage.id, expires))
    return JobSpec(
        run_id=str(run.id),
        run_name=run.name,
        stage_id=str(stage.id),
        stage_key=stage.key,
        kind=stage.kind.value,
        argv=rendered.argv,
        inputs=inputs,
        ingest=ingest,
        progress_csv=stage_def.progress_csv,
        git_sha=run.git_sha,
        timeout_minutes=timeout,
    )


async def _producing_ref(session: AsyncSession, stage_id: uuid.UUID | None) -> ExternalRef | None:
    stage = await session.get(Stage, stage_id) if stage_id else None
    return ExternalRef.from_json(stage.external_ref) if stage and stage.external_ref else None


async def stage_inputs(
    ctx: AppContext, session: AsyncSession, run: Run, refs: list[StageInputRef], in_dir: Path
) -> list[InputFile]:
    """Downloads every input from storage to `in_dir/<name>` and describes it for the backend."""
    wanted: dict[str, Artifact] = {}
    for ref in refs:
        if isinstance(ref, StageOutputInput):
            artifact = await session.scalar(
                sa.select(Artifact).where(
                    Artifact.run_id == run.id,
                    Artifact.uri == artifact_key(run.id, ref.stage_key, ref.filename),
                )
            )
            if artifact is None:
                raise StageSetupError(f"Input '{ref.name}' was not produced by stage '{ref.stage_key}'")
            wanted[ref.name] = artifact
        elif isinstance(ref, ArtifactInput):
            artifact = await session.get(Artifact, ref.artifact_id)
            if artifact is None:
                raise StageSetupError(f"Input '{ref.name}' no longer exists")
            wanted[ref.name] = artifact
            if artifact.kind in CHECKPOINT_KINDS:
                folder = PurePosixPath(ref.name).parent
                for sibling in await _siblings(session, artifact):
                    wanted.setdefault(str(folder / PurePosixPath(sibling.name).name), sibling)

    inputs = []
    for name, artifact in wanted.items():
        local = in_dir / name
        if not (local.is_file() and local.stat().st_size == artifact.size_bytes):
            await ctx.storage.download_file(artifact.uri, local)
        inputs.append(
            InputFile(
                name=name, local_path=local, source_ref=await _producing_ref(session, artifact.stage_id)
            )
        )
    return inputs


async def _siblings(session: AsyncSession, artifact: Artifact) -> list[Artifact]:
    folder = PurePosixPath(artifact.uri).parent
    uris = [str(folder / name) for name in CHECKPOINT_SIBLINGS]
    return list(
        (
            await session.scalars(
                sa.select(Artifact).where(
                    Artifact.run_id == artifact.run_id, Artifact.uri.in_(uris), Artifact.id != artifact.id
                )
            )
        ).all()
    )

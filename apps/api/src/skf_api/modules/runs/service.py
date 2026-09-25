"""Run lifecycle commands from users: estimate, create, cancel, retry, launch approval, release review,
evaluate a checkpoint. Execution itself belongs to the orchestrator; this module only records intent,
applies the permission rules (SPEC §4) and enqueues work."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendKind
from skf_api.context import AppContext
from skf_api.core.auth import Principal
from skf_api.core.db import utcnow
from skf_api.core.errors import Conflict, Forbidden, Invalid, NotFound
from skf_api.core.pagination import Keyset, PageParams
from skf_api.core.permissions import Permission
from skf_api.modules.artifacts.models import Artifact
from skf_api.modules.artifacts.service import CHECKPOINT_KINDS, artifacts_by_id
from skf_api.modules.audit import service as audit
from skf_api.modules.compute.backends import KINDS_NEEDING_SECRET
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.compute.service import Usage, get_target, usage_by_target
from skf_api.modules.gates.models import GateDecision, GateVerdict, ReviewStatus
from skf_api.modules.runs import policy, schemas, transitions
from skf_api.modules.runs.estimate import EstimateInput, estimate_run, is_smoke
from skf_api.modules.runs.models import Evaluation, Run, RunsOn, RunStatus, Stage, StageKind, StageStatus
from skf_api.modules.runs.notify import publish_run, publish_stages
from skf_api.modules.runs.views import main_stages, run_detail, run_summaries
from skf_api.modules.skills.models import Skill, SkillStatus
from skf_api.modules.skills.service import get_skill
from skf_api.skills_registry.manifest import Manifest, ParamType
from skf_api.skills_registry.params import ParamsError, checkpoint_params, validate_params


async def get_run(session: AsyncSession, run_id: uuid.UUID, *, lock: bool = False) -> Run:
    stmt = sa.select(Run).where(Run.id == run_id)
    if lock:
        stmt = stmt.with_for_update()
    run = await session.scalar(stmt)
    if run is None:
        raise NotFound("Run not found")
    return run


# ---------------------------------------------------------------------------------------------- listing

_KEYSET = Keyset(Run.created_at, Run.id)


async def list_runs(
    session: AsyncSession,
    page: PageParams,
    principal: Principal,
    *,
    skill_id: str | None,
    status: RunStatus | None,
    created_by_me: bool,
    q: str | None,
) -> schemas.RunPage:
    stmt = sa.select(Run)
    if skill_id:
        stmt = stmt.where(Run.skill_id == skill_id)
    if status:
        stmt = stmt.where(Run.status == status)
    if created_by_me:
        stmt = stmt.where(Run.created_by_id == principal.id)
    if q:
        escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        stmt = stmt.where(sa.or_(Run.name.ilike(pattern, escape="\\"), Run.notes.ilike(pattern, escape="\\")))
    rows = (await session.scalars(_KEYSET.apply(stmt, page))).all()
    return schemas.RunPage(
        items=await run_summaries(session, rows[: page.limit]),
        next_cursor=_KEYSET.next_cursor(rows, page, lambda r: (r.created_at, r.id)),
    )


# ---------------------------------------------------------------------------------------------- launch


@dataclass(frozen=True)
class LaunchPlan:
    skill: Skill
    manifest: Manifest
    params: dict[str, Any]
    preset_id: str | None
    from_preset: bool  # params equal the preset's: an operator may launch it
    target: ComputeTarget
    usage: Usage
    parent_run_id: uuid.UUID | None
    parent_checkpoint: Artifact | None
    estimate: schemas.Estimate


def _validated(manifest: Manifest, params: dict[str, Any]) -> dict[str, Any]:
    try:
        return validate_params(manifest, params)
    except ParamsError as exc:
        raise Invalid("Invalid params", details={"params": exc.errors}) from exc


async def _resolve_checkpoint(
    session: AsyncSession, manifest: Manifest, params: dict[str, Any], body: schemas.EstimateRequest
) -> tuple[dict[str, Any], Artifact | None]:
    """Warm start: the checkpoint param and `parent_checkpoint_id` are two ways to say the same thing."""
    ckpt_params = [n for n, s in manifest.params.items() if s.type is ParamType.CHECKPOINT]
    body_ckpt = body.parent_checkpoint_id if isinstance(body, schemas.RunCreate) else None
    chosen = checkpoint_params(manifest, params)
    if body_ckpt is not None:
        if not ckpt_params:
            raise Invalid(f"Skill '{manifest.id}' does not support warm starts")
        name = ckpt_params[0]
        if name in chosen and chosen[name] != body_ckpt:
            raise Invalid(f"parent_checkpoint_id and params.{name} disagree")
        chosen[name] = body_ckpt
        params = {**params, name: str(body_ckpt)}
    if len(chosen) > 1:
        raise Invalid("Only one warm-start checkpoint is supported per run")
    if not chosen:
        return params, None
    artifact = await session.get(Artifact, next(iter(chosen.values())))
    if artifact is None or artifact.kind not in CHECKPOINT_KINDS:
        raise Invalid(
            "Warm-start checkpoint not found", details={"params": {next(iter(chosen)): "not found"}}
        )
    return params, artifact


async def plan_launch(
    ctx: AppContext, session: AsyncSession, principal: Principal, body: schemas.EstimateRequest
) -> LaunchPlan:
    skill, manifest = await get_skill(session, body.skill_id)
    if skill.status is SkillStatus.DEPRECATED:
        raise Invalid(f"Skill '{skill.id}' is deprecated")
    preset = None
    if body.preset_id is not None:
        preset = manifest.preset(body.preset_id)
        if preset is None:
            raise Invalid(f"Unknown preset '{body.preset_id}'")
    params = _validated(manifest, {**(preset.params if preset else {}), **body.params})
    params, checkpoint = await _resolve_checkpoint(session, manifest, params, body)
    from_preset = preset is not None and checkpoint is None and params == _validated(manifest, preset.params)

    parent_run_id = checkpoint.run_id if checkpoint else None
    if isinstance(body, schemas.RunCreate) and body.parent_run_id is not None:
        if parent_run_id is not None and parent_run_id != body.parent_run_id:
            raise Invalid("parent_run_id does not match the warm-start checkpoint's run")
        await get_run(session, body.parent_run_id)
        parent_run_id = body.parent_run_id

    target = await get_target(session, body.compute_target_id)
    usage = (await usage_by_target(session, [target.id])).get(target.id, Usage())
    estimate = estimate_run(
        EstimateInput(
            manifest=manifest,
            params=params,
            target=target,
            usage=usage,
            role=principal.role,
            allow_local_training=ctx.settings.environment == "development",
        )
    )
    return LaunchPlan(
        skill=skill,
        manifest=manifest,
        params=params,
        preset_id=body.preset_id,
        from_preset=from_preset,
        target=target,
        usage=usage,
        parent_run_id=parent_run_id,
        parent_checkpoint=checkpoint,
        estimate=estimate,
    )


async def estimate(
    ctx: AppContext, session: AsyncSession, principal: Principal, body: schemas.EstimateRequest
) -> schemas.Estimate:
    return (await plan_launch(ctx, session, principal, body)).estimate


def _check_launch_allowed(ctx: AppContext, principal: Principal, plan: LaunchPlan) -> None:
    custom = not plan.from_preset or plan.parent_run_id is not None
    if custom and not principal.can(Permission.RUN_CREATE_CUSTOM):
        raise Forbidden(
            "Custom params and warm starts need permission run:create_custom; launch a preset as is"
        )
    target = plan.target
    if not target.enabled:
        raise Invalid(f"Compute target '{target.name}' is disabled")
    if target.kind in KINDS_NEEDING_SECRET and not target.has_secret:
        raise Invalid(f"Compute target '{target.name}' has no credentials")
    if (
        target.kind is BackendKind.LOCAL_CPU
        and plan.manifest.train_stages
        and not is_smoke(plan.params)
        and ctx.settings.environment != "development"
    ):
        raise Invalid("Only smoke runs may train on the local CPU target")


async def _local_target(session: AsyncSession) -> ComputeTarget | None:
    return await session.scalar(
        sa.select(ComputeTarget)
        .where(ComputeTarget.kind == BackendKind.LOCAL_CPU, ComputeTarget.enabled)
        .order_by(ComputeTarget.created_at)
        .limit(1)
    )


async def _default_name(session: AsyncSession, skill_id: str, preset_id: str | None) -> str:
    prefix = f"{skill_id}-{preset_id or 'custom'}"[:55]
    names = await session.scalars(sa.select(Run.name).where(Run.name.like(f"{prefix}-%")))
    numbers = [int(m.group(1)) for n in names if (m := re.fullmatch(rf"{re.escape(prefix)}-(\d+)", n))]
    return f"{prefix}-{max(numbers, default=0) + 1}"


async def create_run(
    ctx: AppContext, session: AsyncSession, principal: Principal, body: schemas.RunCreate, ip: str | None
) -> schemas.RunDetail:
    plan = await plan_launch(ctx, session, principal, body)
    _check_launch_allowed(ctx, principal, plan)
    needs_local = any(
        s.runs_on is RunsOn.LOCAL and s.kind is not StageKind.GATE for s in plan.manifest.pipeline
    )
    local = await _local_target(session) if needs_local else None
    if needs_local and local is None:
        raise Invalid("No enabled local_cpu target to run the evaluation stages on")

    name = body.name or await _default_name(session, plan.skill.id, plan.preset_id)
    if await session.scalar(sa.select(Run.id).where(Run.name == name)) is not None:
        raise Conflict(f"A run named '{name}' already exists")
    status = RunStatus.PENDING_APPROVAL if plan.estimate.needs_approval else RunStatus.QUEUED
    run = Run(
        id=uuid.uuid4(),
        name=name,
        skill_id=plan.skill.id,
        preset_id=plan.preset_id,
        params=plan.params,
        status=status,
        compute_target_id=plan.target.id,
        parent_run_id=plan.parent_run_id,
        parent_checkpoint_id=plan.parent_checkpoint.id if plan.parent_checkpoint else None,
        created_by_id=principal.id,
        created_by_name=principal.name,
        created_by_role=principal.role,
        git_sha=plan.skill.git_sha,
        estimate=plan.estimate.model_dump(),
        notes=body.notes,
        created_at=utcnow(),
    )
    session.add(run)
    for position, stage_def in enumerate(plan.manifest.pipeline):
        target_id = None
        if stage_def.kind is not StageKind.GATE:
            target_id = plan.target.id if stage_def.runs_on is RunsOn.TARGET else local.id if local else None
        session.add(
            Stage(
                run_id=run.id,
                key=stage_def.id,
                kind=stage_def.kind,
                title=stage_def.title,
                position=position,
                runs_on=stage_def.runs_on,
                status=StageStatus.PENDING,
                compute_target_id=target_id,
                attempt=1,
            )
        )
    audit.record(
        session,
        audit.Actor.of(principal),
        "run.create",
        "run",
        run.id,
        {
            "name": name,
            "skill_id": plan.skill.id,
            "preset_id": plan.preset_id,
            "target": plan.target.name,
            "status": status.value,
            "gpu_hours": plan.estimate.gpu_hours,
        },
        ip,
    )
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise Conflict(f"A run named '{name}' already exists") from exc
    if status is RunStatus.QUEUED:
        await ctx.jobs.advance_run(run.id)
    await publish_run(session, ctx.events, run)
    return await run_detail(session, run, principal)


# ---------------------------------------------------------------------------------------------- commands


async def _stages(session: AsyncSession, run_id: uuid.UUID, *, lock: bool = False) -> list[Stage]:
    stmt = sa.select(Stage).where(Stage.run_id == run_id).order_by(Stage.position, Stage.attempt)
    if lock:
        stmt = stmt.with_for_update()
    return list((await session.scalars(stmt)).all())


async def cancel(
    ctx: AppContext, session: AsyncSession, principal: Principal, run_id: uuid.UUID, ip: str | None
) -> schemas.RunDetail:
    run = await get_run(session, run_id, lock=True)
    if not policy.may_manage(principal, run):
        raise Forbidden("You can only cancel your own runs")
    if run.status not in policy.CANCELLABLE:
        raise Conflict(f"Run is {run.status.value}; only waiting or running runs can be cancelled")
    if not run.cancel_requested:
        stages = await _stages(session, run.id, lock=True)
        # Nothing started yet: finish here. Otherwise the worker tears remote jobs down and releases the
        # targets first (a queued stage may be backing off after a submit that already booted a box).
        tear_down = any(s.status.active for s in stages)
        run.cancel_requested = True
        if not tear_down:
            transitions.cancel_run(run, stages)
        audit.record(session, audit.Actor.of(principal), "run.cancel", "run", run.id, {"name": run.name}, ip)
        await session.commit()
        if tear_down:
            await ctx.jobs.cancel_run(run.id)
        else:
            await publish_stages(session, ctx.events, stages)
        await publish_run(session, ctx.events, run)
    return await run_detail(session, run, principal)


async def retry(
    ctx: AppContext, session: AsyncSession, principal: Principal, run_id: uuid.UUID, ip: str | None
) -> schemas.RunDetail:
    run = await get_run(session, run_id, lock=True)
    if not policy.may_manage(principal, run):
        raise Forbidden("You can only retry your own runs")
    if run.status is not RunStatus.FAILED:
        raise Conflict(f"Run is {run.status.value}; only failed runs can be retried")
    stages = main_stages(await _stages(session, run.id, lock=True))
    failed = next((s for s in stages if s.status in (StageStatus.FAILED, StageStatus.CANCELLED)), None)
    if failed is None:
        raise Conflict("Run has no failed stage to retry")
    fresh = Stage(
        run_id=run.id,
        key=failed.key,
        kind=failed.kind,
        title=failed.title,
        position=failed.position,
        runs_on=failed.runs_on,
        status=StageStatus.PENDING,
        compute_target_id=failed.compute_target_id,
        attempt=failed.attempt + 1,
    )
    session.add(fresh)
    later = [s for s in stages if s.position > failed.position]
    for stage in later:
        transitions.reset_stage(stage)
    approval = await _retry_approval(ctx, session, principal, run, failed)
    if approval is not None:
        run.status = RunStatus.PENDING_APPROVAL
        run.estimate = approval.model_dump()
        run.launch_decided_by_id = run.launch_decided_by_name = run.launch_decided_at = None
    else:
        run.status = RunStatus.QUEUED
    run.error = None
    run.finished_at = None
    run.cancel_requested = False
    audit.record(
        session,
        audit.Actor.of(principal),
        "run.retry",
        "run",
        run.id,
        {"stage": failed.key, "attempt": fresh.attempt, "status": run.status.value},
        ip,
    )
    await session.commit()
    if run.status is RunStatus.QUEUED:
        await ctx.jobs.advance_run(run.id)
    await publish_stages(session, ctx.events, [fresh, *later])
    await publish_run(session, ctx.events, run)
    return await run_detail(session, run, principal)


async def _retry_approval(
    ctx: AppContext, session: AsyncSession, principal: Principal, run: Run, stage: Stage
) -> schemas.Estimate | None:
    """Retrying a stage on the run's GPU target spends its GPU hours again, so the launch rule (SPEC §4)
    applies to the retry too: returns the estimate if the retrier would need approval to launch the run."""
    if stage.runs_on is not RunsOn.TARGET or run.compute_target_id is None:
        return None
    target = await session.get(ComputeTarget, run.compute_target_id)
    if target is None:
        return None
    _, manifest = await get_skill(session, run.skill_id)
    estimate = estimate_run(
        EstimateInput(
            manifest=manifest,
            params=run.params,
            target=target,
            usage=(await usage_by_target(session, [target.id])).get(target.id, Usage()),
            role=principal.role,
            allow_local_training=ctx.settings.environment == "development",
        )
    )
    return estimate if estimate.needs_approval else None


async def decide_launch(
    ctx: AppContext,
    session: AsyncSession,
    principal: Principal,
    run_id: uuid.UUID,
    body: schemas.Decision,
    ip: str | None,
) -> schemas.RunDetail:
    run = await get_run(session, run_id, lock=True)
    if policy.owns(principal, run):
        raise Forbidden("You cannot approve your own launch")
    if run.status is not RunStatus.PENDING_APPROVAL:
        raise Conflict(f"Run is {run.status.value}, not waiting for launch approval")
    run.launch_decided_by_id = principal.id
    run.launch_decided_by_name = principal.name
    run.launch_decided_at = utcnow()
    stages: list[Stage] = []
    if body.decision == "approve":
        run.status = RunStatus.QUEUED
    else:
        stages = await _stages(session, run.id, lock=True)
        reason = f"launch rejected: {body.comment}" if body.comment else "launch rejected"
        transitions.cancel_run(run, stages, error=reason)
    audit.record(
        session,
        audit.Actor.of(principal),
        f"run.launch_{body.decision}",
        "run",
        run.id,
        {"name": run.name, "comment": body.comment},
        ip,
    )
    await session.commit()
    if run.status is RunStatus.QUEUED:
        await ctx.jobs.advance_run(run.id)
    await publish_stages(session, ctx.events, stages)
    await publish_run(session, ctx.events, run)
    return await run_detail(session, run, principal)


async def review(
    ctx: AppContext,
    session: AsyncSession,
    principal: Principal,
    run_id: uuid.UUID,
    body: schemas.Decision,
    ip: str | None,
) -> schemas.RunDetail:
    run = await get_run(session, run_id, lock=True)
    if policy.owns(principal, run):
        raise Forbidden("You cannot review a release of your own run")
    decision = await session.get(GateDecision, run.id, with_for_update=True)
    if (
        run.status is not RunStatus.AWAITING_REVIEW
        or decision is None
        or decision.verdict is not GateVerdict.PASS
    ):
        raise Conflict(f"Run is {run.status.value}, not awaiting release review")
    if body.decision == "reject" and not (body.comment or "").strip():
        raise Invalid("A comment is required to reject a release", details={"comment": "required"})
    approved = body.decision == "approve"
    run.status = RunStatus.APPROVED if approved else RunStatus.REJECTED
    decision.review_status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
    decision.reviewer_id = principal.id
    decision.reviewer_name = principal.name
    decision.comment = body.comment
    decision.reviewed_at = utcnow()
    audit.record(
        session,
        audit.Actor.of(principal),
        f"release.{body.decision}",
        "run",
        run.id,
        {"name": run.name, "comment": body.comment},
        ip,
    )
    await session.commit()
    await publish_run(session, ctx.events, run)
    return await run_detail(session, run, principal)


async def evaluate_checkpoint(
    ctx: AppContext,
    session: AsyncSession,
    principal: Principal,
    run_id: uuid.UUID,
    body: schemas.EvaluateCheckpoint,
    ip: str | None,
) -> schemas.RunDetail:
    run = await get_run(session, run_id, lock=True)
    if run.status in (RunStatus.PENDING_APPROVAL, RunStatus.CANCELLED):
        raise Conflict(f"Run is {run.status.value}")
    checkpoint = await session.get(Artifact, body.checkpoint_id)
    if checkpoint is None or checkpoint.run_id != run.id or checkpoint.kind not in CHECKPOINT_KINDS:
        raise NotFound("Checkpoint not found in this run")
    _, manifest = await get_skill(session, run.skill_id)
    stage_def = (
        manifest.stage(body.stage_key) if body.stage_key else next(iter(manifest.evaluate_stages), None)
    )
    if stage_def is None or stage_def.kind is not StageKind.EVALUATE:
        raise Invalid("stage_key must name an evaluate stage of this skill")
    if stage_def.runs_on is RunsOn.LOCAL:
        local = await _local_target(session)
        if local is None:
            raise Invalid("No enabled local_cpu target to run the evaluation on")
        target_id = local.id
    else:
        target_id = run.compute_target_id

    stages = await _stages(session, run.id)
    key = transitions.side_stage_key(stage_def.id, checkpoint)
    same_key = [s for s in stages if s.key == key]
    if any(s.status.active for s in same_key):
        raise Conflict("This checkpoint is already being evaluated")
    stage = Stage(
        run_id=run.id,
        key=key,
        kind=StageKind.EVALUATE,
        title=f"{stage_def.title} @ {checkpoint.name}",
        position=max(s.position for s in stages) + 1,
        runs_on=stage_def.runs_on,
        status=StageStatus.QUEUED,
        compute_target_id=target_id,
        attempt=max((s.attempt for s in same_key), default=0) + 1,
        checkpoint_id=checkpoint.id,
    )
    session.add(stage)
    audit.record(
        session,
        audit.Actor.of(principal),
        "run.evaluate_checkpoint",
        "run",
        run.id,
        {"checkpoint": checkpoint.name, "stage_key": key},
        ip,
    )
    await session.commit()
    await ctx.jobs.execute_stage(stage.id)
    await publish_stages(session, ctx.events, [stage])
    return await run_detail(session, run, principal)


async def list_evaluations(session: AsyncSession, run_id: uuid.UUID) -> list[schemas.Evaluation]:
    await get_run(session, run_id)
    rows = (
        await session.execute(
            sa.select(Evaluation, Stage.key)
            .join(Stage, Stage.id == Evaluation.stage_id)
            .where(Evaluation.run_id == run_id)
            .order_by(Evaluation.created_at, Evaluation.id)
        )
    ).all()
    checkpoints = await artifacts_by_id(session, [e.checkpoint_id for e, _ in rows if e.checkpoint_id])
    return [
        schemas.Evaluation(
            id=e.id,
            stage_id=e.stage_id,
            stage_key=key,
            checkpoint=checkpoints.get(e.checkpoint_id) if e.checkpoint_id else None,
            suite=e.suite,
            summary=e.summary,
            created_at=e.created_at,
        )
        for e, key in rows
    ]

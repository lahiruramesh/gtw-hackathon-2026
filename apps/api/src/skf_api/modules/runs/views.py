"""Read models for runs (RunSummary, RunDetail, Stage), batch-loaded so a list costs a fixed query count."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.auth import Principal
from skf_api.modules.artifacts import service as artifacts
from skf_api.modules.artifacts.models import Artifact as ArtifactRow
from skf_api.modules.artifacts.schemas import Artifact
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.gates import schemas as gate_schemas
from skf_api.modules.gates.metrics import resolve_metric
from skf_api.modules.gates.models import GateDecision
from skf_api.modules.gates.service import Summaries, latest_summaries
from skf_api.modules.refs import RunRef, UserRef
from skf_api.modules.runs import policy, schemas
from skf_api.modules.runs.models import Run, Stage, StageStatus
from skf_api.modules.skills.models import Skill
from skf_api.skills_registry.manifest import Manifest


def latest_attempts(stages: Iterable[Stage]) -> list[Stage]:
    """One row per stage key (its highest attempt), in pipeline order."""
    by_key: dict[str, Stage] = {}
    for stage in stages:
        if stage.key not in by_key or stage.attempt > by_key[stage.key].attempt:
            by_key[stage.key] = stage
    return sorted(by_key.values(), key=lambda s: (s.position, s.key))


def main_stages(stages: Iterable[Stage]) -> list[Stage]:
    return [s for s in latest_attempts(stages) if not s.is_side_stage]


def stage_schema(stage: Stage, checkpoint: Artifact | None) -> schemas.Stage:
    return schemas.Stage(
        id=stage.id,
        key=stage.key,
        kind=stage.kind,
        title=stage.title,
        position=stage.position,
        runs_on=stage.runs_on,
        status=stage.status,
        compute_target_id=stage.compute_target_id,
        attempt=stage.attempt,
        progress=stage.progress,
        message=stage.message,
        started_at=stage.started_at,
        finished_at=stage.finished_at,
        last_heartbeat_at=stage.last_heartbeat_at,
        gpu_seconds=stage.gpu_seconds,
        cost=stage.cost,
        noise_dropped=stage.noise_dropped,
        error=stage.error,
        external_url=(stage.external_ref or {}).get("url"),
        checkpoint=checkpoint,
    )


def gate_schema(decision: GateDecision) -> gate_schemas.GateDecision:
    return gate_schemas.GateDecision(
        verdict=decision.verdict,
        criteria=[gate_schemas.GateCriterionResult.model_validate(c) for c in decision.criteria],
        evaluated_at=decision.evaluated_at,
        review_status=decision.review_status,
        reviewer=UserRef(id=decision.reviewer_id, name=decision.reviewer_name or decision.reviewer_id)
        if decision.reviewer_id
        else None,
        comment=decision.comment,
        reviewed_at=decision.reviewed_at,
    )


def headline_values(manifest: Manifest, summaries: Summaries) -> list[schemas.HeadlineValue]:
    values = []
    for h in manifest.headline:
        actual = resolve_metric(summaries, h.metric)
        if actual is not None and h.scale is not None:
            actual *= h.scale
        values.append(schemas.HeadlineValue(label=h.label, value=actual, unit=h.unit))
    return values


@dataclass
class _Related:
    manifests: dict[str, tuple[str, Manifest]]
    targets: dict[uuid.UUID, ComputeTarget]
    stages: dict[uuid.UUID, list[Stage]]
    gates: dict[uuid.UUID, GateDecision]
    summaries: dict[uuid.UUID, Summaries]
    parents: dict[uuid.UUID, Run]
    artifacts: dict[uuid.UUID, Artifact]  # stage checkpoints and warm-start checkpoints


async def _load_related(session: AsyncSession, runs: Sequence[Run]) -> _Related:
    run_ids = [r.id for r in runs]
    skills = (await session.scalars(sa.select(Skill).where(Skill.id.in_({r.skill_id for r in runs})))).all()
    target_ids = {r.compute_target_id for r in runs if r.compute_target_id}
    targets = (await session.scalars(sa.select(ComputeTarget).where(ComputeTarget.id.in_(target_ids)))).all()
    stages: dict[uuid.UUID, list[Stage]] = defaultdict(list)
    for stage in await session.scalars(sa.select(Stage).where(Stage.run_id.in_(run_ids))):
        stages[stage.run_id].append(stage)
    gates = (await session.scalars(sa.select(GateDecision).where(GateDecision.run_id.in_(run_ids)))).all()
    parent_ids = {r.parent_run_id for r in runs if r.parent_run_id}
    parents = (
        (await session.scalars(sa.select(Run).where(Run.id.in_(parent_ids)))).all() if parent_ids else []
    )
    stage_checkpoints = [s.checkpoint_id for ss in stages.values() for s in ss if s.checkpoint_id]
    return _Related(
        manifests={s.id: (s.name, Manifest.model_validate(s.manifest)) for s in skills},
        targets={t.id: t for t in targets},
        stages=stages,
        gates={g.run_id: g for g in gates},
        summaries=await latest_summaries(session, run_ids),
        parents={p.id: p for p in parents},
        artifacts=await artifacts.artifacts_by_id(
            session, [*stage_checkpoints, *(r.parent_checkpoint_id for r in runs if r.parent_checkpoint_id)]
        ),
    )


def _parent(run: Run, rel: _Related) -> schemas.RunParent | None:
    parent = rel.parents.get(run.parent_run_id) if run.parent_run_id else None
    if parent is None:
        return None
    return schemas.RunParent(
        run=RunRef(id=parent.id, name=parent.name, status=parent.status),
        checkpoint=rel.artifacts.get(run.parent_checkpoint_id) if run.parent_checkpoint_id else None,
    )


def _summary(run: Run, rel: _Related) -> schemas.RunSummary:
    skill_name, manifest = rel.manifests[run.skill_id]
    all_stages = rel.stages.get(run.id, [])
    current = next(
        (s for s in main_stages(all_stages) if s.status not in (StageStatus.SUCCEEDED, StageStatus.SKIPPED)),
        None,
    )
    target = rel.targets.get(run.compute_target_id) if run.compute_target_id else None
    gate = rel.gates.get(run.id)
    return schemas.RunSummary(
        id=run.id,
        name=run.name,
        skill_id=run.skill_id,
        skill_name=skill_name,
        preset_id=run.preset_id,
        status=run.status,
        compute_target=schemas.TargetRef(id=target.id, name=target.name, kind=target.kind)
        if target
        else None,
        created_by=UserRef(id=run.created_by_id, name=run.created_by_name),
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        current_stage=schemas.CurrentStage(
            key=current.key, title=current.title, status=current.status, progress=current.progress
        )
        if current
        else None,
        gate_verdict=gate.verdict if gate else None,
        parent=_parent(run, rel),
        imported=run.imported,
        gpu_hours=round(sum(s.gpu_seconds for s in all_stages) / 3600.0, 3),
        cost=round(sum(s.cost for s in all_stages), 2),
        headline=headline_values(manifest, rel.summaries.get(run.id, {})),
    )


async def run_summaries(session: AsyncSession, runs: Sequence[Run]) -> list[schemas.RunSummary]:
    if not runs:
        return []
    rel = await _load_related(session, runs)
    return [_summary(run, rel) for run in runs]


async def run_details(
    session: AsyncSession, runs: Sequence[Run], principal: Principal
) -> list[schemas.RunDetail]:
    if not runs:
        return []
    rel = await _load_related(session, runs)
    run_ids = [r.id for r in runs]
    children: dict[uuid.UUID, list[RunRef]] = defaultdict(list)
    for child in await session.scalars(
        sa.select(Run).where(Run.parent_run_id.in_(run_ids)).order_by(Run.created_at)
    ):
        assert child.parent_run_id is not None
        children[child.parent_run_id].append(RunRef(id=child.id, name=child.name, status=child.status))
    with_checkpoints = set(
        await session.scalars(
            sa.select(ArtifactRow.run_id)
            .distinct()
            .where(ArtifactRow.run_id.in_(run_ids), ArtifactRow.kind.in_(artifacts.CHECKPOINT_KINDS))
        )
    )

    details = []
    for run in runs:
        gate = rel.gates.get(run.id)
        details.append(
            schemas.RunDetail(
                **_summary(run, rel).model_dump(),
                params=run.params,
                git_sha=run.git_sha,
                notes=run.notes,
                error=run.error,
                estimate=schemas.Estimate.model_validate(run.estimate) if run.estimate else None,
                children=children.get(run.id, []),
                stages=[
                    stage_schema(s, rel.artifacts.get(s.checkpoint_id) if s.checkpoint_id else None)
                    for s in latest_attempts(rel.stages.get(run.id, []))
                ],
                gate=gate_schema(gate) if gate else None,
                launch_decided_by=UserRef(id=run.launch_decided_by_id, name=run.launch_decided_by_name or "")
                if run.launch_decided_by_id
                else None,
                launch_decided_at=run.launch_decided_at,
                permissions=schemas.RunPermissions(
                    can_cancel=policy.can_cancel(principal, run),
                    can_retry=policy.can_retry(principal, run),
                    can_approve_launch=policy.can_decide_launch(principal, run),
                    can_review=policy.can_review(principal, run),
                    can_evaluate=policy.can_evaluate(principal, run, run.id in with_checkpoints),
                ),
            )
        )
    return details


async def run_detail(session: AsyncSession, run: Run, principal: Principal) -> schemas.RunDetail:
    return (await run_details(session, [run], principal))[0]

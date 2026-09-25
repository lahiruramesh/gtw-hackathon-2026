"""Home page KPIs: activity, GPU usage and cost, gate pass rate, per-target and per-skill summaries."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.db import utcnow
from skf_api.modules.compute import service as compute
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.dashboard import schemas
from skf_api.modules.gates.models import GateDecision, GateVerdict
from skf_api.modules.runs.models import Run, RunStatus, Stage
from skf_api.modules.runs.views import run_summaries
from skf_api.modules.skills.models import Skill
from skf_api.modules.skills.service import best_runs

RECENT_RUNS = 10
PASS_RATE_WINDOW = timedelta(days=30)


async def _pairs(session: AsyncSession, stmt: sa.Select[Any, Any]) -> dict[Any, Any]:
    return {key: value for key, value in await session.execute(stmt)}


async def _count_by_status(session: AsyncSession) -> dict[RunStatus, int]:
    return await _pairs(session, sa.select(Run.status, sa.func.count()).group_by(Run.status))


async def _gate_pass_rate(session: AsyncSession) -> float | None:
    stmt = sa.select(sa.func.count().filter(GateDecision.verdict == GateVerdict.PASS), sa.func.count()).where(
        GateDecision.evaluated_at >= utcnow() - PASS_RATE_WINDOW
    )
    passed, total = (await session.execute(stmt)).one()
    return round(passed / total, 3) if total else None


async def _skills(session: AsyncSession) -> list[schemas.DashboardSkill]:
    skills = (await session.scalars(sa.select(Skill).order_by(Skill.name))).all()
    runs = await _pairs(session, sa.select(Run.skill_id, sa.func.count()).group_by(Run.skill_id))
    gpu = await _pairs(
        session,
        sa.select(Run.skill_id, sa.func.coalesce(sa.func.sum(Stage.gpu_seconds), 0.0))
        .join(Stage, Stage.run_id == Run.id)
        .group_by(Run.skill_id),
    )
    verdicts = await _pairs(
        session,
        sa.select(Run.skill_id, GateDecision.verdict)
        .join(GateDecision, GateDecision.run_id == Run.id)
        .order_by(Run.skill_id, GateDecision.evaluated_at.desc())
        .ext(distinct_on(Run.skill_id)),
    )
    best = await best_runs(session, [s.id for s in skills])
    return [
        schemas.DashboardSkill(
            id=s.id,
            name=s.name,
            runs=runs.get(s.id, 0),
            gpu_hours_total=round(gpu.get(s.id, 0.0) / 3600.0, 3),
            best_run=best.get(s.id),
            latest_verdict=verdicts.get(s.id),
        )
        for s in skills
    ]


async def dashboard(session: AsyncSession) -> schemas.Dashboard:
    counts = await _count_by_status(session)
    targets = (await session.scalars(sa.select(ComputeTarget).order_by(ComputeTarget.name))).all()
    usage = await compute.usage_by_target(session, [t.id for t in targets])
    target_rows = []
    for t in targets:
        view = compute.to_schema(t, usage.get(t.id, compute.Usage()))
        target_rows.append(
            schemas.DashboardTarget(
                id=t.id,
                name=t.name,
                kind=t.kind,
                gpu_label=t.gpu_label,
                active_stages=view.usage.active_stages,
                quota_left_hours=view.usage.quota_left_hours,
                health=view.health,
            )
        )
    recent = (await session.scalars(sa.select(Run).order_by(Run.created_at.desc()).limit(RECENT_RUNS))).all()
    return schemas.Dashboard(
        active_runs=counts.get(RunStatus.QUEUED, 0) + counts.get(RunStatus.RUNNING, 0),
        awaiting_launch_approval=counts.get(RunStatus.PENDING_APPROVAL, 0),
        awaiting_review=counts.get(RunStatus.AWAITING_REVIEW, 0),
        gpu_hours_7d=round(sum(u.gpu_hours_7d for u in usage.values()), 3),
        cost_7d=round(sum(u.cost_7d for u in usage.values()), 2),
        gate_pass_rate_30d=await _gate_pass_rate(session),
        targets=target_rows,
        recent_runs=await run_summaries(session, recent),
        skills=await _skills(session),
    )

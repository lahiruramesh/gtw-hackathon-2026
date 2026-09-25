"""Skill catalog: manifests synced from the pipeline repo, plus run statistics per skill."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.db import utcnow
from skf_api.core.errors import NotFound
from skf_api.core.pagination import Keyset, PageParams
from skf_api.modules.audit import service as audit
from skf_api.modules.gates.models import GateDecision, GateVerdict
from skf_api.modules.refs import RunRef
from skf_api.modules.runs.models import Run, RunStatus
from skf_api.modules.skills import schemas
from skf_api.modules.skills.models import Skill
from skf_api.skills_registry.git import head_sha
from skf_api.skills_registry.manifest import Manifest
from skf_api.skills_registry.registry import SkillRegistry

_COMPLETED = (RunStatus.AWAITING_REVIEW, RunStatus.APPROVED, RunStatus.REJECTED, RunStatus.GATE_FAILED)


async def get_skill(session: AsyncSession, skill_id: str) -> tuple[Skill, Manifest]:
    skill = await session.get(Skill, skill_id)
    if skill is None:
        raise NotFound(f"Skill '{skill_id}' not found")
    return skill, Manifest.model_validate(skill.manifest)


async def sync_skills(
    session: AsyncSession, registry: SkillRegistry, actor: audit.Actor | None = None, ip: str | None = None
) -> schemas.SyncResult:
    """Reload manifests from disk and upsert them. Skills removed from the repo are left as they are:
    their runs still reference them. `actor` is None for startup/CLI syncs (not user-initiated)."""
    result = registry.reload()
    git_sha = await head_sha(registry.skills_dir.parent) or registry.fallback_git_sha
    for loaded in result.skills.values():
        m = loaded.manifest
        values = {
            "id": m.id,
            "name": m.name,
            "summary": m.summary,
            "description": m.description,
            "robot": m.robot,
            "category": m.category,
            "method": m.method,
            "status": m.status,
            "manifest": m.model_dump(mode="json"),
            "manifest_sha": loaded.manifest_sha,
            "git_sha": git_sha,
            "synced_at": utcnow(),
        }
        stmt = insert(Skill).values(**values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[Skill.id], set_={k: stmt.excluded[k] for k in values if k != "id"}
            )
        )
    synced = sorted(result.skills)
    if actor is not None:
        audit.record(
            session, actor, "skill.sync", "skill", None, {"synced": synced, "errors": len(result.errors)}, ip
        )
    await session.commit()
    return schemas.SyncResult(
        synced=synced, errors=[schemas.SyncError(file=e.file, message=e.message) for e in result.errors]
    )


async def _run_stats(session: AsyncSession, skill_ids: list[str]) -> dict[str, tuple[int, datetime | None]]:
    rows = await session.execute(
        sa.select(Run.skill_id, sa.func.count(), sa.func.max(Run.created_at))
        .where(Run.skill_id.in_(skill_ids))
        .group_by(Run.skill_id)
    )
    return {skill_id: (count, last) for skill_id, count, last in rows}


async def best_runs(session: AsyncSession, skill_ids: Iterable[str]) -> dict[str, RunRef]:
    """Latest approved run, else latest unrejected gate pass, else latest run that finished its pipeline."""
    rows = await session.execute(
        sa.select(Run.id, Run.name, Run.status, Run.skill_id, GateDecision.verdict)
        .outerjoin(GateDecision, GateDecision.run_id == Run.id)
        .where(Run.skill_id.in_(list(skill_ids)), Run.status.in_(_COMPLETED))
        .order_by(sa.func.coalesce(Run.finished_at, Run.created_at).desc(), Run.id.desc())
    )

    def rank(status: RunStatus, verdict: GateVerdict | None) -> int:
        if status is RunStatus.APPROVED:
            return 0
        if verdict is GateVerdict.PASS and status is not RunStatus.REJECTED:
            return 1
        return 2

    best: dict[str, tuple[int, RunRef]] = {}
    for run_id, name, status, skill_id, verdict in rows:
        r = rank(status, verdict)
        if skill_id not in best or r < best[skill_id][0]:
            best[skill_id] = (r, RunRef(id=run_id, name=name, status=status))
    return {skill_id: ref for skill_id, (_, ref) in best.items()}


def _summary(skill: Skill, stats: tuple[int, datetime | None], best: RunRef | None) -> schemas.SkillSummary:
    return schemas.SkillSummary(
        id=skill.id,
        name=skill.name,
        summary=skill.summary,
        robot=skill.robot,
        category=skill.category,
        method=skill.method,
        status=skill.status,
        git_sha=skill.git_sha,
        run_count=stats[0],
        last_run_at=stats[1],
        best_run=best,
    )


_KEYSET = Keyset(Skill.id, descending=False)


async def list_skills(session: AsyncSession, page: PageParams) -> schemas.SkillPage:
    rows = (await session.scalars(_KEYSET.apply(sa.select(Skill), page))).all()
    skills = rows[: page.limit]
    ids = [s.id for s in skills]
    stats = await _run_stats(session, ids)
    best = await best_runs(session, ids)
    return schemas.SkillPage(
        items=[_summary(s, stats.get(s.id, (0, None)), best.get(s.id)) for s in skills],
        next_cursor=_KEYSET.next_cursor(rows, page, lambda s: (s.id,)),
    )


async def skill_detail(session: AsyncSession, skill_id: str) -> schemas.SkillDetail:
    skill, manifest = await get_skill(session, skill_id)
    stats = (await _run_stats(session, [skill.id])).get(skill.id, (0, None))
    summary = _summary(skill, stats, (await best_runs(session, [skill.id])).get(skill.id))
    return schemas.SkillDetail(
        **summary.model_dump(),
        description=manifest.description,
        params_schema=manifest.params_schema(),
        presets=[
            schemas.Preset(id=p.id, name=p.name, description=p.description, params=p.params)
            for p in manifest.presets
        ],
        pipeline=[
            schemas.PipelineStageDef(id=s.id, kind=s.kind, title=s.title, runs_on=s.runs_on)
            for s in manifest.pipeline
        ],
        gate=[
            schemas.GateCriterion(metric=c.metric, op=c.op, value=c.value, label=c.display_label)
            for c in manifest.gate
        ],
        headline=[
            schemas.Headline(label=h.label, metric=h.metric, unit=h.unit, scale=h.scale, digits=h.digits)
            for h in manifest.headline
        ],
        metrics=schemas.SkillMetrics(keys=manifest.metrics.keys, primary=manifest.metrics.primary),
    )

"""Release gate (SPEC §8): machine-checked criteria from the manifest, then human review."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.db import utcnow
from skf_api.modules.gates.metrics import compare, resolve_metric
from skf_api.modules.gates.models import GateDecision, GateVerdict, ReviewStatus
from skf_api.modules.runs.models import Evaluation, Run, RunStatus, Stage, StageKind
from skf_api.modules.skills.service import get_skill
from skf_api.skills_registry.manifest import GateLevel, Manifest

type Summaries = dict[str, dict[str, Any]]


async def latest_summaries(session: AsyncSession, run_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, Summaries]:
    """run id -> stage key -> summary of that stage key's most recent evaluation."""
    ids = list(run_ids)
    result: dict[uuid.UUID, Summaries] = {run_id: {} for run_id in ids}
    if not ids:
        return result
    rows = await session.execute(
        sa.select(Evaluation.run_id, Stage.key, Evaluation.summary)
        .join(Stage, Stage.id == Evaluation.stage_id)
        .where(Evaluation.run_id.in_(ids))
        .order_by(Evaluation.created_at, Evaluation.id)
    )
    for run_id, key, summary in rows:
        result[run_id][key] = summary
    return result


def check_criteria(manifest: Manifest, summaries: Summaries) -> tuple[GateVerdict, list[dict[str, Any]]]:
    criteria = []
    for c in manifest.gate:
        actual = resolve_metric(summaries, c.metric)
        criteria.append(
            {
                "metric": c.metric,
                "label": c.display_label,
                "op": c.op,
                "value": c.value,
                "actual": actual,
                "passed": compare(actual, c.op, c.value),
                "level": c.level,
            }
        )
    verdict = GateVerdict.PASS if all(c["passed"] for c in criteria) else GateVerdict.FAIL
    return verdict, criteria


def level_results(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-level verdicts of stored criteria. The release gate covers every criterion, so it passes only
    together with the simulation gate. Decisions stored before levels existed count as release only."""
    levels: dict[GateLevel, list[bool]] = {
        "simulation": [c["passed"] for c in criteria if c.get("level") == "simulation"],
        "release": [c["passed"] for c in criteria],
    }
    return [
        {
            "level": level,
            "verdict": GateVerdict.PASS if all(results) else GateVerdict.FAIL,
            "passed": sum(results),
            "total": len(results),
        }
        for level, results in levels.items()
        if results
    ]


def simulation_verdict(criteria: list[dict[str, Any]]) -> GateVerdict | None:
    return next((r["verdict"] for r in level_results(criteria) if r["level"] == "simulation"), None)


def describe(criteria: list[dict[str, Any]]) -> str:
    """One line for the gate stage, e.g. "simulation pass (2/2), release fail (3/4)"."""
    return ", ".join(
        f"{r['level']} {r['verdict'].value} ({r['passed']}/{r['total']})" for r in level_results(criteria)
    )


async def evaluate_gate(session: AsyncSession, run: Run, manifest: Manifest) -> GateDecision:
    """Record the verdict and move the run to awaiting_review (pass) or gate_failed (fail).
    Re-evaluating replaces the previous decision, including any review of it."""
    summaries = (await latest_summaries(session, [run.id]))[run.id]
    verdict, criteria = check_criteria(manifest, summaries)
    decision = await session.get(GateDecision, run.id)
    if decision is None:
        decision = GateDecision(run_id=run.id)
        session.add(decision)
    decision.verdict = verdict
    decision.criteria = criteria
    decision.evaluated_at = utcnow()
    decision.review_status = (
        ReviewStatus.PENDING if verdict is GateVerdict.PASS else ReviewStatus.NOT_REQUIRED
    )
    decision.reviewer_id = decision.reviewer_name = decision.comment = None
    decision.reviewed_at = None
    run.status = RunStatus.AWAITING_REVIEW if verdict is GateVerdict.PASS else RunStatus.GATE_FAILED
    run.finished_at = run.finished_at or utcnow()
    return decision


async def regate(session: AsyncSession, run_names: Iterable[str] = ()) -> list[str]:
    """Re-evaluate stored gate decisions against the skills' current manifests (after gate criteria change).
    Reviewed decisions (approved or rejected) are left alone: a review applies to the criteria it saw."""
    query = (
        sa.select(Run, GateDecision)
        .join(GateDecision, GateDecision.run_id == Run.id)
        .where(GateDecision.review_status.not_in([ReviewStatus.APPROVED, ReviewStatus.REJECTED]))
        .order_by(Run.created_at)
    )
    if names := list(run_names):
        query = query.where(Run.name.in_(names))
    lines = []
    for run, _ in (await session.execute(query)).all():
        _, manifest = await get_skill(session, run.skill_id)
        decision = await evaluate_gate(session, run, manifest)
        gate_stage = await session.scalar(
            sa.select(Stage)
            .where(Stage.run_id == run.id, Stage.kind == StageKind.GATE)
            .order_by(Stage.attempt.desc())
        )
        if gate_stage is not None:
            gate_stage.title = manifest.pipeline[-1].title  # a manifest's last stage is always its gate
            gate_stage.message = describe(decision.criteria)
        lines.append(f"{run.name}: {describe(decision.criteria)}")
    await session.commit()
    return lines

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
from skf_api.modules.runs.models import Evaluation, Run, RunStatus, Stage
from skf_api.skills_registry.manifest import Manifest

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
            }
        )
    verdict = GateVerdict.PASS if all(c["passed"] for c in criteria) else GateVerdict.FAIL
    return verdict, criteria


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

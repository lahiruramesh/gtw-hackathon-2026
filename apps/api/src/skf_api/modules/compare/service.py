"""Side-by-side comparison of 2-4 runs (e.g. stairs v9 / v10 / v11): headline metrics, gate criteria and
the metric keys available for an overlaid chart."""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.auth import Principal
from skf_api.core.errors import Invalid, NotFound
from skf_api.modules.compare.schemas import Comparison, GateRow, HeadlineRow
from skf_api.modules.runs.models import MetricPoint, Run
from skf_api.modules.runs.views import run_details

MIN_RUNS, MAX_RUNS = 2, 4


async def compare(session: AsyncSession, principal: Principal, run_ids: list[uuid.UUID]) -> Comparison:
    unique = list(dict.fromkeys(run_ids))
    if not MIN_RUNS <= len(unique) <= MAX_RUNS:
        raise Invalid(f"Pick {MIN_RUNS} to {MAX_RUNS} different runs to compare")
    found = {r.id: r for r in await session.scalars(sa.select(Run).where(Run.id.in_(unique)))}
    missing = [str(i) for i in unique if i not in found]
    if missing:
        raise NotFound("Run not found", details={"run_ids": missing})
    details = await run_details(session, [found[i] for i in unique], principal)

    headline: dict[str, HeadlineRow] = {}
    gate: dict[str, GateRow] = {}
    for index, run in enumerate(details):
        for h in run.headline:
            row = headline.setdefault(
                h.label, HeadlineRow(label=h.label, unit=h.unit, values=[None] * len(details))
            )
            row.values[index] = h.value
        for c in run.gate.criteria if run.gate else []:
            grow = gate.setdefault(
                c.label, GateRow(label=c.label, level=c.level, values=[None] * len(details))
            )
            grow.values[index] = None if c.actual is None else c.passed

    keys = await session.scalars(
        sa.select(MetricPoint.key).distinct().where(MetricPoint.run_id.in_(unique)).order_by(MetricPoint.key)
    )
    return Comparison(
        runs=details,
        headline_rows=list(headline.values()),
        gate_rows=list(gate.values()),
        metric_keys=list(keys),
    )

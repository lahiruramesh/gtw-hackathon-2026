"""Metric series for charts, downsampled to at most MAX_POINTS per key."""

from __future__ import annotations

import uuid
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.modules.runs import schemas
from skf_api.modules.runs.models import MetricPoint

MAX_POINTS = 1000


def downsample(points: list[tuple[int, float]], limit: int = MAX_POINTS) -> list[tuple[int, float]]:
    """Evenly spaced points, always keeping the first and the last."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


async def series(
    session: AsyncSession, run_id: uuid.UUID, *, keys: list[str] | None, stage_id: uuid.UUID | None
) -> schemas.Metrics:
    scope = [MetricPoint.run_id == run_id]
    if stage_id:
        scope.append(MetricPoint.stage_id == stage_id)
    available = list(
        await session.scalars(sa.select(MetricPoint.key).distinct().where(*scope).order_by(MetricPoint.key))
    )
    wanted = [k for k in keys if k in available] if keys else available
    points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if wanted:
        rows = await session.execute(
            sa.select(MetricPoint.key, MetricPoint.step, MetricPoint.value)
            .where(*scope, MetricPoint.key.in_(wanted))
            .order_by(MetricPoint.key, MetricPoint.step)
        )
        for key, step, value in rows:
            points[key].append((step, value))
    return schemas.Metrics(
        keys=available, series=[schemas.MetricSeries(key=k, points=downsample(points[k])) for k in wanted]
    )

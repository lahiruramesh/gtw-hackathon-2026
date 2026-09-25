"""Publish `run` and `stage` events (SPEC §9.3) after a transition is committed."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.events import EventBus
from skf_api.modules.artifacts.service import artifacts_by_id
from skf_api.modules.runs.models import Run, Stage
from skf_api.modules.runs.views import run_summaries, stage_schema


async def publish_run(session: AsyncSession, events: EventBus, run: Run) -> None:
    summary = (await run_summaries(session, [run]))[0]
    await events.publish(run.id, "run", [summary.model_dump(mode="json")])


async def publish_stages(session: AsyncSession, events: EventBus, stages: Iterable[Stage]) -> None:
    stages = list(stages)
    if not stages:
        return
    checkpoints = await artifacts_by_id(session, [s.checkpoint_id for s in stages if s.checkpoint_id])
    by_run: dict[object, list[dict[str, object]]] = {}
    for s in stages:
        payload = stage_schema(s, checkpoints.get(s.checkpoint_id) if s.checkpoint_id else None)
        by_run.setdefault(s.run_id, []).append(payload.model_dump(mode="json"))
    for run_id, items in by_run.items():
        await events.publish(str(run_id), "stage", items)

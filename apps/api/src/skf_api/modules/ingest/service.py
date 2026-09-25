"""Where job output lands: log lines, metric points and heartbeats, from the in-job reporter (via /ingest)
or from the orchestrator (fetch_logs fallback, progress.csv). Every write is followed by a live event."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import sqlalchemy as sa
from redis.asyncio import Redis
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.db import utcnow
from skf_api.core.events import EventBus
from skf_api.modules.runs.logs import to_schema as log_schema
from skf_api.modules.runs.models import LogLevel, LogLine, MetricPoint, Run, Stage, StageStatus
from skf_api.modules.runs.notify import publish_stages
from skf_api.modules.skills.models import Skill
from skf_api.skills_registry.manifest import MetricsSpec
from skf_api.skills_registry.metrics import MetricFilter

MAX_LINE_CHARS = 4000
_ERROR = re.compile(r"Traceback|Error|ERROR|FAILED|Exception")
_WARN = re.compile(r"warn", re.IGNORECASE)
_COUNTER_TTL_S = 14 * 24 * 3600


@dataclass(frozen=True)
class Line:
    text: str
    ts: datetime | None = None


@dataclass(frozen=True)
class Point:
    step: int
    values: dict[str, float]
    wall_s: float | None = None


def level_of(text: str) -> LogLevel:
    if _ERROR.search(text):
        return LogLevel.ERROR
    if _WARN.search(text):
        return LogLevel.WARN
    return LogLevel.INFO


def clip(text: str) -> str:
    """A stored line is at most MAX_LINE_CHARS; a runaway line (a solver dumping an array) keeps its head."""
    text = text.rstrip("\r\n").replace("\x00", "")
    if len(text) <= MAX_LINE_CHARS:
        return text
    marker = f" … [{len(text) - MAX_LINE_CHARS} chars truncated]"
    return text[: MAX_LINE_CHARS - len(marker)] + marker


class IngestService:
    def __init__(self, redis: Redis, events: EventBus, max_lines_per_run: int):
        self._redis = redis
        self._events = events
        self._max_lines = max_lines_per_run

    async def _reserve(self, session: AsyncSession, run_id: uuid.UUID, wanted: int) -> tuple[int, bool]:
        """How many of `wanted` lines fit under LOG_MAX_LINES_PER_RUN, and whether this batch crossed the cap.
        The counter lives in Redis (one INCRBY per batch rather than a COUNT), seeded from the table."""
        key = f"run:{run_id}:log_lines"
        if not await self._redis.exists(key):
            stored = await session.scalar(
                sa.select(sa.func.count()).select_from(LogLine).where(LogLine.run_id == run_id)
            )
            await self._redis.set(key, stored or 0, nx=True, ex=_COUNTER_TTL_S)
        total = int(await self._redis.incrby(key, wanted))
        await self._redis.expire(key, _COUNTER_TTL_S)
        before = total - wanted
        allowed = max(0, min(wanted, self._max_lines - before))
        return allowed, before < self._max_lines <= total

    async def append_logs(
        self, session: AsyncSession, stage: Stage, lines: Sequence[Line], noise_dropped: int = 0
    ) -> tuple[int, int]:
        """Stores lines (capped per run) and commits. Returns (accepted, dropped) batch lines."""
        allowed, crossed = await self._reserve(session, stage.run_id, len(lines))
        now = utcnow()
        rows = [
            {
                "run_id": stage.run_id,
                "stage_id": stage.id,
                "ts": line.ts or now,
                "level": level_of(line.text),
                "text": clip(line.text),
            }
            for line in lines[:allowed]
        ]
        if crossed:
            rows.append(
                {
                    "run_id": stage.run_id,
                    "stage_id": stage.id,
                    "ts": now,
                    "level": LogLevel.WARN,
                    "text": f"Log limit of {self._max_lines} lines per run reached; further lines are "
                    "dropped and counted.",
                }
            )
        dropped = len(lines) - allowed
        stored: list[LogLine] = []
        if rows:
            stored = list(
                await session.scalars(insert(LogLine).returning(LogLine, sort_by_parameter_order=True), rows)
            )
        promoted = await self._account(session, stage, noise_dropped + dropped)
        await session.commit()
        await self._events.publish(
            stage.run_id, "log", [log_schema(r).model_dump(mode="json") for r in stored]
        )
        await self._announce(session, stage, promoted)
        return allowed, dropped

    async def write_metrics(
        self, session: AsyncSession, stage: Stage, points: Sequence[Point], *, touch: bool = True
    ) -> int:
        """Upserts points whose keys the skill tracks. Returns the number of values stored."""
        spec = await session.scalar(
            sa.select(Skill.manifest["metrics"])
            .join(Run, Run.skill_id == Skill.id)
            .where(Run.id == stage.run_id)
        )
        accept = MetricFilter(MetricsSpec.model_validate(spec or {})).accepts
        rows, events = [], []
        for p in points:
            values = {k: v for k, v in p.values.items() if accept(k) and math.isfinite(v)}
            if not values:
                continue
            events.append({"stage_id": str(stage.id), "step": p.step, "values": values})
            rows += [
                {
                    "run_id": stage.run_id,
                    "stage_id": stage.id,
                    "step": p.step,
                    "key": k,
                    "value": v,
                    "wall_s": p.wall_s,
                }
                for k, v in values.items()
            ]
        if rows:
            stmt = insert(MetricPoint).values(rows)
            await session.execute(
                stmt.on_conflict_do_update(
                    index_elements=[MetricPoint.stage_id, MetricPoint.key, MetricPoint.step],
                    set_={"value": stmt.excluded.value, "wall_s": stmt.excluded.wall_s},
                )
            )
        promoted = await self._account(session, stage, 0) if touch else False
        await session.commit()
        await self._events.publish(stage.run_id, "metric", events)
        await self._announce(session, stage, promoted)
        return len(rows)

    async def heartbeat(
        self, session: AsyncSession, stage: Stage, progress: float | None, message: str | None
    ) -> None:
        values: dict[str, object] = {}
        if progress is not None:
            values["progress"] = progress
        if message is not None:
            values["message"] = message
        if values:
            await session.execute(sa.update(Stage).where(Stage.id == stage.id).values(**values))
        promoted = await self._account(session, stage, 0)
        await session.commit()
        await self._announce(session, stage, promoted or bool(values))

    async def _announce(self, session: AsyncSession, stage: Stage, changed: bool) -> None:
        if changed:
            await session.refresh(stage)
            await publish_stages(session, self._events, [stage])

    async def _account(self, session: AsyncSession, stage: Stage, dropped: int) -> bool:
        """Heartbeat bookkeeping shared by all ingest calls; True if the stage moved to running. A job that
        phones home is running, so a stage still marked provisioning moves on here rather than waiting for
        the next status poll."""
        now = utcnow()
        await session.execute(
            sa.update(Stage)
            .where(Stage.id == stage.id)
            .values(last_heartbeat_at=now, noise_dropped=Stage.noise_dropped + dropped)
        )
        promoted = await session.execute(
            sa.update(Stage)
            .where(Stage.id == stage.id, Stage.status == StageStatus.PROVISIONING)
            .values(status=StageStatus.RUNNING, started_at=sa.func.coalesce(Stage.started_at, now))
            .returning(Stage.id)
        )
        return promoted.first() is not None

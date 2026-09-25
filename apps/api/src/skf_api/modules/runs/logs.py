"""Reading run logs: pages for the UI, a plain-text download, and the SSE replay."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.db import Database
from skf_api.modules.runs import schemas
from skf_api.modules.runs.models import LogLevel, LogLine, Stage

MAX_PAGE = 2000
_EXPORT_BATCH = 5000


def to_schema(line: LogLine) -> schemas.LogLine:
    return schemas.LogLine(id=line.id, stage_id=line.stage_id, ts=line.ts, level=line.level, text=line.text)


async def page(
    session: AsyncSession,
    run_id: uuid.UUID,
    *,
    after_id: int = 0,
    limit: int = 500,
    stage_id: uuid.UUID | None = None,
    level: LogLevel | None = None,
    q: str | None = None,
) -> schemas.LogPage:
    stmt = sa.select(LogLine).where(LogLine.run_id == run_id, LogLine.id > after_id)
    if stage_id:
        stmt = stmt.where(LogLine.stage_id == stage_id)
    if level:
        stmt = stmt.where(LogLine.level == level)
    if q:
        stmt = stmt.where(sa.func.strpos(sa.func.lower(LogLine.text), q.lower()) > 0)
    rows = (await session.scalars(stmt.order_by(LogLine.id).limit(limit))).all()
    return schemas.LogPage(
        items=[to_schema(r) for r in rows], next_after_id=rows[-1].id if len(rows) == limit else None
    )


async def _all(session: AsyncSession, stmt: sa.Select) -> Sequence[sa.Row]:
    return (await session.execute(stmt)).all()


async def export_text(db: Database, run_id: uuid.UUID) -> AsyncIterator[str]:
    """Streams every line as `ts [stage] LEVEL text`, in batches so memory stays flat for 200k-line runs."""
    after = 0
    while True:
        stmt = (
            sa.select(LogLine.id, LogLine.ts, Stage.key, LogLine.level, LogLine.text)
            .join(Stage, Stage.id == LogLine.stage_id)
            .where(LogLine.run_id == run_id, LogLine.id > after)
            .order_by(LogLine.id)
            .limit(_EXPORT_BATCH)
        )
        rows = await db.detached_read(lambda session, stmt=stmt: _all(session, stmt))
        if not rows:
            return
        yield "".join(
            f"{ts.isoformat()} [{key}] {level.value.upper():5} {text}\n" for _, ts, key, level, text in rows
        )
        after = rows[-1][0]

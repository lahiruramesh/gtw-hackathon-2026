"""Public ingest API for in-job reporters (SPEC §9.2): mounted at /ingest/v1, per-stage token auth only."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.auth import bearer_token
from skf_api.core.deps import ContextDep, SessionDep
from skf_api.core.errors import ERROR_RESPONSES, PayloadTooLarge, RateLimited, Unauthorized
from skf_api.core.ingest_tokens import InvalidIngestToken
from skf_api.modules.ingest import schemas
from skf_api.modules.ingest.service import IngestService, Line, Point
from skf_api.modules.runs.models import Stage

router = APIRouter(prefix="/stages/{stage_id}", tags=["ingest"], responses=ERROR_RESPONSES)

RATE_LIMIT_PER_SECOND = 20


async def authorized_stage_id(
    stage_id: uuid.UUID,
    request: Request,
    ctx: ContextDep,
    authorization: Annotated[str | None, Header()] = None,
) -> uuid.UUID:
    try:
        claims = ctx.ingest_tokens.verify(bearer_token(authorization))
    except InvalidIngestToken as exc:
        raise Unauthorized("Invalid ingest token") from exc
    if claims.stage_id != stage_id:
        raise Unauthorized("Token is not valid for this stage")
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > schemas.MAX_BATCH_BYTES:
        raise PayloadTooLarge(f"Body exceeds {schemas.MAX_BATCH_BYTES} bytes")
    key = f"ingest:rate:{stage_id}:{int(time.time())}"
    count = await ctx.redis.incr(key)
    if count == 1:
        await ctx.redis.expire(key, 2)
    if count > RATE_LIMIT_PER_SECOND:
        raise RateLimited("Too many ingest requests for this stage")
    return stage_id


StageIdDep = Annotated[uuid.UUID, Depends(authorized_stage_id)]


async def _live_stage(session: AsyncSession, stage_id: uuid.UUID) -> Stage:
    # A token dies with its stage: once terminal (or deleted), the reporter is told to stop.
    stage = await session.get(Stage, stage_id)
    if stage is None or stage.status.terminal:
        raise Unauthorized("Stage is not accepting data")
    return stage


def _service(ctx: ContextDep) -> IngestService:
    return IngestService(ctx.redis, ctx.events, ctx.settings.log_max_lines_per_run)


ServiceDep = Annotated[IngestService, Depends(_service)]


@router.post("/logs", response_model=schemas.IngestAck)
async def ingest_logs(
    stage_id: StageIdDep, body: schemas.LogBatch, session: SessionDep, service: ServiceDep
) -> schemas.IngestAck:
    if sum(len(line.text) for line in body.lines) > schemas.MAX_BATCH_BYTES:
        raise PayloadTooLarge(f"Batch exceeds {schemas.MAX_BATCH_BYTES} bytes")
    stage = await _live_stage(session, stage_id)
    stored, dropped = await service.append_logs(
        session, stage, [Line(text=line.text, ts=line.ts) for line in body.lines], body.noise_dropped
    )
    return schemas.IngestAck(accepted=stored, dropped=dropped)


@router.post("/metrics", response_model=schemas.IngestAck)
async def ingest_metrics(
    stage_id: StageIdDep, body: schemas.MetricBatch, session: SessionDep, service: ServiceDep
) -> schemas.IngestAck:
    stage = await _live_stage(session, stage_id)
    stored = await service.write_metrics(
        session, stage, [Point(step=p.step, values=p.values, wall_s=p.wall_s) for p in body.points]
    )
    return schemas.IngestAck(accepted=stored)


@router.post("/heartbeat", response_model=schemas.IngestAck)
async def ingest_heartbeat(
    stage_id: StageIdDep, body: schemas.Heartbeat, session: SessionDep, service: ServiceDep
) -> schemas.IngestAck:
    stage = await _live_stage(session, stage_id)
    await service.heartbeat(session, stage, body.progress, body.message)
    return schemas.IngestAck(accepted=1)

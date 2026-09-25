from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from skf_api.core.auth import Principal
from skf_api.core.deps import ContextDep, SessionDep, client_ip, require
from skf_api.core.errors import ERROR_RESPONSES, JSON_ERROR_RESPONSES
from skf_api.core.pagination import PageParams, page_params
from skf_api.core.permissions import Permission
from skf_api.modules.artifacts import service as artifacts
from skf_api.modules.artifacts.schemas import Artifact
from skf_api.modules.runs import logs, metrics, schemas, service, stream
from skf_api.modules.runs.models import LogLevel, RunStatus
from skf_api.modules.runs.views import run_detail

router = APIRouter(prefix="/runs", tags=["runs"], responses=ERROR_RESPONSES)

CanRead = Annotated[Principal, Depends(require(Permission.RUN_READ))]
CanLaunch = Annotated[Principal, Depends(require(Permission.RUN_CREATE_PRESET))]


async def existing_run_id(run_id: uuid.UUID, ctx: ContextDep) -> uuid.UUID:
    """404 check with its own short session, for streaming endpoints that must not pin a connection."""
    async with ctx.db.session() as session:
        await service.get_run(session, run_id)
    return run_id


@router.post("/estimate", response_model=schemas.Estimate)
async def runs_estimate(
    body: schemas.EstimateRequest, ctx: ContextDep, session: SessionDep, principal: CanLaunch
) -> schemas.Estimate:
    return await service.estimate(ctx, session, principal, body)


@router.get("", response_model=schemas.RunPage)
async def runs_list(
    session: SessionDep,
    principal: CanRead,
    page: Annotated[PageParams, Depends(page_params)],
    skill_id: Annotated[str | None, Query()] = None,
    status: Annotated[RunStatus | None, Query()] = None,
    created_by: Annotated[Literal["me"] | None, Query()] = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
) -> schemas.RunPage:
    return await service.list_runs(
        session, page, principal, skill_id=skill_id, status=status, created_by_me=created_by == "me", q=q
    )


@router.post("", response_model=schemas.RunDetail, status_code=201)
async def runs_create(
    body: schemas.RunCreate, request: Request, ctx: ContextDep, session: SessionDep, principal: CanLaunch
) -> schemas.RunDetail:
    return await service.create_run(ctx, session, principal, body, client_ip(request))


@router.get("/{run_id}", response_model=schemas.RunDetail)
async def runs_get(run_id: uuid.UUID, session: SessionDep, principal: CanRead) -> schemas.RunDetail:
    return await run_detail(session, await service.get_run(session, run_id), principal)


@router.post("/{run_id}/cancel", response_model=schemas.RunDetail)
async def runs_cancel(
    run_id: uuid.UUID, request: Request, ctx: ContextDep, session: SessionDep, principal: CanRead
) -> schemas.RunDetail:
    return await service.cancel(ctx, session, principal, run_id, client_ip(request))


@router.post("/{run_id}/retry", response_model=schemas.RunDetail)
async def runs_retry(
    run_id: uuid.UUID, request: Request, ctx: ContextDep, session: SessionDep, principal: CanRead
) -> schemas.RunDetail:
    return await service.retry(ctx, session, principal, run_id, client_ip(request))


@router.post("/{run_id}/launch-decision", response_model=schemas.RunDetail)
async def runs_launch_decision(
    run_id: uuid.UUID,
    body: schemas.Decision,
    request: Request,
    ctx: ContextDep,
    session: SessionDep,
    principal: Annotated[Principal, Depends(require(Permission.RUN_APPROVE_LAUNCH))],
) -> schemas.RunDetail:
    return await service.decide_launch(ctx, session, principal, run_id, body, client_ip(request))


@router.post("/{run_id}/review", response_model=schemas.RunDetail)
async def runs_review(
    run_id: uuid.UUID,
    body: schemas.Decision,
    request: Request,
    ctx: ContextDep,
    session: SessionDep,
    principal: Annotated[Principal, Depends(require(Permission.RELEASE_REVIEW))],
) -> schemas.RunDetail:
    return await service.review(ctx, session, principal, run_id, body, client_ip(request))


@router.post("/{run_id}/evaluate", response_model=schemas.RunDetail)
async def runs_evaluate(
    run_id: uuid.UUID,
    body: schemas.EvaluateCheckpoint,
    request: Request,
    ctx: ContextDep,
    session: SessionDep,
    principal: Annotated[Principal, Depends(require(Permission.RUN_EVALUATE))],
) -> schemas.RunDetail:
    return await service.evaluate_checkpoint(ctx, session, principal, run_id, body, client_ip(request))


@router.get("/{run_id}/logs", response_model=schemas.LogPage)
async def runs_logs(
    run_id: uuid.UUID,
    session: SessionDep,
    _: CanRead,
    stage_id: Annotated[uuid.UUID | None, Query()] = None,
    after_id: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=logs.MAX_PAGE)] = 500,
    level: Annotated[LogLevel | None, Query()] = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
) -> schemas.LogPage:
    await service.get_run(session, run_id)
    return await logs.page(
        session, run_id, after_id=after_id, limit=limit, stage_id=stage_id, level=level, q=q
    )


@router.get(
    "/{run_id}/logs.txt",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}}}},
)
async def runs_logs_text(
    ctx: ContextDep, _: CanRead, run_id: Annotated[uuid.UUID, Depends(existing_run_id)]
) -> StreamingResponse:
    return StreamingResponse(
        logs.export_text(ctx.db, run_id),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.log"'},
    )


@router.get("/{run_id}/metrics", response_model=schemas.Metrics)
async def runs_metrics(
    run_id: uuid.UUID,
    session: SessionDep,
    _: CanRead,
    keys: Annotated[str | None, Query(description="Comma-separated metric keys")] = None,
    stage_id: Annotated[uuid.UUID | None, Query()] = None,
) -> schemas.Metrics:
    await service.get_run(session, run_id)
    wanted = [k for k in (keys or "").split(",") if k] or None
    return await metrics.series(session, run_id, keys=wanted, stage_id=stage_id)


@router.get("/{run_id}/artifacts", response_model=list[Artifact])
async def runs_artifacts(run_id: uuid.UUID, session: SessionDep, _: CanRead) -> list[Artifact]:
    await service.get_run(session, run_id)
    return await artifacts.list_for_run(session, run_id)


@router.get("/{run_id}/evaluations", response_model=list[schemas.Evaluation])
async def runs_evaluations(run_id: uuid.UUID, session: SessionDep, _: CanRead) -> list[schemas.Evaluation]:
    return await service.list_evaluations(session, run_id)


# Separate router: the SSE route documents its errors as JSON, not as text/event-stream.
stream_router = APIRouter(prefix="/runs", tags=["runs"])


@stream_router.get("/{run_id}/events", response_class=EventSourceResponse, responses=JSON_ERROR_RESPONSES)
async def runs_events(
    ctx: ContextDep,
    _: CanRead,
    run_id: Annotated[uuid.UUID, Depends(existing_run_id)],
    after_log_id: Annotated[int, Query(ge=0)] = 0,
) -> AsyncIterator[ServerSentEvent]:
    async for event in stream.run_events(ctx, run_id, after_log_id):
        yield event

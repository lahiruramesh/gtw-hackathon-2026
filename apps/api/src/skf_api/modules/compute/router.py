from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from skf_api.backends.base import BackendKind
from skf_api.core.auth import Principal
from skf_api.core.deps import ContextDep, SessionDep, client_ip, require
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.core.permissions import Permission
from skf_api.modules.audit.service import Actor
from skf_api.modules.compute import schemas, service

router = APIRouter(prefix="/compute-targets", tags=["compute"], responses=ERROR_RESPONSES)

CanRead = Annotated[Principal, Depends(require(Permission.COMPUTE_READ))]
CanWrite = Annotated[Principal, Depends(require(Permission.COMPUTE_WRITE))]


@router.get("", response_model=list[schemas.ComputeTarget])
async def compute_targets_list(session: SessionDep, _: CanRead) -> list[schemas.ComputeTarget]:
    return await service.list_targets(session)


@router.get("/schema/{kind}", response_model=schemas.BackendSchema)
async def compute_targets_schema(kind: BackendKind, ctx: ContextDep, _: CanWrite) -> schemas.BackendSchema:
    return service.backend_schema(ctx.backends, kind)


@router.post("", response_model=schemas.ComputeTarget, status_code=201)
async def compute_targets_create(
    body: schemas.ComputeTargetCreate,
    request: Request,
    ctx: ContextDep,
    session: SessionDep,
    principal: CanWrite,
) -> schemas.ComputeTarget:
    return await service.create_target(session, ctx.backends, body, Actor.of(principal), client_ip(request))


@router.patch("/{target_id}", response_model=schemas.ComputeTarget)
async def compute_targets_update(
    target_id: uuid.UUID,
    body: schemas.ComputeTargetUpdate,
    request: Request,
    ctx: ContextDep,
    session: SessionDep,
    principal: CanWrite,
) -> schemas.ComputeTarget:
    return await service.update_target(
        session, ctx.backends, target_id, body, Actor.of(principal), client_ip(request)
    )


@router.put("/{target_id}/secret", status_code=204, response_class=Response)
async def compute_targets_set_secret(
    target_id: uuid.UUID,
    body: schemas.TargetSecret,
    request: Request,
    ctx: ContextDep,
    session: SessionDep,
    principal: CanWrite,
) -> Response:
    await service.set_secret(
        session, ctx.backends, ctx.secrets, target_id, body.secret, Actor.of(principal), client_ip(request)
    )
    return Response(status_code=204)


@router.post("/{target_id}/check", response_model=schemas.ComputeTarget)
async def compute_targets_check(
    target_id: uuid.UUID, request: Request, ctx: ContextDep, session: SessionDep, principal: CanWrite
) -> schemas.ComputeTarget:
    return await service.check_health(
        session, ctx.backends, target_id, Actor.of(principal), client_ip(request)
    )

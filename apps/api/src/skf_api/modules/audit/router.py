from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from skf_api.core.auth import Principal
from skf_api.core.deps import SessionDep, client_ip, require
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.core.pagination import PageParams, page_params
from skf_api.core.permissions import Permission
from skf_api.modules.audit import schemas, service

router = APIRouter(prefix="/audit-events", tags=["audit"], responses=ERROR_RESPONSES)


@router.get("", response_model=schemas.AuditEventPage)
async def audit_list(
    session: SessionDep,
    _: Annotated[Principal, Depends(require(Permission.AUDIT_READ))],
    page: Annotated[PageParams, Depends(page_params)],
    actor_id: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
    entity_type: Annotated[str | None, Query()] = None,
) -> schemas.AuditEventPage:
    return await service.list_events(session, page, actor_id=actor_id, action=action, entity_type=entity_type)


@router.post("", response_model=schemas.AuditEvent, status_code=201)
async def audit_create(
    body: schemas.AuditEventCreate,
    request: Request,
    session: SessionDep,
    principal: Annotated[Principal, Depends(require(Permission.USER_MANAGE))],
) -> schemas.AuditEvent:
    return await service.create_event(session, principal, body, client_ip(request))

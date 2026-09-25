from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from skf_api.core.auth import Principal
from skf_api.core.deps import ContextDep, SessionDep, client_ip, require
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.core.pagination import PageParams, page_params
from skf_api.core.permissions import Permission
from skf_api.modules.audit import service as audit
from skf_api.modules.skills import schemas, service

router = APIRouter(prefix="/skills", tags=["skills"], responses=ERROR_RESPONSES)

CanRead = Annotated[Principal, Depends(require(Permission.SKILL_READ))]


@router.get("", response_model=schemas.SkillPage)
async def skills_list(
    session: SessionDep, _: CanRead, page: Annotated[PageParams, Depends(page_params)]
) -> schemas.SkillPage:
    return await service.list_skills(session, page)


@router.post("/sync", response_model=schemas.SyncResult)
async def skills_sync(
    ctx: ContextDep,
    session: SessionDep,
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.SKILL_WRITE))],
) -> schemas.SyncResult:
    return await service.sync_skills(session, ctx.registry, audit.Actor.of(principal), client_ip(request))


@router.get("/{skill_id}", response_model=schemas.SkillDetail)
async def skills_get(skill_id: str, session: SessionDep, _: CanRead) -> schemas.SkillDetail:
    return await service.skill_detail(session, skill_id)

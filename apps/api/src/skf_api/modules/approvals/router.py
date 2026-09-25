from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from skf_api.core.auth import Principal
from skf_api.core.deps import SessionDep, require_any
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.core.permissions import Permission
from skf_api.modules.approvals import service
from skf_api.modules.approvals.schemas import Approvals

router = APIRouter(tags=["approvals"], responses=ERROR_RESPONSES)


@router.get("/approvals", response_model=Approvals)
async def approvals_list(
    session: SessionDep,
    principal: Annotated[
        Principal, Depends(require_any(Permission.RELEASE_REVIEW, Permission.RUN_APPROVE_LAUNCH))
    ],
) -> Approvals:
    return await service.approvals(session, principal)

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from skf_api.core.auth import Principal
from skf_api.core.deps import SessionDep, require
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.core.permissions import Permission
from skf_api.modules.dashboard import service
from skf_api.modules.dashboard.schemas import Dashboard

router = APIRouter(tags=["dashboard"], responses=ERROR_RESPONSES)


@router.get("/dashboard", response_model=Dashboard)
async def dashboard_get(
    session: SessionDep, _: Annotated[Principal, Depends(require(Permission.RUN_READ))]
) -> Dashboard:
    return await service.dashboard(session)

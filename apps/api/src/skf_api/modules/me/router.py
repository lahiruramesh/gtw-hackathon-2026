from __future__ import annotations

from fastapi import APIRouter

from skf_api.core.deps import PrincipalDep
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.modules.me.schemas import Me

router = APIRouter(tags=["me"], responses=ERROR_RESPONSES)


@router.get("/me", response_model=Me)
async def me_get(principal: PrincipalDep) -> Me:
    return Me(
        id=principal.id,
        email=principal.email,
        name=principal.name,
        role=principal.role,
        permissions=sorted(principal.permissions),
    )

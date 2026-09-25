from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from skf_api.core.auth import Principal
from skf_api.core.deps import SessionDep, require
from skf_api.core.errors import ERROR_RESPONSES, Invalid
from skf_api.core.permissions import Permission
from skf_api.modules.compare import service
from skf_api.modules.compare.schemas import Comparison

router = APIRouter(tags=["compare"], responses=ERROR_RESPONSES)


def _run_ids(run_ids: Annotated[str, Query(description="2-4 comma-separated run ids")]) -> list[uuid.UUID]:
    try:
        return [uuid.UUID(part) for part in run_ids.split(",") if part.strip()]
    except ValueError as exc:
        raise Invalid("run_ids must be comma-separated run ids") from exc


@router.get("/compare", response_model=Comparison)
async def compare_runs(
    session: SessionDep,
    principal: Annotated[Principal, Depends(require(Permission.RUN_READ))],
    run_ids: Annotated[list[uuid.UUID], Depends(_run_ids)],
) -> Comparison:
    return await service.compare(session, principal, run_ids)

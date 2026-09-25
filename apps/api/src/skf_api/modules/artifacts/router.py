from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from skf_api.core.auth import Principal
from skf_api.core.deps import ContextDep, SessionDep, require
from skf_api.core.errors import ERROR_RESPONSES
from skf_api.core.permissions import Permission
from skf_api.modules.artifacts import schemas, service

router = APIRouter(prefix="/artifacts", tags=["artifacts"], responses=ERROR_RESPONSES)


@router.get("/{artifact_id}/url", response_model=schemas.ArtifactUrl)
async def artifacts_url(
    artifact_id: uuid.UUID,
    ctx: ContextDep,
    session: SessionDep,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> schemas.ArtifactUrl:
    return await service.presigned_url(session, ctx.storage, artifact_id)

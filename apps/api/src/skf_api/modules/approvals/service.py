"""The two review queues: launches waiting for approval, releases waiting for review."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.auth import Principal
from skf_api.core.permissions import Permission
from skf_api.modules.approvals.schemas import Approvals
from skf_api.modules.runs.models import Run, RunStatus
from skf_api.modules.runs.schemas import RunSummary
from skf_api.modules.runs.views import run_summaries

QUEUE_LIMIT = 200


async def _queue(session: AsyncSession, status: RunStatus) -> list[RunSummary]:
    runs = (
        await session.scalars(
            sa.select(Run).where(Run.status == status).order_by(Run.created_at).limit(QUEUE_LIMIT)
        )
    ).all()
    return await run_summaries(session, runs)


async def approvals(session: AsyncSession, principal: Principal) -> Approvals:
    """Each list is only filled for callers who may act on it."""
    return Approvals(
        launches=await _queue(session, RunStatus.PENDING_APPROVAL)
        if principal.can(Permission.RUN_APPROVE_LAUNCH)
        else [],
        releases=await _queue(session, RunStatus.AWAITING_REVIEW)
        if principal.can(Permission.RELEASE_REVIEW)
        else [],
    )

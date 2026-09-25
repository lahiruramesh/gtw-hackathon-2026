from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from skf_api.modules.gates.models import GateVerdict, ReviewStatus
from skf_api.modules.refs import UserRef
from skf_api.skills_registry.manifest import GateOp


class GateCriterionResult(BaseModel):
    metric: str
    label: str
    op: GateOp
    value: float
    actual: float | None
    passed: bool


class GateDecision(BaseModel):
    verdict: GateVerdict
    criteria: list[GateCriterionResult]
    evaluated_at: datetime
    review_status: ReviewStatus
    reviewer: UserRef | None
    comment: str | None
    reviewed_at: datetime | None

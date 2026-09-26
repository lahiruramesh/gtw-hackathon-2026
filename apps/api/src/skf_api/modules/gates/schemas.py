from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from skf_api.modules.gates.models import GateVerdict, ReviewStatus
from skf_api.modules.refs import UserRef
from skf_api.skills_registry.manifest import GateLevel, GateOp


class GateCriterionResult(BaseModel):
    metric: str
    label: str
    op: GateOp
    value: float
    actual: float | None
    passed: bool
    level: GateLevel = "release"


class GateLevelResult(BaseModel):
    level: GateLevel
    verdict: GateVerdict
    passed: int
    total: int


class GateDecision(BaseModel):
    verdict: GateVerdict  # the release gate's verdict
    criteria: list[GateCriterionResult]
    levels: list[GateLevelResult]
    evaluated_at: datetime
    review_status: ReviewStatus
    reviewer: UserRef | None
    comment: str | None
    reviewed_at: datetime | None

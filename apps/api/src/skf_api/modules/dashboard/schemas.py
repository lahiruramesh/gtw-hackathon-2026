from __future__ import annotations

import uuid

from pydantic import BaseModel

from skf_api.backends.base import BackendKind
from skf_api.modules.compute.schemas import TargetHealth
from skf_api.modules.gates.models import GateVerdict
from skf_api.modules.refs import RunRef
from skf_api.modules.runs.schemas import RunSummary


class DashboardTarget(BaseModel):
    id: uuid.UUID
    name: str
    kind: BackendKind
    gpu_label: str | None
    active_stages: int
    quota_left_hours: float | None
    health: TargetHealth


class DashboardSkill(BaseModel):
    id: str
    name: str
    runs: int
    gpu_hours_total: float
    best_run: RunRef | None
    latest_verdict: GateVerdict | None


class Dashboard(BaseModel):
    active_runs: int
    awaiting_launch_approval: int
    awaiting_review: int
    gpu_hours_7d: float
    cost_7d: float
    gate_pass_rate_30d: float | None
    targets: list[DashboardTarget]
    recent_runs: list[RunSummary]
    skills: list[DashboardSkill]

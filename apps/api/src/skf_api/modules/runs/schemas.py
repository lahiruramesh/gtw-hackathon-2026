from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from skf_api.backends.base import BackendKind
from skf_api.core.pagination import Page
from skf_api.modules.artifacts.schemas import Artifact
from skf_api.modules.gates.models import GateVerdict
from skf_api.modules.gates.schemas import GateDecision
from skf_api.modules.refs import RunRef, UserRef
from skf_api.modules.runs.models import LogLevel, RunsOn, RunStatus, StageKind, StageStatus


class Estimate(BaseModel):
    train_minutes: float
    total_minutes: float
    gpu_hours: float
    cost: float
    needs_approval: bool
    reasons: list[str]
    warnings: list[str]
    blockers: list[str]  # why POST /runs would refuse this launch; empty when it can start


class EstimateRequest(BaseModel):
    skill_id: str
    preset_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    compute_target_id: uuid.UUID


class RunCreate(EstimateRequest):
    name: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{2,62}$")
    notes: str | None = Field(default=None, max_length=4000)
    parent_run_id: uuid.UUID | None = None
    parent_checkpoint_id: uuid.UUID | None = None


class Stage(BaseModel):
    id: uuid.UUID
    key: str
    kind: StageKind
    title: str
    position: int
    runs_on: RunsOn
    status: StageStatus
    compute_target_id: uuid.UUID | None
    attempt: int
    progress: float | None
    message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    last_heartbeat_at: datetime | None
    gpu_seconds: float
    cost: float
    noise_dropped: int
    error: str | None
    external_url: str | None
    checkpoint: Artifact | None


class TargetRef(BaseModel):
    id: uuid.UUID
    name: str
    kind: BackendKind


class CurrentStage(BaseModel):
    key: str
    title: str
    status: StageStatus
    progress: float | None


class HeadlineValue(BaseModel):
    label: str
    value: float | None
    unit: str | None


class RunParent(BaseModel):
    run: RunRef
    checkpoint: Artifact | None


class RunSummary(BaseModel):
    id: uuid.UUID
    name: str
    skill_id: str
    skill_name: str
    preset_id: str | None
    status: RunStatus
    compute_target: TargetRef | None
    created_by: UserRef
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    current_stage: CurrentStage | None
    gate_verdict: GateVerdict | None  # release gate
    simulation_verdict: GateVerdict | None  # None when the skill has no simulation-level criteria
    parent: RunParent | None  # warm-start source
    imported: bool
    gpu_hours: float
    cost: float
    headline: list[HeadlineValue]


class RunPermissions(BaseModel):
    can_cancel: bool
    can_retry: bool
    can_approve_launch: bool
    can_review: bool
    can_evaluate: bool


class RunDetail(RunSummary):
    params: dict[str, Any]
    git_sha: str | None
    notes: str | None
    error: str | None
    estimate: Estimate | None
    children: list[RunRef]
    stages: list[Stage]
    gate: GateDecision | None
    launch_decided_by: UserRef | None
    launch_decided_at: datetime | None
    permissions: RunPermissions


class RunPage(Page[RunSummary]):
    pass


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    comment: str | None = Field(default=None, max_length=4000)


class EvaluateCheckpoint(BaseModel):
    checkpoint_id: uuid.UUID
    stage_key: str | None = None


class LogLine(BaseModel):
    id: int
    stage_id: uuid.UUID
    ts: datetime
    level: LogLevel
    text: str


class LogPage(BaseModel):
    items: list[LogLine]
    next_after_id: int | None


class MetricSeries(BaseModel):
    key: str
    points: list[tuple[int, float]]


class Metrics(BaseModel):
    keys: list[str]
    series: list[MetricSeries]


class Evaluation(BaseModel):
    id: uuid.UUID
    stage_id: uuid.UUID
    stage_key: str
    checkpoint: Artifact | None
    suite: str
    summary: dict[str, Any]
    created_at: datetime

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from skf_api.core.pagination import Page
from skf_api.modules.refs import RunRef
from skf_api.modules.runs.models import RunsOn, StageKind
from skf_api.modules.skills.models import SkillCategory, SkillStatus
from skf_api.skills_registry.manifest import GateOp


class SkillSummary(BaseModel):
    id: str
    name: str
    summary: str
    robot: str
    category: SkillCategory
    method: str
    status: SkillStatus
    git_sha: str | None
    run_count: int
    last_run_at: datetime | None
    best_run: RunRef | None


class Preset(BaseModel):
    id: str
    name: str
    description: str | None
    params: dict[str, Any]


class PipelineStageDef(BaseModel):
    id: str
    kind: StageKind
    title: str
    runs_on: RunsOn


class GateCriterion(BaseModel):
    metric: str
    op: GateOp
    value: float
    label: str


class Headline(BaseModel):
    label: str
    metric: str
    unit: str | None
    scale: float | None
    digits: int | None


class SkillMetrics(BaseModel):
    keys: list[str]
    primary: str | None


class SkillDetail(SkillSummary):
    description: str
    params_schema: dict[str, Any]
    presets: list[Preset]
    pipeline: list[PipelineStageDef]
    gate: list[GateCriterion]
    headline: list[Headline]
    metrics: SkillMetrics


class SkillPage(Page[SkillSummary]):
    pass


class SyncError(BaseModel):
    file: str
    message: str


class SyncResult(BaseModel):
    synced: list[str]
    errors: list[SyncError]

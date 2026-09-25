from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from skf_api.backends.base import BackendKind


class TargetHealth(BaseModel):
    status: Literal["ok", "degraded", "down", "unknown"]
    message: str
    checked_at: datetime | None


type QuotaSource = Literal["ledger", "provider"]


class TargetUsage(BaseModel):
    gpu_hours_7d: float
    cost_7d: float
    active_stages: int
    quota_left_hours: float | None
    # ledger: weekly quota minus the studio's own usage; provider: reported by the provider at the last check
    quota_source: QuotaSource | None


class ComputeTarget(BaseModel):
    id: uuid.UUID
    name: str
    kind: BackendKind
    description: str | None
    enabled: bool
    config: dict[str, Any]
    has_secret: bool
    gpu_label: str | None
    steps_per_second: float
    overhead_minutes: float
    cost_per_gpu_hour: float
    weekly_quota_gpu_hours: float | None
    max_unapproved_gpu_hours: float
    max_concurrent: int
    health: TargetHealth
    usage: TargetUsage
    created_at: datetime
    updated_at: datetime


class _TargetFields(BaseModel):
    description: str | None = None
    enabled: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    gpu_label: str | None = None
    steps_per_second: float = Field(gt=0)
    overhead_minutes: float = Field(default=0, ge=0)
    cost_per_gpu_hour: float = Field(default=0, ge=0)
    weekly_quota_gpu_hours: float | None = Field(default=None, gt=0)
    max_unapproved_gpu_hours: float = Field(default=0, ge=0)
    max_concurrent: int = Field(default=1, ge=1, le=64)


class ComputeTargetCreate(_TargetFields):
    name: str = Field(min_length=1, max_length=100)
    kind: BackendKind


class ComputeTargetUpdate(BaseModel):
    """Partial update: only fields present in the request body change."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    enabled: bool | None = None
    config: dict[str, Any] | None = None
    gpu_label: str | None = None
    steps_per_second: float | None = Field(default=None, gt=0)
    overhead_minutes: float | None = Field(default=None, ge=0)
    cost_per_gpu_hour: float | None = Field(default=None, ge=0)
    weekly_quota_gpu_hours: float | None = Field(default=None, gt=0)
    max_unapproved_gpu_hours: float | None = Field(default=None, ge=0)
    max_concurrent: int | None = Field(default=None, ge=1, le=64)


class TargetSecret(BaseModel):
    secret: dict[str, Any]


class BackendSchema(BaseModel):
    config: dict[str, Any]
    secret: dict[str, Any]

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from skf_api.core.pagination import Page


class AuditActor(BaseModel):
    id: str
    name: str
    role: str


class AuditEvent(BaseModel):
    id: int
    ts: datetime
    actor: AuditActor
    action: str
    entity_type: str
    entity_id: str | None
    detail: dict[str, Any]


class AuditEventPage(Page[AuditEvent]):
    pass


class AuditEventCreate(BaseModel):
    action: str = Field(pattern=r"^[a-z_]+\.[a-z_]+$", max_length=64)
    entity_type: str = Field(min_length=1, max_length=64)
    entity_id: str | None = Field(default=None, max_length=200)
    detail: dict[str, Any] = Field(default_factory=dict)

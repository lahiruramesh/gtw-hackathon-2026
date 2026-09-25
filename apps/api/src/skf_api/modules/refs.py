"""Small reference shapes shared by several modules' responses."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from skf_api.modules.runs.models import RunStatus


class UserRef(BaseModel):
    id: str
    name: str


class RunRef(BaseModel):
    id: uuid.UUID
    name: str
    status: RunStatus

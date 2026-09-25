from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from skf_api.modules.artifacts.models import ArtifactKind


class Artifact(BaseModel):
    id: uuid.UUID
    stage_id: uuid.UUID | None
    stage_key: str | None
    kind: ArtifactKind
    name: str
    size_bytes: int
    content_type: str
    step: int | None
    created_at: datetime


class ArtifactUrl(BaseModel):
    url: str
    expires_at: datetime

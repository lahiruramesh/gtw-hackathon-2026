from __future__ import annotations

import enum
import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from skf_api.core.db import Base, created_at_column, str_enum, uuid_pk


class ArtifactKind(enum.StrEnum):
    PARAMS = "params"
    CHECKPOINT = "checkpoint"
    VIDEO = "video"
    IMAGE = "image"
    CSV = "csv"
    JSON = "json"
    LOG = "log"
    CONFIG = "config"
    OTHER = "other"


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (sa.UniqueConstraint("run_id", "uri"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("runs.id", ondelete="CASCADE"))
    stage_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("stages.id", ondelete="SET NULL"))
    kind: Mapped[ArtifactKind] = mapped_column(str_enum(ArtifactKind, "artifact_kind"))
    name: Mapped[str] = mapped_column(sa.Text)
    uri: Mapped[str] = mapped_column(sa.Text)
    size_bytes: Mapped[int] = mapped_column(sa.BigInteger)
    sha256: Mapped[str] = mapped_column(sa.Text)
    content_type: Mapped[str] = mapped_column(sa.Text)
    step: Mapped[int | None] = mapped_column(sa.BigInteger)
    created_at: Mapped[datetime] = created_at_column()


def artifact_key(run_id: uuid.UUID, stage_key: str, name: str) -> str:
    return f"runs/{run_id}/{stage_key}/{name}"

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from skf_api.backends.base import BackendKind
from skf_api.core.db import Base, created_at_column, str_enum, utcnow, uuid_pk


class ComputeTarget(Base):
    __tablename__ = "compute_targets"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(sa.Text, unique=True)
    kind: Mapped[BackendKind] = mapped_column(str_enum(BackendKind, "backend_kind"))
    description: Mapped[str | None] = mapped_column(sa.Text)
    enabled: Mapped[bool] = mapped_column(sa.Boolean, default=True, server_default=sa.true())
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=sa.text("'{}'::jsonb"))
    secret_enc: Mapped[bytes | None] = mapped_column(sa.LargeBinary)
    gpu_label: Mapped[str | None] = mapped_column(sa.Text)
    steps_per_second: Mapped[float] = mapped_column(sa.Float)
    overhead_minutes: Mapped[float] = mapped_column(sa.Float, default=0.0, server_default="0")
    cost_per_gpu_hour: Mapped[float] = mapped_column(sa.Float, default=0.0, server_default="0")
    weekly_quota_gpu_hours: Mapped[float | None] = mapped_column(sa.Float)
    max_unapproved_gpu_hours: Mapped[float] = mapped_column(sa.Float, default=0.0, server_default="0")
    max_concurrent: Mapped[int] = mapped_column(sa.Integer, default=1, server_default="1")
    health: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    health_checked_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=sa.func.now()
    )

    @property
    def has_secret(self) -> bool:
        return self.secret_enc is not None

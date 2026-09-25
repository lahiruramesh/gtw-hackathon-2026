from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from skf_api.core.db import Base, utcnow


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (sa.Index("ix_audit_events_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=utcnow, server_default=sa.func.now()
    )
    actor_id: Mapped[str] = mapped_column(sa.Text, index=True)
    actor_name: Mapped[str] = mapped_column(sa.Text)
    actor_role: Mapped[str] = mapped_column(sa.Text)
    action: Mapped[str] = mapped_column(sa.Text, index=True)
    entity_type: Mapped[str] = mapped_column(sa.Text)
    entity_id: Mapped[str | None] = mapped_column(sa.Text)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=sa.text("'{}'::jsonb"))
    ip: Mapped[str | None] = mapped_column(sa.Text)

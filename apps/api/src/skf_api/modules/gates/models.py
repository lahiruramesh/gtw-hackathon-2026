from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from skf_api.core.db import Base, str_enum


class GateVerdict(enum.StrEnum):
    PASS = "pass"  # noqa: S105 (a verdict, not a password)
    FAIL = "fail"


class ReviewStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NOT_REQUIRED = "not_required"


class GateDecision(Base):
    __tablename__ = "gate_decisions"

    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    verdict: Mapped[GateVerdict] = mapped_column(str_enum(GateVerdict, "gate_verdict"))
    criteria: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    evaluated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    review_status: Mapped[ReviewStatus] = mapped_column(str_enum(ReviewStatus, "review_status"))
    reviewer_id: Mapped[str | None] = mapped_column(sa.Text)
    reviewer_name: Mapped[str | None] = mapped_column(sa.Text)
    comment: Mapped[str | None] = mapped_column(sa.Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

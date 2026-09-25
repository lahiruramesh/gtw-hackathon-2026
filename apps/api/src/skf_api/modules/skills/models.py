from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from skf_api.core.db import Base, str_enum, utcnow


class SkillCategory(enum.StrEnum):
    LOCOMOTION = "locomotion"
    MANIPULATION = "manipulation"
    WORKFLOW = "workflow"


class SkillStatus(enum.StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text)
    summary: Mapped[str] = mapped_column(sa.Text)
    description: Mapped[str] = mapped_column(sa.Text)
    robot: Mapped[str] = mapped_column(sa.Text)
    category: Mapped[SkillCategory] = mapped_column(str_enum(SkillCategory, "skill_category"))
    method: Mapped[str] = mapped_column(sa.Text)
    status: Mapped[SkillStatus] = mapped_column(str_enum(SkillStatus, "skill_status"))
    # json, not jsonb: jsonb reorders keys, and the manifest's param order is the run form's field order.
    manifest: Mapped[dict[str, Any]] = mapped_column(sa.JSON)
    manifest_sha: Mapped[str] = mapped_column(sa.Text)
    git_sha: Mapped[str | None] = mapped_column(sa.Text)
    synced_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=utcnow, server_default=sa.func.now()
    )

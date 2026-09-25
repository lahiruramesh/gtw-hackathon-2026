from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from skf_api.core.db import Base, created_at_column, str_enum, uuid_pk


class RunStatus(enum.StrEnum):
    PENDING_APPROVAL = "pending_approval"
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    GATE_FAILED = "gate_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def executing(self) -> bool:
        return self in (RunStatus.QUEUED, RunStatus.RUNNING)

    @property
    def finished(self) -> bool:
        """The pipeline is done (a run awaiting review has nothing left to execute)."""
        return self not in (RunStatus.PENDING_APPROVAL, RunStatus.QUEUED, RunStatus.RUNNING)


class StageStatus(enum.StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COLLECTING = "collecting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def active(self) -> bool:
        return self in ACTIVE_STAGE_STATUSES

    @property
    def terminal(self) -> bool:
        return self in (StageStatus.SUCCEEDED, StageStatus.FAILED, StageStatus.CANCELLED, StageStatus.SKIPPED)


ACTIVE_STAGE_STATUSES = (
    StageStatus.QUEUED,
    StageStatus.PROVISIONING,
    StageStatus.RUNNING,
    StageStatus.COLLECTING,
)
# Stages holding (or about to hold) remote resources: what max_concurrent and release() count.
OCCUPYING_STAGE_STATUSES = (StageStatus.PROVISIONING, StageStatus.RUNNING, StageStatus.COLLECTING)


class StageKind(enum.StrEnum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    GATE = "gate"


class RunsOn(enum.StrEnum):
    TARGET = "target"  # the run's compute target
    LOCAL = "local"  # the local_cpu target on the worker host


class LogLevel(enum.StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        sa.CheckConstraint("name ~ '^[a-z0-9][a-z0-9-]{2,62}$'", name="name_slug"),
        sa.Index("ix_runs_skill_id_created_at", "skill_id", sa.text("created_at DESC")),
        sa.Index("ix_runs_status", "status"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(sa.Text, unique=True)
    skill_id: Mapped[str] = mapped_column(sa.ForeignKey("skills.id", ondelete="RESTRICT"))
    preset_id: Mapped[str | None] = mapped_column(sa.Text)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[RunStatus] = mapped_column(str_enum(RunStatus, "run_status"))
    compute_target_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("compute_targets.id", ondelete="SET NULL")
    )
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("runs.id", ondelete="SET NULL"))
    parent_checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("artifacts.id", ondelete="SET NULL", use_alter=True)
    )
    created_by_id: Mapped[str] = mapped_column(sa.Text)
    created_by_name: Mapped[str] = mapped_column(sa.Text)
    created_by_role: Mapped[str] = mapped_column(sa.Text)
    git_sha: Mapped[str | None] = mapped_column(sa.Text)
    estimate: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(sa.Text)
    error: Mapped[str | None] = mapped_column(sa.Text)
    imported: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    cancel_requested: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    launch_decided_by_id: Mapped[str | None] = mapped_column(sa.Text)
    launch_decided_by_name: Mapped[str | None] = mapped_column(sa.Text)
    launch_decided_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_column()
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))


class Stage(Base):
    __tablename__ = "stages"
    __table_args__ = (sa.UniqueConstraint("run_id", "key", "attempt"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(sa.Text)
    kind: Mapped[StageKind] = mapped_column(str_enum(StageKind, "stage_kind"))
    title: Mapped[str] = mapped_column(sa.Text)
    position: Mapped[int] = mapped_column(sa.Integer)
    runs_on: Mapped[RunsOn] = mapped_column(str_enum(RunsOn, "runs_on"))
    status: Mapped[StageStatus] = mapped_column(str_enum(StageStatus, "stage_status"), index=True)
    compute_target_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("compute_targets.id", ondelete="SET NULL")
    )
    attempt: Mapped[int] = mapped_column(sa.Integer, default=1, server_default="1")
    external_ref: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    log_cursor: Mapped[str | None] = mapped_column(sa.Text)
    progress: Mapped[float | None] = mapped_column(sa.Float)
    message: Mapped[str | None] = mapped_column(sa.Text)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    gpu_seconds: Mapped[float] = mapped_column(sa.Float, default=0.0, server_default="0")
    cost: Mapped[float] = mapped_column(sa.Float, default=0.0, server_default="0")
    noise_dropped: Mapped[int] = mapped_column(sa.BigInteger, default=0, server_default="0")
    error: Mapped[str | None] = mapped_column(sa.Text)
    # Set on "evaluate this checkpoint" stages, which sit outside the run's main pipeline.
    checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("artifacts.id", ondelete="SET NULL", use_alter=True)
    )

    @property
    def is_side_stage(self) -> bool:
        return self.checkpoint_id is not None


class LogLine(Base):
    __tablename__ = "log_lines"
    __table_args__ = (sa.Index("ix_log_lines_run_id_id", "run_id", "id"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("runs.id", ondelete="CASCADE"))
    stage_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("stages.id", ondelete="CASCADE"))
    ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    level: Mapped[LogLevel] = mapped_column(str_enum(LogLevel, "log_level"))
    text: Mapped[str] = mapped_column(sa.Text)


class MetricPoint(Base):
    __tablename__ = "metric_points"
    __table_args__ = (
        sa.UniqueConstraint("stage_id", "key", "step"),
        sa.Index("ix_metric_points_run_id_key", "run_id", "key"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("runs.id", ondelete="CASCADE"))
    stage_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("stages.id", ondelete="CASCADE"))
    step: Mapped[int] = mapped_column(sa.BigInteger)
    key: Mapped[str] = mapped_column(sa.Text)
    value: Mapped[float] = mapped_column(sa.Float)
    wall_s: Mapped[float | None] = mapped_column(sa.Float)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    stage_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("stages.id", ondelete="CASCADE"))
    checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("artifacts.id", ondelete="SET NULL")
    )
    suite: Mapped[str] = mapped_column(sa.Text)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_at_column()

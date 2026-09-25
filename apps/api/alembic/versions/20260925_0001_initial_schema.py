"""Initial schema: skills, compute targets, runs, stages, logs, metrics, artifacts, evaluations, gate, audit.

Revision ID: 0001
Revises:
Create Date: 2026-09-25

Enums are VARCHAR(32) + CHECK constraints (no native PG enums), matching skf_api.core.db.str_enum.
runs.parent_checkpoint_id and stages.checkpoint_id point at artifacts, which in turn point at runs and
stages, so those two foreign keys are added after the artifacts table exists.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = "app"


def _enum(column: str, name: str, *values: str) -> sa.CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in values)
    return sa.CheckConstraint(f"{column} IN ({allowed})", name=op.f(name))


def _ts(name: str, *, nullable: bool = True, now: bool = False) -> sa.Column:
    return sa.Column(
        name, sa.DateTime(timezone=True), nullable=nullable, server_default=sa.text("now()") if now else None
    )


def _uuid_pk() -> sa.Column:
    return sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()"))


def upgrade() -> None:
    op.create_table(
        "skills",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("robot", sa.Text(), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),  # json keeps key order (the form's field order)
        sa.Column("manifest_sha", sa.Text(), nullable=False),
        sa.Column("git_sha", sa.Text()),
        _ts("synced_at", nullable=False, now=True),
        _enum("category", "ck_skills_skill_category", "locomotion", "manipulation", "workflow"),
        _enum("status", "ck_skills_skill_status", "draft", "active", "deprecated"),
        schema=S,
    )

    op.create_table(
        "compute_targets",
        _uuid_pk(),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("config", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("secret_enc", sa.LargeBinary()),
        sa.Column("gpu_label", sa.Text()),
        sa.Column("steps_per_second", sa.Float(), nullable=False),
        sa.Column("overhead_minutes", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost_per_gpu_hour", sa.Float(), nullable=False, server_default="0"),
        sa.Column("weekly_quota_gpu_hours", sa.Float()),
        sa.Column("max_unapproved_gpu_hours", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_concurrent", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("health", JSONB()),
        _ts("health_checked_at"),
        _ts("created_at", nullable=False, now=True),
        _ts("updated_at", nullable=False, now=True),
        _enum("kind", "ck_compute_targets_backend_kind", "local_cpu", "kaggle", "aws_ec2"),
        schema=S,
    )

    op.create_table(
        "runs",
        _uuid_pk(),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "skill_id", sa.Text(), sa.ForeignKey(f"{S}.skills.id", ondelete="RESTRICT"), nullable=False
        ),
        sa.Column("preset_id", sa.Text()),
        sa.Column("params", JSONB(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "compute_target_id", sa.Uuid(), sa.ForeignKey(f"{S}.compute_targets.id", ondelete="SET NULL")
        ),
        sa.Column("parent_run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="SET NULL")),
        sa.Column("parent_checkpoint_id", sa.Uuid()),
        sa.Column("created_by_id", sa.Text(), nullable=False),
        sa.Column("created_by_name", sa.Text(), nullable=False),
        sa.Column("created_by_role", sa.Text(), nullable=False),
        sa.Column("git_sha", sa.Text()),
        sa.Column("estimate", JSONB()),
        sa.Column("notes", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("imported", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("launch_decided_by_id", sa.Text()),
        sa.Column("launch_decided_by_name", sa.Text()),
        _ts("launch_decided_at"),
        _ts("created_at", nullable=False, now=True),
        _ts("started_at"),
        _ts("finished_at"),
        sa.CheckConstraint("name ~ '^[a-z0-9][a-z0-9-]{2,62}$'", name=op.f("ck_runs_name_slug")),
        _enum(
            "status",
            "ck_runs_run_status",
            "pending_approval",
            "queued",
            "running",
            "awaiting_review",
            "approved",
            "rejected",
            "gate_failed",
            "failed",
            "cancelled",
        ),
        schema=S,
    )
    op.create_index("ix_runs_skill_id_created_at", "runs", ["skill_id", sa.text("created_at DESC")], schema=S)
    op.create_index("ix_runs_status", "runs", ["status"], schema=S)

    op.create_table(
        "stages",
        _uuid_pk(),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("runs_on", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "compute_target_id", sa.Uuid(), sa.ForeignKey(f"{S}.compute_targets.id", ondelete="SET NULL")
        ),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("external_ref", JSONB()),
        sa.Column("log_cursor", sa.Text()),
        sa.Column("progress", sa.Float()),
        sa.Column("message", sa.Text()),
        _ts("started_at"),
        _ts("finished_at"),
        _ts("last_heartbeat_at"),
        sa.Column("gpu_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("noise_dropped", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text()),
        sa.Column("checkpoint_id", sa.Uuid()),
        sa.UniqueConstraint("run_id", "key", "attempt", name="uq_stages_run_id_key_attempt"),
        _enum("kind", "ck_stages_stage_kind", "train", "evaluate", "gate"),
        _enum("runs_on", "ck_stages_runs_on", "target", "local"),
        _enum(
            "status",
            "ck_stages_stage_status",
            "pending",
            "queued",
            "provisioning",
            "running",
            "collecting",
            "succeeded",
            "failed",
            "cancelled",
            "skipped",
        ),
        schema=S,
    )
    op.create_index("ix_stages_run_id", "stages", ["run_id"], schema=S)
    op.create_index("ix_stages_status", "stages", ["status"], schema=S)

    op.create_table(
        "artifacts",
        _uuid_pk(),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage_id", sa.Uuid(), sa.ForeignKey(f"{S}.stages.id", ondelete="SET NULL")),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("step", sa.BigInteger()),
        _ts("created_at", nullable=False, now=True),
        sa.UniqueConstraint("run_id", "uri", name="uq_artifacts_run_id_uri"),
        _enum(
            "kind",
            "ck_artifacts_artifact_kind",
            "params",
            "checkpoint",
            "video",
            "image",
            "csv",
            "json",
            "log",
            "config",
            "other",
        ),
        schema=S,
    )
    op.create_foreign_key(
        "fk_runs_parent_checkpoint_id_artifacts",
        "runs",
        "artifacts",
        ["parent_checkpoint_id"],
        ["id"],
        source_schema=S,
        referent_schema=S,
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_stages_checkpoint_id_artifacts",
        "stages",
        "artifacts",
        ["checkpoint_id"],
        ["id"],
        source_schema=S,
        referent_schema=S,
        ondelete="SET NULL",
    )

    op.create_table(
        "log_lines",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage_id", sa.Uuid(), sa.ForeignKey(f"{S}.stages.id", ondelete="CASCADE"), nullable=False),
        _ts("ts", nullable=False),
        sa.Column("level", sa.String(32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        _enum("level", "ck_log_lines_log_level", "info", "warn", "error"),
        schema=S,
    )
    op.create_index("ix_log_lines_run_id_id", "log_lines", ["run_id", "id"], schema=S)

    op.create_table(
        "metric_points",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage_id", sa.Uuid(), sa.ForeignKey(f"{S}.stages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step", sa.BigInteger(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("wall_s", sa.Float()),
        sa.UniqueConstraint("stage_id", "key", "step", name="uq_metric_points_stage_id_key_step"),
        schema=S,
    )
    op.create_index("ix_metric_points_run_id_key", "metric_points", ["run_id", "key"], schema=S)

    op.create_table(
        "evaluations",
        _uuid_pk(),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage_id", sa.Uuid(), sa.ForeignKey(f"{S}.stages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("checkpoint_id", sa.Uuid(), sa.ForeignKey(f"{S}.artifacts.id", ondelete="SET NULL")),
        sa.Column("suite", sa.Text(), nullable=False),
        sa.Column("summary", JSONB(), nullable=False),
        _ts("created_at", nullable=False, now=True),
        schema=S,
    )
    op.create_index("ix_evaluations_run_id", "evaluations", ["run_id"], schema=S)

    op.create_table(
        "gate_decisions",
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey(f"{S}.runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("verdict", sa.String(32), nullable=False),
        sa.Column("criteria", JSONB(), nullable=False),
        _ts("evaluated_at", nullable=False),
        sa.Column("review_status", sa.String(32), nullable=False),
        sa.Column("reviewer_id", sa.Text()),
        sa.Column("reviewer_name", sa.Text()),
        sa.Column("comment", sa.Text()),
        _ts("reviewed_at"),
        _enum("verdict", "ck_gate_decisions_gate_verdict", "pass", "fail"),
        _enum(
            "review_status",
            "ck_gate_decisions_review_status",
            "pending",
            "approved",
            "rejected",
            "not_required",
        ),
        schema=S,
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        _ts("ts", nullable=False, now=True),
        sa.Column("actor_id", sa.Text(), nullable=False),
        sa.Column("actor_name", sa.Text(), nullable=False),
        sa.Column("actor_role", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text()),
        sa.Column("detail", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("ip", sa.Text()),
        schema=S,
    )
    op.create_index("ix_audit_events_actor_id", "audit_events", ["actor_id"], schema=S)
    op.create_index("ix_audit_events_action", "audit_events", ["action"], schema=S)
    op.create_index("ix_audit_events_entity", "audit_events", ["entity_type", "entity_id"], schema=S)


def downgrade() -> None:
    for table in ("audit_events", "gate_decisions", "evaluations", "metric_points", "log_lines"):
        op.drop_table(table, schema=S)
    op.drop_constraint("fk_stages_checkpoint_id_artifacts", "stages", schema=S, type_="foreignkey")
    op.drop_constraint("fk_runs_parent_checkpoint_id_artifacts", "runs", schema=S, type_="foreignkey")
    for table in ("artifacts", "stages", "runs", "compute_targets", "skills"):
        op.drop_table(table, schema=S)

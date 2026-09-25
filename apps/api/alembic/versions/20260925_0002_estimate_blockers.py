"""Stored run estimates gain `blockers` (Estimate schema, SPEC §9.1): older rows get an empty list.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE app.runs SET estimate = estimate || '{\"blockers\": []}'::jsonb "
        "WHERE estimate IS NOT NULL AND NOT estimate ? 'blockers'"
    )


def downgrade() -> None:
    op.execute("UPDATE app.runs SET estimate = estimate - 'blockers' WHERE estimate IS NOT NULL")

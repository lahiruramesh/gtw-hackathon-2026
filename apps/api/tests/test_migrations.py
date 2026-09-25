"""The hand-written Alembic migration builds exactly the ORM models, and downgrades cleanly."""

from __future__ import annotations

import asyncio

import sqlalchemy as sa
from api_services import DATABASE_URL, alembic_config
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command


def test_migration_matches_models(migrated: None) -> None:
    command.check(alembic_config())  # raises if autogenerate would emit any operation


def test_downgrade_and_upgrade_again(migrated: None) -> None:
    config = alembic_config()
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    command.check(config)


def test_stored_estimates_gain_blockers(migrated: None) -> None:
    """0002: runs created before Estimate.blockers existed must still render (the field is required)."""
    config = alembic_config()
    command.downgrade(config, "0001")

    async def execute(*statements: str) -> list[sa.Row]:
        engine = create_async_engine(DATABASE_URL)
        try:
            async with engine.begin() as conn:
                result = None
                for statement in statements:
                    result = await conn.execute(sa.text(statement))
                return list(result.all()) if result is not None and result.returns_rows else []
        finally:
            await engine.dispose()

    asyncio.run(
        execute(
            "INSERT INTO app.skills (id, name, summary, description, robot, category, method, status,"
            " manifest, manifest_sha, synced_at)"
            " VALUES ('s', 's', 's', 's', 'g1', 'locomotion', 'rl', 'active', '{}', 'x', now())",
            "INSERT INTO app.runs (name, skill_id, params, status, created_by_id, created_by_name,"
            " created_by_role, estimate)"
            " VALUES ('old-run', 's', '{}', 'queued', 'u', 'u', 'viewer', '{\"gpu_hours\": 1.0}')",
        )
    )
    command.upgrade(config, "head")
    try:
        rows = asyncio.run(execute("SELECT estimate FROM app.runs WHERE name = 'old-run'"))
        assert rows[0][0] == {"gpu_hours": 1.0, "blockers": []}
    finally:
        asyncio.run(execute("DELETE FROM app.runs", "DELETE FROM app.skills"))

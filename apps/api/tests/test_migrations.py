"""The hand-written Alembic migration builds exactly the ORM models, and downgrades cleanly."""

from __future__ import annotations

from api_services import alembic_config

from alembic import command


def test_migration_matches_models(migrated: None) -> None:
    command.check(alembic_config())  # raises if autogenerate would emit any operation


def test_downgrade_and_upgrade_again(migrated: None) -> None:
    config = alembic_config()
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    command.check(config)

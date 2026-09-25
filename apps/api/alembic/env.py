"""Alembic environment: async engine, everything in schema `app` (version table included)."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from skf_api.core.db import SCHEMA
from skf_api.models import Base
from skf_api.settings import get_settings

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)


def _database_url() -> str:
    return config.attributes.get("database_url") or get_settings().database_url


def _only_app_schema(name: str | None, type_: str, _parent_names: object) -> bool:
    return name == SCHEMA if type_ == "schema" else True


def _configure(connection: Connection | None = None, url: str | None = None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=Base.metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=_only_app_schema,
        compare_type=True,
    )


def run_migrations_offline() -> None:
    _configure(url=_database_url())
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url())
    async with engine.begin() as connection:
        await connection.run_sync(_run_sync)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

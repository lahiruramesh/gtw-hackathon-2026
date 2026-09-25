"""Async SQLAlchemy engine, sessions and the declarative base (all tables live in schema `app`)."""

from __future__ import annotations

import enum
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import anyio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "app"

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = sa.MetaData(schema=SCHEMA, naming_convention=NAMING_CONVENTION)


def utcnow() -> datetime:
    return datetime.now(UTC)


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        sa.Uuid, primary_key=True, default=uuid.uuid4, server_default=sa.text("gen_random_uuid()")
    )


def created_at_column() -> Mapped[datetime]:
    return mapped_column(sa.DateTime(timezone=True), default=utcnow, server_default=sa.func.now())


def str_enum(enum_cls: type[enum.StrEnum], name: str) -> sa.Enum:
    """Enum stored as VARCHAR + CHECK (not a native PG enum: adding a value needs no type migration)."""
    return sa.Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


class Database:
    def __init__(self, url: str, *, echo: bool = False):
        self.engine: AsyncEngine = create_async_engine(url, echo=echo, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as session:
            yield session

    async def detached_read[T](self, read: Callable[[AsyncSession], Awaitable[T]]) -> T:
        """Runs `read` in its own short session, shielded from cancellation. For streaming responses: their
        generator is cancelled when the client disconnects, and anyio's level-triggered cancellation would
        cancel the session's close as well, so the connection would never return to the pool."""
        with anyio.CancelScope(shield=True):
            async with self.session() as session:
                result = await read(session)
        # CancelScope.__exit__ is typed `-> bool`, but a shielded scope nobody cancels never swallows errors.
        return result  # pyright: ignore[reportPossiblyUnboundVariable, reportReturnType]

    async def ping(self) -> None:
        async with self.engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))

    async def dispose(self) -> None:
        await self.engine.dispose()

"""FastAPI dependencies: context, DB session, authenticated principal, permission checks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.context import AppContext
from skf_api.core.auth import Principal, bearer_token
from skf_api.core.errors import Forbidden


def get_context(request: Request) -> AppContext:
    return request.app.state.context


ContextDep = Annotated[AppContext, Depends(get_context)]


async def get_session(ctx: ContextDep) -> AsyncIterator[AsyncSession]:
    async with ctx.db.session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def current_principal(
    ctx: ContextDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    return await ctx.verifier.verify(bearer_token(authorization))


PrincipalDep = Annotated[Principal, Depends(current_principal)]


def require(permission: str) -> Callable[..., Coroutine[Any, Any, Principal]]:
    """Dependency returning the principal if it holds `permission`, else 403."""
    return require_any(permission)


def require_any(*permissions: str) -> Callable[..., Coroutine[Any, Any, Principal]]:
    async def check(principal: PrincipalDep) -> Principal:
        if not any(principal.can(p) for p in permissions):
            raise Forbidden(f"Requires permission {' or '.join(permissions)}")
        return principal

    return check


def client_ip(request: Request) -> str | None:
    # The web app proxies every call, so the browser's address arrives in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None

"""FastAPI app factory: `uv run uvicorn skf_api.main:app` (or `create_app(context)` in tests)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from skf_api.context import AppContext, build_context
from skf_api.core.deps import ContextDep
from skf_api.core.errors import install_error_handlers
from skf_api.core.logging import AccessLogMiddleware, configure_logging
from skf_api.modules.approvals.router import router as approvals_router
from skf_api.modules.artifacts.router import router as artifacts_router
from skf_api.modules.audit.router import router as audit_router
from skf_api.modules.compare.router import router as compare_router
from skf_api.modules.compute.router import router as compute_router
from skf_api.modules.dashboard.router import router as dashboard_router
from skf_api.modules.ingest.router import router as ingest_router
from skf_api.modules.me.router import router as me_router
from skf_api.modules.runs.router import router as runs_router
from skf_api.modules.runs.router import stream_router as run_events_router
from skf_api.modules.skills.router import router as skills_router
from skf_api.modules.skills.service import sync_skills
from skf_api.settings import get_settings

log = logging.getLogger(__name__)


class Health(BaseModel):
    status: str
    checks: dict[str, str]


def _operation_id(route: APIRoute) -> str:
    # Route handlers are named `<module>_<action>` (runs_list, runs_get): stable ids for the TS client.
    return route.name


async def _prepare(ctx: AppContext) -> None:
    await ctx.storage.ensure_bucket()
    try:
        async with ctx.db.session() as session:
            result = await sync_skills(session, ctx.registry)
        for error in result.errors:
            log.warning("skill manifest rejected", extra={"file": error.file, "reason": error.message})
    except SQLAlchemyError:
        log.exception("skill sync at startup failed (has `skf-api migrate` run?)")


def create_app(context: AppContext | None = None) -> FastAPI:
    """With `context`, the caller owns the resources (tests); otherwise they live for the app's lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if context is not None:
            yield
            return
        settings = get_settings()
        configure_logging(settings.log_level)
        ctx = await build_context(settings)
        await _prepare(ctx)
        app.state.context = ctx
        try:
            yield
        finally:
            await ctx.close()

    app = FastAPI(
        title="SKF Skill Studio API",
        version=version("skf-api"),
        summary="Skills, runs, compute targets, live logs, evaluation and the release gate.",
        lifespan=lifespan,
        generate_unique_id_function=_operation_id,
    )
    if context is not None:
        app.state.context = context
    install_error_handlers(app)
    app.add_middleware(AccessLogMiddleware)

    api = APIRouter(prefix="/api/v1")
    for router in (
        me_router,
        skills_router,
        compute_router,
        runs_router,
        run_events_router,
        artifacts_router,
        approvals_router,
        compare_router,
        dashboard_router,
        audit_router,
    ):
        api.include_router(router)
    app.include_router(api)
    app.include_router(ingest_router, prefix="/ingest/v1")

    @app.get("/healthz", response_model=Health, tags=["health"])
    async def health_live() -> Health:
        return Health(status="ok", checks={})

    @app.get("/readyz", response_model=Health, tags=["health"], responses={503: {"model": Health}})
    async def health_ready(ctx: ContextDep) -> JSONResponse:
        checks: dict[str, str] = {}
        for name, probe in (("database", ctx.db.ping), ("redis", ctx.redis.ping)):
            try:
                await probe()
                checks[name] = "ok"
            except Exception as exc:
                checks[name] = f"error: {type(exc).__name__}"
        ready = all(v == "ok" for v in checks.values())
        body = Health(status="ok" if ready else "unavailable", checks=checks)
        return JSONResponse(body.model_dump(), status_code=200 if ready else 503)

    return app


app = create_app()

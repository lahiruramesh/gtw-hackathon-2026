"""arq worker: `uv run arq skf_api.orchestrator.tasks.WorkerSettings`."""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from arq import cron, func
from arq.connections import RedisSettings
from arq.cron import CronJob
from arq.worker import Function

from skf_api.context import AppContext, build_context
from skf_api.core.jobs import ADVANCE_RUN, CANCEL_RUN, EXECUTE_STAGE
from skf_api.core.logging import configure_logging
from skf_api.orchestrator import executor, reconciler, state_machine
from skf_api.settings import get_settings

# Submitting can boot an instance and rsync the repo; collecting uploads checkpoints.
LONG_JOB_TIMEOUT_S = 3600


def _app(ctx: dict[str, Any]) -> AppContext:
    return ctx["app"]


async def advance_run(ctx: dict[str, Any], run_id: str) -> None:
    await state_machine.advance_run(_app(ctx), uuid.UUID(run_id))


async def execute_stage(ctx: dict[str, Any], stage_id: str) -> None:
    await executor.execute_stage(_app(ctx), uuid.UUID(stage_id))


async def cancel_run(ctx: dict[str, Any], run_id: str) -> None:
    await state_machine.cancel_run(_app(ctx), uuid.UUID(run_id))


async def reconcile(ctx: dict[str, Any]) -> None:
    await reconciler.reconcile(_app(ctx))


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    ctx["app"] = await build_context(settings)


async def shutdown(ctx: dict[str, Any]) -> None:
    app: AppContext | None = ctx.get("app")
    if app is not None:
        await app.close()


class WorkerSettings:
    functions: ClassVar[list[Function]] = [
        func(advance_run, name=ADVANCE_RUN, timeout=60),
        func(execute_stage, name=EXECUTE_STAGE, timeout=LONG_JOB_TIMEOUT_S),
        func(cancel_run, name=CANCEL_RUN, timeout=600),
    ]
    cron_jobs: ClassVar[list[CronJob]] = [
        cron(reconcile, second={0, 20, 40}, run_at_startup=True, timeout=LONG_JOB_TIMEOUT_S),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 20
    keep_result = 60
    # Every task is idempotent, so a job interrupted by a worker restart is simply run again.
    max_tries = 3

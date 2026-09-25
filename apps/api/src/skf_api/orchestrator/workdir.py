"""Per-stage scratch space on the worker: `WORK_DIR/<run>/<stage>/{in,out}`.

`in/` holds the inputs downloaded for submit, `out/` what collect copied back before it is uploaded. Both
are only needed until the stage is finished (its outputs are then in object storage), so they are removed
as soon as a stage reaches a terminal state; otherwise every run would leave a full copy of its checkpoints.
"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import suppress
from pathlib import Path

from skf_api.context import AppContext
from skf_api.modules.runs.models import Stage


def stage_dir(ctx: AppContext, stage: Stage) -> Path:
    return ctx.settings.work_dir / str(stage.run_id) / str(stage.id)


def reset_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


async def discard_stage_dir(ctx: AppContext, stage: Stage) -> None:
    path = stage_dir(ctx, stage)
    await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
    with suppress(OSError):  # another stage of the run still has its dir
        path.parent.rmdir()

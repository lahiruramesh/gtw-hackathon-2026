"""State changes on run/stage rows shared by the API service and the orchestrator (SPEC §6).

Callers hold the run row lock and commit; these functions only mutate rows.
"""

from __future__ import annotations

from collections.abc import Iterable

from skf_api.core.db import utcnow
from skf_api.modules.artifacts.models import Artifact
from skf_api.modules.runs.models import Run, RunStatus, Stage, StageStatus

_SIDE_KEY_MARKER = "-ckpt-"


def side_stage_key(stage_def_id: str, checkpoint: Artifact) -> str:
    """Key of an "evaluate this checkpoint" stage, e.g. `evaluate-ckpt-254279680` (`evaluate-ckpt-final` for
    the final params). The manifest stage it runs is recoverable with `side_stage_def_id`."""
    return f"{stage_def_id}{_SIDE_KEY_MARKER}{checkpoint.step if checkpoint.step is not None else 'final'}"


def side_stage_def_id(key: str) -> str:
    return key.rsplit(_SIDE_KEY_MARKER, 1)[0]


def finish_stage(stage: Stage, status: StageStatus, *, error: str | None = None) -> None:
    stage.status = status
    stage.finished_at = stage.finished_at or utcnow()
    if error is not None:
        stage.error = error
    if status is StageStatus.SUCCEEDED:
        stage.progress = 1.0


def reset_stage(stage: Stage) -> None:
    stage.status = StageStatus.PENDING
    stage.external_ref = None
    stage.log_cursor = None
    stage.progress = None
    stage.message = None
    stage.error = None
    stage.started_at = stage.finished_at = stage.last_heartbeat_at = None


def skip_pending(stages: Iterable[Stage]) -> None:
    for stage in stages:
        if stage.status is StageStatus.PENDING:
            finish_stage(stage, StageStatus.SKIPPED)


def fail_run(run: Run, stages: Iterable[Stage], error: str) -> None:
    skip_pending(stages)
    run.status = RunStatus.FAILED
    run.error = error
    run.finished_at = utcnow()


def cancel_stages(stages: Iterable[Stage]) -> None:
    """Pending stages are skipped, started ones cancelled. Remote teardown happens before this."""
    for stage in stages:
        if stage.status is StageStatus.PENDING:
            finish_stage(stage, StageStatus.SKIPPED)
        elif not stage.status.terminal:
            finish_stage(stage, StageStatus.CANCELLED)
            stage.message = None  # the last live status ("running for 54s") no longer describes the stage


def cancel_run(run: Run, stages: Iterable[Stage], *, error: str | None = None) -> None:
    """Marks every unfinished stage and the run cancelled."""
    cancel_stages(stages)
    run.status = RunStatus.CANCELLED
    run.cancel_requested = True
    run.finished_at = utcnow()
    if error is not None:
        run.error = error

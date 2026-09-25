"""Who may do what to a run (SPEC §4). Shared by enforcement in the service and `RunDetail.permissions`."""

from __future__ import annotations

from skf_api.core.auth import Principal
from skf_api.core.permissions import Permission
from skf_api.modules.runs.models import Run, RunStatus

CANCELLABLE = (RunStatus.PENDING_APPROVAL, RunStatus.QUEUED, RunStatus.RUNNING)


def owns(principal: Principal, run: Run) -> bool:
    return run.created_by_id == principal.id


def may_manage(principal: Principal, run: Run) -> bool:
    """Cancel/retry: own runs with run:create_preset, anyone's with run:cancel_any."""
    return principal.can(Permission.RUN_CANCEL_ANY) or (
        principal.can(Permission.RUN_CREATE_PRESET) and owns(principal, run)
    )


def can_cancel(principal: Principal, run: Run) -> bool:
    return run.status in CANCELLABLE and not run.cancel_requested and may_manage(principal, run)


def can_retry(principal: Principal, run: Run) -> bool:
    return run.status is RunStatus.FAILED and may_manage(principal, run)


def can_decide_launch(principal: Principal, run: Run) -> bool:
    return (
        run.status is RunStatus.PENDING_APPROVAL
        and principal.can(Permission.RUN_APPROVE_LAUNCH)
        and not owns(principal, run)
    )


def can_review(principal: Principal, run: Run) -> bool:
    return (
        run.status is RunStatus.AWAITING_REVIEW
        and principal.can(Permission.RELEASE_REVIEW)
        and not owns(principal, run)
    )


def can_evaluate(principal: Principal, run: Run, has_checkpoints: bool) -> bool:
    return (
        principal.can(Permission.RUN_EVALUATE)
        and has_checkpoints
        and run.status not in (RunStatus.PENDING_APPROVAL, RunStatus.CANCELLED)
    )

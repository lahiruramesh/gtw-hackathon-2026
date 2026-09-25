"""Process-runner seam shared by the backends.

Every external command a backend runs (the kaggle CLI, ssh, rsync, scp, a pipeline interpreter probe)
goes through a `ProcessRunner`, so tests swap in a fake that records argv and returns canned output.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skf_api.backends.base import BackendError


@dataclass(frozen=True)
class ProcResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def error_text(self) -> str:
        """Last lines of stderr (or stdout when stderr is empty), for user-visible error messages."""
        raw = (self.stderr or self.stdout).decode("utf-8", errors="replace").strip()
        return raw[-600:]


class ProcessRunner(Protocol):
    async def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ProcResult: ...


class SubprocessRunner:
    """Runs argv without a shell and captures stdout/stderr."""

    async def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ProcResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=dict(env) if env is not None else None,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise BackendError(f"command not found: {argv[0]}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            message = f"{Path(argv[0]).name} timed out after {timeout_s:.0f}s"
            raise BackendError(message, retryable=True) from None
        return ProcResult(tuple(argv), proc.returncode if proc.returncode is not None else -1, stdout, stderr)

"""Fakes for backend tests: a scripted process runner that records every command."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from skf_api.backends._proc import ProcResult

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class Call:
    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None

    @property
    def command(self) -> str:
        """The remote command of an ssh call (its last argument)."""
        return self.argv[-1]


def result(stdout: str | bytes = "", returncode: int = 0, stderr: str = "") -> ProcResult:
    out = stdout.encode() if isinstance(stdout, str) else stdout
    return ProcResult(("fake",), returncode, out, stderr.encode())


Handler = Callable[[Call], ProcResult]


@dataclass
class FakeRunner:
    """Records every command and answers with `handler(call)` (default: success, no output)."""

    handler: Handler = field(default=lambda call: result())
    calls: list[Call] = field(default_factory=list)

    async def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ProcResult:
        call = Call(list(argv), cwd, dict(env) if env is not None else None)
        self.calls.append(call)
        return self.handler(call)

    def programs(self) -> list[str]:
        return [c.argv[0] for c in self.calls]

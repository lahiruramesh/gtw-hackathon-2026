"""ssh / scp / rsync to one remote host, through the process-runner seam.

The private key exists on disk only inside `async with SshClient(...)`, as a 0600 temp file.
Host keys are pinned per target (`known_hosts` under the target's work dir) with accept-new, keyed
by a stable alias (the EC2 instance id) because the public IP changes on every start.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from skf_api.backends._proc import ProcessRunner, ProcResult
from skf_api.backends.base import BackendError

SSH_UNREACHABLE = 255  # ssh's own exit code for connection/auth failures


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    user: str
    host_key_alias: str


class SshClient:
    def __init__(
        self,
        runner: ProcessRunner,
        endpoint: SshEndpoint,
        *,
        private_key: str,
        known_hosts: Path,
        connect_timeout: int = 10,
    ) -> None:
        self.runner = runner
        self.endpoint = endpoint
        self._private_key = private_key
        self.known_hosts = known_hosts
        self.connect_timeout = connect_timeout
        self._key_path: Path | None = None

    async def __aenter__(self) -> SshClient:
        self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="skf-ssh-")
        try:
            os.fchmod(fd, 0o600)
            key = self._private_key.strip() + "\n"  # OpenSSH rejects keys without the final newline
            os.write(fd, key.encode())
        finally:
            os.close(fd)
        self._key_path = Path(path)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._key_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._key_path.unlink()
            self._key_path = None

    @property
    def destination(self) -> str:
        return f"{self.endpoint.user}@{self.endpoint.host}"

    def options(self) -> list[str]:
        if self._key_path is None:
            raise RuntimeError("SshClient used outside `async with`")
        return [
            "-i",
            str(self._key_path),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            f"HostKeyAlias={self.endpoint.host_key_alias}",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "ServerAliveInterval=30",
        ]

    async def run(self, command: str, *, timeout_s: float = 120) -> ProcResult:
        """Run a remote shell command. Callers build `command` with shlex.quote for every value."""
        return await self.runner(["ssh", *self.options(), self.destination, command], timeout_s=timeout_s)

    async def check(self, command: str, *, timeout_s: float = 120) -> ProcResult:
        result = await self.run(command, timeout_s=timeout_s)
        if not result.ok:
            raise self.error("remote command failed", result)
        return result

    async def upload(self, local: Sequence[Path], remote: str, *, timeout_s: float = 900) -> None:
        """scp files to `remote`: a file path for one file, or a directory ending in `/`."""
        result = await self.runner(
            ["scp", "-q", *self.options(), *(str(p) for p in local), f"{self.destination}:{remote}"],
            timeout_s=timeout_s,
        )
        if not result.ok:
            raise self.error("upload failed", result)

    async def rsync_up(
        self,
        sources: Sequence[str],
        remote_dir: str,
        *,
        cwd: Path,
        excludes: Sequence[str] = (),
        timeout_s: float = 900,
    ) -> None:
        """Mirror `sources` (paths relative to `cwd`) into `remote_dir`, keeping their relative paths."""
        argv = [
            "rsync",
            "-az",
            "--delete",
            "--relative",
            *(f"--exclude={e}" for e in excludes),
            "-e",
            self._rsync_shell(),
            *sources,
            f"{self.destination}:{remote_dir}/",
        ]
        result = await self.runner(argv, cwd=cwd, timeout_s=timeout_s)
        if not result.ok:
            raise self.error("rsync to the instance failed", result)

    async def rsync_down(self, remote: str, local: Path, *, timeout_s: float = 3600) -> None:
        # --no-links: outputs from the box must never become links into the worker's filesystem.
        source = f"{self.destination}:{remote}"
        argv = ["rsync", "-az", "--no-links", "-e", self._rsync_shell(), source, str(local)]
        result = await self.runner(argv, timeout_s=timeout_s)
        if not result.ok:
            raise self.error("rsync from the instance failed", result)

    def error(self, what: str, result: ProcResult) -> BackendError:
        if result.returncode == SSH_UNREACHABLE:
            message = f"cannot reach {self.endpoint.host} over SSH: {result.error_text}"
            return BackendError(message, retryable=True)
        return BackendError(f"{what}: {result.error_text}")

    def _rsync_shell(self) -> str:
        return shlex.join(["ssh", *self.options()])

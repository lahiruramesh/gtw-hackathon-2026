"""The pipeline repo's commit, recorded on skills and runs (runs are pinned to a SHA)."""

from __future__ import annotations

import asyncio
from pathlib import Path


async def head_sha(repo_dir: Path) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo_dir),
            "rev-parse",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    stdout, _ = await proc.communicate()
    sha = stdout.decode().strip()
    return sha if proc.returncode == 0 and sha else None

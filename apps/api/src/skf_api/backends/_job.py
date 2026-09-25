"""Helpers shared by the backends: argv rendering, the reporter command line, log chunking."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from skf_api.backends.base import BackendError, IngestConfig, JobSpec

REPORTER_RELPATH = "apps/reporter/skf_reporter.py"
LOG_CHUNK_BYTES = 1024 * 1024
JOB_LOG = "job.log"


def render_argv(spec: JobSpec, interpreter: list[str], out_dir: str, input_dir: str) -> list[str]:
    """Swap the literal `python` for the backend's interpreter and fill in {out_dir}/{input_dir}."""
    if not spec.argv or spec.argv[0] != "python":
        raise BackendError(f"stage {spec.stage_key}: argv must start with 'python'")
    rest = [a.replace("{out_dir}", out_dir).replace("{input_dir}", input_dir) for a in spec.argv[1:]]
    return [*interpreter, *rest]


def input_relpath(name: str) -> PurePosixPath:
    """Validated relative path of an input inside the job's input dir (no absolute paths, no `..`)."""
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BackendError(f"invalid input file name {name!r}")
    return path


def reporter_args(*, log: str, exit_file: str, progress: str | None, timeout_minutes: int) -> list[str]:
    """Reporter options (everything between the script path and `--`)."""
    args = ["--log", log, "--exit-file", exit_file, "--timeout", str(timeout_minutes * 60)]
    if progress:
        args += ["--progress", progress]
    return args


def ingest_env(ingest: IngestConfig | None, stage_id: str) -> dict[str, str]:
    if ingest is None:
        return {}
    return {"SKF_INGEST_URL": ingest.url, "SKF_INGEST_TOKEN": ingest.token, "SKF_STAGE_ID": stage_id}


def split_lines(chunk: bytes, *, final: bool) -> tuple[list[str], int]:
    """Complete lines in `chunk` and how many bytes they use. A trailing partial line is kept for the
    next read unless the job has finished (`final`)."""
    end = len(chunk) if final else chunk.rfind(b"\n") + 1
    if end == 0 and len(chunk) >= LOG_CHUNK_BYTES:
        end = len(chunk)  # one enormous line: pass it on in pieces
    lines = chunk[:end].decode("utf-8", errors="replace").splitlines()
    return lines, end


def copy_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)

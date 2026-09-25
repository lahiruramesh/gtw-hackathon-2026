from __future__ import annotations

from pathlib import Path

import pytest
from backend_fakes import REPO_ROOT

from skf_api.backends import config_schema, create_backend
from skf_api.backends._job import LOG_CHUNK_BYTES, input_relpath, render_argv, split_lines
from skf_api.backends.base import BackendError, BackendKind, ComputeBackend, JobSpec, TargetContext

CONFIGS = {
    BackendKind.LOCAL_CPU: {},
    BackendKind.KAGGLE: {"username": "skf-ml"},
    BackendKind.AWS_EC2: {"region": "us-east-1", "instance_name": "g1-train"},
}


def spec(argv: list[str]) -> JobSpec:
    return JobSpec(run_id="r", run_name="n", stage_id="s", stage_key="train", kind="train", argv=argv)


def test_render_argv_swaps_interpreter_and_fills_dirs() -> None:
    job = spec(["python", "-m", "x", "{input_dir}/train/params.pkl", "--out", "{out_dir}/a.json"])
    argv = render_argv(job, ["uv", "run", "python"], "/o", "/i")
    assert argv == ["uv", "run", "python", "-m", "x", "/i/train/params.pkl", "--out", "/o/a.json"]


def test_render_argv_requires_python() -> None:
    with pytest.raises(BackendError):
        render_argv(spec(["bash", "-c", "rm -rf /"]), ["python3"], "/o", "/i")


@pytest.mark.parametrize("name", ["/etc/passwd", "../x", "a/../../b", ""])
def test_input_names_must_stay_inside_the_input_dir(name: str) -> None:
    with pytest.raises(BackendError):
        input_relpath(name)


def test_split_lines_keeps_partial_line_until_final() -> None:
    assert split_lines(b"a\nb\npar", final=False) == (["a", "b"], 4)
    assert split_lines(b"a\nb\npar", final=True) == (["a", "b", "par"], 7)
    assert split_lines(b"no newline yet", final=False) == ([], 0)
    huge = b"x" * LOG_CHUNK_BYTES
    assert split_lines(huge, final=False)[1] == LOG_CHUNK_BYTES


@pytest.mark.parametrize("kind", list(BackendKind))
def test_registry_builds_every_backend(kind: BackendKind, tmp_path: Path) -> None:
    ctx = TargetContext("t", "t", CONFIGS[kind], {}, tmp_path, REPO_ROOT)
    backend = create_backend(kind, ctx)
    assert isinstance(backend, ComputeBackend)
    assert backend.kind is kind


@pytest.mark.parametrize("kind", list(BackendKind))
def test_config_schema_marks_secrets_write_only(kind: BackendKind) -> None:
    schema = config_schema(kind)
    assert schema["config"]["type"] == "object"
    for prop in schema["secret"].get("properties", {}).values():
        assert prop.get("writeOnly") is True

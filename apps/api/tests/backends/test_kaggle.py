"""kaggle backend against a fake `kaggle` CLI: nothing here talks to Kaggle."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from backend_fakes import REPO_ROOT, Call, FakeRunner, result

from skf_api.backends._proc import ProcResult
from skf_api.backends.base import (
    BackendError,
    BackendKind,
    ExternalRef,
    HealthStatus,
    IngestConfig,
    InputFile,
    JobSpec,
    JobState,
    TargetContext,
)
from skf_api.backends.kaggle import KaggleBackend, kernel_state, load_pipeline_bundle, parse_kernel_log

KEY = "kaggle-key-not-real"
CONFIG = {"username": "skf-ml"}
KERNEL = "skf-ml/skf-g1-stairs-v12-train-stage123"


def train_spec(**overrides: object) -> JobSpec:
    fields: dict = {
        "run_id": "run-1",
        "run_name": "g1-stairs-v12",
        "stage_id": "stage123-4567",
        "stage_key": "train",
        "kind": "train",
        "timeout_minutes": 600,
        "argv": [
            "python",
            "-m",
            "g1pipe.train",
            "--task",
            "stairs",
            "--out",
            "{out_dir}",
            "--timesteps",
            "20000",
        ],
    }
    fields.update(overrides)
    return JobSpec(**fields)


class FakeKaggle:
    """Scripted `kaggle` CLI. `status` is what `kernels status` reports (None = 404)."""

    def __init__(self, status: str | None = None) -> None:
        self.status = status
        self.pushed: dict[str, object] = {}
        self.output_files: dict[str, bytes] = {}
        self.config_dirs: list[str] = []

    def all_config_dirs_removed(self) -> bool:
        return not any(os.path.exists(d) for d in self.config_dirs)

    def __call__(self, call: Call) -> ProcResult:
        assert KEY not in " ".join(call.argv), "the API key must never be on a command line"
        assert call.env is not None and call.env["KAGGLE_KEY"] == KEY
        config_dir = call.env["KAGGLE_CONFIG_DIR"]
        assert stat.S_IMODE(os.stat(config_dir).st_mode) == 0o700 and not os.listdir(config_dir)
        self.config_dirs.append(config_dir)
        args = call.argv[3:]  # after `python -m kaggle`
        if args[:2] == ["kernels", "status"]:
            if self.status is None:
                return result("", 1, "404 Client Error: Not Found for url")
            return result(f'{args[2]} has status "KernelWorkerStatus.{self.status.upper()}"\n')
        if args[:2] == ["kernels", "push"]:
            folder = Path(args[args.index("-p") + 1])
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            self.pushed = {"args": args, "metadata": metadata, "source": (folder / "kernel.py").read_text()}
            self.status = "queued"
            return result(
                f"Kernel version 1 successfully pushed.  Please check progress at "
                f"https://www.kaggle.com/code/{KERNEL}\n"
            )
        if args[:2] == ["kernels", "output"]:
            dest = Path(args[args.index("-p") + 1])
            pattern = args[args.index("--file-pattern") + 1] if "--file-pattern" in args else None
            for name, data in self.output_files.items():
                if pattern is None or name == "job.log" or name.endswith(".log"):
                    (dest / name).parent.mkdir(parents=True, exist_ok=True)
                    (dest / name).write_bytes(data)
            return result()
        if args[:2] == ["kernels", "list"]:
            return result("ref,title\nskf-ml/x,x\n")
        if args[:1] == ["quota"]:
            return result("resource,used,remaining,total,refreshAt\nGPU,12.50h,17.50h,30.00h,2026-09-27\n")
        raise AssertionError(f"unexpected kaggle call {args}")


def make(
    make_ctx: Callable[..., TargetContext],
    fake: FakeKaggle,
    clock: Callable[[], float] = lambda: 1000.0,
    secret: dict | None = None,
) -> tuple[KaggleBackend, FakeRunner]:
    runner = FakeRunner(fake)
    ctx = make_ctx(CONFIG, {"key": KEY} if secret is None else secret)
    return KaggleBackend(ctx, runner=runner, clock=clock), runner


def test_pipeline_bundle_comes_from_kaggle_job() -> None:
    bundle = load_pipeline_bundle(REPO_ROOT / "scripts" / "kaggle_job.py")
    assert "jax[cuda12]==0.7.2" in bundle.pins
    assert "train.py" in bundle.bundle
    assert bundle.noise[0] == "iterations limit reached"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('a/b has status "KernelWorkerStatus.RUNNING"', "running"),
        ('a/b has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "cancelacknowledged"),
        ('a/b has status "cancelAcknowledged"', "cancelacknowledged"),
        ('a/b has status "complete"', "complete"),
        ("404 Not Found", None),
    ],
)
def test_kernel_state_parses_old_and_new_cli_output(output: str, expected: str | None) -> None:
    assert kernel_state(output)[0] == expected


async def test_submit_pushes_a_private_kernel(make_ctx: Callable[..., TargetContext]) -> None:
    fake = FakeKaggle()
    backend, runner = make(make_ctx, fake)
    spec = train_spec(
        ingest=IngestConfig("https://studio.example.com/ingest/v1", "ingest-token"),
        env={"XLA_FLAGS": "--xla_gpu_triton_gemm_any=true"},
    )
    ref = await backend.submit(spec)

    assert ref == ExternalRef(
        BackendKind.KAGGLE, KERNEL, {"pushed_at": 1000.0}, f"https://www.kaggle.com/code/{KERNEL}"
    )
    metadata = fake.pushed["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["id"] == KERNEL and metadata["title"] == KERNEL.split("/")[1]
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    assert metadata["kernel_sources"] == []
    args = fake.pushed["args"]
    assert isinstance(args, list)
    assert args[args.index("--accelerator") + 1] == "NvidiaTeslaT4"
    assert args[args.index("--timeout") + 1] == "36000"

    source = fake.pushed["source"]
    assert isinstance(source, str)
    compile(source, "kernel.py", "exec")
    assert "jax[cuda12]==0.7.2" in source
    assert "def main(argv" in source  # the embedded reporter
    assert "'/kaggle/working/run'" in source and "{out_dir}" not in source
    assert "'--stop-when-revoked'" in source and "'SKF_INGEST_TOKEN': 'ingest-token'" in source
    assert "'XLA_FLAGS'" in source
    assert fake.config_dirs and fake.all_config_dirs_removed(), "temp config dirs are removed"
    assert [c.argv[3:5] for c in runner.calls] == [["kernels", "status"], ["kernels", "push"]]


async def test_submit_is_idempotent(make_ctx: Callable[..., TargetContext]) -> None:
    backend, runner = make(make_ctx, FakeKaggle(status="running"))
    ref = await backend.submit(train_spec())
    assert ref.job_id == KERNEL
    assert [c.argv[3:5] for c in runner.calls] == [["kernels", "status"]]


async def test_submit_never_pushes_when_the_kernel_state_is_unknown(
    make_ctx: Callable[..., TargetContext],
) -> None:
    runner = FakeRunner(lambda call: result("", 1, "503 Server Error: Service Unavailable"))
    backend = KaggleBackend(make_ctx(CONFIG, {"key": KEY}), runner=runner)
    with pytest.raises(BackendError) as info:
        await backend.submit(train_spec())
    assert info.value.retryable
    assert [c.argv[3:5] for c in runner.calls] == [["kernels", "status"]]


async def test_only_train_stages_run_on_kaggle(make_ctx: Callable[..., TargetContext]) -> None:
    backend, _ = make(make_ctx, FakeKaggle())
    with pytest.raises(BackendError, match="train stages only"):
        await backend.submit(train_spec(kind="evaluate"))


async def test_warm_start_mounts_the_parent_kernel(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    fake = FakeKaggle()
    backend, _ = make(make_ctx, fake)
    parent = ExternalRef(BackendKind.KAGGLE, "skf-ml/skf-g1-stairs-v11-train-aaaa")
    spec = train_spec(
        inputs=[InputFile("parent/params.pkl", tmp_path / "params.pkl", source_ref=parent)],
        argv=["python", "-m", "g1pipe.train", "--init-from", "{input_dir}/parent/params.pkl"],
    )
    await backend.submit(spec)
    metadata = fake.pushed["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["kernel_sources"] == ["skf-ml/skf-g1-stairs-v11-train-aaaa"]
    assert "'/tmp/skf-in/parent/params.pkl'" in str(fake.pushed["source"])


async def test_warm_start_from_elsewhere_is_refused(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    backend, _ = make(make_ctx, FakeKaggle())
    aws_parent = ExternalRef(BackendKind.AWS_EC2, "skf-1234")
    for source_ref in (None, aws_parent):
        spec = train_spec(inputs=[InputFile("parent/params.pkl", tmp_path / "p.pkl", source_ref=source_ref)])
        with pytest.raises(BackendError, match="needs a parent trained on Kaggle"):
            await backend.submit(spec)


async def test_push_error_is_reported(make_ctx: Callable[..., TargetContext]) -> None:
    fake = FakeKaggle()

    def handler(call: Call) -> ProcResult:
        if call.argv[3:5] == ["kernels", "push"]:
            return result("Kernel push error: Maximum weekly GPU quota reached\n")
        return fake(call)

    backend = KaggleBackend(make_ctx(CONFIG, {"key": KEY}), runner=FakeRunner(handler))
    with pytest.raises(BackendError, match="GPU quota"):
        await backend.submit(train_spec())


async def test_status_maps_states_and_counts_gpu_time(make_ctx: Callable[..., TargetContext]) -> None:
    now = [1000.0]
    fake = FakeKaggle(status="queued")
    backend, _ = make(make_ctx, fake, clock=lambda: now[0])
    ref = ExternalRef(BackendKind.KAGGLE, KERNEL, {"pushed_at": 900.0})

    status = await backend.status(ref)
    assert status.state is JobState.PROVISIONING and status.gpu_seconds is None

    fake.status, now[0] = "running", 1100.0
    status = await backend.status(ref)
    assert status.state is JobState.RUNNING and status.ref is not None
    ref = status.ref

    fake.status, now[0] = "complete", 1400.0
    status = await backend.status(ref)
    assert status.state is JobState.SUCCEEDED
    assert status.gpu_seconds == 300.0
    assert status.ref is not None and status.ref.data["ended_at"] == 1400.0

    for kaggle_status, state in (("error", JobState.FAILED), ("cancel_acknowledged", JobState.CANCELLED)):
        fake.status = kaggle_status
        assert (await backend.status(ref)).state is state


async def test_status_of_a_kernel_kaggle_does_not_list_yet(make_ctx: Callable[..., TargetContext]) -> None:
    now = [1000.0]
    backend, _ = make(make_ctx, FakeKaggle(status=None), clock=lambda: now[0])
    ref = ExternalRef(BackendKind.KAGGLE, KERNEL, {"pushed_at": 950.0})
    assert (await backend.status(ref)).state is JobState.PROVISIONING
    now[0] = 5000.0
    assert (await backend.status(ref)).state is JobState.FAILED


async def test_logs_arrive_once_the_kernel_finished(make_ctx: Callable[..., TargetContext]) -> None:
    fake = FakeKaggle(status="running")
    fake.output_files = {"job.log": b"line 1\nline 2\n", "run/params.pkl": b"weights"}
    backend, runner = make(make_ctx, fake)
    ref = ExternalRef(BackendKind.KAGGLE, KERNEL, {"pushed_at": 900.0})

    batch = await backend.fetch_logs(ref, None)
    assert batch.lines == [] and batch.cursor is None

    fake.status = "complete"
    batch = await backend.fetch_logs(ref, None)
    assert batch.lines == ["line 1", "line 2"] and batch.cursor == "done"
    output_call = next(c for c in runner.calls if c.argv[3:5] == ["kernels", "output"])
    assert output_call.argv[output_call.argv.index("--file-pattern") + 1] == r"^job\.log$"

    calls = len(runner.calls)
    assert (await backend.fetch_logs(ref, "done")).lines == []
    assert len(runner.calls) == calls


async def test_collect_unpacks_run_outputs_and_the_session_log(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    fake = FakeKaggle(status="complete")
    session_log = [
        {"stream_name": "stdout", "time": 1.0, "data": "pip install\n"},
        {"stream_name": "stderr", "time": 2.0, "data": "Traceback: boom\n"},
    ]
    fake.output_files = {
        "run/params.pkl": b"weights",
        "run/ckpt_00000100.pkl": b"c",
        "run/progress.csv": b"step\n",
        f"{KERNEL.split('/')[1]}.log": json.dumps(session_log).encode(),
    }
    backend, _ = make(make_ctx, fake)
    dest = tmp_path / "dest"
    await backend.collect(ExternalRef(BackendKind.KAGGLE, KERNEL), dest)
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["ckpt_00000100.pkl", "job.log", "params.pkl", "progress.csv"]
    assert (dest / "job.log").read_text() == "pip install\nTraceback: boom\n"


def test_parse_kernel_log_falls_back_to_text() -> None:
    assert parse_kernel_log("plain\ntext") == ["plain", "text"]


async def test_validate_reports_quota(make_ctx: Callable[..., TargetContext]) -> None:
    backend, runner = make(make_ctx, FakeKaggle())
    report = await backend.validate()
    assert report.status is HealthStatus.OK
    assert report.details == {"gpu_used_hours": 12.5, "gpu_remaining_hours": 17.5, "gpu_total_hours": 30.0}
    assert all(c.argv[3] in ("kernels", "quota") and "push" not in c.argv for c in runner.calls)


async def test_validate_without_key_or_with_bad_key(make_ctx: Callable[..., TargetContext]) -> None:
    backend, runner = make(make_ctx, FakeKaggle(), secret={})
    assert (await backend.validate()).status is HealthStatus.DOWN
    assert runner.calls == []

    def unauthorized(call: Call) -> ProcResult:
        return result("", 1, "401 Client Error: Unauthorized")

    backend = KaggleBackend(make_ctx(CONFIG, {"key": KEY}), runner=FakeRunner(unauthorized))
    report = await backend.validate()
    assert report.status is HealthStatus.DOWN and "401" in report.message


async def test_cancel_and_release_touch_nothing(make_ctx: Callable[..., TargetContext]) -> None:
    backend, runner = make(make_ctx, FakeKaggle(status="running"))
    await backend.cancel(ExternalRef(BackendKind.KAGGLE, KERNEL))
    await backend.release()
    assert runner.calls == []

"""aws_ec2 backend: EC2 via moto, ssh/scp/rsync via a fake runner. Nothing here reaches AWS."""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from backend_fakes import Call, FakeRunner, result
from botocore.exceptions import ClientError
from moto import mock_aws

from skf_api.backends._proc import ProcResult
from skf_api.backends.aws_ec2 import AwsEc2Backend, AwsEc2Config, AwsEc2Secret
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

REGION = "us-east-1"
SSH_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nnot-a-real-key\n-----END OPENSSH PRIVATE KEY-----"
CONFIG = {"region": REGION, "instance_name": "g1-train"}


@pytest.fixture
def ec2(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(var, "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        yield boto3.client("ec2", region_name=REGION)


def launch(ec2: Any, name: str = "g1-train", *, stopped: bool = True) -> str:
    response = ec2.run_instances(
        ImageId="ami-12c6146b",
        MinCount=1,
        MaxCount=1,
        InstanceType="g6e.2xlarge",
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": name}]}],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    if stopped:
        ec2.stop_instances(InstanceIds=[instance_id])
    return instance_id


def describe(ec2: Any, instance_id: str) -> dict[str, Any]:
    return ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]


def state(ec2: Any, instance_id: str) -> str:
    return describe(ec2, instance_id)["State"]["Name"]


def ref_for(instance_id: str, started_at: float = 0) -> ExternalRef:
    return ExternalRef(
        BackendKind.AWS_EC2,
        "skf-abcdef12",
        {
            "instance_id": instance_id,
            "home": "/home/ubuntu",
            "stage_id": "abcdef12",
            "started_at": started_at,
        },
    )


def bash_syntax_ok(script: str) -> bool:
    return subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def removed(paths: set[str]) -> bool:
    return not any(os.path.exists(p) for p in paths)


class RecordingClient:
    """Wraps the moto client: records method names, optionally fails one of them."""

    def __init__(self, client: Any, fail: dict[str, str] | None = None) -> None:
        self.client = client
        self.fail = fail or {}
        self.methods: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def call(**kwargs: Any) -> Any:
            self.methods.append(name)
            if name in self.fail:
                raise ClientError({"Error": {"Code": self.fail[name], "Message": "no capacity"}}, name)
            return getattr(self.client, name)(**kwargs)

        return call


class FakeBox:
    """Answers ssh/scp/rsync the way the box would; checks the key file while a command runs."""

    def __init__(self) -> None:
        self.job = "running"  # running | exit N | cancelled | gone
        self.sessions = ""
        self.log = b""
        self.uploaded: dict[str, str] = {}
        self.key_files: set[str] = set()

    def __call__(self, call: Call) -> ProcResult:
        program = call.argv[0]
        if program in ("ssh", "scp"):
            key = call.argv[call.argv.index("-i") + 1]
            assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
            assert Path(key).read_text().startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
            self.key_files.add(key)
            assert "StrictHostKeyChecking=accept-new" in call.argv
        if program == "scp":
            for local in call.argv[call.argv.index("ServerAliveInterval=30") + 1 : -1]:
                self.uploaded[Path(local).name] = Path(local).read_text()
            return result()
        if program == "rsync":
            return result()
        command = call.command
        if command.startswith('printf %s "$HOME"'):
            return result("/home/ubuntu")
        if "then echo started" in command:
            return result("")
        if command.startswith("if [ -e") and "echo cancelled" in command:
            return result(self.job + "\n")
        if "then echo done; else echo live" in command:
            marker = b"live" if self.job == "running" else b"done"
            return result(marker + b"\n" + self.log)
        if command.startswith("tmux list-sessions"):
            assert 'for d in "$HOME"/skf/*/' in command
            return result(self.sessions)
        return result()


def make(
    ctx: TargetContext, ec2: Any, box: FakeBox, fail: dict[str, str] | None = None
) -> tuple[AwsEc2Backend, FakeRunner, RecordingClient]:
    client = RecordingClient(ec2, fail)
    runner = FakeRunner(box)
    backend = AwsEc2Backend(
        ctx,
        runner=runner,
        ec2_factory=lambda config, secret: client,
        clock=lambda: 5000.0,
        poll_seconds=0,
        start_timeout=5,
        ssh_timeout=5,
    )
    return backend, runner, client


@pytest.fixture
def ctx(make_ctx: Any) -> TargetContext:
    return make_ctx(CONFIG, {"ssh_private_key": SSH_KEY})


def train_spec(tmp_path: Path) -> JobSpec:
    params = tmp_path / "params.pkl"
    params.write_bytes(b"w")
    return JobSpec(
        run_id="run-1",
        run_name="g1-stairs-v12",
        stage_id="abcdef12-3456",
        stage_key="train",
        kind="train",
        argv=[
            "python",
            "-m",
            "g1pipe.train",
            "--task",
            "stairs",
            "--out",
            "{out_dir}",
            "--init-from",
            "{input_dir}/parent/params.pkl",
            "--lr",
            "1e-4",
        ],
        inputs=[InputFile("parent/params.pkl", params)],
        env={"XLA_FLAGS": "a b"},
        ingest=IngestConfig("https://studio.example.com/ingest/v1", "tok'en"),
        timeout_minutes=60,
    )


async def test_submit_starts_the_box_and_the_tmux_session(
    ctx: TargetContext, ec2: Any, tmp_path: Path
) -> None:
    instance_id = launch(ec2)
    box = FakeBox()
    backend, runner, client = make(ctx, ec2, box)

    ref = await backend.submit(train_spec(tmp_path))

    assert state(ec2, instance_id) == "running"
    assert "start_instances" in client.methods
    assert ref.backend is BackendKind.AWS_EC2 and ref.job_id == "skf-abcdef12"
    assert ref.data == {
        "instance_id": instance_id,
        "home": "/home/ubuntu",
        "stage_id": "abcdef12-3456",
        "started_at": 5000.0,
    }
    assert ref.url is not None and instance_id in ref.url

    ssh = [c for c in runner.calls if c.argv[0] == "ssh"]
    assert all(f"HostKeyAlias={instance_id}" in c.argv for c in ssh)
    known_hosts = f"UserKnownHostsFile={ctx.work_dir / 'ssh' / 'known_hosts'}"
    assert all(known_hosts in c.argv for c in ssh)
    assert ssh[-1].command == "tmux new-session -d -s skf-abcdef12 bash /home/ubuntu/skf/abcdef12-3456/run.sh"

    rsync = next(c for c in runner.calls if c.argv[0] == "rsync")
    assert rsync.cwd == ctx.pipeline_repo_dir
    assert rsync.argv[-1].endswith(":/home/ubuntu/gtw/")
    assert "g1pipe" in rsync.argv and "apps/reporter" in rsync.argv and "--exclude=runs" in rsync.argv

    scp_input = next(c for c in runner.calls if c.argv[0] == "scp" and c.argv[-2].endswith("params.pkl"))
    assert scp_input.argv[-1].endswith(":/home/ubuntu/skf/abcdef12-3456/in/parent/params.pkl")

    run_sh, job_sh = box.uploaded["run.sh"], box.uploaded["job.sh"]
    assert "export SKF_INGEST_TOKEN='tok'\"'\"'en'" in run_sh
    assert (
        "exec python3 /home/ubuntu/gtw/apps/reporter/skf_reporter.py "
        "--log /home/ubuntu/skf/abcdef12-3456/job.log"
    ) in run_sh
    assert "--timeout 3600" in run_sh
    assert "flock /home/ubuntu/skf/.bootstrap.lock bash -c" in job_sh
    assert "MJX_ONLY=1 bash ~/gtw/scripts/aws_bootstrap.sh" in job_sh
    assert (
        "uv run --no-sync python -m g1pipe.train --task stairs --out /home/ubuntu/skf/abcdef12-3456/out "
        "--init-from /home/ubuntu/skf/abcdef12-3456/in/parent/params.pkl --lr 1e-4"
    ) in job_sh
    assert "XLA_FLAGS='a b'" in job_sh
    assert bash_syntax_ok(run_sh) and bash_syntax_ok(job_sh)
    assert box.key_files and removed(box.key_files), "key file removed after use"


async def test_submit_again_does_not_start_a_second_session(
    ctx: TargetContext, ec2: Any, tmp_path: Path
) -> None:
    launch(ec2, stopped=False)
    box = FakeBox()
    backend, runner, _ = make(ctx, ec2, box)
    original = box.__call__

    def already_started(call: Call) -> ProcResult:
        if call.argv[0] == "ssh" and "then echo started" in call.command:
            return result("started\n")
        return original(call)

    runner.handler = already_started
    await backend.submit(train_spec(tmp_path))
    assert not any(c.argv[0] in ("rsync", "scp") for c in runner.calls)
    assert not any("tmux new-session" in c.command for c in runner.calls if c.argv[0] == "ssh")


async def test_capacity_errors_are_retryable(ctx: TargetContext, ec2: Any, tmp_path: Path) -> None:
    launch(ec2)
    backend, runner, _ = make(ctx, ec2, FakeBox(), fail={"start_instances": "InsufficientInstanceCapacity"})
    with pytest.raises(BackendError) as info:
        await backend.submit(train_spec(tmp_path))
    assert info.value.retryable is True
    assert runner.calls == []


async def test_missing_box_is_an_error_unless_it_may_be_created(
    make_ctx: Any, ec2: Any, tmp_path: Path
) -> None:
    ctx = make_ctx(CONFIG, {"ssh_private_key": SSH_KEY})
    backend, _, _ = make(ctx, ec2, FakeBox())
    with pytest.raises(BackendError, match="no EC2 instance named g1-train"):
        await backend.submit(train_spec(tmp_path))

    subnet = ec2.describe_subnets()["Subnets"][0]["SubnetId"]
    group = ec2.describe_security_groups()["SecurityGroups"][0]["GroupId"]
    config = {
        **CONFIG,
        "create_if_missing": True,
        "ami_id": "ami-12c6146b",
        "subnet_id": subnet,
        "security_group_id": group,
        "key_name": "g1-train",
    }
    backend, _, client = make(make_ctx(config, {"ssh_private_key": SSH_KEY}), ec2, FakeBox())
    ref = await backend.submit(train_spec(tmp_path))
    assert "run_instances" in client.methods
    tags = describe(ec2, ref.data["instance_id"])["Tags"]
    assert {"Key": "Name", "Value": "g1-train"} in tags


def test_create_if_missing_needs_launch_settings() -> None:
    with pytest.raises(ValueError, match="create_if_missing needs"):
        AwsEc2Config.model_validate({**CONFIG, "create_if_missing": True})


@pytest.mark.parametrize(
    ("job", "expected", "message"),
    [
        ("exit 0", JobState.SUCCEEDED, None),
        ("exit 3", JobState.FAILED, "exit code 3"),
        ("exit 124", JobState.FAILED, "timed out"),
        ("cancelled", JobState.CANCELLED, "cancelled"),
        ("gone", JobState.FAILED, "the tmux session ended without an exit code"),
    ],
)
async def test_status_reads_exit_file_and_tmux(
    ctx: TargetContext, ec2: Any, job: str, expected: JobState, message: str | None
) -> None:
    instance_id = launch(ec2, stopped=False)
    box = FakeBox()
    backend, runner, _ = make(ctx, ec2, box)
    ref = ref_for(instance_id, 4000)

    running = await backend.status(ref)
    assert running.state is JobState.RUNNING and running.gpu_seconds == 1000.0

    box.job = job
    status = await backend.status(ref)
    assert (status.state, status.message) == (expected, message)
    assert status.gpu_seconds == 1000.0 and status.ref is not None

    calls = len(runner.calls)
    ec2.stop_instances(InstanceIds=[instance_id])  # idle auto-stop after the job ended
    again = await backend.status(status.ref)
    assert again.state is expected and len(runner.calls) == calls


async def test_status_when_ssh_is_unreachable_is_retryable(ctx: TargetContext, ec2: Any) -> None:
    instance_id = launch(ec2, stopped=False)
    backend, runner, _ = make(ctx, ec2, FakeBox())
    runner.handler = lambda call: result("", 255, "ssh: connect to host: Connection timed out")
    ref = ref_for(instance_id)
    with pytest.raises(BackendError) as info:
        await backend.status(ref)
    assert info.value.retryable


async def test_stopped_box_fails_an_unfinished_stage(ctx: TargetContext, ec2: Any) -> None:
    instance_id = launch(ec2, stopped=True)
    backend, runner, _ = make(ctx, ec2, FakeBox())
    ref = ref_for(instance_id)
    status = await backend.status(ref)
    assert status.state is JobState.FAILED and "stopped" in (status.message or "")
    assert runner.calls == []


async def test_fetch_logs_tails_by_byte_offset(ctx: TargetContext, ec2: Any) -> None:
    instance_id = launch(ec2, stopped=False)
    box = FakeBox()
    box.log = b"line 1\nline 2\npart"
    backend, runner, _ = make(ctx, ec2, box)
    ref = ref_for(instance_id)

    batch = await backend.fetch_logs(ref, "100")
    assert batch.lines == ["line 1", "line 2"] and batch.cursor == str(100 + 14)
    assert "tail -c +101 /home/ubuntu/skf/abcdef12/job.log" in runner.calls[-1].command

    box.job, box.log = "exit 0", b"part of the last line"
    batch = await backend.fetch_logs(ref, "114")
    assert batch.lines == ["part of the last line"] and batch.cursor == str(114 + 21)


async def test_collect_rsyncs_outputs_and_log(ctx: TargetContext, ec2: Any, tmp_path: Path) -> None:
    instance_id = launch(ec2, stopped=False)
    backend, runner, _ = make(ctx, ec2, FakeBox())
    ref = ref_for(instance_id)
    dest = tmp_path / "out"
    await backend.collect(ref, dest)
    rsyncs = [c.argv[-2:] for c in runner.calls if c.argv[0] == "rsync"]
    ip = describe(ec2, instance_id)["PublicIpAddress"]
    assert rsyncs == [
        [f"ubuntu@{ip}:/home/ubuntu/skf/abcdef12/out/", str(dest)],
        [f"ubuntu@{ip}:/home/ubuntu/skf/abcdef12/job.log", str(dest / "job.log")],
    ]


async def test_cancel_kills_the_session_and_marks_it(ctx: TargetContext, ec2: Any) -> None:
    instance_id = launch(ec2, stopped=False)
    backend, runner, _ = make(ctx, ec2, FakeBox())
    ref = ref_for(instance_id)
    await backend.cancel(ref)
    assert runner.calls[-1].command == (
        "touch /home/ubuntu/skf/abcdef12/cancelled; tmux kill-session -t skf-abcdef12 2>/dev/null; true"
    )

    ec2.stop_instances(InstanceIds=[instance_id])
    calls = len(runner.calls)
    await backend.cancel(ref)  # nothing runs on a stopped box
    assert len(runner.calls) == calls


@pytest.mark.parametrize(
    ("sessions", "stop_when_idle", "expected"),
    [
        ("", True, "stopped"),
        ("skf-other123\nnotebook\n", True, "running"),
        ("unreported /home/ubuntu/skf/abcdef12/\n", True, "running"),
        ("", False, "running"),
    ],
)
async def test_release_stops_an_idle_box(
    make_ctx: Any, ec2: Any, sessions: str, stop_when_idle: bool, expected: str
) -> None:
    instance_id = launch(ec2, stopped=False)
    box = FakeBox()
    box.sessions = sessions
    ctx = make_ctx({**CONFIG, "stop_when_idle": stop_when_idle}, {"ssh_private_key": SSH_KEY})
    backend, _, _ = make(ctx, ec2, box)
    await backend.release()
    assert state(ec2, instance_id) == expected


async def test_release_removes_the_dirs_of_reported_stages(
    ctx: TargetContext, ec2: Any, tmp_path: Path
) -> None:
    launch(ec2, stopped=False)
    backend, runner, _ = make(ctx, ec2, FakeBox())
    await backend.release()
    command = next(c.command for c in runner.calls if "tmux list-sessions" in c.command)

    stages = tmp_path / "skf"
    for name, markers in {"done": ["exit_code", "reported"], "unread": ["exit_code"], "live": []}.items():
        (stages / name / "out").mkdir(parents=True)
        for marker in markers:
            (stages / name / marker).write_text("0")
    (stages / ".bootstrapped").write_text("")
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
    shell = await asyncio.to_thread(
        subprocess.run, ["bash", "-c", command], env=env, capture_output=True, text=True
    )
    assert shell.returncode == 0
    assert sorted(p.name for p in stages.iterdir()) == [".bootstrapped", "live", "unread"]
    assert f"unreported {stages}/unread/" in shell.stdout


async def test_validate_only_describes(ctx: TargetContext, ec2: Any) -> None:
    instance_id = launch(ec2, stopped=True)
    backend, runner, client = make(ctx, ec2, FakeBox())
    report = await backend.validate()
    assert report.status is HealthStatus.OK
    assert report.details["instance_id"] == instance_id and report.details["state"] == "stopped"
    assert client.methods == ["describe_instances"] and runner.calls == []
    assert state(ec2, instance_id) == "stopped"


async def test_validate_reports_missing_box_or_key(make_ctx: Any, ec2: Any) -> None:
    backend, _, _ = make(make_ctx(CONFIG, {"ssh_private_key": SSH_KEY}), ec2, FakeBox())
    assert (await backend.validate()).status is HealthStatus.DOWN
    launch(ec2)
    backend, _, _ = make(make_ctx(CONFIG, {}), ec2, FakeBox())
    report = await backend.validate()
    assert report.status is HealthStatus.DOWN and "SSH private key" in report.message


def test_secret_fields_are_hidden_from_repr() -> None:
    secret = AwsEc2Secret(ssh_private_key=SSH_KEY, aws_secret_access_key="very-secret")  # noqa: S106
    assert "very-secret" not in repr(secret) and "not-a-real-key" not in repr(secret)

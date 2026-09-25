"""aws_ec2: runs a stage on a long-lived EC2 GPU box, the way scripts/aws_box.sh does by hand.

submit: find the box by its Name tag (optionally create it), start it if stopped, wait for SSH, rsync
the pipeline subset of the repo, upload inputs, and start a tmux session that runs the job under the
reporter. GPU capacity comes and goes per availability zone, so a target can list fallback boxes
(same setup, other zones): submit uses one that is already up, else starts the first that AWS has
capacity for. The one-time bootstrap (scripts/aws_bootstrap.sh) runs inside that session, guarded by a
marker file and a lock, so its output shows up in the stage log. Remote layout per stage:

    ~/skf/<stage_id>/{in/, out/, run.sh, job.sh, job.log, exit_code, cancelled, reported}

Everything needed to resume after a worker restart is in the ExternalRef (instance id, remote home,
tmux session). All boto3 calls run in a thread; every remote value is quoted with shlex.quote.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from skf_api.backends._job import (
    JOB_LOG,
    LOG_CHUNK_BYTES,
    REPORTER_RELPATH,
    ingest_env,
    input_relpath,
    render_argv,
    reporter_args,
    split_lines,
)
from skf_api.backends._proc import ProcessRunner, SubprocessRunner
from skf_api.backends._ssh import SshClient, SshEndpoint
from skf_api.backends.base import (
    BackendError,
    BackendKind,
    ExternalRef,
    HealthReport,
    HealthStatus,
    JobSpec,
    JobState,
    JobStatus,
    LogBatch,
    TargetContext,
)

log = logging.getLogger(__name__)

# What the box needs from the repo (the rest - runs/, .venv, third_party, results - stays local).
SYNC_PATHS = ("g1pipe", "jev_agent", "scripts", "skills", "apps/reporter", "pyproject.toml", "uv.lock")
SYNC_EXCLUDES = ("__pycache__", "*.pyc", ".venv", "runs", "third_party", "kaggle_jobs")
ACTIVE_STATES = ("pending", "running", "stopping", "stopped")
CAPACITY_ERRORS = {
    "InsufficientInstanceCapacity",
    "InsufficientHostCapacity",
    "InsufficientCapacity",
    "RequestLimitExceeded",
}
SESSION_PREFIX = "skf-"

# Idle auto-stop, as in scripts/aws_box.sh: every 5 min count idle minutes (GPU < 5 %, 5-min load < 0.5,
# nobody logged in); shut down (= stop, see InstanceInitiatedShutdownBehavior) after 60.
IDLE_STOP_USER_DATA = """#!/bin/bash
cat >/usr/local/bin/idle-stop.sh <<'EOS'
#!/bin/bash
STATE=/var/tmp/idle-minutes
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
logins=$(who | wc -l)
busy=$(awk '{print ($2 >= 0.5)}' /proc/loadavg)
if [ "${util:-0}" -lt 5 ] && [ "$logins" -eq 0 ] && [ "$busy" -eq 0 ]; then
  n=$(( $(cat $STATE 2>/dev/null || echo 0) + 5 ))
else
  n=0
fi
echo $n > $STATE
if [ "$n" -ge 60 ]; then echo 0 > $STATE; shutdown -h now; fi
EOS
chmod +x /usr/local/bin/idle-stop.sh
echo "*/5 * * * * root /usr/local/bin/idle-stop.sh" > /etc/cron.d/idle-stop
"""


class AwsEc2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", title="AWS EC2 GPU box")

    region: str = Field(..., min_length=1, description="e.g. us-east-1")
    instance_name: str = Field(..., min_length=1, description="Name tag of the training box")
    fallback_instance_names: list[str] = Field(
        default_factory=list,
        description="Other boxes (Name tags) to try in order when AWS has no capacity for the first, "
        "e.g. the same box in other availability zones",
    )
    instance_type: str = Field("g6e.2xlarge", description="Used when the box is created")
    ami_id: str | None = Field(None, description="Deep Learning AMI (Ubuntu); needed to create the box")
    subnet_id: str | None = None
    security_group_id: str | None = Field(None, description="Must allow SSH from the worker")
    key_name: str | None = Field(None, description="EC2 key pair matching the SSH private key")
    volume_gb: int = Field(250, ge=50, description="Root volume size when the box is created")
    ssh_user: str = "ubuntu"
    remote_repo_dir: str = "~/gtw"
    bootstrap_command: str = Field(
        "MJX_ONLY=1 bash ~/gtw/scripts/aws_bootstrap.sh", description="Runs once per box before the first job"
    )
    create_if_missing: bool = False
    stop_when_idle: bool = Field(True, description="Stop the box when no stage is using it any more")
    aws_profile: str | None = Field(None, description="Named AWS profile; empty = keys or instance role")

    @property
    def instance_names(self) -> list[str]:
        """The primary box first, then the fallbacks, without duplicates."""
        return list(dict.fromkeys([self.instance_name, *self.fallback_instance_names]))

    @model_validator(mode="after")
    def _creatable(self) -> AwsEc2Config:
        needed = ("ami_id", "subnet_id", "security_group_id", "key_name")
        missing = [f for f in needed if not getattr(self, f)]
        if self.create_if_missing and missing:
            raise ValueError(f"create_if_missing needs {', '.join(missing)}")
        return self


def _secret_field(description: str, *, required: bool = False) -> Any:
    extra = {"format": "password", "writeOnly": True}
    if required:
        return Field(..., min_length=1, repr=False, json_schema_extra=extra, description=description)
    return Field(None, repr=False, json_schema_extra=extra, description=description)


class AwsEc2Secret(BaseModel):
    model_config = ConfigDict(extra="forbid", title="AWS EC2 credentials")

    ssh_private_key: str = _secret_field("OpenSSH private key for the box's key pair", required=True)
    aws_access_key_id: str | None = _secret_field("Empty: default credential chain / instance role")
    aws_secret_access_key: str | None = _secret_field("Empty: default credential chain / instance role")
    aws_session_token: str | None = _secret_field("Only for temporary credentials")


Ec2ClientFactory = Callable[[AwsEc2Config, AwsEc2Secret | None], Any]


def boto3_ec2_client(config: AwsEc2Config, secret: AwsEc2Secret | None) -> Any:
    keys = {}
    if secret and secret.aws_access_key_id and secret.aws_secret_access_key:
        keys = {
            "aws_access_key_id": secret.aws_access_key_id,
            "aws_secret_access_key": secret.aws_secret_access_key,
            "aws_session_token": secret.aws_session_token or None,
        }
    session = boto3.Session(profile_name=config.aws_profile or None, region_name=config.region, **keys)
    return session.client("ec2")


@dataclass(frozen=True)
class RemotePaths:
    home: str
    stage_id: str
    repo_setting: str

    @property
    def repo(self) -> str:
        if self.repo_setting == "~" or self.repo_setting.startswith("~/"):
            return self.home + self.repo_setting[1:]
        return self.repo_setting

    @property
    def root(self) -> str:
        return f"{self.home}/skf"

    @property
    def stage(self) -> str:
        return f"{self.root}/{self.stage_id}"

    @property
    def in_dir(self) -> str:
        return f"{self.stage}/in"

    @property
    def out_dir(self) -> str:
        return f"{self.stage}/out"

    @property
    def log(self) -> str:
        return f"{self.stage}/{JOB_LOG}"

    @property
    def exit_file(self) -> str:
        return f"{self.stage}/exit_code"

    @property
    def cancelled(self) -> str:
        return f"{self.stage}/cancelled"

    @property
    def reported(self) -> str:
        return f"{self.stage}/reported"

    @property
    def run_script(self) -> str:
        return f"{self.stage}/run.sh"

    @property
    def job_script(self) -> str:
        return f"{self.stage}/job.sh"


class AwsError(BackendError):
    """A failed AWS API call, with the AWS error code."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, retryable=code in CAPACITY_ERRORS)
        self.code = code


class AwsEc2Backend:
    kind = BackendKind.AWS_EC2

    def __init__(
        self,
        ctx: TargetContext,
        *,
        runner: ProcessRunner | None = None,
        ec2_factory: Ec2ClientFactory = boto3_ec2_client,
        clock: Callable[[], float] = time.time,
        poll_seconds: float = 5.0,
        start_timeout: float = 600.0,
        ssh_timeout: float = 300.0,
    ) -> None:
        self.ctx = ctx
        self.config = AwsEc2Config.model_validate(ctx.config)
        self.secret = AwsEc2Secret.model_validate(ctx.secret) if ctx.secret else None
        self.runner = runner or SubprocessRunner()
        self.clock = clock
        self.poll_seconds = poll_seconds
        self.start_timeout = start_timeout
        self.ssh_timeout = ssh_timeout
        self._ec2_factory = ec2_factory
        self._client: Any = None

    # ---- ComputeBackend -------------------------------------------------------------------------

    async def validate(self) -> HealthReport:
        try:
            boxes = await self._find_instances()
        except BackendError as exc:
            return HealthReport(HealthStatus.DOWN, str(exc))
        if self.secret is None:
            return HealthReport(HealthStatus.DOWN, "SSH private key not set for this target")
        found = [(name, box) for name, box in boxes.items() if box is not None]
        if not found:
            if self.config.create_if_missing:
                message = f"no instance named {self.config.instance_name} yet; created on first run"
                return HealthReport(HealthStatus.OK, message)
            names = ", ".join(self.config.instance_names)
            return HealthReport(HealthStatus.DOWN, f"no instance named {names} in {self.config.region}")
        summaries = [
            {
                "name": name,
                "instance_id": box["InstanceId"],
                "instance_type": box["InstanceType"],
                "state": box["State"]["Name"],
                "availability_zone": box.get("Placement", {}).get("AvailabilityZone"),
                "public_ip": box.get("PublicIpAddress"),
            }
            for name, box in found
        ]
        missing = [name for name, box in boxes.items() if box is None]
        message = "; ".join(
            f"{b['instance_id']} ({b['instance_type']}, {b['availability_zone']}) is {b['state']}"
            for b in summaries
        )
        if missing:
            message += f"; not found: {', '.join(missing)}"
        details = {**summaries[0], "boxes": summaries}
        status = HealthStatus.DEGRADED if missing else HealthStatus.OK
        return HealthReport(status, message, details)

    async def submit(self, spec: JobSpec) -> ExternalRef:
        instance = await self._acquire()
        async with self._ssh(instance) as ssh:
            await self._wait_for_ssh(ssh)
            home = (await ssh.check('printf %s "$HOME"')).text.strip()
            paths = RemotePaths(home, spec.stage_id, self.config.remote_repo_dir)
            session = SESSION_PREFIX + spec.stage_id[:8]
            ref = self._ref(instance["InstanceId"], paths, session)
            started = await ssh.check(
                f"if [ -e {q(paths.exit_file)} ] || tmux has-session -t {q(session)} 2>/dev/null; "
                "then echo started; fi"
            )
            if started.text.strip() == "started":  # submitted before (worker restarted mid-submit)
                return ref
            await self._upload(ssh, spec, paths)
            await ssh.check(f"tmux new-session -d -s {q(session)} bash {q(paths.run_script)}")
        return ref

    async def status(self, ref: ExternalRef) -> JobStatus:
        data = dict(ref.data)
        if "exit_code" in data:  # finished earlier; the box may be stopped by now
            return self._finished_status(ref, data)
        instance = await self._describe(data["instance_id"])
        state = instance["State"]["Name"] if instance else "terminated"
        if state == "pending":
            return JobStatus(JobState.PROVISIONING, "instance starting")
        if instance is None or state != "running":
            return JobStatus(JobState.FAILED, f"instance is {state}; the stage did not finish")
        paths = self._paths(ref)
        # A finished stage is marked `reported` once its outcome is read here: release() must not stop
        # the box between a job ending and the reconciler seeing its exit code.
        reported = f"touch {q(paths.reported)}"
        async with self._ssh(instance) as ssh:
            probe = await ssh.run(
                f"if [ -e {q(paths.cancelled)} ]; then echo cancelled; {reported}; "
                f"elif [ -e {q(paths.exit_file)} ]; then echo exit $(cat {q(paths.exit_file)}); {reported}; "
                f"elif tmux has-session -t {q(ref.job_id)} 2>/dev/null; then echo running; "
                f"else echo gone; {reported}; fi",
                timeout_s=60,
            )
        if not probe.ok:
            raise BackendError(
                f"cannot check the stage on {data['instance_id']}: {probe.error_text}", retryable=True
            )
        word, _, rest = probe.text.strip().partition(" ")
        now = self.clock()
        if word == "running":
            return JobStatus(JobState.RUNNING, gpu_seconds=now - float(data["started_at"]))
        data["ended_at"] = now
        if word == "cancelled":
            data["exit_code"] = "cancelled"
        elif word == "exit" and rest.strip().lstrip("-").isdigit():
            data["exit_code"] = int(rest)
        else:
            data["exit_code"] = "lost"
        return self._finished_status(ExternalRef(ref.backend, ref.job_id, data, ref.url), data)

    async def fetch_logs(self, ref: ExternalRef, cursor: str | None) -> LogBatch:
        offset = int(cursor or 0)
        instance = await self._describe(ref.data["instance_id"])
        if instance is None or instance["State"]["Name"] != "running":
            return LogBatch([], cursor)
        paths = self._paths(ref)
        async with self._ssh(instance) as ssh:
            result = await ssh.run(
                f"if [ -e {q(paths.exit_file)} ]; then echo done; else echo live; fi; "
                f"tail -c +{offset + 1} {q(paths.log)} 2>/dev/null | head -c {LOG_CHUNK_BYTES}",
                timeout_s=120,
            )
        if not result.ok:
            raise BackendError(f"cannot read the stage log: {result.error_text}", retryable=True)
        marker, _, chunk = result.stdout.partition(b"\n")
        lines, used = split_lines(chunk, final=marker == b"done" and len(chunk) < LOG_CHUNK_BYTES)
        return LogBatch(lines, str(offset + used))

    async def collect(self, ref: ExternalRef, dest: Path) -> None:
        instance = await self._ensure_running(ref.data["instance_id"])
        paths = self._paths(ref)
        await asyncio.to_thread(dest.mkdir, parents=True, exist_ok=True)
        async with self._ssh(instance) as ssh:
            await self._wait_for_ssh(ssh)
            await ssh.rsync_down(f"{paths.out_dir}/", dest)
            await ssh.rsync_down(paths.log, dest / JOB_LOG)

    async def cancel(self, ref: ExternalRef) -> None:
        instance = await self._describe(ref.data["instance_id"])
        if instance is None or instance["State"]["Name"] != "running":
            return  # nothing runs on a stopped box
        paths = self._paths(ref)
        async with self._ssh(instance) as ssh:
            # kill-session hangs up the reporter, which stops the job and flushes its last lines
            await ssh.check(
                f"touch {q(paths.cancelled)}; tmux kill-session -t {q(ref.job_id)} 2>/dev/null; true"
            )

    async def release(self) -> None:
        for instance in (await self._find_instances()).values():
            if instance is not None and instance["State"]["Name"] == "running":
                await self._release_box(instance)

    async def _release_box(self, instance: dict[str, Any]) -> None:
        async with self._ssh(instance) as ssh:
            # A `reported` stage's outcome has been read, and it is collected by now (release is only called
            # once no stage uses the target), so its dir - inputs, checkpoints, log - is removed.
            busy = await ssh.run(
                "tmux list-sessions -F '#{session_name}' 2>/dev/null; "
                'for d in "$HOME"/skf/*/; do '
                '[ -e "$d/exit_code" ] && [ ! -e "$d/reported" ] && echo "unreported $d"; '
                '[ -e "$d/reported" ] && rm -rf "$d"; done; true',
                timeout_s=60,
            )
        if not self.config.stop_when_idle:
            return
        if not busy.ok:
            # leave it running; the box's own idle auto-stop is the safety net
            log.warning("not stopping %s: SSH check failed: %s", instance["InstanceId"], busy.error_text)
            return
        if any(line.startswith((SESSION_PREFIX, "unreported ")) for line in busy.text.splitlines()):
            return
        await self._ec2("stop_instances", InstanceIds=[instance["InstanceId"]])

    # ---- instance --------------------------------------------------------------------------------

    async def _ec2(self, method: str, **kwargs: Any) -> Any:
        try:
            if self._client is None:
                self._client = await asyncio.to_thread(self._ec2_factory, self.config, self.secret)
            return await asyncio.to_thread(getattr(self._client, method), **kwargs)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            raise AwsError(f"AWS {method}: {error.get('Message', exc)}", error.get("Code", "")) from exc
        except BotoCoreError as exc:
            raise BackendError(f"AWS {method}: {exc}") from exc

    async def _find_instances(self) -> dict[str, dict[str, Any] | None]:
        """Every candidate box by Name tag, in preference order (None where no such box exists)."""
        names = self.config.instance_names
        response = await self._ec2(
            "describe_instances",
            Filters=[
                {"Name": "tag:Name", "Values": names},
                {"Name": "instance-state-name", "Values": list(ACTIVE_STATES)},
            ],
        )
        by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for reservation in response["Reservations"]:
            for instance in reservation["Instances"]:
                tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                if tags.get("Name") in by_name:
                    by_name[tags["Name"]].append(instance)
        for name, instances in by_name.items():
            if len(instances) > 1:
                raise BackendError(f"{len(instances)} instances are named {name}; keep one")
        return {name: instances[0] if instances else None for name, instances in by_name.items()}

    async def _acquire(self) -> dict[str, Any]:
        """The box for a new stage: a candidate that is already up, else the first one AWS can start."""
        boxes = await self._find_instances()
        for instance in boxes.values():
            if instance is not None and instance["State"]["Name"] in ("pending", "running"):
                return await self._wait_running(instance["InstanceId"])
        busy: list[str] = []
        for name, instance in boxes.items():
            if instance is None:
                if name != self.config.instance_name or not self.config.create_if_missing:
                    continue
                instance = await self._create_instance()
            try:
                return await self._ensure_running(instance["InstanceId"])
            except BackendError as exc:
                if not exc.retryable:
                    raise
                busy.append(f"{name}: {exc}")  # no capacity in its zone, or still stopping: try the next
        if not busy:
            c = self.config
            raise BackendError(f"no EC2 instance named {', '.join(c.instance_names)} in {c.region}")
        raise BackendError("; ".join(busy), retryable=True)

    async def _describe(self, instance_id: str) -> dict[str, Any] | None:
        try:
            response = await self._ec2("describe_instances", InstanceIds=[instance_id])
        except AwsError as exc:
            if exc.code.startswith("InvalidInstanceID"):
                return None
            raise
        instances = [i for r in response["Reservations"] for i in r["Instances"]]
        return instances[0] if instances else None

    async def _ensure_running(self, instance_id: str) -> dict[str, Any]:
        instance = await self._describe(instance_id)
        if instance is None:
            raise BackendError(f"EC2 instance {instance_id} no longer exists")
        state = instance["State"]["Name"]
        if state == "stopping":
            raise BackendError("the instance is still stopping; retrying shortly", retryable=True)
        if state == "stopped":
            await self._ec2("start_instances", InstanceIds=[instance["InstanceId"]])
        elif state not in ("pending", "running"):
            raise BackendError(f"instance {instance['InstanceId']} is {state}")
        return await self._wait_running(instance["InstanceId"])

    async def _wait_running(self, instance_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.start_timeout
        while True:
            instance = await self._describe(instance_id)
            if instance and instance["State"]["Name"] == "running" and instance.get("PublicIpAddress"):
                return instance
            if time.monotonic() > deadline:
                message = f"instance {instance_id} did not reach running with a public IP"
                raise BackendError(message, retryable=True)
            await asyncio.sleep(self.poll_seconds)

    async def _create_instance(self) -> dict[str, Any]:
        c = self.config
        tags = [{"Key": "Name", "Value": c.instance_name}, {"Key": "ManagedBy", "Value": "skf-skill-studio"}]
        response = await self._ec2(
            "run_instances",
            ImageId=c.ami_id,
            InstanceType=c.instance_type,
            KeyName=c.key_name,
            MinCount=1,
            MaxCount=1,
            UserData=IDLE_STOP_USER_DATA,
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": c.subnet_id,
                    "Groups": [c.security_group_id],
                    "AssociatePublicIpAddress": True,
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": c.volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
                }
            ],
            MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
            InstanceInitiatedShutdownBehavior="stop",
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": tags},
                {"ResourceType": "volume", "Tags": tags},
            ],
        )
        return response["Instances"][0]

    # ---- remote job ------------------------------------------------------------------------------

    def _ssh(self, instance: dict[str, Any]) -> SshClient:
        if self.secret is None:
            raise BackendError("SSH private key not set for this target")
        endpoint = SshEndpoint(instance["PublicIpAddress"], self.config.ssh_user, instance["InstanceId"])
        return SshClient(
            self.runner,
            endpoint,
            private_key=self.secret.ssh_private_key,
            known_hosts=self.ctx.work_dir / "ssh" / "known_hosts",
        )

    async def _wait_for_ssh(self, ssh: SshClient) -> None:
        deadline = time.monotonic() + self.ssh_timeout
        while True:
            result = await ssh.run("true", timeout_s=30)
            if result.ok:
                return
            if time.monotonic() > deadline:
                raise ssh.error("SSH did not come up", result)
            await asyncio.sleep(self.poll_seconds)

    async def _upload(self, ssh: SshClient, spec: JobSpec, paths: RemotePaths) -> None:
        inputs = [(item, f"{paths.in_dir}/{input_relpath(item.name)}") for item in spec.inputs]
        dirs = {paths.in_dir, paths.out_dir, paths.repo, *(r.rsplit("/", 1)[0] for _, r in inputs)}
        await ssh.check("mkdir -p " + " ".join(q(d) for d in sorted(dirs)))
        sources = [p for p in SYNC_PATHS if (self.ctx.pipeline_repo_dir / p).exists()]
        await ssh.rsync_up(sources, paths.repo, cwd=self.ctx.pipeline_repo_dir, excludes=SYNC_EXCLUDES)
        for item, remote in inputs:
            await ssh.upload([item.local_path], remote)
        with tempfile.TemporaryDirectory(prefix="skf-aws-") as tmp:
            scripts = {"run.sh": self._run_script(spec, paths), "job.sh": self._job_script(spec, paths)}
            for name, text in scripts.items():
                (Path(tmp) / name).write_text(text)
                (Path(tmp) / name).chmod(0o600)  # run.sh holds the stage's ingest token
            await ssh.upload([Path(tmp) / name for name in scripts], paths.stage + "/")

    def _run_script(self, spec: JobSpec, paths: RemotePaths) -> str:
        """Session entry point: the reporter wrapping job.sh."""
        progress = f"{paths.out_dir}/{spec.progress_csv}" if spec.progress_csv else None
        reporter = [
            f"{paths.repo}/{REPORTER_RELPATH}",
            *reporter_args(
                log=paths.log,
                exit_file=paths.exit_file,
                progress=progress,
                timeout_minutes=spec.timeout_minutes,
            ),
        ]
        exports = "".join(f"export {k}={q(v)}\n" for k, v in ingest_env(spec.ingest, spec.stage_id).items())
        return (
            "#!/usr/bin/env bash\n"
            f"# SKF Skill Studio: run {spec.run_name}, stage {spec.stage_key} ({spec.stage_id})\n"
            "umask 077\n"
            f"{exports}"
            f"exec python3 {shlex.join(reporter)} -- bash {q(paths.job_script)}\n"
        )

    def _job_script(self, spec: JobSpec, paths: RemotePaths) -> str:
        """Bootstrap once per box, then the pipeline command itself."""
        marker, lock = f"{paths.root}/.bootstrapped", f"{paths.root}/.bootstrap.lock"
        bootstrap = f"[ -e {q(marker)} ] || {{ {self.config.bootstrap_command} && touch {q(marker)}; }}"
        # `uv run --no-sync`: use the bootstrapped venv as is; a sync would drop the CUDA jax wheels
        argv = render_argv(spec, ["uv", "run", "--no-sync", "python"], paths.out_dir, paths.in_dir)
        env = {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl", **spec.env}
        return (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'export PATH="$HOME/.local/bin:$PATH"\n'
            f"flock {q(lock)} bash -c {q(bootstrap)}\n"
            f"cd {q(paths.repo)}\n"
            f"exec env {' '.join(f'{k}={q(v)}' for k, v in env.items())} {shlex.join(argv)}\n"
        )

    def _paths(self, ref: ExternalRef) -> RemotePaths:
        return RemotePaths(ref.data["home"], ref.data["stage_id"], self.config.remote_repo_dir)

    def _ref(self, instance_id: str, paths: RemotePaths, session: str) -> ExternalRef:
        region = self.config.region
        url = f"https://{region}.console.aws.amazon.com/ec2/home?region={region}#InstanceDetails:instanceId={instance_id}"
        return ExternalRef(
            BackendKind.AWS_EC2,
            session,
            {
                "instance_id": instance_id,
                "home": paths.home,
                "stage_id": paths.stage_id,
                "started_at": self.clock(),
            },
            url,
        )

    def _finished_status(self, ref: ExternalRef, data: dict[str, Any]) -> JobStatus:
        code = data["exit_code"]
        gpu_seconds = float(data["ended_at"]) - float(data["started_at"])
        if code == "cancelled":
            return JobStatus(JobState.CANCELLED, "cancelled", gpu_seconds=gpu_seconds, ref=ref)
        if code == "lost":
            return JobStatus(
                JobState.FAILED,
                "the tmux session ended without an exit code",
                gpu_seconds=gpu_seconds,
                ref=ref,
            )
        if code == 0:
            return JobStatus(JobState.SUCCEEDED, gpu_seconds=gpu_seconds, ref=ref)
        message = "timed out" if code == 124 else f"exit code {code}"
        return JobStatus(JobState.FAILED, message, gpu_seconds=gpu_seconds, ref=ref)


def q(value: str) -> str:
    return shlex.quote(value)

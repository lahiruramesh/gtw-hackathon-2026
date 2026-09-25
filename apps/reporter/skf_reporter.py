#!/usr/bin/env python3
"""SKF Skill Studio in-job reporter (stdlib only, Python >= 3.8).

Wraps one pipeline command wherever it runs (worker host, AWS box, Kaggle kernel):

    python3 skf_reporter.py [--log FILE] [--exit-file FILE] [--progress CSV] [--noise SUBSTR]... -- CMD...

- runs CMD with stdout and stderr merged, drops known solver noise at the source and counts it,
  writes the kept lines to --log and to stdout;
- when SKF_INGEST_URL, SKF_INGEST_TOKEN and SKF_STAGE_ID are set, ships log batches (every 2 s or
  500 lines), tails --progress into metrics and heartbeats every 30 s. Ingest problems are retried
  with backoff and then dropped: they never slow down or stop CMD;
- forwards SIGTERM/SIGINT/SIGHUP to CMD and exits with CMD's exit code, also written to --exit-file.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Sequence

# Must stay identical to NOISE in scripts/kaggle_job.py (tests/backends/test_reporter.py checks it).
# MuJoCo Warp prints a solver-overflow note every step (~14 MB/s) that would otherwise flood the log.
DEFAULT_NOISE = (
    "iterations limit reached",
    "To disable the print warning",
    "Warning",
    "warnings.warn",
    # GPU printf output from parallel threads interleaves mid-line; these tails survive intact
    "warn_overflow",
    "mjw.OverflowType",
)

MAX_LINE_CHARS = 4000  # the API truncates to this anyway
FLUSH_LINES = 500  # ship a batch at least every --flush-seconds or this many lines
MAX_BATCH_LINES = 1000  # ingest limits: <= 1000 lines and <= 1 MB per request
MAX_BATCH_BYTES = 900_000
MAX_BUFFERED_LINES = 50_000  # beyond this (ingest down for long), new lines are not shipped
METRICS_PER_REQUEST = 200
KILL_GRACE_SECONDS = 20.0  # after forwarding a signal, SIGKILL the job if it is still running
SHUTDOWN_GRACE_SECONDS = 30.0  # time allowed for the final flush once the job has exited
REVOKED_STATUSES = (401, 403, 404, 410)
REVOKED_LIMIT = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestClient:
    """POSTs JSON to the studio's ingest API with retries (exponential backoff, full jitter)."""

    def __init__(
        self,
        base_url: str,
        token: str,
        stage_id: str,
        attempts: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        timeout: float = 10.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("SKF_INGEST_URL must be an http(s) URL")
        self.base = f"{base_url.rstrip('/')}/stages/{stage_id}"
        self.token = token
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.deadline: float | None = None  # set during shutdown to bound the final retries
        self.last_status: int | None = None

    def post(self, endpoint: str, payload: dict) -> bool:
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.attempts):
            request = urllib.request.Request(  # noqa: S310 - the scheme is checked in __init__
                f"{self.base}/{endpoint}",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    self.last_status = response.status
                    return True
            except urllib.error.HTTPError as exc:
                self.last_status = exc.code
                if 400 <= exc.code < 500 and exc.code not in (408, 429):
                    return False  # the request itself is wrong; retrying won't help
            except (urllib.error.URLError, OSError, ValueError):
                self.last_status = None
            delay = random.uniform(0, min(self.max_delay, self.base_delay * 2**attempt))  # noqa: S311 - jitter
            if attempt == self.attempts - 1 or (self.deadline and time.time() + delay > self.deadline):
                break
            time.sleep(delay)
        return False


class ProgressTail:
    """Incrementally reads a CSV that the job appends to (g1pipe.train's progress.csv)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.offset = 0
        self.header: list[str] | None = None

    def read_points(self) -> list[dict]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self.offset:  # file was rewritten: start again
            self.offset, self.header = 0, None
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        points = []
        for cells in csv.reader(data[:end].decode("utf-8", errors="replace").splitlines()):
            if not cells:
                continue
            if self.header is None:
                self.header = cells
                continue
            point = self._point(dict(zip(self.header, cells)))
            if point is not None:
                points.append(point)
        return points

    @staticmethod
    def _point(row: dict[str, str]) -> dict | None:
        step = _number(row.get("step"))
        if step is None:
            return None
        values = {}
        for key, cell in row.items():
            if key in ("step", "wall_s"):
                continue
            value = _number(cell)
            if value is not None:
                values[key] = value
        point = {"step": int(step), "values": values}
        wall_s = _number(row.get("wall_s"))
        if wall_s is not None:
            point["wall_s"] = wall_s
        return point


def _number(cell: str | None) -> float | None:
    try:
        value = float(cell) if cell not in (None, "") else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


class Shipper(threading.Thread):
    """Background thread: log batches, progress metrics and heartbeats to the ingest API."""

    def __init__(
        self,
        client: IngestClient,
        progress: str | None,
        flush_seconds: float,
        metrics_seconds: float,
        heartbeat_seconds: float,
        on_revoked: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(name="skf-reporter-shipper", daemon=True)
        self.client = client
        self.progress = ProgressTail(progress) if progress else None
        self.flush_seconds = flush_seconds
        self.metrics_seconds = metrics_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.on_revoked = on_revoked
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.lines: list[dict] = []
        self.noise_pending = 0
        self.unshipped = 0
        self.last_step: int | None = None
        self.revoked_count = 0

    def add_line(self, text: str) -> None:
        with self.lock:
            if len(self.lines) >= MAX_BUFFERED_LINES:
                self.unshipped += 1
                return
            self.lines.append({"ts": utc_now(), "text": text[:MAX_LINE_CHARS], "stream": "stdout"})
            full = len(self.lines) >= FLUSH_LINES
        if full:
            self.wake.set()

    def add_noise(self) -> None:
        with self.lock:
            self.noise_pending += 1

    def run(self) -> None:
        next_flush = next_metrics = time.time()
        next_heartbeat = time.time() + min(5.0, self.heartbeat_seconds)
        while not self.stopping.is_set():
            self.wake.wait(timeout=0.5)
            self.wake.clear()
            now = time.time()
            if now >= next_flush or len(self.lines) >= FLUSH_LINES:
                self.flush_logs()
                next_flush = now + self.flush_seconds
            if self.progress and now >= next_metrics:
                self.ship_metrics()
                next_metrics = now + self.metrics_seconds
            if now >= next_heartbeat:
                self.heartbeat()
                next_heartbeat = now + self.heartbeat_seconds

    def finish(self, grace: float) -> None:
        """Stop the loop and ship whatever is left, within about `grace` seconds."""
        self.client.deadline = time.time() + grace
        self.stopping.set()
        self.wake.set()
        self.join(timeout=grace)
        self.flush_logs(everything=True)
        if self.progress:
            self.ship_metrics()
        self.heartbeat()

    def flush_logs(self, everything: bool = False) -> None:
        while True:
            with self.lock:
                batch, size = [], 0
                for line in self.lines[:MAX_BATCH_LINES]:
                    size += len(json.dumps(line)) + 2
                    if batch and size > MAX_BATCH_BYTES:
                        break
                    batch.append(line)
                del self.lines[: len(batch)]
                noise, self.noise_pending = self.noise_pending, 0
            if not batch and not noise:
                return
            payload: dict = {"lines": batch}
            if noise:
                payload["noise_dropped"] = noise
            if not self.client.post("logs", payload):
                with self.lock:
                    self.unshipped += len(batch)
                    self.noise_pending += noise
                return
            if not everything and len(self.lines) < FLUSH_LINES:
                return

    def ship_metrics(self) -> None:
        assert self.progress is not None
        points = self.progress.read_points()
        for i in range(0, len(points), METRICS_PER_REQUEST):
            self.client.post("metrics", {"points": points[i : i + METRICS_PER_REQUEST]})
        if points:
            self.last_step = points[-1]["step"]

    def heartbeat(self) -> None:
        # The studio shows how long the stage has run; the message only adds what it cannot know.
        message = f"step {self.last_step:,}" if self.last_step is not None else ""
        if self.client.post("heartbeat", {"message": message}):
            self.revoked_count = 0
        elif self.client.last_status in REVOKED_STATUSES:
            self.revoked_count += 1
            if self.revoked_count >= REVOKED_LIMIT and self.on_revoked:
                self.on_revoked()


class Job:
    """The wrapped command: output filtering, signal forwarding, timeout."""

    def __init__(
        self, cmd: Sequence[str], noise: Sequence[str], log_path: str | None, shipper: Shipper | None
    ) -> None:
        self.cmd = list(cmd)
        self.noise = tuple(noise)
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None  # noqa: SIM115
        self.shipper = shipper
        self.proc: subprocess.Popen | None = None
        self.dropped = 0
        self.forced_code: int | None = None
        self.notes: list[str] = []

    def emit(self, text: str) -> None:
        if self.log:
            self.log.write(text + "\n")
            self.log.flush()
        try:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass  # nobody reads our stdout any more; keep the job going
        if self.shipper:
            self.shipper.add_line(text)

    def stop(self, sig: int, reason: str, code: int | None = None) -> None:
        # Called from signal handlers and timer threads, so it only records why and signals the job;
        # the main thread writes the notes to the log once the job has exited.
        self.notes.append(reason)
        if code is not None and self.forced_code is None:
            self.forced_code = code
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(sig)
        except OSError:
            return
        timer = threading.Timer(KILL_GRACE_SECONDS, self._kill)
        timer.daemon = True
        timer.start()

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(OSError):
                self.proc.kill()

    def run(self, timeout: float | None) -> int:
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            self.emit(f"[skf-reporter] cannot start {self.cmd[0]}: {exc}")
            return 127
        if timeout:
            reason = f"timeout after {timeout:.0f}s"
            timer = threading.Timer(timeout, self.stop, (signal.SIGTERM, reason, 124))
            timer.daemon = True
            timer.start()
        assert self.proc.stdout is not None
        for raw in iter(lambda: self.proc.stdout.readline(65536), b""):  # type: ignore[union-attr]
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if any(n in line for n in self.noise):
                self.dropped += 1
                if self.shipper:
                    self.shipper.add_noise()
                continue
            self.emit(line)
        rc = self.proc.wait()
        if rc < 0:
            rc = 128 - rc  # killed by signal N -> 128 + N, like a shell
        return self.forced_code if self.forced_code is not None else rc


def write_exit_file(path: str, code: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{code}\n")
    os.replace(tmp, path)


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    argv = list(argv)
    if "--" not in argv:
        raise SystemExit("usage: skf_reporter.py [options] -- CMD...")
    split = argv.index("--")
    ap = argparse.ArgumentParser(prog="skf_reporter.py")
    ap.add_argument("--log", help="append kept output lines here")
    ap.add_argument("--exit-file", help="write the job's exit code here when it ends")
    ap.add_argument("--progress", help="CSV the job appends metrics to (step, wall_s, ...)")
    ap.add_argument(
        "--noise",
        action="append",
        default=None,
        help="drop lines containing this substring (repeatable; replaces the defaults)",
    )
    ap.add_argument("--timeout", type=float, default=None, help="stop the job after this many seconds")
    ap.add_argument(
        "--stop-when-revoked",
        action="store_true",
        help="stop the job when the ingest API keeps rejecting heartbeats (stage cancelled)",
    )
    ap.add_argument("--flush-seconds", type=float, default=2.0)
    ap.add_argument("--metrics-seconds", type=float, default=5.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = ap.parse_args(argv[:split])
    cmd = argv[split + 1 :]
    if not cmd:
        raise SystemExit("skf_reporter.py: no command after --")
    return args, cmd


def main(argv: Sequence[str] | None = None) -> int:
    args, cmd = parse_args(sys.argv[1:] if argv is None else argv)
    for path in (args.log, args.exit_file):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    url, token, stage_id = (os.environ.get(k) for k in ("SKF_INGEST_URL", "SKF_INGEST_TOKEN", "SKF_STAGE_ID"))
    job: Job | None = None

    def revoked() -> None:
        if job is not None:
            job.stop(signal.SIGTERM, "stage is no longer active in the studio; stopped the job", 143)

    shipper = None
    if url and token and stage_id:
        shipper = Shipper(
            IngestClient(url, token, stage_id),
            args.progress,
            args.flush_seconds,
            args.metrics_seconds,
            args.heartbeat_seconds,
            on_revoked=revoked if args.stop_when_revoked else None,
        )
    job = Job(cmd, DEFAULT_NOISE if args.noise is None else args.noise, args.log, shipper)

    def forward(signum, _frame) -> None:
        assert job is not None
        job.stop(signum, f"received signal {signum}; stopped the job")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)

    if shipper:
        shipper.start()
    code = job.run(args.timeout)
    for note in job.notes:
        job.emit("[skf-reporter] " + note)
    summary = f"[skf-reporter] job exited with code {code}"
    if job.dropped:
        summary += f"; {job.dropped:,} noise lines dropped"
    job.emit(summary)
    if shipper:
        shipper.finish(SHUTDOWN_GRACE_SECONDS)
        if shipper.unshipped:
            sys.stderr.write(f"[skf-reporter] {shipper.unshipped} lines could not be shipped\n")
    if args.exit_file:
        write_exit_file(args.exit_file, code)
    return code


if __name__ == "__main__":
    sys.exit(main())

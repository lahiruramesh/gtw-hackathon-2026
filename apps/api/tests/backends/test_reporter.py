"""apps/reporter/skf_reporter.py against a local fake ingest API (http.server)."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from backend_fakes import REPO_ROOT

from skf_api.backends.kaggle import load_pipeline_bundle

REPORTER = REPO_ROOT / "apps" / "reporter" / "skf_reporter.py"


def load_reporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("skf_reporter_under_test", REPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeIngest:
    """Records POSTs; `statuses` lets a test script failures (popped per request, then 200)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.statuses: list[int] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
                status = fake.statuses.pop(0) if fake.statuses else 200
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/ingest/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def bodies(self, endpoint: str) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["path"].endswith("/" + endpoint)]

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def ingest() -> Iterator[FakeIngest]:
    fake = FakeIngest()
    yield fake
    fake.close()


def reporter_env(ingest: FakeIngest) -> dict[str, str]:
    return dict(
        os.environ,
        SKF_INGEST_URL=ingest.url,
        SKF_INGEST_TOKEN="tok-123",  # noqa: S106
        SKF_STAGE_ID="stage-9",
    )


CHILD = """
import sys, time
csv = open(sys.argv[1], "w")
csv.write("step,wall_s,eval/episode_reward,eval/label,eval/sps\\n"); csv.flush()
for i in range(3):
    print(f"line {i}", flush=True)
    print("Warp: iterations limit reached", flush=True)
    csv.write(f"{i * 1000},{i}.5,{-1 + i},good,nan\\n"); csv.flush()
print("to stderr", file=sys.stderr, flush=True)
csv.write("3000,3.5,2"); csv.flush()   # a row still being written
sys.exit(7)
"""


def test_noise_list_matches_kaggle_job() -> None:
    bundle = load_pipeline_bundle(REPO_ROOT / "scripts" / "kaggle_job.py")
    assert bundle.noise == load_reporter().DEFAULT_NOISE


def test_ships_logs_metrics_and_heartbeats(ingest: FakeIngest, tmp_path: Path) -> None:
    progress, log, exit_file = tmp_path / "progress.csv", tmp_path / "job.log", tmp_path / "exit_code"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "--log",
            str(log),
            "--exit-file",
            str(exit_file),
            "--progress",
            str(progress),
            "--heartbeat-seconds",
            "0.2",
            "--metrics-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            CHILD,
            str(progress),
        ],
        env=reporter_env(ingest),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 7
    assert exit_file.read_text().strip() == "7"
    assert "line 0" in proc.stdout and "iterations limit" not in proc.stdout
    assert log.read_text().splitlines()[:4] == ["line 0", "line 1", "line 2", "to stderr"]

    assert {r["auth"] for r in ingest.requests} == {"Bearer tok-123"}
    assert all("/ingest/v1/stages/stage-9/" in r["path"] for r in ingest.requests)
    logs = ingest.bodies("logs")
    texts = [line["text"] for body in logs for line in body["lines"]]
    assert texts[:4] == ["line 0", "line 1", "line 2", "to stderr"]
    assert "[skf-reporter] job exited with code 7; 3 noise lines dropped" in texts
    assert sum(body.get("noise_dropped", 0) for body in logs) == 3
    assert all(line["ts"].endswith("+00:00") for body in logs for line in body["lines"])

    points = [p for body in ingest.bodies("metrics") for p in body["points"]]
    assert points == [
        {"step": 0, "wall_s": 0.5, "values": {"eval/episode_reward": -1.0}},
        {"step": 1000, "wall_s": 1.5, "values": {"eval/episode_reward": 0.0}},
        {"step": 2000, "wall_s": 2.5, "values": {"eval/episode_reward": 1.0}},
    ]
    assert ingest.bodies("heartbeat"), "at least the final heartbeat"


def test_runs_without_ingest_and_passes_the_exit_code(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SKF_INGEST")}
    proc = subprocess.run(
        [sys.executable, str(REPORTER), "--", sys.executable, "-c", "print('hi')"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0] == "hi"


def test_missing_command_exits_127(tmp_path: Path) -> None:
    exit_file = tmp_path / "exit_code"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "--exit-file",
            str(exit_file),
            "--",
            str(tmp_path / "no-such-program"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 127
    assert exit_file.read_text().strip() == "127"


def test_sigterm_is_forwarded_to_the_job(tmp_path: Path) -> None:
    child = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: (print('child got TERM', flush=True), sys.exit(5)))\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    log = tmp_path / "job.log"
    proc = subprocess.Popen(
        [sys.executable, str(REPORTER), "--log", str(log), "--", sys.executable, "-c", child],
        stdout=subprocess.DEVNULL,
    )
    for _ in range(100):
        if log.exists() and "ready" in log.read_text():
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) == 5
    text = log.read_text()
    assert "child got TERM" in text
    assert "[skf-reporter] received signal 15; stopped the job" in text


def test_timeout_stops_the_job_with_124(tmp_path: Path) -> None:
    started = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "--timeout",
            "0.5",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 124
    assert time.monotonic() - started < 15
    assert "timeout after 0s" in proc.stdout or "timeout after 1s" in proc.stdout


def test_ingest_client_retries_server_errors_but_not_client_errors(ingest: FakeIngest) -> None:
    reporter = load_reporter()
    client = reporter.IngestClient(ingest.url, "tok", "s1", attempts=4, base_delay=0.01)
    ingest.statuses = [503, 502]
    assert client.post("heartbeat", {"message": "x"}) is True
    assert len(ingest.requests) == 3

    ingest.requests.clear()
    ingest.statuses = [422]
    assert client.post("logs", {"lines": []}) is False
    assert len(ingest.requests) == 1


def test_ingest_client_rejects_non_http_urls() -> None:
    with pytest.raises(ValueError):
        load_reporter().IngestClient("file:///etc/passwd", "tok", "s1")


def test_dead_ingest_never_holds_up_the_job(
    ingest: FakeIngest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporter = load_reporter()
    monkeypatch.setattr(reporter, "SHUTDOWN_GRACE_SECONDS", 1.0)
    monkeypatch.setattr(reporter.random, "uniform", lambda a, b: 0.05)
    ingest.statuses = [500] * 10_000
    monkeypatch.setenv("SKF_INGEST_URL", ingest.url)
    monkeypatch.setenv("SKF_INGEST_TOKEN", "tok")
    monkeypatch.setenv("SKF_STAGE_ID", "s1")
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    started = time.monotonic()
    try:
        code = reporter.main(["--", sys.executable, "-c", "print('x'); raise SystemExit(2)"])
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert code == 2
    assert time.monotonic() - started < 10


def test_revoked_stage_stops_the_job_when_asked(ingest: FakeIngest, tmp_path: Path) -> None:
    ingest.statuses = [401] * 1000
    started = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "--stop-when-revoked",
            "--heartbeat-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        env=reporter_env(ingest),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 143
    assert time.monotonic() - started < 20
    assert "stage is no longer active" in proc.stdout


def test_progress_tail_handles_partial_rows_and_rewrites(tmp_path: Path) -> None:
    reporter = load_reporter()
    path = tmp_path / "progress.csv"
    tail = reporter.ProgressTail(str(path))
    assert tail.read_points() == []
    path.write_text("step,wall_s,loss\n10,1.0,0.5\n20,2.0")
    assert [p["step"] for p in tail.read_points()] == [10]
    with path.open("a") as f:
        f.write(",0.25\n")
    assert tail.read_points() == [{"step": 20, "wall_s": 2.0, "values": {"loss": 0.25}}]
    path.write_text("step,wall_s,loss\n5,0.1,9\n")  # a new run rewrote the file
    assert [p["step"] for p in tail.read_points()] == [5]

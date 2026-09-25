"""One command for a training run on AWS: box -> preflight -> train -> rank on the GPU -> pull -> stop.

    uv run python scripts/g1job.py --run g1-stairs-v14 \\
        --init-from runs/g1-stairs-v13/run/params.pkl --budget-min 90 \\
        -- --task stairs --leg-action-scale 1.0 --lr 1e-4 --timesteps 300000000 --scan-model camera

What it does (everything that was done by hand for v9-v13):
  1. box       start a stopped g1-train* box, else launch one (g6e.2xlarge / g6e.xlarge over the zones,
               then g6.2xlarge), retrying for --wait-min; bootstrap the MJX pipeline if it is new
  2. budget    hard shutdown on the box after min(--budget-min, what is left of --daily-cap-h today);
               box minutes are kept in ~/.g1job_ledger.json
  3. preflight g1pipe.preflight on the box (engine parity, throughput, warm start); stop on failure
  4. train     g1pipe.train in tmux, Warp's per-step solver notes filtered out of the log
  5. watch     every 60 s: process alive, errors in the log, log size, disk, GPU power; checkpoints
               are copied to runs/<run>/run/ as they appear
  6. rank      g1pipe.gpu_eval on the box over every checkpoint (true scan and camera, 2048 episodes)
  7. stop      pull the rankings, stop the box, print the best checkpoints to certify
               (scripts/certify.py) on the Mac
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "scripts" / "aws_box.sh"
LEDGER = Path.home() / ".g1job_ledger.json"
AWS = {"AWS_PROFILE": os.environ.get("AWS_PROFILE", "sylonik-sso"), "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1")}
SUBNETS = {"us-east-1a": "subnet-0dc6bf369fb078ba3", "us-east-1b": "subnet-0b25ee4dea04d93ab",
           "us-east-1c": "subnet-0b9f32c0980b56c49", "us-east-1d": "subnet-01766fd0097ceef5d"}
LAUNCH_TYPES = ["g6e.2xlarge", "g6e.xlarge"]
FALLBACK_TYPES = ["g6.2xlarge"]            # L4: ~3x slower than the L40S, still faster than a Kaggle T4
NOISE = ["iterations limit", "To disable the print", "Warning", "warnings.warn"]


def sh(cmd, check=True, capture=True, env=None, input=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=capture, text=input is None or isinstance(input, str),
                       env={**os.environ, **AWS, **(env or {})}, cwd=ROOT, input=input)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr}")
    return r.stdout.strip() if capture else ""


def log(msg):
    print(f"[{dt.datetime.now():%H:%M}] {msg}", flush=True)


class Box:
    def __init__(self, name):
        self.name = name

    def run(self, *args, check=True, input=None):
        return sh(["bash", str(BOX), *args], check=check, env={"NAME": self.name}, input=input)

    def ssh(self, command, check=True):
        return self.run("ssh", "-o", "ConnectTimeout=15", command, check=check)

    def put(self, local, remote):
        with open(local, "rb") as f:
            subprocess.run(["bash", str(BOX), "ssh", f"mkdir -p $(dirname {remote}) && cat > {remote}"], stdin=f,
                           check=True, env={**os.environ, **AWS, "NAME": self.name}, cwd=ROOT)

    def get(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as f:
            subprocess.run(["bash", str(BOX), "ssh", f"cat {remote}"], stdout=f, check=True,
                           env={**os.environ, **AWS, "NAME": self.name}, cwd=ROOT)

    def reachable(self):
        return subprocess.run(["bash", str(BOX), "ssh", "-o", "ConnectTimeout=10", "true"], capture_output=True,
                              env={**os.environ, **AWS, "NAME": self.name}, cwd=ROOT).returncode == 0

    def state(self):
        return sh(f"aws ec2 describe-instances --filters Name=tag:Name,Values={self.name} "
                  "Name=instance-state-name,Values=pending,running,stopping,stopped "
                  "--query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text", check=False)


def boxes():
    out = sh("aws ec2 describe-instances --filters 'Name=tag:Name,Values=g1-train*' "
             "Name=instance-state-name,Values=running,stopped,pending,stopping "
             "--query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceId,State.Name]' --output text",
             check=False)
    return [line.split("\t") for line in out.splitlines() if line.strip()]


def acquire(wait_min):
    """A running g1-train* box: start a stopped one, else launch one; retry until wait_min."""
    end = time.time() + wait_min * 60
    tried_fallback = False
    while True:
        for name, iid, state in boxes():
            if state == "running":
                return Box(name), False
        busy = [(name, iid, state) for name, iid, state in boxes() if state in ("stopping", "pending")]
        if busy:
            # a box that is still stopping is set up already: wait for it rather than launching a new one
            name, iid, state = busy[0]
            log(f"{name} is {state}; waiting for it")
            sh(f"aws ec2 wait instance-{'stopped' if state == 'stopping' else 'running'} --instance-ids {iid}", check=False)
            continue
        for name, iid, state in boxes():
            if state == "stopped" and subprocess.run(f"aws ec2 start-instances --instance-ids {iid}", shell=True,
                                                     capture_output=True, env={**os.environ, **AWS}).returncode == 0:
                log(f"started {name}")
                sh(f"aws ec2 wait instance-running --instance-ids {iid}")
                return Box(name), False
        n = len(boxes()) + 1
        types = LAUNCH_TYPES + (FALLBACK_TYPES if time.time() > end - 300 else [])
        for t in types:
            for zone, subnet in SUBNETS.items():
                name = f"g1-train-{n}"
                r = subprocess.run(["bash", str(BOX), "launch"], capture_output=True, text=True, cwd=ROOT,
                                   env={**os.environ, **AWS, "NAME": name, "TYPE": t, "SUBNET": subnet})
                if r.returncode == 0:
                    log(f"launched {name} ({t}, {zone})")
                    return Box(name), True
        if time.time() > end:
            if tried_fallback:
                raise SystemExit("no GPU capacity; try later or use Kaggle (scripts/kaggle_job.py)")
            tried_fallback = True
        log("no capacity; retrying in 60 s")
        time.sleep(60)


def ledger_minutes_today():
    d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    return d.get(dt.date.today().isoformat(), 0.0)


def ledger_add(minutes):
    d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    k = dt.date.today().isoformat()
    d[k] = d.get(k, 0.0) + minutes
    LEDGER.write_text(json.dumps(d, indent=1))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--run", required=True, help="run name, e.g. g1-stairs-v14")
    ap.add_argument("--init-from", default=None, help="local params.pkl / ckpt_*.pkl to warm start from")
    ap.add_argument("--budget-min", type=float, default=120, help="hard stop for this job (box minutes)")
    ap.add_argument("--daily-cap-h", type=float, default=10, help="box hours per day, across jobs")
    ap.add_argument("--wait-min", type=float, default=30, help="how long to wait for GPU capacity")
    ap.add_argument("--min-gpu-w", type=float, default=60, help="alert when the GPU draws less once training runs")
    ap.add_argument("--keep", action="store_true", help="leave the box running afterwards")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="-- then g1pipe.train arguments")
    a = ap.parse_args()
    train_args = [x for x in a.train_args if x != "--"]

    left = a.daily_cap_h * 60 - ledger_minutes_today()
    cap = int(min(a.budget_min, left))
    if cap < 20:
        raise SystemExit(f"only {left:.0f} box minutes left today (--daily-cap-h {a.daily_cap_h})")
    box, new = acquire(a.wait_min)
    t0 = time.time()
    try:
        for _ in range(40):
            if box.reachable():
                break
            time.sleep(10)
        else:
            raise SystemExit(f"{box.name} not reachable over ssh")
        box.ssh(f"sudo shutdown -h +{cap} 'g1job budget'")
        log(f"{box.name}: hard stop in {cap} min ({left:.0f} min left today)")
        box.run("sync")
        if new or box.ssh("test -x gtw/.venv/bin/python && echo ok", check=False) != "ok":
            log("bootstrapping the MJX pipeline (~5 min)")
            box.ssh("MJX_ONLY=1 bash gtw/scripts/aws_bootstrap.sh > bootstrap.log 2>&1")

        init_remote = None
        if a.init_from:
            src = Path(a.init_from).resolve().relative_to(ROOT)
            for f in {src, src.with_name("params.pkl"), src.with_name("config.json")}:
                if f.exists():
                    box.put(f, f"gtw/{f}")
            init_remote = str(src)
            train_args = ["--init-from", init_remote, *train_args]

        # 3. preflight
        leg = train_args[train_args.index("--leg-action-scale") + 1] if "--leg-action-scale" in train_args else None
        pre_cmd = "cd gtw && PYTHONPATH=. .venv/bin/python -m g1pipe.preflight" + \
            (f" --init-from {init_remote}" if init_remote else "") + (f" --leg-action-scale {leg}" if leg else "")
        out = box.ssh(pre_cmd + " 2>&1 | grep -v -e Warning -e warnings.warn", check=False)
        print(out, flush=True)
        if "PREFLIGHT OK" not in out:
            raise SystemExit("preflight failed; not training")

        # 4. train, then rank every checkpoint on the GPU
        run = a.run
        grep = " ".join(f"-e {shlex.quote(n)}" for n in NOISE)
        rank = (f"PYTHONPATH=. .venv/bin/python -m g1pipe.gpu_eval runs/{run}/ckpt_*.pkl runs/{run}/params.pkl "
                f"--episodes 2048 --out runs/{run}/rank_true.json > ~/{run}.rank.log 2>&1; "
                f"PYTHONPATH=. .venv/bin/python -m g1pipe.gpu_eval runs/{run}/ckpt_*.pkl runs/{run}/params.pkl "
                f"--episodes 2048 --scan camera --out runs/{run}/rank_camera.json >> ~/{run}.rank.log 2>&1; "
                f"touch ~/{run}.done")
        job = (f"cd ~/gtw && PYTHONPATH=. .venv/bin/python -u -m g1pipe.train --out runs/{run} "
               f"{' '.join(map(shlex.quote, train_args))} 2>&1 | grep --line-buffered -v {grep} > ~/{run}.log; {rank}")
        box.ssh(f"rm -f ~/{run}.done; tmux new -d -s {run} {shlex.quote(job)}")
        log(f"training {run}: {' '.join(train_args)}")

        # 5. watch
        local = ROOT / "runs" / run / "run"
        local.mkdir(parents=True, exist_ok=True)
        seen, low_gpu = set(), 0
        while True:
            time.sleep(60)
            st = box.ssh(f"echo $(tmux has-session -t {run} 2>/dev/null && echo up || echo down) "
                         f"$(test -f ~/{run}.done && echo done || echo -) "
                         f"$(( $(stat -c %s ~/{run}.log 2>/dev/null || echo 0) / 1048576 )) "
                         f"$(df --output=pcent / | tail -1 | tr -dc 0-9) "
                         f"$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | cut -d. -f1) "
                         f"$(grep -c -i traceback ~/{run}.log); grep '^\\[' ~/{run}.log | tail -1; "
                         f"ls gtw/runs/{run}/ 2>/dev/null", check=False).splitlines()
            if not st:
                raise SystemExit("box unreachable (hard stop reached?)")
            tmux, done, log_mb, disk, gpu_w, tb = st[0].split()[:6]
            last = st[1] if len(st) > 1 and st[1].startswith("[") else ""
            files = [f for f in st[1:] if f.endswith((".pkl", ".json"))]
            for f in files:
                if f.startswith("ckpt_") and f not in seen:
                    box.get(f"gtw/runs/{run}/{f}", local / f)
                    seen.add(f)
            if "config.json" in files and not (local / "config.json").exists():
                box.get(f"gtw/runs/{run}/config.json", local / "config.json")
            log(f"{last or 'compiling'} | ckpts {len(seen)} gpu {gpu_w} W disk {disk} % log {log_mb} MB")
            if int(tb):
                raise SystemExit(f"traceback in the training log: see ~/{run}.log on {box.name}")
            if int(log_mb) > 200 or int(disk) > 85:
                raise SystemExit("log or disk growing: stopping")
            training = last and "params.pkl" not in files
            low_gpu = low_gpu + 1 if training and int(gpu_w) < a.min_gpu_w else 0
            if low_gpu >= 3:
                raise SystemExit(f"GPU under {a.min_gpu_w} W for 3 min while training: stopping")
            if done == "done":
                break
            if tmux == "down":
                raise SystemExit("training session ended without finishing")

        # 7. pull and report
        for f in ("params.pkl", "progress.csv", "rank_true.json", "rank_camera.json"):
            box.get(f"gtw/runs/{run}/{f}", local / f)
        for scan in ("true", "camera"):
            r = json.loads((local / f"rank_{scan}.json").read_text())
            log(f"GPU ranking, {scan} scan (2048 episodes each, best first):")
            for x in r[:5]:
                lo, hi = x["fall_ci95"]
                print(f"    {Path(x['params']).name:24} crossed {x['crossed']}/{x['episodes']} "
                      f"fell {x['fell']} ({x['fall_rate']*100:.1f} %, CI {lo*100:.1f}-{hi*100:.1f})", flush=True)
        best = json.loads((local / "rank_camera.json").read_text())[0]["params"]
        log(f"certify the best: uv run python scripts/certify.py {local / Path(best).name}")
    finally:
        minutes = (time.time() - t0) / 60
        ledger_add(minutes)
        if not a.keep:
            box.run("stop", check=False)
            log(f"{box.name} stopped after {minutes:.0f} min")


if __name__ == "__main__":
    main()

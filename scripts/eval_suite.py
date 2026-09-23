"""Evaluation suite for a trained step-length policy (runs on the Mac CPU).

  E2  step-length tracking over a (speed x step-length) grid
  E3  robustness/stress: friction, payload, pushes, action latency
  +   a scheduled run (step length changes mid-walk) for the demo video

    uv run scripts/eval_suite.py runs/<name>/run/params.pkl --tag v1
Outputs results/eval_<tag>/{grid.csv, stress.csv, *.png, demo.mp4, summary.json}
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from g1pipe.evaluate import Perturb, PolicyRunner


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path("results") / f"eval_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    r = PolicyRunner(a.params)

    # ---- E2: tracking grid -------------------------------------------------
    speeds = [0.3, 0.6, 0.9] if a.quick else [0.3, 0.5, 0.7, 0.9]
    steps = [0.15, 0.25, 0.35] if a.quick else [0.15, 0.20, 0.25, 0.30, 0.35]
    grid = []
    for vx in speeds:
        for sl in steps:
            for seed in range(a.seeds):
                s, *_ = r.run(vx=vx, step_len=sl, duration=10.0, seed=seed)
                grid.append({"vx": vx, "step_req": sl, "seed": seed, **s})
                print(f"E2 vx={vx:.1f} step={sl:.2f} -> {s['step_achieved_mean_m']:.3f} m "
                      f"(cmd {s['step_cmd_mean_m']:.3f}) fell={s['fell']}", flush=True)
    write_csv(out / "grid.csv", grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    ok = [g for g in grid if not g["fell"]]
    ax.scatter([g["step_cmd_mean_m"] * 100 for g in ok], [g["step_achieved_mean_m"] * 100 for g in ok],
               c=[g["vx"] for g in ok], cmap="viridis")
    lim = [5, 45]
    ax.plot(lim, lim, "k--", lw=1, label="perfect tracking")
    ax.set_xlabel("commanded step length (cm)")
    ax.set_ylabel("achieved step length (cm)")
    ax.set_title("E2: step-length tracking (MuJoCo C, unseen engine)")
    ax.legend()
    fig.colorbar(ax.collections[0], label="speed (m/s)")
    fig.tight_layout()
    fig.savefig(out / "e2_tracking.png", dpi=140)

    # ---- E3: stress tests ---------------------------------------------------
    cases = {
        "nominal": Perturb(),
        "low_friction_0.5": Perturb(friction=0.5),
        "low_friction_0.3": Perturb(friction=0.3),
        "payload_+3kg": Perturb(payload_kg=3.0),
        "payload_+6kg": Perturb(payload_kg=6.0),
        "push_0.5mps": Perturb(push_every_s=3.0, push_vel=0.5),
        "push_1.0mps": Perturb(push_every_s=3.0, push_vel=1.0),
        "latency_20ms": Perturb(action_delay_steps=1),
        "latency_40ms": Perturb(action_delay_steps=2),
    }
    stress = []
    for name, p in cases.items():
        res = [r.run(vx=0.6, step_len=0.25, duration=12.0, perturb=p, seed=s)[0] for s in range(max(3, a.seeds))]
        row = {"case": name,
               "fall_rate": float(np.mean([x["fell"] for x in res])),
               "step_abs_err_cm": float(np.nanmean([x["step_abs_err_m"] for x in res]) * 100),
               "vx_abs_err": float(np.nanmean([x["vx_abs_err"] for x in res]))}
        stress.append(row)
        print("E3", row, flush=True)
    write_csv(out / "stress.csv", stress)

    # ---- demo: change step length while walking -----------------------------
    def schedule(t):
        return (0.6, 0.18) if t < 5 else (0.6, 0.30) if t < 10 else (0.6, 0.22)
    ren = mujoco.Renderer(r.m, 540, 960)
    s, frames, step_log, _ = r.run(duration=15.0, schedule=schedule, renderer=ren)
    import imageio.v2 as imageio
    imageio.mimsave(out / "demo_step_change.mp4", frames, fps=30)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.step([x[0] for x in step_log], [x[3] * 100 for x in step_log], where="post", label="command")
    ax.plot([x[0] for x in step_log], [x[2] * 100 for x in step_log], "o", ms=4, label="achieved (per step)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("step length (cm)"); ax.legend()
    ax.set_title("Step length changed on the fly at 0.6 m/s")
    fig.tight_layout(); fig.savefig(out / "demo_step_change.png", dpi=140)

    ok = [g for g in grid if not g["fell"]]
    summary = {
        "grid_fall_rate": float(np.mean([g["fell"] for g in grid])),
        "grid_step_abs_err_cm": float(np.nanmean([g["step_abs_err_m"] for g in ok]) * 100) if ok else None,
        "grid_vx_abs_err": float(np.nanmean([g["vx_abs_err"] for g in ok])) if ok else None,
        "stress": stress,
        "demo": s,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

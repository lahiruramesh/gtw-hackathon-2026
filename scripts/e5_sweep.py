"""E5 — Do we need to retrain for step length?

Sweeps the pre-trained policy over forward-speed commands and gait periods (the
policy was trained with a fixed 0.8 s period) and measures achieved step length,
speed and falls across several seeds with light pushes / observation noise.

    uv run scripts/e5_sweep.py            # writes results/e5/*.csv + *.png
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g1pipe.sim import G1Sim, Perturb, summarize

OUT = Path("results/e5")
_sim = None


def job(args):
    global _sim
    vx, period, seed = args
    if _sim is None:
        import torch
        torch.set_num_threads(1)
        _sim = G1Sim()
    p = Perturb(obs_noise=0.01, push_every_s=4.0, push_vel=0.3)
    ep, _ = _sim.run(cmd=(vx, 0, 0), period=period, duration=12.0, perturb=p, seed=seed)
    r = summarize(ep, vx, period)
    r["seed"] = seed
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    vxs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    periods = [0.6, 0.7, 0.8, 0.9, 1.0]
    grid = list(product(vxs, periods, range(a.seeds)))
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(job, grid))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "e5_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # aggregate per (vx, period)
    agg = {}
    for r in rows:
        agg.setdefault((r["cmd_vx"], r["period"]), []).append(r)
    step = np.full((len(periods), len(vxs)), np.nan)
    fall = np.zeros_like(step)
    for (vx, per), rs in agg.items():
        i, j = periods.index(per), vxs.index(vx)
        fall[i, j] = np.mean([r["fell"] for r in rs])
        ok = [r["step_mean_m"] for r in rs if not r["fell"]]
        step[i, j] = np.nanmean(ok) if ok else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    im = axes[0].imshow(step * 100, origin="lower", cmap="viridis", aspect="auto")
    axes[1].imshow(fall * 100, origin="lower", cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for ax, data, title, fmt in ((axes[0], step * 100, "Achieved step length (cm)", "{:.0f}"),
                                 (axes[1], fall * 100, "Fall rate (%)", "{:.0f}")):
        ax.set_xticks(range(len(vxs)), [f"{v:.1f}" for v in vxs])
        ax.set_yticks(range(len(periods)), [f"{p:.1f}" for p in periods])
        ax.set_xlabel("commanded forward speed (m/s)")
        ax.set_ylabel("gait period (s)  [trained: 0.8]")
        ax.set_title(title)
        for i in range(len(periods)):
            for j in range(len(vxs)):
                v = data[i, j]
                ax.text(j, i, "–" if np.isnan(v) else fmt.format(v), ha="center", va="center",
                        color="white" if ax is axes[0] else "black", fontsize=9)
    fig.colorbar(im, ax=axes[0])
    fig.suptitle("E5: step-length range of the pre-trained G1 policy without retraining")
    fig.tight_layout()
    fig.savefig(OUT / "e5_heatmap.png", dpi=140)

    print(f"{'period':>6} " + " ".join(f"{v:>7.1f}" for v in vxs))
    for i, per in enumerate(periods):
        print(f"{per:>6.1f} " + " ".join(
            f"{step[i, j]*100:6.1f}{'*' if fall[i, j] > 0 else ' '}" for j in range(len(vxs))))
    print("(cm; * = at least one fall)  ->", OUT)


if __name__ == "__main__":
    main()

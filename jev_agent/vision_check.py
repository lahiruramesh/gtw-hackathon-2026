"""Walk the G1 toward a staircase with the simulated head depth camera, and score what it sees.

    python -m jev_agent.vision_check --rise 0.08 --out results/vision

Every 0.25 s: capture depth + RGB, fuse the points into the elevation map, and compare the
reconstructed ground height with the simulator's true terrain, (a) along the walking line and
(b) on the 11 x 5 scan patch the stairs policy observes. Writes a figure and metrics JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1pipe.sim import yaw_of
from g1pipe.stairs_terrain import SCAN_PTS, SCAN_X, SCAN_Y
from jev_agent import stairs as S
from jev_agent.vision import ElevationMap, HeadCamera, compare, scan_points

LINE = np.linspace(0.0, 3.0, 61)          # m ahead of the pelvis, along the heading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rise", type=float, default=0.08)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=7.5)
    ap.add_argument("--out", default="results/vision")
    a = ap.parse_args()

    st = S.Staircase(rise=a.rise)
    sim = S.build(st, camera=True)
    cam = HeadCamera(sim.m)
    emap = ElevationMap()
    frames, metrics = [], []

    def cmd_fn(t):
        k = int(round(t / 0.02))
        if k % 12 == 0 and t > 0.3:                       # ~4 Hz
            shot = cam.capture(sim.d, rgb=True)
            emap.integrate(shot["points"], t)
            base, yaw = sim.d.qpos[:3].copy(), yaw_of(sim.d.qpos[3:7])
            line_xy = scan_points(base[:2], yaw, np.stack([LINE, np.zeros_like(LINE)], -1))
            patch_xy = scan_points(base[:2], yaw, SCAN_PTS)
            m = {"t": round(t, 2), "x": round(float(base[0]), 2),
                 "dist_to_stairs_m": round(st.start_x - float(base[0]), 2),
                 "points": int(len(shot["points"])), "self_pixels": shot["self_pixels"],
                 "line": compare(emap.height(line_xy), st.ground(line_xy)),
                 "patch": compare(emap.height(patch_xy), st.ground(patch_xy))}
            metrics.append(m)
            frames.append({**m, "rgb": shot["rgb"], "depth": shot["depth"],
                           "line_true": st.ground(line_xy), "line_est": emap.height(line_xy),
                           "patch_true": st.ground(patch_xy), "patch_est": emap.height(patch_xy)})
        return (a.vx if t > 0.5 else 0.0, 0.0, float(np.clip(-1.5 * yaw_of(sim.d.qpos[3:7]), -0.5, 0.5)))

    sim.run(duration=a.duration, cmd_fn=cmd_fn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"camera_check_{int(a.rise * 100)}cm.json").write_text(json.dumps(metrics, indent=1))

    for m in metrics[::4]:
        print(f"t={m['t']:4.1f}s  {m['dist_to_stairs_m']:4.2f} m to stairs | line: seen {m['line']['coverage']:.0%}"
              f" err {m['line']['mae_cm'] or 0:.1f} cm (p95 {m['line']['p95_cm'] or 0:.1f}) | policy patch: seen"
              f" {m['patch']['coverage']:.0%} err {m['patch']['mae_cm'] or 0:.1f} cm | robot pixels {m['self_pixels']}")
    fig_path = out / f"camera_check_{int(a.rise * 100)}cm.png"
    figure(frames, st, fig_path)
    print("figure:", fig_path)


def figure(frames, st, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [frames[int(i)] for i in np.linspace(len(frames) * 0.15, len(frames) - 1, 3)]
    fig, ax = plt.subplots(3, 4, figsize=(17, 11), gridspec_kw={"width_ratios": [1.1, 1.1, 1.6, 1.0]})
    for r, f in enumerate(picks):
        ax[r, 0].imshow(f["rgb"]); ax[r, 0].set_title(f"head RGB  ({f['dist_to_stairs_m']:.1f} m to first step)")
        im = ax[r, 1].imshow(f["depth"], cmap="turbo", vmin=0.3, vmax=3.0)
        ax[r, 1].set_title("depth (m); black = no return / robot")
        fig.colorbar(im, ax=ax[r, 1], fraction=0.035)
        a2 = ax[r, 2]
        a2.plot(LINE, f["line_true"] * 100, color="#555", lw=2.5, label="true ground (simulator)")
        seen = ~np.isnan(f["line_est"])
        a2.plot(LINE[seen], f["line_est"][seen] * 100, "o", ms=4, color="#e4572e", label="camera elevation map")
        a2.set_xlabel("distance ahead of robot (m)"); a2.set_ylabel("height (cm)")
        a2.set_title(f"walking line: seen {f['line']['coverage']:.0%}, mean err "
                     f"{f['line']['mae_cm'] or 0:.1f} cm")
        a2.set_ylim(-3, st.n_up * st.rise * 100 + 5); a2.grid(alpha=.3); a2.legend(loc="upper left", fontsize=8)
        a3 = ax[r, 3]
        shape = (len(SCAN_X), len(SCAN_Y))
        err = np.abs(f["patch_est"] - f["patch_true"]).reshape(shape) * 100
        im = a3.imshow(err.T, origin="lower", cmap="magma_r", vmin=0, vmax=5, aspect="auto",
                       extent=[SCAN_X[0] - .075, SCAN_X[-1] + .075, SCAN_Y[0] - .075, SCAN_Y[-1] + .075])
        unseen = np.isnan(err)
        for (i, j) in zip(*np.nonzero(unseen)):
            a3.text(SCAN_X[i], SCAN_Y[j], "?", ha="center", va="center", color="#3a5bd9", fontsize=9, weight="bold")
        a3.set_title(f"policy scan patch error (cm)\nseen {f['patch']['coverage']:.0%}, ? = not seen yet")
        a3.set_xlabel("ahead (m)"); a3.set_ylabel("left (m)")
        fig.colorbar(im, ax=a3, fraction=0.05)
        for c in ax[r, :2]:
            c.axis("off")
    fig.suptitle(f"Simulated head depth camera (D435i-like) vs. simulator ground truth: {st.rise * 100:.0f} cm stairs",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


if __name__ == "__main__":
    main()

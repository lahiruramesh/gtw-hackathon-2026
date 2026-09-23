"""Evaluate a stairs policy (g1pipe.stairs_env) in plain MuJoCo, one staircase per step height.

For every pyramid in the terrain the robot starts on the flat strip at its west edge, facing
east, and is commanded to walk straight over it: up the steps, across the platform, down the
other side. Success = reached the top and got off the far side without falling.

    python -m g1pipe.stairs_eval runs/g1-steplength-stairs/run/params.pkl --video results/videos/stairs.mp4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from g1pipe import stairs_terrain as T
from g1pipe.evaluate import PolicyRunner


class StairsPolicyRunner(PolicyRunner):
    def __init__(self, params_path, terrain_seed: int = 0):
        from g1pipe.stairs_env import StairsStepLength, default_config
        cfg = default_config()
        cfg.impl = "jax"
        super().__init__(params_path, env=StairsStepLength(config=cfg, terrain_seed=terrain_seed))
        self.grid = T.heights(terrain_seed)
        self.rises = T.cell_rises(terrain_seed)

    def _obs(self, *args):
        obs = super()._obs(*args)
        q = self.d.qpos
        yaw = np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        scan = T.scan(self.grid, q[:3], yaw).astype(np.float32)
        return {"state": np.hstack([np.asarray(obs["state"]), scan]),
                "privileged_state": np.zeros(self._priv.shape[0], np.float32)}

    def place(self, x: float, y: float, yaw: float = 0.0):
        ring = np.array([[x + dx, y + dy] for dx in (-0.2, 0, 0.2) for dy in (-0.2, 0, 0.2)])
        self.init_q[:3] = (x, y, T.lookup(self.grid, ring).max() + 0.755 + 0.02)
        self.init_q[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


def cross_cell(r: StairsPolicyRunner, ix: int, iy: int, vx=0.5, step=0.25, renderer=None, duration=16.0):
    x0 = -T.HALF + ix * T.CELL
    y = -T.HALF + (iy + 0.5) * T.CELL
    r.place(x0 + 0.15, y)
    track = []

    def schedule(t):
        track.append(r.d.qpos[:3].copy())
        return (vx, step)

    summary, frames, _, _ = r.run(vx=vx, step_len=step, duration=duration, renderer=renderer, schedule=schedule)
    p = np.array(track)
    ground = T.lookup(r.grid, p[:, :2])
    top = r.rises[iy, ix] * int((T.CELL / 2 - T.BORDER - T.PLATFORM / 2) // T.TREAD)
    return {
        "rise_m": float(r.rises[iy, ix]), "cell": [ix, iy], "fell": summary["fell"],
        "reached_top": bool(ground.max() >= top - 0.01),
        "crossed": bool(not summary["fell"] and p[:, 0].max() > x0 + T.CELL - T.BORDER),
        "max_ground_m": round(float(ground.max()), 3),
    }, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--video", default=None, help="record one crossing per step height")
    ap.add_argument("--out", default=None, help="write per-cell results as JSON")
    a = ap.parse_args()

    r = StairsPolicyRunner(a.params)
    renderer = None
    if a.video:
        import mujoco
        renderer = mujoco.Renderer(r.m, 480, 640)
    results, frames_all, filmed = [], [], set()
    for iy in range(T.GRID):
        for ix in range(T.GRID):
            rise = float(r.rises[iy, ix])
            film = renderer if (renderer and rise > 0 and rise not in filmed) else None
            res, frames = cross_cell(r, ix, iy, a.vx, a.step, renderer=film)
            if film:
                filmed.add(rise)
                frames_all += frames
            results.append(res)
            print(json.dumps(res), flush=True)

    by = defaultdict(list)
    for x in results:
        by[x["rise_m"]].append(x)
    print("\nstep height | tries | reached top | crossed | fell")
    for rise in sorted(by):
        xs = by[rise]
        print(f"   {rise * 100:4.0f} cm  |  {len(xs):3d}  |    {sum(x['reached_top'] for x in xs):3d}      |   {sum(x['crossed'] for x in xs):3d}   | {sum(x['fell'] for x in xs):3d}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))
    if a.video and frames_all:
        import imageio.v2 as imageio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(a.video, frames_all, fps=30)
        print("video:", a.video)


if __name__ == "__main__":
    main()

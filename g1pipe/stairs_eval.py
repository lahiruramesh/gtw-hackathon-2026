"""Evaluate a stairs policy (g1pipe.stairs_env) in plain MuJoCo on every staircase of a layout.

By default this is the held-out "test" layout: step heights between the training levels and a
different tread depth, so the policy never saw these stairs. For every staircase the robot starts
on the flat strip at its west edge, facing east, and is commanded to walk straight over it. A
pyramid is up the steps, across the platform, down the other side; a pit is down, across, up.
Success = reached the centre (top of a pyramid, bottom of a pit) and got off the far side
without falling.

    python -m g1pipe.stairs_eval runs/stairs_v2/run/params.pkl --out results/stairs/eval_v2.json
    python -m g1pipe.stairs_eval runs/stairs_v2/run/params.pkl --layout train --video results/videos/stairs.mp4
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
    """scan="true": the height scan is read from the known terrain (as in training).
    scan="camera": it comes from the simulated head depth camera (jev_agent.vision, D435i-like:
    noise, dropouts, self-filtering, limited field of view) fused into an elevation map, as it
    would on the robot. Cells the camera has not seen are assumed level with the feet."""

    CAMERA_EVERY = 2   # control steps per depth frame (25 Hz)

    def __init__(self, params_path, layout: str = "test", scan: str = "true"):
        from g1pipe.stairs_env import HEADING_WZ_MAX, StairsStepLength, default_config
        self.heading_wz_max = HEADING_WZ_MAX   # deploy the heading loop the policy was trained with (v4+)
        cfg = default_config()
        cfg.impl = "jax"
        run_cfg = Path(params_path).with_name("config.json")   # env settings the policy was trained with
        if run_cfg.exists():
            cfg.leg_action_scale = json.loads(run_cfg.read_text())["env"].get("leg_action_scale", 0.0)
        super().__init__(params_path, env=StairsStepLength(config=cfg, layout=layout))
        self.layout = layout
        self.grid = T.heights(layout)
        self.rises = T.cell_rises(layout)
        self.kinds = T.cell_kinds(layout)
        self.scan_mode = scan
        self.scan_log = []            # (true scan, used scan, unseen mask) per control step, for sensor-model fitting
        if scan == "camera":
            import mujoco
            from jev_agent.vision import CameraSpec, HeadCamera, add_head_camera
            assets = self.env._model_assets
            spec = mujoco.MjSpec.from_string(
                Path(self.env._xml_path).read_text(),
                include={k: v for k, v in assets.items() if k.endswith(".xml")}, assets=assets)
            add_head_camera(spec, CameraSpec())
            m = spec.compile()
            m.opt.timestep = self.m.opt.timestep
            m.jnt_range[:], m.actuator_ctrlrange[:] = self.m.jnt_range, self.m.actuator_ctrlrange
            self.m, self.d = m, mujoco.MjData(m)
            self._nominal = (self.m.pair_friction.copy(), self.m.body_mass.copy())
            self.cam = HeadCamera(self.m)

    def reset_scan(self):
        if self.scan_mode == "camera":
            from jev_agent.vision import ElevationMap
            self.emap, self._k = ElevationMap(half=T.HALF, res=T.RES), 0

    def _obs(self, *args):
        obs = super()._obs(*args)
        q = self.d.qpos
        yaw = np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        scan = true = T.scan(self.grid, q[:3], yaw).astype(np.float32)
        unseen = np.zeros(T.N_SCAN, bool)
        if self.scan_mode == "camera":
            from jev_agent.vision import scan_points
            if self._k % self.CAMERA_EVERY == 0:
                self.emap.integrate(self.cam.capture(self.d, rgb=False)["points"], self._k * self.ctrl_dt)
            self._k += 1
            h = self.emap.height(scan_points(q[:2], yaw, T.SCAN_PTS))
            feet_z = self.d.site_xpos[self.feet_sites][:, 2].min() - 0.037   # sole: foot box bottom is 3.7 cm below the foot site
            unseen = np.isnan(h)
            h = np.where(unseen, feet_z, h)
            scan = np.clip(h - (q[2] - T.NOMINAL_HEIGHT), -1.0, 1.0).astype(np.float32)
        self.scan_log.append((true, scan, unseen))
        return {"state": np.hstack([np.asarray(obs["state"]), scan]),
                "privileged_state": np.zeros(self._priv.shape[0], np.float32)}

    def _ground_z(self, xy) -> float:
        return float(T.lookup(self.grid, np.asarray(xy)))

    def place(self, x: float, y: float, yaw: float = 0.0):
        ring = np.array([[x + dx, y + dy] for dx in (-0.2, 0, 0.2) for dy in (-0.2, 0, 0.2)])
        self.init_q[:3] = (x, y, T.lookup(self.grid, ring).max() + 0.755 + 0.02)
        self.init_q[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


# Stability gate for one step height: every staircase crossed, no falls, torso never tilted past
# GATE_TILT_DEG and the pelvis never sank below GATE_PELVIS_M above the ground under it.
GATE_TILT_DEG = 25.0
GATE_PELVIS_M = 0.55


def stable(runs) -> bool:
    return all(x["crossed"] and not x["fell"] and x["tilt_deg"][1] < GATE_TILT_DEG
               and x["pelvis_rel_m"][0] > GATE_PELVIS_M for x in runs)


def certified_height(results) -> float:
    """Tallest step height (m) such that it and every lower height pass the stability gate; 0 if none."""
    by = defaultdict(list)
    for x in results:
        by[x["rise_m"]].append(x)
    cert = 0.0
    for rise in sorted(by):
        if not stable(by[rise]):
            break
        cert = rise
    return cert


def cross_cell(r: StairsPolicyRunner, ix: int, iy: int, vx=0.5, step=0.25, renderer=None, duration=16.0,
               perturb=None):
    x0 = -T.GRID_HALF + ix * T.CELL
    y = -T.GRID_HALF + (iy + 0.5) * T.CELL
    r.place(x0 + 0.15, y)
    r.reset_scan()
    track, tilt, yaws, torque, near_limit = [], [], [], [], []
    m = r.m
    jnt = m.actuator_trnid[:, 0]
    tau_max = np.abs(m.jnt_actfrcrange[jnt]).max(1)            # motor torque limits (G1 spec)
    lo, hi = m.jnt_range[jnt].T
    qadr = m.jnt_qposadr[jnt]

    def schedule(t):
        # the PD request is clipped at the joint's torque limit: count ticks where any motor saturates
        torque.append(bool(np.any(np.abs(r.d.actuator_force) > tau_max)))
        q = r.d.qpos[qadr]
        near_limit.append(bool(np.any((q < lo + 0.02) | (q > hi - 0.02))))
        track.append(r.d.qpos[:3].copy())
        q = r.d.qpos
        yaws.append(np.degrees(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))))
        tilt.append(np.degrees(np.arccos(np.clip(r.d.xmat[r.torso].reshape(3, 3)[2, 2], -1, 1))))
        return (vx, step)

    summary, frames, _, _ = r.run(vx=vx, step_len=step, duration=duration, renderer=renderer, schedule=schedule,
                                  perturb=perturb)
    p = np.array(track)
    ground = T.lookup(r.grid, p[:, :2])
    kind = int(r.kinds[iy, ix])
    centre = kind * r.rises[iy, ix] * T.n_steps(r.layout)[iy, ix]
    reached = ground.max() >= centre - 0.01 if kind > 0 else ground.min() <= centre + 0.01
    return {
        "rise_m": round(float(r.rises[iy, ix]), 4), "kind": "pyramid" if kind > 0 else "pit",
        "cell": [ix, iy], "fell": summary["fell"], "reached_centre": bool(reached),
        "crossed": bool(not summary["fell"] and p[:, 0].max() > x0 + T.CELL - T.BORDER),
        "ground_range_m": [round(float(ground.min()), 3), round(float(ground.max()), 3)],
        # stability: pelvis height above the ground under it (standing ~0.75 m; low = crouching),
        # torso tilt, and how far east the robot got across the cell
        "pelvis_rel_m": [round(float(np.min(p[:, 2] - ground)), 3), round(float(np.median(p[:, 2] - ground)), 3)],
        "tilt_deg": [round(float(np.median(tilt)), 1), round(float(np.max(tilt)), 1)],
        "progress_m": round(float(p[:, 0].max() - x0), 2),
        "heading_dev_deg": round(float(np.max(np.abs(yaws))), 1),   # started facing east (0 deg)
        # robot safety: share of time with a motor at its torque limit, and with a joint at its range limit
        "torque_sat_frac": round(float(np.mean(torque)), 3),
        "joint_limit_frac": round(float(np.mean(near_limit)), 3),
    }, frames


START_VARIANTS = [(0.0, 0.0), (-0.35, 0.12), (0.35, -0.12)]   # (lateral offset m, heading rad)
PERTURBS = {   # disturbance presets for certification (g1pipe.evaluate.Perturb)
    "none": {},
    "push": {"push_every_s": 3.0, "push_vel": 0.3},       # 0.3 m/s shove in a random direction every 3 s
    "lowfric": {"friction": 0.5},                         # half the nominal foot-floor friction
    "payload": {"payload_kg": 5.0},                       # 5 kg on the torso
    "delay": {"action_delay_steps": 1},                   # 20 ms actuation delay
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", help="params.pkl, or a ckpt_*.pkl next to one")
    ap.add_argument("--vx", type=float, default=0.7)
    ap.add_argument("--step", type=float, default=None,
                    help="step length; default vx / (2 * STAIRS_CADENCE), the 1.4 Hz stairs cadence")
    ap.add_argument("--layout", default="test", choices=["test", "train", "train30"],
                    help="test: held-out step heights and tread (default); train: the training curriculum")
    ap.add_argument("--cols", type=int, default=4, help="staircases per step height (alternating pyramid, pit)")
    ap.add_argument("--video", default=None, help="record one crossing per step height")
    ap.add_argument("--out", default=None, help="write per-cell results as JSON")
    ap.add_argument("--starts", type=int, default=1, choices=[1, 3],
                    help="3: also start every crossing 35 cm to each side and turned 7 deg (3x the runs)")
    ap.add_argument("--perturb", default="none", choices=list(PERTURBS),
                    help="disturbance preset: pushes, low friction, payload, actuation delay")
    ap.add_argument("--scan", default="true", choices=["true", "camera"],
                    help="height scan from the known terrain, or from the simulated head depth camera")
    a = ap.parse_args()

    from g1pipe.stairs_env import STAIRS_CADENCE
    if a.step is None:
        a.step = a.vx / (2 * STAIRS_CADENCE)
    r = StairsPolicyRunner(a.params, layout=a.layout, scan=a.scan)
    renderer = None
    if a.video:
        import mujoco
        renderer = mujoco.Renderer(r.m, 480, 640)
    results, frames_all, filmed = [], [], set()
    place = r.place
    from g1pipe.evaluate import Perturb
    perturb = Perturb(**PERTURBS[a.perturb])
    cells = [(ix, iy) for iy in range(T.GRID) for ix in range(min(a.cols, T.GRID))]
    for dy, yaw in START_VARIANTS[:a.starts]:
        r.place = lambda x, y, yaw0=0.0, dy=dy, yaw=yaw: place(x, y + dy, yaw)
        for ix, iy in cells:
            rise = float(r.rises[iy, ix])
            film = renderer if (renderer and rise not in filmed) else None
            res, frames = cross_cell(r, ix, iy, a.vx, a.step, renderer=film, perturb=perturb)
            if film:
                filmed.add(rise)
                frames_all += frames
            results.append({**res, "start": [dy, yaw]})
            print(json.dumps(results[-1]), flush=True)

    by = defaultdict(list)
    for x in results:
        by[(x["rise_m"], x["kind"])].append(x)
    print(f"\n{a.layout} layout\nstep height | stairs  | tries | reached centre | crossed | fell | pelvis med | tilt max")
    for rise, kind in sorted(by):
        xs = by[(rise, kind)]
        print(f"  {rise * 100:5.1f} cm  | {kind:7s} |  {len(xs):3d}  |      {sum(x['reached_centre'] for x in xs):3d}       |   {sum(x['crossed'] for x in xs):3d}   | {sum(x['fell'] for x in xs):3d}  |"
              f"   {np.mean([x['pelvis_rel_m'][1] for x in xs]):.2f} m   | {max(x['tilt_deg'][1] for x in xs):5.1f}°")
    n = len(results)
    print(f"total: crossed {sum(x['crossed'] for x in results)}/{n}, fell {sum(x['fell'] for x in results)}/{n}")
    print(f"certified step height (all crossed, no falls, tilt < {GATE_TILT_DEG:.0f} deg, pelvis > {GATE_PELVIS_M} m): "
          f"{certified_height(results) * 100:.1f} cm")
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

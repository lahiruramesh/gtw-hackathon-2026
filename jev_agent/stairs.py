"""Stairs world for the G1, plus a height-scan "sensor" that detects them.

The scene is Unitree's G1 scene with a staircase of box geoms added in front of the
robot: flat approach -> N steps up -> landing -> N steps down -> flat.

Detection runs in code (terrain geometry is arithmetic, not a judgment): a row of
downward rays ahead of the robot, like a depth camera or LiDAR heightmap on the real
G1, gives the ground height profile; we find step edges in it and describe what is
ahead in words for Jev ("stairs going up about 1 m ahead, 4 steps of about 8 cm").
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from g1pipe.sim import G1Sim, yaw_of

SCAN_AHEAD = np.linspace(0.15, 3.0, 58)   # m ahead of the pelvis along the heading
SCAN_TOP = 2.0                              # rays start this far above the pelvis
EDGE_DZ = 0.025                             # height jump that counts as a step edge
LOW_STEP_MAX = 0.045                        # m: below this, "low steps"
FLAT_TOL = 0.012                            # m: neighbouring samples this close belong to one tread


@dataclass
class Staircase:
    start_x: float = 2.5
    n_up: int = 4
    n_down: int = 4
    rise: float = 0.06
    tread: float = 0.35
    landing: float = 1.2
    width: float = 2.0

    def boxes(self):
        """(center_x, length, height) of each solid block, from the floor up."""
        out, x = [], self.start_x
        for i in range(self.n_up):
            out.append((x + self.tread / 2, self.tread, (i + 1) * self.rise))
            x += self.tread
        top = self.n_up * self.rise
        out.append((x + self.landing / 2, self.landing, top))
        x += self.landing
        for i in range(self.n_down):
            out.append((x + self.tread / 2, self.tread, top - (i + 1) * self.rise))
            x += self.tread
        return [b for b in out if b[2] > 1e-4]

    def height_at(self, x: float) -> float:
        for cx, length, h in self.boxes():
            if abs(x - cx) <= length / 2:
                return h
        return 0.0

    def ground(self, xy) -> np.ndarray:
        """True ground height at points xy (..., 2): the simulator's answer key."""
        xy = np.asarray(xy, float)
        h = np.zeros(xy.shape[:-1])
        inside = np.abs(xy[..., 1]) <= self.width / 2
        for cx, length, top in self.boxes():
            h = np.where(inside & (np.abs(xy[..., 0] - cx) <= length / 2), np.maximum(h, top), h)
        return h

    @property
    def end_x(self) -> float:
        return self.start_x + (self.n_up + self.n_down) * self.tread + self.landing


def build(stairs: Staircase, policy_path: str | None = None, camera: bool = False) -> G1Sim:
    """A G1Sim whose model has the staircase added. Everything else (policy, PD, obs) is unchanged.
    camera=True also mounts the simulated head depth camera (jev_agent.vision)."""
    sim = G1Sim(policy_path=policy_path)
    spec = mujoco.MjSpec.from_file(sim.xml_path)
    if camera:
        from jev_agent.vision import CameraSpec, add_head_camera
        add_head_camera(spec, CameraSpec())
    mat = spec.add_material(name="stair", rgba=[0.78, 0.74, 0.66, 1])
    edge = spec.add_material(name="stair_edge", rgba=[0.95, 0.75, 0.2, 1])
    for i, (cx, length, h) in enumerate(stairs.boxes()):
        spec.worldbody.add_geom(name=f"stair_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[length / 2, stairs.width / 2, h / 2], pos=[cx, 0, h / 2],
                                material="stair", friction=[1.0, 0.005, 0.0001])
        # thin visual stripe on the nose of each step, so edges read clearly in the video
        spec.worldbody.add_geom(name=f"stair_nose_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.015, stairs.width / 2, 0.002], pos=[cx - length / 2 + 0.015, 0, h + 0.001],
                                material="stair_edge", contype=0, conaffinity=0)
    m = spec.compile()
    m.opt.timestep = sim.dt
    sim.m, sim.d = m, mujoco.MjData(m)
    sim.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    sim.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    sim.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    sim._nominal = (m.geom_friction.copy(), m.body_mass.copy())
    sim.stairs = stairs
    return sim


# -- the sensor ------------------------------------------------------------------
def height_scan(m, d, pelvis_body: int) -> tuple[np.ndarray, float]:
    """Ground height at SCAN_AHEAD points along the heading, relative to the ground under the robot."""
    base = d.xpos[pelvis_body]
    yaw = yaw_of(d.xquat[pelvis_body])
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    geomid = np.zeros(1, np.int32)

    def ground(p):
        dist = mujoco.mj_ray(m, d, p, down, None, 1, pelvis_body, geomid)   # static geoms only
        return p[2] - dist if dist >= 0 else 0.0

    top = base[2] + SCAN_TOP
    under = ground(np.array([base[0], base[1], top]))
    h = np.array([ground(np.array([*(base[:2] + a * fwd[:2]), top])) for a in SCAN_AHEAD])
    return h - under, under


def analyse(scan: np.ndarray) -> dict:
    """Find step edges in a height profile; pure geometry.
    The profile is split into flat treads (>= 3 samples level within FLAT_TOL); every change of
    level between neighbouring treads is a step edge, and its height is the difference of the
    tread levels. Samples between treads (a sensor's blurred edge, riser points) are ignored, so
    a camera map measures the same step heights as a perfect raycast."""
    flat = np.abs(np.diff(scan)) < FLAT_TOL
    treads, i = [], 0
    while i < len(scan):
        j = i
        while j < len(flat) and flat[j]:
            j += 1
        if j - i + 1 >= 3:
            treads.append((i, j, float(np.median(scan[i:j + 1]))))
        i = j + 1
    edges = []
    for (_, e0, h0), (s1, _, h1) in zip(treads, treads[1:]):
        if abs(h1 - h0) > EDGE_DZ:
            edges.append((float(SCAN_AHEAD[min(e0 + 1, len(SCAN_AHEAD) - 1)]), h1 - h0))
    ups, downs = [e for e in edges if e[1] > 0], [e for e in edges if e[1] < 0]
    first = edges[0] if edges else None
    return {
        "n_edges": len(edges),
        "n_up": len(ups), "n_down": len(downs),
        "first_edge_m": float(first[0]) if first else None,
        "first_dir": (None if not first else "up" if first[1] > 0 else "down"),
        "mean_rise_m": float(np.mean([abs(e[1]) for e in edges])) if edges else 0.0,
        "max_rise_m": float(max([abs(e[1]) for e in edges], default=0.0)),
        "net_change_m": float(scan[-1]),
    }


def describe_terrain(a: dict, on_stairs: bool) -> str:
    """Numbers -> words for Jev."""
    if not a["n_edges"]:
        return "on the stairs, no more steps ahead" if on_stairs else "flat ground ahead"
    dist = a["first_edge_m"]
    where = ("right in front of the feet" if dist < 0.45 else "about half a metre ahead" if dist < 0.8
             else "about one metre ahead" if dist < 1.4 else "a couple of metres ahead")
    rise = a["mean_rise_m"]
    size = ("very low steps (about 4 cm or less)" if rise < LOW_STEP_MAX else "low steps (about 5 to 8 cm)" if rise < 0.085
            else "normal steps (about 8 to 12 cm)" if rise < 0.125 else "tall steps (over 12 cm)")
    n = a["n_up"] if a["first_dir"] == "up" else a["n_down"]
    count = "one step" if n == 1 else "a few steps" if n <= 3 else "a flight of several steps"
    kind = "going up" if a["first_dir"] == "up" else "going down"
    prefix = "on the stairs; next edge " if on_stairs else "stairs "
    return f"{prefix}{kind} {where}: {count}, {size}"


# -- the agent on the stairs world ---------------------------------------------------
# Gait modes for Unitree's policy: (forward speed m/s, gait period s). Step length = v * period / 2.
# "stride" uses a long period because on low steps the blind policy only gets over them with
# long, fast strides (see README: stairs sweep); slow short steps stub the toe.
VENDOR_MODES = {
    "stop":     (0.00, 0.8),
    "cautious": (0.30, 0.8),
    "normal":   (0.50, 0.8),
    "stride":   (0.70, 1.0),
}
SUP_MODES = {m: (v, v * p / 2) for m, (v, p) in VENDOR_MODES.items()}
MISSIONS = [
    "Walk to the stairs, climb them and continue across to the other side.",
    "Walk to the stairs and wait there. Do not climb them.",
]


def camera_scan(heights_fn, d, pelvis_body: int) -> tuple[np.ndarray, float, float]:
    """Same profile as height_scan, but from an elevation map (e.g. the head camera's). Unseen points
    are filled from the nearest seen point behind them; returns (scan, ground under robot, fraction seen)."""
    base = d.xpos[pelvis_body]
    yaw = yaw_of(d.xquat[pelvis_body])
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    h = heights_fn(base[:2] + SCAN_AHEAD[:, None] * fwd)
    under = heights_fn(base[None, :2])[0]
    seen = float(np.mean(~np.isnan(h)))
    under = 0.0 if np.isnan(under) else float(under)
    last = under
    for i in range(len(h)):
        h[i] = last = (last if np.isnan(h[i]) else h[i])
    return h - under, under, seen


def make_terrain(sim, heights_fn=None):
    """Terrain sensor for the Supervisor: (words for Jev, features for code).
    heights_fn(xy) -> ground heights (NaN = unseen) switches from perfect raycasts to e.g. the camera map."""
    def sense():
        if heights_fn is None:
            (scan, under), seen = height_scan(sim.m, sim.d, sim.pelvis), 1.0
        else:
            scan, under, seen = camera_scan(heights_fn, sim.d, sim.pelvis)
        a = analyse(scan)
        on = bool(under > 0.01)
        near = a["n_edges"] > 0 and a["first_edge_m"] < 1.5
        rise = a["mean_rise_m"] if (near or (on and a["n_edges"])) else 0.0
        # class edges sit between typical heights (4 | 5 cm) so a few mm of sensor error can't flip them
        size = ("flat" if rise == 0 else "low_steps" if rise < LOW_STEP_MAX
                else "mid_steps" if rise < 0.085 else "tall_steps")
        tag = ("on_" if on else "") + ("stairs_top" if on and rise == 0 else size)
        feats = {
            "tag": tag, "on_stairs": on, "ground_m": round(float(under), 3),
            "edge_dist_m": round(float(a["first_edge_m"]), 2) if a["n_edges"] else None,
            "edge_dir": a["first_dir"], "rise_ahead_m": round(rise, 3),
            "edge_close": bool(a["n_edges"] and a["first_edge_m"] < 1.0),
            "n_up": a["n_up"], "n_down": a["n_down"],
            "source": "raycast" if heights_fn is None else "camera", "seen": round(seen, 2),
        }
        text = describe_terrain(a, on)
        if seen < 0.6:
            text += " (much of the ground ahead not seen yet)"
        return text, feats
    return sense


def run_episode(sim, sup, duration: float = 25.0, perturb=None, seed: int = 0, on_tick=None):
    """Roll out Unitree's policy with the Supervisor choosing the gait; heading held in code."""
    period = [0.8]

    def cmd_fn(t):
        mode = sup(t)
        vx, period[0] = VENDOR_MODES[mode]
        cmd = (vx, 0.0, float(np.clip(-1.5 * yaw_of(sim.d.qpos[3:7]), -0.5, 0.5)))
        if on_tick:
            on_tick(t, mode, cmd, period[0])
        return cmd

    ep, _ = sim.run(duration=duration, cmd_fn=cmd_fn, period_fn=lambda t: period[0], perturb=perturb, seed=seed)
    xs = np.array(ep.base_pos)[:, 0] if ep.base_pos else np.zeros(1)
    st = sim.stairs
    top_start = st.start_x + st.n_up * st.tread
    return ep, {
        "fell": ep.fell, "fall_time": ep.fall_time, "max_x": round(float(xs.max()), 2),
        "reached_stairs": bool(xs.max() > st.start_x - 0.6),
        "reached_top": bool(xs.max() > top_start + 0.2),
        "crossed": bool(xs.max() > st.end_x + 0.3),
        "stopped_before": bool(not ep.fell and st.start_x - 0.9 < xs.max() < st.start_x),
    }


def main():
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    from jev_agent.brain import make_brain
    from jev_agent.learner import Learner
    from jev_agent.perception import VendorRobot
    from jev_agent.supervisor import Supervisor

    ap = argparse.ArgumentParser(description="Jev agent + Unitree G1 policy on a staircase")
    ap.add_argument("--brain", choices=["jev", "offline"], default="jev")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--rises", default="0.04,0.08", help="step heights (m) to cycle through")
    ap.add_argument("--mission", help="default: cycle through MISSIONS")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--video", default=None, help="record the last episode")
    ap.add_argument("--require-jev", action="store_true", help="fail instead of going offline without a key")
    ap.add_argument("--terrain", choices=["raycast", "camera"], default="raycast",
                    help="what the stair detector reads: perfect raycasts or the head depth camera's elevation map")
    a = ap.parse_args()

    tag = a.tag or f"stairs_{a.brain}"
    out = Path("runs/jev_agent") / tag
    out.mkdir(parents=True, exist_ok=True)
    learner = Learner(out / "learner.json")
    brain = make_brain(a.brain, terrain=True, require_jev=a.require_jev)
    rises = [float(r) for r in a.rises.split(",")]
    missions = [a.mission] if a.mission else MISSIONS
    with open(out / "episodes.jsonl", "a") as ep_f, open(out / "decisions.jsonl", "a") as dec_f:
        for i in range(a.episodes):
            rise, mission = rises[i % len(rises)], missions[(i // len(rises)) % len(missions)]
            cam_on = a.terrain == "camera"
            sim = build(Staircase(rise=rise), camera=cam_on)
            heights_fn, capture = None, None
            if cam_on:
                from jev_agent.vision import ElevationMap, HeadCamera
                cam, emap = HeadCamera(sim.m, seed=i), ElevationMap()
                heights_fn = emap.height

                def capture(t, *_):
                    if int(round(t / 0.02)) % 10 == 0:      # 5 Hz
                        emap.integrate(cam.capture(sim.d, rgb=False)["points"], t)
            sup = Supervisor(VendorRobot(sim), brain, learner, mission, SUP_MODES,
                             terrain=make_terrain(sim, heights_fn), seed=i)
            frames, grab = [], None
            if a.video and i == a.episodes - 1:
                ren = mujoco.Renderer(sim.m, 480, 640)

                def grab(t, *_):
                    if int(round(t / 0.02)) % 2 == 0:
                        ren.update_scene(sim.d, camera=sim._camera())
                        frames.append(ren.render())
            hooks = [h for h in (capture, grab) if h]
            ep, res = run_episode(sim, sup, a.duration, seed=i,
                                  on_tick=(lambda *x: [h(*x) for h in hooks]) if hooks else None)
            decs = sup.finish(ep.fall_time)
            learner.learn(decs)
            learner.save()
            rec = {"episode": i, "rise_m": rise, "mission": mission, **res,
                   "stairs_attempt": sup.attempt,
                   "modes": dict(Counter(d["mode"] for d in decs)),
                   "reasons": dict(Counter(d["reason"] for d in decs)),
                   "brain_errors": sum(bool(d["jev"]["error"]) for d in decs)}
            for d in decs:
                dec_f.write(json.dumps({"episode": i, "rise_m": rise, **d}) + "\n")
            ep_f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if frames:
                import imageio.v2 as imageio
                imageio.mimsave(a.video, frames, fps=25)
                print("video:", a.video)
    print("stairs skill (gait: [crossed, failed]) per step size:")
    for size, g in learner.stairs.stats.items():
        print(f"  {size:10s} {g}  -> {'TOO HARD, will stop before these' if learner.stairs.too_hard(size) else 'still learning / climbable'}")


if __name__ == "__main__":
    main()

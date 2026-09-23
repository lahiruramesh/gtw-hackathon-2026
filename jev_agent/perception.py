"""Turn raw MuJoCo sensor readings into a compact, text-friendly snapshot.

Runs every control tick (50 Hz) and aggregates a window between decisions.
All arithmetic stays here in code; Jev only ever sees named buckets
("slightly tilted forward", "feet slipping"), because jev-1.13 is weak at
comparing raw numbers (docs: model-jaggedness/jev-1.13, "Math and Numbers").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

PUSH_DV = 0.3          # m/s jump in base velocity within one tick = external push
SLIP_SPEED = 0.15      # m/s horizontal speed of a foot that stays in contact
PUSH_MEMORY_S = 1.5    # how long a push counts as "recent"
SLIP_MEMORY_WINDOWS = 3  # slip is reported as the worst of this window and the previous 3 (~2 s): a slippery
                         # floor doesn't become grippy because one half-second happened to go well
STANCE_FORCE_N = 60.0  # normal force per contact point above which the foot is bearing weight
MIN_AIR_S = 0.1        # s a foot must be airborne before its landing counts as a step (as in g1pipe.evaluate)


@dataclass
class Window:
    """Stats accumulated since the last decision."""
    max_tilt_deg: float = 0.0
    max_gyro: float = 0.0
    max_slip: float = 0.0
    vx_err_sum: float = 0.0
    lat_speed_sum: float = 0.0
    n: int = 0
    step_errs: list = field(default_factory=list)
    start_xy: np.ndarray | None = None
    progress_m: float = 0.0


def contact_slip(m, d, feet) -> np.ndarray:
    """Per foot body, max horizontal speed of the foot at its weight-bearing floor contacts (0 if unloaded).
    Measured at the contact point and only under load, so heel-strike, toe-off and a normal
    heel-to-toe roll don't read as slipping."""
    out, vel, force = np.zeros(len(feet)), np.zeros(6), np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for k, fb in enumerate(feet):
            if fb in (b1, b2) and m.body_rootid[b2 if b1 == fb else b1] == 0:
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] < STANCE_FORCE_N:            # only a foot carrying weight can "slip"
                    continue
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb, vel, 0)
                # mj_objectVelocity gives the velocity at the body COM (xipos), so the lever arm starts there
                v = vel[3:] + np.cross(vel[:3], c.pos - d.xipos[fb])
                out[k] = max(out[k], float(np.linalg.norm(v[:2])))
    return out


class PPORobot:
    """Sensor access for g1pipe.evaluate.PolicyRunner (Playground G1 model)."""
    name = "v1"

    def __init__(self, runner):
        self.r, self.dt = runner, runner.ctrl_dt
        self.feet = runner.m.site_bodyid[runner.feet_sites]   # ankle-roll links that carry the foot geoms

    def up_z(self):
        return float(self.r.d.xmat[self.r.torso].reshape(3, 3)[2, 2])

    def gravity_local(self):
        return self.r.d.site_xmat[self.r.pelvis_imu].reshape(3, 3).T @ np.array([0, 0, -1.0])

    def gyro(self):
        return np.asarray(self.r._sensor("gyro_pelvis"))

    def local_vel(self):
        return np.asarray(self.r._sensor("local_linvel_pelvis"))

    def feet_xy(self):
        return self.r.d.site_xpos[self.r.feet_sites][:, :2].copy()

    def contacts(self):
        return np.array([self.r.d.sensordata[a] > 0 for a in self.r.floor_sensors])

    def contact_slip(self):
        return contact_slip(self.r.m, self.r.d, self.feet)

    def fwd_xy(self):
        f = self.r.d.xmat[self.r.torso].reshape(3, 3)[:2, 0]
        return f / (np.linalg.norm(f) + 1e-9)

    def base_vel_xy(self):
        return self.r.d.qvel[:2].copy()

    def base_xy(self):
        return self.r.d.qpos[:2].copy()

    def height(self):
        return float(self.r.d.qpos[2])


class VendorRobot:
    """Sensor access for g1pipe.sim.G1Sim (Unitree 12-DoF model)."""
    name = "vendor"

    def __init__(self, sim, dt: float = 0.02):
        self.s, self.dt = sim, dt

    def _R(self):
        return self.s.d.xmat[self.s.pelvis].reshape(3, 3)

    def up_z(self):
        return float(self._R()[2, 2])

    def gravity_local(self):
        return self._R().T @ np.array([0, 0, -1.0])

    def gyro(self):
        return self.s.d.qvel[3:6].copy()          # free joint angular velocity is in the body frame

    def local_vel(self):
        return self._R().T @ self.s.d.qvel[:3]

    def feet_xy(self):
        return self.s.d.xpos[self.s.feet][:, :2].copy()

    def contacts(self):
        return np.array(self.s._foot_contacts())

    def contact_slip(self):
        return contact_slip(self.s.m, self.s.d, self.s.feet)

    def fwd_xy(self):
        f = self._R()[:2, 0]
        return f / (np.linalg.norm(f) + 1e-9)

    def base_vel_xy(self):
        return self.s.d.qvel[:2].copy()

    def base_xy(self):
        return self.s.d.qpos[:2].copy()

    def height(self):
        return float(self.s.d.qpos[2])


def load_calibration(name: str) -> dict | None:
    """This robot's normal-walking levels (jev_agent/calibration.json), or None to use absolute buckets."""
    import json
    from pathlib import Path
    f = Path(__file__).with_name("calibration.json")
    return json.loads(f.read_text()).get(name) if f.is_file() else None


class Perception:
    def __init__(self, robot):
        self.r = robot
        self.calib = load_calibration(getattr(robot, "name", ""))
        self.reset()

    def reset(self):
        self.win = Window()
        self.prev_vel = None
        self.prev_feet = None
        self.prev_contact = np.array([True, True])
        self.air_time = np.zeros(2)
        self.last_push_t = -1e9
        self.t = 0.0
        self.slip_hist = []   # last few per-foot contact-slip readings
        self.slip_windows = []  # max_slip of recent windows

    def _tilt(self):
        """Tilt angle of the torso from vertical, and gravity in the pelvis frame."""
        return float(np.degrees(np.arccos(np.clip(self.r.up_z(), -1, 1)))), self.r.gravity_local()

    def update(self, t: float, cmd_vx: float, step_cmd: float):
        """Call once per control tick with the command currently applied."""
        r = self.r
        self.t = t
        tilt, _ = self._tilt()
        gyro = float(np.linalg.norm(r.gyro()))
        vx, vy = r.local_vel()[:2]
        vel, feet, contact, fwd = r.base_vel_xy(), r.feet_xy(), r.contacts(), r.fwd_xy()

        if self.prev_vel is not None and np.linalg.norm(vel - self.prev_vel) > PUSH_DV:
            self.last_push_t = t
        if hasattr(r, "contact_slip"):
            # sustained slip: the foot kept sliding for 3 ticks (60 ms); one-tick heel-strike spikes don't count
            self.slip_hist = (self.slip_hist + [r.contact_slip()])[-3:]
            if len(self.slip_hist) == 3:
                self.win.max_slip = max(self.win.max_slip, float(np.min(self.slip_hist, axis=0).max()))
        elif self.prev_feet is not None:
            both = contact & self.prev_contact
            speed = np.linalg.norm(feet - self.prev_feet, axis=1) / r.dt
            if both.any():
                self.win.max_slip = max(self.win.max_slip, float(speed[both].max()))
        # touchdown step length, same definition as the evaluator: only after a real swing, so contact flicker isn't a step
        for i in (0, 1):
            if contact[i] and not self.prev_contact[i] and self.air_time[i] >= MIN_AIR_S and abs(cmd_vx) > 0.15:
                self.win.step_errs.append(float(np.dot(feet[i] - feet[1 - i], fwd)) - step_cmd)
        self.air_time = np.where(contact, 0.0, self.air_time + r.dt)

        w = self.win
        if w.start_xy is None:
            w.start_xy = r.base_xy()
        w.max_tilt_deg = max(w.max_tilt_deg, tilt)
        w.max_gyro = max(w.max_gyro, gyro)
        w.vx_err_sum += abs(vx - cmd_vx)
        w.lat_speed_sum += abs(vy)
        w.n += 1
        w.progress_m = float(np.dot(r.base_xy() - w.start_xy, fwd))
        self.prev_vel, self.prev_feet, self.prev_contact = vel, feet, contact

    # -- decision-time summary -----------------------------------------------
    def snapshot(self) -> dict:
        """Numeric features for the current window (used by code and the learner)."""
        w = self.win
        tilt, g = self._tilt()
        n = max(w.n, 1)
        return {
            "tilt_deg": tilt,
            "max_tilt_deg": w.max_tilt_deg,
            "lean_fwd": float(g[0]),
            "lean_left": float(g[1]),
            "max_gyro": w.max_gyro,
            "max_slip": max([w.max_slip, *self.slip_windows]),
            "vx_err": w.vx_err_sum / n,
            "lat_speed": w.lat_speed_sum / n,
            "step_err": float(np.mean(np.abs(w.step_errs))) if w.step_errs else 0.0,
            "since_push_s": min(self.t - self.last_push_t, 99.0),
            "height": self.r.height(),
            "progress_m": w.progress_m,
        }

    def end_window(self):
        self.slip_windows = (self.slip_windows + [self.win.max_slip])[-SLIP_MEMORY_WINDOWS:]
        self.win = Window()


# -- numbers -> words ----------------------------------------------------------
def _bucket(x, edges, names):
    for e, n in zip(edges, names):
        if x < e:
            return n
    return names[-1]


# Words for "how far from this robot's normal walking": ratio to its calibrated p90 level.
RATIO_EDGES = (1.3, 2.0, 3.5)


def describe(s: dict, calib: dict | None = None) -> dict:
    """Semantic description of a snapshot. This is what goes into Jev's state.
    With a calibration, each reading is judged against this robot's own normal walking (a lively
    policy that always wobbles a bit is "calm" when it wobbles as usual); without, absolute buckets."""
    def level(key, absolute_edges):
        if calib and calib.get(key):
            return _bucket(s[key] / calib[key], RATIO_EDGES, (0, 1, 2, 3))
        return _bucket(s[key], absolute_edges, (0, 1, 2, 3))

    t = level("max_tilt_deg", (6, 12, 20))
    if s["max_tilt_deg"] > 20:                       # absolute: a big tilt is bad whatever is "normal"
        t = 3
    tilt = ("upright", "slightly tilted", "clearly tilted", "severely tilted")[t]
    if t:
        dirs = []
        if abs(s["lean_fwd"]) > 0.08:
            dirs.append("forward" if s["lean_fwd"] > 0 else "backward")
        if abs(s["lean_left"]) > 0.08:
            dirs.append("to the left" if s["lean_left"] > 0 else "to the right")
        tilt += (" " + " and ".join(dirs)) if dirs else ""
    return {
        "torso_posture": tilt,
        "body_rotation": ("calm", "some wobble", "strong wobble", "spinning or tumbling")[level("max_gyro", (1.0, 2.5, 4.5))],
        "foot_grip": ("feet firmly planted", "tiny foot movement while planted", "feet slipping on the ground",
                      "feet sliding badly")[level("max_slip", (0.08, 0.15, 0.4))],
        "speed_tracking": ("on target", "slightly off target", "well off target",
                           "not following the command")[level("vx_err", (0.1, 0.25, 0.5))],
        "sideways_drift": ("none", "some", "strong", "strong")[level("lat_speed", (0.1, 0.25, 0.4))],
        "step_placement": ("accurate", "a little off", "inaccurate", "erratic")[level("step_err", (0.04, 0.08, 0.15))],
        "recent_push": ("pushed within the last second" if s["since_push_s"] < 1.0
                        else "pushed a moment ago" if s["since_push_s"] < PUSH_MEMORY_S
                        else "no push"),
    }

"""Evaluate a trained step-length policy in plain MuJoCo (C engine, CPU).

Training ran in MJX/Warp on a GPU; evaluation runs the same model in the
reference MuJoCo engine with no observation noise. That is our cheap
sim-to-sim check: the policy never saw this solver during training.

    python -m g1pipe.evaluate runs/steplength_v1/params.pkl --vx 0.6 --step 0.25 --video out.mp4
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from g1pipe.steplength_env import GAIT_FREQ_RANGE, StepLength, default_config


MIN_AIR_S = 0.1   # s a foot must be airborne before its landing counts as a step
HEADING_KP = 1.5  # rad/s of yaw-rate command per rad of heading error (outer loop)


@dataclass
class Perturb:
    friction: float = 1.0        # multiplier on foot-floor friction
    payload_kg: float = 0.0      # added to torso
    push_every_s: float = 0.0
    push_vel: float = 0.0        # m/s velocity kick, random direction
    action_delay_steps: int = 0  # 20 ms each


def command_to_gait(vx: float, step_len: float) -> tuple[float, float]:
    """Operator gives (speed, step length); return (gait_freq, achievable step length)."""
    if abs(vx) < 1e-3 or step_len <= 0:
        return 1.35, 0.0
    f = float(np.clip(abs(vx) / (2 * step_len), *GAIT_FREQ_RANGE))
    return f, vx / (2 * f)


class PolicyRunner:
    heading_wz_max = 0.5   # rad/s, yaw-rate limit of the heading-hold loop
    def __init__(self, params_path: str | Path, env=None):
        blob = pickle.load(open(params_path, "rb"))
        if not (isinstance(blob, dict) and "params" in blob):
            # a periodic checkpoint (ckpt_*.pkl) holds raw params; network shape comes from the run's params.pkl
            meta = pickle.load(open(Path(params_path).with_name("params.pkl"), "rb"))
            blob = {**meta, "params": blob}
        if env is None:
            cfg = default_config()
            cfg.impl = "jax"
            env = StepLength(config=cfg)
        self.env = env
        cfg = env._config
        self.m = self.env.mj_model
        self.d = mujoco.MjData(self.m)
        self.ctrl_dt, self.sim_dt = cfg.ctrl_dt, cfg.sim_dt
        self.n_sub = int(round(cfg.ctrl_dt / cfg.sim_dt))
        self.action_scale = cfg.action_scale
        self.default = np.array(self.env._default_pose)
        self.init_q = np.array(self.env._init_q)
        self.feet_sites = np.array(self.env._feet_site_id)
        self.torso = self.env._torso_body_id
        self.pelvis_imu = self.env._pelvis_imu_site_id
        self.floor_sensors = [self.m.sensor_adr[s] for s in self.env._feet_floor_found_sensor]
        self._nominal = (self.m.pair_friction.copy(), self.m.body_mass.copy())

        nets = ppo_networks.make_ppo_networks(
            blob["obs_size"], blob["action_size"],
            preprocess_observations_fn=running_statistics.normalize, **blob["network_factory"])
        make_inf = ppo_networks.make_inference_fn(nets)
        self._policy = jax.jit(make_inf(blob["params"], deterministic=True))
        self._priv = np.zeros(blob["obs_size"]["privileged_state"][0], np.float32)
        self._key = jax.random.PRNGKey(0)

    def _ground_z(self, xy) -> float:
        """Ground height under xy; flat floor here, terrain in subclasses (fall = pelvis too low above it)."""
        return 0.0

    def _sensor(self, name):
        s = self.m.sensor(name)
        return self.d.sensordata[s.adr[0]: s.adr[0] + s.dim[0]]

    def _obs(self, cmd3, last_act, phase, step_cmd, f):
        gravity = self.d.site_xmat[self.pelvis_imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        state = np.hstack([
            self._sensor("local_linvel_pelvis"), self._sensor("gyro_pelvis"), gravity, cmd3,
            self.d.qpos[7:] - self.default, self.d.qvel[6:], last_act,
            np.cos(phase), np.sin(phase), step_cmd * 4.0, f - 1.35,
        ]).astype(np.float32)
        return {"state": jp.asarray(state), "privileged_state": jp.asarray(self._priv)}

    def run(self, vx=0.5, step_len=0.25, duration=12.0, perturb: Perturb | None = None,
            renderer=None, fps=30, seed=0, settle_s=2.0, schedule=None, heading_hold=True):
        """schedule(t) -> (vx, step_len) lets you change the command mid-run.

        heading_hold: the policy tracks a yaw *rate*, so small errors integrate into heading
        drift (20-35 deg over 9 s even in the training engine). A classical P-loop on heading
        feeds a small yaw-rate command, the way a joystick policy is deployed in practice.
        """
        p = perturb or Perturb()
        rng = np.random.default_rng(seed)
        pf, bm = self._nominal
        self.m.pair_friction[:] = pf
        self.m.pair_friction[:, 0:2] = pf[:, 0:2] * p.friction
        self.m.body_mass[:] = bm
        self.m.body_mass[self.torso] += p.payload_kg

        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.init_q
        self.d.ctrl[:] = self.init_q[7:]
        mujoco.mj_forward(self.m, self.d)

        # Match the training env's timing exactly: after each physics step it builds the next
        # observation *before* advancing the phase and *before* storing the action just applied,
        # so obs_k carries phase_{k-1} and the action from two steps back (a_{k-2}).
        phase = np.array([0.0, np.pi])
        obs_phase = phase.copy()
        obs_act = np.zeros(self.m.nu, np.float32)
        prev_act = np.zeros(self.m.nu, np.float32)
        queue = [prev_act.copy()] * (p.action_delay_steps + 1)
        prev_contact = np.array([True, True])
        air_time = np.zeros(2)
        next_push = p.push_every_s if p.push_every_s > 0 else np.inf
        frames, steps, log = [], [], {"t": [], "vx": [], "cmd_vx": [], "cmd_step": []}
        fell_at = None
        frame_every = max(1, int(round(1 / (fps * self.ctrl_dt))))

        for k in range(int(duration / self.ctrl_dt)):
            t = k * self.ctrl_dt
            cvx, cstep = schedule(t) if schedule else (vx, step_len)
            f, step_cmd = command_to_gait(cvx, cstep)
            R = self.d.xmat[self.torso].reshape(3, 3)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            if k == 0:
                yaw0 = yaw
            wz = float(np.clip(HEADING_KP * np.angle(np.exp(1j * (yaw0 - yaw))), -self.heading_wz_max, self.heading_wz_max)) if heading_hold else 0.0
            obs = self._obs(np.array([cvx, 0.0, wz]), obs_act, obs_phase, step_cmd, f)
            self._key, sub = jax.random.split(self._key)
            act, _ = self._policy(obs, sub)
            act = np.asarray(act, np.float32)
            obs_act, prev_act = prev_act, act
            queue.append(act.copy())
            applied = queue.pop(0)
            self.d.ctrl[:] = self.default + applied * self.action_scale
            for _ in range(self.n_sub):
                mujoco.mj_step(self.m, self.d)
            if t >= next_push:
                ang = rng.uniform(0, 2 * np.pi)
                self.d.qvel[:2] += p.push_vel * np.array([np.cos(ang), np.sin(ang)])
                next_push += p.push_every_s
            obs_phase = phase
            phase = np.fmod(phase + 2 * np.pi * self.ctrl_dt * f + np.pi, 2 * np.pi) - np.pi

            contact = np.array([self.d.sensordata[a] > 0 for a in self.floor_sensors])
            # A touchdown counts only after a real swing (>= MIN_AIR_S airborne); scuffs and
            # contact flicker otherwise register as spurious "steps" of a few cm.
            touchdown = contact & ~prev_contact & (air_time >= MIN_AIR_S)
            if t >= settle_s and abs(cvx) > 0.15:
                fwd = self.d.xmat[self.torso].reshape(3, 3)[:2, 0]
                fwd /= np.linalg.norm(fwd) + 1e-9
                feet = self.d.site_xpos[self.feet_sites][:, :2]
                for i in (0, 1):
                    if touchdown[i]:
                        steps.append((t, i, float(np.dot(feet[i] - feet[1 - i], fwd)), step_cmd))
            air_time = np.where(contact, 0.0, air_time + self.ctrl_dt)
            prev_contact = contact
            log["t"].append(t); log["vx"].append(float(self._sensor("local_linvel_pelvis")[0]))
            log["cmd_vx"].append(cvx); log["cmd_step"].append(step_cmd)

            up = self.d.xmat[self.torso].reshape(3, 3)[2, 2]
            if up < 0.3 or self.d.qpos[2] - self._ground_z(self.d.qpos[:2]) < 0.4:
                fell_at = t
                break
            if renderer is not None and k % frame_every == 0:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = self.d.qpos[:3] + np.array([0, 0, -0.25])
                cam.distance, cam.azimuth, cam.elevation = 2.8, 120, -12
                renderer.update_scene(self.d, camera=cam)
                frames.append(renderer.render())
        return summarize(steps, log, fell_at, settle_s), frames, steps, log


def summarize(steps, log, fell_at, settle_s):
    t = np.array(log["t"])
    vx, cvx = np.array(log["vx"]), np.array(log["cmd_vx"])
    m = t >= settle_s
    ach = np.array([s[2] for s in steps]) if steps else np.array([np.nan])
    cmd = np.array([s[3] for s in steps]) if steps else np.array([np.nan])
    return {
        "fell": fell_at is not None,
        "fall_time": fell_at,
        "n_steps": len(steps),
        "step_cmd_mean_m": float(np.nanmean(cmd)),
        "step_achieved_mean_m": float(np.nanmean(ach)),
        "step_abs_err_m": float(np.nanmean(np.abs(ach - cmd))),
        "vx_abs_err": float(np.mean(np.abs(vx[m] - cvx[m]))) if m.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    r = PolicyRunner(a.params)
    ren = mujoco.Renderer(r.m, 480, 640) if a.video else None
    s, frames, _, _ = r.run(vx=a.vx, step_len=a.step, duration=a.duration, renderer=ren)
    print(json.dumps(s, indent=2))
    if a.video:
        import imageio.v2 as imageio
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(a.video, frames, fps=30)
        print("video:", a.video)


if __name__ == "__main__":
    main()

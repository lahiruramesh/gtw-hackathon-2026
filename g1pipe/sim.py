"""Headless MuJoCo runner for G1 locomotion policies.

Mirrors unitree_rl_gym/deploy/deploy_mujoco (same observation layout, PD gains,
50 Hz policy / 500 Hz physics) so policies trained in Isaac Gym, MJX or here are
evaluated identically. Adds gait metrics, perturbations and video capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
UNITREE = ROOT / "third_party" / "unitree_rl_gym"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def gravity_orientation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def yaw_of(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


@dataclass
class Perturb:
    """Stress-test knobs. Defaults reproduce the nominal deploy setup."""
    friction: float = 1.0          # multiplier on floor sliding friction
    payload_kg: float = 0.0        # extra mass added to the pelvis
    motor_strength: float = 1.0    # multiplier on PD gains
    action_delay_steps: int = 0    # policy steps of latency (20 ms each)
    push_every_s: float = 0.0      # 0 = no pushes
    push_vel: float = 0.0          # lateral velocity kick (m/s) per push
    obs_noise: float = 0.0         # gaussian noise std on joint observations


@dataclass
class Episode:
    t: list = field(default_factory=list)
    base_pos: list = field(default_factory=list)
    base_vel: list = field(default_factory=list)
    steps: list = field(default_factory=list)   # (time, foot, step_length_m)
    fell: bool = False
    fall_time: float | None = None


class G1Sim:
    def __init__(self, config: str = "g1.yaml", policy_path: str | None = None, scene: str | None = None):
        cfg_path = UNITREE / "deploy" / "deploy_mujoco" / "configs" / config
        cfg = yaml.safe_load(cfg_path.read_text())
        sub = lambda s: s.replace("{LEGGED_GYM_ROOT_DIR}", str(UNITREE))
        self.cfg = cfg
        self.xml_path = scene or sub(cfg["xml_path"])
        self.policy_path = policy_path or sub(cfg["policy_path"])
        self.dt = cfg["simulation_dt"]
        self.decimation = cfg["control_decimation"]
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default = np.array(cfg["default_angles"], dtype=np.float32)
        self.n_act = cfg["num_actions"]
        self.n_obs = cfg["num_obs"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.policy = torch.jit.load(self.policy_path)
        self.policy.eval()

        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.m.opt.timestep = self.dt
        self.d = mujoco.MjData(self.m)
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._nominal = (self.m.geom_friction.copy(), self.m.body_mass.copy())

    # ------------------------------------------------------------------ setup
    def _apply(self, p: Perturb):
        fr, mass = self._nominal
        self.m.geom_friction[:] = fr
        self.m.body_mass[:] = mass
        # MuJoCo uses the larger friction of the two touching geoms, so scale the feet as well as the
        # floor; scaling only the floor had no effect while the feet stayed at 1.0.
        feet = np.isin(self.m.geom_bodyid, self.feet)
        self.m.geom_friction[self.floor, 0] = fr[self.floor, 0] * p.friction
        self.m.geom_friction[feet, 0] = fr[feet, 0] * p.friction
        self.m.body_mass[self.pelvis] += p.payload_kg

    def _foot_contacts(self) -> list[bool]:
        """Foot k is in contact if any of its geoms touches a non-robot (world) geom."""
        on = [False, False]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for k, fb in enumerate(self.feet):
                if (b1 == fb and self.m.body_rootid[b2] == 0) or (b2 == fb and self.m.body_rootid[b1] == 0):
                    on[k] = True
        return on

    def reset_policy_memory(self):
        """Unitree's G1 policy is an LSTM that stores its state in module buffers; zero them per episode."""
        for name in ("hidden_state", "cell_state"):
            if hasattr(self.policy, name):
                getattr(self.policy, name).zero_()

    # -------------------------------------------------------------------- run
    def run(self, cmd=(0.5, 0.0, 0.0), period: float = 0.8, duration: float = 10.0,
            perturb: Perturb | None = None, seed: int = 0, renderer=None, fps: int = 30,
            settle_s: float = 2.0, cmd_fn=None, period_fn=None, target_fn=None) -> tuple[Episode, list]:
        """Roll out the policy. `cmd_fn(t)` / `period_fn(t)` override constants for scheduled commands.
        `target_fn(target, phase, t)` may adjust the PD joint targets each physics step (e.g. swing-leg shaping).

        Step lengths recorded after `settle_s` only, so start-up transients are excluded.
        """
        p = perturb or Perturb()
        rng = np.random.default_rng(seed)
        self._apply(p)
        self.reset_policy_memory()
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[7:] = self.default
        mujoco.mj_forward(self.m, self.d)

        ep, frames = Episode(), []
        action = np.zeros(self.n_act, dtype=np.float32)
        target = self.default.copy()
        queue = [action.copy()] * (p.action_delay_steps + 1)
        obs = np.zeros(self.n_obs, dtype=np.float32)
        prev_contact = [True, True]
        phase_t = 0.0
        next_push = p.push_every_s if p.push_every_s > 0 else np.inf
        frame_every = max(1, int(round(1 / (fps * self.dt))))
        kp, kd = self.kps * p.motor_strength, self.kds * p.motor_strength

        n = int(duration / self.dt)
        for k in range(n):
            t = k * self.dt
            tgt = target_fn(target, phase_t % 1.0, t) if target_fn else target
            self.d.ctrl[:] = (tgt - self.d.qpos[7:]) * kp - self.d.qvel[6:] * kd
            mujoco.mj_step(self.m, self.d)

            if t >= next_push:
                self.d.qvel[1] += p.push_vel * rng.choice([-1, 1])
                next_push += p.push_every_s

            cur_period = period_fn(t) if period_fn else period
            phase_t += self.dt / cur_period  # integrate phase so period changes stay continuous

            if (k + 1) % self.decimation == 0:
                c = np.asarray(cmd_fn(t) if cmd_fn else cmd, dtype=np.float32)
                qj = (self.d.qpos[7:] - self.default) + rng.normal(0, p.obs_noise, self.n_act)
                dqj = self.d.qvel[6:] * self.cfg["dof_vel_scale"]
                ph = phase_t % 1.0
                obs[:3] = self.d.qvel[3:6] * self.cfg["ang_vel_scale"]
                obs[3:6] = gravity_orientation(self.d.qpos[3:7])
                obs[6:9] = c * self.cmd_scale
                obs[9:9 + self.n_act] = qj * self.cfg["dof_pos_scale"]
                obs[9 + self.n_act:9 + 2 * self.n_act] = dqj
                obs[9 + 2 * self.n_act:9 + 3 * self.n_act] = action
                obs[9 + 3 * self.n_act:9 + 3 * self.n_act + 2] = (np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph))
                with torch.no_grad():
                    action = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
                queue.append(action.copy())
                target = queue.pop(0) * self.cfg["action_scale"] + self.default

                # metrics at policy rate
                ep.t.append(t)
                ep.base_pos.append(self.d.qpos[:3].copy())
                ep.base_vel.append(self.d.qvel[:3].copy())
                contact = self._foot_contacts()
                heading = yaw_of(self.d.qpos[3:7])
                fwd = np.array([np.cos(heading), np.sin(heading)])
                for f in (0, 1):
                    if contact[f] and not prev_contact[f] and t >= settle_s:
                        this = self.d.xpos[self.feet[f]][:2]
                        other = self.d.xpos[self.feet[1 - f]][:2]
                        ep.steps.append((t, f, float(np.dot(this - other, fwd))))
                prev_contact = contact

                if self.d.qpos[2] < 0.45 or gravity_orientation(self.d.qpos[3:7])[2] > -0.5:
                    ep.fell, ep.fall_time = True, t
                    break

            if renderer is not None and k % frame_every == 0:
                renderer.update_scene(self.d, camera=self._camera())
                frames.append(renderer.render())
        return ep, frames

    def _camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.d.qpos[:3] + np.array([0, 0, -0.2])
        cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -15
        return cam


def summarize(ep: Episode, cmd_vx: float, period: float) -> dict:
    lengths = np.array([s[2] for s in ep.steps]) if ep.steps else np.array([np.nan])
    vel = np.array(ep.base_vel)
    t = np.array(ep.t)
    steady = vel[t >= 2.0] if len(t) and (t >= 2.0).any() else vel
    return {
        "cmd_vx": cmd_vx,
        "period": period,
        "expected_step_m": cmd_vx * period / 2,
        "step_mean_m": float(np.nanmean(lengths)),
        "step_std_m": float(np.nanstd(lengths)),
        "n_steps": int(len(ep.steps)),
        "cadence_steps_per_s": float(len(ep.steps) / max(1e-6, (t[-1] - 2.0))) if len(t) and t[-1] > 2 else float("nan"),
        "vx_mean": float(steady[:, 0].mean()) if len(steady) else float("nan"),
        "fell": ep.fell,
        "fall_time": ep.fall_time,
    }

"""G1 step-length task: MuJoCo Playground's G1 joystick env + explicit step-length control.

Operator command: forward speed vx and step length l*. Cadence follows from them:
    gait_freq f = |vx| / (2 l*)      (two steps per gait cycle)
so during training we sample (vx, f) over a wide range and give the policy both
f and l* = vx / (2 f) as observations. A touchdown reward pays for landing the
swing foot l* ahead of the stance foot along the heading.

Changes vs. upstream G1 Joystick (kept deliberately small, see docs/pipeline.md):
  * gait_freq range widened 1.25–1.5 Hz  ->  GAIT_FREQ_RANGE
  * obs "state" gets [l*, f] appended (+2)
  * new reward term "step_length"
"""
from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

mjx_env.ensure_menagerie_exists()  # downloads G1 meshes on first use

GAIT_FREQ_RANGE = (0.9, 1.8)      # Hz  -> gait period 0.56–1.11 s
STEP_SIGMA = 0.05                 # m, width of touchdown reward
MIN_WALK_SPEED = 0.15             # m/s, below this the step reward is off


def default_config() -> config_dict.ConfigDict:
    cfg = g1_joystick.default_config()
    cfg.reward_config.scales.step_length = 1.5
    cfg.lin_vel_x = [-0.5, 1.2]
    return cfg


class StepLength(g1_joystick.Joystick):

    def __init__(self, task: str = "flat_terrain", config=None, config_overrides=None):
        super().__init__(task=task, config=config or default_config(), config_overrides=config_overrides)

    # -- command ---------------------------------------------------------------
    def _sample_gait(self, rng, command):
        f = jax.random.uniform(rng, (), minval=GAIT_FREQ_RANGE[0], maxval=GAIT_FREQ_RANGE[1])
        step_cmd = command[0] / (2.0 * f)
        return f, step_cmd

    def _set_gait(self, info, rng):
        f, step_cmd = self._sample_gait(rng, info["command"])
        info["gait_freq"] = f
        info["step_cmd"] = step_cmd
        info["phase_dt"] = 2 * jp.pi * self.dt * jp.array([f])
        return info

    # -- env API ---------------------------------------------------------------
    def reset(self, rng):
        rng, gait_rng = jax.random.split(rng)
        state = super().reset(rng)
        info = self._set_gait(dict(state.info), gait_rng)
        state.metrics["step_len_err"] = jp.zeros(())
        contact = self._contact(state.data)
        obs = self._get_obs(state.data, info, contact)
        return state.replace(obs=obs, info=info)

    def step(self, state, action):
        state = super().step(state, action)
        # Upstream resamples the velocity command when info["step"] wraps to 0; resample cadence too.
        info = dict(state.info)
        info["rng"], gait_rng = jax.random.split(info["rng"])
        new = self._set_gait(dict(info), gait_rng)
        wrapped = info["step"] == 0
        for k in ("gait_freq", "step_cmd", "phase_dt"):
            info[k] = jp.where(wrapped, new[k], info[k])
        return state.replace(info=info)

    def _contact(self, data):
        return jp.array([
            data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._feet_floor_found_sensor
        ])

    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        extra = jp.hstack([
            info.get("step_cmd", jp.zeros(())) * 4.0,          # ~[-1, 1] scale
            info.get("gait_freq", jp.array(1.35)) - 1.35,
        ])
        return {
            "state": jp.hstack([obs["state"], extra]),
            "privileged_state": jp.hstack([obs["privileged_state"], extra]),
        }

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        rewards["step_length"] = self._reward_step_length(data, info, first_contact, metrics)
        return rewards

    def _reward_step_length(self, data, info, first_contact, metrics):
        feet_xy = data.site_xpos[self._feet_site_id][:, :2]
        fwd = data.xmat[self._torso_body_id][:2, 0]
        fwd = fwd / (jp.linalg.norm(fwd) + 1e-6)
        # step length of foot i = how far it landed ahead of the other foot
        step = jp.array([
            jp.dot(feet_xy[0] - feet_xy[1], fwd),
            jp.dot(feet_xy[1] - feet_xy[0], fwd),
        ])
        err = step - info.get("step_cmd", jp.zeros(()))
        per_foot = jp.exp(-jp.square(err / STEP_SIGMA)) * first_contact
        walking = jp.abs(info["command"][0]) > MIN_WALK_SPEED
        # metric: abs error on touchdown steps (0 when no touchdown this tick)
        n_td = jp.maximum(jp.sum(first_contact), 1.0)
        metrics["step_len_err"] = jp.sum(jp.abs(err) * first_contact) / n_td * walking
        return jp.sum(per_foot) * walking

"""G1 step-length task on stairs, with a height scan in the observation.

Same task, rewards and PPO recipe as g1pipe.steplength_env.StepLength (Playground G1
joystick + step-length command), plus what a policy needs to climb stairs:

  * terrain: g1pipe.stairs_terrain grid of pyramid staircases (3–15 cm steps) and flat cells
  * spawn anywhere on the map, at the right height, facing a random direction
  * observation += 55-point height scan (11 x 5 patch, 0.3 m behind to 1.2 m ahead)
  * feet_phase reward measures swing-foot height above the ground under the foot,
    not above z = 0 (otherwise standing on a step looks like a raised foot)
  * episodes end if the robot walks off the 24 m map

    python -m g1pipe.train --task stairs --timesteps 200_000_000 --out runs/stairs_v1
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import gait
from mujoco_playground._src.locomotion.g1 import base as g1_base

from g1pipe import stairs_terrain as T
from g1pipe.steplength_env import StepLength, default_config as flat_config

SCAN_NOISE = 0.02     # m, uniform noise on the height scan
SPAWN_MARGIN = 3.0    # m from the map edge


def default_config():
    cfg = flat_config()
    cfg.lin_vel_x = [-0.5, 1.0]
    cfg.reward_config.max_foot_height = 0.18
    cfg.reward_config.scales.step_length = 0.5   # stairs set their own step length; keep it a hint
    return cfg


class StairsStepLength(StepLength):

    def __init__(self, config=None, config_overrides=None, terrain_seed: int = 0):
        xml = Path(tempfile.gettempdir()) / f"g1_stairs_{terrain_seed}.xml"
        xml.write_text(T.scene_xml(terrain_seed))
        # StepLength/Joystick pick the XML from a task name; go straight to the G1 base instead.
        g1_base.G1Env.__init__(self, xml_path=xml.as_posix(), config=config or default_config(),
                               config_overrides=config_overrides)
        self._post_init()
        self._hgrid = jp.asarray(T.heights(terrain_seed), dtype=jp.float32)

    # -- terrain helpers -------------------------------------------------------------
    def _yaw(self, data):
        q = data.qpos[3:7]
        return jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def _scan(self, data):
        return T.scan(self._hgrid, data.qpos[:3], self._yaw(data), xp=jp)

    # -- env API ---------------------------------------------------------------------
    def reset(self, rng):
        rng, spawn_rng = jax.random.split(rng)
        state = super().reset(rng)
        lim = T.HALF - SPAWN_MARGIN
        xy = jax.random.uniform(spawn_rng, (2,), minval=-lim, maxval=lim)
        qpos = state.data.qpos
        # highest ground under the footprint, so a spawn on a step edge never puts a foot inside the step
        ring = xy + jp.array([[0, 0], [0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2], [0.2, 0.2], [-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2]])
        ground = jp.max(T.lookup(self._hgrid, ring, xp=jp))
        qpos = qpos.at[0:2].set(xy).at[2].set(ground + qpos[2] + 0.02)
        data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
        contact = self._contact(data)
        obs = self._get_obs(data, state.info, contact)
        return state.replace(data=data, obs=obs)

    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        scan = self._scan(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy = scan + (2 * jax.random.uniform(noise_rng, scan.shape) - 1) * SCAN_NOISE * self._config.noise_config.level
        return {
            "state": jp.hstack([obs["state"], noisy]),
            "privileged_state": jp.hstack([obs["privileged_state"], scan]),
        }

    def _get_termination(self, data):
        off_map = jp.any(jp.abs(data.qpos[:2]) > T.HALF - 0.5)
        return super()._get_termination(data) | off_map

    def _reward_feet_phase(self, data, phase, foot_height, command):
        feet = data.site_xpos[self._feet_site_id]
        foot_z = feet[..., -1] - T.lookup(self._hgrid, feet[..., :2], xp=jp)
        rz = gait.get_rz(phase, swing_height=foot_height)
        reward = jp.exp(-jp.sum(jp.square(foot_z - rz)) / 0.01)
        body_linvel = self.get_global_linvel(data, "pelvis")[:2]
        body_angvel = self.get_global_angvel(data, "pelvis")[2]
        moving = (jp.linalg.norm(body_linvel) > 0.1) | (jp.abs(body_angvel) > 0.1)
        return reward * (moving | (jp.linalg.norm(command) > 0.01))

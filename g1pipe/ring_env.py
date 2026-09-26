"""Pick a bearing ring from a tray and drop it on a fixture: the manipulation part of the SKF use case
(docs/report §11: carry rings from the washing line to the inspection station, pick and put
tray <-> fixture), as a quick reinforcement-learning run through this pipeline.

Stand-in, stated plainly: the Franka arm and parallel gripper of MuJoCo Playground (PandaPickCube)
take the place of the G1's arm and hand. The Playground G1 model has no hands, and the documented
method for this skill (teleoperated demonstrations -> imitation learning, RL fine-tune for grasp
robustness) needs demonstrations we do not have yet. The task, the part, the success measures and the
pipeline around them (train -> batched evaluation with intervals -> video) carry over.

The part: the outer ring of a 6206-class deep-groove ball bearing, 62 mm outside diameter, 48 mm bore
(the raceway side), 16 mm wide, 0.15 kg of steel, modelled as 12 box segments (boxes collide well
in MJX). It is grasped across the outside diameter, which never touches the raceway.

An episode (RING_EPISODE_S): the ring lies flat somewhere in the tray area (TRAY), the fixture is
somewhere in the fixture area (FIXTURE), 25-55 cm away. Stages, each rewarded:
  reach     gripper to the ring
  lift      ring 4+ cm off the table (latched); the placing reward only counts after a lift, so
            dragging the ring along the table does not pay
  place     ring on the fixture: centre within PLACE_TOL of it, resting on the table, upright
  release   placed, and the gripper 6+ cm away from the ring; then back to the home pose
            (v2: once placed, the reach reward stops and a smooth reward grows with the gripper's
            distance from the ring while the ring stays put. v1 learned to pick, carry and place in
            34M steps but never let go: moving away first lost reach reward before the release paid)
Penalties: a finger pad inside the bore (a raceway-contact proxy: the report's "no contact with the
ring raceway"), the hand touching the table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State

RING_OD, RING_ID, RING_W = 0.062, 0.048, 0.016    # m: outside diameter, bore, width
RING_MASS = 0.15                                   # kg
N_SEG = 12
REST_Z = RING_W / 2                                # ring centre height lying flat on the table
TRAY = ((0.45, 0.62), (-0.28, -0.12))              # x, y ranges where rings are picked (m, robot base frame)
FIXTURE = ((0.45, 0.62), (0.12, 0.28))             # x, y ranges of the fixture
CARRY_Z = 0.10                                     # m: carry height before lowering onto the fixture
LIFT_Z = REST_Z + 0.04                             # a lift counts above this
PLACE_TOL = 0.015                                  # m, ring centre to fixture centre
RELEASE_DIST = 0.06                                # m, gripper to ring centre after release
RING_EPISODE_S = 5.0


def ring_xml() -> str:
    r_mid = (RING_OD + RING_ID) / 4
    half_t = (RING_OD - RING_ID) / 4
    half_len = r_mid * np.tan(np.pi / N_SEG) + 0.0015            # a little overlap closes the gaps
    seg_mass = RING_MASS / N_SEG
    segs = []
    for i in range(N_SEG):
        a = 2 * np.pi * i / N_SEG
        segs.append(
            f'<geom name="ring_seg{i}" type="box" size="{half_t:.4f} {half_len:.4f} {RING_W / 2:.4f}" '
            f'pos="{r_mid * np.cos(a):.5f} {r_mid * np.sin(a):.5f} 0" euler="0 0 {a:.5f}" mass="{seg_mass:.4f}" '
            f'condim="3" friction="1 .03 .003" rgba=".72 .74 .78 1" contype="2" conaffinity="1" solref="0.01 1"/>')
    tray = f'{(TRAY[0][0] + TRAY[0][1]) / 2} {(TRAY[1][0] + TRAY[1][1]) / 2} 0.0005'
    tray_size = f'{(TRAY[0][1] - TRAY[0][0]) / 2 + 0.04} {(TRAY[1][1] - TRAY[1][0]) / 2 + 0.04} 0.0005'
    fix_size = f'{(FIXTURE[0][1] - FIXTURE[0][0]) / 2 + 0.04} {(FIXTURE[1][1] - FIXTURE[1][0]) / 2 + 0.04} 0.0005'
    fix = f'{(FIXTURE[0][0] + FIXTURE[0][1]) / 2} {(FIXTURE[1][0] + FIXTURE[1][1]) / 2} 0.0005'
    return f"""<mujoco model="panda bearing ring pick and drop">
  <include file="mjx_scene.xml"/>
  <worldbody>
    <geom name="tray_area" type="box" pos="{tray}" size="{tray_size}" rgba=".35 .55 .85 .25" contype="0" conaffinity="0"/>
    <geom name="fixture_area" type="box" pos="{fix}" size="{fix_size}" rgba=".95 .6 .2 .18" contype="0" conaffinity="0"/>
    <body name="ring" pos="0.53 -0.2 {REST_Z}">
      <freejoint/>
      {chr(10).join('      ' + s for s in segs)}
    </body>
    <body mocap="true" name="mocap_target">
      <geom type="cylinder" size="{RING_OD / 2 + 0.004} 0.001" rgba=".95 .45 .1 .9" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="{RING_ID / 2 - 0.004} 0.0012" rgba=".95 .95 .95 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="left_finger_pad_floor_found" geom1="left_finger_pad" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="right_finger_pad_floor_found" geom1="right_finger_pad" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="hand_capsule_floor_found" geom1="hand_capsule" geom2="floor" reduce="mindist" num="1" data="found"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04 0.53 -0.2 {REST_Z} 1 0 0 0"
      ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
  </keyframe>
</mujoco>
"""


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=int(RING_EPISODE_S / 0.02),
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(scales=config_dict.create(
            gripper_ring=4.0,          # reach
            ring_target=8.0,           # carry to the fixture (after a reach)
            placed=4.0,                # resting on the fixture
            released=8.0,              # placed and let go
            let_go=6.0,                # v2: placed, and the gripper moving away from the ring
            no_floor_collision=0.25,
            robot_target_qpos=0.3,     # home pose: after the release, go back
            raceway=-2.0,              # a finger pad inside the bore
        )),
        impl="warp",
        # a ring lying on the table: 12 segments x up to 4 box-plane contacts x 4 friction-pyramid rows
        # = 192 constraint rows before the gripper touches it (160 overflowed on the first GPU run)
        naconmax=96 * 2048,
        naccdmax=96 * 2048,
        njmax=400,
        quiet_warp=True,     # no per-step overflow / line-search prints from the GPU kernels (stairs lesson 5)
    )


class RingPickDrop(panda.PandaBase):
    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides=None):
        import mujoco
        import tempfile
        config = config or default_config()
        # the scene includes Playground's Panda XMLs by name: they come in as assets
        path = Path(tempfile.gettempdir()) / "g1pipe_bearing_ring.xml"
        path.write_text(ring_xml())
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._xml_path = path.as_posix()
        self._model_assets = panda.get_assets()
        mj_model = mujoco.MjModel.from_xml_string(path.read_text(), assets=self._model_assets)
        mj_model.opt.timestep = self.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
        if self._config.impl == "warp" and self._config.quiet_warp:
            opt = self._mjx_model.opt
            self._mjx_model = self._mjx_model.replace(opt=opt.replace(_impl=opt._impl.replace(warn_overflow=0)))
        self._action_scale = config.action_scale
        self._post_init(obj_name="ring", keyframe="home")
        self._floor_hand_found_sensor = [self._mj_model.sensor(f"{g}_floor_found").id
                                         for g in ("left_finger_pad", "right_finger_pad", "hand_capsule")]
        self._finger_geoms = np.array([self._left_finger_geom, self._right_finger_geom])

    # -- episode -------------------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> State:
        rng, k_ring, k_fix, k_yaw = jax.random.split(rng, 4)
        lo = jp.array([TRAY[0][0], TRAY[1][0]])
        hi = jp.array([TRAY[0][1], TRAY[1][1]])
        ring_xy = jax.random.uniform(k_ring, (2,), minval=lo, maxval=hi)
        fix_xy = jax.random.uniform(k_fix, (2,), minval=jp.array([FIXTURE[0][0], FIXTURE[1][0]]),
                                    maxval=jp.array([FIXTURE[0][1], FIXTURE[1][1]]))
        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        q = jp.array(self._init_q)
        q = q.at[self._obj_qposadr:self._obj_qposadr + 3].set(jp.array([ring_xy[0], ring_xy[1], REST_Z]))
        q = q.at[self._obj_qposadr + 3:self._obj_qposadr + 7].set(jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)]))
        data = mjx_env.make_data(self._mj_model, qpos=q, qvel=jp.zeros(self._mjx_model.nv), ctrl=self._init_ctrl,
                                 impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
                                 naccdmax=self._config.naccdmax, njmax=self._config.njmax)
        target = jp.array([fix_xy[0], fix_xy[1], REST_Z])
        data = data.replace(mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target - jp.array([0, 0, REST_Z])))
        metrics = {"out_of_bounds": jp.zeros(()), "lifted": jp.zeros(()), "success": jp.zeros(()),
                   **{k: jp.zeros(()) for k in self._config.reward_config.scales.keys()}}
        info = {"rng": rng, "target_pos": target, "reached": jp.zeros(()), "lifted": jp.zeros(()),
                "raceway_steps": jp.zeros(())}
        obs = self._get_obs(data, info)
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        ctrl = jp.clip(state.data.ctrl + action * self._action_scale, self._lowers, self._uppers)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        raw, parts = self._get_reward(data, state.info)
        reward = jp.clip(sum(v * self._config.reward_config.scales[k] for k, v in raw.items()), -1e4, 1e4)
        ring = data.xpos[self._obj_body]
        out = jp.any(jp.abs(ring[:2]) > 1.0) | (ring[2] < -0.01)
        done = (out | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(float)
        state.metrics.update(**raw, out_of_bounds=out.astype(float), lifted=state.info["lifted"],
                             success=parts["success"])
        return State(data, self._get_obs(data, state.info), reward, done, state.metrics, state.info)

    def outcome(self, data, info) -> dict[str, jax.Array]:
        """The success measures, for evaluation (see g1pipe.ring_eval)."""
        ring = data.xpos[self._obj_body]
        gripper = data.site_xpos[self._gripper_site]
        upright = data.xmat[self._obj_body].reshape(3, 3)[2, 2] > 0.95
        placed = (jp.linalg.norm(ring[:2] - info["target_pos"][:2]) < PLACE_TOL) & (ring[2] < REST_Z + 0.004) & upright
        released = placed & (jp.linalg.norm(gripper - ring) > RELEASE_DIST)
        return {"placed": placed, "released": released, "lifted": info["lifted"] > 0,
                "xy_err": jp.linalg.norm(ring[:2] - info["target_pos"][:2])}

    def _get_reward(self, data, info):
        ring = data.xpos[self._obj_body]
        gripper = data.site_xpos[self._gripper_site]
        target = info["target_pos"]
        xy_err = jp.linalg.norm(ring[:2] - target[:2])
        # carry above the table until over the fixture, then lower onto it
        way_z = jp.where(xy_err > 0.03, CARRY_Z, REST_Z)
        way = jp.stack([target[0], target[1], way_z])
        d_grip = jp.linalg.norm(ring - gripper)
        info["reached"] = jp.maximum(info["reached"], (d_grip < 0.02).astype(float))
        info["lifted"] = jp.maximum(info["lifted"], (ring[2] > LIFT_Z).astype(float))
        out = self.outcome(data, info)
        # a finger pad inside the bore: horizontally within the bore radius of the ring's axis, at ring height
        pads = data.geom_xpos[self._finger_geoms]
        axis = data.xmat[self._obj_body].reshape(3, 3)[:, 2]
        rel = pads - ring
        along = rel @ axis
        radial = jp.linalg.norm(rel - along[:, None] * axis[None, :], axis=1)
        in_bore = jp.any((radial < RING_ID / 2 - 0.002) & (jp.abs(along) < RING_W / 2 + 0.005))
        info["raceway_steps"] = info["raceway_steps"] + in_bore
        floor = sum(data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._floor_hand_found_sensor) > 0
        home = 1 - jp.tanh(jp.linalg.norm(data.qpos[self._robot_arm_qposadr] - self._init_q[self._robot_arm_qposadr]))
        raw = {
            "gripper_ring": (1 - jp.tanh(5 * d_grip)) * (1 - out["placed"]),
            "let_go": out["placed"] * info["lifted"] * jp.tanh(jp.maximum(d_grip - 0.02, 0.0) / RELEASE_DIST * 2),
            "ring_target": (1 - jp.tanh(5 * jp.linalg.norm(ring - way))) * info["reached"] * info["lifted"],
            "placed": out["placed"].astype(float) * info["lifted"],
            "released": out["released"].astype(float) * info["lifted"],
            "no_floor_collision": 1.0 - floor.astype(float),
            "robot_target_qpos": home * (0.3 + 0.7 * out["released"]),
            "raceway": in_bore.astype(float),
        }
        return raw, {"success": (out["released"] & (info["lifted"] > 0)).astype(float)}

    def _get_obs(self, data, info):
        gripper = data.site_xpos[self._gripper_site]
        ring = data.xpos[self._obj_body]
        return jp.concatenate([
            data.qpos, data.qvel, gripper, data.site_xmat[self._gripper_site].ravel()[3:],
            data.xmat[self._obj_body].ravel()[3:], ring - gripper, info["target_pos"] - ring,
            jp.array([info["lifted"]]), data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        ])

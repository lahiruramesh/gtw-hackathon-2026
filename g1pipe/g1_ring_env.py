"""Unitree G1 humanoid with Dex3 hands: pick a bearing ring from a tray and drop it on a fixture at a
workbench (docs/report §11, "pick and put rings, tray <-> fixture").

The same part, stages, success measures and evaluation as g1pipe.ring_env (the Franka stand-in), on
the humanoid:

  robot     MuJoCo Menagerie unitree_g1/g1_with_hands.xml (Dex3 three-finger hands). The pelvis is
            fixed at standing height and the legs, left arm, waist roll and pitch are held rigid:
            the robot stands at the bench and works with waist yaw, the right arm (7 joints) and the
            right hand (7 joints), 15 position-controlled joints. Whole-body balance while the arms
            work is a separate standing skill in the report's plan (§11.3), not part of this run.
  hand      the right hand's collision meshes are replaced by boxes fitted to them (boxes collide
            reliably and fast in MuJoCo Warp); the visual meshes stay.
  scene     a bench top at BENCH_Z (waist height); the tray area on the robot's right, the fixture
            in front-left, 15-40 cm apart, within the right hand's reach.
  grasp     the grasp point is midway between the thumb tip and the index/middle finger tips.
  held      (v2) thumb and index or middle tip on opposite sides of the ring, grasp point within
            HOLD_R of its centre. Lift, carry reward and success count only while it is held: v1
            reached 96 % "success" by flicking the ring 2 cm up and sliding it onto the fixture,
            its fingers never closer than 37 mm to the ring's centre (radius 31 mm).

    python -m g1pipe.train --task g1ring --timesteps 100_000_000 --out runs/g1-ring-g1-v1
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

from g1pipe.ring_env import LIFT_Z, N_SEG, PLACE_TOL, RELEASE_DIST, RING_ID, RING_MASS, RING_OD, RING_W

G1_XML = mjx_env.MENAGERIE_PATH / "unitree_g1" / "g1_with_hands.xml"
BENCH_Z = 0.76                                  # bench top height (m); the G1's pelvis is at 0.793
REST_Z = BENCH_Z + RING_W / 2
TRAY = ((0.27, 0.36), (-0.27, -0.15))           # x, y ranges (m, robot frame: x forward, y left)
FIXTURE = ((0.27, 0.36), (-0.02, 0.10))
CARRY_H = 0.08                                  # carry height above the bench before lowering
EPISODE_S = 5.0
HOLD_R = 0.030                                  # m, grasp point to ring centre while held
ACTIVE = (["waist_yaw_joint"]
          + [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                            "wrist_roll", "wrist_pitch", "wrist_yaw")]
          + [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1",
                                                 "middle_0", "middle_1")])
READY = {"right_elbow_joint": 0.1}   # right hand forward, grasp point ~10 cm over the bench, tips 6+ cm clear
HAND_BODIES = ["right_wrist_yaw_link", "right_hand_thumb_0_link", "right_hand_thumb_1_link",
               "right_hand_thumb_2_link", "right_hand_index_0_link", "right_hand_index_1_link",
               "right_hand_middle_0_link", "right_hand_middle_1_link"]
TIPS = ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")


def build_model(sim_dt: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.option.timestep = sim_dt
    # a fixed base: the robot stands at the bench
    pelvis = spec.body("pelvis")
    for j in list(pelvis.joints):
        spec.delete(j)
    # hold every joint that is not active rigid (actuators first: they name their joints)
    for a in list(spec.actuators):
        if a.target not in ACTIVE:
            spec.delete(a)
    for j in list(spec.joints):
        if j.name not in ACTIVE:
            spec.delete(j)
    for k in list(spec.keys):   # the model's keyframes are for all 50 coordinates
        spec.delete(k)
    # collisions: only the right hand touches the world, as fitted boxes; nothing else collides
    probe = spec.compile()   # for the hand meshes' bounding boxes
    for g in spec.geoms:
        g.contype, g.conaffinity = 0, 0
    for bid in {probe.body(b).id for b in HAND_BODIES}:
        body = spec.body(probe.body(bid).name)
        for gi in range(probe.ngeom):
            if probe.geom_bodyid[gi] != bid or probe.geom_group[gi] != 3:
                continue
            c, h = probe.geom_aabb[gi][:3], probe.geom_aabb[gi][3:]    # centre, half sizes in the geom frame
            q = probe.geom_quat[gi]
            pos = probe.geom_pos[gi] + _rotate(q, c)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=np.maximum(h, 0.004), pos=pos, quat=q,
                          contype=1, conaffinity=2, condim=3, friction=[1.2, 0.03, 0.003],
                          rgba=[0, 0, 0, 0], group=4, name=f"hand_box_{probe.body(bid).name}_{gi}")
    # the bench, tray and fixture marks, the ring
    wb = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.82, .84, .86], rgb2=[.74, .76, .79], width=300, height=300)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6], texuniform=True)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="floor",
                contype=0, conaffinity=0)
    wb.add_light(pos=[0.3, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_geom(name="bench", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.42, -0.05, BENCH_Z / 2],
                size=[0.22, 0.40, BENCH_Z / 2], rgba=[.45, .47, .5, 1], contype=1, conaffinity=3)
    for name, (xr, yr), rgba in (("tray_area", TRAY, [.35, .55, .85, .45]), ("fixture_area", FIXTURE, [.95, .6, .2, .35])):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[sum(xr) / 2, sum(yr) / 2, BENCH_Z + 0.0005],
                    size=[(xr[1] - xr[0]) / 2 + 0.04, (yr[1] - yr[0]) / 2 + 0.04, 0.0005], rgba=rgba,
                    contype=0, conaffinity=0)
    ring = wb.add_body(name="ring", pos=[0.31, -0.2, REST_Z])
    ring.add_freejoint()
    r_mid, half_t = (RING_OD + RING_ID) / 4, (RING_OD - RING_ID) / 4
    half_len = r_mid * np.tan(np.pi / N_SEG) + 0.0015
    for i in range(N_SEG):
        a = 2 * np.pi * i / N_SEG
        ring.add_geom(name=f"ring_seg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_t, half_len, RING_W / 2],
                      pos=[r_mid * np.cos(a), r_mid * np.sin(a), 0], euler=[0, 0, a], mass=RING_MASS / N_SEG,
                      condim=3, friction=[1, .03, .003], rgba=[.72, .74, .78, 1], contype=2, conaffinity=1)
    fix = wb.add_body(name="mocap_target", mocap=True)
    fix.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[RING_OD / 2 + 0.004, 0.001, 0], rgba=[.95, .45, .1, .9],
                 contype=0, conaffinity=0)
    fix.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[RING_ID / 2 - 0.004, 0.0012, 0], rgba=[.95, .95, .95, 1],
                 contype=0, conaffinity=0)
    m = spec.compile()
    m.opt.timestep = sim_dt
    return m


def _rotate(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=int(EPISODE_S / 0.02),
        action_repeat=1,
        action_scale=0.05,          # rad per control step, around the current target
        reward_config=config_dict.create(scales=config_dict.create(
            gripper_ring=4.0, held=2.0, ring_target=8.0, placed=4.0, released=8.0, let_go=6.0,
            robot_target_qpos=0.3, raceway=-2.0, bench_push=-1.0,
        )),
        impl="warp",
        # the ring on the bench alone takes ~192 constraint rows; 22 hand boxes pressing on the bench or
        # the ring add more: headroom, since an overflow drops contacts (the first arm run: 160 overflowed)
        naconmax=160 * 2048,
        naccdmax=160 * 2048,
        njmax=800,
        quiet_warp=True,
    )


class G1RingPickDrop(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides=None):
        config = config or default_config()
        super().__init__(config, config_overrides)
        self._mj_model = build_model(self.sim_dt)
        self._xml_path = str(G1_XML)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        if self._config.impl == "warp" and self._config.quiet_warp:
            opt = self._mjx_model.opt
            self._mjx_model = self._mjx_model.replace(opt=opt.replace(_impl=opt._impl.replace(warn_overflow=0)))
        m = self._mj_model
        self._act_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ACTIVE])
        self._act_vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ACTIVE])
        self._lowers, self._uppers = m.actuator_ctrlrange.T
        self._ring = m.body("ring").id
        self._ring_q = m.jnt_qposadr[m.body("ring").jntadr[0]]
        self._tips = np.array([m.body(b).id for b in TIPS])
        self._palm = m.body("right_wrist_yaw_link").id
        self._mocap = m.body("mocap_target").mocapid
        q = m.qpos0.copy()
        for j, v in READY.items():
            q[m.jnt_qposadr[m.joint(j).id]] = v
        self._init_q = q
        self._init_ctrl = np.array([q[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)], np.float32)

    # -- helpers ---------------------------------------------------------------------------------
    def grasp_point(self, data):
        tips = data.xpos[self._tips]
        return (tips[0] + (tips[1] + tips[2]) / 2) / 2

    def held(self, data):
        """Thumb and a finger on opposite sides of the ring, the grasp point near its centre."""
        ring = data.xpos[self._ring]
        tips = data.xpos[self._tips] - ring
        axis = data.xmat[self._ring].reshape(3, 3)[:, 2]
        flat = tips - (tips @ axis)[:, None] * axis[None, :]        # in the ring's plane
        opposed = (flat[0] @ flat[1] < 0) | (flat[0] @ flat[2] < 0)
        return opposed & (jp.linalg.norm(self.grasp_point(data) - ring) < HOLD_R)

    def outcome(self, data, info):
        ring = data.xpos[self._ring]
        upright = data.xmat[self._ring].reshape(3, 3)[2, 2] > 0.95
        placed = (jp.linalg.norm(ring[:2] - info["target_pos"][:2]) < PLACE_TOL) & (ring[2] < REST_Z + 0.004) & upright
        released = placed & (jp.linalg.norm(self.grasp_point(data) - ring) > RELEASE_DIST)
        return {"placed": placed, "released": released, "lifted": info["lifted"] > 0,
                "xy_err": jp.linalg.norm(ring[:2] - info["target_pos"][:2])}

    # -- episode -----------------------------------------------------------------------------------
    def reset(self, rng):
        rng, k_ring, k_fix, k_yaw = jax.random.split(rng, 4)
        ring_xy = jax.random.uniform(k_ring, (2,), minval=jp.array([TRAY[0][0], TRAY[1][0]]),
                                     maxval=jp.array([TRAY[0][1], TRAY[1][1]]))
        fix_xy = jax.random.uniform(k_fix, (2,), minval=jp.array([FIXTURE[0][0], FIXTURE[1][0]]),
                                    maxval=jp.array([FIXTURE[0][1], FIXTURE[1][1]]))
        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        q = jp.array(self._init_q)
        q = q.at[self._ring_q:self._ring_q + 3].set(jp.array([ring_xy[0], ring_xy[1], REST_Z]))
        q = q.at[self._ring_q + 3:self._ring_q + 7].set(jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)]))
        data = mjx_env.make_data(self._mj_model, qpos=q, qvel=jp.zeros(self._mjx_model.nv), ctrl=self._init_ctrl,
                                 impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
                                 naccdmax=self._config.naccdmax, njmax=self._config.njmax)
        target = jp.array([fix_xy[0], fix_xy[1], REST_Z])
        data = data.replace(mocap_pos=data.mocap_pos.at[self._mocap, :].set(target - jp.array([0, 0, RING_W / 2])))
        metrics = {"out_of_bounds": jp.zeros(()), "lifted": jp.zeros(()), "success": jp.zeros(()),
                   **{k: jp.zeros(()) for k in self._config.reward_config.scales.keys()}}
        info = {"rng": rng, "target_pos": target, "reached": jp.zeros(()), "lifted": jp.zeros(())}
        return State(data, self._get_obs(data, info), jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        ctrl = jp.clip(state.data.ctrl + action * self._config.action_scale, self._lowers, self._uppers)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        raw, success = self._get_reward(data, state.info)
        reward = jp.clip(sum(v * self._config.reward_config.scales[k] for k, v in raw.items()), -1e4, 1e4)
        ring = data.xpos[self._ring]
        out = (ring[2] < BENCH_Z - 0.05) | jp.any(jp.abs(ring[:2] - jp.array([0.42, -0.05])) > jp.array([0.3, 0.5]))
        done = (out | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(float)
        state.metrics.update(**raw, out_of_bounds=out.astype(float), lifted=state.info["lifted"], success=success)
        return State(data, self._get_obs(data, state.info), reward, done, state.metrics, state.info)

    def _get_reward(self, data, info):
        ring = data.xpos[self._ring]
        grasp = self.grasp_point(data)
        target = info["target_pos"]
        xy_err = jp.linalg.norm(ring[:2] - target[:2])
        way = jp.stack([target[0], target[1], jp.where(xy_err > 0.03, REST_Z + CARRY_H, REST_Z)])
        d_grip = jp.linalg.norm(ring - grasp)
        held = self.held(data)
        info["reached"] = jp.maximum(info["reached"], (d_grip < 0.03).astype(float))
        info["lifted"] = jp.maximum(info["lifted"], (held & (ring[2] > REST_Z + (LIFT_Z - RING_W / 2))).astype(float))
        out = self.outcome(data, info)
        # a fingertip inside the bore (raceway-contact proxy)
        tips = data.xpos[self._tips]
        axis = data.xmat[self._ring].reshape(3, 3)[:, 2]
        rel = tips - ring
        along = rel @ axis
        radial = jp.linalg.norm(rel - along[:, None] * axis[None, :], axis=1)
        in_bore = jp.any((radial < RING_ID / 2 - 0.002) & (jp.abs(along) < RING_W / 2 + 0.005))
        # pressing the hand into the bench
        low = jp.clip(BENCH_Z + 0.005 - jp.min(tips[:, 2]), 0.0, None)
        q = data.qpos[self._act_qadr]
        home = 1 - jp.tanh(jp.linalg.norm(q[:8] - jp.asarray(self._init_q)[self._act_qadr][:8]))
        raw = {
            "gripper_ring": (1 - jp.tanh(5 * d_grip)) * (1 - out["placed"]),
            "held": held.astype(float) * (1 - out["placed"]),
            "ring_target": (1 - jp.tanh(5 * jp.linalg.norm(ring - way))) * held * info["lifted"],
            "placed": out["placed"].astype(float) * info["lifted"],
            "released": out["released"].astype(float) * info["lifted"],
            "let_go": out["placed"] * info["lifted"] * jp.tanh(jp.maximum(d_grip - 0.02, 0.0) / RELEASE_DIST * 2),
            "robot_target_qpos": home * (0.3 + 0.7 * out["released"]),
            "raceway": in_bore.astype(float),
            "bench_push": low * 100,
        }
        return raw, (out["released"] & (info["lifted"] > 0)).astype(float)

    def _get_obs(self, data, info):
        grasp = self.grasp_point(data)
        ring = data.xpos[self._ring]
        return jp.concatenate([
            data.qpos[self._act_qadr], data.qvel[self._act_vadr], grasp, data.xmat[self._palm].ravel()[3:],
            data.xpos[self._tips].ravel() - jp.tile(ring, 3),
            data.xmat[self._ring].ravel()[3:], ring - grasp, info["target_pos"] - ring,
            jp.array([info["lifted"]]), data.ctrl - data.qpos[self._act_qadr],
        ])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

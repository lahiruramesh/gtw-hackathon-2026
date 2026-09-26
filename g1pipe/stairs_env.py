"""G1 step-length task on stairs, with a height scan and a terrain curriculum.

Same task, rewards and PPO recipe as g1pipe.steplength_env.StepLength (Playground G1
joystick + step-length command), plus what a policy needs to climb stairs:

  * terrain: g1pipe.stairs_terrain curriculum grid (row = level, 2 -> 16 cm steps;
    pyramids to walk down, pits to climb out of)
  * observation += 55-point height scan (11 x 5 patch, 0.3 m behind to 1.2 m ahead)
  * feet_phase reward measures swing-foot height above the ground under the foot,
    not above z = 0 (otherwise standing on a step looks like a raised foot)
  * terrain curriculum (legged_gym): each env has a level and spawns in the middle of a
    staircase of that level. At the end of an episode it moves up a level if it walked
    off its staircase (> PROMOTE_DIST from spawn), down if it fell or covered less than
    half the commanded distance. Envs that pass the top level get a random level, so
    easy terrain is not forgotten.
  * walking off the map ends the episode but is not punished as a fall

v3 (stable climbing; v2 stalled at the first riser and crouched or sat on the steps):

  * spawn either in the middle of a staircase facing out, or on its border facing in, always
    square to the steps (+-YAW_JITTER): half the episodes now start with a climb
  * commands mostly forward (STAIRS_VX), little sideways walking or turning; stairs are
    crossed head on, the way a supervisor would command them
  * a fall is also a torso tilted past MAX_TILT_DEG or a pelvis lower than MIN_PELVIS_REL
    above the ground (feet-only collisions let a collapsed robot sink through the steps
    without ever tipping 90 deg, so v2 was never punished for sitting down)
  * feet_phase clearance is measured against the highest ground under the foot or up to
    TOE_LOOKAHEAD ahead of it, so the swing foot lifts before it reaches a riser
  * feet_edge: cost while a loaded foot straddles a step edge (heel and toe on different
    steps); base_height_rel: cost for the pelvis sinking below BASE_HEIGHT_REL above the ground
  * step_length reward off: on stairs the tread sets where the feet land

v4 (v3 turned away from climbs it found hard and walked off along the flat: the velocity reward
is in the body frame and promotion counted distance in any direction):

  * heading command (legged_gym / Isaac Lab): each episode has a target heading square to the
    steps; the yaw-rate command is HEADING_KP x heading error, recomputed every step, so turning
    away costs tracking_ang_vel reward. The evaluator's heading hold does the same.
  * promotion and the "crossed" metric count progress along the target heading only
  * feet_edge only at touchdown (landing a foot across an edge), not every loaded tick: with an
    18 cm foot on 27-30 cm treads the per-tick cost dominated the return and taught lunging
  * step_length reward back at a small weight (0.25) as a stride-length hint

v5 (v4 climbed pyramids but still turned right round at the exit of a pit and walked back:
the tracking_ang_vel reward was too weak a price for skipping the climb):

  * turning more than MAX_HEADING_DEV_DEG off the target heading while commanded to walk ends
    the episode as a fall (termination penalty, curriculum demotion)

v6: the curriculum stops at TRAIN_LEVELS (2-12 cm). With Playground's action_scale 0.5 the joint
targets stay within +-0.5 rad of the default pose, which lifts the foot at most ~22 cm with every
joint at its limit, so 14-16 cm steps are near the edge of what the leg can reach; training time
goes to making the reachable range reliable. The held-out test still runs to 15 cm.
(v6 was no better than v5, so v7 trains on all levels again.)

v7: training stairs mix tread depths per staircase (stairs_terrain "train": 25/29/33 cm). v5
certified 8 cm on its own 30 cm-tread stairs but 0-4.7 cm on the held-out 27 cm treads.

v8 (falls on 8+ cm steps are the pelvis sinking while stepping up, with the knee and hip-pitch
targets pinned at the +-0.5 rad action limit: a PD target at most 0.5 rad from the default pose
caps the knee's extension torque near 75 Nm at step-up angles, about half of what the G1 knee
motor gives): leg_action_scale (config) widens the target range of hip pitch, knee and ankle
pitch only; motor torque limits are unchanged. g1pipe.train rescales the warm-start policy's
outputs for those joints so it starts from the same gait.

v11 (v10 on the held-out stairs, by commanded cadence f = vx / (2 step): 1.0 Hz long strides
14-15/32 crossed at any speed, 1.4 Hz short steps 30-31/32 with 13-15 cm crossed): on stairs the
gait is sampled from STAIRS_GAIT_FREQ (config gait_freq_range) instead of the flat task's 0.9-1.8 Hz,
and the operator commands STAIRS_CADENCE (step length = vx / (2 * 1.4), about one step per tread).

v14 (the first certification suite: every policy since v1 falls with a 20 ms actuation delay, and
half the crossings fall at friction 0.3): each episode draws an actuation delay of 0 or 1 control step
(config action_delay_max), and the floor friction is sampled from FRICTION_RANGE (0.25-1.0).

From v15 on, changes are experiment arms of scripts/improve.py (a control arm with the champion's
recipe plus one change per arm), not new defaults: the defaults below reproduce v14. The knobs arms
change are config fields: action_delay_p (share of episodes with the delay), friction_range,
payload_kg (extra torso mass on top of Playground's +-1 kg), top_replay_p, and any other config value
through g1pipe.train --set.

v12 (v11: 2 falls in 72 strict runs up to 11.6 cm, but 1 in 3 at 15 cm): TOP_REPLAY_P of the
robots that beat the top level replay the tallest TOP_REPLAY_LEVELS (12-16 cm) instead of a random one.

scan_model = "camera" (config): the policy's height scan gets the errors of the simulated head
depth camera + elevation map (jev_agent.vision), measured in plain MuJoCo on the held-out stairs
(results/stairs/camera_scan_v2.json): ~1 cm noise, rare 5 cm outliers at step edges, points near
and behind the feet not yet seen (read as level with the feet), plus a per-episode map offset for
odometry drift, which the simulated camera (perfect pose) cannot show. The critic keeps the true scan.

The curriculum lives in state.info["curriculum"]. Playground's auto-reset wrapper
(full_reset=False) restores data and obs on done but keeps info, and every episode
starts from the cached reset data at time 0, so step() re-places the robot on the first
step of each episode. Train with num_resets_per_eval = 0, or host-side resets would wipe
the levels (g1pipe.train does this).

    python -m g1pipe.train --task stairs --init-from runs/g1-stairs-v2/run/params.pkl \
        --timesteps 200_000_000 --out runs/stairs_v3
"""
from __future__ import annotations

import tempfile
from unittest import mock
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import gait
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize

from g1pipe import stairs_terrain as T
from g1pipe.steplength_env import StepLength, default_config as flat_config

SCAN_NOISE = 0.02        # m, uniform noise on the height scan
N_LEVELS = T.GRID
TRAIN_LEVELS = N_LEVELS  # all 8 levels (v6 capped at 6 = 2-12 cm; no better, so v7 reverts it)
MAX_INIT_LEVEL = 2       # training envs start on 2-6 cm steps
SPAWN_JITTER = 0.3       # m around the cell centre (platform half-width is 0.5 m)
TOP_REPLAY_P, TOP_REPLAY_LEVELS = 0.7, 3   # v12: where top-level graduates go (see _next_episode)
PROMOTE_DIST = T.CELL / 2 - 0.1   # m from spawn: past the last step, on the flat walkway
DEMOTE_FRAC = 0.5        # fraction of the commanded distance an episode must cover
EDGE_SPAWN_P = 0.5       # share of episodes that start on a cell border, facing the stairs
YAW_JITTER = 0.3         # rad around square to the steps
STAIRS_VX = [0.4, 0.9]   # m/s forward command (v2: momentum helped; 0.3 m/s stalled)
STAIRS_GAIT_FREQ = (1.25, 1.6)   # Hz, sampled in training (v11+)
STAIRS_CADENCE = 1.4             # Hz, commanded on stairs: step length = vx / (2 * STAIRS_CADENCE)
HEADING_KP = 1.5         # rad/s of yaw-rate command per rad of heading error (as g1pipe.evaluate)
HEADING_WZ_MAX = 0.8     # rad/s
MAX_HEADING_DEV_DEG = 60.0   # off the target heading = turned away from the stairs = fall (v5)
MAX_TILT_DEG = 60.0      # torso tilt that counts as a fall
MIN_PELVIS_REL = 0.45    # m, pelvis above the ground under it that counts as a fall (stand ~0.75)
BASE_HEIGHT_REL = 0.70   # m, pelvis height below which base_height_rel starts to cost
TOE_LOOKAHEAD = (0.08, 0.16)   # m ahead of the foot site checked for a riser (toe is at +0.13)
HEEL, TOE = -0.05, 0.13  # m, foot box extent along the foot from the foot site
SOLE = 0.037             # m, foot box bottom below the foot site

# camera scan model (scan_model="camera"), fitted to results/stairs/camera_scan_v2.json
CAM_SIGMA = 0.01                  # m, noise on seen points (measured std 0.9-1.3 cm)
CAM_OUTLIER_P, CAM_OUTLIER_SIGMA = 0.01, 0.05   # edge outliers (measured p99 5.2 cm)
CAM_BLIND_P = {-0.3: 0.21, -0.15: 0.18, 0.0: 0.15, 0.15: 0.11, 0.3: 0.05, 0.45: 0.01}   # unseen share by row
CAM_DRIFT_XY, CAM_DRIFT_Z = 0.03, 0.02          # m, per-episode map offset (odometry drift)


def default_config():
    cfg = flat_config()
    cfg.lin_vel_x = STAIRS_VX
    cfg.lin_vel_y = [-0.15, 0.15]
    cfg.ang_vel_yaw = [-0.4, 0.4]
    cfg.reward_config.max_foot_height = 0.18
    cfg.reward_config.scales.step_length = 0.25  # hint only: on stairs the tread sets where the feet land
    cfg.reward_config.scales.feet_edge = -1.0
    cfg.reward_config.scales.base_height_rel = -20.0
    cfg.scan_model = "uniform"   # or "camera"
    cfg.gait_freq_range = list(STAIRS_GAIT_FREQ)
    cfg.action_delay_max = 1     # v14: actuation latency, control steps (20 ms each), sampled per episode
    cfg.action_delay_p = 0.5     # share of episodes with a delay (v14: uniform over 0..max, i.e. 0.5)
    cfg.friction_range = list(FRICTION_RANGE)   # floor friction, sampled per env (domain_randomize)
    cfg.payload_kg = list(PAYLOAD_KG)           # extra torso mass, sampled per env (domain_randomize)
    cfg.top_replay_p = TOP_REPLAY_P             # v12: share of top-level graduates sent to the tallest rows
    cfg.spawn_mode = "train"                    # "strict": the strict test's start (g1pipe.bench), see _place
    cfg.leg_action_scale = 0.0   # >0: action scale of LEG_JOINTS (v8); 0 keeps Playground's 0.5 everywhere
    # contact buffers (Playground: 8 contacts per robot): a foot on the stairs now has up to 7
    cfg.naconmax = 20 * 8192
    cfg.njmax = 29 * 2 + 20 * 4
    return cfg


FRICTION_RANGE = (0.25, 1.0)   # v14: floor friction; Playground samples 0.4-1.0
PAYLOAD_KG = (0.0, 0.0)        # extra torso mass on top of Playground's +-1 kg (carrying, a battery pack)


def floor_pairs(model):
    """Indices of the foot-floor contact pairs. The compiler orders pairs with the floor pairs
    first, so pair 0 is one of them."""
    return np.flatnonzero(np.asarray(model.pair_geom1) == int(np.asarray(model.pair_geom1)[0]))


def domain_randomize(model, rng, friction_range=FRICTION_RANGE, payload_kg=PAYLOAD_KG):
    """Playground's G1 randomisation (masses, joint friction, armature, qpos0), with the floor
    friction drawn from friction_range for every foot-floor pair (Playground sets pairs 0-1 only, its
    two sole boxes) and payload_kg of extra torso mass. g1pipe.train binds both from the env config."""
    floor = floor_pairs(model)
    model, in_axes = g1_randomize.domain_randomize(model, rng)
    mu = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 17), (), minval=friction_range[0],
                                               maxval=friction_range[1]))(rng)
    pf = model.pair_friction.at[:, floor, 0:2].set(mu[:, None, None])
    load = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 29), (), minval=payload_kg[0],
                                                 maxval=payload_kg[1]))(rng)
    mass = model.body_mass.at[:, g1_randomize.TORSO_BODY_ID].add(load)
    model = model.tree_replace({"pair_friction": pf, "body_mass": mass})
    return model, in_axes


LEG_JOINTS = ("hip_pitch", "knee", "ankle_pitch")


def action_scales(mj_model, config):
    """Per-actuator action scale: config.leg_action_scale on LEG_JOINTS, config.action_scale elsewhere."""
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mj_model.nu)]
    leg = config.leg_action_scale or config.action_scale   # 0: not widened
    return np.array([leg if any(j in n for j in LEG_JOINTS) else config.action_scale for n in names], np.float32)


class StairsStepLength(StepLength):

    def __init__(self, config=None, config_overrides=None, layout: str = "train", eval_levels: bool = False):
        """eval_levels: start every episode on a uniformly random level (for the PPO eval env),
        instead of the curriculum's easy start."""
        xml = Path(tempfile.gettempdir()) / f"g1_stairs_{layout}.xml"
        xml.write_text(T.scene_xml(layout))
        # StepLength/Joystick pick the XML from a task name; go straight to the G1 base instead.
        assets = g1_base.get_assets()
        assets[T.FEET_XML] = T.feet_xml(assets["g1_mjx_feetonly.xml"].decode()).encode()
        with mock.patch.object(g1_base, "get_assets", lambda: assets):   # the G1 model with sole spheres
            g1_base.G1Env.__init__(self, xml_path=xml.as_posix(), config=config or default_config(),
                                   config_overrides=config_overrides)
        self._post_init()
        if self._config.impl == "warp":
            # Stair contacts often use all 3 of the model's solver iterations, and MuJoCo Warp then
            # printf's a note from the GPU kernel on every step: 14 MB/s of log, a 3x slower run
            opt = self._mjx_model.opt
            self._mjx_model = self._mjx_model.replace(opt=opt.replace(_impl=opt._impl.replace(warn_overflow=0)))
        if self._config.leg_action_scale > 0:   # Joystick.step reads config.action_scale; make it per joint
            scales = action_scales(self.mj_model, self._config)
            self._config = self._config.copy_and_resolve_references()
            self._config._fields["action_scale"] = scales
        self._hgrid = jp.asarray(T.heights(layout), dtype=jp.float32)
        self._eval_levels = eval_levels

    def _delay(self, rng):
        """Actuation latency for an episode: 0..action_delay_max control steps."""
        k1, k2 = jax.random.split(jax.random.fold_in(rng, 3))
        delayed = jax.random.bernoulli(k1, self._config.action_delay_p)
        return jp.where(delayed, jax.random.randint(k2, (), 1, self._config.action_delay_max + 1), 0)

    def _sample_gait(self, rng, command):
        lo, hi = self._config.gait_freq_range
        f = jax.random.uniform(rng, (), minval=lo, maxval=hi)
        return f, command[0] / (2.0 * f)

    # -- terrain helpers -------------------------------------------------------------
    def _yaw(self, data):
        q = data.qpos[3:7]
        return jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def _scan(self, data):
        return T.scan(self._hgrid, data.qpos[:3], self._yaw(data), xp=jp)

    def _place(self, data, rng, level):
        """Robot on a random staircase of `level`, square to its steps: in the middle facing out
        (walks down a pyramid, up out of a pit) or on the border facing in (up a pyramid, down
        into a pit). Playground's joint and velocity perturbations. Returns (data, spawn xy)."""
        k_ix, k_xy, k_yaw, k_q, k_v, k_dir, k_edge = jax.random.split(rng, 7)
        ix = jax.random.randint(k_ix, (), 0, T.GRID)
        ang = jax.random.randint(k_dir, (), 0, 4) * (jp.pi / 2)
        out = jp.array([jp.cos(ang), jp.sin(ang)])
        # spawn_mode "strict" (g1pipe.bench): as the plain-MuJoCo strict test, always on the border,
        # square to the steps, at the border's middle; only the sideways position varies
        strict = self._config.spawn_mode == "strict"
        edge = jax.random.bernoulli(k_edge, EDGE_SPAWN_P) | strict
        jitter = jax.random.uniform(k_xy, (2,), minval=-SPAWN_JITTER, maxval=SPAWN_JITTER) * jp.array([1.0 - strict, 1.0])
        # on the border strip: along the axis stay on the strip, across it anywhere near the middle
        along = jp.where(edge, T.CELL / 2 - T.BORDER / 2 + jitter[0] / 3, jitter[0])
        across = jitter[1] * jp.where(edge, 1.5, 1.0)
        xy = T.cell_center(ix, level, xp=jp) + along * out + across * jp.array([-out[1], out[0]])
        # highest ground under the footprint, so a spawn near an edge never puts a foot inside a step
        ring = xy + jp.array([[0, 0], [0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2], [0.2, 0.2], [-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2]])
        ground = jp.max(T.lookup(self._hgrid, ring, xp=jp))
        yaw = ang + jp.where(edge, jp.pi, 0.0) + jax.random.uniform(k_yaw, (), minval=-YAW_JITTER, maxval=YAW_JITTER) * (1.0 - strict)
        qpos = (self._init_q
                .at[0:2].set(xy).at[2].set(ground + self._init_q[2] + 0.02)
                .at[3:7].set(jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)]))
                .at[7:].multiply(jax.random.uniform(k_q, (self._init_q.shape[0] - 7,), minval=0.5, maxval=1.5)))
        qvel = jp.zeros_like(data.qvel).at[0:6].set(jax.random.uniform(k_v, (6,), minval=-0.5, maxval=0.5))
        return data.replace(qpos=qpos, qvel=qvel), xy, ang + jp.where(edge, jp.pi, 0.0)

    # -- env API ---------------------------------------------------------------------
    def reset(self, rng):
        rng, lvl_rng, place_rng, drift_rng = jax.random.split(rng, 4)
        state = super().reset(rng)
        top = TRAIN_LEVELS if self._eval_levels else MAX_INIT_LEVEL + 1
        level = jax.random.randint(lvl_rng, (), 0, top)
        data, xy, heading = self._place(state.data, place_rng, level)
        data = mjx.forward(self.mjx_model, data)
        info = dict(state.info)
        info["curriculum"] = {"level": level, "origin": xy, "heading": heading, "dist": jp.zeros(()), "cmd_dist": jp.zeros(()),
                              "scan_offset": self._scan_offset(drift_rng), "delay": self._delay(drift_rng),
                              "fell": jp.zeros((), bool), "crossed": jp.zeros((), bool)}
        metrics = dict(state.metrics, crossed=jp.zeros(()), turned=jp.zeros(()), terrain_level=level.astype(jp.float32))
        obs = self._get_obs(data, info, self._contact(data))
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    def _next_episode(self, state):
        """First step of an episode: update the level from the last episode, then re-place."""
        cur = state.info["curriculum"]
        up = cur["dist"] > PROMOTE_DIST
        down = ~up & (cur["fell"] | (cur["dist"] < DEMOTE_FRAC * cur["cmd_dist"]))
        level = cur["level"] + up.astype(int) - down.astype(int)
        info = dict(state.info)
        info["rng"], lvl_rng, top_rng, place_rng, drift_rng = jax.random.split(info["rng"], 5)
        # legged_gym sends a robot that beats the top level to a random level; since v12 most go
        # back to the tallest TOP_REPLAY_LEVELS, where the falls are, instead
        lo = jp.where(jax.random.bernoulli(top_rng, self._config.top_replay_p), TRAIN_LEVELS - TOP_REPLAY_LEVELS, 0)
        level = jp.where(level >= TRAIN_LEVELS, jax.random.randint(lvl_rng, (), lo, TRAIN_LEVELS), jp.maximum(level, 0))
        data, xy, heading = self._place(state.data, place_rng, level)
        info["curriculum"] = {"level": level, "origin": xy, "heading": heading, "dist": jp.zeros(()), "cmd_dist": jp.zeros(()),
                              "scan_offset": self._scan_offset(drift_rng), "delay": self._delay(drift_rng),
                              "fell": jp.zeros((), bool), "crossed": jp.zeros((), bool)}
        return state.replace(data=data, info=info)

    def step(self, state, action):
        new_episode = state.data.time == 0.0
        placed = self._next_episode(state)
        pick = lambda a, b: jp.where(new_episode, a, b)
        # Only the keys _next_episode changes: info also holds the auto-reset wrapper's cached
        # Data, whose MuJoCo Warp contact buffers are shared by the whole batch and must not be
        # selected per env.
        info = dict(state.info, rng=pick(placed.info["rng"], state.info["rng"]),
                    curriculum=jax.tree.map(pick, placed.info["curriculum"], state.info["curriculum"]))
        state = state.replace(
            data=state.data.replace(qpos=pick(placed.data.qpos, state.data.qpos), qvel=pick(placed.data.qvel, state.data.qvel)),
            info=info)
        # heading command: yaw rate from the heading error (zero command stays zero)
        cur = state.info["curriculum"]
        d = cur["heading"] - self._yaw(state.data)
        err = jp.arctan2(jp.sin(d), jp.cos(d))
        cmd = state.info["command"]
        wz = jp.where(jp.linalg.norm(cmd[:2]) > 0.01, jp.clip(HEADING_KP * err, -HEADING_WZ_MAX, HEADING_WZ_MAX), 0.0)
        state = state.replace(info=dict(state.info, command=cmd.at[2].set(wz)))
        # actuation latency: with a delay the motors get the previous control step's action; the
        # policy still sees its own actions in the observation (last_act), as on the robot
        prev = state.info["last_act"]
        state = super().step(state, jp.where(cur["delay"] > 0, prev, action))
        state = state.replace(info=dict(state.info, last_act=action, last_last_act=prev))
        # turned away from the stairs while commanded to walk: end the episode as a fall
        d = cur["heading"] - self._yaw(state.data)
        turned = ((jp.abs(jp.arctan2(jp.sin(d), jp.cos(d))) > jp.deg2rad(MAX_HEADING_DEV_DEG))
                  & (jp.linalg.norm(cmd[:2]) > 0.01))
        penalty = turned & (state.done == 0)
        state = state.replace(
            reward=state.reward + penalty * self._config.reward_config.scales.termination * self.dt,
            done=jp.maximum(state.done, turned.astype(state.done.dtype)))
        info = dict(state.info)
        info["step"] = jp.where(turned, 0, info["step"])
        cur = dict(info["curriculum"])
        fwd = jp.array([jp.cos(cur["heading"]), jp.sin(cur["heading"])])
        cur["dist"] = jp.dot(state.data.qpos[:2] - cur["origin"], fwd)   # progress along the target heading
        cur["cmd_dist"] = cur["cmd_dist"] + jp.linalg.norm(info["command"][:2]) * self.dt
        cur["fell"] = self._fell(state.data) | turned
        crossed_now = (cur["dist"] > PROMOTE_DIST) & ~cur["crossed"]
        cur["crossed"] = cur["crossed"] | crossed_now
        info["curriculum"] = cur
        metrics = dict(state.metrics, crossed=crossed_now.astype(jp.float32), turned=turned.astype(jp.float32),
                       terrain_level=cur["level"].astype(jp.float32))
        return state.replace(info=info, metrics=metrics)

    def _scan_offset(self, rng):
        """Per-episode elevation-map offset (dx, dy, dz): odometry drift, camera model only."""
        lim = jp.array([CAM_DRIFT_XY, CAM_DRIFT_XY, CAM_DRIFT_Z]) * self._config.noise_config.level
        return jax.random.uniform(rng, (3,), minval=-lim, maxval=lim) * (self._config.scan_model == "camera")

    def _camera_scan(self, data, info, rng):
        """Height scan with the head depth camera's errors (see module docstring)."""
        k_n, k_o, k_os, k_b = jax.random.split(rng, 4)
        level = self._config.noise_config.level
        off = info["curriculum"]["scan_offset"] if "curriculum" in info else jp.zeros(3)
        base = data.qpos[:3] + jp.array([off[0], off[1], 0.0])
        s = T.scan(self._hgrid, base, self._yaw(data), xp=jp) + off[2]
        s = s + jax.random.normal(k_n, s.shape) * CAM_SIGMA * level
        outlier = jax.random.bernoulli(k_o, CAM_OUTLIER_P * level, s.shape)
        s = s + outlier * jax.random.normal(k_os, s.shape) * CAM_OUTLIER_SIGMA
        blind_p = jp.asarray([CAM_BLIND_P.get(round(float(x), 2), 0.0) for x, _ in T.SCAN_PTS])
        blind = jax.random.bernoulli(k_b, blind_p * level)
        feet = jp.min(data.site_xpos[self._feet_site_id][:, 2]) - SOLE - (data.qpos[2] - T.NOMINAL_HEIGHT)
        return jp.clip(jp.where(blind, feet, s), -1.0, 1.0)

    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        scan = self._scan(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        if self._config.scan_model == "camera":
            noisy = self._camera_scan(data, info, noise_rng)
        else:
            noisy = scan + (2 * jax.random.uniform(noise_rng, scan.shape) - 1) * SCAN_NOISE * self._config.noise_config.level
        return {
            "state": jp.hstack([obs["state"], noisy]),
            "privileged_state": jp.hstack([obs["privileged_state"], scan]),
        }

    def _fell(self, data):
        up = self.get_gravity(data, "torso")[-1]   # torso z axis, world z component (1 = upright)
        pelvis_rel = data.qpos[2] - T.lookup(self._hgrid, data.qpos[:2], xp=jp)
        return (super()._get_termination(data) | (up < jp.cos(jp.deg2rad(MAX_TILT_DEG)))
                | (pelvis_rel < MIN_PELVIS_REL))

    def _get_termination(self, data):
        off_map = jp.any(jp.abs(data.qpos[:2]) > T.HALF - 0.5)
        return self._fell(data) | off_map

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        # the termination penalty is for falling, not for reaching the edge of the map
        rewards = super()._get_reward(data, action, info, metrics, self._fell(data), first_contact, contact)
        rewards["feet_edge"] = self._cost_feet_edge(data, first_contact)
        pelvis_rel = data.qpos[2] - T.lookup(self._hgrid, data.qpos[:2], xp=jp)
        rewards["base_height_rel"] = jp.square(jp.clip(BASE_HEIGHT_REL - pelvis_rel, 0.0, None))
        return rewards

    def _foot_axes(self, data):
        """Foot positions (2, 3) and unit forward directions in the ground plane (2, 2)."""
        feet = data.site_xpos[self._feet_site_id]
        fwd = data.site_xmat[self._feet_site_id][:, :2, 0]
        return feet, fwd / (jp.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-6)

    def _cost_feet_edge(self, data, contact):
        """Feet landing with heel and toe on different steps (foot across an edge); contact = touchdowns."""
        feet, fwd = self._foot_axes(data)
        heel = T.lookup(self._hgrid, feet[:, :2] + HEEL * fwd, xp=jp)
        toe = T.lookup(self._hgrid, feet[:, :2] + TOE * fwd, xp=jp)
        return jp.sum((jp.abs(toe - heel) > 0.02) * contact)

    def _reward_feet_phase(self, data, phase, foot_height, command):
        feet, fwd = self._foot_axes(data)
        ground = T.lookup(self._hgrid, feet[:, :2], xp=jp)
        for d in TOE_LOOKAHEAD:   # a riser just ahead raises the ground the swing foot must clear
            ground = jp.maximum(ground, T.lookup(self._hgrid, feet[:, :2] + d * fwd, xp=jp))
        foot_z = feet[..., -1] - ground
        rz = gait.get_rz(phase, swing_height=foot_height)
        reward = jp.exp(-jp.sum(jp.square(foot_z - rz)) / 0.01)
        body_linvel = self.get_global_linvel(data, "pelvis")[:2]
        body_angvel = self.get_global_angvel(data, "pelvis")[2]
        moving = (jp.linalg.norm(body_linvel) > 0.1) | (jp.abs(body_angvel) > 0.1)
        return reward * (moving | (jp.linalg.norm(command) > 0.01))

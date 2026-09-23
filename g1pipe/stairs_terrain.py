"""Stairs terrain for training and evaluating a stair-climbing G1 policy.

A grid of CELL x CELL m cells, each either flat or a "pyramid" staircase (steps up
from every side to a central platform, legged_gym style), with step heights spread
from easy to hard. One heightfield named "floor" holds it all, so the G1 model's
foot-floor contact pairs and sensors work unchanged, in MJX/Warp and in plain MuJoCo.

The same grid gives the height scan: a patch of points around the robot, ground
height looked up bilinearly (numpy or jax), relative to the robot's nominal foot level.
On hardware this would come from the head depth camera / LiDAR elevation map.
"""
from __future__ import annotations

import numpy as np

CELL = 4.0            # m
GRID = 6              # cells per side -> 24 m square
RES = 0.05            # m per heightfield sample
TREAD = 0.30          # m
PLATFORM = 1.0        # m, flat top of each pyramid
BORDER = 0.35         # m, flat strip around each cell (spawn area, walkways)
RISES = [0.0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]   # m, cycled over the cells

# height scan pattern, robot yaw frame: 11 x 5 points from 0.3 m behind to 1.2 m ahead
SCAN_X = np.linspace(-0.3, 1.2, 11)
SCAN_Y = np.linspace(-0.3, 0.3, 5)
SCAN_PTS = np.stack(np.meshgrid(SCAN_X, SCAN_Y, indexing="ij"), -1).reshape(-1, 2)
N_SCAN = len(SCAN_PTS)
NOMINAL_HEIGHT = 0.755   # pelvis height above the feet when standing (knees_bent keyframe)

HALF = GRID * CELL / 2
N = int(round(GRID * CELL / RES)) + 1


def cell_rises(seed: int = 0) -> np.ndarray:
    """Step height of each cell (GRID x GRID), a shuffled mix of RISES."""
    rng = np.random.default_rng(seed)
    r = np.resize(np.array(RISES), GRID * GRID)
    rng.shuffle(r)
    return r.reshape(GRID, GRID)


def heights(seed: int = 0) -> np.ndarray:
    """Ground height grid, shape (N, N), indexed [iy, ix]; x, y from -HALF to +HALF."""
    xs = np.linspace(-HALF, HALF, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")          # [iy, ix]
    rises = cell_rises(seed)
    cx = np.clip(((X + HALF) // CELL).astype(int), 0, GRID - 1)
    cy = np.clip(((Y + HALF) // CELL).astype(int), 0, GRID - 1)
    lx, ly = X + HALF - cx * CELL, Y + HALF - cy * CELL  # position within the cell
    d_edge = np.minimum(np.minimum(lx, CELL - lx), np.minimum(ly, CELL - ly)) - BORDER
    n_max = int((CELL / 2 - BORDER - PLATFORM / 2) // TREAD)
    steps = np.clip(np.floor(d_edge / TREAD) + 1, 0, n_max) * (d_edge >= 0)
    return steps * rises[cy, cx]


def scene_xml(seed: int = 0) -> str:
    """MJCF scene: Playground's feet-only G1 + sensors on the stairs heightfield."""
    h = heights(seed)
    zmax = max(float(h.max()), 1e-3)
    # MJCF inline elevation lists rows from +y down to -y; our grid is indexed from -y up
    elev = " ".join(f"{v:.4f}" for v in (h[::-1] / zmax).ravel())
    return f"""<mujoco model="g1 stairs">
  <include file="g1_mjx_feetonly.xml"/>
  <statistic center="0 0 0.7" extent="1.2" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1=".85 .82 .76" rgb2=".78 .75 .7"
      markrgb=".3 .3 .3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="40 40" reflectance="0"/>
    <hfield name="stairs" nrow="{N}" ncol="{N}" size="{HALF} {HALF} {zmax:.4f} 0.1" elevation="{elev}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="stairs" material="groundplane"/>
  </worldbody>
  <include file="sensor.xml"/>
  <keyframe>
    <key name="knees_bent"
      qpos="0 0 0.755 1 0 0 0
      -0.312 0 0 0.669 -0.363 0 -0.312 0 0 0.669 -0.363 0 0 0 0.073
      0.2 0.2 0 0.6 0 0 0 0.2 -0.2 0 0.6 0 0 0"
      ctrl="-0.312 0 0 0.669 -0.363 0 -0.312 0 0 0.669 -0.363 0 0 0 0.073
      0.2 0.2 0 0.6 0 0 0 0.2 -0.2 0 0.6 0 0 0"/>
  </keyframe>
</mujoco>
"""


def lookup(grid, xy, xp=np):
    """Bilinear ground height at points xy (..., 2). Works with numpy or jax.numpy as xp."""
    f = (xy + HALF) / RES
    f = xp.clip(f, 0.0, N - 1.001)
    i0 = xp.floor(f).astype(int)
    w = f - i0
    ix, iy = i0[..., 0], i0[..., 1]
    wx, wy = w[..., 0], w[..., 1]
    h00, h01 = grid[iy, ix], grid[iy, ix + 1]
    h10, h11 = grid[iy + 1, ix], grid[iy + 1, ix + 1]
    return (h00 * (1 - wx) + h01 * wx) * (1 - wy) + (h10 * (1 - wx) + h11 * wx) * wy


def scan(grid, base_xyz, yaw, xp=np):
    """Height scan: ground height at SCAN_PTS (robot yaw frame) relative to nominal foot level."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    pts = xp.asarray(SCAN_PTS)
    world = base_xyz[:2] + xp.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], -1)
    return xp.clip(lookup(grid, world, xp) - (base_xyz[2] - NOMINAL_HEIGHT), -1.0, 1.0)

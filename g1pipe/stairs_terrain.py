"""Stairs terrain for training and evaluating a stair-climbing G1 policy.

A GRID x GRID grid of CELL x CELL m staircases (legged_gym-style terrain curriculum):

  * row iy = difficulty level: every staircase in a row has the same step height
  * columns alternate between a pyramid (steps up to a central platform) and a pit
    (steps down to a central floor), so a robot spawned in the middle of a cell walks
    down stairs (pyramid) or up stairs (pit) whichever way it heads
  * a flat MARGIN around the grid; flat ground is at z = 0, pits go below it

Two layouts: "train" (the curriculum) and "test" (held out: step heights between the
training levels and a different tread depth), so evaluation never replays training stairs.

Since v7 the training layout mixes tread depths per staircase (25, 29, 33 cm): with one tread
(30 cm, now layout "train30") the v5 policy certified 8 cm on its training stairs but 0-4.7 cm on
the held-out 27 cm treads, i.e. it had fitted its stride to one staircase geometry.

One heightfield named "floor" holds it all, so the G1 model's foot-floor contact pairs and
sensors work unchanged, in MJX/Warp and in plain MuJoCo. But MuJoCo Warp makes at most ONE
contact between a box and a heightfield, where plain MuJoCo makes 3-20 per foot: through v8 the
policies trained balancing on one point per foot and were evaluated standing flat-footed.
Since v9 each sole also carries FOOT_SPHERES (heel, middle, toe on both sides), one contact
each in both engines (feet_xml); the box stays for Playground's foot-contact sensors. The same grid gives the height scan:
a patch of points around the robot, ground height looked up bilinearly (numpy or jax),
relative to the robot's nominal foot level. On hardware this comes from the head depth
camera / LiDAR elevation map (see jev_agent/vision_check.py).
"""
from __future__ import annotations

import numpy as np

CELL = 4.0            # m
GRID = 8              # cells per side; also the number of curriculum levels
MARGIN = 2.0          # m of flat ground around the grid
RES = 0.05            # m per heightfield sample
PLATFORM = 1.0        # m, flat centre of each pyramid / pit
BORDER = 0.35         # m, flat strip around each cell (walkway between staircases)

LAYOUTS = {
    "train": {"rises": np.linspace(0.02, 0.16, GRID), "tread": (0.25, 0.29, 0.33)},   # 2, 4, ... 16 cm; tread per cell
    "train30": {"rises": np.linspace(0.02, 0.16, GRID), "tread": 0.30},   # v1-v6 training layout
    "test": {"rises": np.linspace(0.03, 0.15, GRID), "tread": 0.27},    # 3.0, 4.7, ... 15 cm
}

# height scan pattern, robot yaw frame: 11 x 5 points from 0.3 m behind to 1.2 m ahead
SCAN_X = np.linspace(-0.3, 1.2, 11)
SCAN_Y = np.linspace(-0.3, 0.3, 5)
SCAN_PTS = np.stack(np.meshgrid(SCAN_X, SCAN_Y, indexing="ij"), -1).reshape(-1, 2)
N_SCAN = len(SCAN_PTS)
NOMINAL_HEIGHT = 0.755   # pelvis height above the feet when standing (knees_bent keyframe)

GRID_HALF = GRID * CELL / 2
HALF = GRID_HALF + MARGIN
N = int(round(2 * HALF / RES)) + 1


def cell_treads(layout: str = "train") -> np.ndarray:
    """Tread depth of each cell (m), indexed [iy, ix]; mixed layouts cycle so every level and
    both pyramids and pits get every tread."""
    t = LAYOUTS[layout]["tread"]
    if np.isscalar(t):
        return np.full((GRID, GRID), float(t))
    iy, ix = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    return np.asarray(t)[(ix // 2 + iy) % len(t)]


def n_steps(layout: str = "train", tread=None):
    """Steps per flight, from the cell border to the central platform (per cell if tread is an array)."""
    tread = cell_treads(layout) if tread is None else tread
    return ((CELL / 2 - BORDER - PLATFORM / 2) // tread).astype(int)


def cell_rises(layout: str = "train") -> np.ndarray:
    """Step height of each cell, indexed [iy, ix]."""
    return np.repeat(LAYOUTS[layout]["rises"][:, None], GRID, axis=1)


def cell_kinds(layout: str = "train") -> np.ndarray:
    """+1 for a pyramid (centre raised), -1 for a pit (centre sunk), indexed [iy, ix]."""
    return np.tile(np.where(np.arange(GRID) % 2 == 0, 1, -1), (GRID, 1))


def cell_center(ix, iy, xp=np):
    return xp.stack([-GRID_HALF + (ix + 0.5) * CELL, -GRID_HALF + (iy + 0.5) * CELL], -1)


def heights(layout: str = "train") -> np.ndarray:
    """Ground height grid (world z), shape (N, N), indexed [iy, ix]; x, y from -HALF to +HALF."""
    xs = np.linspace(-HALF, HALF, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")          # [iy, ix]
    inside = (np.abs(X) < GRID_HALF) & (np.abs(Y) < GRID_HALF)
    cx = np.clip(((X + GRID_HALF) // CELL).astype(int), 0, GRID - 1)
    cy = np.clip(((Y + GRID_HALF) // CELL).astype(int), 0, GRID - 1)
    lx, ly = X + GRID_HALF - cx * CELL, Y + GRID_HALF - cy * CELL  # position within the cell
    d_edge = np.minimum(np.minimum(lx, CELL - lx), np.minimum(ly, CELL - ly)) - BORDER
    tread = cell_treads(layout)[cy, cx]
    steps = np.clip(np.floor(d_edge / tread) + 1, 0, n_steps(layout, tread)) * (d_edge >= 0)
    return steps * cell_rises(layout)[cy, cx] * cell_kinds(layout)[cy, cx] * inside


# sphere centres in the ankle-roll body frame: the sole box spans x -0.05..0.13, y +-0.03, bottom z -0.037
FOOT_SPHERE_R = 0.012
FOOT_SPHERES = [(x, y, -0.037 + FOOT_SPHERE_R) for x in (-0.038, 0.04, 0.118) for y in (-0.018, 0.018)]
FEET_XML = "g1_stairs_feet.xml"


def feet_xml(base: str) -> str:
    """Playground's feet-only G1 MJCF (text) plus FOOT_SPHERES on each sole, each paired with the
    floor like the sole box (same friction)."""
    pairs = []
    for side in ("left", "right"):
        box = f'<geom name="{side}_foot" class="foot"'
        i = base.index(box)
        end = base.index("/>", i) + 2
        spheres = "".join(f'\n                  <geom name="{side}_sole_{k}" class="collision" type="sphere" '
                          f'size="{FOOT_SPHERE_R}" pos="{x} {y} {z:.4f}"/>'
                          for k, (x, y, z) in enumerate(FOOT_SPHERES))
        base = base[:end] + spheres + base[end:]
        pairs += [f'\n    <pair name="{side}_sole_{k}_floor" geom1="{side}_sole_{k}" geom2="floor" condim="3" friction="0.6 0.6"/>'
                  for k in range(len(FOOT_SPHERES))]
    anchor = '<pair name="left_foot_right_foot"'
    end = base.index("/>", base.index(anchor)) + 2
    return base[:end] + "".join(pairs) + base[end:]


def scene_xml(layout: str = "train") -> str:
    """MJCF scene: Playground's feet-only G1 + sensors on the stairs heightfield."""
    h = heights(layout)
    lo, hi = float(h.min()), float(h.max())
    span = max(hi - lo, 1e-3)
    # MJCF inline elevation lists rows from +y down to -y; our grid is indexed from -y up.
    # MuJoCo scales elevation to [0, 1] * size_z, so the geom is shifted down to the pit floor.
    elev = " ".join(f"{v:.5f}" for v in ((h[::-1] - lo) / span).ravel())
    return f"""<mujoco model="g1 stairs">
  <include file="{FEET_XML}"/>
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
    <hfield name="stairs" nrow="{N}" ncol="{N}" size="{HALF} {HALF} {span:.5f} 0.1" elevation="{elev}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="stairs" pos="0 0 {lo:.5f}" material="groundplane"/>
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


def scan_stats(layout: str = "train", n: int = 20000, seed: int = 0):
    """Mean and std of each scan point over random standing poses on the grid, for
    initialising the observation normaliser when warm-starting from a flat policy."""
    rng = np.random.default_rng(seed)
    grid = heights(layout)
    xy = rng.uniform(-GRID_HALF, GRID_HALF, (n, 2))
    yaw = rng.uniform(-np.pi, np.pi, n)
    z = lookup(grid, xy) + NOMINAL_HEIGHT
    s = np.stack([scan(grid, np.array([*p, h]), a) for p, h, a in zip(xy, z, yaw)])
    return s.mean(0), s.std(0) + 1e-3

"""Simulated head depth camera -> point cloud -> elevation map -> height scan, checked against truth.

Models the G1's head RealSense D435i: mounted on the head, pitched down, 87 x 58 deg field of
view, 0.3-3 m range, depth noise that grows with distance, random dropouts. Pixels that land
on the robot itself (hands, arms, feet) are removed, as a real pipeline does with the robot's own
kinematics. What's left is fused into a world-frame elevation map, which gives the same kind
of ground-height scan the planner and the policy use, but built from what the camera saw.
The simulator knows the true terrain, so every estimate can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import mujoco
import numpy as np

CAM_NAME = "head_depth"


@dataclass
class CameraSpec:
    pos: tuple = (0.075, 0.0, 0.44)   # in the pelvis frame: front of the head
    pitch_deg: float = 45.0           # downward tilt
    fovy_deg: float = 58.0            # D435 depth vertical FOV (horizontal ~87 at 16:9)
    width: int = 212                  # D435 depth is 848x480; a quarter of that per side
    height: int = 120
    near: float = 0.3
    far: float = 3.0
    noise: bool = True


def add_head_camera(spec: mujoco.MjSpec, cam: CameraSpec, body: str = "pelvis"):
    """Attach the camera to the robot in an MjSpec before compiling."""
    s, c = np.sin(np.radians(cam.pitch_deg)), np.cos(np.radians(cam.pitch_deg))
    # MuJoCo cameras look along -z with +y up: x = robot right, y = up tilted forward -> looks forward-down
    spec.body(body).add_camera(name=CAM_NAME, pos=list(cam.pos), fovy=cam.fovy_deg,
                               xyaxes=[0, -1, 0, s, 0, c])


class HeadCamera:
    def __init__(self, m: mujoco.MjModel, cam: CameraSpec = CameraSpec(), seed: int = 0):
        self.m, self.cam = m, cam
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        self.depth_r = mujoco.Renderer(m, cam.height, cam.width)
        self.depth_r.enable_depth_rendering()
        self.seg_r = mujoco.Renderer(m, cam.height, cam.width)
        self.seg_r.enable_segmentation_rendering()
        self.rgb_r = mujoco.Renderer(m, 240, 320)
        self.rng = np.random.default_rng(seed)
        # geoms that belong to the robot (anything not rooted in the static world body)
        self.robot_geom = m.body_rootid[m.geom_bodyid] != 0
        # per-pixel ray directions in the camera frame (MuJoCo: -z forward, +y up)
        f = 0.5 * cam.height / np.tan(np.radians(cam.fovy_deg) / 2)
        u, v = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
        self.ray_x = (u - cam.width / 2) / f
        self.ray_y = -(v - cam.height / 2) / f

    def close(self):
        for r in (self.depth_r, self.seg_r, self.rgb_r):
            r.close()

    def capture(self, d: mujoco.MjData, rgb: bool = True) -> dict:
        self.depth_r.update_scene(d, camera=self.cam_id)
        depth = self.depth_r.render().astype(np.float64)       # metres along the optical axis
        self.seg_r.update_scene(d, camera=self.cam_id)
        seg = self.seg_r.render()[..., 0]                       # geom id per pixel, -1 = background
        valid = (depth > self.cam.near) & (depth < self.cam.far) & (seg >= 0)
        valid &= ~self.robot_geom[np.clip(seg, 0, None)]       # self-filter: drop the robot's own body
        if self.cam.noise:
            depth = depth + self.rng.normal(0, 1, depth.shape) * (0.001 + 0.004 * depth ** 2)
            valid &= self.rng.random(depth.shape) > 0.03        # random dropouts
        pts_cam = np.stack([self.ray_x * depth, self.ray_y * depth, -depth], -1)[valid]
        R, p = d.cam_xmat[self.cam_id].reshape(3, 3), d.cam_xpos[self.cam_id]
        out = {"depth": np.where(valid, depth, np.nan), "points": pts_cam @ R.T + p,
               "cam_pos": p.copy(), "self_pixels": int((self.robot_geom[np.clip(seg, 0, None)] & (seg >= 0)).sum())}
        if rgb:
            self.rgb_r.update_scene(d, camera=self.cam_id)
            out["rgb"] = self.rgb_r.render()
        return out


class ElevationMap:
    """World-frame height grid fused from camera points: per-cell mean each frame, smoothed over frames."""

    def __init__(self, half: float = 12.0, res: float = 0.05, alpha: float = 0.5):
        self.half, self.res, self.alpha = half, res, alpha
        n = int(2 * half / res) + 1
        self.h = np.full((n, n), np.nan)
        self.t = np.full((n, n), -np.inf)

    def _idx(self, xy):
        return np.clip(np.floor((np.asarray(xy) + self.half) / self.res).astype(int), 0, self.h.shape[0] - 1)

    def integrate(self, points: np.ndarray, t: float):
        if not len(points):
            return
        ix, iy = self._idx(points[:, :2]).T
        key = iy * self.h.shape[0] + ix
        cells, inv = np.unique(key, return_inverse=True)
        mean = np.bincount(inv, points[:, 2]) / np.bincount(inv)
        old = self.h.flat[cells]
        self.h.flat[cells] = np.where(np.isnan(old), mean, (1 - self.alpha) * old + self.alpha * mean)
        self.t.flat[cells] = t

    def height(self, xy) -> np.ndarray:
        """Height of the cell under each point; if unseen, mean of seen neighbours; NaN if none."""
        ix, iy = self._idx(xy).T
        n = self.h.shape[0]
        out = self.h[iy, ix].copy()
        miss = np.isnan(out)
        if miss.any():
            nb = np.stack([self.h[np.clip(iy[miss] + dy, 0, n - 1), np.clip(ix[miss] + dx, 0, n - 1)]
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN neighbourhoods stay NaN
                out[miss] = np.nanmean(nb, axis=0)
        return out


def scan_points(base_xy, yaw, pts):
    c, s = np.cos(yaw), np.sin(yaw)
    return base_xy + np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], -1)


def compare(est: np.ndarray, truth: np.ndarray) -> dict:
    seen = ~np.isnan(est)
    err = np.abs(est[seen] - truth[seen])
    return {
        "coverage": float(seen.mean()),
        "mae_cm": float(err.mean() * 100) if seen.any() else None,
        "p95_cm": float(np.percentile(err, 95) * 100) if seen.any() else None,
        "max_cm": float(err.max() * 100) if seen.any() else None,
    }


def composite(rgb: np.ndarray, depth: np.ndarray, line_x, truth, est, far: float = 3.0) -> np.ndarray:
    """One 960x240 image for the live view: head RGB | depth | true vs camera ground profile."""
    from matplotlib import colormaps
    from PIL import Image, ImageDraw

    lut = (colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    d = np.nan_to_num((depth - 0.3) / (far - 0.3), nan=-1)
    img = np.where((d < 0)[..., None], 0, lut[np.clip((d * 255).astype(int), 0, 255)]).astype(np.uint8)
    dimg = Image.fromarray(img).resize((320, int(320 * depth.shape[0] / depth.shape[1])), Image.NEAREST)
    canvas = Image.new("RGB", (960, 240), (18, 19, 21))
    canvas.paste(Image.fromarray(rgb).resize((320, 240)), (0, 0))
    canvas.paste(dimg, (320, (240 - dimg.height) // 2))
    g = ImageDraw.Draw(canvas)
    x0, y0, w, h = 672, 20, 272, 190
    top = max(float(np.nanmax(truth)), 0.1) * 1.2
    px = lambda x: x0 + x / line_x[-1] * w
    py = lambda z: y0 + h - (z + 0.02) / (top + 0.02) * h
    g.rectangle([x0, y0, x0 + w, y0 + h], outline=(70, 72, 78))
    g.line([(px(x), py(z)) for x, z in zip(line_x, truth)], fill=(200, 200, 200), width=3)
    for x, z in zip(line_x, est):
        if not np.isnan(z):
            g.ellipse([px(x) - 2.5, py(z) - 2.5, px(x) + 2.5, py(z) + 2.5], fill=(228, 87, 46))
    g.text((x0, 4), "ground ahead: truth (grey) vs camera (orange)", fill=(200, 200, 200))
    g.text((x0, y0 + h + 6), f"0 m{'':>52}{line_x[-1]:.0f} m ahead", fill=(150, 150, 150))
    g.text((6, 6), "head RGB", fill=(255, 255, 255))
    g.text((326, 6), "head depth (black = no return / robot)", fill=(255, 255, 255))
    return np.asarray(canvas)

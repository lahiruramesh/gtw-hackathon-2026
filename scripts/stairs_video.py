"""Annotated stairs video of a trained policy on the held-out test staircases (plain MuJoCo, CPU).

One crossing per step height (alternating pyramid and pit), each with a side panel stating the
staircase geometry (step height, tread depth = the stairs' step length, steps per flight), the
command (speed, cadence, commanded step length), the robot's measured step length at every
touchdown, a live side profile of the staircase with the robot on it, and the outcome. Title and
summary cards at the start and end.

    uv run python scripts/stairs_video.py runs/g1-stairs-v14/run/params.pkl --out results/videos/stairs_v14_annotated.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1pipe import stairs_terrain as T  # noqa: E402
from g1pipe.evaluate import MIN_AIR_S  # noqa: E402
from g1pipe.stairs_eval import StairsPolicyRunner  # noqa: E402

W, H = 1280, 720
VIEW_W = 880                      # MuJoCo render width; the panel takes the rest
PANEL_X = VIEW_W
FPS = 25                          # the runner renders every 2nd 20 ms control step: real time
BG, FG, DIM, ACCENT, OK, BAD = (18, 20, 24), (236, 238, 241), (150, 156, 165), (86, 156, 255), (80, 200, 120), (235, 90, 80)
STAIR, ROBOT = (120, 128, 140), (255, 196, 60)


def font(size, bold=False):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_H, F_B, F_S, F_BIG = font(34, True), font(19, True), font(17), font(14), font(24, True)
try:
    F_MONO = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 17)
except OSError:
    F_MONO = ImageFont.load_default()


def card(lines, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 150
    for i, (text, f, col) in enumerate(lines):
        d.text((90, y), text, font=f, fill=col)
        y += f.size + (22 if i == 0 else 12)
    if sub:
        d.text((90, H - 70), sub, font=F_S, fill=DIM)
    return np.asarray(img)


def profile(r, ix, iy, x0, y):
    """Ground height along the crossing line (x from the cell's west edge), for the side diagram."""
    xs = np.linspace(x0 - 0.3, x0 + T.CELL + 0.3, 400)
    zs = T.lookup(r.grid, np.stack([xs, np.full_like(xs, y)], -1))
    return xs, zs


def panel(img, info, live, prof, robot_xz):
    d = ImageDraw.Draw(img)
    d.rectangle([PANEL_X, 0, W, H], fill=BG)
    x, y = PANEL_X + 24, 20
    d.text((x, y), info["title"], font=F_H, fill=FG); y += 26
    d.text((x, y), info["subtitle"], font=F_S, fill=DIM); y += 34

    d.text((x, y), "TEST STAIRCASE (held out)", font=F_S, fill=ACCENT); y += 24
    rows = [("Type", info["kind_text"]),
            ("Step height (rise)", f"{info['rise_cm']:.1f} cm"),
            ("Tread depth (stair step length)", f"{info['tread_cm']:.0f} cm"),
            ("Steps per flight", f"{info['n_steps']}  (+ 1.0 m platform)"),
            ("Training treads", "25 / 29 / 33 cm")]
    for k, v in rows:
        d.text((x, y), k, font=F_S, fill=DIM)
        d.text((W - 24, y), v, font=F_B, fill=FG, anchor="ra"); y += 24
    y += 12

    d.text((x, y), "COMMAND", font=F_S, fill=ACCENT); y += 24
    for k, v in [("Speed", f"{info['vx']:.2f} m/s"), ("Cadence", f"{info['cadence']:.1f} Hz"),
                 ("Commanded step length", f"{info['step_cmd_cm']:.1f} cm")]:
        d.text((x, y), k, font=F_S, fill=DIM)
        d.text((W - 24, y), v, font=F_B, fill=FG, anchor="ra"); y += 24
    y += 12

    d.text((x, y), "LIVE", font=F_S, fill=ACCENT); y += 24
    last = f"{live['last_cm']:.1f} cm" if live["last_cm"] is not None else "-"
    mean = f"{live['mean_cm']:.1f} cm ({live['n']} steps)" if live["n"] else "-"
    for k, v in [("Time", f"{live['t']:.1f} s"), ("Measured step length", last), ("Mean so far", mean),
                 ("Height above start", f"{live['dz_cm']:+.1f} cm")]:
        d.text((x, y), k, font=F_S, fill=DIM)
        d.text((W - 24, y), v, font=F_B, fill=FG, anchor="ra"); y += 24
    y += 10

    # side profile of the staircase, the robot's position on it
    xs, zs = prof
    bx0, bx1, by0, by1 = x, W - 24, y + 6, y + 120
    d.rectangle([bx0, by0, bx1, by1], outline=(45, 50, 58))
    zmin, zmax = min(zs.min(), -0.02), max(zs.max(), 0.02)
    span = max(zmax - zmin, 0.25)
    mid = (zmax + zmin) / 2
    sx = lambda v: bx0 + 6 + (v - xs[0]) / (xs[-1] - xs[0]) * (bx1 - bx0 - 12)
    sz = lambda v: by1 - 12 - (v - (mid - span / 2)) / span * (by1 - by0 - 24)
    pts = [(sx(a), sz(b)) for a, b in zip(xs, zs)]
    d.line(pts, fill=STAIR, width=3)
    rx, rz = robot_xz
    d.ellipse([sx(rx) - 6, sz(rz) - 6, sx(rx) + 6, sz(rz) + 6], fill=ROBOT)
    d.text((bx0 + 6, by1 + 6), "side profile (east ->), vertical scale exaggerated", font=F_S, fill=DIM)
    y = by1 + 34

    col = {"walking": FG, "crossed": OK, "fell": BAD}[live["status"]]
    d.text((x, y), live["status"].upper(), font=F_BIG, fill=col)
    for i, line in enumerate(info["footer"]):
        d.text((x, H - 48 + 18 * i), line, font=F_S, fill=DIM)
    return img


def crossing(r, ix, iy, vx, step_cmd, cadence, renderer, info_base, duration, write):
    x0 = -T.GRID_HALF + ix * T.CELL
    y = -T.GRID_HALF + (iy + 0.5) * T.CELL
    r.place(x0 + 0.15, y)
    r.reset_scan()
    rise = float(r.rises[iy, ix])
    kind = int(r.kinds[iy, ix])
    tread = float(T.cell_treads("test")[iy, ix])
    n = int(T.n_steps("test")[iy, ix])
    info = {**info_base, "rise_cm": rise * 100, "tread_cm": tread * 100, "n_steps": n,
            "kind_text": "pyramid: up, across, down" if kind > 0 else "pit: down, across, up"}
    prof = profile(r, ix, iy, x0, y)
    z_start = float(T.lookup(r.grid, np.array([x0 + 0.15, y])))
    tele, steps = [], []
    prev_contact = np.array([True, True])
    air = np.zeros(2)

    def schedule(t):
        nonlocal prev_contact, air
        contact = np.array([r.d.sensordata[a] > 0 for a in r.floor_sensors])
        touchdown = contact & ~prev_contact & (air >= MIN_AIR_S)
        fwd = r.d.xmat[r.torso].reshape(3, 3)[:2, 0]
        fwd /= np.linalg.norm(fwd) + 1e-9
        feet = r.d.site_xpos[r.feet_sites][:, :2]
        for i in (0, 1):
            if touchdown[i] and t > 0.5:
                steps.append(float(np.dot(feet[i] - feet[1 - i], fwd)))
        air = np.where(contact, 0.0, air + r.ctrl_dt)
        prev_contact = contact
        p = r.d.qpos[:3]
        ground = float(T.lookup(r.grid, p[:2]))
        tele.append({"t": t, "x": float(p[0]), "ground": ground, "steps": list(steps)})
        return (vx, step_cmd)

    summary, frames, _, _ = r.run(vx=vx, step_len=step_cmd, duration=duration, renderer=renderer,
                                  fps=FPS, schedule=schedule)
    far = x0 + T.CELL - T.BORDER
    crossed = (not summary["fell"]) and max(t["x"] for t in tele) > far
    every = max(1, int(round(1 / (FPS * r.ctrl_dt))))
    if crossed:   # the clip and the step statistics end when the robot is past the far border (+1 s of video)
        k_cross = next(k for k, t in enumerate(tele) if t["x"] > far)
        frames = frames[: (k_cross // every) + FPS]
        steps = tele[k_cross]["steps"]
    last = None
    for j, fr in enumerate(frames):
        k = min(j * every, len(tele) - 1)
        tl = tele[k]
        st = tl["steps"]
        status = "crossed" if tl["x"] > x0 + T.CELL - T.BORDER and not summary["fell"] else "walking"
        if summary["fell"] and j >= len(frames) - 3:
            status = "fell"
        live = {"t": tl["t"], "last_cm": st[-1] * 100 if st else None, "n": len(st),
                "mean_cm": float(np.mean(st)) * 100 if st else 0.0, "dz_cm": (tl["ground"] - z_start) * 100,
                "status": status}
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(fr), (0, 0))
        last = np.asarray(panel(img, info, live, prof, (tl["x"], tl["ground"])))
        write(last)
    for _ in range(FPS if summary["fell"] else FPS // 2):   # hold the last frame (longer after a fall)
        write(last)
    result = {"rise_cm": round(rise * 100, 1), "kind": "pyramid" if kind > 0 else "pit", "tread_cm": tread * 100,
              "steps_per_flight": n, "crossed": bool(crossed), "fell": bool(summary["fell"]),
              "step_len_mean_cm": round(float(np.mean(steps)) * 100, 1) if steps else None, "n_steps": len(steps)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="?", default="runs/g1-stairs-v14/run/params.pkl")
    ap.add_argument("--out", default="results/videos/stairs_v14_annotated.mp4")
    ap.add_argument("--vx", type=float, default=0.7)
    ap.add_argument("--duration", type=float, default=16.0, help="longest crossing, s (as the strict test)")
    ap.add_argument("--label", default=None, help="policy name on the cards (default: the run directory)")
    a = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from g1pipe.stairs_env import STAIRS_CADENCE

    params = Path(a.params)
    label = a.label or params.parent.parent.name
    cfg = json.loads(params.with_name("config.json").read_text()) if params.with_name("config.json").exists() else {}
    step_cmd = a.vx / (2 * STAIRS_CADENCE)
    r = StairsPolicyRunner(str(params), layout="test")
    r.m.vis.global_.offwidth, r.m.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(r.m, H, VIEW_W)

    steps_m = cfg.get("ppo", {}).get("num_timesteps")
    info_base = {"title": f"Unitree G1 · {label}", "vx": a.vx, "cadence": STAIRS_CADENCE, "step_cmd_cm": step_cmd * 100,
                 "subtitle": f"{params}" + (f" · {steps_m / 1e6:.0f}M steps" if steps_m else ""),
                 "footer": ["plain MuJoCo (CPU), true height scan", "20 ms actuation latency (this evaluator)"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=FPS, quality=8, macro_block_size=8)
    results, n_frames = [], [0]

    def write(frame, times=1):
        for _ in range(times):
            writer.append_data(frame)
        n_frames[0] += times

    write(card([(f"{label}: stair climbing on held-out stairs", F_TITLE, FG),
                     (f"Test layout: step heights 3.0 to 15.0 cm, tread depth 27 cm (training: 25 / 29 / 33 cm)", F_B, FG),
                     (f"Command: {a.vx:.2f} m/s at {STAIRS_CADENCE:.1f} Hz = {step_cmd * 100:.1f} cm step length", F_B, FG),
                     ("One crossing per step height, alternating pyramid (up, across, down) and pit (down, across, up)", F_B, DIM)],
                    sub=str(params)), FPS * 4)
    for iy in range(T.GRID):
        ix = iy % 2   # alternate the staircase kind
        res = crossing(r, ix, iy, a.vx, step_cmd, STAIRS_CADENCE, renderer, info_base, a.duration, write)
        results.append(res)
        print(json.dumps(res), flush=True)

    rows = [(f"{'Step height':>11}  {'Type':8} {'Tread':>6} {'Steps':>5}  {'Mean step length':>16}  Result", F_MONO, DIM)]
    for x in results:
        outcome = "crossed" if x["crossed"] else ("FELL" if x["fell"] else "did not cross")
        sl = f"{x['step_len_mean_cm']:.1f} cm" if x["step_len_mean_cm"] is not None else "-"
        rows.append((f"{x['rise_cm']:>8.1f} cm  {x['kind']:8} {x['tread_cm']:>3.0f} cm {x['steps_per_flight']:>5}  "
                     f"{sl:>16}  {outcome}", F_MONO, OK if x["crossed"] else BAD))
    crossed = sum(x["crossed"] for x in results)
    write(card([(f"{label}: {crossed}/{len(results)} held-out staircases crossed in this video", F_TITLE, FG),
                     (f"Commanded step length {step_cmd * 100:.1f} cm on 27 cm treads", F_B, DIM), *rows],
                    sub="Full strict test (96 crossings): scripts/certify.py -> policy card"), FPS * 6)
    writer.close()
    Path(a.out).with_suffix(".json").write_text(json.dumps(results, indent=1))
    print("video:", a.out, f"({n_frames[0] / FPS:.0f} s)")


if __name__ == "__main__":
    main()

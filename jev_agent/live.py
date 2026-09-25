"""Live preview: the G1 walking in MuJoCo in real time, with the Jev agent's decisions beside it.

    python -m jev_agent.live runs/g1-steplength-v1/run/params.pkl --port 8765
    python -m jev_agent.live --world stairs --port 8766      # Unitree policy + staircase + height scan
    open http://localhost:8765

The sim runs at wall-clock speed and calls Jev asynchronously (as on hardware), so the video
never waits on the network. Episodes restart after a fall or timeout; the learner updates
after each one (runs/jev_agent/live/learner.json). Scenario and mission can be changed from
the page and apply at the next episode.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np
from PIL import Image

from g1pipe.evaluate import PolicyRunner, command_to_gait
from jev_agent import questions as Q
from jev_agent.brain import make_brain
from jev_agent.learner import Learner
from jev_agent.run import MISSIONS, SCENARIOS
from jev_agent import stairs as S
from jev_agent.perception import PPORobot, VendorRobot
from jev_agent.supervisor import Supervisor
from jev_agent import vision as V

FPS = 25
PAGE = Path(__file__).with_name("live.html")


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = b""
        self.frame_id = 0
        self.cam_frame, self.cam_id = b"", 0
        self.status = {}
        self.episodes = deque(maxlen=20)
        self.control = {"scenario": None, "mission": None}
        self.scenarios, self.missions, self.modes = list(SCENARIOS), MISSIONS, Q.GAIT_MODES


def sim_loop(sh: Shared, params: str, brain_kind: str, episode_s: float):
    runner = PolicyRunner(params)
    renderer = mujoco.Renderer(runner.m, 480, 640)
    learner = Learner(Path("runs/jev_agent/live/learner.json"))
    brain = make_brain(brain_kind)
    scen_names, ep = list(SCENARIOS), 0
    every = max(1, int(round(1 / (FPS * runner.ctrl_dt))))

    while True:
        with sh.lock:
            scen = sh.control["scenario"] or scen_names[ep % len(scen_names)]
            mission = sh.control["mission"] or MISSIONS[ep % len(MISSIONS)]
        sup = Supervisor(PPORobot(runner), brain, learner, mission, Q.GAIT_MODES, async_brain=True)
        wall0, tick = time.perf_counter(), [0]

        def schedule(t):
            cmd = Q.GAIT_MODES[sup(t)]
            if tick[0] % every == 0:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = runner.d.qpos[:3] + np.array([0, 0, -0.25])
                cam.distance, cam.azimuth, cam.elevation = 2.8, 120, -12
                renderer.update_scene(runner.d, camera=cam)
                buf = io.BytesIO()
                Image.fromarray(renderer.render()).save(buf, "JPEG", quality=80)
                last = sup.decisions[-1] if sup.decisions else None
                with sh.lock:
                    sh.frame, sh.frame_id = buf.getvalue(), sh.frame_id + 1
                    sh.status = {
                        "episode": ep, "scenario": scen, "mission": mission, "t": round(t, 2),
                        "mode": sup.mode, "override": bool(last) and sup.mode != last["mode"], "cmd": {"vx": cmd[0], "step": cmd[1],
                                                  "cadence_hz": round(command_to_gait(*cmd)[0], 2)},
                        "jev_pending": sup._pending is not None, "brain": brain_kind,
                        "live": sup.perception.snapshot() if t > 0 else None,
                        "last": last, "recent": [
                            {k: d[k] for k in ("t", "prev_mode", "mode", "reason")} |
                            {"conf": d["jev"]["gait_conf"], "latency_s": d["jev"]["latency_s"]}
                            for d in sup.decisions[-12:]][::-1],
                    }
                # pace to wall clock
                lag = t - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
            tick[0] += 1
            return cmd

        summary, _, _, log = runner.run(duration=episode_s, perturb=SCENARIOS[scen], seed=ep, schedule=schedule)
        decs = sup.finish(summary["fall_time"])
        log_decisions(Path("runs/jev_agent/live"), ep, scen, decs)
        learner.learn(decs)
        learner.save()
        with sh.lock:
            sh.episodes.appendleft({
                "episode": ep, "scenario": scen, "mission": mission, "fell": summary["fell"],
                "time_s": round(log["t"][-1], 1), "distance_m": round(sum(log["vx"]) * runner.ctrl_dt, 2),
                "decisions": len(decs), "modes": sorted({d["mode"] for d in decs}, key=Q.MODE_ORDER.index),
            })
            sh.status["between"] = True
        time.sleep(1.2)   # hold the last frame so a fall is visible
        ep += 1


def log_decisions(folder: Path, ep: int, scen: str, decs: list):
    """Append every decision (state sent to Jev, Jev's answer, what the agent did, outcome)."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "decisions.jsonl", "a") as f:
        for d in decs:
            f.write(json.dumps({"episode": ep, "scenario": scen, "wall_time": time.time(), **d}) + "\n")


STAIR_RISES = {"4 cm steps": 0.04, "5 cm steps": 0.05, "8 cm steps": 0.08, "12 cm steps": 0.12}


def status_of(sup, ep, scen, mission, t, cmd_txt, brain_kind, extra=None):
    last = sup.decisions[-1] if sup.decisions else None
    return {
        "episode": ep, "scenario": scen, "mission": mission, "t": round(t, 2),
        "mode": sup.mode, "override": bool(last) and sup.mode != last["mode"], "reflex": sup.reflex,
        "cmd": cmd_txt,
        "jev_pending": sup._pending is not None, "brain": brain_kind,
        "live": sup.perception.snapshot() if t > 0 else None,
        "last": last, "recent": [
            {k: d[k] for k in ("t", "prev_mode", "mode", "reason")} |
            {"conf": d["jev"]["gait_conf"], "latency_s": d["jev"]["latency_s"]}
            for d in sup.decisions[-12:]][::-1],
        **(extra or {}),
    }


def stairs_loop(sh: Shared, brain_kind: str, episode_s: float, learner_path: str, terrain_src: str = "camera"):
    learner = Learner(Path(learner_path))
    brain = make_brain(brain_kind, terrain=True)
    names, ep = list(STAIR_RISES), 0
    while True:
        with sh.lock:
            scen = sh.control["scenario"] or names[ep % len(names)]
            mission = sh.control["mission"] or S.MISSIONS[(ep // len(names)) % len(S.MISSIONS)]
        stairs = S.Staircase(rise=STAIR_RISES[scen])
        sim = S.build(stairs, camera=True)
        renderer = mujoco.Renderer(sim.m, 480, 640)
        cam, emap = V.HeadCamera(sim.m, seed=ep), V.ElevationMap()
        terrain = S.make_terrain(sim, heights_fn=emap.height if terrain_src == "camera" else None)
        sup = Supervisor(VendorRobot(sim), brain, learner, mission, S.SUP_MODES,
                         async_brain=True, terrain=terrain, seed=ep)
        wall0 = time.perf_counter()
        vis = {}
        line_x = np.linspace(0.0, 3.0, 61)

        def on_tick(t, mode, cmd, period):
            k = int(round(t / 0.02))
            if k % 10 == 0:                        # head camera at 5 Hz -> elevation map
                shot = cam.capture(sim.d)
                emap.integrate(shot["points"], t)
                base, yaw = sim.d.qpos[:2].copy(), S.yaw_of(sim.d.qpos[3:7])
                lxy = V.scan_points(base, yaw, np.stack([line_x, np.zeros_like(line_x)], -1))
                truth, est = stairs.ground(lxy), emap.height(lxy)
                vis.update(V.compare(est, truth), points=int(len(shot["points"])),
                           self_pixels=shot["self_pixels"], source=terrain_src)
                buf2 = io.BytesIO()
                Image.fromarray(V.composite(shot["rgb"], shot["depth"], line_x, truth, est)).save(buf2, "JPEG", quality=80)
                with sh.lock:
                    sh.cam_frame, sh.cam_id = buf2.getvalue(), sh.cam_id + 1
            if k % 2:
                return
            view = mujoco.MjvCamera()
            view.lookat[:] = sim.d.qpos[:3] + np.array([0, 0, -0.3])
            view.distance, view.azimuth, view.elevation = 3.4, 105, -14
            renderer.update_scene(sim.d, camera=view)
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, "JPEG", quality=80)
            terr = sup.terrain_now
            with sh.lock:
                sh.frame, sh.frame_id = buf.getvalue(), sh.frame_id + 1
                sh.status = status_of(sup, ep, scen, mission, t, {
                    "vx": cmd[0], "step": cmd[0] * period / 2, "cadence_hz": round(2 / period, 2)}, brain_kind, {
                    "terrain": terr[0] if terr else None, "terrain_features": terr[1] if terr else None,
                    "vision": dict(vis), "attempt": sup.attempt, "skill": {
                        size: {"gaits": g, "too_hard": learner.stairs.too_hard(size)}
                        for size, g in learner.stairs.stats.items()},
                })
            lag = t - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)

        epi, res = S.run_episode(sim, sup, episode_s, seed=ep, on_tick=on_tick)
        decs = sup.finish(epi.fall_time)
        log_decisions(Path(learner_path).parent, ep, scen, decs)
        learner.learn(decs)
        learner.save()
        renderer.close()
        cam.close()
        result = ("fell" if res["fell"] else "crossed" if res["crossed"] else
                  "stopped before stairs" if res["stopped_before"] else "on the way")
        with sh.lock:
            sh.episodes.appendleft({
                "episode": ep, "scenario": scen, "mission": mission, "fell": res["fell"], "result": result,
                "time_s": round(epi.t[-1], 1) if epi.t else 0, "distance_m": res["max_x"],
                "decisions": len(decs), "modes": sorted({d["mode"] for d in decs}, key=Q.MODE_ORDER.index),
            })
        time.sleep(1.5)
        ep += 1


def make_handler(sh: Shared):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/state":
                with sh.lock:
                    body = json.dumps({"status": sh.status, "episodes": list(sh.episodes),
                                       "scenarios": sh.scenarios, "missions": sh.missions,
                                       "modes": sh.modes, "control": sh.control})
                self._send(body.encode(), "application/json")
            elif url.path == "/control":
                q = parse_qs(url.query)
                with sh.lock:
                    for k in ("scenario", "mission"):
                        if k in q:
                            v = q[k][0].strip()
                            sh.control[k] = v if v and (k != "scenario" or v in sh.scenarios) else None
                self._send(b"ok", "text/plain")
            elif url.path in ("/stream", "/camera"):
                attr = ("frame", "frame_id") if url.path == "/stream" else ("cam_frame", "cam_id")
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with sh.lock:
                            frame, fid = getattr(sh, attr[0]), getattr(sh, attr[1])
                        if fid != seen and frame:
                            seen = fid
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                        time.sleep(1 / (FPS * 2))
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="?", help="PPO params.pkl (flat world)")
    ap.add_argument("--world", choices=["flat", "stairs"], default="flat")
    ap.add_argument("--learner", default="runs/jev_agent/live_stairs/learner.json",
                    help="stairs world: learner file (copy a trained one here to start experienced)")
    ap.add_argument("--brain", choices=["jev", "offline"], default="jev")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--episode-s", type=float, default=22.0)
    ap.add_argument("--terrain", choices=["camera", "raycast"], default="camera",
                    help="stairs world: what the stair detector reads (head depth camera map, or perfect raycasts)")
    a = ap.parse_args()
    sh = Shared()
    if a.world == "stairs":
        sh.scenarios, sh.missions = list(STAIR_RISES), S.MISSIONS
        sh.modes = {m: (v, v * p / 2) for m, (v, p) in S.VENDOR_MODES.items()}
    elif not a.params:
        ap.error("params is required for the flat world")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sh))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"live preview: http://localhost:{a.port}", flush=True)
    # main thread: MuJoCo's GL context lives here
    if a.world == "stairs":
        stairs_loop(sh, a.brain, a.episode_s, a.learner, a.terrain)
    else:
        sim_loop(sh, a.params, a.brain, a.episode_s)


if __name__ == "__main__":
    main()

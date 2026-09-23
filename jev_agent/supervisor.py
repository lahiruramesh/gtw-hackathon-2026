"""High-level agent: perceive -> ask Jev -> decide in code -> command the low-level walking policy.

Robot-agnostic: sensors come through an adapter (perception.PPORobot / VendorRobot),
commands go out as a gait-mode name that the caller maps to its policy's command
(see questions.GAIT_MODES, stairs.VENDOR_MODES). Call sup(t) every control tick.

Jev is consulted every DECIDE_EVERY_S. By default the call blocks (sim time
pauses), so results are reproducible regardless of network latency. With
async_brain=True the call runs in the background and its answer is applied when
it arrives, as it would have to be on hardware or in a real-time preview; the
tilt safety filter is then also checked every tick, without waiting for Jev.

Optional terrain sensing: terrain() -> (text for Jev, features dict for code). With it,
Jev also answers the stairs questions (does the mission want them?), code finds the edges,
and the learner's stairs skill picks the gait to climb with, or declines stairs it has
learned are too tall for this robot.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from jev_agent import questions as Q
from jev_agent.brain import Answers, build_state
from jev_agent.learner import FALL_HORIZON_S, Learner, context_key, risk_features
from jev_agent.perception import Perception, describe


class Supervisor:
    def __init__(self, robot, brain, learner: Learner, mission: str, modes: dict,
                 start_mode: str = "cautious", async_brain: bool = False,
                 terrain: Callable[[], tuple[str, dict]] | None = None, seed: int = 0):
        """modes: gait name -> (forward speed m/s, step length m), used for perception's tracking stats."""
        self.brain, self.learner, self.mission, self.modes = brain, learner, mission, modes
        self.perception = Perception(robot)
        self.terrain = terrain
        self.terrain_now: tuple[str, dict] | None = None
        self.mode = start_mode
        self.next_decision = Q.DECIDE_EVERY_S
        self.decisions: list[dict] = []
        self._pool = ThreadPoolExecutor(1) if async_brain else None
        self._pending = None   # (future, t, snap, desc, terrain)
        self._rng = np.random.default_rng(seed)
        self.attempt: dict | None = None   # stairs attempt: {size, gait, was_on, left}
        self.stairs_intent: str | None = None   # Jev's latest confident stairs_action
        self.reflex: str | None = None          # reason, while a code reflex holds the gait
        self._tick = 0

    def __call__(self, t: float) -> str:
        vx, step = self.modes[self.mode]
        if t > 0:
            self.perception.update(t, vx, step)
        if self._pool:
            if self._pending and self._pending[0].done():
                fut, *ctx = self._pending
                self._pending = None
                self._apply(*ctx, self._result(fut.result))
            if self.perception.snapshot()["tilt_deg"] > Q.SAFETY_TILT_DEG:
                self.mode = "stop"
        # stairs reflex: fresh height scan at 10 Hz, independent of Jev's latency
        if self.terrain and self._tick % 5 == 0:
            f = self._track(self.terrain())[1]
            r = self._stairs(f)
            self.reflex = r[1] if r else None
            if r and self.perception.snapshot()["tilt_deg"] <= Q.SAFETY_TILT_DEG:
                self.mode = r[0]
        self._tick += 1
        if t >= self.next_decision:
            if not (self._pool and self._pending):   # async: skip a tick while a call is in flight
                self._decide(t)
            self.next_decision += Q.DECIDE_EVERY_S
        return self.mode

    # ------------------------------------------------------------------------
    def _decide(self, t: float):
        snap = self.perception.snapshot()
        desc = describe(snap)
        terr = self._track(self.terrain()) if self.terrain else None
        self._close_outcome(snap, fell=False)
        self.perception.end_window()

        state = build_state(self.mission, desc, self.mode, terr[0] if terr else None)
        if self._pool:
            self._pending = (self._pool.submit(self.brain.ask, state), t, snap, desc, terr)
        else:
            self._apply(t, snap, desc, terr, self._result(lambda: self.brain.ask(state)))

    def _track(self, terr):
        """Remember the latest terrain reading and advance any stairs attempt."""
        self.terrain_now = terr
        if self.attempt:
            on = terr[1]["on_stairs"]
            self.attempt["left"] |= self.attempt["was_on"] and not on and terr[1]["edge_dist_m"] is None
            self.attempt["was_on"] |= on
        return terr

    @staticmethod
    def _result(call) -> Answers:
        try:
            return call()
        except Exception as e:  # network, rate limit, auth... never let the robot wait on it
            return Answers(gait_probs={m: 0.0 for m in Q.MODE_ORDER}, gait_conf=0.0,
                           stability=3.0, stability_conf=0.0, nouls={}, error=f"{type(e).__name__}: {e}")

    def _apply(self, t: float, snap: dict, desc: dict, terr, ans: Answers):
        feats = terr[1] if terr else None
        sa = ans.choices.get("stairs_action")
        if sa and sa["confidence"] >= Q.CONF_ACT:
            self.stairs_intent = sa["choice"]
        ctx = context_key(desc, ans.nouls, feats["tag"] if feats else None)
        mode, reason, bonus, risks = self._choose(snap, ans, ctx, feats)
        self.decisions.append({
            "t": round(t, 3), "mission": self.mission, "describe": desc,
            "terrain": terr[0] if terr else None, "terrain_features": feats,
            "snap": {k: round(v, 4) for k, v in snap.items()},
            "jev": {"gait_probs": ans.gait_probs, "gait_conf": ans.gait_conf, "stability": ans.stability,
                    "stability_conf": ans.stability_conf, "nouls": ans.nouls, "choices": ans.choices,
                    "latency_s": round(ans.latency_s, 4), "model": ans.model, "error": ans.error,
                    "fallback": ans.fallback},
            "context": ctx, "bonus": bonus, "fall_risk": risks,
            "prev_mode": self.mode, "mode": mode, "reason": reason,
            "risk_x": risk_features(snap, ans, mode, feats).tolist(), "outcome": None,
        })
        self.mode = mode

    def _choose(self, snap, ans: Answers, ctx: str, feats: dict | None):
        cur = Q.MODE_ORDER.index(self.mode)
        bonus = self.learner.bonus(ctx)
        risks = {m: self.learner.fall_risk(snap, ans, m, feats) for m in Q.MODE_ORDER}

        if ans.error:
            r = self._stairs(self.terrain_now[1]) if feats and self.terrain_now else None
            if r:                                   # still stop before an edge while Jev is down
                return (*r, bonus, risks)
            return Q.MODE_ORDER[min(cur, 1)], "brain_error_fallback", bonus, risks
        # 1. hard safety filter: numbers are judged in code, not by the model
        if snap["tilt_deg"] > Q.SAFETY_TILT_DEG or ans.stability < Q.SAFETY_STABILITY:
            return "stop", "safety_filter", bonus, risks
        # 2. stairs: Jev decides whether the mission wants them, code finds the edge,
        #    the skill memory decides which gait (or whether) to attempt them with.
        if feats:
            r = self._stairs(self.terrain_now[1] if self.terrain_now else feats)   # freshest scan
            if r:
                return (*r, bonus, risks)

        # 3. Jev's belief + learned experience, minus modes the risk model vetoes
        score = {}
        for m in Q.MODE_ORDER:
            score[m] = math.log(ans.gait_probs.get(m, 0.0) + 1e-3) + bonus[m]
            if m != "stop" and risks[m] is not None and risks[m] > Q.RISK_VETO and not (feats and feats["tag"] != "flat"):
                score[m] = -math.inf
        best = max(score, key=score.get)
        want = Q.MODE_ORDER.index(best)
        if all(math.isinf(v) for m, v in score.items() if m != "stop") and best == "stop":
            return "stop", "risk_veto", bonus, risks

        # 4. confidence gating: slowing down is always allowed, speeding up must be earned
        if want < cur:
            return best, "slow_down", bonus, risks
        if want > cur:
            if ans.gait_conf < Q.CONF_UPSHIFT and not bonus[best] > 0:
                return self.mode, "hold_low_confidence", bonus, risks
            return Q.MODE_ORDER[cur + 1], "speed_up_one_level", bonus, risks
        return self.mode, "keep", bonus, risks

    def _stairs(self, f: dict):
        """Stairs rules on a terrain reading: Jev's intent + geometry + learned skill. None = no opinion."""
        if self.attempt and not self.attempt["left"]:
            return self.attempt["gait"], "stairs_attempt_hold"          # commit: no gait changes mid-flight
        near = f["edge_dist_m"] is not None and not f["on_stairs"]
        if not near:
            return None
        intent = self.stairs_intent or "stop_before"   # fail safe: no confident answer yet = don't climb
        if intent != "climb":
            if intent == "stop_before" and f["edge_dist_m"] < Q.STAIRS_STOP_DIST_M:
                return "stop", "stop_before_stairs"
            return None
        if f["edge_dist_m"] >= Q.STAIRS_COMMIT_DIST_M or self.attempt:
            return None
        size = f["tag"].removeprefix("on_")
        gait = self.learner.stairs.choose(size, self._rng)
        if gait is None:
            return ("stop", "learned_too_hard") if f["edge_dist_m"] < Q.STAIRS_STOP_DIST_M else None
        self.attempt = {"size": size, "gait": gait, "was_on": False, "left": False}
        return gait, "stairs_attempt_start"

    def _close_outcome(self, snap: dict, fell: bool):
        if self.decisions and self.decisions[-1]["outcome"] is None:
            self.decisions[-1]["outcome"] = {
                "progress_m": snap["progress_m"], "vx_err": snap["vx_err"],
                "max_tilt_deg": snap["max_tilt_deg"], "fell_soon": fell,
            }

    def finish(self, fell_at: float | None):
        """Call after the episode; labels the decisions that led up to a fall, records any stairs attempt."""
        self._close_outcome(self.perception.snapshot(), fell=fell_at is not None)
        if self.attempt:
            self.attempt["success"] = bool(self.attempt["left"] and fell_at is None)
            self.learner.stairs.record(self.attempt["size"], self.attempt["gait"], self.attempt["success"])
        if fell_at is not None:
            for d in self.decisions:
                if d["t"] >= fell_at - FALL_HORIZON_S:
                    d["outcome"]["fell_soon"] = True
        return self.decisions

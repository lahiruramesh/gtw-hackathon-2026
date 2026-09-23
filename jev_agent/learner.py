"""Self-learning layer on top of Jev.

Jev's weights are shared and cannot be fine-tuned (docs: models, "Customizing Jev"),
so the agent learns in code, from its own outcomes:

  1. Experience bandit: for each situation (context) and gait mode, the running mean
     reward of choosing that mode. The advantage of a mode over the context average
     is added to Jev's log-probability, so with experience the agent drifts toward
     what actually worked in that situation on this robot.
  2. Fall-risk model: logistic regression on sensor features + Jev's answers,
     predicting "fell within the next second". Modes it rates too risky are vetoed.
     This is the autoresearch pattern: Jev outputs become features for a small
     classical model trained on real outcomes.

  3. Stairs skill memory: for each step size seen by the height scan, how often each gait
     got the robot across the staircase. Untried gaits are explored (Thompson sampling);
     once every gait keeps failing on a step size, the agent concludes those stairs are
     beyond this robot and stops in front of them instead of trying.

State is a JSON file, so learning carries over between runs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jev_agent import questions as Q

SHRINK = 5.0          # pseudo-count: a mode needs a few tries before its bonus counts
BONUS_SCALE = 1.0
FALL_HORIZON_S = 1.0  # decisions this close to a fall are labelled "led to a fall"
MIN_CLASS = 5         # samples per class before the risk model is trusted
N_RISK = 11           # length of risk_features()


def reward(outcome: dict) -> float:
    return (outcome["progress_m"] - 0.5 * outcome["vx_err"]
            - 0.03 * outcome["max_tilt_deg"] - 10.0 * outcome["fell_soon"])


def context_key(desc: dict, nouls: dict, terrain_tag: str | None = None) -> str:
    posture = "tilted" if not desc["torso_posture"].startswith("upright") else "upright"
    slip = "slip" if ("slipping" in desc["foot_grip"] or "sliding" in desc["foot_grip"]) else "grip"
    push = "pushed" if desc["recent_push"] != "no push" else "calm"
    goal = ("care" if nouls.get("mission_care", 0) > 0.5
            else "speed" if nouls.get("mission_speed", 0) > 0.5 else "plain")
    key = f"{posture}|{slip}|{push}|{goal}"
    return f"{terrain_tag}|{key}" if terrain_tag else key


def risk_features(snap: dict, answers, mode: str, terrain: dict | None = None) -> np.ndarray:
    t = terrain or {}
    return np.array([
        1.0,
        snap["max_tilt_deg"] / 20.0,
        snap["max_gyro"] / 3.0,
        snap["max_slip"] / 0.3,
        float(snap["since_push_s"] < 1.5),
        (3.0 - answers.stability) / 3.0,
        answers.nouls.get("slipping", 0.0),
        answers.nouls.get("disturbed", 0.0),
        Q.MODE_ORDER.index(mode) / 3.0,
        t.get("rise_ahead_m", 0.0) / 0.1,          # step height within reach (0 on flat ground)
        float(t.get("edge_close", False)),
    ])


STAIR_GAITS = ("cautious", "normal", "stride")
GIVE_UP_TRIES = 2       # tries per gait before a step size can be declared too hard
GIVE_UP_RATE = 0.3      # ...and success rate below which it is


class StairsSkill:
    def __init__(self, stats: dict | None = None):
        self.stats = {k: dict(v) for k, v in (stats or {}).items()}   # size tag -> gait -> [success, fail]

    def _s(self, size):
        return self.stats.setdefault(size, {g: [0, 0] for g in STAIR_GAITS})

    def too_hard(self, size) -> bool:
        s = self._s(size)
        return all(sum(s[g]) >= GIVE_UP_TRIES and s[g][0] / sum(s[g]) < GIVE_UP_RATE for g in STAIR_GAITS)

    def choose(self, size, rng) -> str | None:
        """Gait to attempt these stairs with, or None if experience says they can't be climbed."""
        if self.too_hard(size):
            return None
        s = self._s(size)
        draws = {g: rng.beta(1 + s[g][0], 1 + s[g][1]) for g in STAIR_GAITS}
        return max(draws, key=draws.get)

    def record(self, size, gait, success: bool):
        self._s(size)[gait][0 if success else 1] += 1


class Learner:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.stairs = StairsSkill()
        self.stats = defaultdict(lambda: {m: [0, 0.0] for m in Q.MODE_ORDER})  # ctx -> mode -> [n, sum]
        self.w = None
        self.X, self.y = [], []
        if self.path and self.path.is_file():
            blob = json.loads(self.path.read_text())
            for k, v in blob["stats"].items():
                self.stats[k] = v
            self.stairs = StairsSkill(blob.get("stairs"))
            self.X, self.y = blob.get("X", []), blob.get("y", [])
            self.w = np.array(blob["w"]) if blob.get("w") else None
            if self.X and len(self.X[0]) != N_RISK:      # saved by an older feature layout: drop the risk data
                self.X, self.y, self.w = [], [], None

    # -- used at decision time ------------------------------------------------
    def bonus(self, ctx: str) -> dict:
        s = self.stats[ctx]
        n_tot = sum(n for n, _ in s.values())
        if n_tot == 0:
            return {m: 0.0 for m in s}
        base = sum(t for _, t in s.values()) / n_tot
        return {m: BONUS_SCALE * ((t / n - base) * n / (n + SHRINK) if n else 0.0) for m, (n, t) in s.items()}

    def fall_risk(self, snap: dict, answers, mode: str, terrain: dict | None = None) -> float | None:
        if self.w is None:
            return None
        z = float(risk_features(snap, answers, mode, terrain) @ self.w)
        return 1.0 / (1.0 + np.exp(-z))

    # -- used after an episode -------------------------------------------------
    def learn(self, decisions: list[dict]):
        for d in decisions:
            if d.get("outcome") is None or d["jev"].get("error"):   # no answer = nothing to learn from
                continue
            n_t = self.stats[d["context"]][d["mode"]]
            n_t[0] += 1
            n_t[1] += reward(d["outcome"])
            self.X.append(d["risk_x"])
            self.y.append(int(d["outcome"]["fell_soon"]))
        self._fit_risk()

    def _fit_risk(self, steps=800, lr=0.5, l2=1e-2):
        y = np.array(self.y, float)
        if y.sum() < MIN_CLASS or (1 - y).sum() < MIN_CLASS:
            return
        X = np.array(self.X, float)
        pos_w = (1 - y).sum() / y.sum()          # falls are rare: reweight them
        sw = np.where(y == 1, pos_w, 1.0)
        w = np.zeros(X.shape[1]) if self.w is None else self.w.copy()
        for _ in range(steps):
            p = 1 / (1 + np.exp(-X @ w))
            w -= lr * (X.T @ ((p - y) * sw) / sw.sum() + l2 * w)
        self.w = w

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "stats": self.stats, "stairs": self.stairs.stats, "X": self.X, "y": self.y,
                "w": None if self.w is None else self.w.tolist(),
            }))

    def summary(self) -> dict:
        return {ctx: {m: (n, round(t / n, 2) if n else None) for m, (n, t) in s.items()}
                for ctx, s in self.stats.items()}

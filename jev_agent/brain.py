"""The "System One" brain: state + typed questions -> typed answers.

JevBrain calls TypeSafe's Jev. OfflineBrain is a hand-written stand-in with the
same answer shape, for dry runs and tests without an API key; it is NOT Jev.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from jev_agent import questions as Q


@dataclass
class Answers:
    gait_probs: dict                 # mode -> probability
    gait_conf: float
    stability: float                 # expected score on 0..3
    stability_conf: float
    nouls: dict                      # name -> P(yes)
    choices: dict = field(default_factory=dict)   # other Choice questions: name -> {choice, confidence, probabilities}
    latency_s: float = 0.0
    model: str = ""
    error: str | None = None
    fallback: str | None = None      # set when OfflineBrain answered because Jev is unreachable
    raw: dict = field(default_factory=dict)

    @property
    def gait(self) -> str:
        return max(self.gait_probs, key=self.gait_probs.get)


REPO_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_api_key(env_file: str | Path = ".env") -> str | None:
    """Read the key from the environment or a .env file (accepts TYPESAFE_API_KEY or TYPE_SAFE_API_KEY).
    A variable set but empty disables Jev instead of falling through to .env."""
    names = ("TYPESAFE_API_KEY", "TYPE_SAFE_API_KEY")
    for n in names:
        if n in os.environ:
            return os.environ[n].strip() or None
    for p in (Path(env_file), REPO_ENV):   # cwd first, then the repo root
        if p.is_file():
            for line in p.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip().removeprefix("export ").strip() in names and v.strip():
                    return v.strip().strip("'\"")
    return None


def build_state(mission: str, description: dict, current_gait: str, terrain: str | None = None) -> dict:
    # Small, named state: only what the questions need (docs: concepts/state, jaggedness #5).
    state = {"mission": mission, "robot_now": description, "current_gait": current_gait}
    if terrain is not None:
        state["terrain_ahead"] = terrain
    return state


class JevBrain:
    # Jev latency varies a lot over a day (0.3 s .. 2+ s observed), so the per-request limit leaves room;
    # the live loop is asynchronous and the reflexes don't wait on it anyway.
    def __init__(self, model: str = Q.MODEL, timeout_s: float = 4.0, terrain: bool = False):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        key = load_api_key()
        if not key:
            raise RuntimeError("No TypeSafe key: set TYPESAFE_API_KEY or put TYPE_SAFE_API_KEY in .env")
        # Few, fast retries: a stale decision is worse than falling back to a safe default.
        # timeout= is the per-request limit; RetryPolicy.timeout is only the total retry budget.
        self.client = TypeSafeClient(api_key=key, model=model, timeout=timeout_s,
                                     retry=RetryPolicy(max_retries=1, backoff_max=0.2, timeout=timeout_s))
        self.questions = Q.questions(terrain)

    def ask(self, state: dict) -> Answers:
        t0 = time.perf_counter()
        r = self.client.system_one(state, self.questions)
        g, s = r.choices["gait_mode"], r.scores["stability"]
        return Answers(
            gait_probs=dict(g.probabilities), gait_conf=g.confidence,
            stability=s.score, stability_conf=s.confidence,
            nouls={k: v.noul for k, v in r.nouls.items()},
            choices={k: {"choice": v.choice, "confidence": v.confidence, "probabilities": dict(v.probabilities)}
                     for k, v in r.choices.items() if k != "gait_mode"},
            latency_s=time.perf_counter() - t0, model=r.model,
            raw=r.model_dump(mode="json")["answers"],
        )


class OfflineBrain:
    """Rule-based stand-in reading the same words Jev sees. For dry runs only."""

    def __init__(self, terrain: bool = False):
        self.terrain = terrain

    def ask(self, state: dict) -> Answers:
        d, mission = state["robot_now"], state["mission"].lower()
        bad = sum([
            d["torso_posture"].startswith(("clearly", "severely")) * 2,
            d["body_rotation"] in ("strong wobble", "spinning or tumbling"),
            d["foot_grip"].startswith(("feet slipping", "feet sliding")),
            d["recent_push"] != "no push",
        ])
        care = any(w in mission for w in ("careful", "fragile", "gently", "slow"))
        speed = any(w in mission for w in ("quick", "fast", "hurry", "brisk"))
        if bad >= 3:
            p = {"stop": .7, "cautious": .25, "normal": .05, "stride": 0}
        elif bad >= 1 or care:
            p = {"stop": .1, "cautious": .7, "normal": .2, "stride": 0}
        elif speed:
            p = {"stop": 0, "cautious": .05, "normal": .3, "stride": .65}
        else:
            p = {"stop": 0, "cautious": .15, "normal": .75, "stride": .1}
        n = len(p)
        conf = max(0.0, (n * max(p.values()) - 1) / (n - 1))
        choices, stairs_mission = {}, any(w in mission for w in ("stairs", "upstairs", "landing"))
        if self.terrain:
            terr = state.get("terrain_ahead", "")
            wait = any(w in mission for w in ("wait at", "stop at", "don't climb", "do not climb"))
            c = ("no_stairs" if "flat" in terr and "stairs" not in terr
                 else "climb" if stairs_mission and not wait else "stop_before")
            choices["stairs_action"] = {"choice": c, "confidence": 0.8,
                                        "probabilities": {k: (0.9 if k == c else 0.05) for k in ("no_stairs", "climb", "stop_before")}}
        return Answers(
            gait_probs=p, gait_conf=conf, stability=max(0.0, 3.0 - bad), stability_conf=0.8,
            nouls={"slipping": float("slipping" in d["foot_grip"] or "sliding" in d["foot_grip"]),
                   "disturbed": float(d["recent_push"] != "no push"),
                   "mission_speed": float(speed), "mission_care": float(care),
                   **({"mission_stairs": float(stairs_mission)} if self.terrain else {})},
            choices=choices, model="offline-stub",
        )


class FallbackBrain:
    """Jev first; after max_errors consecutive failures, answer from the offline stub and
    probe Jev again every probe_every_s or every probe_every_n questions, whichever comes first
    (fast offline simulation can run many decisions per wall-clock second). Errors before the switch
    still reach the supervisor."""

    def __init__(self, primary, backup, max_errors: int = 3, probe_every_s: float = 15.0, probe_every_n: int = 20):
        self.primary, self.backup = primary, backup
        self.max_errors, self.probe_every_s, self.probe_every_n = max_errors, probe_every_s, probe_every_n
        self.errors, self.last_error, self.next_probe, self.since_probe = 0, "", 0.0, 0

    @property
    def offline(self) -> bool:
        return self.errors >= self.max_errors

    def ask(self, state: dict) -> Answers:
        self.since_probe += 1
        if not self.offline or time.monotonic() >= self.next_probe or self.since_probe >= self.probe_every_n:
            self.since_probe = 0
            try:
                ans = self.primary.ask(state)
                self.errors = 0
                return ans
            except Exception as e:
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                self.next_probe = time.monotonic() + self.probe_every_s
                if not self.offline:
                    raise
                if self.errors == self.max_errors:
                    print(f"Jev unreachable after {self.errors} errors -> OFFLINE MODE ({self.last_error})", flush=True)
        ans = self.backup.ask(state)
        ans.model = "offline-stub (Jev unreachable)"
        ans.fallback = f"Jev unreachable: {self.last_error}"
        return ans


def make_brain(kind: str, terrain: bool = False, require_jev: bool = False):
    """'jev' -> Jev with automatic offline fallback; 'offline' -> the stub. Without a key, degrade to
    the stub (loudly) unless require_jev."""
    if kind == "offline":
        return OfflineBrain(terrain=terrain)
    try:
        jev = JevBrain(terrain=terrain)
    except RuntimeError as e:
        if require_jev:
            raise
        print(f"Jev unavailable ({e}) -> OFFLINE MODE", flush=True)
        fb = FallbackBrain(None, OfflineBrain(terrain=terrain), max_errors=0, probe_every_s=float("inf"),
                           probe_every_n=10**9)
        fb.last_error = str(e)
        return fb
    return FallbackBrain(jev, OfflineBrain(terrain=terrain))

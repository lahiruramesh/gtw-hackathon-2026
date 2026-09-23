"""Every Jev question and every threshold, in one place so they are easy to review.

TypeSafe's advice: keep questions atomic, keep numbers in code, and gate
actions on confidence. Edit this file to tune the agent's behaviour.
"""
from __future__ import annotations

from typesafe_sdk import Choice, Noul, Score

MODEL = "jev-1.13.0"   # pinned: thresholds below were set against this version

# Gait modes the agent can pick. Order = speed, slowest (safest) first.
# (forward speed m/s, step length m) -> fed to the PPO step-length policy.
# Cadence f = v / 2l stays inside the policy's trained 0.9–1.8 Hz band.
GAIT_MODES = {
    "stop":     (0.00, 0.00),
    "cautious": (0.30, 0.12),
    "normal":   (0.60, 0.25),
    "stride":   (0.90, 0.33),
}
MODE_ORDER = list(GAIT_MODES)

DECIDE_EVERY_S = 0.5     # supervisor rate (2 Hz); the PPO policy runs at 50 Hz underneath

# Confidence gating (see docs.typesafe.ai/confidence)
CONF_ACT = 0.35          # below this, don't change mode unless the change is to a safer one
CONF_UPSHIFT = 0.60      # speeding up needs a confident answer, and only one mode at a time
TOP_GAIT_CLEAN_WINDOWS = 3  # the fastest gait also needs this many clean decision windows in a row (1.5 s):
                            # on unknown ground you only find out it's slippery by slipping

# Hard safety filter, enforced in code whatever Jev says
SAFETY_TILT_DEG = 25.0   # torso tilt that forces "stop"
SAFETY_STABILITY = 0.75  # expected stability score (0..3) below which we force "stop"
RISK_VETO = 0.5          # learned fall probability above which a mode is vetoed

# Stairs (terrain-aware runs)
STAIRS_STOP_DIST_M = 0.7 # "stop before the stairs" is enforced once the first edge is this close
STAIRS_COMMIT_DIST_M = 1.4  # when climbing, the stairs gait is chosen (and held) from this distance (room to speed up)


def questions(terrain: bool = False) -> dict:
    """One request, several speculative questions (docs: patterns/fan-out)."""
    q = {
        "gait_mode": Choice(
            instructions="Which gait should the walking robot use for the next half second, "
                         "given its current body state and its mission?",
            criteria={
                "stop": "Stand still. Right when the robot is losing balance, is tumbling, "
                        "or the mission says to stop or wait.",
                "cautious": "Short, careful steps at low speed. Right when the feet are slipping, "
                            "the robot was just pushed, it is wobbling but still in control, "
                            "or the mission asks for care.",
                "normal": "Normal steps at moderate speed. Right when the robot is steady and "
                          "nothing unusual is happening.",
                "stride": "Long steps at high speed. Right only when the robot is fully steady, "
                          "the feet grip well, and the mission asks for speed.",
            },
        ),
        "stability": Score(
            instructions="How stable is the robot's balance right now?",
            criteria=[
                "Falling or about to fall",
                "Badly off balance, needs to recover",
                "Minor wobble, in control",
                "Completely steady",
            ],
        ),
        "slipping": Noul(instructions="The robot's feet are slipping on the ground."),
        "disturbed": Noul(instructions="Something external recently pushed or disturbed the robot."),
        "mission_speed": Noul(instructions="The mission asks the robot to move quickly."),
        "mission_care": Noul(instructions="The mission asks the robot to be careful or protect what it carries."),
    }
    if terrain:
        q["stairs_action"] = Choice(
            instructions="What should the robot do about the stairs described in terrain_ahead?",
            criteria={
                "no_stairs": "Keep walking. Right when terrain_ahead says the ground ahead is flat.",
                "climb": "Walk onto the stairs and keep going. Right when there are stairs ahead or underfoot "
                         "and the mission asks the robot to use the stairs or get past them.",
                "stop_before": "Stop before the first step. Right when there are stairs ahead and the mission "
                               "does not ask the robot to use them, or asks it to wait at the stairs.",
            },
        )
        q["mission_stairs"] = Noul(instructions="The mission asks the robot to go up or down the stairs.")
    return q

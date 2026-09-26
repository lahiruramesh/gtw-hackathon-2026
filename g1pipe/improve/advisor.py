"""Advisors: given the results so far, which experiments should the next round run?

An advisor only proposes. Keeping or dropping a change, promoting a champion, budgets and stopping
are decided in code (g1pipe.improve.stats, campaign.py), and every proposed change is validated
against the campaign's knob whitelist before it can become a run.

* rules    ranks the queued experiments by how far the conditions they aim at are from their targets.
* bedrock  a cheap model on Amazon Bedrock (Converse API, default Amazon Nova Lite) reads the results,
           writes an analysis for the round and may propose new one-change experiments.
* jev      TypeSafe's Jev picks among the queued experiments (a Choice question over their
           hypotheses, with the results described in words: Jev is weak at arithmetic).

bedrock and jev fall back to rules when they fail (no credentials, timeout, invalid answer); the round
records which advisor actually answered.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")


@dataclass
class Arm:
    name: str
    hypothesis: str
    set: dict[str, Any]
    aims: list[str] = field(default_factory=list)   # benchmark conditions it should improve
    source: str = "queue"

    def key(self) -> str:
        return json.dumps(self.set, sort_keys=True)

    def to_dict(self) -> dict:
        return {"name": self.name, "hypothesis": self.hypothesis, "set": self.set, "aims": self.aims, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> Arm:
        return cls(d["name"], d.get("hypothesis", ""), dict(d["set"]), list(d.get("aims", [])), d.get("source", "queue"))


@dataclass
class Plan:
    arms: list[Arm]
    analysis: str
    advisor: str                 # who actually answered
    raw: Any = None              # the advisor's raw answer, for the round record


class InvalidArm(ValueError):
    pass


def validate_arm(arm: Arm, knobs: dict[str, dict], targets: list[str]) -> Arm:
    """Raise InvalidArm unless every change is a whitelisted knob with a value in range."""
    if not SLUG.match(arm.name):
        raise InvalidArm(f"{arm.name!r}: name must be a lowercase slug")
    if not arm.set:
        raise InvalidArm(f"{arm.name}: no change")
    for key, value in arm.set.items():
        spec = knobs.get(key)
        if spec is None:
            raise InvalidArm(f"{arm.name}: {key} is not a whitelisted knob")
        kind, lo, hi = spec["type"], spec.get("min"), spec.get("max")
        if kind == "choice":
            if value not in spec["choices"]:
                raise InvalidArm(f"{arm.name}: {key}={value!r} not in {spec['choices']}")
            continue
        vals = value if kind == "range" else [value]
        if kind == "range" and (not isinstance(value, list) or len(value) != 2 or value[0] > value[1]):
            raise InvalidArm(f"{arm.name}: {key} must be [low, high]")
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InvalidArm(f"{arm.name}: {key}={value!r} is not a number")
            if kind == "int" and int(v) != v:
                raise InvalidArm(f"{arm.name}: {key}={v} is not an integer")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                raise InvalidArm(f"{arm.name}: {key}={v} outside [{lo}, {hi}]")
    arm.aims = [a for a in arm.aims if a in targets]
    return arm


# -- rules ---------------------------------------------------------------------------------------

def rule_rank(candidates: list[Arm], shortfall: dict[str, float]) -> list[Arm]:
    """Queue order, moved up by how far the conditions an arm aims at are from their targets."""
    def score(i_arm):
        i, arm = i_arm
        return (-sum(shortfall.get(c, 0.0) for c in arm.aims), i)
    return [a for _, a in sorted(enumerate(candidates), key=score)]


class RulesAdvisor:
    name = "rules"

    def plan(self, ctx: dict, candidates: list[Arm], k: int) -> Plan:
        ranked = rule_rank(candidates, ctx["shortfall"])[:k]
        worst = sorted(ctx["shortfall"].items(), key=lambda x: -x[1])
        far = ", ".join(f"{c} ({s*100:.1f} pts)" for c, s in worst if s > 0)
        lines = [f"Furthest from target: {far or 'not benchmarked yet' if not ctx['shortfall'] else far or 'none'}"]
        lines += [f"- {a.name}: {a.hypothesis}" for a in ranked]
        return Plan(ranked, "\n".join(lines), self.name)


# -- Amazon Bedrock (cheap model, Converse API) ----------------------------------------------------

PLAN_TOOL = {
    "name": "plan_round",
    "description": "Record the analysis of the results so far and the experiments for the next round.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "description": "What the results show and why, 3-8 sentences, for the engineers."},
            "ranking": {"type": "array", "items": {"type": "string"},
                        "description": "Names of the experiments to run next, best first (queued names or new ones)."},
            "new_arms": {"type": "array", "description": "New one-change experiments, only if the queue lacks a better idea.",
                         "items": {"type": "object", "properties": {
                             "name": {"type": "string", "description": "lowercase slug"},
                             "hypothesis": {"type": "string"},
                             "aims": {"type": "array", "items": {"type": "string"}},
                             "set": {"type": "object", "description": "knob -> value; one knob, or two that only make sense together"},
                         }, "required": ["name", "hypothesis", "set"]}},
        },
        "required": ["analysis", "ranking"],
    }},
}

SYSTEM = """You advise an automated reinforcement-learning improvement loop for a Unitree G1 humanoid \
that climbs stairs (MuJoCo Playground, Brax PPO, warm-started from the current champion policy).
Each round trains a control (the champion's recipe, unchanged) and a few arms that each change one \
training knob, for the same number of steps, then scores every policy on a fixed benchmark: fall \
rates under 7 conditions. Code, not you, decides whether an arm helped (statistics against the \
control) and which policy becomes champion. Your job: explain what the results show, rank the \
queued experiments for the next round, and propose new one-change experiments only when the queue \
has no good idea for the conditions furthest from target. Only use knobs from the whitelist, with \
values inside their ranges. Do not repeat a change that was already tried unless a result was \
inconclusive and you say why. Always answer by calling the plan_round tool."""


class BedrockAdvisor:
    name = "bedrock"

    def __init__(self, model_id: str = "us.amazon.nova-lite-v1:0", region: str = "us-east-1",
                 profile: str | None = None, max_new_arms: int = 3, max_tokens: int = 2000, client=None):
        self.model_id, self.max_new_arms, self.max_tokens = model_id, max_new_arms, max_tokens
        if client is None:
            import boto3
            session = boto3.Session(profile_name=profile, region_name=region)
            client = session.client("bedrock-runtime")
        self.client = client

    def plan(self, ctx: dict, candidates: list[Arm], k: int) -> Plan:
        prompt = json.dumps({
            "targets": ctx["targets"], "champion": ctx["champion"], "recipe": ctx["recipe"],
            "history": ctx["history"][-24:], "queue": [a.to_dict() for a in candidates],
            "knob_whitelist": ctx["knobs"], "arms_this_round": k, "max_new_arms": self.max_new_arms,
            "notes": ctx.get("notes", []),
        }, indent=1)
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig={"tools": [{"toolSpec": PLAN_TOOL}], "toolChoice": {"any": {}}},
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.2},
        )
        blocks = resp["output"]["message"]["content"]
        call = next((b["toolUse"]["input"] for b in blocks if "toolUse" in b), None)
        if call is None:
            raise ValueError(f"{self.model_id} did not call plan_round (stop reason {resp.get('stopReason')})")
        by_name = {a.name: a for a in candidates}
        tried = set(ctx.get("tried_keys", []))
        rejected = []
        for d in (call.get("new_arms") or [])[: self.max_new_arms]:
            try:
                arm = validate_arm(Arm(str(d.get("name", "")), str(d.get("hypothesis", "")), dict(d.get("set") or {}),
                                       list(d.get("aims") or []), source=f"bedrock:{self.model_id}"),
                                   ctx["knobs"], list(ctx["targets"]))
            except InvalidArm as e:
                rejected.append(str(e))
                continue
            if arm.key() in tried or arm.name in by_name:
                rejected.append(f"{arm.name}: already queued or tried")
                continue
            by_name[arm.name] = arm
        picked = [by_name[n] for n in dict.fromkeys(call.get("ranking", [])) if n in by_name]   # dedupe, keep order
        fill = [a for a in rule_rank(list(by_name.values()), ctx["shortfall"]) if a not in picked]
        arms = (picked + fill)[:k]
        analysis = str(call.get("analysis", "")).strip()
        if rejected:
            analysis += "\n\nRejected proposals: " + "; ".join(rejected)
        usage = resp.get("usage", {})
        return Plan(arms, analysis, f"bedrock:{self.model_id}",
                    raw={"call": call, "usage": usage, "new_arms": [a.to_dict() for a in by_name.values() if a.source != "queue"]})


# -- TypeSafe Jev ---------------------------------------------------------------------------------

class JevAdvisor:
    name = "jev"

    def __init__(self, model: str = "jev-1.13.0", timeout_s: float = 20.0):
        from typesafe_sdk import TypeSafeClient
        from jev_agent.brain import load_api_key
        key = load_api_key()
        if not key:
            raise RuntimeError("No TypeSafe key: set TYPESAFE_API_KEY")
        self.client = TypeSafeClient(api_key=key, model=model, timeout=timeout_s)

    def plan(self, ctx: dict, candidates: list[Arm], k: int) -> Plan:
        from typesafe_sdk import Choice
        pool = rule_rank(candidates, ctx["shortfall"])[:8]   # a short, pre-ranked menu
        if len(pool) <= k:
            return Plan(pool, "Jev not asked: no more candidates than slots.", "rules")
        state = {"goal": "a stair-climbing humanoid policy that meets every benchmark target",
                 "benchmark_now": ctx["champion"]["in_words"],
                 "recent_results": [f"{h['arm']}: {h['verdict_words']}" for h in ctx["history"][-10:]]}
        q = {"next": Choice(instructions="Which one-change training experiment is most likely to move the "
                                         "conditions furthest from target, given the results so far?",
                            criteria={a.name: a.hypothesis for a in pool})}
        r = self.client.system_one(state, q)
        probs = dict(r.choices["next"].probabilities)
        ranked = sorted(pool, key=lambda a: -probs.get(a.name, 0.0))[:k]
        analysis = "Jev's preference: " + ", ".join(f"{a.name} {probs.get(a.name, 0):.2f}" for a in ranked) + \
                   f" (confidence {r.choices['next'].confidence:.2f})"
        return Plan(ranked, analysis, f"jev:{r.model}", raw={"probabilities": probs})


def make_advisor(cfg: dict):
    kind = (cfg or {}).get("kind", "rules")
    if kind == "bedrock":
        b = cfg.get("bedrock", {})
        return BedrockAdvisor(b.get("model_id", "us.amazon.nova-lite-v1:0"), b.get("region", "us-east-1"),
                              b.get("profile"), int(b.get("max_new_arms", 3)))
    if kind == "jev":
        return JevAdvisor(cfg.get("jev", {}).get("model", "jev-1.13.0"))
    return RulesAdvisor()


def plan_with_fallback(advisor, ctx: dict, candidates: list[Arm], k: int, log=print) -> Plan:
    if not isinstance(advisor, RulesAdvisor):
        try:
            p = advisor.plan(ctx, candidates, k)
            if p.arms:
                return p
            log(f"advisor {advisor.name} proposed nothing; using rules")
        except Exception as e:   # the loop must not stop because an advisor is down
            log(f"advisor {advisor.name} failed ({type(e).__name__}: {e}); using rules")
    return RulesAdvisor().plan(ctx, candidates, k)

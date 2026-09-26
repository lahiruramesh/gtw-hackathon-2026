"""Tests for the improvement loop (g1pipe.improve) and the g1-stairs-bench skill summarizer.
No JAX, no network: the studio is simulated and Bedrock is a fake client.

    uv run pytest tests/test_improve.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from g1pipe.improve import stats
from g1pipe.improve.advisor import Arm, BedrockAdvisor, InvalidArm, RulesAdvisor, plan_with_fallback, validate_arm
from g1pipe.improve.campaign import Campaign, dry_run_studio, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "stairs" / "campaign.yaml"
TARGETS = {"strict": 0.01, "camera": 0.02, "delay_20ms": 0.02, "low_friction": 0.05,
           "payload_5kg": 0.02, "push": 0.02, "speed_06": 0.02}


def summary(rates: dict[str, float], n: int = 1280) -> dict:
    out = {}
    for k in TARGETS:
        fell = round(rates.get(k, 0.0) * n)
        out[k] = {"n": n, "fell": fell, "rate": fell / n, "rate_hi": stats_hi(fell, n)}
    return out


def stats_hi(k, n):
    return load_summarizer().wilson(k, n)[1]


def load_summarizer():
    spec = importlib.util.spec_from_file_location("bench_summarize", ROOT / "skills" / "stairs_bench" / "summarize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = {"strict": 0.008, "camera": 0.07, "delay_20ms": 0.45, "low_friction": 0.05, "payload_5kg": 0.11,
        "push": 0.01, "speed_06": 0.005}


# -- stats ------------------------------------------------------------------------------------------

def test_gap_and_pass():
    s = summary(BASE)
    assert stats.gap(s, TARGETS) == pytest.approx((0.07 - 0.02) + (0.45 - 0.02) + (0.11 - 0.02), abs=2e-3)
    assert not stats.passes(s, TARGETS)
    good = summary({k: 0.0 for k in TARGETS})
    assert stats.gap(good, TARGETS) == 0 and stats.passes(good, TARGETS)


def test_targets_from_gate_reads_only_bench_bounds():
    gate = [{"metric": "bench.delay_20ms.rate_hi", "op": "<=", "value": 0.02},
            {"metric": "evaluate.fell", "op": "<=", "value": 5},
            {"metric": "bench.strict.rate", "op": "<=", "value": 0.5}]
    assert stats.targets_from_gate(gate) == {"delay_20ms": 0.02}


def test_compare_keeps_a_large_gain_despite_noise_elsewhere():
    ctrl = summary(BASE)
    arm = summary({**BASE, "payload_5kg": 0.025, "delay_20ms": 0.49})   # delay moved within seed noise
    noise = {"gap": 0.06, "cond": {**{k: 0.005 for k in TARGETS}, "delay_20ms": 0.04, "payload_5kg": 0.012}}
    v = stats.compare(arm, ctrl, TARGETS, noise=noise)
    assert v.better and v.improved == ["payload_5kg"] and not v.regressions


def test_compare_rejects_a_significant_regression():
    ctrl = summary(BASE)
    arm = summary({**BASE, "payload_5kg": 0.02, "strict": 0.06})
    v = stats.compare(arm, ctrl, TARGETS, noise={"gap": 0.0, "cond": {k: 0.004 for k in TARGETS}})
    assert not v.better and v.regressions == ["strict"]


def test_compare_ignores_gains_on_conditions_already_on_target():
    ctrl = summary(BASE)
    arm = summary({**BASE, "push": 0.0})   # push is under its 2 % target in both
    assert not stats.compare(arm, ctrl, TARGETS).better


def test_seed_noise_pools_as_rms():
    a, b = summary(BASE), summary({**BASE, "delay_20ms": 0.41})
    n1 = stats.seed_noise(a, b, TARGETS)
    pooled = stats.pool_noise([n1, n1])
    assert pooled["cond"]["delay_20ms"] == pytest.approx(0.04, abs=1e-3) and pooled["samples"] == 2


# -- advisors ---------------------------------------------------------------------------------------

KNOBS = yaml.safe_load(CONFIG.read_text())["knobs"]


def test_validate_arm_whitelist_and_ranges():
    ok = validate_arm(Arm("delay-share-70", "h", {"env.action_delay_p": 0.7}, ["delay_20ms", "nope"]), KNOBS, list(TARGETS))
    assert ok.aims == ["delay_20ms"]
    for bad in [Arm("x-y", "h", {"env.action_delay_p": 1.5}), Arm("x-y", "h", {"env.secret": 1}),
                Arm("Bad Name", "h", {"env.action_delay_p": 0.5}), Arm("x-y", "h", {"env.payload_kg": [4, 1]}),
                Arm("x-y", "h", {"env.scan_model": "lidar"}), Arm("x-y", "h", {"env.action_delay_max": 1.5}),
                Arm("x-y", "h", {})]:
        with pytest.raises(InvalidArm):
            validate_arm(bad, KNOBS, list(TARGETS))


class FakeBedrock:
    def __init__(self, tool_input=None, fail=False):
        self.tool_input, self.fail, self.calls = tool_input, fail, []

    def converse(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("AccessDeniedException")
        return {"output": {"message": {"content": [{"toolUse": {"name": "plan_round", "input": self.tool_input}}]}},
                "stopReason": "tool_use", "usage": {"inputTokens": 900, "outputTokens": 200}}


def ctx(shortfall=None):
    return {"targets": TARGETS, "champion": {"label": "v14"}, "recipe": {}, "history": [], "knobs": KNOBS,
            "shortfall": shortfall or {"delay_20ms": 0.4, "payload_5kg": 0.09}, "tried_keys": [
                json.dumps({"env.action_delay_p": 0.7})]}


QUEUE = [Arm("lr-2e-4", "h", {"ppo.learning_rate": 0.0002}),
         Arm("payload-0-4kg", "h", {"env.payload_kg": [0.0, 4.0]}, ["payload_5kg"])]


def test_bedrock_ranking_new_arms_and_validation():
    fake = FakeBedrock({"analysis": "Delay dominates.", "ranking": ["delay-40ms", "payload-0-4kg", "unknown"],
                        "new_arms": [{"name": "delay-40ms", "hypothesis": "margin", "set": {"env.action_delay_max": 2}},
                                     {"name": "too-far", "hypothesis": "x", "set": {"env.action_delay_p": 3}},
                                     {"name": "again", "hypothesis": "x", "set": {"env.action_delay_p": 0.7}}]})
    plan = BedrockAdvisor(model_id="us.amazon.nova-lite-v1:0", client=fake).plan(ctx(), list(QUEUE), 3)
    assert [a.name for a in plan.arms] == ["delay-40ms", "payload-0-4kg", "lr-2e-4"]
    assert plan.arms[0].source.startswith("bedrock:")
    assert "too-far" in plan.analysis and "again: already queued or tried" in plan.analysis
    req = fake.calls[0]
    assert req["modelId"] == "us.amazon.nova-lite-v1:0" and req["toolConfig"]["toolChoice"] == {"any": {}}


def test_advisor_failure_falls_back_to_rules():
    logs = []
    plan = plan_with_fallback(BedrockAdvisor(client=FakeBedrock(fail=True)), ctx(), list(QUEUE), 2, log=logs.append)
    assert plan.advisor == "rules" and plan.arms[0].name == "payload-0-4kg"   # aims at a condition off target
    assert "AccessDeniedException" in logs[0]


def test_rules_rank_by_shortfall():
    plan = RulesAdvisor().plan(ctx(), list(QUEUE), 1)
    assert [a.name for a in plan.arms] == ["payload-0-4kg"]


# -- summarizer and skill ---------------------------------------------------------------------------

def test_summarizer_thresholds_match_the_skill_gate():
    skill = yaml.safe_load((ROOT / "skills" / "stairs_bench" / "skill.yaml").read_text())
    assert stats.targets_from_gate(skill["gate"]) == load_summarizer().THRESHOLDS


def test_summarizer_reads_bench_json(tmp_path):
    levels = [(3.0, 256, 0), (6.4, 256, 2), (9.9, 256, 3), (13.3, 256, 40)]
    cond = {"episodes": 1024, "per_level": [{"level": i, "rise_cm": r, "n": n, "fell": f, "crossed": n - f}
                                            for i, (r, n, f) in enumerate(levels)]}
    (tmp_path / "bench.json").write_text(json.dumps({"p.pkl": {"key": "v1", "conditions": {"strict": cond}}}))
    s = load_summarizer().summarize(tmp_path)
    assert s["strict"]["n"] == 768 and s["strict"]["fell"] == 5 and s["strict"]["fell_all"] == 45
    assert s["strict"]["rate_hi"] > s["strict"]["rate"] and s["bench_key"] == "v1"


# -- campaign (simulated studio) --------------------------------------------------------------------

def dry_cfg(tmp_path, **budget):
    cfg = load_config(CONFIG)
    cfg["state_dir"] = str(tmp_path / "state")
    cfg["champion"]["checkpoint_id"] = "ckpt-champion"
    cfg["advisor"] = {"kind": "rules"}
    cfg["poll_s"] = 0
    cfg["budget"].update(budget)
    return cfg


def test_campaign_finds_the_real_effects(tmp_path):
    cfg = dry_cfg(tmp_path, daily_gpu_hours=1000)
    camp = Campaign(cfg, dry_run_studio(cfg, seed=0), RulesAdvisor(), log=lambda m: None, sleep=lambda s: None)
    status = camp.run()
    s = camp.state
    assert status.startswith("stopped: queue exhausted") or status.startswith(("done", "stopped: max_rounds"))
    assert s["recipe"]["overrides"]["env.action_delay_p"] == 0.7          # the big simulated effects were kept
    assert s["recipe"]["overrides"]["env.payload_kg"] == [0.0, 4.0]
    first = s["rounds"][0]
    assert set(first["runs"]) >= {"control", "control-s1"} and s["noise"]["samples"] == 1
    assert s["champion"]["gap"] < stats.gap(first["runs"]["control"]["summary"], TARGETS) / 3
    report = (tmp_path / "state" / "report.md").read_text()
    assert "## Round 1" in report and "kept" in report
    events = [json.loads(line)["type"] for line in (tmp_path / "state" / "events.jsonl").read_text().splitlines()]
    assert events[0] == "round_planned" and "champion" in events and events[-1] == "campaign_stopped"


def test_campaign_stops_at_the_daily_budget(tmp_path):
    cfg = dry_cfg(tmp_path, daily_gpu_hours=3.0)   # a round of 5 runs x 0.8 h does not fit, 3 runs do
    camp = Campaign(cfg, dry_run_studio(cfg, seed=0), RulesAdvisor(), log=lambda m: None, sleep=lambda s: None)
    status = camp.run()
    assert status.startswith("stopped: daily GPU budget")
    assert len(camp.state["rounds"][0]["runs"]) == 3   # trimmed to control, second seed, one arm


def test_campaign_resumes_an_interrupted_round(tmp_path):
    cfg = dry_cfg(tmp_path, daily_gpu_hours=1000)
    studio = dry_run_studio(cfg, seed=1)
    camp = Campaign(cfg, studio, RulesAdvisor(), log=lambda m: None, sleep=lambda s: None)
    camp._start_round(1)                       # launched, then the process "dies"
    camp.save()
    again = Campaign(cfg, studio, RulesAdvisor(), log=lambda m: None, sleep=lambda s: None)
    assert again.state["rounds"][0]["phase"] == "running"
    assert again.step() is None                 # waits for the same runs and decides; no duplicate launch
    assert again.state["rounds"][0]["phase"] == "decided" and len(studio.runs) == 5

"""Improvement campaign: rounds of experiments through the SKF Skill Studio until the release gate passes.

Each round:
  1. budget   estimate the round on the compute target; stop (or wait for tomorrow) when it would take
              the day's GPU hours on that target, across all users, past `budget.daily_gpu_hours`
  2. plan     the advisor (rules / cheap Bedrock model / Jev) ranks the queued one-change experiments,
              may propose new ones (validated against the knob whitelist) and writes an analysis
  3. launch   a control run (the champion's recipe, unchanged) and one run per arm, all warm-started
              from the champion checkpoint and trained for the same steps; round 1 (every round with
              `seed_controls: every`) adds a second control with another seed to measure how much two
              trainings of the same recipe differ
  4. wait     until every run is finished (runs over the approval budget wait for a person to approve
              the launch in the web app)
  5. decide   in code (g1pipe.improve.stats): an arm is kept when its benchmark is closer to the
              targets than the control's by more than evaluation noise and seed noise, with no
              condition significantly worse; the best kept change joins the recipe; an inconclusive
              arm gets one retry on the new recipe; among the control and the kept arms, the policy
              closest to the targets becomes champion when it beats the current one
  6. record   state.json, report.md, events.jsonl under the campaign's state dir; optional webhook

It stops when the champion's run passes the whole release gate (then a person reviews the release in
the web app), when the queue is exhausted, or at the round / GPU-hour limits.

    python -m g1pipe.improve.campaign run experiments/stairs/campaign.yaml            # SKF_URL/EMAIL/PASSWORD set
    python -m g1pipe.improve.campaign run experiments/stairs/campaign.yaml --dry-run  # simulated studio
    python -m g1pipe.improve.campaign status experiments/stairs/campaign.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from g1pipe.improve import stats
from g1pipe.improve.advisor import Arm, InvalidArm, make_advisor, plan_with_fallback, validate_arm
from g1pipe.improve.studio import TERMINAL, FakeStudio, Studio, StudioError

PASSED = {"awaiting_review", "approved"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("state_dir", str(Path(path).with_suffix("")))   # experiments/stairs/campaign/
    cfg.setdefault("run_prefix", cfg["name"])
    cfg.setdefault("arms_per_round", 3)
    cfg.setdefault("seed_controls", "first")   # first | every | never: rounds with a second-seed control
    cfg.setdefault("retry_inconclusive", 1)
    cfg.setdefault("poll_s", 120)
    cfg.setdefault("advisor", {"kind": "rules"})
    b = cfg.setdefault("budget", {})
    b.setdefault("daily_gpu_hours", 10.0)
    b.setdefault("campaign_gpu_hours", 40.0)
    b.setdefault("max_rounds", 8)
    b.setdefault("on_daily_cap", "stop")
    b.setdefault("max_round_hours", 24.0)
    return cfg


class Campaign:
    def __init__(self, cfg: dict, studio, advisor, log=None, sleep=time.sleep):
        self.cfg, self.studio, self.advisor, self.sleep = cfg, studio, advisor, sleep
        self.dir = Path(cfg["state_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = log or (lambda m: print(f"[{dt.datetime.now():%H:%M}] {m}", flush=True))
        skill = studio.skill(cfg["skill_id"])
        self.targets = stats.targets_from_gate(skill["gate"])
        if not self.targets:
            raise SystemExit(f"skill {cfg['skill_id']} has no bench.<condition>.rate_hi gate criteria")
        self.knobs = cfg["knobs"]
        self.state = self._load_state()
        self.target_id = self._target_id(cfg["compute_target"])

    # -- state ----------------------------------------------------------------------------------
    def _load_state(self) -> dict:
        f = self.dir / "state.json"
        if f.exists():
            return json.loads(f.read_text())
        c = self.cfg
        queue = []
        for d in c.get("queue", []):
            try:
                queue.append(validate_arm(Arm.from_dict(d), c["knobs"], list(self.targets)).to_dict())
            except InvalidArm as e:
                raise SystemExit(f"campaign queue: {e}")
        return {"campaign": c["name"], "created": now(), "status": "running", "rounds": [], "tried": [],
                "queue": queue, "noise_samples": [], "noise": None, "gpu_hours": 0.0,
                "recipe": {"params": dict(c["recipe"]["params"]), "overrides": dict(c["recipe"].get("overrides") or {})},
                "champion": {"label": c["champion"]["label"], "run_id": c["champion"].get("run_id"),
                             "checkpoint_id": c["champion"]["checkpoint_id"], "summary": None, "gap": None}}

    def save(self) -> None:
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.dir / "state.json")
        (self.dir / "report.md").write_text(self.report())

    def event(self, kind: str, **data) -> None:
        e = {"ts": now(), "campaign": self.cfg["name"], "type": kind, **data}
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(e) + "\n")
        url = (self.cfg.get("notify") or {}).get("webhook_url") or os.environ.get("IMPROVE_WEBHOOK_URL")
        if url:
            try:
                req = urllib.request.Request(url, data=json.dumps(e).encode(), method="POST",
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).close()
            except Exception as ex:   # notifications never stop the loop
                self._log(f"webhook failed: {ex}")

    def log(self, msg: str) -> None:
        self._log(msg)

    # -- helpers --------------------------------------------------------------------------------
    def _target_id(self, name_or_id: str) -> str:
        for t in self.studio.targets():
            if name_or_id in (t["id"], t["name"]):
                return t["id"]
        raise SystemExit(f"compute target {name_or_id!r} not found in the web app")

    def _params(self, overrides: dict, seed: int) -> dict:
        p = dict(self.state["recipe"]["params"], seed=seed)
        p["overrides"] = json.dumps(overrides, sort_keys=True) if overrides else None
        return p

    def _shortfall(self) -> dict[str, float]:
        s = self.state["champion"]["summary"]
        if not s:
            return {}
        return {k: max(0.0, s[k]["rate"] - thr) for k, thr in self.targets.items()}

    def _day_gpu_hours(self) -> float:
        start = dt.datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        runs = self.studio.runs_since(start.astimezone(dt.timezone.utc))
        return sum(r.get("gpu_hours") or 0.0 for r in runs
                   if (r.get("compute_target") or {}).get("id", self.target_id) == self.target_id)

    def _checkpoint_of(self, run_id: str) -> str:
        arts = self.studio.artifacts(run_id)
        a = next((a for a in arts if a["name"] == "params.pkl"), None)
        if a is None:
            raise RuntimeError(f"run {run_id} has no params.pkl artifact")
        return a["id"]

    def _bench(self, run_id: str) -> dict | None:
        ev = [e for e in self.studio.evaluations(run_id) if e["stage_key"] == "bench"]
        return ev[-1]["summary"] if ev else None

    # -- the loop -------------------------------------------------------------------------------
    def run(self, once: bool = False) -> str:
        while True:
            reason = self.step()
            self.save()
            if reason:
                self.state["status"] = reason if reason.startswith(("done", "stopped", "waiting")) else f"stopped: {reason}"
                self.save()
                self.event("campaign_stopped", reason=self.state["status"])
                self.log(self.state["status"])
                if reason.startswith("waiting"):
                    self._sleep_until_tomorrow()
                    self.state["status"] = "running"
                    continue
                return self.state["status"]
            if once:
                return "running"

    def _sleep_until_tomorrow(self) -> None:
        t = dt.datetime.now().astimezone()
        wake = (t + dt.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        self.log(f"daily GPU budget used; sleeping until {wake:%Y-%m-%d %H:%M}")
        self.sleep((wake - t).total_seconds())

    def step(self) -> str | None:
        """One round (or the rest of an interrupted one). Returns a stop reason, or None to go on."""
        rounds = self.state["rounds"]
        if rounds and rounds[-1]["phase"] != "decided":
            return self._finish_round(rounds[-1])
        b = self.cfg["budget"]
        if len(rounds) >= b["max_rounds"]:
            return f"max_rounds ({b['max_rounds']}) reached"
        if self.state["gpu_hours"] >= b["campaign_gpu_hours"]:
            return f"campaign GPU budget ({b['campaign_gpu_hours']} h) used"
        return self._start_round(len(rounds) + 1)

    def _start_round(self, n: int) -> str | None:
        k = self.cfg["arms_per_round"]
        sc = self.cfg["seed_controls"]
        calibrate = sc == "every" or (sc == "first" and not self.state["noise_samples"])
        tried = {json.dumps(t["set"], sort_keys=True) for t in self.state["tried"]}
        retried = {t["arm"] for t in self.state["tried"]}
        # retries and stack tests repeat a tried change on a newer recipe: they are matched by name
        again = ("retry", "stack")
        candidates = [Arm.from_dict(d) for d in self.state["queue"]
                      if (d.get("source") in again and d["name"] not in retried)
                      or (d.get("source") not in again and json.dumps(d["set"], sort_keys=True) not in tried)]
        champ = self.state["champion"]
        ctx = {"targets": self.targets, "recipe": self.state["recipe"], "knobs": self.knobs,
               "shortfall": self._shortfall(), "tried_keys": sorted(tried), "notes": self.cfg.get("notes", []),
               "champion": {"label": champ["label"], "gap": champ["gap"],
                            "rates": {c: champ["summary"][c]["rate"] for c in self.targets} if champ["summary"] else None,
                            "in_words": stats.describe(champ["summary"], self.targets) if champ["summary"] else "not benchmarked yet"},
               "history": [dict(t, verdict_words=t["reason"]) for t in self.state["tried"]]}
        plan = plan_with_fallback(self.advisor, ctx, candidates, k, log=self.log)
        if not plan.arms:
            return "queue exhausted: no experiment left to try (add ideas to the campaign queue or enable an advisor that proposes them)"
        for a in plan.arms:   # new arms from the advisor join the queue record
            if all(d["name"] != a.name for d in self.state["queue"]):
                self.state["queue"].append(a.to_dict())

        roles = [("control", None, 0)] + ([("control-s1", None, 1)] if calibrate else []) + [(a.name, a, 0) for a in plan.arms]
        # budget: what the round needs against what is left today on this target
        est = [self.studio.estimate({"skill_id": self.cfg["skill_id"], "compute_target_id": self.target_id,
                                     "params": self._params({**self.state["recipe"]["overrides"], **(a.set if a else {})}, seed),
                                     "parent_checkpoint_id": champ["checkpoint_id"]}) for _, a, seed in roles]
        blockers = [x for e in est for x in e.get("blockers", [])]
        if blockers:
            return f"the web app refuses the launch: {blockers[0]}"
        used = self._day_gpu_hours()
        cap = self.cfg["budget"]["daily_gpu_hours"]
        need = [e["gpu_hours"] for e in est]
        while roles and used + sum(need) > cap and len(roles) > (3 if calibrate else 2):
            roles.pop(), need.pop()   # drop the lowest-ranked arm to fit
        if used + sum(need) > cap:
            msg = f"daily GPU budget: {used:.1f} h used today, the round needs {sum(need):.1f} h, cap {cap} h"
            if self.cfg["budget"]["on_daily_cap"] == "wait":
                return f"waiting: {msg}"
            return msg
        if len(roles) < len(est):
            self.log(f"round {n}: only {len(roles)} runs fit today's budget")

        rnd = {"n": n, "phase": "launching", "started": now(), "advisor": plan.advisor, "analysis": plan.analysis,
               "advisor_raw": plan.raw, "champion": champ["label"], "recipe": json.loads(json.dumps(self.state["recipe"])),
               "planned": [[role, arm.to_dict() if arm else None, seed] for role, arm, seed in roles], "runs": {}}
        self.state["rounds"].append(rnd)
        self.save()
        self.event("round_planned", round=n, advisor=plan.advisor, arms=[r for r, _, _ in roles], analysis=plan.analysis)
        self.log(f"round {n} ({plan.advisor}): {', '.join(r for r, _, _ in roles)}")
        for role, arm, seed in roles:
            self._launch(rnd, role, arm, seed)
        rnd["phase"] = "running"
        self.save()
        return None

    def _launch(self, rnd: dict, role: str, arm: Arm | None, seed: int) -> None:
        name = f"{self.cfg['run_prefix']}-r{rnd['n']:02d}-{role}"
        change = arm.set if arm else {}
        overrides = {**self.state["recipe"]["overrides"], **change}
        champ = self.state["champion"]
        notes = (f"Improvement campaign '{self.cfg['name']}', round {rnd['n']}, {role}. "
                 + (f"Hypothesis: {arm.hypothesis} Change: {json.dumps(change)}. " if arm else
                    f"Control: the champion's recipe unchanged{' (second seed, for the noise floor)' if seed else ''}. ")
                 + f"Warm start: champion {champ['label']}. Compared against {self.cfg['run_prefix']}-r{rnd['n']:02d}-control.")
        try:
            run = self.studio.create_run({"skill_id": self.cfg["skill_id"], "compute_target_id": self.target_id,
                                          "name": name, "notes": notes, "params": self._params(overrides, seed),
                                          "parent_checkpoint_id": champ["checkpoint_id"]})
        except StudioError as e:
            if e.status != 409:
                raise
            run = self.studio.find_run(name)   # launched before an interruption
        rnd["runs"][role] = {"name": name, "run_id": run["id"], "status": run["status"], "set": change,
                             "hypothesis": arm.hypothesis if arm else "control", "seed": seed}
        self.save()
        self.event("run_launched", round=rnd["n"], role=role, run_id=run["id"], name=name, change=change)

    def _finish_round(self, rnd: dict) -> str | None:
        if rnd["phase"] == "launching":   # interrupted while launching: launch what is missing, then go on
            for role, arm, seed in rnd["planned"]:
                if role not in rnd["runs"]:
                    self._launch(rnd, role, Arm.from_dict(arm) if arm else None, seed)
            rnd["phase"] = "running"
        deadline = dt.datetime.fromisoformat(rnd["started"]) + dt.timedelta(hours=self.cfg["budget"]["max_round_hours"])
        told_approval = False
        while True:
            pending = []
            for role, r in rnd["runs"].items():
                if r["status"] in TERMINAL:
                    continue
                cur = self.studio.run(r["run_id"])
                if cur["status"] != r["status"]:
                    self.event("run_status", round=rnd["n"], role=role, run_id=r["run_id"], status=cur["status"])
                r["status"], r["gpu_hours"] = cur["status"], cur.get("gpu_hours", 0.0)
                if cur["status"] not in TERMINAL:
                    pending.append(role)
                if cur["status"] == "pending_approval" and not told_approval:
                    self.log(f"round {rnd['n']}: {r['name']} waits for launch approval in the web app")
                    told_approval = True
            self.save()
            if not pending:
                break
            if dt.datetime.now(dt.timezone.utc) > deadline:
                return f"round {rnd['n']} still running after {self.cfg['budget']['max_round_hours']} h: {', '.join(pending)}"
            self.sleep(self.cfg["poll_s"])
        return self._decide(rnd)

    def _decide(self, rnd: dict) -> str | None:
        runs = rnd["runs"]
        self.state["gpu_hours"] += sum(r.get("gpu_hours") or 0.0 for r in runs.values())
        for r in runs.values():
            r["summary"] = self._bench(r["run_id"]) if r["status"] not in ("failed", "cancelled") else None
            r["gap"] = stats.gap(r["summary"], self.targets) if r["summary"] else None
        ctrl = runs["control"].get("summary")
        decisions: dict[str, Any] = {}
        if ctrl is None:
            rnd["phase"] = "decided"
            rnd["decisions"] = {"error": f"control run {runs['control']['name']} has no benchmark ({runs['control']['status']})"}
            self.event("round_failed", round=rnd["n"], reason=rnd["decisions"]["error"])
            failed = sum(1 for x in self.state["rounds"][-2:] if "error" in x.get("decisions", {}))
            return "two rounds in a row without a control benchmark: check the runs in the web app" if failed >= 2 else None
        if "control-s1" in runs and runs["control-s1"].get("summary"):
            self.state["noise_samples"].append(stats.seed_noise(ctrl, runs["control-s1"]["summary"], self.targets))
            self.state["noise"] = stats.pool_noise(self.state["noise_samples"])
            decisions["seed_noise"] = self.state["noise"]
        noise = self.state["noise"]

        winners = []
        for role, r in runs.items():
            if role.startswith("control"):
                continue
            if r["summary"] is None:
                verdict, v = "failed", None
            else:
                v = stats.compare(r["summary"], ctrl, self.targets, noise=noise)
                verdict = "kept" if v.better else ("worse" if v.regressions else "inconclusive")
                if v.better:
                    winners.append((v.improvement, role, r))
            reason = v.reason if v else f"run {r['status']}"
            decisions[role] = {"verdict": verdict, "reason": reason, "improvement": v.improvement if v else None,
                               "z": v.z if v else None}
            self.state["tried"].append({"round": rnd["n"], "arm": role, "set": r["set"], "verdict": verdict,
                                        "reason": reason, "run_id": r["run_id"]})
            if verdict == "inconclusive" and role.count("-retry") < self.cfg["retry_inconclusive"]:
                self.state["queue"].append({"name": f"{role}-retry", "set": r["set"], "aims": [], "source": "retry",
                                            "hypothesis": f"{role} looked better in round {rnd['n']} but within "
                                                          f"noise ({reason}); again on the current recipe",
                                            "retry_of": json.dumps(r["set"], sort_keys=True)})
            r["verdict"] = verdict
        winners.sort(key=lambda x: -x[0])
        if winners:
            _, role, r = winners[0]
            self.state["recipe"]["overrides"].update(r["set"])
            decisions["recipe"] = f"kept {role}: {json.dumps(r['set'])}"
            for _, other, ro in winners[1:]:   # other real gains: test them again on top of the new recipe
                self.state["queue"].insert(0, {"name": f"{other}-stacked", "hypothesis": f"{other} helped alone "
                                               f"in round {rnd['n']}; does it add to {role}?", "set": ro["set"],
                                               "aims": [], "source": "stack"})

        # champion: among the controls and the kept arms, the policy closest to the targets, if it beats
        # the current champion (arms that lost to their control never become champion)
        champ = self.state["champion"]
        pool = [(k, r) for k, r in runs.items() if r.get("summary") and (k.startswith("control") or r.get("verdict") == "kept")]
        best_role, best = min(pool, key=lambda kr: kr[1]["gap"])
        if champ["summary"] is None:
            promote, why = True, "first benchmarked policy of the campaign"
        else:
            v = stats.compare(best["summary"], champ["summary"], self.targets, noise=noise)
            promote, why = v.better, v.reason
        decisions["champion"] = {"candidate": best_role, "promoted": promote, "reason": why}
        if promote:
            self.state["champion"] = {"label": best["name"], "run_id": best["run_id"],
                                      "checkpoint_id": self._checkpoint_of(best["run_id"]),
                                      "summary": best["summary"], "gap": best["gap"], "run_status": best["status"]}
            self.event("champion", round=rnd["n"], name=best["name"], run_id=best["run_id"], gap=best["gap"])
        rnd["decisions"], rnd["phase"], rnd["finished"] = decisions, "decided", now()
        self.event("round_decided", round=rnd["n"], decisions=decisions)
        self.log(f"round {rnd['n']} decided: " + "; ".join(f"{k}: {d['verdict']}" for k, d in decisions.items()
                                                          if isinstance(d, dict) and "verdict" in d))
        champ = self.state["champion"]
        if champ.get("run_status") in PASSED:
            return f"done: {champ['label']} passed the release gate; review the release in the web app"
        if champ["summary"] and stats.passes(champ["summary"], self.targets):
            return (f"stopped: {champ['label']} meets every benchmark target but its run did not pass the gate "
                    f"({champ.get('run_status')}); look at its strict test in the web app")
        return None

    # -- report ---------------------------------------------------------------------------------
    def report(self) -> str:
        s, t = self.state, self.targets
        pct = lambda x: "n/a" if x is None else f"{x*100:.1f} %"
        champ_gap = "n/a" if s["champion"]["gap"] is None else f"{s['champion']['gap']:.4f}"
        noise_s = "not measured" if not s["noise"] else f"{s['noise']['gap']:.4f} ({s['noise']['samples']} pairs)"
        lines = [f"# Improvement campaign: {s['campaign']}", "",
                 f"Status: **{s['status']}** · rounds {len(s['rounds'])} · GPU hours {s['gpu_hours']:.1f} · "
                 f"seed noise on the gap {noise_s}", "",
                 f"Champion: **{s['champion']['label']}** · gap to targets {champ_gap}", "",
                 f"Recipe: `{json.dumps(s['recipe'])}`", "",
                 "Targets (95 % upper bound of the fall rate, steps up to the target height): "
                 + ", ".join(f"{k} <= {v*100:g} %" for k, v in t.items()), ""]
        for rnd in s["rounds"]:
            lines += [f"## Round {rnd['n']} ({rnd['phase']}, advisor {rnd['advisor']})", ""]
            if rnd.get("analysis"):
                lines += ["> " + rnd["analysis"].replace("\n", "\n> "), ""]
            lines += ["| run | change | status | gap | " + " | ".join(t) + " | verdict |",
                      "|---|---|---|---|" + "---|" * len(t) + "---|"]
            dec = rnd.get("decisions", {})
            for role, r in rnd["runs"].items():
                summ = r.get("summary") or {}
                cells = [pct(summ[c]["rate"]) if c in summ else "" for c in t]
                d = dec.get(role, {})
                verdict = (d.get("verdict", "") + (f": {d.get('reason')}" if d.get("reason") else "")) if d else \
                    ("reference" if role.startswith("control") else "")
                gap_s = "" if r.get("gap") is None else f"{r['gap']:.4f}"
                lines.append(f"| {r['name']} | `{json.dumps(r['set']) if r['set'] else 'control'}` | {r['status']} | "
                             f"{gap_s} | " + " | ".join(cells) + f" | {verdict} |")
            for key in ("recipe", "champion", "seed_noise", "error"):
                if key in dec:
                    lines.append(f"- **{key}**: {json.dumps(dec[key]) if not isinstance(dec[key], str) else dec[key]}")
            lines.append("")
        return "\n".join(lines)


# -- CLI --------------------------------------------------------------------------------------------

def dry_run_studio(cfg: dict, seed: int) -> FakeStudio:
    d = cfg.get("dry_run", {})
    gate = [{"metric": f"bench.{k}.rate_hi", "op": "<=", "value": v} for k, v in d["targets"].items()]
    return FakeStudio(gate, d["base_rates"], d.get("effects", {}), seed=seed, skill_id=cfg["skill_id"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run rounds until the gate passes or a limit is hit (resumes from state.json)")
    r.add_argument("config")
    r.add_argument("--once", action="store_true", help="one round, then exit")
    r.add_argument("--dry-run", action="store_true", help="simulated studio (the config's dry_run section); "
                                                          "state goes to <state_dir>-dry-run")
    r.add_argument("--advisor", default=None, choices=["rules", "bedrock", "jev"], help="override the config's advisor")
    r.add_argument("--seed", type=int, default=0, help="dry run: simulation seed")
    s = sub.add_parser("status", help="print the campaign report")
    s.add_argument("config")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "status":
        f = Path(cfg["state_dir"]) / "report.md"
        print(f.read_text() if f.exists() else "no campaign state yet")
        return
    if a.advisor:
        cfg["advisor"] = {**cfg["advisor"], "kind": a.advisor}
    if a.dry_run:
        cfg["state_dir"] += "-dry-run"
        studio, sleep = dry_run_studio(cfg, a.seed), (lambda s: None)
        cfg["poll_s"] = 0
    else:
        studio, sleep = Studio(), time.sleep
    try:
        advisor = make_advisor(cfg["advisor"])
    except Exception as e:
        print(f"advisor {cfg['advisor'].get('kind')} unavailable ({e}); using rules", file=sys.stderr)
        advisor = make_advisor({"kind": "rules"})
    camp = Campaign(cfg, studio, advisor, sleep=sleep)
    status = camp.run(once=a.once)
    print(status)
    sys.exit(0 if status.startswith(("done", "running")) else 3)


if __name__ == "__main__":
    main()

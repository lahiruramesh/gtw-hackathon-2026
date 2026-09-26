"""Client for the SKF Skill Studio API (docs/webapp/SPEC.md §9), stdlib only.

The loop signs in like a person (Better Auth email + password; create a dedicated user such as
automation@... with the ml_engineer role in the web app's admin page) and calls the API through the
web app's same-origin proxy (/api/backend/<path>), which attaches the short-lived JWT. Credentials
come from the environment (SKF_URL, SKF_EMAIL, SKF_PASSWORD), never from the campaign file.

FakeStudio simulates the API (runs finish at once with a synthetic benchmark) for dry runs and tests.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import random
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

TERMINAL = {"awaiting_review", "approved", "rejected", "gate_failed", "failed", "cancelled"}


class StudioError(RuntimeError):
    def __init__(self, status: int, message: str, body: Any = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.body = status, body


class Studio:
    def __init__(self, base_url: str | None = None, email: str | None = None, password: str | None = None,
                 timeout_s: float = 60.0):
        self.base = (base_url or os.environ.get("SKF_URL", "")).rstrip("/")
        self.email = email or os.environ.get("SKF_EMAIL")
        self._password = password or os.environ.get("SKF_PASSWORD")
        if not (self.base and self.email and self._password):
            raise SystemExit("set SKF_URL, SKF_EMAIL and SKF_PASSWORD (the automation user of the web app)")
        self.timeout = timeout_s
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self._signed_in = False

    def _send(self, method: str, url: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Origin": self.base, "Accept": "application/json",
            **({"Content-Type": "application/json"} if data is not None else {})})
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = raw.decode(errors="replace")[:500]
            msg = payload.get("error", {}).get("message") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) \
                else (payload.get("message") if isinstance(payload, dict) else payload)
            raise StudioError(e.code, str(msg), payload) from None

    def sign_in(self) -> None:
        self._send("POST", f"{self.base}/api/auth/sign-in/email", {"email": self.email, "password": self._password})
        self._signed_in = True

    def api(self, method: str, path: str, body: Any = None) -> Any:
        if not self._signed_in:
            self.sign_in()
        url = f"{self.base}/api/backend/{path.lstrip('/')}"
        try:
            return self._send(method, url, body)
        except StudioError as e:
            if e.status != 401:
                raise
            self.sign_in()   # session expired: once more
            return self._send(method, url, body)

    # -- resources ------------------------------------------------------------------------------
    def me(self) -> dict:
        return self.api("GET", "me")

    def skill(self, skill_id: str) -> dict:
        return self.api("GET", f"skills/{skill_id}")

    def targets(self) -> list[dict]:
        return self.api("GET", "compute-targets")

    def estimate(self, body: dict) -> dict:
        return self.api("POST", "runs/estimate", body)

    def create_run(self, body: dict) -> dict:
        return self.api("POST", "runs", body)

    def run(self, run_id: str) -> dict:
        return self.api("GET", f"runs/{run_id}")

    def find_run(self, name: str) -> dict | None:
        page = self.api("GET", f"runs?q={urllib.request.quote(name)}&limit=50")
        return next((r for r in page["items"] if r["name"] == name), None)

    def runs_since(self, since: datetime) -> list[dict]:
        """Runs created at or after `since` (newest first, paged)."""
        out, cursor = [], None
        while True:
            page = self.api("GET", "runs?limit=200" + (f"&cursor={cursor}" if cursor else ""))
            for r in page["items"]:
                if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) < since:
                    return out
                out.append(r)
            cursor = page.get("next_cursor")
            if not cursor:
                return out

    def evaluations(self, run_id: str) -> list[dict]:
        return self.api("GET", f"runs/{run_id}/evaluations")

    def artifacts(self, run_id: str) -> list[dict]:
        return self.api("GET", f"runs/{run_id}/artifacts")


# -- dry runs -------------------------------------------------------------------------------------

class FakeStudio:
    """In-memory stand-in: every run finishes on the next poll with a synthetic benchmark whose fall
    rates come from `effects` (knob key -> per-condition multiplier on the fall rate) applied to the
    parent's rates, plus binomial noise. For exercising the loop's logic end to end, not for results."""

    CONDS = ("strict", "camera", "delay_20ms", "low_friction", "payload_5kg", "push", "speed_06")

    def __init__(self, gate: list[dict], base_rates: dict[str, float], effects: dict[str, dict[str, float]],
                 seed: int = 0, gpu_hours_per_run: float = 0.8, skill_id: str = "g1-stairs-bench"):
        self.gate, self.effects, self.gpu_h, self.skill_id = gate, effects, gpu_hours_per_run, skill_id
        self.rng = random.Random(seed)
        self.runs: dict[str, dict] = {}
        self.rates: dict[str, dict[str, float]] = {}   # checkpoint id -> true fall rates
        self.base_ckpt = "ckpt-champion"
        self.rates[self.base_ckpt] = dict(base_rates)

    def me(self):
        return {"id": "bot", "role": "ml_engineer", "permissions": ["run:create_custom"]}

    def skill(self, skill_id):
        return {"id": skill_id, "gate": self.gate, "params_schema": {"properties": {}}}

    def targets(self):
        return [{"id": "t-aws", "name": "AWS L40S", "kind": "aws_ec2", "usage": {}}]

    def estimate(self, body):
        return {"gpu_hours": self.gpu_h, "total_minutes": self.gpu_h * 60, "needs_approval": False, "blockers": []}

    def create_run(self, body):
        if any(r["name"] == body.get("name") for r in self.runs.values()):
            raise StudioError(409, "duplicate name")
        rid = str(uuid.uuid4())
        parent = body.get("parent_checkpoint_id") or self.base_ckpt
        rates = dict(self.rates[parent])
        over = json.loads(body["params"].get("overrides") or "{}")
        for key in over:
            for c, m in self.effects.get(key, {}).items():
                rates[c] = min(0.95, rates[c] * m)
        drift = self.effects.get("_training", {})
        for c in rates:   # more training on the same recipe: a small drift plus seed noise
            rates[c] = min(0.95, max(0.0, rates[c] * drift.get(c, 1.0) * self.rng.uniform(0.9, 1.1)))
        ckpt = f"ckpt-{rid[:8]}"
        self.rates[ckpt] = rates
        self.runs[rid] = {"id": rid, "name": body.get("name"), "status": "queued", "gpu_hours": self.gpu_h,
                          "created_at": datetime.now(timezone.utc).isoformat(), "params": body["params"],
                          "_ckpt": ckpt, "_polls": 0}
        return self.run(rid)

    def run(self, run_id):
        r = self.runs[run_id]
        r["_polls"] += 1
        if r["_polls"] > 1 and r["status"] not in TERMINAL:
            s = self._summary(r["_ckpt"])
            r["_summary"] = s
            ok = all(s[k.split(".")[1]]["rate_hi"] <= c["value"] for c in self.gate if (k := c["metric"]).startswith("bench."))
            r["status"] = "awaiting_review" if ok else "gate_failed"
        return {k: v for k, v in r.items() if not k.startswith("_")} | {"gate_verdict": None}

    def find_run(self, name):
        return next((self.run(r["id"]) for r in self.runs.values() if r["name"] == name), None)

    def runs_since(self, since):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.runs.values()]

    def evaluations(self, run_id):
        r = self.runs[run_id]
        return [{"stage_key": "bench", "summary": r["_summary"]}] if "_summary" in r else []

    def artifacts(self, run_id):
        return [{"id": self.runs[run_id]["_ckpt"], "name": "params.pkl", "kind": "params"}]

    def _summary(self, ckpt):
        import math
        out = {}
        for c, p in self.rates[ckpt].items():
            n = 1280
            fell = sum(self.rng.random() < p for _ in range(n))
            z = 1.96
            ph = fell / n
            d = 1 + z * z / n
            hi = min(1.0, (ph + z * z / (2 * n)) / d + z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d)
            out[c] = {"n": n, "fell": fell, "rate": ph, "rate_hi": hi, "n_all": 2048, "fell_all": fell, "rate_all": ph}
        return out

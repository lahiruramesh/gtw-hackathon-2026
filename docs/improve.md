# Improvement loop: stairs until the release gate passes

`g1pipe/improve` runs experiment rounds through the SKF Skill Studio (the web app) until a policy
passes the `g1-stairs-bench` release gate, or a budget or idea limit is reached. Before this, each
version (v1-v14) changed several things at once, was compared on 32-96 crossings, and training was
never repeated with a second seed. So we could not tell which change helped, or whether a
difference was just noise.

```
            champion checkpoint + recipe
                       │
   advisor (rules / Bedrock Nova Lite / Jev) ── picks 2-3 one-change experiments from the queue,
                       │                          may propose new ones (whitelisted knobs only)
                       ▼
   web app runs (skill g1-stairs-bench), all warm-started from the champion, same steps:
     control        champion recipe, unchanged          (round 1: + a second seed → seed noise)
     arm A          recipe + one change
     arm B          recipe + one change
       each: train (GPU) → bench: 7 conditions x 2048 episodes (GPU) → strict test (CPU) → gate
                       │
                       ▼
   decide in code (g1pipe/improve/stats.py)
     kept      an off-target condition improves at z >= 2.5 (binomial + seed noise), nothing gets
               significantly worse, the gap to the targets shrinks
     recipe    += the best kept change (other kept changes are re-tested on top of it)
     champion  = the control or kept arm closest to the targets, if it beats the current champion
     retry     an inconclusive arm runs once more on the new recipe
                       │
                       └── until: the champion's run passes the whole gate → a person reviews the
                           release in the web app │ queue exhausted │ round / GPU-hour limit │ daily cap
```

The goal is the `g1-stairs-bench` gate in [skills/stairs_bench/skill.yaml](../skills/stairs_bench/skill.yaml):

- **Fixed benchmark.** For each of the 7 conditions, the 95 % upper bound of the fall rate on steps
  up to 10 cm must be under its threshold.
- **Strict test.** The plain-MuJoCo strict test must certify 10 cm.

To change a target, edit the gate and `THRESHOLDS` in
[summarize.py](../skills/stairs_bench/summarize.py); a test checks the two stay in step.

## Why these rules

- **Same test for every policy.** `g1pipe.bench` gives every policy the same conditions and start
  states, whatever the policy was trained with. `gpu_eval` ranked each run under its own training
  mix, so it could not compare runs.
- **One change per arm, against a control trained just as long.** Otherwise more training steps
  look like a win.
- **Seed noise is measured, not assumed.** Two trainings of the same recipe differ far more than
  two evaluations do. In the dry runs this spread hid a real 4x payload gain until the tests went
  per condition.
- **The advisor proposes; code decides.** An LLM never keeps a change, promotes a champion, spends
  budget or changes targets. It writes the round's analysis and suggests what to try next, and code
  checks every change against the knob whitelist in the campaign file.

## Set up (once)

1. **Web app.** Deploy this branch, then `POST /skills/sync` (or restart the API) so `g1-stairs-bench` appears.
   The aws_ec2 backend rsyncs the repo to the box at submit, so `g1pipe/bench.py` and `train --overrides`
   travel with each run.
2. **Champion.** Import v14 into the web app (it needs an entry in `apps/api/src/skf_api/history/catalog.py`, then
   `skf-api import-history`). Put its `params.pkl` artifact id in `champion.checkpoint_id` in
   [experiments/stairs/campaign.yaml](../experiments/stairs/campaign.yaml).
3. **Automation user.** In the web app's admin page, create e.g. `automation@sylonik.com` with the `ml_engineer` role
   (custom params need `run:create_custom`). Give the loop its credentials through the environment only:
   `SKF_URL=https://sylonik.com SKF_EMAIL=... SKF_PASSWORD=...`. With the `operator` role, runs over the target's
   `max_unapproved_gpu_hours` wait for a person to approve them in the web app. Use that role if every launch
   should be approved.
4. **Advisor (optional; rules work without it).** Enable model access to Amazon Nova Lite in Bedrock (us-east-1).
   The loop's AWS identity needs `bedrock:InvokeModel` on `us.amazon.nova-lite-v1:0`. Any Converse model with tool
   use works: set `advisor.bedrock.model_id`. Nova Micro is cheaper; for Nova Pro, set it here. A round's request
   is a few thousand tokens. For Jev, set `advisor.kind: jev` and `TYPESAFE_API_KEY`.
5. **Budget.** `budget.daily_gpu_hours` (default 10) counts every run on the target that day, by anyone. With
   `on_daily_cap: stop` the loop stops and waits for a person; with `wait` it sleeps until tomorrow.
   `campaign_gpu_hours` and `max_rounds` cap the whole campaign. The aws_ec2 target's own weekly quota and
   stop-when-idle still apply.

## Run

```bash
uv sync --extra train --extra improve          # boto3 for the Bedrock advisor
uv run python -m g1pipe.improve.campaign run experiments/stairs/campaign.yaml --dry-run   # simulated studio
uv run python -m g1pipe.improve.campaign run experiments/stairs/campaign.yaml             # for real
uv run python -m g1pipe.improve.campaign status experiments/stairs/campaign.yaml
uv run pytest tests/test_improve.py
```

`run` resumes from `experiments/stairs/campaign/state.json`. After a crash or restart it waits for
the runs it already launched instead of launching them again. Run it under a supervisor that restarts
it (systemd, a compose service with `restart: on-failure`), with the state directory on a persistent
volume. It needs no GPU and no JAX: only the repo, PyYAML and boto3.

## Where to look

- **Web app:** runs named `stairs-auto-rNN-<arm>`. Their notes carry the hypothesis and the change. Use
  **Compare** for a round's control against its arms, and review the champion's release once its gate passes.
- **`experiments/stairs/campaign/report.md`:** every round, with the advisor's analysis, per-condition
  fall rates, verdicts, recipe and champion changes.
- **`events.jsonl`:** every launch, status change, decision and champion. Set `notify.webhook_url` (or
  `IMPROVE_WEBHOOK_URL`) to have each event POSTed as JSON.

## What the loop cannot do

It searches training knobs. A fix that needs code ends the campaign with "queue exhausted", and the
advisor's analysis says what the benchmark still misses. One example is teacher-student distillation
for the head-camera gap, which v13 showed a knob will not close. Adding the code and a knob for it
(for example `env.distill_weight`) makes it searchable.

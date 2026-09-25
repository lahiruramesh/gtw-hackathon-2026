# SKF Skill Studio runbook

Day-to-day operations for developers and platform admins. The contract is [SPEC.md](SPEC.md), the reasoning is
[PLAN.md](PLAN.md), and production hosting on AWS is covered in [infra/aws/README.md](../../infra/aws/README.md).

| Service | Dev URL | Notes |
|---|---|---|
| Web (Next.js) | http://localhost:3100 | the only UI; it calls the API server-side |
| API (FastAPI) | http://localhost:8000 | `/docs` for the OpenAPI UI (never exposed in production) |
| Postgres 17 | `localhost:55432` | user/password/db `skf`; schemas `auth` (Better Auth) and `app` (Alembic); `skf_test` for tests |
| Redis 7 | `localhost:56379` | arq queue (db 0), live log pub/sub; tests use db 15 |
| S3 (MinIO or RustFS) | http://localhost:59000, console :59001 | `skfminio` / `skfminio-dev-secret`; bucket `skf-artifacts` is created by the API |

## 1. Local development from zero

Prerequisites: Docker Desktop, [uv](https://docs.astral.sh/uv/) 0.8+, Node 24 with pnpm 10 (`corepack enable`), `make`.

```bash
# 1. Dev services (postgres, redis, S3). If `minio/minio` can't be pulled (Docker Hub and quay.io no longer
#    serve it), use the drop-in RustFS container instead: same port, same credentials.
make infra-up                 # or: make infra-up S3=rustfs

# 2. Configuration
cp apps/api/.env.example apps/api/.env          # set SECRET_KEY:      openssl rand -base64 32
cp apps/web/.env.example apps/web/.env.local    # set BETTER_AUTH_SECRET (openssl rand -base64 32) and ADMIN_PASSWORD

# 3. Dependencies
make api-install web-install
uv sync --extra train         # the research pipeline, used by local_cpu stages (repo .venv = PIPELINE_PYTHON)

# 4. Schema, skills, compute targets (Local CPU enabled; Kaggle T4 and AWS L40S disabled until they have
#    credentials) and the first admin. All idempotent. KAGGLE_USERNAME=... pre-fills the Kaggle target.
make migrate seed

# 5. Run (one terminal each)
make api
make worker
make web                      # then sign in at http://localhost:3100 with ADMIN_EMAIL / ADMIN_PASSWORD

# Optional: the existing research runs as historical runs (v1, no-DR, stairs v1-v11). runs/ is gitignored,
# so point at the checkout that trained them; results/ is in the repo.
make import-history HISTORY_DIR=/path/to/checkout/runs
```

The first `local_cpu` stage on a fresh `.venv` downloads the MuJoCo Menagerie G1 model (MuJoCo Playground does
this on first import). The worker image has it baked in.

**Everything in containers** (closest to production; no hot reload):

```bash
cp .env.example .env          # SECRET_KEY, BETTER_AUTH_SECRET, ADMIN_EMAIL, ADMIN_PASSWORD
make up                       # builds api, worker, web; runs both migrations; waits until healthy
docker compose -f infra/compose.yaml --profile app exec api skf-api sync-skills
docker compose -f infra/compose.yaml --profile app exec api skf-api seed-targets
docker compose -f infra/compose.yaml --profile app run --rm web-migrate pnpm seed:admin
# history import in containers: mount results/ and a runs/ directory into a one-off worker
docker compose -f infra/compose.yaml --profile app run --rm -v "$PWD/results:/data/results:ro" \
  -v /path/to/checkout/runs:/data/runs:ro worker skf-api import-history --results-dir /data/results --runs-dir /data/runs
make down                     # stop (volumes are kept)
```

The worker image is large (about 3.5 GB: JAX, Brax, MuJoCo, Playground) because it runs `local_cpu` stages
itself. Stop a host-run `make api` / `make web` first: `make up` binds the same ports (8000, 3100).

Checks before pushing: `make lint test` (API tests need `make infra-up`).

## 2. Users and roles

There is no public sign-up. Admins create accounts in the web app: **Admin → Users → New user** (name, email,
temporary password, role). The user changes the password after signing in. Every user-management action is
written to the audit log (**Admin → Audit**).

| Role | Can |
|---|---|
| `admin` | everything, including users, compute targets and secrets |
| `ml_engineer` | launch any run (custom params, warm starts), evaluate checkpoints, approve operator launches |
| `operator` | launch runs from approved presets; runs over the target's GPU-hour limit wait for approval |
| `safety_reviewer` | approve or reject releases after the gate passes; can't launch runs |
| `viewer` | read-only |

The matrix lives in `shared/permissions.json` and is enforced by the API (the UI only hides what a role can't use).
Changing a role takes effect within 15 minutes (the lifetime of the API token) or at the user's next sign-in.
Banning a user revokes their sessions immediately. The first admin comes from `make seed` (`ADMIN_EMAIL`,
`ADMIN_PASSWORD`); re-running it never changes an existing user.

## 3. Compute targets and their secrets

Targets are managed in **Compute** (admin only). `make seed` (`skf-api seed-targets`) creates the three this
repo's experiments ran on, with their measured throughput: **Local CPU** (enabled), **Kaggle T4** and
**AWS L40S** (disabled until an admin adds credentials and enables them). Secrets are write-only: stored encrypted with a key derived from
`SECRET_KEY`, never returned by the API, never logged. **Check health** runs a read-only probe (it never starts
a GPU).

| Kind | Config | Secret | Probe |
|---|---|---|---|
| `local_cpu` | `max_concurrent_hint`, `python` (default `PIPELINE_PYTHON`), `work_subdir` | none | the pipeline interpreter imports the training modules |
| `kaggle` | `username`, `accelerator` (`NvidiaTeslaT4`), `kernel_prefix`, `enable_internet` | `{"key": "<Kaggle API key>"}` | `kaggle kernels list --mine` and `kaggle quota` (degraded below 1 GPU hour) |
| `aws_ec2` | `region`, `instance_name`, `instance_type`, `ami_id`, `subnet_id`, `security_group_id`, `key_name`, `volume_gb`, `ssh_user`, `remote_repo_dir`, `bootstrap_command`, `create_if_missing`, `stop_when_idle`, `aws_profile` | `{"ssh_private_key": "-----BEGIN ..."}` plus optional `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token` | describe the instance (never starts it) |

**Kaggle.** Kaggle → Settings → API → *Create New Token* downloads `kaggle.json`; paste only its `key` value as
the secret, put the `username` in the config, **Check health**, then enable the target. The account must be
phone-verified (GPU and internet in kernels). The seeded weekly quota (30 GPU hours) drives "quota left".

**AWS EC2 (the L40S training box).** The box is found by its `Name` tag (for example `g1-train`) and is started
and stopped by the worker; `create_if_missing` stays `false`, so the app never creates instances.
- Secret: the private key of the box's EC2 key pair (the contents of `~/.ssh/g1-train.pem`). Leave the AWS keys
  empty on the AWS app host: the instance role is used. For local development, either run with an AWS profile
  (`aws_profile` in the config) or paste short-lived keys.
- Network: the box's security group must allow TCP 22 from wherever the worker runs (the app host's Elastic IP
  in production, your IP in development).
- Idle auto-stop on the box (cron from `scripts/aws_box.sh launch`) stays on as a safety net.

Cost guards per target: `max_concurrent`, `max_unapproved_gpu_hours` (operator launches above it need approval),
`cost_per_gpu_hour` (drives the estimate and the 7-day cost).

## 4. Launching a smoke run

1. **Runs → New run**, skill **Step-length adjustment**, preset **Smoke test**.
2. Target **Local CPU**. The estimate shows a few minutes and no GPU hours.
3. Review and **Launch**. The run page streams the training log live; when training ends the evaluate stage runs,
   then the release gate. A smoke policy is not expected to pass the gate: `gate_failed` is a successful smoke test.

A real run is the same flow with a GPU target and a real preset (`v1`, `v11-finetune`). On Kaggle the log appears
live if `PUBLIC_INGEST_URL` is set and reachable; otherwise it appears when the kernel finishes (see §6).

## 5. Where logs and outputs live

| What | Where |
|---|---|
| Stage log lines (what the UI shows, noise filtered, capped at `LOG_MAX_LINES_PER_RUN`) | `app.log_lines`; download from the Logs tab |
| Full raw job log | `job.log` in the stage's work dir, uploaded as a `log` artifact when the stage is collected |
| Local stage work dirs | `WORK_DIR/targets/<target_id>/local/<run_id>/<stage_key>-<stage_id[:8]>/{in,out,job.log}`: dev `.skf-work/`, containers the `workdata` volume |
| Artifacts (params, checkpoints, videos, JSON) | S3 `s3://skf-artifacts/runs/<run_id>/<stage_key>/<name>`; MinIO/RustFS console on :59001 in dev |
| On an AWS training box | `~/skf/<stage_id>/job.log`, outputs in `~/skf/<stage_id>/out`, tmux session `skf-<stage_id[:8]>` |
| On Kaggle | private kernel `<kernel_prefix>-<run>-<stage>-<id8>`; its output log becomes the stage's `job.log` |
| API / worker / web process logs | the terminals in dev; `docker compose -f infra/compose.yaml logs -f api worker web` in containers |
| Audit trail | `app.audit_events`; **Admin → Audit** |

## 6. Troubleshooting

**Every API call returns 401 / "invalid token" (JWKS mismatch).** The API verifies Better Auth JWTs against
`AUTH_JWKS_URL` with `AUTH_ISSUER` and `AUTH_AUDIENCE`.
- `curl -s http://localhost:3100/api/auth/jwks` must return an `Ed25519` key; the API must reach that URL
  (in containers it is `http://web:3000/api/auth/jwks`).
- `AUTH_ISSUER` / `AUTH_AUDIENCE` must be identical in `apps/api/.env` and `apps/web/.env.local`.
- After `BETTER_AUTH_SECRET` changed, stored signing keys can't be decrypted: see §8.
- Unknown role → 403: the user's role isn't one of the five in `shared/permissions.json`.

**Remote logs only appear at the end (ingest unreachable → polling fallback).** Kaggle and AWS jobs POST logs to
`PUBLIC_INGEST_URL`. When it is empty or unreachable from the job, the reporter keeps the full log locally and
the worker falls back to polling: AWS logs are tailed over SSH every 20 s, Kaggle logs arrive when the kernel
completes. This is expected in local development (localhost isn't reachable from Kaggle). To test live remote
logs from a laptop, expose port 8000 through a tunnel and set `PUBLIC_INGEST_URL=https://<tunnel>/ingest/v1`.
In production check `curl -si https://<domain>/ingest/v1/stages/00000000-0000-0000-0000-000000000000/heartbeat -X POST`
returns `401` (reachable, token rejected). Stages without a heartbeat for 10 minutes are reconciled via
`backend.status()`.

**Kaggle quota or verification.** `Kaggle quota exceeded` / kernels stuck in `queued`: the free tier is 30 GPU
hours per week, reset weekly; the compute page shows what is left. Use the AWS target or wait. `phone
verification required`: verify the account on kaggle.com, then **Check health**. A 401/403 on health means the
key was rotated: paste the new key as the target secret.

**AWS capacity (`InsufficientInstanceCapacity`).** Provisioning is retried with backoff and the stage stays in
`provisioning`. If it persists, change the target's `instance_type` (for example `g6e.xlarge` instead of
`g6e.2xlarge`) or subnet/AZ, or stop and restart the box from another AZ with `scripts/aws_box.sh`.
`UnauthorizedOperation` on start/stop: the instance lacks the `Project=skf-studio` tag or a `Name` starting with
`g1-train` (see `infra/aws/iam/app-host-policy.json`). SSH timeouts: the box's security group doesn't allow the
worker's address.

**`minio/minio` can't be pulled.** Use `make infra-up S3=rustfs` (same ports and credentials, separate volume).

**A local stage failed with "worker restarted".** Local jobs are supervised by the worker process; restarting
it (code reload, `make worker` stopped) fails them. Use **Retry** on the run.

**Ports already in use.** Host-run and containerised stacks share 8000/3100; stop one. 55432/56379/59000 belong
to the dev services.

## 7. Backup and restore

Development data is disposable: `docker compose -f infra/compose.yaml down -v` resets everything, then redo §1
steps 4 onwards. To keep a copy:

```bash
docker compose -f infra/compose.yaml exec -T postgres pg_dump -U skf -d skf --format=custom > skf.dump
docker compose -f infra/compose.yaml exec -T postgres pg_restore -U skf -d skf --clean --if-exists < skf.dump
```

Production (see infra/aws/README.md, "Backups"): RDS automated backups and snapshots, or with the containerised
database a nightly `pg_dump` to `s3://<bucket>/backups/postgres/` kept 35 days (`bin/backup-db.sh`, restore with
`bin/backup-db.sh --restore <s3-uri>`). Artifacts are protected by S3 versioning. Both schemas (`auth` and `app`)
are in the same dump, so users, sessions and runs are restored together.

## 8. Rotating secrets

Production values live in SSM Parameter Store (`/skf-studio/api/*`, `/skf-studio/web/*`); after changing a
parameter, run `infra/aws/deploy.sh <live-tag>` to re-render the env files and restart what changed.

**`SECRET_KEY`** (API and worker) derives two keys: the encryption key for compute-target secrets and the HMAC
key for ingest tokens. There is a single active key, so rotation is a short maintenance step:
1. Wait until no stage is running (dashboard: no active runs). Running jobs hold ingest tokens signed with the
   old key; after rotation their posts are rejected, and a Kaggle job's reporter stops the job when that keeps
   happening (it can't tell a rotated key from a cancelled stage).
2. Generate and store the new key (`openssl rand -base64 32`), redeploy / restart api and worker.
3. Re-enter every compute target secret (**Compute → Set secret**): the old ciphertexts no longer decrypt, and
   **Check health** reports it.

**`BETTER_AUTH_SECRET`** (web) signs sessions and encrypts the JWT signing keys stored in `auth.jwks`.
1. Store the new value and restart web. Everyone is signed out.
2. Delete the old signing keys so a new pair is generated on the next sign-in:
   `psql "$DATABASE_URL" -c 'DELETE FROM auth.jwks;'`
3. The API picks up the new public key automatically (it refetches the JWKS for an unknown key id).

**Kaggle key / SSH keys:** rotate at the provider, then paste the new value as the target secret. **Database
password:** change it in RDS (or the container), update `/skf-studio/api/DATABASE_URL` and
`/skf-studio/web/DATABASE_URL`, redeploy.

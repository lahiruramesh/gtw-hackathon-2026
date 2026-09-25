# syntax=docker/dockerfile:1.7
# SKF Skill Studio worker: the arq orchestrator plus everything a local_cpu stage needs (smoke training,
# cross-engine evaluation, videos). Build from the repo root:
#   docker build -f infra/docker/worker.Dockerfile -t skf-studio/worker .
#
# Layout under /app/pipeline (= PIPELINE_REPO_DIR) mirrors the repo, so the aws_ec2 backend can rsync g1pipe/,
# jev_agent/, scripts/, skills/, apps/reporter/, pyproject.toml and uv.lock to a training box straight from here.
#   /app/pipeline/apps/api/.venv   API + orchestrator (Python 3.12)
#   /app/pipeline/.venv            research pipeline, `uv sync --extra train` (= PIPELINE_PYTHON)

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.8

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3

FROM builder AS api-env
WORKDIR /app/pipeline/apps/api
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=apps/api/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/api/uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project
COPY apps/api/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pipeline environment: exactly the locked `--extra train` set, minus torch and its CUDA wheels
# (torch, triton, nvidia-*, cuda-*). torch is only used by g1pipe/sim.py to play back Unitree's vendor
# policy on a desktop; no pipeline stage imports it, and on Linux its CUDA stack adds ~5 GB.
FROM builder AS pipeline-env
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Debian packages follow the (pinned) base image; pinning their versions too would break on every point release.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app/pipeline
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv export --frozen --extra train --no-dev --no-emit-project --no-hashes --no-header --no-annotate \
      | grep -Ev '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z-]+)==' > /tmp/pipeline-requirements.txt \
 && uv venv /app/pipeline/.venv \
 && uv pip install --python /app/pipeline/.venv/bin/python --no-deps -r /tmp/pipeline-requirements.txt \
 && uv pip check --python /app/pipeline/.venv/bin/python
# MuJoCo Playground clones the whole mujoco_menagerie (~2 GB) on first use. Stages only need the G1 model,
# so fetch that directory at the commit the installed Playground pins, once, at build time.
RUN set -eu; \
    menagerie() { /app/pipeline/.venv/bin/python -c "from mujoco_playground._src import mjx_env; print(mjx_env.$1)"; }; \
    sha=$(menagerie MENAGERIE_COMMIT_SHA); \
    dest=$(menagerie MENAGERIE_PATH); \
    git init -q "$dest"; \
    git -C "$dest" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git; \
    git -C "$dest" sparse-checkout set unitree_g1; \
    git -C "$dest" fetch -q --depth 1 --filter=blob:none origin "$sha"; \
    git -C "$dest" checkout -q FETCH_HEAD; \
    rm -rf "$dest/.git"

FROM python:${PYTHON_VERSION}-slim AS runtime
# Commit the image was built from; recorded on runs when the repo has no .git (containers).
ARG GIT_SHA=
LABEL org.opencontainers.image.title="skf-studio-worker" \
      org.opencontainers.image.revision="${GIT_SHA}"
# Headless MuJoCo rendering (EGL via Mesa llvmpipe, OSMesa fallback), ffmpeg for videos, rsync and ssh for
# the aws_ec2 backend, tini to reap the detached local_cpu job processes.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates tini ffmpeg rsync openssh-client \
      libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglu1-mesa libosmesa6 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 skf \
 && useradd --system --uid 10001 --gid skf --home-dir /home/skf --create-home --shell /usr/sbin/nologin skf \
 && install -d -o skf -g skf /var/lib/skf/work
WORKDIR /app/pipeline
COPY --from=pipeline-env /app/pipeline/.venv .venv
COPY --from=api-env /app/pipeline/apps/api apps/api
COPY pyproject.toml uv.lock ./
COPY g1pipe/ g1pipe/
COPY jev_agent/ jev_agent/
COPY scripts/ scripts/
COPY apps/reporter/ apps/reporter/
COPY shared/ shared/
COPY skills/ skills/
ENV PATH=/app/pipeline/apps/api/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIPELINE_REPO_DIR=/app/pipeline \
    PIPELINE_PYTHON=/app/pipeline/.venv/bin/python \
    WORK_DIR=/var/lib/skf/work \
    MUJOCO_GL=egl \
    JAX_PLATFORMS=cpu \
    GIT_SHA=${GIT_SHA}
WORKDIR /app/pipeline/apps/api
USER 10001:10001
VOLUME ["/var/lib/skf/work"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["arq", "--check", "skf_api.orchestrator.tasks.WorkerSettings"]
ENTRYPOINT ["tini", "-g", "--"]
CMD ["arq", "skf_api.orchestrator.tasks.WorkerSettings"]

# syntax=docker/dockerfile:1.7
# SKF Skill Studio API (FastAPI + Alembic). Build from the repo root:
#   docker build -f infra/docker/api.Dockerfile -t skf-studio/api .
#
# The image mirrors the repo layout under /app/pipeline (PIPELINE_REPO_DIR) so that paths resolved relative
# to the package (alembic.ini, shared/permissions.json, skills/) are identical in dev and in containers.
# It carries only apps/api plus the data it reads (shared/, skills/); the heavy training stack lives in the
# worker image.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.8

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3
WORKDIR /app/pipeline/apps/api
# Dependencies first: this layer is reused until uv.lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=apps/api/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/api/uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project
COPY apps/api/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
COPY shared/ /app/pipeline/shared/
COPY skills/ /app/pipeline/skills/

FROM python:${PYTHON_VERSION}-slim AS runtime
# Commit the image was built from; recorded on runs when the repo has no .git (containers).
ARG GIT_SHA=
LABEL org.opencontainers.image.title="skf-studio-api" \
      org.opencontainers.image.revision="${GIT_SHA}"
RUN groupadd --system --gid 10001 skf \
 && useradd --system --uid 10001 --gid skf --home-dir /home/skf --create-home --shell /usr/sbin/nologin skf
# Code stays root-owned and read-only for the service user.
COPY --from=build /app/pipeline /app/pipeline
ENV PATH=/app/pipeline/apps/api/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIPELINE_REPO_DIR=/app/pipeline \
    GIT_SHA=${GIT_SHA}
WORKDIR /app/pipeline/apps/api
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
# Caddy terminates TLS in front of the API, so X-Forwarded-* from the compose network is trusted.
CMD ["uvicorn", "skf_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]

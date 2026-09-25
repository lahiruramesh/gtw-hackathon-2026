"""Ingest tokens and the public /ingest/v1 API: auth, rate limit, truncation, run cap, metrics, heartbeat."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from api_fakes import RecordingJobQueue

from skf_api.context import AppContext
from skf_api.core.ingest_tokens import IngestTokenSigner, InvalidIngestToken
from skf_api.modules.compute.seeds import KAGGLE_T4
from skf_api.modules.ingest.service import MAX_LINE_CHARS, IngestService, Line, clip, level_of
from skf_api.modules.runs.models import LogLine, Stage, StageStatus

Auth = Callable[..., dict[str, str]]


# ------------------------------------------------------------------------------------------------ tokens


def test_token_round_trip_and_tampering() -> None:
    signer = IngestTokenSigner("k" * 40)
    stage_id = uuid.uuid4()
    token = signer.mint(stage_id, int(time.time()) + 60)
    assert signer.verify(token).stage_id == stage_id

    encoded, exp, sig = token.split(".")
    with pytest.raises(InvalidIngestToken, match="signature"):
        signer.verify(f"{encoded}.{int(exp) + 3600}.{sig}")  # extended expiry
    other = IngestTokenSigner("k" * 40).mint(uuid.uuid4(), int(exp))
    with pytest.raises(InvalidIngestToken, match="signature"):
        signer.verify(f"{other.split('.')[0]}.{exp}.{sig}")  # swapped stage
    with pytest.raises(InvalidIngestToken, match="signature"):
        IngestTokenSigner("j" * 40).verify(token)  # different SECRET_KEY
    with pytest.raises(InvalidIngestToken, match="expired"):
        signer.verify(signer.mint(stage_id, int(time.time()) - 1))
    with pytest.raises(InvalidIngestToken, match="malformed"):
        signer.verify("garbage")


def test_level_heuristic_and_clipping() -> None:
    assert level_of("Traceback (most recent call last):") == "error"
    assert level_of("ValueError: bad") == "error"
    assert level_of("1 test FAILED") == "error"
    assert level_of("UserWarning: deprecated") == "warn"
    assert level_of("step 100 reward 3.2") == "info"
    long = "x" * 10_000
    clipped = clip(long + "\n")
    assert len(clipped) == MAX_LINE_CHARS and clipped.endswith("chars truncated]")
    assert clip("line\r\n") == "line"


# ------------------------------------------------------------------------------------------------ API


@pytest.fixture
async def running_stage(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    targets: dict[str, uuid.UUID],
) -> tuple[uuid.UUID, str]:
    """The train stage of a Kaggle run, submitted (provisioning), and an ingest token for it."""
    response = await client.post(
        "/api/v1/runs",
        headers=auth("ml_engineer"),
        json={
            "skill_id": "g1-stairs",
            "preset_id": "v11-finetune",
            "compute_target_id": str(targets[KAGGLE_T4]),
        },
    )
    run_id = uuid.UUID(response.json()["id"])
    await queue.drain(ctx)
    async with ctx.db.session() as session:
        stage_id = await session.scalar(
            sa.select(Stage.id).where(Stage.run_id == run_id, Stage.key == "train")
        )
    assert stage_id is not None
    return stage_id, ctx.ingest_tokens.mint(stage_id, int(time.time()) + 3600)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def post(
    client: httpx.AsyncClient, stage_id: uuid.UUID, kind: str, token: str, body: dict[str, Any]
) -> httpx.Response:
    return await client.post(f"/ingest/v1/stages/{stage_id}/{kind}", headers=bearer(token), json=body)


async def test_logs_are_stored_and_move_stage_to_running(
    client: httpx.AsyncClient, ctx: AppContext, running_stage: tuple[uuid.UUID, str]
) -> None:
    stage_id, token = running_stage
    response = await post(
        client,
        stage_id,
        "logs",
        token,
        {
            "lines": [
                {"text": "compiling"},
                {"text": "RuntimeError: boom", "stream": "stderr"},
                {"text": "y" * 5000, "ts": "2026-09-25T10:00:00Z"},
            ],
            "noise_dropped": 42,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"accepted": 3, "dropped": 0}
    async with ctx.db.session() as session:
        stage = await session.get(Stage, stage_id)
        lines = (await session.scalars(sa.select(LogLine).order_by(LogLine.id))).all()
    assert stage is not None
    assert stage.status is StageStatus.RUNNING and stage.noise_dropped == 42 and stage.last_heartbeat_at
    assert [line.level for line in lines] == ["info", "error", "info"]
    assert len(lines[2].text) == MAX_LINE_CHARS and lines[2].ts.year == 2026


async def test_token_must_match_path_and_live_stage(
    client: httpx.AsyncClient, ctx: AppContext, running_stage: tuple[uuid.UUID, str]
) -> None:
    stage_id, token = running_stage
    body = {"lines": [{"text": "x"}]}
    assert (await post(client, uuid.uuid4(), "logs", token, body)).status_code == 401
    assert (await post(client, stage_id, "logs", token + "0", body)).status_code == 401
    assert (await client.post(f"/ingest/v1/stages/{stage_id}/logs", json=body)).status_code == 401
    expired = ctx.ingest_tokens.mint(stage_id, int(time.time()) - 5)
    assert (await post(client, stage_id, "logs", expired, body)).status_code == 401

    async with ctx.db.session() as session:
        await session.execute(
            sa.update(Stage).where(Stage.id == stage_id).values(status=StageStatus.SUCCEEDED)
        )
        await session.commit()
    response = await post(client, stage_id, "heartbeat", token, {})
    assert response.status_code == 401  # the reporter's --stop-when-revoked relies on this


async def test_rate_limit(
    client: httpx.AsyncClient, ctx: AppContext, running_stage: tuple[uuid.UUID, str]
) -> None:
    stage_id, token = running_stage
    assert (await post(client, stage_id, "heartbeat", token, {})).status_code == 200
    now = int(time.time())
    for second in (now, now + 1):  # fill this second's window (and the next, in case it rolls over)
        await ctx.redis.set(f"ingest:rate:{stage_id}:{second}", 20, ex=5)
    limited = await post(client, stage_id, "heartbeat", token, {})
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "rate_limited"


async def test_batch_limits(client: httpx.AsyncClient, running_stage: tuple[uuid.UUID, str]) -> None:
    stage_id, token = running_stage
    too_many = await post(client, stage_id, "logs", token, {"lines": [{"text": "x"}] * 1001})
    assert too_many.status_code == 422
    too_big = await post(client, stage_id, "logs", token, {"lines": [{"text": "x" * 600_000}] * 2})
    assert too_big.status_code == 413


async def test_run_line_cap(ctx: AppContext, running_stage: tuple[uuid.UUID, str]) -> None:
    stage_id, _ = running_stage
    service = IngestService(ctx.redis, ctx.events, max_lines_per_run=5)
    async with ctx.db.session() as session:
        stage = await session.get(Stage, stage_id)
        assert stage is not None
        assert await service.append_logs(session, stage, [Line(f"l{i}") for i in range(3)]) == (3, 0)
        assert await service.append_logs(session, stage, [Line(f"m{i}") for i in range(4)]) == (2, 2)
        assert await service.append_logs(session, stage, [Line("n")]) == (0, 1)
        texts = list(await session.scalars(sa.select(LogLine.text).order_by(LogLine.id)))
        await session.refresh(stage)
    assert texts[:5] == ["l0", "l1", "l2", "m0", "m1"]
    assert texts[5].startswith("Log limit of 5 lines per run reached") and len(texts) == 6
    assert stage.noise_dropped == 3


async def test_metrics_filter_and_upsert(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, running_stage: tuple[uuid.UUID, str]
) -> None:
    stage_id, token = running_stage
    points = [
        {
            "step": 100,
            "wall_s": 5.0,
            "values": {
                "eval/episode_reward": 1.5,
                "eval/episode_reward_std": 0.1,
                "eval/episode_reward/alive": 3.0,
                "train/loss": 9.0,
            },
        }
    ]
    response = await post(client, stage_id, "metrics", token, {"points": points})
    assert response.json() == {"accepted": 1, "dropped": 0}
    points[0]["values"]["eval/episode_reward"] = 2.5
    await post(client, stage_id, "metrics", token, {"points": points})
    async with ctx.db.session() as session:
        run_id = await session.scalar(sa.select(Stage.run_id).where(Stage.id == stage_id))
    series = (await client.get(f"/api/v1/runs/{run_id}/metrics", headers=auth("viewer"))).json()
    assert series == {
        "keys": ["eval/episode_reward"],
        "series": [{"key": "eval/episode_reward", "points": [[100, 2.5]]}],
    }


async def test_heartbeat_updates_progress(
    client: httpx.AsyncClient, ctx: AppContext, running_stage: tuple[uuid.UUID, str]
) -> None:
    stage_id, token = running_stage
    response = await post(client, stage_id, "heartbeat", token, {"progress": 0.25, "message": "step 5M"})
    assert response.status_code == 200
    async with ctx.db.session() as session:
        stage = await session.get(Stage, stage_id)
    assert stage is not None and stage.progress == 0.25 and stage.message == "step 5M"
    assert (await post(client, stage_id, "heartbeat", token, {"progress": 2})).status_code == 422

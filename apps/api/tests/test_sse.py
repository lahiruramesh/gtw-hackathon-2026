"""SSE (SPEC §9.3): replay stored log lines after `after_log_id`, then live events from Redis."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable

import httpx
import sqlalchemy as sa
import uvicorn
from api_fakes import RecordingJobQueue

from skf_api.context import AppContext
from skf_api.main import create_app
from skf_api.modules.compute.seeds import KAGGLE_T4
from skf_api.modules.ingest.service import IngestService, Line
from skf_api.modules.runs.models import Stage
from skf_api.modules.runs.stream import run_events

Auth = Callable[..., dict[str, str]]


async def test_replay_then_live(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    targets: dict[str, uuid.UUID],
) -> None:
    created = await client.post(
        "/api/v1/runs",
        headers=auth("ml_engineer"),
        json={
            "skill_id": "g1-stairs",
            "preset_id": "v11-finetune",
            "compute_target_id": str(targets[KAGGLE_T4]),
        },
    )
    run_id = uuid.UUID(created.json()["id"])
    await queue.drain(ctx)
    ingest = IngestService(ctx.redis, ctx.events, 1000)
    async with ctx.db.session() as session:
        stage = await session.scalar(sa.select(Stage).where(Stage.run_id == run_id, Stage.key == "train"))
        assert stage is not None
        await ingest.append_logs(session, stage, [Line("one"), Line("two"), Line("three")])

    events = run_events(ctx, run_id, after_log_id=1)
    opened = await anext(events)
    assert (opened.event, opened.comment) == (None, "connected")
    replayed = [await anext(events), await anext(events)]
    assert [(e.event, e.id, e.data["text"]) for e in replayed] == [("log", "2", "two"), ("log", "3", "three")]

    async def write_live() -> None:
        async with ctx.db.session() as session:
            live_stage = await session.get(Stage, stage.id)
            assert live_stage is not None
            await ingest.append_logs(session, live_stage, [Line("four"), Line("five")])
            await ingest.heartbeat(session, live_stage, 0.5, "halfway")

    writer = asyncio.create_task(write_live())
    live = [await asyncio.wait_for(anext(events), 5) for _ in range(3)]
    await writer
    await events.aclose()
    assert [(e.event, e.id) for e in live[:2]] == [("log", "4"), ("log", "5")]
    assert live[1].data["text"] == "five" and live[1].data["level"] == "info"
    assert live[2].event == "stage" and live[2].data["progress"] == 0.5 and live[2].data["key"] == "train"


async def test_duplicates_after_replay_are_dropped(ctx: AppContext) -> None:
    run_id = uuid.uuid4()
    events = run_events(ctx, run_id, after_log_id=10)
    assert (await anext(events)).comment == "connected"

    async def publish() -> None:
        await asyncio.sleep(0.1)
        stale = {
            "id": 7,
            "stage_id": str(uuid.uuid4()),
            "ts": "2026-09-25T00:00:00Z",
            "level": "info",
            "text": "old",
        }
        await ctx.events.publish(run_id, "log", [stale, {**stale, "id": 11, "text": "new"}])
        await ctx.events.publish(run_id, "run", [{"id": str(run_id), "status": "running"}])

    task = asyncio.create_task(publish())
    first, second = [await asyncio.wait_for(anext(events), 5) for _ in range(2)]
    await task
    await events.aclose()
    assert (first.event, first.id, first.data["text"]) == ("log", "11", "new")
    assert second.event == "run" and second.data["status"] == "running"


async def test_events_endpoint_checks_the_run(
    client: httpx.AsyncClient, auth: Auth, skills: list[str]
) -> None:
    response = await client.get(f"/api/v1/runs/{uuid.uuid4()}/events", headers=auth("viewer"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_disconnecting_clients_do_not_leak_db_connections(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, targets: dict[str, uuid.UUID]
) -> None:
    """EventSource reconnects and page navigations drop streams right after `: connected`. Served by a real
    uvicorn (the ASGI test transport does not model disconnects), no stream may keep a pooled connection."""
    created = await client.post(
        "/api/v1/runs",
        headers=auth("ml_engineer"),
        json={
            "skill_id": "g1-stairs",
            "preset_id": "v11-finetune",
            "compute_target_id": str(targets[KAGGLE_T4]),
        },
    )
    run_id = created.json()["id"]
    server = uvicorn.Server(
        uvicorn.Config(create_app(ctx), host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 - uvicorn only exposes a flag
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    pool = ctx.db.engine.pool
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
            for linger in (0, 0.5, 0, 0.5, 0):
                url = f"/api/v1/runs/{run_id}/events"
                async with http.stream("GET", url, headers=auth("viewer")) as response:
                    assert response.status_code == 200
                    await asyncio.sleep(linger)
                await asyncio.sleep(0.2)
                assert pool.checkedout() == 0  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        server.should_exit = True
        await serving

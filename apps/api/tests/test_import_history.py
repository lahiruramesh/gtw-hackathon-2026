"""import-history on the repo's real results/ and past training outputs (skipped where those are absent)."""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

from skf_api.context import AppContext
from skf_api.history.importer import HistoryImporter
from skf_api.modules.artifacts.models import Artifact
from skf_api.modules.compute.seeds import AWS_L40S_COST_PER_GPU_HOUR
from skf_api.modules.gates.service import regate
from skf_api.modules.runs.models import Evaluation, Run, Stage
from skf_api.settings import REPO_ROOT


def _runs_dir() -> Path:
    """Training outputs are not in git: they live in runs/ of the main checkout (also for a worktree)."""
    if "SKF_TEST_RUNS_DIR" in os.environ:
        return Path(os.environ["SKF_TEST_RUNS_DIR"])
    common = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return Path(common).parent / "runs" if common else REPO_ROOT / "runs"


RUNS_DIR = _runs_dir()
pytestmark = pytest.mark.skipif(
    not (RUNS_DIR / "g1-stairs-v11" / "run" / "params.pkl").is_file(),
    reason=f"no past training outputs in {RUNS_DIR}",
)

Auth = Callable[..., dict[str, str]]


async def import_history(ctx: AppContext) -> list[str]:
    lines: list[str] = []
    importer = HistoryImporter(
        ctx, REPO_ROOT / "results", RUNS_DIR, with_checkpoints=False, echo=lines.append
    )
    await importer.run()
    return lines


async def counts(ctx: AppContext) -> tuple[int, ...]:
    async with ctx.db.session() as session:
        return tuple(
            [
                await session.scalar(sa.select(sa.func.count()).select_from(model))
                for model in (Run, Stage, Artifact, Evaluation)
            ]
        )


async def test_import_history(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, targets: dict[str, uuid.UUID]
) -> None:
    lines = await import_history(ctx)
    assert "  g1-stairs-v11 <- g1-stairs-v10 @ ckpt_00253624320.pkl" in lines
    assert "  g1-stairs-v10 <- g1-stairs-v9 @ params.pkl" in lines

    runs = {r["name"]: r for r in (await client.get("/api/v1/runs", headers=auth("viewer"))).json()["items"]}
    assert set(runs) == {
        "g1-steplength-v1",
        "g1-steplength-nodr",
        "g1-stairs-v9",
        "g1-stairs-v10",
        "g1-stairs-v11",
        "g1-stairs-v12",
        "g1-stairs-v14",
    }
    assert all(r["imported"] and r["status"] == "gate_failed" for r in runs.values())
    assert runs["g1-steplength-v1"]["compute_target"]["name"] == "Kaggle T4"
    assert runs["g1-steplength-v1"]["gpu_hours"] == 1.6  # docs/effort_log.csv
    assert runs["g1-stairs-v11"]["compute_target"]["name"] == "AWS L40S"
    assert runs["g1-stairs-v11"]["cost"] == pytest.approx(AWS_L40S_COST_PER_GPU_HOUR)  # 1.0 GPU-h
    # Lineage is part of the list view, so the skill page builds the tree from one request.
    assert runs["g1-stairs-v11"]["parent"]["run"]["name"] == "g1-stairs-v10"
    assert runs["g1-stairs-v11"]["parent"]["checkpoint"]["step"] == 253624320
    assert runs["g1-stairs-v10"]["parent"]["checkpoint"]["name"] == "params.pkl"
    assert runs["g1-stairs-v9"]["parent"] is None
    # v14 was certified by scripts/certify.py: its final strict test lives next to the weights.
    assert runs["g1-stairs-v14"]["parent"]["run"]["name"] == "g1-stairs-v12"
    assert runs["g1-stairs-v14"]["parent"]["checkpoint"]["step"] == 254279680
    assert runs["g1-stairs-v14"]["gpu_hours"] == 0.7

    v1 = (await client.get(f"/api/v1/runs/{runs['g1-steplength-v1']['id']}", headers=auth("viewer"))).json()
    criteria = {c["metric"]: c for c in v1["gate"]["criteria"]}
    assert criteria["evaluate.grid_fall_rate"]["passed"] is True
    assert criteria["evaluate.grid_step_abs_err_cm"]["actual"] == pytest.approx(10.6826, abs=1e-3)
    assert criteria["evaluate.stress.latency_20ms.fall_rate"]["actual"] == 1.0
    assert v1["params"]["timesteps"] == 200_000_000 and v1["params"]["no_dr"] is False
    assert v1["headline"][0] == {"label": "Step error", "value": pytest.approx(10.68, abs=0.01), "unit": "cm"}

    v11 = (await client.get(f"/api/v1/runs/{runs['g1-stairs-v11']['id']}", headers=auth("viewer"))).json()
    assert v11["parent"]["run"]["name"] == "g1-stairs-v10"
    assert v11["parent"]["checkpoint"]["name"] == "ckpt_00253624320.pkl"
    assert v11["params"]["init_from"] == v11["parent"]["checkpoint"]["id"]
    assert v11["params"]["lr"] == 0.0001 and v11["params"]["leg_action_scale"] == 1.0
    assert {s["key"] for s in v11["stages"]} == {
        "train",
        "evaluate",
        "gate",
        "evaluate-ckpt-111247360",
        "evaluate-ckpt-190709760",
        "evaluate-ckpt-222494720",
    }
    strict = {c["metric"]: c["actual"] for c in v11["gate"]["criteria"]}
    assert strict["evaluate.crossed_rate"] == pytest.approx(85 / 96)
    assert strict["evaluate.fell"] == 11.0

    evaluations = (await client.get(f"/api/v1/runs/{v11['id']}/evaluations", headers=auth("viewer"))).json()
    best = next(e for e in evaluations if e["stage_key"] == "evaluate-ckpt-190709760")
    assert (best["summary"]["crossed"], best["summary"]["fell"]) == (88, 8)  # "191M ckpt 88/96, 8 falls"
    assert best["checkpoint"]["step"] == 190709760 and best["suite"] == "strict-96"

    v9 = (await client.get(f"/api/v1/runs/{runs['g1-stairs-v9']['id']}", headers=auth("viewer"))).json()
    assert v9["parent"] is None  # v7 was never imported
    assert all(c["actual"] is None for c in v9["gate"]["criteria"])  # only a checkpoint was strict-tested
    assert v9["children"] == [
        {"id": runs["g1-stairs-v10"]["id"], "name": "g1-stairs-v10", "status": "gate_failed"}
    ]

    artifacts = (await client.get(f"/api/v1/runs/{v11['id']}/artifacts", headers=auth("viewer"))).json()
    names = {(a["stage_key"], a["name"]) for a in artifacts}
    assert {
        ("train", "params.pkl"),
        ("train", "progress.csv"),
        ("evaluate", "strict.json"),
        ("evaluate", "crossing.mp4"),
    } <= names
    assert len([a for a in artifacts if a["kind"] == "checkpoint"]) == 3  # only the strict-tested ones
    video = next(a for a in artifacts if a["name"] == "crossing.mp4")
    url = (await client.get(f"/api/v1/artifacts/{video['id']}/url", headers=auth("viewer"))).json()
    assert url["url"].startswith("http://files.example.test/skf-artifacts-test/runs/")

    metrics = (
        await client.get(
            f"/api/v1/runs/{v11['id']}/metrics?keys=eval/episode_crossed", headers=auth("viewer")
        )
    ).json()
    assert len(metrics["series"][0]["points"]) == 20

    compare = (
        await client.get(
            f"/api/v1/compare?run_ids={runs['g1-stairs-v9']['id']},{runs['g1-stairs-v10']['id']},{v11['id']}",
            headers=auth("viewer"),
        )
    ).json()
    crossed = next(r for r in compare["headline_rows"] if r["label"] == "Crossed")
    assert crossed["values"][0] is None and crossed["values"][2] == pytest.approx(100 * 85 / 96)
    # v9 was only strict-tested at a checkpoint, never on its final params: "not measured", not "failed".
    assert all(row["values"][0] is None for row in compare["gate_rows"])
    assert all(row["values"][2] is False for row in compare["gate_rows"])


async def test_import_is_idempotent(ctx: AppContext, targets: dict[str, uuid.UUID]) -> None:
    await import_history(ctx)
    first = await counts(ctx)
    await import_history(ctx)
    # 7 runs; v12 adds 4 checkpoint evaluations and v14 its certified final strict test
    assert await counts(ctx) == first == (7, 30, first[2], 14)


async def test_import_needs_seeded_targets(ctx: AppContext, skills: list[str]) -> None:
    lines = await import_history(ctx)
    assert lines[0].startswith("skip g1-steplength-v1: compute target Kaggle T4 not found")
    assert await counts(ctx) == (0, 0, 0, 0)


async def test_import_history_certified_run_keeps_its_card(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, targets: dict[str, uuid.UUID]
) -> None:
    await import_history(ctx)
    runs = {r["name"]: r for r in (await client.get("/api/v1/runs", headers=auth("viewer"))).json()["items"]}
    v14 = runs["g1-stairs-v14"]["id"]
    evaluations = (await client.get(f"/api/v1/runs/{v14}/evaluations", headers=auth("viewer"))).json()
    final = next(e for e in evaluations if e["stage_key"] == "evaluate")
    assert final["summary"]["n"] == 96 and final["summary"]["crossed"] == 95 and final["summary"]["fell"] == 1
    artifacts = (await client.get(f"/api/v1/runs/{v14}/artifacts", headers=auth("viewer"))).json()
    names = {a["name"] for a in artifacts}
    assert {"policy_card.md", "crossing.mp4", "strict.json", "params.pkl"} <= names


async def test_regate_reports_both_gate_levels(ctx: AppContext, targets: dict[str, uuid.UUID]) -> None:
    await import_history(ctx)
    async with ctx.db.session() as session:
        lines = set(await regate(session))
    assert "g1-steplength-v1: simulation pass (2/2), release fail (2/4)" in lines
    assert "g1-steplength-nodr: simulation fail (1/2), release fail (1/4)" in lines
    assert "g1-stairs-v14: simulation pass (2/2), release fail (2/3)" in lines
    assert "g1-stairs-v9: simulation fail (0/2), release fail (0/3)" in lines  # final policy never tested

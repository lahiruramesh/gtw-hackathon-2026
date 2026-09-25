"""The past experiments `skf-api import-history` brings into the studio, and where their evidence lives.

Training outputs come from `<runs-dir>/<name>/run/` (params.pkl, ckpt_*.pkl, progress.csv, config.json);
evaluation outputs from `<results-dir>`; GPU hours from docs/effort_log.csv. Warm-start lineage is not listed
here: it is read from each run's config.json (`init_from`).
"""

from __future__ import annotations

from dataclasses import dataclass

from skf_api.modules.compute.seeds import AWS_L40S, KAGGLE_T4


@dataclass(frozen=True)
class StepLengthResults:
    """scripts/eval_suite.py output directory under results/, e.g. `eval_v1`."""

    directory: str


@dataclass(frozen=True)
class StairsResults:
    """g1pipe.stairs_eval strict tests: results/stairs/strict_v<N>_params.json (final policy) and
    strict_v<N>_ckpt_<step>.json (periodic checkpoints), plus an optional crossing video."""

    version: int
    video: str | None = None


@dataclass(frozen=True)
class Experiment:
    name: str  # directory under --runs-dir; also the imported run's name
    skill_id: str
    target: str  # seeded compute target it trained on
    preset_id: str | None
    effort_match: str  # text that identifies its training row in docs/effort_log.csv
    notes: str
    results: StepLengthResults | StairsResults


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="g1-steplength-v1",
        skill_id="g1-step-length",
        target=KAGGLE_T4,
        preset_id="v1",
        effort_match="v1 finished",
        results=StepLengthResults("eval_v1"),
        notes="Reference step-length policy: 202M steps in 95 min on a Kaggle T4, with domain randomisation.",
    ),
    Experiment(
        name="g1-steplength-nodr",
        skill_id="g1-step-length",
        target=KAGGLE_T4,
        preset_id="e3-no-dr",
        effort_match="E3 no-DR run",
        results=StepLengthResults("eval_nodr"),
        notes="Experiment E3: the same recipe without randomisation, pushes or noise.",
    ),
    Experiment(
        name="g1-stairs-v9",
        skill_id="g1-stairs",
        target=AWS_L40S,
        preset_id=None,
        effort_match="AWS g1-stairs-v9",
        results=StairsResults(9),
        notes="Sole spheres (6 contacts per foot) and leg_action_scale 1.0; 500M steps in 85 min on an L40S.",
    ),
    Experiment(
        name="g1-stairs-v10",
        skill_id="g1-stairs",
        target=AWS_L40S,
        preset_id=None,
        effort_match="AWS g1-stairs-v10",
        results=StairsResults(10, video="stairs_v10.mp4"),
        notes="Fine-tune of v9 at lr 1e-4, 400M steps; the 254M checkpoint is the best of the run.",
    ),
    Experiment(
        name="g1-stairs-v11",
        skill_id="g1-stairs",
        target=AWS_L40S,
        preset_id=None,
        effort_match="g1-stairs-v11",
        results=StairsResults(11, video="stairs_v11.mp4"),
        notes="Stairs cadence 1.25-1.6 Hz, warm start from v10 @ 254M, lr 1e-4, 300M steps in 55 min.",
    ),
)

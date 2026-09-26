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
    strict_v<N>_ckpt_<step>.json (periodic checkpoints), plus an optional crossing video.

    Runs certified by scripts/certify.py keep the final policy's strict test next to the weights instead:
    `final_strict` is then that file, relative to --runs-dir. `card` is the policy card under results/stairs/.
    """

    version: int
    video: str | None = None
    final_strict: str | None = None
    card: str | None = None


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
        results=StairsResults(9, video="stairs_v9.mp4"),
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
    Experiment(
        name="g1-stairs-v12",
        skill_id="g1-stairs",
        target=AWS_L40S,
        preset_id=None,
        effort_match="AWS g1-stairs-v12",
        results=StairsResults(12, video="stairs_v12.mp4"),
        notes="Curriculum sends 70% of top-level graduates back to the 12-16 cm rows; "
        "warm start from v11 @ 191M, 300M steps. "
        "The 254M checkpoint (91/96 crossed, 5 falls) is the one later runs start from.",
    ),
    Experiment(
        name="g1-stairs-v14",
        skill_id="g1-stairs",
        target=AWS_L40S,
        preset_id=None,
        effort_match="First full g1job run: g1-stairs-v14",
        results=StairsResults(
            14,
            video="stairs_v14.mp4",
            final_strict="g1-stairs-v14/run/cert_params/strict.json",
            card="card_v14.md",
        ),
        notes="Actuation delay 0/20 ms and friction 0.25-1.0 randomised; "
        "warm start from v12 @ 254M, 100M steps. "
        "Strict test 95/96 crossed with 1 fall, certified 9.9 cm; see the policy card.",
    ),
)

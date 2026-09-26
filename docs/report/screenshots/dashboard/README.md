# SKF Skill Studio screenshots

Taken on 2026-09-26 from the cloud deployment at https://sylonik.com (release `6024ef8`, AWS us-east-1),
3200 × 1800 px. All numbers are the real, imported or live results; nothing was staged for the screenshots.

| File | Shows |
|---|---|
| `01-dashboard.png`, `02-dashboard-full.png` | Dashboard: GPU-hours and cost, compute targets (AWS L40S, Kaggle T4, local CPU), recent runs with simulation and release gates |
| `03-skills.png`, `04-skill-stairs.png` | Skill catalog; the stairs skill with its pipeline, presets and both gate levels |
| `05-runs-with-gates.png` | All runs with their simulation / hardware-release verdicts and headline metrics |
| `06`–`10` `run-stairs-v14-*` | Stairs v14: pipeline, gates (simulation passed; release 9.86 of 10 cm), training metrics, strict-test evaluation, crossing video |
| `11`, `12` `run-steplength-v1-*` | Step-length v1: gates (simulation passed; release failed on step error and 20 ms latency) and the evaluation grid |
| `13`–`15` `run-aws-l40s-*` | The first training launched from the dashboard: AWS g6e (L40S) box via zone fallback, live logs and metrics |
| `16-compare-stairs-v12-v14.png`, `16b-compare-stairs-v11-v14.png` | Version comparison; v11 → v14: crossed 89 % → 99 %, falls 11 → 1, certified 6.4 → 9.9 cm |
| `17-compare-steplength-dr-vs-nodr.png` | Experiment E3: domain randomisation vs none |
| `18-compute-targets.png` | Compute targets with health and quota |
| `19-learning-methods.png` | Learning-methods comparison matrix |

The 20-second pitch video from the same deployment is `docs/pitch/skf-skill-studio-tour.mp4`.

# Learning pipeline: G1 step-length adjustment

Living document: update the result sections as experiments finish.

## 1. Task specification

| Item | Value |
|---|---|
| Robot | Unitree G1 (29-DoF Playground model for training; 12-DoF Unitree model for vendor baseline) |
| Command | forward speed `v` ∈ [0.3, 0.9] m/s, step length `ℓ` ∈ [0.15, 0.35] m |
| Derived | cadence `f = v / 2ℓ`, limited to 0.9–1.8 Hz (gait period 0.56–1.11 s) |
| Success (provisional) | step-length error < 3 cm, speed error < 0.1 m/s, 0 falls in 100 nominal episodes, survives 0.5 m/s pushes |

## 2. E5 — do we need learning at all? (vendor policy, no retraining)

Unitree's pre-trained G1 policy (LSTM, trained in Isaac Gym, run here in MuJoCo) takes a speed
command and an internal gait clock (trained at 0.8 s). Swept 7 speeds × 5 gait periods × 3 seeds
with light pushes and sensor noise (105 runs, 15 s on a laptop CPU):

| gait period (s) | 0.2 m/s | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 |
|---|---|---|---|---|---|---|
| 0.6 | 7.2 cm | 14.1 | 20.1 | 25.8 | 30.8 | 35.2 |
| 0.7 | 7.2 | 14.2 | 21.2 | 27.6 | 33.4 | 37.9 |
| **0.8 (trained)** | 7.7 | 15.4 | 22.9 | 29.5 | 35.0 | 38.6 |
| 0.9 | 8.0 | 16.3 | 24.3 | 31.1 | 36.0 | 39.9 |
| 1.0 | 8.6 | 17.6 | 26.0 | 32.6 | 33.8 | 33.6 |

Findings:
* **No falls in any of the 105 runs.** Step length from about 7 cm to 40 cm is reachable without retraining.
* **Cadence follows the clock exactly** (for example period 0.6 s gives 3.3 steps/s), so step length
  is set by speed ÷ cadence.
* **But speed and step length are not independent.** Off the trained 0.8 s period, speed tracking
  degrades. At 0.4 m/s the achieved speed ranges from 0.48 (period 0.6) to 0.36 m/s (period 1.0). At 1.2 m/s
  with period 1.0 the gait breaks down (step variability 16 cm, speed 0.99 m/s instead of 1.2).
* **Conclusion:** a coarse step-length change is possible by re-parameterising an existing policy
  (zero training cost). *Precise, independent* step length at a given speed needs the step length
  in the task definition. That is what the trained `StepLength` policy adds, and it's the comparison
  that shows SKF what extra training buys.

![E5](../results/e5/e5_heatmap.png)

## 3. Training (E2)

*Fill in once the Kaggle run is back: GPU, steps, wall time, reward curve.*

## 4. Cross-engine evaluation and stress tests (E2/E3)

*Fill in from `results/eval_<tag>/summary.json`.*

## 5. Why it works / why it doesn't

*Fill in.*

## 6. Release gate before hardware (proposed)

1. Pass the success criteria in §1 in the unseen engine (MuJoCo C) with randomisation on.
2. No fall under: friction ≥ 0.5, payload ≤ 3 kg, pushes ≤ 0.5 m/s, 20 ms latency.
3. Joint torque and velocity within 90 % of limits over the whole evaluation suite.
4. Then: gantry (harness) tests on the real robot, then supervised floor trials, then shadow operation.

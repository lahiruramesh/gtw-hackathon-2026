# Lessons from the stairs runs (v1-v13): the rules the pipeline now enforces

Each rule comes from a run that cost time. The tool that now enforces it is in brackets.

## Physics and simulation

1. **The training engine must match the test engine.** MuJoCo Warp made one contact between the
   box sole and the stairs heightfield; plain MuJoCo made 3-20. v1-v8 trained standing on one point
   per foot and were tested flat-footed. Fixed with six sole spheres per foot (v9: certified height
   3 cm to 9.9 cm). [`g1pipe.preflight` parity check, before every run]
2. **Know the gap that is left.** With sole spheres, feet landing across a step edge agree between
   Warp and plain MuJoCo to 0.1 cm; a sole lying lengthwise on an edge still slides off in Warp and
   holds in MuJoCo (the 5 cm heightfield makes each edge a ramp; Warp gives the box one contact).
   Harmless for straight crossings, relevant for turning on stairs. [preflight reports it]
3. **Hold a pose for a short time when comparing engines.** Over 1 s an unstable pose tips over
   differently in each engine; over 0.25 s a real contact gap still shows and noise does not.
4. **Joint targets must be able to reach the motion.** Playground's +-0.5 rad action range capped knee
   extension torque at ~75 Nm when stepping up (the G1 knee motor gives 139 Nm). Falls at 8+ cm were
   the pelvis sinking with the knee target pinned at its limit. [fall analysis in the failure report]

## Infrastructure

5. **Check throughput in the first minutes.** MuJoCo Warp printf's a solver note from the GPU kernel
   every step: a 30 GB log in 40 min and 3-5x slower training until `warn_overflow = 0`.
   [`g1pipe.preflight` throughput, `g1job` GPU-power watch]
6. **GPU capacity is not guaranteed.** g6e capacity in us-east-1 came and went all day. Keep one box
   set up, try every zone and size, and have a fallback (g6.2xlarge, Kaggle). [`g1job` acquire]
7. **Budgets are enforced on the box, not by memory.** A shutdown timer on the box itself, a post-run
   stop, and a daily ledger of box hours. [`g1job`, `~/.g1job_ledger.json`]

## Evaluation

8. **Never judge a policy on 32 crossings.** Changing the step length by 4 mm swung the result from
   30/32 to 27/32. Rank checkpoints on thousands of episodes with confidence intervals; certify the
   finalists on 96+ crossings from several start poses. [`g1pipe.gpu_eval`, `scripts/certify.py`]
9. **Test on terrain the policy never saw.** v5 certified 8 cm on its own 30 cm-tread stairs and
   0-4.7 cm on held-out 27 cm treads: it had fitted its stride to one staircase. [held-out `test`
   layout, tread randomisation in training]
10. **Test with the sensor you deploy with.** v12 fell twice as often with the head camera (10/96) as
   with the true terrain (5/96). v13, fine-tuned on the camera's error model, did not close the gap
   (11/96), though its 32-crossing camera test showed 0 falls. The fix is structural (teacher-student
   distillation onto camera input), not a fine-tune. [`strict_camera` is a required certification condition]
11. **Find out why it fails, not just that it fails.** Every fix since v8 came from looking at the
    falls: joint targets at their limits, cadence, the tallest rows, the camera. [failure report]

## Training

12. **Warm start from the closest skill; rescale outputs when the action space changes.** Each run
    since v9 takes about an hour instead of days. [`g1pipe.train --init-from`, preflight warm-start check]
13. **Watch for reward loopholes.** v3 learned to turn away from climbs; a heading command and
    "turning away counts as a fall" closed it.
14. **Command the gait the skill needs.** On stairs, 1.4 Hz short steps (about one per tread) crossed
    30-31/32 where 1.0 Hz long strides crossed 14-15/32. Train on the cadence you will command.
15. **Practise where it fails.** Sending most top-level graduates back to the 12-16 cm rows cut the
    15 cm falls from 1 in 3 to 1 in 12 (v12).
16. **Checkpoints swing; pick by evaluation, not by the last one.** The final checkpoint was rarely
    the best. [`g1job` ranks every checkpoint on the GPU]
17. **Train with the robot's latency, not an ideal actuator.** Playground's recipe applies each action
    in the same control step. With a 20 ms actuation delay every policy since v1, flat walking
    included, falls within seconds; the first certification suite (v12) found it. Randomise action
    latency (0-20 ms) and friction below the nominal range in training before any hardware test.
    [`delay_20ms`, `low_friction` in `scripts/certify.py`]

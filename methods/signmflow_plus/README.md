---
status: exploring   # exploring | promising | final | abandoned
created: 2026-11-20
---

# signmflow_plus

Experiment variants of SignNMFlow (arXiv:2609.33339) for the plan in
`docs/proposal7/07_lo_trinh_uu_tien.md` (critique and rationale: `01`, `04`, `05`, `06`).

The core modules are a self-contained copy of `methods/signmflow` (v2 replica). Methods never
import each other. **With the default config the method is the paper replica:**
`tests/signmflow_plus/test_paths_schedules.py::test_defaults_reproduce_the_signmflow_replica_exactly`
checks that the loss and the sampler are bit-identical to `methods/signmflow`. Every new feature
is behind a flag whose default reproduces the paper.

Status: code and unit tests only. **No arm has been trained or evaluated yet.**

## Hypothesis
DTW-JPE rewards regression to the mean pose, and in the v2 replica the naturalness path behaves
like a time re-parametrisation of the geodesic (SD3 shift s ≈ 2.5). The plan first measures
properly (fixed lengths, shrink-to-mean frontier, kinematic ratios, retrieval evaluator) and checks
whether NG beats GE. It then tests training changes that could give real gains: hygiene,
position losses, noise schedules, guidance, source and self-conditioning.

## Flags (task → config key → experiment)

| Task | Config key(s) | Experiment | Code |
|---|---|---|---|
| C1 | `sampler.lengths_file` | every arm (fixed GT lengths) | `method.read_lengths`, `tools/lengths.py` |
| C2 | `flow.path ∈ {naturalness, geodesic, geoshift, ng_parallel, linear, linear_renorm}`, `flow.shift` | E1-gate, E1-H1, E1-lin | `paths.py` |
| C3 | `flow.train_shift`, `sampler.grid_power`, `sampler.shift` | E2-F | `paths.time_shift`, `flow.time_grid` |
| C4 | `generator.ema`, `generator.ema_eval`, `generator.text_lr`, `generator.warmup_steps` | E3 | `training.EMA` |
| C5 | `flow.joint_weights=lever_arm`, `flow.joint_weights_floor` | E4-c | `levers.py` |
| C6 | `flow.fk_loss.{lambda_body, lambda_hand, lambda_vel}` | E4-a, E4-b | `fk.py` (sparse SMPL-X LBS, matches the evaluator to 0.0004 mm) |
| C7 | `generator.p_uncond`, `sampler.cfg_w` | E5 | `model.drop_text`, `flow.guided` |
| C8 | `flow.source=wrapped_normal`, `flow.source_sigma` | E6 | `geometry.wrapped_normal` |
| C9 | `flow.self_cond`, `flow.self_cond_p` | E7 | `model.forward(sc=...)`, `flow.clean_estimate` |
| C10 | metrics `soke_{speed,accel}_ratio_{body,hand}`, `r_precision`, `frechet_distance`, `paired_bootstrap_ci`, `frontier_effect` | every arm | `src/slp/evaluation/metrics.py`, `analysis/score.py`, `analysis/evaluator.py` |
| E0.4 | — | trivial baselines | `methods/text_retrieval` (mean pose, TF-IDF retrieval) |

Design notes:
- **Lever-arm weights (C5).** A joint's weight is the squared lever arm of the points each
  reported DTW column measures: JPE body pelvis-aligned, JPE hand wrist-aligned, PA hand fingers
  only. Lever arms are taken on the SOKE skeleton at rest. Each column is normalised separately,
  because otherwise the 50 cm arm swamps the 3 cm fingers. Weights are floored at 0.1 × mean (the
  jaw) and normalised to mean 1.
- **FK loss (C6).** Squared distances are in cm². The initial `lambda_body=0.1, lambda_hand=1.0`
  in `experiments.py` is a rough scale match to `10 · loss_pose`. Check `loss_fk_*` against
  `loss_pose` in the first epochs before the sweep.
- **CFG on the manifold (C7):** x̂ = Exp_{x̂u}(w · Log_{x̂u} x̂c) per joint. The length head is
  always trained on the real text.
- **Wrapped-normal source (C8).** The source is centred on the per-joint mean rotation of the
  training frames (`source_mean.pt`). The correction network is trained with the same p0.

Not implemented on purpose (07 §2, P5–P6, conditional on earlier results):
- E9: separate hand head. Only worth building if hands remain the bottleneck after P2.
- E10: phonological prior. Needs reliable sign boundaries.
- E11: MFM-style metric. Dropped.

E8 (6D) already exists as `flow.representation=6d flow.path=linear`.

## How to run (the user runs; nothing here starts a job)
```bash
python -m methods.signmflow_plus.experiments list                  # arms, priorities, GPU-h
python -m methods.signmflow_plus.experiments plan --prio P0,P1     # exact commands, in order
# P0 preparation: fixed-length table (val GT), M1 evaluator + calibration
python -m methods.signmflow_plus.tools.lengths --data phoenix_smplx --out outputs/signmflow_plus/lengths/phoenix_smplx_val_gt.json
python -m methods.signmflow_plus.analysis.evaluator train --data phoenix_smplx
python -m methods.signmflow_plus.analysis.evaluator calibrate --data phoenix_smplx
# score arms against the reference B_s0 (shrink frontier, bootstrap CI, M1-M4)
python -m methods.signmflow_plus.analysis.score --data phoenix_smplx --reference <B_s0>/predictions.npz \
    --arm E3=<run>/predictions.npz --text-swap --evaluator outputs/signmflow_plus/evaluator/phoenix_smplx/evaluator.pt \
    --out outputs/signmflow_plus/scores/phoenix_E3.json
```
An arm counts as a real improvement only if the 95% CI of its matched-speed `frontier_effect`
lies below 0 and the M1/M2/M3 conditions of `docs/proposal7/04` §1.3 hold. All of this uses the
val split only.

## Results
| Experiment | Data | Seeds | Metric (mean ± std) | Run dir / notes |
|---|---|---|---|---|
| — | — | — | not run yet | — |

## Notes / log
- 2026-11-20: created as a copy of `methods/signmflow` (v2); C1–C10 implemented with unit tests;
  arm registry in `experiments.py`.

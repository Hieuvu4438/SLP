---
status: exploring   # exploring | promising | final | abandoned
created: 2026-11-16
---

# pma_flow — Posterior-Mean-Anchored residual flow (PMA-Flow)

Design document, research questions and gates: `docs/proposal5/SignNMFlow_Improvement_Proposal_vi.md`
(Sec. 3-5). Baseline: the SignNMFlow replication in `methods/signmflow`.

## Hypothesis
On small, weakly-conditioned SLP data, DTW-JPE is decided by how well the model estimates the
**conditional mean** in joint space, while back-translation is decided by **sharpness and text
dependence**. Splitting the two — a joint-space regression to the mean (Stage A) followed by a
short flow from that mean to the data (Stage B), with a knob `alpha` between them — should be
Pareto-better than SignNMFlow's flow from uniform noise.

## Method
State: 41 joint quaternions on (S³)⁴¹ (SOKE 133-D layout, physical SMPL-X rotations) + 10-D
expression, as in SignNMFlow.

| | Component | Where |
|---|---|---|
| A | Anchor μ_ψ(c, T): 8-layer decoder whose per-frame input is a learned query + absolute / relative position; x1-regression with the loss below | `model.py` (`mode='anchor'`), `flow.anchor_loss` |
| A' | **Cross-fitted training anchors**: K = 2 fold models predict μ for the half they never saw; the full-data model predicts val / test anchors | `method._fit_stage_a` |
| B | Residual geodesic flow: x0 = Exp_μ(σ ξ), ξ ~ N(0, I₃) in T_μS³; x_t = Exp_{x0}(t Log_{x0} x1); x1-prediction, φ also reads μ | `flow.flow_loss`, `flow.sample` |
| σ | Per-joint source scale = RMS of the cross-fitted residual Log_μ(x1) (`sigma: auto`) | `flow.residual_scales` |
| C | α knob: y_α = Exp_μ(α Log_μ x); α = 0 → μ, α = 1 → the sample; `export_alphas` writes the curve | `flow.interpolate`, `sweep.py` |
| D | CFG: text dropout 10 %; x̂ = Exp_{x̂_u}(w Log_{x̂_u} x̂_c), optional interval | `model.drop_text`, `flow._predict` |
| E | Hygiene: EMA, best-val (proxy) selection, logit-normal t, mBART lr 5e-5 + frozen 5 epochs, warm-up, clip, bf16 | `training.py` |

Loss (both stages, per frame, normalised by the batch's valid frames):

```
L = 10 Σ_j |q̂_j − q_j|²  +  100 · Huber_1cm(|p̂ − p|)  +  50 · Huber_2mm(|Δp̂ − Δp|)  +  0.1 |ê − e|²
```

p = the points the SOKE metrics read (12 upper-body points; 20 wrist-relative points per hand), from
`slp.skeleton_fk.SkeletonFK`: exact SMPL-X LBS restricted to the 55 skeleton joints and the 1,484
vertices the hand regressor / face points use (matches `SMPLXJoints` to 4e-7 m, differentiable,
~7x cheaper than the full mesh). Position term = mean(body) + mean(hands), as the table columns.

Not implemented (proposal 5, optional): the semantic feature loss of RQ4 (needs a separate
text–pose teacher) and the naturalness correction ε_θ in Stage B (RQ5).

### Pipelines (`method.pipeline`)
* `anchor_flow` — PMA-Flow (default).
* `anchor_only` — Stage A alone (= α = 0; the RQ2/RQ3 regression reference).
* `uniform_flow` — Stage B from the uniform hemisphere, no anchor: SignNMFlow-GE with the same
  loss / hygiene → the matched baseline. `configs/experiments/pma_flow/_common.sh` (`REPLICA`)
  turns everything off to re-implement SignNMFlow-GE exactly inside this code base.

### Choices made without experiments (defaults, to revisit after P1)
1. **Loss weights** 100 / 50: chosen so the three terms have the same order of magnitude on
   Phoenix train (48 clips): chordal×10 ≈ 3–5, position ≈ 0.03–0.05 m, velocity ≈ 0.035–0.09 for a
   GT blurred over 9 frames or GT + 0.05 rad tangent noise (static mean pose: chordal 1.95,
   position 0.11 m, velocity 0.036).
2. **Cross-fitting** (K = 2). Without it Stage B would see in-sample anchors that are much closer to
   x1 than test anchors and learn to copy μ. Cost: Stage A ≈ 2× one full training. At test time
   anchors come from the full-data model (slightly better than the fold models), so the source
   scale estimated on the folds is a little conservative.
3. **σ = auto** (residual RMS): the source covers the anchor's actual error, so the flow's support
   matches; `sigma_scale` 0.5 is the P2 alternative. A fixed float (rad) is also accepted.
4. **Checkpoint selection proxy**: frame-aligned mean point error (body + hands, metres) on 256 val
   clips, with the run's sampler (NFE 5, fixed noise). It is not the reported metric (no DTW), and
   it favours mean-like predictions in the same direction as DTW-JPE; set `train.select=last` to
   disable.
5. **α = 1.0** by default (the method's actual sample). The reported α must be picked on dev from
   `alpha_sweep.json`; never on test.
6. **bf16** autocast for both stages (fp32 in the replication). The matched comparison is
   `uniform_flow` vs `anchor_flow` inside this method; `REPLICA` uses fp32 for the sanity check
   against `methods/signmflow`.
7. **t sampling**: logit-normal(0, 1) mixed with 20 % uniform. Logit-normal alone almost never
   draws t < 0.05 (≈ 0.2 %), but the sampler's first call — the whole of NFE 1 — is at t = 0.
8. `align_pairs`: the flow pairs x0 with the representative of x1 in x0's hemisphere (shortest
   rotation); SignNMFlow pairs the sign-convention representatives (REPLICA sets it off).

## How to run
```bash
# P1 (RQ2): replica / hygiene / hygiene+FK / Stage A only - Phoenix dev, seed 0
bash configs/experiments/pma_flow/p1_rq2.sh
# P2 (RQ3, RQ4) after the gates: PMA-Flow, sigma x0.5, CFG w in {1.5, 2.5, 4}, NFE 1/10
bash configs/experiments/pma_flow/p2_rq3_rq4.sh
# single run / alpha curve of a finished run (val only)
python scripts/run.py --method pma_flow --data phoenix_smplx --name pma_flow_phoenix_smplx
python -m methods.pma_flow.sweep --run outputs/runs/pma_flow_phoenix_smplx/pma_flow/seed0_<stamp>
```
Re-use: `method.init.anchor_dir=<run>/checkpoints` (Stage A + anchor caches, symlinked),
`method.init.flow=<run>/checkpoints/flow_net.pt` (sampling-only variants: CFG, NFE, α).

Cost estimate (not measured): Stage A full + 2 half-data folds ≈ one 300-epoch generator
training; Stage B = one. Phoenix ≈ 2× the replication's ~10.5 h (fp32), less with bf16.

## Tests
`tests/pma_flow/test_pma_flow.py` (CPU, tiny nets): codec round trip, Huber, zero loss at the
target, exact micro-batch normalisation, source distribution moments (E|Log_μ x0|² = 3σ²), σ
recovery from synthetic residuals, sampler (NFE 1 = the t = 0 prediction; last step = x̂1), CFG
limits (w = 1 conditional, w = 0 unconditional), α endpoints, anchor-mode length dependence,
trainer (EMA, best-epoch selection, resume state). `tests/test_geometry.py`: S³ maps and the
sparse FK against the full SMPL-X FK.

## Results
| Experiment | Data | Seeds | DTW-JPE body/hand | B-T | Run dir / notes |
|---|---|---|---|---|---|
| — | | | | | not run yet (GPU busy with the SignNMFlow v2 replication) |

## Notes / log
- 2026-11-16: created from proposal 5; CPU smoke run (1-layer net, 16 clips, 2 epochs) of the full
  `anchor_flow` pipeline through `scripts/run.py` (`outputs/runs/pma_flow_smoke`).

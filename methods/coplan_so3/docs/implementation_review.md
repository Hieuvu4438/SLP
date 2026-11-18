# CoPlan-SO3 implementation review

Date: 2026-11-17. Scope: the proposed method (§6), mathematical invariants (§7),
A/B/C objectives and stage transitions (§9), executable integration (§10).
No optimizer steps or benchmark training were performed for this implementation.

## Requirement mapping

| Proposal | Current implementation | Evidence |
|---|---|---|
| §6.1 state / conventions | `geometry/so3.py`, manifest, SOKE codec | matrix/6D/axis-angle round trips, real FK audit |
| §6.2 factorization / no GT inference | `sampling.generate`, prior, length model | text/language/seed-only test; controlled track metadata |
| §6.3 four-branch plan AE | `models/plan_encoder_decoder.py`, `PartInput`, axial blocks | A backward; nonlinear finger heads; masked recon/NCE tests |
| §6.3 train latent coordinates and freezing | `TrainMoments`, `posterior`, `decode`, `freeze` | inverse-normalization, train-only and illegal-transition tests |
| §6.4 stochastic Euclidean plan FM | `flow.plan_fm_loss`, `PlanPrior`, `sample_plan` | B backward; analytic two-condition bimodal CFM oracle test |
| §6.5 plan-conditioned AR(1) source | `SourceConfig`, `flow.source` | stationary covariance/full-rank test; same source config in training/sampling |
| §6.6 exact geodesic target | `flow.interpolate` | finite difference, endpoints, seam test; no division by 1-s |
| §6.7 state-only kinematic metric | sequence features and `geometry/pullback.py` | finite difference JVP; explicit 2 JᵀJδ gradient; positive tangent floor |
| §6.7 physical-time and feature normalization | timestamp differences and frozen per-group/order RMS | nonuniform t² acceleration test; valid-pair/triple masks; padding invariance |
| §6.8 dense architecture | 8×512 axial transformer; text/plan cross-attention; language adapters | full default model GPU A/B/C backward; text/plan/language sensitivity tests |
| §6.9 auxiliary endpoint | separate nonlinear head, off by default | independent head backward test; sampler consumes velocity only |
| §6.10 current-state sampling | `sampling.lie_step` and `generate` | Euler/midpoint convergence, SO(3) validity, actual NFE budgets 5/10/20 |
| §7.3 minimizer conditions | detached base state; train scales; no endpoint confidence in KFM | direct Jacobian gradient check; no hidden-endpoint weights in core loss |
| §8 explicit schema / cache provenance | `MotionManifest`, `CanonicalMotionDataset`, `ProtocolDataset` | explicit mapping, hashes, split/schema/checksum failure tests |
| §8 reference FK | exact sparse shared SMPL-X skinning/regressor | full evaluator comparison on neutral, articulated poses and a training clip |
| §8 padding | collator identity/zero padding and valid masks | attention/output and loss padding-invariance tests |
| §8 train-only statistics | frozen feature RMS / latent Welford buffers | validation-fit rejection; masked observations excluded |
| §8.7 multilingual sampling | adapters, `language_sampler`, per-language output fps | Nᵅ probability and mixed-fps metadata tests |
| §9 A/B/C orchestration | `training.py`, `model.set_stage`, `Method.fit` | loss/gradient checks; plan frozen at B/C; C always includes prior FM |
| §9 length head | log-normal NLL, median/stochastic generation | text-only lengths, explicit max-frames failure tests |
| §10.2 numerical branches | Taylor Exp, quaternion-branch Log, 6D fallback with counters | identity gradcheck/gradgradcheck; near/exact π and degenerate 6D tests |
| §10.4 differentiable JVP | `autograd.functional.jvp(create_graph=True)` | real SMPL-X tangent gradients and central finite differences |
| §10.6 curved-path hook | `flow.exact_path_velocity` | noncommuting curved-path differential test |
| repository contract | local `Method(BaseMethod)`, config, README | registry/interface tests; no imports from other methods |

## Review findings addressed

1. **SOKE vocabulary is reduced.** Its tokenizer IDs cannot directly index the
   19,011-entry embedding table. The wrapper applies the released `map_ids.pkl`,
   including language tokens and unknown-token fallback. German, Chinese and
   English forward passes were executed with the real checkpoint.
2. **Raw axis-angle hand offsets and physical rotations differ.** The codec adds
   the SMPL-X hand mean in axis-angle before Exp; decoding subtracts it after Log.
   The real regressor comparison verifies FK equivalence rather than requiring
   identical axis-angle coordinates across a principal-branch change.
3. **Old local mBART config triggers an unrelated Mistral regex heuristic in
   transformers 5.5.4.** The SentencePiece mBART wrapper explicitly disables
   that regex alteration; source-language tokenization is retained.
4. **Padding must not change normalization.** Counts are per sequence and valid
   group, attention keys exclude padded frames/slots, and temporal terms require
   valid pairs/triples. Arbitrary finite padding was tested against unpadded output.
5. **Physical timing is absent from the existing cropped loader.** The adapter
   requires explicit timestamps/fps or deliberate verified timing metadata. It
   cannot infer physical seconds from a cropped feature tensor. Mixed-language
   output rates can be configured with `architecture.fps_by_language` (DGS/CSL/ASL).
6. **JVP can silently become a logging-only scalar.** The base state is detached
   while tangent predictions retain their graph. Finite-difference energy gradients,
   explicit Jacobian gradients, and real SMPL-X gradients all pass.
7. **Checkpoint shapes alone do not identify a generative law.** Loading checks
   manifest digest, posterior sigma, source AR/scales, latent stride, frame rates
   and metric normalization weights, and loads all frozen statistics strictly.
8. **Midpoint doubles evaluations.** Odd dense budgets end with an Euler interval;
   counters prove exact total flow NFE. Text encoder / coarse decoder cost is not
   disguised as flow NFE and remains part of pipeline runtime.

## Recorded execution evidence

- Full repository suite: **222 passed, 2 skipped**;
  [verification.log](verification.log) records the run and skip reasons.
  Final focused rerun after review: **46 passed** (CoPlan-SO3 plus method contracts).
- Real numerical audit: [smplx_audit.json](smplx_audit.json).
  Neutral/artificial articulated and real training-clip FK errors are below
  `2.4e-7 m`; real SMPL-X JVP/central-difference maximum discrepancy is around
  `1.2e-10` in float64.
- Full architecture GPU smoke: [full_architecture_smoke.json](full_architecture_smoke.json).
  **244,969,383 parameters**, real released mBART, all three objectives/backward,
  finite generated normalized `[8,133]` features and exact 2+3 NFE accounting.
  **Zero optimizer steps.** This is execution evidence for short sequences,
  not a throughput or quality benchmark.
- Exact environment and asset fingerprints: [runtime_environment.json](runtime_environment.json).

## Remaining experimental work

Training, semantic quality, ablation results, long-sequence memory/FLOPs and
benchmark equivalence to the original SignNMFlow cache remain experimental work.
The supplied SOKE schema does not certify unknown extraction/annotation revisions.
Per-clip provenance is enforced by the canonical JSONL cache format; official
split/parent-video overlap and published normalization provenance must be audited
for the actual chosen data before research runs.

Stage D multilingual pooling can use concatenated canonical/protocol datasets
with the balanced sampler; it shares the same A/B/C objectives and language
adapters. Stage E distillation, retrieval, full-mesh contact, and optional
noisy-state FK network inputs are extensions, not part of the main implementation.
Final scoring continues through the repository evaluator, not a new metric inside
this method. The endpoint auxiliary includes labeled physical velocity matching
alongside reconstruction; it is off in the main configuration.

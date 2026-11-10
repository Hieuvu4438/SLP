# Implementation review

Scope: the primary RACE-Sign v2 method in `docs/proposal3/05_revised_proposal.md`, including frozen-base integration, direct flow decoder, geometry losses, robustness preparation, coupled/sentence selection, staged training, calibration, checkpointing and native-length inference. No existing SLP experiment result is used to validate this implementation.

## Requirement-to-evidence audit

| Requirement | Implementation | Verification |
|---|---|---|
| Actual author tokenizer/generator, frozen | `adapters.py`, `madis_adapter.py`; strict module states; original VQ/mBART or MaDiS code imports | Author VQ/mBART fixtures, real SOKE tokenizer, real released MaDiS CSL integration |
| Exact SOKE data layout | 179D released statistics → indices 36:159 plus 169:179; raw poses remain 133D | Real author stats/checkpoint loaded strictly; `soke_tokenizer_integration.json` |
| Correct hand/rotation semantics | Hand means added in axis-angle **before** exponentiation; physical rotation 6D; mean subtracted on raw export | SciPy agreement; zero/pi/degenerate gradients; non-principal hand branches; licensed `smplx.SMPLX` reference |
| Frame-specific motion conditions | Code embedding → learned factor×width projection; native frame offsets, part identities, time/language embeddings | Projection gradients, window correspondence and padding invariance tests |
| Coupled body–two-hands and text conditions | Temporal, inter-part and cross-text attention; baseline relative wrist/head/shoulder geometry | Backward reaches code and part-interaction parameters; text/coupling controls available |
| Correct flow target/sampler | Linear path, target endpoint difference, one-step endpoint estimate, Euler and text CFG | Analytic endpoint test and actual forward-evaluation counts |
| Geometry supervision | Real SMPL-X; equal-weight part joint errors, orientation, velocity and relative hand/body geometry | Numerical reference and gradients with real licensed assets; finite CPU/CUDA training losses |
| Stage A and stage B | Teacher-token stage; nearest-code/repeat/part-dropout controls; optional held-out generated-token tuning | Role guards, deterministic corruption, explicit generator-holdout metadata, normalizer-preserving initialization |
| Training-only correspondence | DTW proxy and SO(3) averaging with length-ratio guard; no access from inference | Alignment rejects test roles; preparation rejects synthetic/generated-mode conflation |
| Expected-improvement selector | Paired surrogate-loss regression + auxiliary win classification; explicit switch cost; coupled blocks or whole sentence | Improvement sign checks, calibration affine recovery, surrogate reliability/coverage diagnostics |
| Preserve coordinated rotations | One body/LH/RH decision per block; shortest-path SLERP at switches; native-length boundary handling | Exact endpoint selection; switch blending invariant to padded batch length |
| No target/GT length at inference | `InferenceBatch`; text-only CLI rejects extra reference fields; base-generated lengths | Typed-input rejection; native-length and exact jaw/expression preservation tests; real text-only author-base run |
| Data/evaluator integrity | Explicit role + original official split; original-video disjointness; asset/codec fingerprints | Training and calibration require official train; test roles rejected; selector cannot use teacher tokens or decoder-development groups |
| Frozen candidate cache consistency | Cache SHA binds decoder; sampling/conditioning policy binds selector | Candidate/asset/policy mismatch failures and staged training checks |
| Save/load and continuation | Version 2 checkpoint; optimizer/RNG state, train/dev/source provenance; separate initialization/resume | Roundtrip prediction equality; bit-exact resumed versus uninterrupted CPU training with dropout |
| Full usable entry points | `prepare`, `train-decoder`, `cache-candidates`, `train-selector`, `infer`; registry bridge and asset fetcher | CLI import/help, method conformance and shared framework regression tests |

## Corrections made during implementation review

1. The author releases **179D**, not pre-cropped 133D, statistics. The loader now uses the exact author reduction rule; it does not infer a different schema from tensor length.
2. Using a flat-hand geometry layer would disagree with SOKE's nonzero hand mean. The learned representation now uses physical full rotations, including the mean before exponentiation, so angles outside the principal branch remain physically correct.
3. MaDiS Qwen may initialize in BF16 while projection heads are FP32. The adapter establishes consistent FP32 inference; the real checkpoint run exposed this mismatch before it was corrected.
4. Author MaDiS evaluation loads the full SFT state after tokenizer initialization. The adapter restores all VQ weights from that state as well as the LM, then checks the registered motion codebooks.
5. Window supervision keeps global token/frame correspondence and masks padding. Switching ramps are clipped by each sample's native length, not padded batch length.
6. Selector inputs/caches cannot silently use teacher tokens, the wrong decoder, different assets, or different sampling settings. Calibration groups remain separate from model-selection/training groups.
7. Generated robustness data cannot be called held-out when generator training membership is unknown; source-group metadata is required for that preparation.
8. Resuming a stage preserves previous best validation state, RNG/optimizer and ancestor training groups. Stage-B initialization preserves the learned normalizer and cannot train on prior validation groups.

## Runtime evidence and limits

`validation.json` records the final test/lint outcomes and exact covered scope. `author_integration.json` exercises the **full default 256-width, six-block, eight-step** RACE decoder and coupled selector on real pretrained MaDiS CSL outputs. It produced finite native-length `[1,56,133]` motion and preserved jaw/expression exactly. RACE weights and calibration in that integration are explicitly untrained, so the report is not a quality result. `soke_tokenizer_integration.json` validates the real public SOKE tokenizer/statistics; a trained SOKE signing-generator checkpoint was not silently substituted by the bare mBART initialization.

The decoder/selector training checks use new controlled synthetic fixtures. They demonstrate executable forward/backward, both stages, calibration and continuation under the tested contracts. They do not establish intelligibility, generalization, paper-score reproduction or SOTA. The latter require training on correctly prepared official data and external evaluation/human assessment. Full face generation remains outside the proposal's manual-motion MVP. Entropy/per-part selection and a post-refiner are separate proposed experiment controls, not components claimed to be part of this coupled implementation.

Source dependencies and author checkpoint provenance are listed in `NOTICE.md`, `upstream_sources.json`, and `pretrained_assets.json`. Downloaded model assets are ignored by Git and are not redistributed. Existing user edits elsewhere in SLP were preserved.

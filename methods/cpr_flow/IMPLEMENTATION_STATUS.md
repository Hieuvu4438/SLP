# CPR-Flow implementation status

Date: 2026-11-10. Authoritative specification: `docs/proposal4/CPR_Flow_Detailed_Implementation_Spec_vi.md`, implementation-spec v1.0. Code is isolated in `methods/cpr_flow/` and installed as editable package `cprflow`.

**Code delivered and engineering-tested. No real-corpus training, real benchmark, linguistic gate, or SOTA reproduction is claimed.** The specification permits unavailable real-data artifacts to be recorded with exact block reasons; those remain pending below. All fixture runs are explicitly `synthetic=true`.

## Milestone evidence

| Milestone | Implemented | Tested evidence | Real-data/training/evaluation state |
|---|---|---|---|
| M0 environment/source lock | Strict config inheritance/overrides, editable package, dependency snapshot, resolved Git/HF revisions and hashes, notices, run provenance/source archive | `verification/imports.json`, `package_build.log`, `package_contents.json`, `source_audit.log`, `unit_tests.log` | Installed/tested on Python3.13.5/Torch2.11.0+cu128. CPU execution tested; CUDA training and NCCL not tested. |
| M1 data | Authorized pinned downloader; exact nine-source routing; lazy readonly LMDB adapter; source/canonical conversion; NFC text; manifests, quarantine, full-utterance/long-clip ledgers; timing verification; SLERP/FIR resampling | `tests/test_schema.py`, `test_preparation.py` (nine **synthetic** LMDBs), `test_resample.py`; CLI gate failures in `verification/cli_acceptance.json` | Actual nine LMDBs, access record and verified feature timing absent. No real split counts or facial-basis compatibility inferred. |
| M2 geometry | Row6D, deterministic degeneracy fallback, robust geodesic loss, exact body/jaw/hand assembly, differentiable chunked matrix-input SMPLXLayer, named hand21 tips, relation33/pool2; offline ground-truth FK cache; skeleton renderer | Asset-backed tests in `test_gradients.py`, mapping tests in `test_rotations.py`, invariant tests in `test_relations.py`, offline/live FK equivalence in `test_geometry_cache.py`; `verification/render.log` | Existing licensed neutral expression50 asset used on CPU. Native fitting/video audit of 64 real clips/corpus is pending; no G0 approval generated. |
| M3 AE | Exact four-part deterministic stride2 AE, channel LayerNorm, true-length replicate boundaries, residual decoder fusion, identity-aware output biases, parameter/rotation/FK/velocity losses, EMA/dev selection | Padding output/gradient invariance, frozen-decoder→FK→latent nonzero gradients; AE2 optimizer updates in `verification/synthetic_integration.log` | AE48-real-clip smoke and full200k AE blocked by M1/G0. Fixture updates do not establish linguistic retention or real smoke convergence. |
| M4 text/duration/stats/cache | Pinned frozen mT5 encoder768; strict token-length policy; duration800→1024→2/log-normal; streaming equal-dataset/equal-clip moments; numeric content-addressed caches and invalidation | Actual pinned mT5 executed by CLI sample and three-route fixture evaluation; duration2 fixture updates; statistics tamper/leakage tests | Real train-fit statistics/cache and duration dev MAE pending. Fixture training text features are random768D and labeled as such. |
| M5 CFM/training/inference | Typed DiT, gathered windows, masks/adaLN/CFG; linear CFM; AdamW/AMP/EMA/scheduler/checkpoints; per-clip/global reductions; worker loader and exact replay; predicted-duration API; Euler/Heun | CFM4 + relational1 updates, exact resumed/uninterrupted weights and deterministic sample; 20-update resume with workers0/2; two-process CPU gloo reduction/gather tests; `test_no_oracle.py` | No real CFM parent200k/continuation100k trained. Generated fixture NPZ exists under the final integration artifact path logged below. |
| M6 relational/controls | P4→P5 copying/key gate, relation FM, geometry/cycle/lag/semantic auxiliary interfaces, sparse auxiliary schedule/ramp; B0 diffusion, B2 aux-only, B3 random stream, shared-AE mono configs | Nontrivial alpha0 B1/C0 output equality; DDIM indexing/ideal endpoint; lag+3/static-prediction tests; actual relation geometry/cycle backward in integration; parameter/tensor-count match in `runtime_cpu.json` | C0/B0/B2/B3/mono real training pending; no comparative gains asserted. Planned jobs in `EXPERIMENT_MATRIX.json` are not results. |
| M7 teacher/guard | Four-part teacher, masked reconstruction, multi-positive same-language negatives with gradient-aware global gather; frozen teacher input gradients; dev hash/calibration gates; bounded correction/rollback; separate C1/C2/C3 modes | Teacher2 fixture updates with actual pool2 explicitly logged; global contrastive-gradient regression; guard rejection/rollback, feature-only vs semantic, unapproved/hash-mismatch tests | Real teacher100k, annotated dev contrasts and `teacher_calibration.json` absent. Semantic auxiliaries/default guard disabled; full config refuses missing/invalid dependencies. |
| M8 evaluation | Per-corpus metrics, exact DTW path-average, global/local hands, decoded-geometry relation A, duration/time diagnostics, retrieval/FID/diversity utilities with annotation boundaries, bootstrap, failure/coverage ledgers, protocol rejection, sealed final-test interface | Three synthetic routes×two prompts×five seeds: 10/10 successful each, actual mT5 encoder; `verification/evaluation_fixture_summary.json`; protocol/oracle tests | No real dev/test benchmark. Official BT/SiBLEU/contact F1/penetration/independent sign-FID/human comprehension unavailable, not zero. Real test not accessed. |
| M9 reproducible experiments/profile | Resolved config/manifest/source hashes, environment/git diff/status/source archive, raw+EMA checkpoints, parent provenance, staged configs/scripts, planned seeds17/42/123 matrix, profiling CLI | Main384/8/6 model CPU forward/backward profile in `verification/runtime_cpu.json`; wheel/import/CLI checks | Real length-bucket100-warmup+200-timed/GPU profiles and confirmatory comparisons pending actual data/AE gates. CPU synthetic timings are not real-data throughput. |

## Acceptance checks and commands

From `/home/haipd/SLP`:

```bash
python -m pip install lmdb
python -m pip install -e 'methods/cpr_flow[dev,eval]' --no-deps
OMP_NUM_THREADS=2 python -m pytest methods/cpr_flow/tests -q
python -m pytest -q
python -m ruff check methods/cpr_flow
```

The method suite currently covers 49 tests; the final recorded count/result is the last line of `verification/unit_tests.log`. The parent suite is recorded independently in `verification/repository_tests.log` (131 passed,2 skipped on the inspected run). Parent skips belong to existing optional upstream tests, not substituted CPR-Flow passes.

From `methods/cpr_flow`:

```bash
python -m cprflow.cli.audit_sources --config configs/base.yaml --resolve-lock
python -m pip wheel --no-deps --no-build-isolation . -w runs/package_check
python scripts/experiment_matrix.py --output EXPERIMENT_MATRIX.json
OMP_NUM_THREADS=2 python scripts/synthetic_integration.py --output runs/verified_pipeline --smplx-model assets/smplx/SMPLX_NEUTRAL.npz
OMP_NUM_THREADS=2 python scripts/verify_fixture_evaluation.py --checkpoint runs/verified_pipeline/runs/cfm_fixture/checkpoints/last.pt --output runs/verified_pipeline/evaluation --summary verification/evaluation_fixture_summary.json
OMP_NUM_THREADS=2 python -m cprflow.cli.render --motion runs/verified_pipeline/sample.npz --smplx-model assets/smplx/SMPLX_NEUTRAL.npz --output runs/verified_pipeline/sample_preview.mp4
OMP_NUM_THREADS=2 python scripts/verify_runtime.py --output verification/runtime_cpu.json
```

Integration output directories must be new. `verification/synthetic_integration.log` records exact sample/checkpoint paths, hash, update counts, actual teacher pool, finite relation losses, and device. The fixture generator uses hidden24/blocks2 explicitly; the separate profile uses the main hidden384/blocks8/heads6 architecture. No small preset is reported as a main-model training run.

The installed module import inventory and wheel content check prove that `cprflow.data.*` and `defaults.yaml` are packaged. Raw data, runs, reference snapshots and licensed assets are excluded from the wheel/git source listing.

## Concrete missing artifacts / next real-data gates

Authoritative path-by-path audit: `verification/asset_audit.json`.

- All nine `data/raw_pose/9fe7e04ef59204c787cd6094c747f889efff6ba0/{train,dev,test}/<corpus>_reopt_<split>.lmdb/data.mdb` files are absent. Exact paths and corpus prefixes are listed in that audit. Source: https://huggingface.co/datasets/LionelLow/SignSparK_data .
- `access_record.json` absent. Lab must record existing source-corpus authorization; downloader does not invent authorization or send access requests.
- `source_timing.json` absent. Effective pose sampling FPS needs release preprocessing evidence or original annotation/video verification. Raw video25/30FPS is not accepted as proof.
- `runs/geometry_audit/review.json` absent. Real native-fitting renders, correct left/right mapping and face basis need review with current asset/source/manifest hashes and64 UIDs/corpus. Code does not manufacture approval.
- Real `runs/ae_joint3_s17/checkpoints/best_ema.pt` and `runs/duration_joint3_s17/checkpoints/best.pt` were not supplied/trained. The duration trainer writes `best.pt` and `best_ema.pt` with raw+EMA state; selection uses EMA. AE/duration/latent-cache/generator stages must run in order after G0.
- Real teacher checkpoint/dev contrast annotations/calibration absent. Guard gate remains closed; geometry-only C1 is explicitly a research control without a semantic score or protection claim.
- Audited official backtranslation adapter/checkpoint and real-fit/AE sanity suite absent. No LLM captioner or different recognizer is substituted. Contact/NMF/human labels and independent evaluator feature release are not supplied.

No SMPL-X model, raw corpus, generated weights or evaluation reports are redistributed as source. Local assets are ignored; only the actual asset hash is recorded.

## Important implementation details and deviations made explicit

- The package/evaluator are colocated inside this method directory as required by the supplied spec. The SLP `Method` bridge obeys input-only prediction; it delegates final metrics to the separate `cprflow` evaluation CLI.
- Environment was already Python3.13/Torch2.11, rather than the suggested starting Python3.10/3.11. Those other versions and CUDA/NCCL training are not labeled tested.
- SourceFPS>target anti-alias was unspecified beyond requiring a versioned policy. `quaternion_fir_v1` is documented in README and included in preprocess hashes; high-frequency attenuation/endpoint tests exist. Linguistic retention remains an empirical gate.
- Teacher training uses a full logical negative pool with differentiable DDP gather. It does not pretend accumulated independent microbatches provide64 negatives. The small fixture pool2 is explicitly reported.
- C1/C2/C3 inference modes share the same backtracking/gradient caps. C1 reports no semantic score; C2 uses calibrated feature budgets; C3 adds semantic score/confidence budgets. Genuine calibration is still required for C2/C3 in the public runtime.
- OracleR is an explicitly labeled diagnostic extension with normalized-time interpolation and a clamped linear relation path; it does not silently imply GT duration. OracleT/R results are rejected from main aggregation.
- G0 native fitting/video comparison, real smoke improvement, AE contrast ceiling, calibration power/sensitivity, linguistic noninferiority and downstream benchmark gates are not inferred from unit tests.

## Failures encountered and resolved

- An asset-backed static-activity backward test exposed NaN derivatives from `sqrt(mean(square(v)))` at zero. Replaced it with the equivalent vector norm divided by sqrt(width); static backward now passes.
- Initial renderer used auto-selected PyAV without an explicit codec, then the explicit FFMPEG plugin was found missing in the lab environment. Renderer now selects FFMPEG/libx264; optional `imageio-ffmpeg` was installed and dependency lock updated with an environment-update record. Render outcome is in `verification/render.log`.
- Padding invariance, all-masked attention, DDP microbatch scaling, global contrastive gradients, cache numeric tampering, unapproved guard calibration, invalid-candidate rollback and worker prefetch/resume are regression-tested.
- Real import/timing/train commands intentionally fail at missing authorization/timing/G0 artifacts; exact commands, nonzero return codes and reasons are in `verification/cli_acceptance.json`. These are blocked real runs, not failed implementation tests disguised as passes.

## Completion boundary

The requested implementation and reproducible engineering checks are delivered. Large training, dataset access/FPS audit, approved calibration, official benchmark comparison and human study remain pending external artifacts/experimentation. Do not describe this implementation as trained, reproduced, semantically certified, or experimentally better until those separate gates actually pass.

## Follow-up synthetic GPU compute timing

A later user question about training duration prompted a short component benchmark on the local RTX5880 Ada48GB. Results: `verification/gpu_training_compute_synthetic.json`; driver: `verification/gpu_training_compute_driver.txt`. It uses initialized main-model weights, synthetic lengths128/256/512, BF16 and strict deterministic algorithms, then extrapolates measured microbatch costs to logical batch128(AE)/64(flow). This is not a real-data CUDA train/resume integration or a throughput100-warmup+200-timed benchmark. For AE200k + duration30k + CFM parent200k + C0 continuation100k, estimated pure compute is approximately31–44days on one dedicated GPU, before real I/O, preprocessing, dev validation, final evaluation or restarts. Real sequence distributions and full semantic/confirmatory runs remain unmeasured. No real dataset was downloaded or trained in this follow-up.

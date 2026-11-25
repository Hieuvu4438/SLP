# CoPlan-SO3

status: exploring

Implementation of the **method in §§6–7, A–C in §9** of
[the proposal](../../docs/proposal6/CoPlan_SO3_Research_Proposal.md).
Code and numerical execution have been checked. Benchmark quality remains to be evaluated.

## Implemented

- Four motion branches with spatial/temporal attention, nonlinear finger
  projectors, per-finger outputs and high-resolution hand skips.
- Motion plan encoder/coarse decoder, stride-4 slots, train-only latent
  normalization with its inverse, normalized-coordinate noisy posterior,
  frozen plan distribution after A, conditional Euclidean FM prior.
- Reconstruction with labeled physical velocities, same-language multi-positive
  contrastive loss with explicit false-negative exclusions, masked reconstruction.
- Log-normal length head; main inference uses its rounded median. Controlled
  lengths are explicitly marked in generation metadata. Resource overflow raises.
- Plan-conditioned full-rank AR(1) source on rotations and expression;
  principal SO(3) geodesic with exact body-angular-velocity targets.
- Matrix-free KFM with FK, torso-relative wrists, hand-local fingertips,
  palm orientations, cross-hand relationships, physical-second velocity and
  acceleration. Every group/order uses frozen train RMS and valid counts.
- Dense text/plan-conditioned flow with separate physical/flow time embeddings,
  DGS/CSL/ASL adapters, optional independent endpoint head (off by default).
- Current-state Lie Euler/midpoint; **total** plan+dense NFE budgets 5/10/20.
  Odd midpoint budgets use a final Euler interval and report actual calls.
- Opt-in A/B/C optimizer orchestration, stage checkpoint selection on validation
  objectives, atomic save/load with manifest and source/posterior-law checks.
  Shared text fine-tuning keeps prior FM active in C to avoid condition drift.
- Repository `BaseMethod` integration: `fit(train, val)` and input-only `predict`.
  Final metrics remain in `slp.evaluation`; no test split is accessed by the method.

## Run checks, without training

From the repository root (PyTorch, NumPy, pytest and ruff installed):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/coplan_so3 tests/test_methods.py -q
python -m ruff check methods/coplan_so3 tests/coplan_so3
```

With the existing licensed SMPL-X/SOKE assets installed:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m methods.coplan_so3.audit \
  --report methods/coplan_so3/docs/smplx_audit.json
```

This checks real model/regressor conventions, codec equivalence, JVP finite
differences and tangent gradients. It does not update model parameters. The
asset-bound schema is in `configs/soke_manifest.json`. To recreate that schema
for a verified cache frame rate, pass `--write-manifest PATH --fps FPS`.
This **does not certify a different cache as SignNMFlow's original data**.

`smoke.py` exercises full configured architecture, mBART, all stage backward
passes and sampling with a short **training** clip, with no optimizer steps:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m methods.coplan_so3.smoke --device cuda
```

## CSL-Daily A/B pilot

From the repository root:

```bash
bash configs/experiments/coplan_so3/pilot_ab.sh RUN_DIR SIGNMFLOW_DEV_PREDICTIONS.npz
```

The launcher holds one local CoPlan lock per GPU. It checks for at least 16 GiB free and
at most 5% utilization three times, 30 seconds apart, before starting the Python job.
Set `COPLAN_WAIT_PID` to wait for a particular training process. In this mode the
launcher checks its process identity every 30 seconds and starts after that process
exits and at least 16 GiB is free, regardless of GPU utilization from other jobs.
Use `COPLAN_GPU_INDEX` and `COPLAN_PYTHON` to select the device index and interpreter.
`RUN_DIR/status` and `RUN_DIR/job.log` show the queue and training state.

The job trains A for 20 epochs and B for 30 epochs on fixed CSL-Daily subsets
(2,000 train and 256 dev, seed 0). It retains the original indices from the MaDiS crop,
drops repeated indices, and uses the dataset's native 30 fps. The dev diagnostic
holds reference duration and latent noise fixed for each paired text swap across
three sampling seeds. The wrong sentences are selected by character overlap and
clip length, with different text and gloss. A frozen CSL back-translator scores
the reference transcript's mean token log probability. The named SignMFlow dev
archive supplies the local baseline on the same dev IDs.

The output includes the ID and configuration manifest, stage checkpoints, paired
motion archives, individual clip scores, and `report.json` with paired bootstrap
intervals. Interpret the prior gap alongside the ground-truth text matching control
and the oracle reconstruction. Stage C is a separate experiment.

## API and data contract

`factory.build(hparams)` assembles the production SOKE model and codec from
`config.yaml`. mBART uses the local `mbart-h2s-csl-phoenix` checkpoint, per-language
source tokens and `map_ids.pkl` for its reduced vocabulary. `ByteTextEncoder` is
an explicitly enabled offline test fixture.

The low-level `CoPlanSO3` accepts a `MotionManifest` and a differentiable FK
backend. `data.dataset.collate` consumes dictionaries with `rotations[T,J,3,3]`,
normalized `expression[T,E]`, `timestamps[T]` in seconds, `text`, and integer
`language` (`DGS=0, CSL=1, ASL=2`). Padding is identity/zero and masks are True
for valid entries. No root translation is added.

`CanonicalMotionDataset` reads NPZ motions plus JSONL clip/parent-video/dataset IDs,
annotation/extractor versions, split, language and file/schema checksums. It verifies
cache integrity and never resamples frames. It is usable directly with `train_stage`.

`ProtocolDataset` adapts existing normalized SOKE `Split` arrays. **Physical
timing must be supplied**: per-clip timestamps or actual post-preprocessing
`fps`, or deliberately verified `split.meta['timing_verified']=True` and
`split.meta['fps']`. The existing `madis_crop`/resampling path omits timestamps;
this adapter therefore fails instead of silently calling every resampled
frame a native 25-fps frame. Audit the reference preprocessing before enabling
that flag. For pooled data, configure `architecture.fps_by_language` with verified
output frame rates in DGS/CSL/ASL order and `language_sampling_alpha` in each stage
(0.5 or 1.0). The balanced sampler also supports concatenated datasets.
There is no resampling, mirroring, confidence fabrication or joint
mapping inference in this method.

All production asset paths must be listed in the manifest hashes. The supplied
manifest binds the local SOKE schema, shape, fixed rotations, hand mean, SMPL-X
asset and released normalization. Clip IDs, parent-video IDs, annotation/extractor
versions and checksums must accompany the chosen dataset cache; the schema alone
does not establish split independence or original-paper protocol equivalence.

After A, collect raw encoder statistics **from train only**, finalize, then call
`model.plan.freeze()`. `set_stage('B')` / `set_stage('C')` enforce this ordering.
The normalized posterior noise is shared by both stages. Feature RMS statistics
must also be fitted on train before reconstruction/KFM. Inference is:

```python
from methods.coplan_so3.sampling import generate

# model loaded from an A/B/C checkpoint with frozen plan and statistics
out = generate(model, ["Guten Morgen"], [0], seed=0, total_budget=10)
# out.rotations is physical SO(3); out.expression is normalized.
# codec.expression_raw(out.expression) returns SMPL-X expression coefficients.
# codec.decode(...) returns existing normalized SOKE [T,133] evaluation features.
```

Generation takes text, language and noise only in the main track. It never
accepts a motion reference, gloss, keyframes, or a reference-based seed selector.
`Method.predict` retains generation metadata in `last_generation_metadata`.
`Method.load` rebuilds from the saved configuration and checks manifest/noise laws.

## Scope and limitations

The optional endpoint loss uses the labeled reconstruction/velocity loss as
an auxiliary objective; it does not drive sampling. Its weight defaults to 0.
Backbone geometry runs in float32; mathematical audits use float64. Mixed
precision, EMA, gradient accumulation, full-mesh contact/collision constraints,
retrieval, phase models and distillation are future experiments, not enabled
parts of the core method. The optional noisy-state FK input is omitted; FK is
used in the pullback objective.

The generic tree backend is for explicit alternate skeletons and tests. The
production backend uses the exact shared sparse SMPL-X skinning/regressor,
including hand-tip vertices and expression blend shapes. It does not claim
valid SO(3) rotations imply anatomically or linguistically valid signing.

See [implementation review](docs/implementation_review.md) for requirement-to-code
mapping and verification evidence. Default epoch/LR values are starting settings
from the proposal; successful training and semantic quality remain untested.

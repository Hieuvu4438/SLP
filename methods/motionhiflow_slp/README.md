# MotionHiFlow for sign language production

status: exploring

Gloss-free text-to-motion generation for Phoenix-2014T (DGS), CSL-Daily (CSL), and How2Sign (ASL).
This adaptation uses the hierarchical flow paths and the denoise–upsample–renoise transition from
[MotionHiFlow](https://arxiv.org/abs/2604.23264). The source checkout used for reference was
`third_party/MotionHiFlow` at `e0a1605`.

## Model

- A temporal VAE learns the repository's normalized 133-D SMPL-X layout. The body, left hand,
  right hand, and face have separate input and output projections. Temporal blocks mix their
  features. Training uses part-weighted reconstruction, first differences, KL, and reconstruction
  after temporal downsampling. The face channels contain jaw rotation and ten expression coefficients.
- After codec training, its weights are frozen. The flow models posterior means normalized with
  statistics estimated on training clips. The default stages use scales `[0.3, 0.6, 1.0]`.
  Each scale uses the same initial Gaussian sample resampled on the clip's valid temporal grid.
  `flow.gamma: 1.0` follows the released scheduler's corrected start sigma. Setting it to `null`
  uses the boundaries in the paper.
- The DiT uses modality-specific blocks followed by shared blocks with joint text-motion attention
  and rotary encodings for time and signing parts. A pretrained mBART encoder supplies token features.
  A duration head predicts the frame count from text and language. The default sampler uses Euler
  integration and classifier-free guidance.

The codec, rotary layout, and text encoder are specific to the signing representation. Train this
codec and generator with the signing data. The HumanML3D/KIT-ML checkpoints use different inputs
and latent shapes.

## Run from the SLP root

The existing data configs and shared evaluation handle the official splits and SOKE-compatible FK.
Use an environment with PyTorch 2.6+, transformers, sentencepiece and smplx. The default mBART
directory is `third_party/SOKE/deps/mbart-h2s-csl-phoenix`. See
[data/README.md](../../data/README.md) for the SMPL-X data and normalization assets.

```bash
python scripts/run.py --method motionhiflow_slp --data phoenix_smplx \
  --name motionhiflow_phoenix --seeds 0
```

Use `--data csl_smplx` or `--data how2sign_smplx_soke` for the other datasets. The latter uses
SOKE's How2Sign frame selection and exclusions. Choose `how2sign_smplx` where the comparison
uses the MaDiS protocol. The default runner scores dev.

A quick training check on a small subset can use `method.debug_limit=16`,
`method.codec.updates=2`, and `method.flow.updates=2`. The runner still scores the full dev split.
The byte encoder (`method.text.backend=bytes`) is available for offline smoke tests. For mBART,
`method.text.finetune=true` enables encoder fine-tuning with a separate learning rate.

## Compare temporal scales

Use a trained codec and a common set of generated durations to isolate the effect of scale.
Set both paths below to artifacts from the three-scale run with the same split and seed.

```bash
REFERENCE_RUN='outputs/runs/motionhiflow_phoenix/motionhiflow_slp/<run_id>'
python scripts/run.py --method motionhiflow_slp --data phoenix_smplx \
  --name motionhiflow_phoenix_single_scale --seeds 0 \
  method.init_codec="$REFERENCE_RUN/checkpoints/codec.pt" \
  'method.flow.scales=[1.0]' \
  method.sampler.lengths_from_npz="$REFERENCE_RUN/predictions.npz"
```

The duration lock reads only the frame counts from those generated sequences. Leave it unset
for independent text-to-sign evaluation. A two-stage ablation can use `[0.5, 1.0]`.
With a trained checkpoint held fixed, tune guidance and step count on dev. The sampler makes
exactly `sampler.steps` Euler evaluations, with both conditional and unconditional predictions
at each step when guidance is active.

The [Phoenix recipe](../../configs/experiments/motionhiflow_slp/phoenix.sh) trains the one- and
three-scale variants with the same codec, durations, seed, and update budget. Its `finish` phase
chooses from snapshots at 8k, 16k, and 24k updates using hand DTW-JPE on fixed tuning broadcasts,
then scores the selected models on all dev clips and on the remaining broadcasts. It trains an
otherwise matched run with auxiliary fingertip and wrist supervision on the selected scale schedule.
`method.flow.endpoint_fk_weight` enables that loss, which uses the clean endpoint estimate at
scale 1.0. The SMPL-X statistics and rig match the shared SOKE evaluator.

## Outputs

- `checkpoints/codec.pt` contains the selected codec and train-only latent normalization.
  Use `method.init_codec` to reuse it with the same codec architecture.
- `checkpoints/model.pt` contains the selected full generator and codec. Restore it with
  `Method.load(run_dir / "checkpoints")` and the matching model config. A frozen mBART encoder
  is loaded from the configured model directory. Fine-tuned text weights are included.
- `predictions.npz` contains normalized `[T, 133]` arrays keyed by clip name. It can be scored
  with the repository's back-translation tools. `metrics.json` contains SOKE-style DTW-JPE
  and DTW-PA-JPE from the shared evaluator. `metrics_history.jsonl` tracks the codec and flow
  validation losses used for checkpoint selection.

## Verification

The CPU tests cover masking and variable-length clips, the cross-scale transition with oracle
velocities, and training, checkpoint reload, and export. CUDA smoke tests with the local mBART
and real SMPL-X clips exercised the train–predict–score path for all three datasets. Those smoke
runs used three training clips and two validation clips per dataset and one update per stage.
Full benchmark training and quality comparisons remain to be measured.

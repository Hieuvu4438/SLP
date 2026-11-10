# RACE-Sign v2
status: exploring

Implementation of [proposal v2](../../docs/proposal3/05_revised_proposal.md): frozen author motion tokens → text-conditioned continuous flow decoder → expected-improvement selector coupled across body and both hands. This is a complete staged implementation, **not a trained RACE checkpoint or a SOTA result**. Existing SLP pilot results and locally trained experiment checkpoints are not used as evidence.

## What is implemented

| Module | Contract |
|---|---|
| `adapters.py` | Exact author SOKE VQ/mBART modules, strict checkpoint loading, physical 133D poses, frozen codebook/text features, native lengths |
| `madis_adapter.py` | Optional real pretrained MaDiS base using the same 133D contract; dataset-specific checkpoint language is explicit |
| `schema.py`, `rotations.py` | Raw SOKE ↔ physical 240D manual rotations; SO(3) projection, stable zero/pi conversion, shortest-path SLERP |
| `geometry.py` | Real licensed SMPL-X, including fingertips and hand means; explicit skeleton FK for controlled checks |
| `model.py`, `flow.py` | Learned subframe token projection, temporal/inter-part/text attention, language/time conditions, endpoint-consistent CFM, Euler/CFG with actual NFE |
| `losses.py` | Masked flow, per-part joint/orientation/velocity and bilateral/wrist-to-body relational training surrogates |
| `data.py`, `preparation.py` | Explicit split roles/video groups, teacher/generated caches, window/halo correspondence, code corruption, training-only DTW correspondence |
| `selector.py` | Paired expected-improvement regression, auxiliary win classifier, held-out affine calibration, joint block or sentence decisions and rotation blending |
| `training.py` | Decoder training, optional stage-B initialization, exact epoch resume, selector training/calibration, strict versioned checkpoints/provenance |
| `pipeline.py`, `cli.py`, `method.py` | Target-free native-length inference, outputs/export, staged CLI and SLP registry bridge |

The code does not automatically download a dataset, split one arbitrarily, reuse an SLP experiment, or start a long training job. Training, candidate caching and inference are separate commands with explicit inputs.

## Installation and author assets

The verified runtime is recorded in `runtime_tested.json` (including Transformers 5.5.4 and Hugging Face Hub 1.14.0); declared dependencies include that tested combination. Run from the repository root:

```bash
pip install -e '.[race-sign,dev]'
python -m methods.race_sign.cli --help
```

The author repositories are dependencies, not copied code. Use the fresh source checkouts under `third_party/proposal3_sources/code/`, or clone the official repositories listed in [NOTICE](NOTICE.md). Two different `mGPT` checkouts cannot be imported into one Python process; the adapter fails explicitly instead of mixing SOKE/MaDiS modules. CLI stages run in separate processes.

For SOKE, copy `assets.example.json`, fill **absolute paths** to the author tokenizer, signing-generator checkpoint, reduced-vocabulary mBART folder and author mean/std. Released 179D statistics are reduced with the author’s exact index rule; already-reduced 133D statistics are also accepted. `pytorch_model.bin` inside the mBART folder is not automatically a trained signing-generator checkpoint. Loading is strict; no random-head or partial-weight fallback. The public SOKE tokenizer was downloaded separately and its provenance is recorded in `pretrained_assets.json`.

The optional MaDiS base has an official per-language generator release. Its complete SFT VQ state is restored after tokenizer initialization, matching author evaluation; strict codebook equality is then checked against the language-model buffers. The adapter uses consistent FP32 inference for the frozen Qwen model and projection heads. For CSL:

```bash
python -m methods.race_sign.fetch_assets \
  --output data/pretrained/race_sign/madis \
  --upstream-repo third_party/proposal3_sources/code/MaDiS
```

`--local-download-cache` may point to a previous **author download cache**; reuse is allowed only after binary SHA256 matches the author's Hugging Face LFS object ID. This never discovers or loads a locally trained experiment. The command verifies tokenizer, CSL generator, stats and Qwen archive, extracts the model, and writes `assets.json` and `download_provenance.json`. Keep the MaDiS tokenizer paired with its generator; the SOKE and MaDiS releases are not assumed to have identical codebook values.

SMPL-X model files are separate licensed assets. Supply the directory containing `SMPLX_NEUTRAL.npz`. No model files or signing datasets are bundled. The geometry provider fixes the 10 shape coefficients used in SOKE; custom coefficients change the geometry signature and must match between training and inference.

## Rotation convention and geometry

Raw input/output is `[T,133]`: body 30, left hand 45, right hand 45, jaw 3, expression 10. Coordinates are **physical axis-angle radians**, not the author's standardized VQ inputs. The adapter alone applies/deapplies the author's 133D mean/std.

The learned flow space is `[T,240]`: 10 + 15 + 15 full physical rotations, six dimensions each (first two matrix **rows**). For hands the SMPL-X mean is added **before** exponentiation; after generating physical rotations, the mean is subtracted in axis-angle coordinates when exporting raw SOKE poses. This preserves physical rotations even when a raw angle is outside the principal ±pi branch. RACE's learned 240D normalization is fitted only on decoder training poses and stored in the checkpoint. Jaw/expression are copied exactly from the frozen baseline at valid frames.

`SMPLXGeometry` is a manual-motion training surrogate with zero jaw/expression and fixed shape, not a replacement for the paper's full benchmark evaluator. It uses full hand rotations and mesh-derived fingertips. `SkeletonGeometry` accepts an explicit 55-joint tree/rest pose and has zero pose mean; it is useful for analytic tests, not a silently substituted production skeleton. Geometry assets/shape/codec and base checkpoints/stats are fingerprinted; mixed caches and incompatible inference providers fail.

## Data preparation

Assign source videos to five non-overlapping development groups **outside** the method:
`decoder_train`, `decoder_val`, `selector_train`, `selector_val`, `selector_calibration`.
Keep the paper test split untouched. Every raw/cache manifest carries `official_split`: trainable stages (decoder, robustness, selector training and calibration) require the original `train` split; only validation roles may use `dev`. The five internal groups can be chosen within the original training split, with original dev used for model selection as declared. Each raw JSONL row is:

```json
{"sample_id":"clip_001","source_group":"original_video_001","role":"decoder_train","official_split":"train","text":"今天的天气很好。","language":"csl","pose":"poses/clip_001.npy"}
```

The `.npy` contains finite physical `[T,133]` SOKE poses, without normalization or root/lower-body coordinates. `pose` is relative to the raw manifest. Supported languages are `how2sign`, `csl`, `phoenix`. A MaDiS CSL checkpoint accepts only `csl`; train a separate RACE model for each per-language MaDiS checkpoint rather than mixing different generator assets in one cache. Group identity must represent the original recording, so adjacent sentences from the same video cannot cross development groups.

```bash
python -m methods.race_sign.cli prepare \
  --raw /data/raw_decoder_train.jsonl \
  --assets data/pretrained/race_sign/madis/assets.json \
  --role decoder_train --mode teacher \
  --output data/processed/race/decoder_train --device cuda
```

Repeat for `decoder_val`. Teacher preparation encodes the target and reconstructs a baseline with the actual author VQ modules. Target tails shorter than one four-frame token are explicitly trimmed; target and frozen-token timeline agree. Frozen encoder states and codebook embeddings are cached, not optimized by RACE.

For selector data, prepare `--mode generated --align-target --geometry /licensed/smplx`. The generator receives only text/language; target motion is used **after generation** to create training correspondence. DTW with a length-ratio guard is only a noisy geometric training proxy, never semantic alignment ground truth. It is unavailable to inference. Sequences with missing/reordered signs should be reviewed/excluded upstream; the method cannot certify their correspondence. Cache source hashes, roles and preparation mode are stored.

## Train the decoder

```bash
python -m methods.race_sign.cli train-decoder \
  --config methods/race_sign/config.yaml \
  --train data/processed/race/decoder_train/manifest.jsonl \
  --val data/processed/race/decoder_val/manifest.jsonl \
  --geometry /licensed/smplx --device cuda \
  --output outputs/race/decoder --epochs 100 --batch-size 8
```

`last.pt` contains optimizer/RNG state for `--resume`; `best.pt` is selected by deterministic validation loss. `--epochs` is the total intended epoch count, not additional epochs. `--window 0` trains whole sequences. Local windows retain global token-to-frame offsets; halo provides boundary context while only the center is supervised. `--no-text` and `--no-coupling` provide decoder controls.

Optional stage B: prepare `robustness_train` records with nearest-code corruption (`--mode teacher --corruption 0.1`) or explicit generated-token/target alignment. Additional synthetic controls are `--repeat-probability` (short token repeats) and `--part-dropout` (zero one or more code-condition streams, with the frozen baseline retained). These controls are limited to the explicit teacher-based robustness stage and cannot be silently combined with generated-token preparation. Fine-tune with `--robustness --initialize outputs/race/decoder/best.pt`. The normalizer is retained, prior training/video provenance is accumulated, and previous validation groups cannot become training data. `--resume` is for exact continuation and cannot be combined with `--initialize`. Generated-token robustness tuning additionally requires `generator_train_groups` in the asset configuration (a JSON list or a path to a JSON list of original video IDs), and the preparation rejects any overlapping source video. The author release is not assumed to be out-of-fold; when its generator training membership is unknown, use the teacher/synthetic stage rather than relabeling in-sample generation as held-out.

## Cache candidates and train the selector

For each selector role, use a manifest prepared from that group's generated tokens:

```bash
python -m methods.race_sign.cli cache-candidates \
  --checkpoint outputs/race/decoder/best.pt \
  --manifest data/processed/race/selector_train/manifest.jsonl \
  --role selector_train --geometry /licensed/smplx --device cuda \
  --output data/processed/race/selector_train_candidates
```

Repeat for `selector_val` and `selector_calibration`. Candidate caches are bound to the decoder checkpoint SHA256 and sampling settings, and cannot overlap any decoder training/validation video group.

```bash
python -m methods.race_sign.cli train-selector \
  --checkpoint outputs/race/decoder/best.pt \
  --train data/processed/race/selector_train_candidates/manifest.jsonl \
  --val data/processed/race/selector_val_candidates/manifest.jsonl \
  --calibration data/processed/race/selector_calibration_candidates/manifest.jsonl \
  --geometry /licensed/smplx --device cuda \
  --output outputs/race/selector.pt --epochs 30
```

The target is paired surrogate improvement, including an explicit local switching penalty. It is regression, not merely classification/confidence. Calibration is fitted on its separate group after model selection. Surrogate MAE, auxiliary win Brier score, reliability bins and coverage/improvement diagnostics are stored in checkpoint provenance; these are training diagnostics, not final SLP benchmark metrics. `--whole-sentence` trains a sentence selector; use that checkpoint only with `--policy sentence`. Block selectors use `--policy coupled`. The head sees native tokens, text and both candidates, never a test target. It does not save flow compute: both alternatives are generated. Changing sampling steps/CFG/conditioning after selector calibration is rejected.

## Text-only inference

Each inference JSONL row has **only** `sample_id`, `text`, `language`; target, gloss, reference length or extra fields are rejected:

```json
{"sample_id":"request_001","text":"今天的天气很好。","language":"csl"}
```

```bash
python -m methods.race_sign.cli infer \
  --checkpoint outputs/race/selector.pt \
  --assets data/pretrained/race_sign/madis/assets.json \
  --inputs /data/requests.jsonl --geometry /licensed/smplx \
  --device cuda --policy coupled --output outputs/race/predictions
```

`base`, `flow`, `random`, `coupled`, `sentence` policies are implemented. A selector is required only for the last two. Per-sample flow noise is reproducible from seed and sample ID, independently of batch padding. Outputs are `.npz` files with physical `pose`, `baseline`, `candidate`, `weights`, plus a manifest containing native lengths and actual model NFE. Final benchmark metrics belong to the external evaluator, not this method. Use the exact same fitting/normalization/body model/evaluator for every baseline comparison.

The SOKE adapter also accepts an explicit dictionary-conditioned `prompt` in its Python `prepare()` API. It never consults the author's name→keyword tables or any test metadata. The CLI currently implements the primary GF track; dictionary keyword extraction and evaluation must be configured as a separate resource-controlled track. Entropy/per-part selection and post-refiner are research controls, not claimed implemented production variants of the coupled method.

## Verification and limits

```bash
python -m pytest tests/race_sign -q
RACE_SMPLX_PATH=/licensed/smplx python -m pytest tests/race_sign -q
python -m ruff check methods/race_sign tests/race_sign
```

Tests cover independent SciPy rotation agreement, zero/pi and degenerate 6D gradients, analytic flow endpoints/NFE, padding/window correspondence, split/asset guards, native-length target-free prediction, both training stages, calibration and checkpoint continuation. Additional integration tests exercise actual author VQ/mBART code with small random fixtures; the licensed-asset check compares geometry numerically with the public `smplx.SMPLX` axis-angle implementation including non-principal hand angles. `author_integration.json`, when present, records the real pretrained-base integration scope; it does not assert signing quality of untrained RACE weights.

A repeatable real-author integration command is:

```bash
python -m methods.race_sign.integration_check \
  --assets data/pretrained/race_sign/madis/assets.json \
  --geometry /licensed/smplx --device cuda \
  --output outputs/race/author_integration.json
```

This command exercises the full default decoder and selector with random RACE weights and a real pretrained author generator. The integration report explicitly marks both RACE modules as untrained. There is no trained RACE model yet. Passing these tests establishes the covered implementation properties, not semantic correctness, intelligibility, benchmark reproduction, or SOTA. Generated-token DTW training can be wrong for lexical omissions; this is a scientific limitation, not something hidden by resampling at inference.

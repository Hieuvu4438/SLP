# SOKE: local setup and readiness

Audit date: 2026-11-03. Repository: https://github.com/2000ZRL/SOKE.
Upstream `main` / `origin/main`: `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`.
Both `git fetch --prune origin` and `git pull --ff-only` completed successfully.

## Location

The independent checkout is `/home/haipd/SLP/third_party/SOKE`.
`/home/haipd/DexAvatar/data/SOKE` is a compatibility symlink to that location.
The other clones at `/home/haipd/SOKE` and `/home/haipd/DexAvatar/SOKE` were
left untouched. The relocated checkout initially had no environment, `deps`,
checkpoints, datasets or experiment outputs.

The upstream checkout is excluded from SLP Git. Local integration files live in
`configs/soke/` and `scripts/`; upstream source files are unchanged.

## Runtime

Dedicated Python 3.10 environment: `third_party/SOKE/.venv`.
The SLP environment and other research environments are independent.

The runtime pins are in `configs/soke/requirements-runtime.txt`:
PyTorch 2.1.2 with CUDA 12.1, Lightning 2.2.5, Transformers 4.39.3,
TorchMetrics 1.3.2, NumPy 1.23.5, SciPy 1.10.1 and SMPL-X 0.1.28.

The upstream requirements are not directly usable as a complete environment:

- `bpy==3.4.0` is not published on PyPI. Blender rendering is optional and is
  separate from the core runtime. No Blender binary was found during the audit.
- NumPy 2.1.3 is incompatible with legacy NumPy aliases used by chumpy and
  legacy code; NumPy 1.23.5 retains these aliases.
- Runtime imports also need packages omitted from upstream requirements,
  including PyTorch, SciPy, pandas, sentencepiece, OpenCV and portalocker.
- MoviePy 1.x preserves the `moviepy.editor` API used by upstream.

Recreate the runtime from the SLP root:

```bash
uv venv --python /usr/bin/python3.10 third_party/SOKE/.venv
uv pip install --python third_party/SOKE/.venv/bin/python pip wheel setuptools==69.5.1 numpy==1.23.5 scipy==1.10.1
uv pip install --python third_party/SOKE/.venv/bin/python 'torch==2.1.2+cu121' --index-url https://download.pytorch.org/whl/cu121
uv pip install --python third_party/SOKE/.venv/bin/python -r configs/soke/requirements-runtime.txt --no-build-isolation
```

Use `bash scripts/soke.sh check` for imports and inventory, or
`bash scripts/soke.sh smoke --device cuda` for a pretrained tokenizer smoke test.
The report defaults to `outputs/soke/readiness.json`.

## Assets and sources

All download URLs below come from the upstream README or project homepage.

| Asset | Local path inside SOKE | Official source |
| --- | --- | --- |
| Decoupled tokenizer | `checkpoints/tokenizer.ckpt` | [Drive](https://drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk/view) |
| Pose mean | `deps/stats/mean.pt` | [Drive](https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view) |
| Pose std | `deps/stats/std.pt` | [Drive](https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view) |
| mBART backbone, tokenizer, map IDs | `deps/mbart-h2s-csl-phoenix/` | [Drive folder](https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi) |
| Evaluators and GloVe vocabulary | `deps/t2m/` | [Drive archive](https://drive.google.com/file/d/1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8/view) |
| Trained SOKE generator | `experiments/mgpt/SOKE/checkpoints/last.ckpt` | **Missing; no link found upstream.** |

SMPL-X files are reused through `deps/smpl_models/smplx`, linked to
`/home/haipd/DexAvatar/SMPLer-X/common/utils/human_model_files/smplx`.
That directory includes the neutral model, J14 regressor, MANO vertex mapping
and FLAME vertex mapping. `deps/smpl_models/smpl` links to the adjacent SMPL
directory. The segmentation JSON links to
`/home/haipd/DexAvatar/DPoser-X/lib/data/smplx_vert_segmentation.json`.
These links depend on the existing DexAvatar assets remaining available.
MANO and SMPL-H standalone models are not required by SOKE's active SMPL-X
code path; the full four-model archive in the README was not duplicated.

`flan-t5-base` is a legacy alternate model. The configured SOKE generator uses
mBART, and the tokenizer does not instantiate a language model. T5 weights are
not needed for these paths and were not downloaded.

## Datasets and local configurations

Official annotations for all three datasets are under `data/raw/soke/` in SLP:

- `How2Sign/{train,val,test}/re_aligned/how2sign_realigned_<split>_preprocessed_fps.csv`.
- `CSL-Daily/csl_clean.{train,val,test}`.
- `Phoenix_2014T/phoenix14t.{train,dev,test}`.

The local configurations point to these directories and the downloaded stats.
They use GPU 0, small batches, four data workers and offline WandB logging.
The wrapper sets `--use_gpus 0`; upstream otherwise defaults to GPUs 2–7
and a six-device configuration, which does not match this one-GPU machine.

How2Sign and CSL-Daily full pose trees have not been installed. Their raw videos
or poses from other pipelines do not automatically satisfy SOKE's file layout.
Multilingual training/evaluation requires the matching SMPL-X poses and verified
coverage for each selected split.

The official Phoenix pose archive is extracted to `Phoenix_2014T`; annotation
names match existing pose directories for all 7,092 train, 519 dev and 642 test
records. This checks directory coverage, not every individual frame. Phoenix-only
configurations are `configs/soke/deto.phoenix.yaml` and `soke.phoenix.yaml`.
Pass one after `--cfg` to the wrapper to override its default config.

```bash
# These dataset workflows require their selected datasets to be prepared.
bash scripts/soke.sh test-deto
bash scripts/soke.sh tokenize
bash scripts/soke.sh train-soke

# Requires the trained generator checkpoint first.
bash scripts/soke.sh test-soke --task t2m
```

The correct tokenization module is `scripts.get_motion_code`; the README's
`python -m get_motion_code` refers to a file that is not at the repository root.

## Remaining blocker

The mBART backbone is **not** the trained SOKE text-to-sign generator.
The tokenizer checkpoint is also **not** the generator checkpoint.
Full pretrained text-to-sign inference requires the trained generator weights
and their matching configuration. [Upstream issue #2](https://github.com/2000ZRL/SOKE/issues/2)
remains open with requests for `last.ckpt`. No replacement weights were invented.
Training a new generator is a separate research run and was not started.

The tokenizer smoke uses synthetic inputs, strict pretrained state-dict loading,
encode/decode and finite-output checks. It does not measure generation quality
or verify full training/evaluation. The GPU had an existing Uni-Sign workload;
that process was left running.

On 2026-11-03, `pip check` reported no broken requirements and the imports in
`scripts/check_soke.py` passed. Strict loading and CPU encode/decode passed for
the body, left-hand and right-hand tokenizer. A separate CPU test loaded the
first official Phoenix test pose sequence (175 frames, 133 features), normalized
its first 16 frames, and encoded/decoded the body component successfully.
GPU execution and full dataset evaluation were not tested.

# 02 — Codebase & data audit (SOKE · MaDiS · SignSparK)

Audit date: 2026-11-03. Everything below was read directly from the code in
`third_party/` (not only the papers). File:line references are clickable.

## 1. Shared representation (SOKE → MaDiS)

Both Imperial-College repos (SOKE ICCV'25, MaDiS arXiv 2601.19577) use the
**same 133-D SMPL-X axis-angle feature** and **the same released tokenizer**:

| slice | content | dims |
|---|---|---|
| 0:30 | upper-body joints 12-21 (spine3/neck/collars/head/shoulders/elbows/wrists) | 10×3 |
| 30:75 | left-hand 15 joints | 45 |
| 75:120 | right-hand 15 joints | 45 |
| 120:123 | jaw | 3 |
| 123:133 | expression (first 10 FLAME-PCA coeffs) | 10 |

* root + lower body (36 dims) and shape (10) are dropped; a fixed mean shape is
  used for FK (`third_party/MaDiS/mGPT/data/H2S.py:52`).
* Frames are subsampled to ≤24 fps for How2Sign; sequences filtered to
  40–400 frames.
* **Decoupled tokenizer** (`third_party/SOKE/configs/vq/*.yaml`): three 1-D conv
  VQ-VAEs, temporal down-sampling ×4 (`down_t=2`), EMA-reset codebooks:
  body+face (43-D input) **96 codes**, each hand (45-D) **192 codes**, code dim 512.
  Very small codebooks → a hard quantization floor (measured in
  [03_diagnostics.md](03_diagnostics.md)).

## 2. SOKE (Signs as Tokens, ICCV 2025)

* Generator: mBART-large-cc25 (`deps/mbart-h2s-csl-phoenix`, vocabulary pruned to
  en/zh/de) with sign tokens added to the vocabulary and **three LM heads**
  (body / lhand / rhand) predicting the three part tokens of a time step jointly
  (`mGPT/archs/lm_multihead.py`). Language tags `en_ASL`, `zh_CSL`, `de_DGS`.
* **Retrieval** (`mGPT/archs/mgpt_mbart.py:214` `get_kw_strings`): up to 3
  keywords per sentence from pre-extracted `scripts/name2kws_{train,val,test}.json`
  are looked up in `scripts/word2code.json` (4,057 dictionary words → body/lhand/rhand
  token lists, ≤10 steps) and appended to the encoder prompt as
  `Key word i, <motion_id_..><hand_id_..><rhand_id_..>`.
  * The keyword lists are substring matches against the dictionary, sorted by
    length (e.g. `['color','mean','rich','ich','aus',...]` for an English
    sentence: German sub-words like `ich/aus` match), so the retrieval is noisy and
    language-agnostic. Only the first 3 are used.
* Trained generator checkpoint is **not released** (upstream issue #2), so SOKE
  cannot be re-evaluated; only the tokenizer is public (`docs/soke_setup.md`).

## 3. MaDiS (masked diffusion LM, arXiv 2601.19577)

Code: `third_party/MaDiS/mGPT/archs/masked_diffusion.py`.

* Backbone: Qwen3-0.6B-Base (vocab pruned to en/zh/de), causal mask replaced by a
  **bidirectional** mask; text prompt is prefix, sign tokens are denoised.
* **Joint masking across parts**: the same mask is applied to body/lhand/rhand at a
  time step; the input embedding of a step is a **mixture-of-parts (MoP)** gate
  over the three codebook vectors (`_mop_embedding`).
* **UTC (unmasking with temporal checkpoints)**: training masks and inference
  unmasking follow a fixed coarse-to-fine grid (every 4th step → every 2nd → all),
  pivots 0.75/0.5, `DIFFUSION_STEPS = 25`.
* Inference (`generate`): greedy argmax, **confidence computed from the body head
  only**, no remasking, no CFG, fixed max length 100 tokens (=400 frames); length
  is implicit via EOS (PAD==EOS).
* **Tri-level pretraining**: token CE + latent regression (MLP from hidden states to
  codebook latent space) + 3D recon through the frozen VQ decoder
  (`mGPT/models/madis.py:112-170`).
* **No retrieval** in the released MaDiS (`forward` deletes `src`, `name`) — the
  SOKE dictionary mechanism was dropped.
* Released on HF `2000zrl/MaDiS`: tokenizer, pretrained (6.3 GB), SFT per dataset
  (CSL/Phoenix/How2Sign, 6.3 GB each), SiCLIP (3.8 GB), Qwen LM, pre-extracted tokens.

### 3.1 Metric implementation details that matter (MaDiS `mGPT/metrics/`)

| metric | what the code actually computes | consequence |
|---|---|---|
| DTW-JPE body/hands (`dtw.py`) | **accumulated** (not path-normalised) DTW cost of per-frame mean joint error, root (pelvis / wrist) translation aligned, averaged over sequences; values are metres×frames | scales with sequence length; long clips dominate; a static mean pose is a strong baseline (see 03) |
| DTW-PA-JPE (SOKE `t2m.py`, `align_idx=None`) | same but Procrustes aligned per frame | rewards shape only |
| **SiBLEU** (`generation.py:16-36`) | **unigram clipped overlap × brevity penalty over VQ token IDs** of the SOKE tokenizer, vs GT tokens (docstring says "SiBLEU-4" but only unigram counts are accumulated) | order-insensitive; only defined for methods that use *this* tokenizer; continuous methods must be re-tokenised (and pay the quantization error twice) |
| SiCLIP R@k (`clip.py`) | text↔motion retrieval with a CLIP model whose motion encoders are **initialised from the same SOKE VQ encoders**, ranked **within a test batch of 45** | evaluator shares representation with token generators; R@k depends on batch size |

## 4. SignSparK (ECCV 2026, arXiv 2603.10446)

Code: `third_party/SignSparK/`.

* Conditional (rectified) flow matching, 1-D temporal U-Net (~large, AdaLN-Zero,
  RoPE attention), **one model per stream** (hand / body / face), 6-D rotations +
  translation, M-CLIP XLM-R text encoder, seq_len 304.
* Losses: flow 0.5 + recon 1.0 + velocity 0.5 + acceleration 0.1
  (`configs/data_v2/common/train.yaml`).
* **Keyframe conditioning by inpainting**: observed keyframes (first/middle/last
  frame of each FAST sign segment) are given as GT values (`signspark/keyframes.py`,
  `pose_datasets_lmdb.py:293`). In the default `sample.py` they come from the
  **ground-truth test sequence** and the **GT length** is used
  (`length_mask_conditioned: True`). In the paper the SOKE comparison (SR-T2P) uses
  text→gloss→dictionary keyframes (3 per gloss, fixed 8-frame spacing); gloss-free
  T2P uses zero keyframes + CFG. Released checkpoints are trained on 15× more data
  (CSL-Daily + How2Sign + BOBSL + CSL-News) than the paper → **not comparable** with
  paper tables.
* Data: re-optimised SMPL-X LMDBs for CSL-Daily / How2Sign / Phoenix on HF
  (`LionelLow/SignSparK_data`, ≈14 GB) — a *different* fitting from SOKE's, so
  SOKE-protocol numbers cannot be computed on them directly.

## 5. Local assets (status 2026-11-03)

| asset | location | status |
|---|---|---|
| SOKE tokenizer, mean/std, mBART, t2m evaluators | `third_party/SOKE/{checkpoints,deps}` | present (sha256 in `configs/soke/assets-sha256.json`); MaDiS mean/std are byte-identical |
| SMPL-X neutral model | `third_party/SOKE/deps/smpl_models` → DexAvatar/SMPLer-X | present (symlinked into MaDiS) |
| Phoenix-2014T SOKE SMPL-X poses | `data/raw/soke/Phoenix_2014T/{train,dev,test}` (per frame) | present; packed for MaDiS by `scripts/madis_build_phoenix.py` → `data/processed/madis/Phoenix_2014T` (7096/519/642) |
| CSL-Daily SOKE fittings | Dropbox (2.9 GB) → `data/raw/soke/_downloads/csl_fit/` | present; streamed into `data/processed/madis/CSL-Daily/poses_video_level` by `scripts/madis_build_csl.py` (20,654 videos) |
| How2Sign SOKE fittings (Neural-Sign-Actors) | Dropbox (7.3 GB) → `data/raw/soke/_downloads/h2s_fit/` | present; packed by `scripts/madis_build_how2sign.py` → `data/processed/madis/How2Sign` (train/val/test 30,965 / 1,724 / 2,318 = CSV counts) |
| MaDiS tokenizer / sft_phoenix / SiCLIP / Qwen / tokens | `third_party/MaDiS/hf_assets/` (via hf-mirror.com, sha256 verified against the HF API) | present; MaDiS tokenizer VQ weights are bit-identical to SOKE's (max abs diff 0) |
| MaDiS venv | `third_party/MaDiS/.venv` (py3.10, torch 2.5.0+cu124, transformers 4.51.0) | ready |

## 6. Other datasets on this machine (useful for gaps)

| dataset | path | content | possible use |
|---|---|---|---|
| CSL-Daily videos + labels | `/home/shared_data/sign_language/CSLDaily` (28 GB), `/home/dongvk/datasets/CSL_Daily_Sentence_Crop` (42 GB, 20,654 2-D COCO-WholeBody-133 keypoint pkl) | sentence-level CSL | 2-D keypoints → BT evaluator, refitting |
| How2Sign videos | `/home/shared_data/sign_language/How2Sign` (86 GB) | ASL sentences | refitting / BT evaluator |
| Phoenix-2014T full release | `/home/dongvk/datasets/phoenix14T` (48 GB) | incl. **frame-level gloss alignments** (`alignments.pami0`) inside the SOKE annotation pickles | alignment supervision / segment dictionary |
| ASL Citizen | `/home/dongvk/datasets/ASL_Citizen` (65 GB) | ~2.7k-gloss isolated ASL, 2-D keypoints | **ASL dictionary** for How2Sign retrieval |
| WLASL2000, MS-ASL | `/home/dongvk/datasets/{WLASL2000,MS-ASL}` | isolated ASL (2k / 1k glosses) | dictionary / lexical evaluator |
| **OurVSL_small** | `/home/dongvk/datasets/OurVSL_small` (37 GB) | **Vietnamese Sign Language**, 4 sources (CED 1.4k, HAD 3.0k, PARD 2.5k, QIPEDC 8.2k ≈ 15.1k clips) with Vietnamese text + signer id, 2-D 133-kp | new low-resource SLP benchmark (no 3-D VSL production benchmark exists) |
| YouTube-ASL | ids only | – | – |

All 2-D keypoints are COCO-WholeBody-133 (`[T,1,133,2]` + scores), i.e. the
RTMPose format used by Uni-Sign. SMPLer-X / HaMeR / WiLoR / DPoser-X exist under
`/home/haipd/{DexAvatar,hamer,DPoser-X}`, so 3-D refitting of new data is possible.

## 7. Compute

One RTX 5880 Ada (48 GB), shared: another user's Uni-Sign pre-training job
holds ~27 GB at 100% utilisation. Disk: 187 GB free on `/home` (95% used).
MaDiS was trained on 4 GPUs at batch 64, so the proposal must be
compute-aware (see [06_experiment_plan.md](06_experiment_plan.md)).

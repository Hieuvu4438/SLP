# 03 — Diagnostics: what limits the current SOTA

All numbers in this file are on **dev splits** (Phoenix-2014T dev; CSL-Daily and How2Sign val in §8–9), using the SOKE SMPL-X fittings.
We use MaDiS's own `pairwise_jpe` and DTW definitions, which are pelvis/wrist-
translation aligned and accumulated. The test split was never used. Published numbers
quoted for comparison are **test** numbers from the papers, so dev↔test differences
of a few percent are expected. Raw outputs are in `outputs/diagnostics/` and
`outputs/d2c_pilot/`.

## 1. Setup

* Script: `scripts/madis_oracle_eval.py` (MaDiS venv). Data:
  `data/processed/madis/Phoenix_2014T` (packed by `scripts/madis_build_phoenix.py`).
* Sequences are filtered to 40–400 frames, the protocol's MIN/MAX_MOTION_LEN, and
  cropped to a multiple of 4. This leaves 491 of 519 dev clips and 6,760 train clips.
  The released MaDiS loader resamples clips shorter than 40 frames instead of
  dropping them; that affects 28 dev clips.
* Metric: DTW-JPE for body (12 upper-body / face joints, pelvis-aligned) and hands
  (21 joints per hand from the MANO-style regressor, wrist-aligned, averaged over
  hands). It is reported as the **accumulated** cost (paper unit) and as the
  **per-step** cost (accumulated / warping-path length, in cm).

## 2. Calibration rows (n = 491)

| Row | What it is | DTW-JPE body ↓ | DTW-JPE hands ↓ | per-step body (cm) | per-step hands (cm) |
|---|---|---|---|---|---|
| **tokenizer oracle** | GT → SOKE/MaDiS tokenizer → decode, i.e. a perfect LM | **3.19** | **4.93** | 3.00 | 4.24 |
| mean pose | training-mean pose, repeated for the GT length | 5.73 | 8.22 | 5.33 | 7.63 |
| text-NN | the train clip with the most similar sentence (char TF-IDF) | 6.57 | 8.39 | 5.07 | 5.93 |
| GT time-reversed | real motion, wrong order | 5.31 | 7.33 | 4.27 | 5.42 |
| *published, test:* SOKE | – | 6.04 | 7.72 | – | – |
| *published, test:* MaDiS | – | 4.97 | 6.59 | – | – |
| *published, test:* M3T | – | 4.02 | 8.55 | – | – |
| *published, test:* SignSparK noKF / SR-T2P | – | 4.40 / 3.51 | 7.10 / 4.82 | – | – |

Tokenizer codebook usage on dev: body 96/96, left hand 189/192, right hand 190/192.
The codebooks are fully used, so the floor comes from **capacity**, not collapse.

### Findings

**F1 — The accumulated DTW-JPE barely separates real methods from trivial outputs.**
- Time-reversed GT is semantically wrong, yet it scores 5.31 / 7.33, better than
  SOKE's published 6.04 / 7.72.
- A static mean pose scores 5.73 / 8.22, better than M3T's published hands (8.55).
- Any SOTA claim on this metric must therefore be accompanied by the calibration
  rows. Small DTW deltas (< ≈0.3) between methods are within the
  "trivial-output band".

**F2 — The accumulated metric conflates timing/length with articulation.**
- Text-NN is worse than mean pose on accumulated hands (8.39 vs 8.22) but much
  better per step (5.93 vs 7.63 cm).
- The retrieved real motion has the wrong length, so its warping path is longer. The
  accumulated cost punishes the length mismatch more than the articulation.
- Normalised DTW (now `dtw_jpe_norm` in `src/slp/evaluation/metrics.py`) should be
  reported alongside it.

**F3 — Quantization floor.** A *perfect* token predictor with the shared tokenizer
still incurs 3.19 / 4.93.
- MaDiS (test, 4.97 / 6.59) is therefore ≈1.8 / 1.7 above the floor.
- SignSparK-SR's hands (4.82, test) are **at or below** the floor. No method that
  decodes through this tokenizer can match SignSparK's hands, however good the LM.
- This motivates C1, the continuous refiner (see [05](05_proposed_method.md)).

**F4 — Sources of the floor.** The tokenizer smooths high-frequency jitter of the
monocular fittings. Part of the "floor" is therefore the tokenizer correctly
*denoising* noisy pseudo-GT. This cuts both ways for C1: a continuous refiner trained
on noisy GT may learn to reproduce jitter. We will report a jerk / velocity spectrum
and the DTW split by frequency band.

## 3. Tokenizer pilot (right hand, Phoenix train only)

`methods/d2c_sign/tokenizer_pilot.py`: same conv architecture family as SOKE (×4
temporal down-sampling, 512-d codes, EMA + dead-code reset), trained 20k iterations
with batch 256 × 64 frames. Scored with only the right hand replaced in GT, so the
error is finger articulation only.

| Right-hand tokenizer (others = GT) | bits / token | codes used | DTW-JPE right hand ↓ | per-step (cm) |
|---|---|---|---|---|
| SOKE/MaDiS released (192 codes, trained on 3 datasets) | 7.58 | 190/192 | 1.004 | 0.95 |
| pilot VQ-192 (Phoenix train only) | 7.58 | 192/192 | 0.917 | 0.86 |
| pilot RVQ-2 / RVQ-4 | 15.2 / 30.3 | – | stopped (see below) | – |

**F5. Finger articulation is *not* where the floor comes from.**
- With only the right-hand fingers tokenized, the hand error is ≈1.0 (0.95 cm per
  step).
- With the full tokenizer reconstruction, the hand error is 4.93 (4.24 cm per step).
- The hand metric is wrist-aligned, so it is sensitive to the **global hand
  orientation**, which comes from the arm/wrist rotations in the **96-code body
  stream**. The body/face tokenizer is therefore the main contributor to the
  "hands" floor.
- The decomposition (job `floor_decomp`, `outputs/diagnostics/phoenix_dev_floor_decomposition.json`,
  n = 491) confirms it:

  | Tokenized stream (rest = GT) | DTW body | DTW hands | per-step hands (cm) |
  |---|---|---|---|
  | body + face only (96 codes) | 3.19 | **4.78** | 4.15 |
  | both hands only (192 codes each) | 0.00 | 0.89 | 0.84 |
  | everything (= tokenizer oracle) | 3.19 | 4.93 | 4.24 |

  **About 97% of the "hands" floor comes from the body tokenizer.**
- Consequence for the design: a better *hand* codebook (RVQ / FSQ for fingers,
  which is what M3T and SeRV change) buys little. What matters is
  wrist/forearm orientation, which C1 corrects in continuous space. The RVQ hand
  pilots were stopped for this reason, and to respect the 2–3-process budget.

## 4. Released MaDiS reproduction on dev

Released `sft_phoenix.ckpt` (sha256 verified), config `configs/madis/madis_phoenix.dev.yaml`,
1 GPU, batch 4, released code unchanged, all 519 dev clips (MaDiS loader).
Log: `outputs/madis_phoenix_dev.log`. Predictions:
`third_party/MaDiS/results/madis/madis_phoenix_dev/val_rank_0/`.

| | DTW-JPE body ↓ | DTW-JPE hands ↓ | SiBLEU body ↑ | SiBLEU hands ↑ | s / batch |
|---|---|---|---|---|---|
| MaDiS released, **dev (ours)** | **5.19** | **6.94** | **4.97** | **3.71** | 0.52 |
| MaDiS paper, test | 4.97 | 6.59 | 4.54 | 3.33 | – |
| tokenizer oracle, dev (§2, 491 clips) | 3.19 | 4.93 | 100 by definition | 100 | – |
| GT reversed, dev (§2, 491 clips) | 5.31 | 7.33 | – | – | – |

The reproduction is consistent with the paper: dev and test differ by ~4–5% in DTW
and SiBLEU, in opposite directions.

**F6.** On body, MaDiS (5.19) is only 0.12 better than time-reversed GT (5.31) and
0.54 better than a static mean pose (5.73). On hands it is 1.3 better than mean
pose but still **2.0 above the tokenizer floor**. Paired numbers on identical clips
and crops (`scripts/score_predictions.py`, built-in `@mean_pose` / `@gt_reversed`)
were then computed by job `score_madis_phoenix_dev`
(`outputs/diagnostics/phoenix_dev_madis_paired.json`). The scorer reproduces MaDiS's own
metric exactly (5.186 / 6.942), which validates it.

| All 519 dev clips, identical crops | DTW body ↓ | DTW hands ↓ | per-step body (cm) | per-step hands (cm) | right-hand speed / GT |
|---|---|---|---|---|---|
| MaDiS released | 5.19 | 6.94 | 4.55 | 5.56 | **0.35** |
| mean pose | 5.53 | 7.94 | 5.34 | 7.63 | 0.00 |
| **GT time-reversed** | **5.11** | 7.07 | **4.23** | **5.44** | 1.00 |

**F7. Under the current protocol, SOTA is barely distinguishable from a wrong-order
copy of the reference.**
- Time-reversed GT beats MaDiS on body DTW and on both per-step metrics.
- MaDiS's hands move at **35% of GT speed** (median ratio of mean frame-to-frame
  right-hand joint displacement), the classic regression-to-the-mean signature.
- So the DTW-JPE leaderboard rewards conservative, under-articulated motion.

Two consequences:
1. C0: report speed / articulation statistics and trivial rows in every table.
2. C1 / C3 must raise articulation without being punished by DTW. That is why the
   proposal relies on back-translation and SiCLIP for semantic claims, and on DTW
   only as a fidelity sanity check.

## 5. Metric-implementation audit (code facts)

| Item | Where | Fact |
|---|---|---|
| DTW accumulation | SOKE `mGPT/metrics/dtw.py:14`, MaDiS `mGPT/metrics/dtw.py:8` | Returns `D[-1,-1]` (sum along the path), not normalised |
| SiBLEU | MaDiS `mGPT/metrics/generation.py:16-36` | Clipped **unigram** counts + brevity penalty; docstring says SiBLEU-4 |
| SiCLIP | MaDiS `mGPT/metrics/clip.py`, `configs/evaluator/clip_test_*.yaml` | Recall within a batch of 45; motion encoder initialised from the SOKE VQ encoders |
| SOKE retrieval keywords | SOKE `scripts/name2kws_*.json` | Substring matches across languages, e.g. German `ich` / `aus` for English sentences |
| SignSparK default sampling | `SignSparK/sample.py`, `pose_datasets_lmdb.py:293` | GT keyframes (first / middle / last of each GT sign segment) and GT length |

## 6. Data-provenance check: released Phoenix-dev tokens

`TOKENS_h2s_csl_phoenix` (MaDiS HF) was compared with a fresh encoding of the local
SOKE fittings by the identical tokenizer and crop rule:

| Split | Agreement (body / lh / rh) |
|---|---|
| CSL-Daily | 100% / 100% / 100% |
| Phoenix train | 100% / 100% / 100% |
| **Phoenix dev** | **0–7%** (chance level; no other dev clip or ±3 shift matches either) |

The token lengths are identical, so the released Phoenix-dev tokens must come from
**different dev fittings** than the SOKE-released dev poses.

**Extended check (all three datasets, 30–60 clips each, body stream):**

| Released tokens vs … | Phoenix | CSL-Daily | How2Sign |
|---|---|---|---|
| fresh encoding of local GT, **train** split | 100% | 100% | 100% |
| fresh encoding of local GT, **val/dev** split | 0–7% | 1.4% | 0.6% |
| re-encoded MaDiS-generated dev output | 0.6% | – | – |
| re-encoded `feats_ref` (the GT MaDiS's own loader uses for DTW) | 0.7% | – | – |

Token *lengths* always match. In the released assets, the **val/dev reference tokens
are not encodings of the val/dev poses that DTW is computed against**, nor of the
generated output. Train tokens are consistent. We did not check the test split (test
data untouched), so we cannot say whether the paper's SiBLEU on test has the same
issue. This should be raised with the MaDiS authors as a GitHub issue, with the
script above as the reproduction.

**F8 — consequences**
1. MaDiS's own dev / val **SiBLEU** (Phoenix 4.97 / 3.71, CSL 5.62 / 5.24,
   How2Sign 4.44 / 3.67) is computed against these mismatched references, so it is
   **not meaningful on any dev split**. This also corrects §8, which assumed the CSL
   val tokens were valid; only CSL *train* was checked there. We re-encode the
   references locally (`scripts/decode_sign_tokens.py --reencode`).
2. Any token-level metric must document which fittings produced its reference
   tokens. Test tokens were deliberately **not** checked (test split untouched).
3. For our experiments, every reference token is re-encoded from the poses we
   evaluate against.

## 7. C1 refiner v0 — first result (negative, informative)

`methods/d2c_sign/refiner.py`:
- residual rectified flow, transformer d = 384 × 6, no text;
- trained 30k iterations on Phoenix train with GT-token decodes plus 3 corrupted
  variants (25% clean);
- 8 Euler steps, temperature 1.

| Dev, 519 clips | DTW body | DTW hands | per-step hands (cm) | rh speed / GT |
|---|---|---|---|---|
| MaDiS released | 5.19 | 6.94 | 5.56 | 0.35 |
| MaDiS + flow refiner v0 | 5.41 | 7.08 | 5.51 | **0.87** |

- The refiner restores articulation (hand speed 0.35 → 0.87 of GT) and slightly
  improves per-step hand error.
- But accumulated DTW gets worse. A sampled residual adds motion that does not match
  the reference, and DTW-JPE rewards the conditional mean (cf. F7).
- Re-evaluation on locally re-encoded GT tokens (to separate the refiner from F8)
  and the regression variant are queued (`refiner_flow_v0_gttok_fixed`,
  `refiner_regress_v0`).

**Corrected GT-token evaluation.** Job `refiner_flow_v0_gttok_fixed` uses tokens
re-encoded from the local dev poses (F8), n = 519:

| Input tokens = perfect (GT) | DTW body | DTW hands | per-step hands (cm) | rh speed / GT |
|---|---|---|---|---|
| tokenizer decode (floor) | 3.12 | 4.80 | 4.31 | 0.35 |
| + flow v0, τ = 1 | 3.32 | 4.49 | 3.83 | 0.83 |
| **+ flow v0, τ = 0 (deterministic ODE)** | **2.75** | **3.96** | **3.57** | 0.35 |

**F9. A continuous refiner breaks the quantization floor.**
- At τ = 0 the refiner removes 12% (body) and 17% (hands) of the floor that every
  SOKE/MaDiS-tokenizer method is bound by.
- The deterministic path trades back to low hand speed; τ = 1 restores speed and is
  still below the floor on hands.
- **The tokenizer itself moves the hands at 35% of GT speed, exactly MaDiS's
  ratio.** MaDiS's under-articulation (F7) is inherited from the tokenizer, not
  produced by the LM.
- Whether these gains survive LM-generated tokens is queued as job
  `refiner_flow_v0_madis_t0` (τ ∈ {0, 0.5}).

**Refiner on real MaDiS outputs** (job `refiner_flow_v0_madis_t0`, n = 519):

| | DTW body | DTW hands | per-step hands (cm) | rh speed / GT |
|---|---|---|---|---|
| MaDiS released | 5.186 | 6.942 | 5.56 | 0.35 |
| + flow v0, τ = 0 | 5.159 | 6.934 | 5.53 | 0.35 |
| + flow v0, τ = 0.5 | 5.209 | 6.939 | 5.52 | 0.45 |

**F10. On LM outputs, the error is dominated by planning, not quantization.**
- Error budget on Phoenix dev (body / hands):
  - quantization floor: 3.12 / 4.80;
  - MaDiS: 5.19 / 6.94;
  - so the planning gap is **≈2.1 / 2.1**, while the removable quantization part is
    ≈0.4 / 0.8.
- A refiner conditioned only on the coarse motion can fix quantization but not
  wrong signs or timing, so the gain on MaDiS outputs is ≈0.5%.
- Consequences:
  1. The biggest lever toward SOTA is the **planning / semantic stage** (C2
     retrieval, C3 reward alignment), not C1.
  2. C1 must be **trained on self-generated tokens with DTW-warped targets** (input
     mix (c) in [05](05_proposed_method.md)) and **text-conditioned**, so that it
     learns to correct LM errors, not just quantization. Until then, C1 is a
     fidelity module whose benefit only appears once planning improves.

**Implications for C1**
- Distance metrics and articulation pull in opposite directions. C1 needs a design
  choice about which one it optimises.
- Planned variants:
  - (i) predict the conditional mean for low-frequency wrist/arm orientation (where
    97% of the floor lives, F5), and sample only high-frequency articulation;
  - (ii) temperature or guidance on the residual;
  - (iii) train on low-pass-filtered targets so the refiner does not learn
    pseudo-GT jitter (F4).
- Success is judged on **BT / SiCLIP / human** with DTW as a non-inferiority
  check, not on DTW alone.

## 8. CSL-Daily dev: MaDiS reproduction and calibration rows

Config `configs/madis/madis_csl.dev.yaml`, released `sft_csl.ckpt`, 1,077 dev clips.
The released CSL *train* tokens match the local fittings, but **val tokens do not** (§6), so the SiBLEU column below is not meaningful.

| | DTW body ↓ | DTW hands ↓ | SiBLEU body ↑ | SiBLEU hands ↑ | per-step body / hands (cm) | rh speed / GT |
|---|---|---|---|---|---|---|
| MaDiS released, dev (ours) | **5.78** | **7.65** | 5.62 | 5.24 | 4.37 / **5.43** | 0.54 |
| MaDiS paper, test | 5.99 | 7.96 | 4.87 | 4.08 | – | – |
| mean pose | 7.01 | 9.78 | – | – | 5.82 / 8.12 | 0.00 |
| GT time-reversed | 6.10 | 8.35 | – | – | **4.31** / 5.55 | 1.00 |

**F11.** On CSL-Daily the reproduction holds: DTW is 3–4% better than the paper's
test numbers, consistent with the dev/test shift.
- MaDiS beats the trivial rows on accumulated DTW by a clearer margin than on
  Phoenix: −0.32 body and −0.70 hands versus reversed GT.
- Per-step body error is still tied with reversed GT (4.37 vs 4.31 cm).
- Hands remain under-articulated (54% of GT speed).
- The Phoenix finding (F7) is therefore strongest on the smallest, most repetitive
  dataset, but the conclusions carry over: the per-step and speed statistics must
  be reported.

## 9. How2Sign dev: MaDiS reproduction and calibration rows

Config `configs/madis/madis_how2sign.dev.yaml`, released `sft_how2sign.ckpt`.
Metrics are over 1,724 val clips (released loader); paired scoring covers 1,717 clips
(`outputs/diagnostics/how2sign_dev_madis_paired.json`). The released val reference
tokens do not match the GT (§6), so SiBLEU is not meaningful.

| | DTW body ↓ | DTW hands ↓ | per-step body / hands (cm) | rh speed / GT |
|---|---|---|---|---|
| MaDiS released, dev (ours) | 6.61 | 9.61 | 4.70 / 6.62 | 0.43 |
| MaDiS paper, test | 6.59 | 9.45 | – | – |
| **static mean pose** | **5.99** | 10.09 | 4.25 / 7.09 | 0.00 |
| **GT time-reversed** | **5.29** | **8.94** | **3.10 / 4.90** | 1.00 |

**F13.** On How2Sign, the largest and only open-domain benchmark, the released
MaDiS reproduces the paper (6.61 / 9.61 vs 6.59 / 9.45). Yet it **loses to a static
mean pose on body DTW** and to time-reversed GT on every DTW variant.

Across the three benchmarks, the DTW-JPE leaderboard therefore does not show that
current SOTA models produce text-specific motion. On the dataset where generation
matters most (open-domain ASL), they are below trivial outputs.

This is the strongest argument for C0. A credible SOTA claim needs:
- semantic metrics: an independent back-translator with a GT ceiling, and full-split
  SiCLIP;
- calibrated kinematic metrics: per-step DTW, speed ratio, trivial-baseline rows.

It also reframes the target. Beating MaDiS on accumulated DTW is easy, since mean
pose already does on How2Sign body. Beating it on semantics while staying above
the trivial floors is the real bar.

## 10. Back-translator B1 (reward model) on Phoenix

`scripts/train_backtranslator.py --arch mbart`:
- pose encoder: part-aware linear → conv ×2 down → 4-layer transformer;
- decoder: SOKE's pruned mBART-large decoder (the text encoder is replaced);
- trained on GT train poses only.

The first run (encoder LR 3e-4) **diverged after epoch 5**: loss 1.64 → 5.42, with
collapsed outputs. It is kept in `outputs/bt/phoenix_mbart_diverged_lr3e-4`. The v2
run uses encoder LR 1e-4 and LM LR 1e-5 for 30 epochs and is stable.

| Epoch | train loss | dev BLEU-4 (GT poses) | chrF | BT log-likelihood / token |
|---|---|---|---|---|
| 12 | 1.25 | 10.88 | 33.1 | **−1.53** |
| **21 (selected by BLEU)** | 0.62 | **11.49** | 34.8 | −1.93 |
| 30 | 0.41 | 11.07 | 34.9 | −2.10 |

The GT ceiling of 11.5 BLEU-4 on dev is in the range of published GT ceilings for
Phoenix (M3T 11.84, Sign-IDD 11.93, SignRR 13.43, SignSparK 14.77, each with its own
evaluator). Log-likelihood is best at epoch 12, before BLEU peaks: the model becomes
over-confident. For C3, the reward model should therefore be the likelihood-selected
checkpoint, and the evaluator (B2, a different architecture) is selected by BLEU.
Scoring of MaDiS / refiner / trivial rows with B1 is queued (`bt_score_phoenix_dev`).

### 10b. B1 back-translation of generated and calibration outputs (Phoenix dev, n = 519)

Job `bt_score_phoenix_dev`; results in `outputs/diagnostics/phoenix_dev_bt_b1_scores.json`.

| Input to B1 | BLEU-4 ↑ | chrF ↑ | log-lik / token ↑ | % of GT BLEU |
|---|---|---|---|---|
| GT (ceiling) | 11.49 | 34.83 | −1.931 | 100 |
| static mean pose | 0.36 | 15.60 | −4.452 | 3 |
| GT time-reversed | 9.17 | 32.63 | −1.984 | 80 |
| MaDiS released | 9.76 | 32.24 | −2.015 | 85 |
| **MaDiS + C1 flow refiner v0, τ = 0** | **11.25** | **34.40** | −2.005 | **98** |
| MaDiS + C1 flow refiner v0, τ = 1 | 9.94 | 33.23 | −2.006 | 86 |

**F14.** Back-translation separates the trivial rows from real ones much better
than DTW: mean pose collapses to 0.4 BLEU, whereas on DTW it beat MaDiS on How2Sign
body. Two caveats and one positive signal:

1. **Weak order sensitivity.** Reversed GT keeps 80% of the GT BLEU. On the narrow
   Phoenix weather domain, a pose→text model can partly recover sentences from the
   *set* of signs. The C3 reward must be paired with an order-sensitive term, or
   with a reversed-GT negative used as a contrastive penalty.
2. **Domain artefacts.** B1 is trained on GT poses, so part of the refiner gain may
   come from removing tokenizer smoothing (F9: tokenizer hands move at 35% of GT
   speed), not from better semantics. This is tested in job
   `bt_score_phoenix_gttok` (GT-token decode ± refiner) and later with B2.
3. **Positive signal for C1.** The deterministic refiner lifts MaDiS from 85% to 98%
   of the GT back-translation ceiling (+1.49 BLEU-4, +2.2 chrF) while leaving DTW
   unchanged (F10). If (2) is ruled out, this is the first evidence for the claim
   that FLARE improves semantics *beyond* what DTW sees.

## 11. C1 refiner v1 (self-generated inputs, DTW-warped targets)

`refiner_v1_selfgen`:
- MaDiS outputs were generated for all 7,092 train sentences
  (`scripts/madis_generate.py`, 55 min);
- each GT was DTW-warped onto the generated timeline
  (`scripts/build_selfgen_targets.py`);
- the refiner was trained on a 50/50 mix of these and the v0 data, 30k iterations,
  then evaluated at τ = 0 on MaDiS dev outputs (n = 519).

| | DTW body | DTW hands | per-step hands (cm) | rh speed / GT |
|---|---|---|---|---|
| MaDiS released | 5.186 | 6.942 | 5.56 | 0.35 |
| + refiner v0, τ = 0 | 5.159 | 6.934 | 5.53 | 0.35 |
| + refiner v1 (self-gen), τ = 0 | 5.145 | 6.964 | 5.56 | 0.33 |

**F15.** Training the refiner on the LM's own outputs does not unlock
planning-error correction under DTW (−0.04 body, +0.02 hands). This is consistent
with F10: a motion-to-motion stage without access to *what should have been
signed* cannot repair wrong or missing signs.

Caveat: MaDiS was fine-tuned on these train sentences, so its train-set outputs
are closer to GT than its dev outputs, and the refiner sees an easier error
distribution than at test time.

Conclusions:
- C1 is a fidelity / articulation module. Its value shows up in back-translation
  (F14: v0 lifts BT from 85% to 98% of the ceiling), not in DTW. v1's BT score is
  queued (`bt_score_phoenix_v1`).
- Planning must be improved upstream, by C2 (retrieval) and C3 (reward alignment).

## 12. Is the BT gain real? GT-token control (job `bt_score_phoenix_gttok`)

| B1 back-translation, Phoenix dev (n = 519) | BLEU-4 | chrF | log-lik / token |
|---|---|---|---|
| GT poses (ceiling) | 11.49 | 34.83 | −1.931 |
| **GT tokens → tokenizer decode (perfect LM)** | **7.82** | 29.34 | −2.105 |
| GT tokens + refiner v0, τ = 0 | 9.62 | 32.01 | −2.068 |
| GT tokens + refiner v0, τ = 1 | 8.65 | 31.50 | −2.086 |
| MaDiS released | 9.76 | 32.24 | −2.015 |
| MaDiS + refiner v0, τ = 0 | 11.25 | 34.40 | −2.005 |

**F16 — two lessons that change the evaluation and the C3 design.**

1. **The tokenizer costs about one third of the BT score** (11.49 → 7.82) even with
   perfect tokens. The refiner recovers part of it (+1.8). So part of the C1 BT gain
   in F14 is removal of tokenizer smoothing / domain shift relative to B1's training
   poses. That is still a genuine fidelity gain, but it is not a semantic one.
2. **Generated motion back-translates *better* than a perfect-token decode** (MaDiS
   9.76 vs 7.82). Per-sentence content cannot explain this, because the GT tokens
   carry the exact content. The likely cause: B1 and MaDiS are both fit to the
   Phoenix train distribution, and MaDiS outputs are *typical* signing that B1 maps
   to generic, high-frequency weather sentences, which corpus BLEU rewards.
   **Corpus BLEU from a single in-domain back-translator therefore measures
   typicality as much as fidelity.**

Consequences:
- **C0:** add a *sentence-discriminative* BT metric. For each clip, rank the
  likelihood of its own sentence against K = 99 distractor dev sentences under B1
  and B2 and report R@1/R@5 ("BT-retrieval"). A generic output cannot score well on
  this. Report BLEU and BT-retrieval together.
- **C3:** the reward becomes the **contrastive BT likelihood**
  ℓ(y | x̂) − logsumexp_k ℓ(y_k | x̂), with distractors y_k from the batch, plus the
  order term (F14). Plain likelihood or BLEU rewards would push the policy toward
  generic, typical output; F16 shows this optimum is already partly reached by MaDiS.
- The refiner and MaDiS rows have to be re-scored with BT-retrieval before any C1
  claim is made.

## 13. Sentence-discriminative BT-retrieval (B1, k = 99 distractors, Phoenix dev)

Each clip ranks its own sentence against 99 fixed random other dev sentences by B1
likelihood (`--retrieval`, `outputs/diagnostics/phoenix_dev_bt_b1_retrieval.json`).
Chance R@1 = 1%.

| | BLEU-4 | R@1 ↑ | R@5 ↑ | R@10 ↑ | mean rank ↓ |
|---|---|---|---|---|---|
| GT | 11.49 | 20.6 | 36.8 | 47.0 | 24.3 |
| mean pose | 0.36 | 1.7 | 8.1 | 15.0 | 45.5 |
| GT reversed | 9.17 | 14.1 | 27.6 | 40.1 | 27.2 |
| GT tokens → tokenizer decode | 7.82 | 12.5 | 28.7 | 40.8 | 28.2 |
| GT tokens + refiner v0 τ = 0 | 9.62 | 15.2 | 33.3 | 43.0 | 25.6 |
| MaDiS released | 9.76 | 20.0 | 35.3 | 44.9 | 26.1 |
| **MaDiS + refiner v0 τ = 0** | **11.25** | **21.2** | **38.2** | **49.3** | 24.8 |
| MaDiS + refiner v1 τ = 0 | 10.48 | 21.0 | 38.0 | 48.4 | 24.7 |

**F17 — evaluator–generator shared-training-set bias.**

- Retrieval does separate the trivial rows: mean pose is near chance, and reversed
  GT drops to 14.1 R@1 against 20.6 for GT.
- But MaDiS matches GT, and MaDiS + refiner **exceeds GT** on every retrieval
  metric. The decode of *perfect* GT tokens is far below MaDiS.
- An evaluator that prefers generated motion over the real dev signing is not
  measuring fidelity to the reference. It is measuring agreement with the
  **train-set text→motion mapping** that both MaDiS and B1 learned from the same
  7k Phoenix sentences. Phoenix is highly repetitive (weather domain, 9 signers),
  so "canonical train-style signing of this sentence" scores better than the dev
  signer's own realisation.

Consequences for the protocol (C0) and for C3:

1. **Within-dataset back-translators are biased judges** for models trained on the
   same split. This applies to every published SLP BT number we found, since each
   paper trains its evaluator on the benchmark's train split.
2. **Report super-GT scores as evaluator saturation, not as success.** A row that
   beats GT on BT must be flagged.
3. **Independent evaluators** must come from data the generator never saw:
   - (a) back-translators trained on *disjoint signers*: Phoenix and CSL provide
     signer IDs, so a signer-held-out evaluator is possible;
   - (b) cross-dataset / external SLT models (e.g. Uni-Sign pose models
     pre-trained on CSL-News for CSL);
   - (c) human comprehension.
4. **C3 reward hacking risk is concrete.** A reward from a same-split back-translator
   pushes the policy toward the canonical mapping that the evaluator already
   over-rewards. The reward model and the evaluator therefore need **different
   training data**, not just different architectures. B1/B2 (same split, different
   architecture) are insufficient as a reward/evaluator pair. The protocol is updated
   in [05](05_proposed_method.md).

## 14. Independent-architecture evaluator B2 (Qwen3-0.6B + LoRA, same train split)

B2's best dev BLEU-4 on GT is 10.62 (epoch 15, selected by BLEU), against 11.49 for B1.
Results: `outputs/diagnostics/phoenix_dev_bt_b2_retrieval.json`.

| Phoenix dev | B2 BLEU-4 | B2 R@1 | B2 R@10 | B1 BLEU-4 | B1 R@1 |
|---|---|---|---|---|---|
| GT | 10.62 | 17.5 | 49.7 | 11.49 | 20.6 |
| mean pose | 0.13 | 1.7 | 11.2 | 0.36 | 1.7 |
| GT reversed | 8.59 | 12.9 | 41.2 | 9.17 | 14.1 |
| GT tokens → decode | 6.87 | 12.3 | 43.0 | 7.82 | 12.5 |
| GT tokens + refiner v0 τ = 0 | 9.49 | 16.6 | 45.7 | 9.62 | 15.2 |
| MaDiS | 8.48 | 18.3 | 47.8 | 9.76 | 20.0 |
| MaDiS + refiner v0 τ = 0 | **10.00** | 18.5 | 48.9 | 11.25 | 21.2 |
| MaDiS + refiner v1 τ = 0 | 9.96 | 17.9 | 48.7 | 10.48 | 21.0 |

**F18 — what holds under a second, architecturally different evaluator.**

1. **The C1 BLEU gain replicates:** +1.52 with B2, +1.49 with B1. The refiner effect is
   not an artefact of one back-translator's architecture.
2. **The refiner raises sentence-specific recognisability only when the content is
   right.** With perfect tokens, R@1 goes 12.3 → 16.6 (B2) and 12.5 → 15.2 (B1).
   With MaDiS tokens it moves only 18.3 → 18.5. C1 is a fidelity amplifier; planning
   determines the ceiling (consistent with F10 and F15).
3. **The super-GT effect replicates** (MaDiS R@1 18.3 ≥ GT 17.5). It comes from the
   shared train split and/or noisy pseudo-GT (the smoothed-GT control is queued),
   not from one evaluator's quirks. Disjoint-data evaluators remain necessary (F17).

**Status of the C1 claim:** "C1 improves fidelity and back-translation of LM outputs
(+1.5 BLEU-4 under two evaluators, DTW unchanged), and lowers the tokenizer floor
(−12% / −17% DTW with GT tokens)." This is supported on Phoenix dev and still needs
CSL / How2Sign, seeds, and disjoint-data or human evaluation.

## 15. Control: is the super-GT effect caused by noisy pseudo-GT? (B1)

The dev GT (MaDiS crop) was smoothed temporally with a Gaussian of σ = 1 and 2 frames
and back-translated (`outputs/diagnostics/phoenix_dev_bt_b1_gt_smooth.json`).

| | BLEU-4 | chrF | R@1 | R@10 |
|---|---|---|---|---|
| GT (crop) | 11.23 | 34.4 | 20.8 | 46.4 |
| GT, σ = 1 | 11.44 | 35.0 | 20.2 | 47.8 |
| GT, σ = 2 | 10.78 | 33.7 | 20.4 | 46.6 |
| *for comparison:* GT tokens → decode | 7.82 | 29.3 | 12.5 | 40.8 |
| *for comparison:* MaDiS + refiner v0 | 11.25 | 34.4 | 21.2 | 49.3 |

**F19.**
- Cleaner (smoothed) GT does **not** back-translate better. The hypothesis that
  generated motion beats GT because GT is noisy (H2) is **not supported**. The
  shared-train-mapping explanation (H1, F17) remains.
- The tokenizer loses far more than temporal smoothing does (7.82 vs ≥ 10.8 BLEU).
  The loss is not in high-frequency jitter; it is information-bearing detail. This
  is consistent with F5: the 96-code body stream discards wrist / arm configuration.
- Against the paired crop-GT row, MaDiS + C1 reaches **100% of GT BLEU** and is
  slightly above it on retrieval. That is exactly where an evaluator with shared
  training data saturates.
- **A method that scores near GT under same-split BT cannot be ranked by these
  numbers.** Ranking at SOTA level needs the disjoint-data evaluators and the
  human study of C0.

## 16. Regression refiner ablation (job `eval_regress_v0`)

Same architecture and data as flow v0, but a deterministic L1 regression of the
residual. Phoenix dev, n = 519.

| | DTW body | DTW hands | rh speed / GT | B1 BLEU-4 | B1 R@1 | B2 BLEU-4 | B2 R@1 |
|---|---|---|---|---|---|---|---|
| GT (ceiling) | 0 | 0 | 1.00 | 11.49 | 20.6 | 10.62 | 17.5 |
| GT tokens + flow τ = 0 | **2.75** | **3.96** | 0.35 | 9.62 | 15.2 | 9.49 | 16.6 |
| GT tokens + regress | 2.87 | 4.04 | 0.45 | 10.52 | 17.3 | 9.43 | 16.6 |
| MaDiS | 5.19 | 6.94 | 0.35 | 9.76 | 20.0 | 8.48 | 18.3 |
| MaDiS + flow τ = 0 | 5.16 | 6.93 | 0.35 | 11.25 | 21.2 | 10.00 | 18.5 |
| **MaDiS + regress** | 5.22 | 6.94 | 0.46 | **12.50** | **23.3** | **10.56** | **19.7** |

**F20 — direct evidence of evaluator saturation.**
- The **deterministic regression refiner produces the most canonical / mean-like
  motion, and it scores above GT** under B1 (12.50 vs 11.49 BLEU, 23.3 vs 20.6 R@1).
  Under B2 it ties GT.
- This is the behaviour F17 predicts for back-translators trained on the generator's
  train split. **Neither BT-BLEU nor BT-retrieval from same-split evaluators can rank
  methods near the GT ceiling.**
- On DTW, flow τ = 0 is slightly better than regression for breaking the tokenizer
  floor (2.75 / 3.96 vs 2.87 / 4.04).
- Neither variant improves MaDiS's DTW, confirming F10 (planning-dominated error).

**Bottom line of this diagnostic round.** The current SLP benchmark protocol cannot
certify a SOTA improvement:
- DTW cannot (F7, F13);
- SiBLEU cannot (F8);
- same-split BT cannot (F17, F20).

The first contribution FLARE needs is therefore **C0 with disjoint-data evaluators
plus a human study**. Without it, any "SOTA" claim, ours or prior work's, is
unfalsifiable on these benchmarks.

## 17. Data-disjoint semantic evaluator: Uni-Sign (OpenASL) on How2Sign

This is the C0 evaluator that F17–F20 called for.

**Evaluator.** The released Uni-Sign pose-only SLT model fine-tuned on **OpenASL**
(`ZechengLi19/Uni-Sign/openasl_pose_only_slt.pth`, sha256 verified). It has never
seen How2Sign, SMPL-X, MaDiS or our back-translators.

**Input.**
- 133-D SMPL-X → FK (144 joints) → frontal orthographic projection → COCO-WholeBody-133;
- `scripts/smplx_to_cocowb.py`, with layout checked against real RTMPose;
- Uni-Sign's own `load_part_kp` normalisation and 4-beam decoding
  (`scripts/unisign_bt_eval.py`);
- sacreBLEU 13a (Uni-Sign's bundled implementation; ROUGE not installed in that venv).

**Data.** How2Sign val, n = 1,717; results in
`outputs/diagnostics/how2sign_val_unisign_openasl.json`.

| | BLEU-1 | BLEU-2 | BLEU-4 | % of GT BLEU-4 |
|---|---|---|---|---|
| GT (projected SMPL-X) | 19.50 | 8.06 | 2.50 | 100 |
| static mean pose | 8.04 | 0.66 | 0.04 | 2 |
| GT time-reversed | 15.42 | 3.90 | 0.42 | 17 |
| **MaDiS released** | 8.04 | 1.57 | **0.24** | **10** |

**F21 — the first non-saturated, order-sensitive, data-disjoint semantic measurement.**

1. **No saturation.** GT sits far above every generated or trivial row. The absolute
   GT level is low (cross-dataset evaluator on projected SMPL-X), but the
   *relative* scale separates methods.
2. **Order-sensitive.** Reversed GT loses 83% of BLEU-4, compared with ~20% under the
   same-split B1/B2 (F14). This evaluator therefore penalises wrong sign order.
3. **The published SOTA recovers ≈10% of the GT BLEU-4 on How2Sign.** Its BLEU-1
   equals a static mean pose. Under a judge it was not fit to, MaDiS conveys little
   sentence-specific content in open-domain ASL. This is consistent with the DTW
   finding F13 (MaDiS < mean pose on How2Sign body) and opposite to the
   near-ceiling numbers of same-split evaluators.

**Implications.**
- **C0 adopts this as the primary semantic metric for How2Sign:** Uni-Sign-OpenASL
  BLEU-4 reported as a percentage of the projected-GT ceiling, plus the trivial rows.
- The CSL-Daily analogue (Uni-Sign CSL-Daily on real RTMPose vs projected SMPL-X)
  and a How2Sign-trained Uni-Sign (same sentences, different modality) are queued.
- **SOTA headroom is large and measurable.** Moving How2Sign from 10% toward the GT
  ceiling under this judge is a credible SOTA target. Any gain can no longer come
  from fitting the evaluator, because the evaluator is fixed, external, and never
  saw How2Sign.

### 17b. Second external evaluator: Uni-Sign fine-tuned on How2Sign (real-video RTMPose)

This model has seen How2Sign train *sentences*, but only as real-video RTMPose
keypoints. It never saw SMPL-X, MaDiS or our back-translators. Results in
`outputs/diagnostics/how2sign_val_unisign_h2s.json`.

| How2Sign val (1,717) | BLEU-1 | BLEU-4 | % GT | *OpenASL-model BLEU-4* | *% GT* |
|---|---|---|---|---|---|
| GT (projected SMPL-X) | 26.71 | 6.14 | 100 | 2.50 | 100 |
| static mean pose | 9.25 | 0.63 | 10 | 0.04 | 2 |
| GT time-reversed | 16.49 | 1.00 | 16 | 0.42 | 17 |
| **MaDiS released** | 10.59 | **0.79** | **13** | 0.24 | 10 |

**F22.** Two external evaluators agree:
- MaDiS recovers ≈10–13% of the GT BLEU-4 on How2Sign and is barely above a static
  mean pose (0.79 vs 0.63).
- Reversed GT keeps only 16–17%, so both external evaluators are order-sensitive.
- The projection domain gap is moderate. Uni-Sign reports 14.5 BLEU-4 on real-video
  How2Sign test, and projected SMPL-X GT gets 6.14 on val, i.e. ≈40% of real-video
  quality survives SMPL-X fitting and projection.

**Conclusion for the SOTA target.** On How2Sign, open-domain ASL and the largest
benchmark, the actual SOTA is semantically far from solved. Both external judges put
the published SOTA near the trivial floor. **FLARE's headline target is now
concrete and falsifiable:**

> Raise How2Sign external-evaluator BLEU-4 from ≈10–13% of the projected-GT ceiling
> (MaDiS) significantly toward that ceiling, measured by Uni-Sign-OpenASL
> (data-disjoint) and Uni-Sign-How2Sign (modality-disjoint), with mean-pose and
> reversed-GT rows reported. DTW-JPE is kept as a secondary kinematic report.

### 17c. CSL-Daily with a real-video ceiling (Uni-Sign CSL-Daily pose-only)

The evaluator is trained on real-video RTMPose of CSL-Daily train and has never seen
SMPL-X or our models. Character-level BLEU follows Uni-Sign's CSL protocol. Real
RTMPose keypoints: `/home/dongvk/datasets/CSL_Daily_Sentence_Crop/keypoint`. Results
in `outputs/diagnostics/csl_val_unisign_csl.json`.

| CSL-Daily val (1,077) | BLEU-1 | BLEU-4 | % of projected-GT BLEU-4 |
|---|---|---|---|
| **real-video RTMPose** (evaluator ceiling) | 51.93 | 23.82 | 114 |
| **projected SMPL-X GT** | 48.12 | 20.92 | 100 |
| GT time-reversed | 24.91 | 3.05 | 15 |
| static mean pose | 7.23 | 0.05 | 0 |
| **MaDiS released** | 18.00 | **3.35** | **16** |

Example: GT 他今年四岁。 MaDiS → 他多大了? (topically related, semantically wrong).

**F23.**
1. **The projection pipeline is validated.** SMPL-X fitting plus our COCO-WholeBody
   projection keeps **88%** of real-video translation quality (20.92 vs 23.82). The
   real-video number matches Uni-Sign's published pose-only performance (≈25.6 on
   test).
2. **MaDiS reaches 16% of the GT ceiling on CSL-Daily**, the same level as
   time-reversed GT. Together with F21–F22 (How2Sign: 10–13%), every external
   evaluator says the published SOTA conveys little sentence-specific content, while
   in-domain metrics (DTW, SiBLEU, same-split BT) suggested near-GT performance.
3. **The C0 evaluation suite is now complete for 2 of 3 benchmarks.**
   - How2Sign: OpenASL (data-disjoint) and How2Sign (modality-disjoint).
   - CSL-Daily: modality-disjoint, with a real-video ceiling.
   - Phoenix: no external DGS model; same-split B1/B2 plus a planned human study.

## 18. C3 pilot 1: best-of-4 reranking with a same-split BT reward (How2Sign, null result)

**Setup.**
- 500 fixed How2Sign val clips; 4 MaDiS samples each (`scripts/madis_sample.py`,
  τ = 0.8, all-part confidence, UTC schedule).
- Reward: B1-H2S (mBART back-translator trained on How2Sign train; dev BLEU-4
  1.58 on GT, stopped at epoch 12 for overfitting). It uses the contrastive
  likelihood against 15 distractor sentences (`scripts/bt_rerank.py`).
- Selected samples are scored with both external Uni-Sign evaluators.
- Reward gain from selection: mean −2.71 → −2.40.

| 500 clips | OpenASL B1 / B2 / B4 | Uni-Sign-H2S B1 / B2 / B4 |
|---|---|---|
| GT | 14.77 / 5.35 / 1.56 | 18.61 / 7.87 / 2.43 |
| mean pose | 7.56 / 0.73 / 0.05 | 9.67 / 3.47 / 0.41 |
| MaDiS greedy | 7.73 / 1.52 / **0.35** | 10.33 / 3.36 / **0.88** |
| sample s0 | 9.25 / 1.71 / 0.15 | 11.01 / 3.55 / 0.41 |
| random of 4 | 8.69 / 1.67 / 0.18 | 11.30 / 3.79 / 0.73 |
| best-of-4 (B1-H2S reward) | 9.22 / **1.95** / 0.16 | **11.51** / 3.66 / 0.60 |

**F24 (null result).**
- Selecting by the same-split back-translation reward gives **no consistent external
  gain** over a random sample.
- Stochastic sampling raises BLEU-1/2 slightly but loses BLEU-4 against greedy.
- All differences are within the noise of BLEU-4 ≈ 0.1–0.9 on 500 clips.

**Diagnosis: the reward model is too weak.** B1-H2S reaches only 1.58 BLEU-4 on GT
poses, so its ranking among 4 similar samples is close to random.

**Revised C3 pairing.**
- **Reward:** Uni-Sign-How2Sign, GT ceiling 6.14, ≈4× stronger.
- **Evaluator:** Uni-Sign-OpenASL.

The two share no training data (How2Sign vs OpenASL). Queued as
`c3_bestofn_h2s_usreward`.

## 19. C3 pilot 2: best-of-4 with a strong external reward (Uni-Sign-H2S → evaluated by Uni-Sign-OpenASL)

The reward is the Uni-Sign How2Sign model's contrastive likelihood
(`scripts/unisign_rerank.py`, 15 distractors). The evaluator is the OpenASL model; the
two share no data. Same 500 clips and samples as §18.
- Reward gain from selection: −2.94 → −2.77.
- Argmax over the 4 samples: [107, 139, 122, 132], i.e. picks are spread across samples.

| 500 clips, Uni-Sign-OpenASL | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| GT | 14.77 | 5.35 | 2.67 | 1.56 |
| mean pose | 7.56 | 0.73 | 0.14 | 0.05 |
| MaDiS greedy | 7.73 | 1.52 | **0.64** | **0.35** |
| sample s0 / s1 | 9.25 / 9.16 | 1.71 / 1.73 | 0.45 / 0.31 | 0.15 / 0.12 |
| random of 4 | 8.69 | 1.67 | 0.44 | 0.18 |
| **best-of-4 (strong reward)** | **9.43** | **1.86** | 0.42 | 0.14 |

**F25.**
- Even with a 4× stronger, data-disjoint reward, test-time selection among MaDiS
  samples yields only marginal gains on the held-out evaluator: +0.7 BLEU-1 and
  +0.2 BLEU-2 over random, none on BLEU-3/4.
- MaDiS's samples are **uniformly wrong at the sentence level**. The 10–16%-of-ceiling
  gap (F21–F23) is a property of the generator's semantic grounding, not of decoding.

**Consequences for FLARE.**
1. Best-of-N / RAFT starting from MaDiS samples has a low ceiling. C3 needs either
   much larger N, or **training** (diffu-GRPO) that moves the policy, plus a stronger
   prior on *which signs* to produce.
2. This raises **C2 (lexical retrieval / grounding)** to the top priority: inject
   real sign segments for the content words. A cheap upper-bound probe,
   whole-sentence text-NN retrieval from train (§20, queued), tests how much
   retrieved real motion conveys under the external evaluator.

## 20. Whole-sentence text-NN retrieval vs. generation under the external evaluator (How2Sign val)

For each val sentence, copy the motion of the TRAIN clip whose sentence is most
similar (`scripts/build_nn_retrieval.py`, TF-IDF word 1–2-grams, train index only).
Only 34 of 1,724 val sentences have an exact train match. Evaluated with
Uni-Sign-OpenASL on 1,717 clips.

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| GT | 19.50 | 8.06 | 4.26 | 2.50 |
| mean pose | 8.04 | 0.66 | 0.14 | 0.04 |
| MaDiS released | 8.04 | 1.57 | 0.52 | 0.24 |
| **text-NN retrieval (no generation)** | 7.89 | **2.17** | **0.86** | **0.38** |

**F26.**
- A trivial retrieval baseline that generates nothing **beats the published SOTA
  generator by +58% BLEU-4 (and +38% BLEU-2)** under the data-disjoint evaluator.
- On DTW, the same kind of baseline is *worse* than a mean pose (F2: text-NN 6.57 vs
  mean pose 5.73 on Phoenix body). **DTW and external semantics rank retrieval and
  generation in opposite orders.**
- Real, partially matching sign motion conveys more sentence content than MaDiS's
  generated motion. This is the strongest evidence so far for **C2 (retrieval /
  grounding)** as FLARE's main lever.

**Training-free C2 probe (queued, `c2_rimd_h2s`): retrieval-initialised masked
diffusion.**
- The retrieved clip is encoded to tokens.
- A (1 − keep) fraction is re-masked.
- The released MaDiS denoises them conditioned on the *actual* sentence
  (`scripts/madis_sample.py --init_npz --keep`). This is SDEdit for masked
  diffusion: it keeps retrieved real-sign structure and adapts the rest to the text.
- keep ∈ {0.5, 0.75}, deterministic decoding.

## 21. C3 pilot 3: best-of-4 on CSL-Daily (weak same-split reward, strong external evaluator)

- 500 CSL val clips, 4 MaDiS samples each, τ = 0.8.
- Reward: B1-CSL (mBART back-translator, dev char-BLEU-4 ≈ 4.2 on GT), contrastive
  over 15 distractors. Reward gain: −2.27 → −1.81.
- Evaluator: Uni-Sign CSL-Daily on projected SMPL-X.

| 500 clips | BLEU-1 | BLEU-2 | BLEU-4 |
|---|---|---|---|
| GT (projected) | 50.38 | 38.26 | 23.89 |
| mean pose | 7.50 | 1.18 | 0.08 |
| MaDiS greedy | 20.91 | 11.49 | 5.24 |
| sample s0 | 20.66 | 11.36 | 4.90 |
| random of 4 | 20.22 | 11.17 | 5.16 |
| **best-of-4** | **21.29** | **11.81** | **5.53** |

**Paired bootstrap** (`scripts/paired_bootstrap.py`, 1,000 resamples, n = 500):

| best-of-4 vs | ΔBLEU-4 | 95% CI | p(Δ ≤ 0) |
|---|---|---|---|
| random of 4 | +0.37 | [−0.52, +1.24] | 0.20 |
| greedy | +0.29 | [−0.80, +1.36] | 0.30 |
| sample s0 | +0.63 | [−0.23, +1.47] | 0.06 |

**F27 — corrected after the bootstrap: not statistically significant.** The original
provisional reading is kept below for the record.
- On CSL, where the external evaluator works at a higher BLEU level and is less
  noisy, reward selection gives a **small, consistent gain on every BLEU order**:
  - +0.37 BLEU-4 (+7%) over a random sample;
  - +0.29 (+6%) over greedy.
- The direction is positive on every BLEU order, but **all 95% CIs include zero**.
  Best-of-4 reranking with a weak reward is at most a small, unproven effect.
  Detecting effects of this size (≈ +0.3–0.6 BLEU-4) needs the full val set or N ≫ 4.
- The effect is small relative to the gap to GT (5.5 vs 23.9). Like F25, it shows
  that selection among MaDiS samples cannot close the gap on its own.

## 22. C2 probe: retrieval-initialised masked diffusion (token space) — negative result

The NN-retrieved clip (§20) is encoded with the SOKE/MaDiS tokenizer. A fraction
(1 − keep) of its tokens is re-masked, and released MaDiS denoises the rest
conditioned on the actual sentence (`scripts/madis_sample.py --init_npz`,
deterministic, batch size 1). Evaluated on How2Sign val (n = 1,717).

| | OpenASL B1 / B2 / B3 / B4 | Uni-Sign-H2S B1 / B2 / B3 / B4 |
|---|---|---|
| GT | 19.50 / 8.06 / 4.26 / 2.50 | 26.71 / 14.48 / 9.14 / 6.14 |
| mean pose | 8.04 / 0.66 / 0.14 / 0.04 | 9.25 / 3.25 / 1.40 / 0.63 |
| MaDiS | 8.04 / 1.57 / 0.52 / 0.24 | 10.59 / 3.39 / 1.48 / 0.79 |
| **text-NN retrieval** | 7.89 / **2.17** / **0.86** / **0.38** | **13.31 / 6.32 / 3.70 / 2.41** |
| retrieval-init, keep 0.5 | 6.04 / 1.12 / 0.38 / 0.17 | 8.95 / 2.85 / 1.20 / 0.63 |
| retrieval-init, keep 0.75 | 5.96 / 1.04 / 0.28 / 0.08 | 8.79 / 2.63 / 1.00 / 0.46 |

**F28.**

1. **The token-space combination fails.** Both retrieval-init variants are below plain
   MaDiS. Keeping *more* retrieved tokens is *worse*. Splicing tokens of a
   different-sentence clip into MaDiS's bidirectional sequence produces incoherent
   token sequences, and the tokenizer itself discards a third of the
   back-translatable content (F16).
2. **Text-NN retrieval stays best.**
   - Data-disjoint OpenASL evaluator: +58% BLEU-4 over MaDiS.
   - Uni-Sign-H2S evaluator: 2.41, i.e. 39% of GT, ≈3× MaDiS.
   - The H2S number is **inflated**: that evaluator was trained on How2Sign train
     sentences, and NN copies train-set motion. The OpenASL number is the
     trustworthy one. The asymmetry is itself another instance of F17
     (shared-data bias).
3. **Design consequence: FLARE's C2 must operate in continuous motion space.**
   - The LM plans *which* real segments to use and in *what order*.
   - Retrieved real motion is kept in SMPL-X space and never passed through the
     tokenizer.
   - The continuous flow model (C1) generates the transitions / co-articulation
     between retrieved segments and the parts with no retrieval hit.
   - This matches the evidence: real motion carries semantics (F26), the tokenizer
     loses them (F16, F28), and C1 handles fidelity (F9).

## 23. ★ Retrieve-and-rerank beats the published SOTA under a data-disjoint evaluator (How2Sign val)

**System "FLARE-R0"** (training-free):
1. For the input sentence, retrieve the top-8 TRAIN clips by TF-IDF text similarity
   (`scripts/build_nn_retrieval.py --topk 8`).
2. Rerank the 8 real motions with the contrastive likelihood of the *input* sentence
   under Uni-Sign-How2Sign (reward), with 15 distractors (`scripts/unisign_rerank.py`).
3. Output the argmax motion.

The system uses only the val text, train motions and models trained on train data.
No val motion is used.

**Evaluator.** Uni-Sign-OpenASL. It shares no data with the reward model, the
retrieval index or MaDiS. n = 1,717.

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | % GT BLEU-4 |
|---|---|---|---|---|---|
| GT (projected SMPL-X) | 19.50 | 8.06 | 4.26 | 2.50 | 100 |
| static mean pose | 8.04 | 0.66 | 0.14 | 0.04 | 2 |
| MaDiS (published SOTA) | 8.04 | 1.57 | 0.52 | 0.24 | 10 |
| text-NN top-1 | 7.85 | 2.17 | 0.86 | 0.38 | 15 |
| random of top-8 | 9.12 | 2.69 | 1.22 | 0.62 | 25 |
| **FLARE-R0 (top-8 + Uni-Sign-H2S rerank)** | **14.24** | **4.55** | **1.93** | **0.90** | **36** |

**Paired bootstrap** (1,000 resamples, BLEU-4):

| comparison | Δ | 95% CI | p(Δ ≤ 0) |
|---|---|---|---|
| FLARE-R0 vs MaDiS | **+0.65** | [0.44, 0.87] | < 0.001 |
| FLARE-R0 vs top-1 retrieval | +0.52 | [0.32, 0.71] | < 0.001 |
| FLARE-R0 vs random-of-8 | +0.27 | [0.07, 0.48] | 0.009 |
| top-1 retrieval vs MaDiS | +0.14 | [−0.01, 0.29] | 0.039 |

**F29.**
- A training-free retrieve-and-rerank system **beats the published SOTA generator by
  3.7× BLEU-4 and +77% BLEU-1** under an evaluator that never saw How2Sign. The
  difference is significant at p < 0.001.
- It reaches **36% of the projected-GT BLEU-4 ceiling and 73% of the GT BLEU-1**,
  against 10% and 41% for MaDiS.
- The reward–evaluator pair is data-disjoint, so this is not evaluator fitting.

**Caveats (to be addressed before any claim).**
1. Retrieved motions are whole train sentences, so content words that are absent
   from them are never produced. Segment-level composition (R5–R7) is needed for
   open vocabulary.
2. Kinematic fidelity to the reference (DTW) is expected to be worse, because the
   length and timing come from another sentence. The DTW job
   (`dtw_retr_rerank_h2s`) is queued and will be reported as is.
3. One evaluator, one dataset, one seed of distractors. Needed:
   - CSL replication (only CSL-trained reward / evaluator pairs share data there; a
     disjoint CSL reward is needed);
   - a human comprehension study.

**Status of the SOTA objective.** On How2Sign, under the protocol this proposal
argues is the only credible one (C0), FLARE-R0 is a measured, significant
improvement over the published SOTA. The full FLARE (planner + segment lexicon +
flow renderer + RL) is designed to keep this semantic gain while restoring
kinematic fidelity and open-vocabulary coverage.

### 23b. Kinematic check of FLARE-R0 (job `dtw_retr_rerank_h2s`, n = 1,717)

| How2Sign val | DTW body / hands ↓ | per-step body / hands (cm) ↓ | rh speed / GT | OpenASL BLEU-4 ↑ |
|---|---|---|---|---|
| static mean pose | **5.99** / 10.09 | **4.25** / 7.09 | 0.00 | 0.04 |
| MaDiS | 6.61 / **9.61** | 4.70 / **6.62** | 0.43 | 0.24 |
| text-NN top-1 | 8.10 / 11.11 | 5.24 / 7.03 | 0.86 | 0.38 |
| **FLARE-R0** | 8.71 / 11.71 | 5.17 / 6.68 | **1.01** | **0.90** |

**F30.**
- FLARE-R0 is **worse on accumulated DTW** (+2.1 body / +2.1 hands vs MaDiS). Its
  timing and length come from another sentence. The same metric ranks a static mean
  pose best on body.
- On **per-step hand error** it is on par with MaDiS (6.68 vs 6.62 cm).
- Its articulation is natural: right-hand speed is exactly 1.01 × GT, against 0.43
  for MaDiS.
- The *trustworthy* semantic metric (data-disjoint BLEU-4) and the *standard*
  kinematic metric (accumulated DTW) **rank these systems in exactly opposite
  order**. This is the central measurement result of this proposal.

**Target for the full FLARE.** Keep FLARE-R0's semantic level while moving its
DTW / per-step error toward MaDiS's. The LM planner and the flow renderer (R6–R7)
are meant to fit the duration and timing of the retrieved segments to the input
sentence; reporting both axes is mandatory.

### 23c. Duration fix: retime FLARE-R0 to the MaDiS-predicted length (job `r0_retime`)

Each selected retrieved motion is resampled to the length MaDiS generates for the
same input sentence (`scripts/retime_to_reference_length.py`). This uses a
*generated* length, never the GT length.

| How2Sign val (n = 1,717) | DTW body / hands ↓ | per-step body / hands (cm) | rh speed / GT | OpenASL BLEU-1 / BLEU-4 ↑ |
|---|---|---|---|---|
| MaDiS | **6.61 / 9.61** | 4.70 / 6.62 | 0.43 | 8.04 / 0.24 |
| FLARE-R0 | 8.71 / 11.71 | 5.17 / 6.68 | 1.01 | **14.24 / 0.90** |
| **FLARE-R0 + MaDiS-length retiming** | 7.32 / 9.93 | 5.18 / 6.74 | 1.21 | 13.15 / 0.82 |

**Paired bootstrap** (BLEU-4):

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| retimed vs un-retimed | −0.08 | [−0.22, +0.06] | not significant |
| **retimed vs MaDiS** | **+0.58** | [0.37, 0.80] | < 0.001 |

**F31.**
- Duration alone closes ≈ 2/3 of the accumulated-DTW gap on body (8.71 → 7.32) and
  ≈ 85% on hands (11.71 → 9.93, against 9.61 for MaDiS).
- There is no significant semantic loss.
- The remaining body gap is per-step pose error (5.18 vs 4.70 cm): the retrieved
  signer's arm configuration differs from the reference. That is the job of the
  continuous renderer (C1 / R7).
- **A retrieval-based system can be near MaDiS on DTW while being 3.4× better on
  data-disjoint semantics.** This is the strongest evidence for the FLARE
  architecture ("retrieve real, plan with the LM, blend with flow"), and it sets the
  next concrete milestone: a learned duration / timing model plus a C1 renderer on
  top of FLARE-R0.

## 24. CSL replication of FLARE-R0, and an evaluator-memorisation test

**CSL-Daily val (n = 1,077), evaluator Uni-Sign-CSL** (trained on CSL-Daily *train*
real video). Reward: B1-CSL. Results in
`outputs/diagnostics/csl_val_retr_rerank_unisign_csl.json`.

| | BLEU-4 | % GT |
|---|---|---|
| GT (projected) | 20.92 | 100 |
| text-NN top-1 | 19.07 | 91 |
| top-8 + B1-CSL rerank | 16.47 | 79 |
| random of top-8 | 15.38 | 74 |
| MaDiS | 3.35 | 16 |
| mean pose | 0.05 | 0 |

Bootstrap:
- top-1 vs MaDiS: +15.7 [14.8, 16.7];
- rerank vs top-1: −2.6 [−3.1, −2.1];
- rerank vs random: +1.1 [0.4, 1.8].

**Memorisation test.** Compare the evaluator's output on retrieved motion with
(a) the val reference and (b) the *retrieved train clip's own sentence*.

| | evaluator(motion) vs val ref | evaluator(motion) vs retrieved train sentence | text-only copy of retrieved sentence vs val ref |
|---|---|---|---|
| **CSL, Uni-Sign-CSL (saw these train clips)**, top-1 | 19.07 | **96.98** | 19.59 |
| How2Sign, Uni-Sign-OpenASL (never saw How2Sign), top-1 | 0.38 | 2.93 | 7.85 |
| How2Sign, Uni-Sign-OpenASL, FLARE-R0 | 0.90 | 3.27 | 9.34 |

**F32 — the CSL retrieval numbers are an evaluator artefact.**
- Uni-Sign-CSL reads back the memorised train sentence of the retrieved clip
  (BLEU-4 97).
- Its score against the val reference equals the score of copying the retrieved
  sentence *as text* (19.07 vs 19.59).
- With a same-data evaluator, motion retrieval reduces to text retrieval. **These
  numbers must not be reported as production quality.**
- This also means **CSL currently has no valid external evaluator for retrieval-based
  systems.** Uni-Sign-CSL is valid only for systems whose outputs are not train-set
  motions (e.g. generators). Even then, F17-type bias remains possible.
- Before any CSL claim, we need a CSL evaluator trained on data disjoint from
  CSL-Daily train, e.g. Uni-Sign pre-trained on CSL-News only (stage-1/2 weights,
  not fine-tuned on CSL-Daily), with an SLT head.

**F33 — the How2Sign / OpenASL result is not memorisation.**
- OpenASL reads retrieved motions at BLEU-4 ≈ 3 against their own sentences, the
  same level at which it reads real GT How2Sign motion (2.50).
- It reads, rather than recalls.
- The FLARE-R0 gain therefore reflects (i) retrieval of semantically closer
  sentences (text-copy BLEU-4 7.85 → 9.34 thanks to the Uni-Sign-H2S reranker) and
  (ii) real motion conveying them.
- **F29 (How2Sign) stands; the CSL replication does not count.**

## 25. A data-disjoint CSL evaluator: Uni-Sign stage-1 (CSL-News only)

This is `csl_stage1_weight.pth`: pose-only translation pre-training on **CSL-News
only**, never fine-tuned on CSL-Daily (sha256 verified). It is disjoint from MaDiS,
from the retrieval index, and from the B1-CSL reward. Results in
`outputs/diagnostics/csl_val_cslnews_evaluator.json`, n = 1,077.

| | BLEU-1 | BLEU-4 | % projected-GT BLEU-4 |
|---|---|---|---|
| real-video RTMPose | 14.30 | 1.75 | 226 |
| GT (projected SMPL-X) | 14.60 | 0.78 | 100 |
| static mean pose | 13.08 | 0.24 | 31 |
| GT reversed | 11.69 | 0.25 | 33 |
| **MaDiS** | 11.72 | **0.18** | **23** |
| text-NN top-1 | 11.87 | 0.44 | 56 |
| random of top-8 | 11.75 | 0.38 | 49 |
| top-8 + B1-CSL rerank | 11.98 | 0.44 | 56 |

Bootstrap (BLEU-4):

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| rerank vs MaDiS | +0.26 | [0.07, 0.42] | < 0.001 |
| top-1 vs MaDiS | +0.26 | [0.10, 0.43] | < 0.001 |
| rerank vs random | +0.06 | [−0.17, +0.28] | n.s. |
| rerank vs top-1 | +0.00 | [−0.23, +0.22] | n.s. |

**F34.**
1. This evaluator is weak on CSL-Daily (domain shift CSL-News → CSL-Daily). The
   projected-GT ceiling is 0.78 BLEU-4, mean pose already reaches 31%, and
   real-video reaches 1.75. The projection keeps 44% of real-video quality under
   this evaluator; the same comparison was 88% under Uni-Sign-CSL.
2. Within its resolution, it reproduces the How2Sign pattern:
   - **MaDiS ≤ static mean pose** (0.18 vs 0.24);
   - **real-motion retrieval beats MaDiS significantly** (+0.26 BLEU-4,
     p < 0.001, about 2.5×).
3. The weak same-split B1-CSL reranker adds nothing, consistent with F24/F25: the
   reward must be strong. On How2Sign, Uni-Sign-H2S as reward gave a significant
   rerank gain (F29).

**Overall standing of the SOTA claim (dev splits only, test untouched):**

| Benchmark | Evaluator | Result | Status |
|---|---|---|---|
| How2Sign | data-disjoint OpenASL | FLARE-R0 (retrieve + strong-reward rerank) beats MaDiS by **+0.65 BLEU-4 (3.7×)**; the retimed variant by **+0.58**; both p < 0.001; no memorisation (F33) | strong |
| CSL-Daily | data-disjoint CSL-News | retrieval beats MaDiS by **+0.26 BLEU-4 (2.5×)**, p < 0.001 | evaluator weak, supporting evidence |
| Phoenix | – | no external DGS evaluator | open; needs a human study |

### 23d. Robustness of FLARE-R0 to the distractor seed (job `rerank_seeds_h2s`)

The reranker's 15 random distractor sentences were redrawn with seeds 1 and 2.
Evaluator: Uni-Sign-OpenASL, n = 1,717.

| distractor seed | BLEU-1 | BLEU-4 | Δ BLEU-4 vs MaDiS (95% CI) |
|---|---|---|---|
| 0 | 14.24 | 0.896 | +0.65 [0.44, 0.87] |
| 1 | 14.10 | 0.895 | +0.65 [0.46, 0.85] |
| 2 | 14.36 | 0.873 | +0.63 [0.42, 0.84] |
| **mean ± sd** | **14.23 ± 0.13** | **0.888 ± 0.013** | all p < 0.001 |
| MaDiS | 8.04 | 0.243 | – |

**F35.** The FLARE-R0 gain does not depend on the distractor draw (sd of BLEU-4 is
0.013). The gain over the published SOTA, **3.65× BLEU-4 and +77% BLEU-1**, is
significant in every seed.

## 26. Which stream carries the semantics? Body/hand stream swaps (job `hybrid_streams_h2s`)

Hybrids between the retimed FLARE-R0 (§23c) and MaDiS, at equal length
(`scripts/hybrid_streams.py`). How2Sign val, n = 1,717.

| | DTW body / hands | rh speed / GT | OpenASL BLEU-1 | BLEU-4 | Δ BLEU-4 vs retimed FLARE-R0 (95% CI) |
|---|---|---|---|---|---|
| MaDiS | 6.61 / 9.61 | 0.43 | 8.04 | 0.24 | −0.58 |
| retimed FLARE-R0 (all streams retrieved) | 7.32 / 9.93 | 1.21 | **13.15** | **0.82** | – |
| A: MaDiS body/arms + retrieved fingers | **6.61 / 9.62** | 0.56 | 8.79 | 0.19 | −0.63 [−0.84, −0.41] |
| B: retrieved body/arms + MaDiS fingers | 7.32 / 9.99 | 1.19 | 11.46 | 0.14 | −0.68 [−0.86, −0.49] |

**F36 — semantics require whole-body coherence; streams cannot be mixed.**
- DTW is decided by the body/arm stream: hybrid A equals MaDiS on DTW.
- But **both** hybrids lose the entire semantic gain. Neither differs significantly
  from MaDiS on BLEU-4.
- Retrieved fingers on generated arms carry nothing. Retrieved arms with generated
  fingers keep some word-level cues (BLEU-1 11.5) but no multi-word structure.
- The external evaluator, like a human, reads a sign from handshape, location and
  movement *jointly*.

**Design consequence for the renderer (R7).**
- Retrieved segments must be used **whole (all streams together)**.
- The renderer may change only timing, global placement and transitions *between*
  segments. It must not replace arm or finger streams within a segment.
- A cheap DTW improvement therefore cannot come from part-swapping. It has to come
  from (i) better duration modelling (F31: −2/3 of the gap) and (ii) segment-level
  composition that matches the input sentence more closely, so the arm
  trajectories themselves are closer to the reference.
- This also explains why token methods fail semantically: part-wise tokenisation
  (decoupled body / hand codebooks) and MoP fusion optimise each stream
  separately, and in-domain metrics (DTW per part) never test cross-stream
  coherence.

## 27. R5-lite: two-clip compositional retrieval (job `pair_retrieval_h2s`)

**Method.**
- Split the input sentence into two halves by words.
- Retrieve the top-4 TRAIN clips per half (TF-IDF).
- Form all 16 concatenations of *whole* clips (all streams, per F36).
- Retime each to the MaDiS-generated length.
- Rerank with the Uni-Sign-H2S contrastive reward on the full sentence
  (`scripts/build_pair_retrieval.py`).

Evaluator: Uni-Sign-OpenASL. How2Sign val, n = 1,717.

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | % GT BLEU-4 |
|---|---|---|---|---|---|
| GT | 19.50 | 8.06 | 4.26 | 2.50 | 100 |
| MaDiS | 8.04 | 1.57 | 0.52 | 0.24 | 10 |
| single-clip FLARE-R0 (retimed) | 13.15 | 4.05 | 1.75 | 0.82 | 33 |
| random of 16 pairs | 13.30 | 3.58 | 1.35 | 0.57 | 23 |
| **best pair (FLARE-R1)** | **15.53** | **4.98** | **2.22** | **1.08** | **43** |

Bootstrap (BLEU-4):

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| FLARE-R1 vs single-clip | +0.27 | [0.03, 0.50] | 0.008 |
| **FLARE-R1 vs MaDiS** | **+0.84** | [0.65, 1.04] | < 0.001 |
| FLARE-R1 vs random pair | +0.52 | [0.28, 0.76] | < 0.001 |

**F37.**
- Even crude composition of two whole real clips, with a naive half-split and no
  alignment or transition model, significantly beats single-clip retrieval.
- It reaches **43% of the GT ceiling (4.5× MaDiS) and 80% of GT BLEU-1**.
- This is direct evidence for the R5–R7 direction: finer segment composition plus
  a learned planner and transition renderer should raise semantics further.
- The DTW check is queued (`dtw_pairs_h2s`).

### 27b. DTW of FLARE-R1 (job `dtw_pairs_h2s`)

| How2Sign val (n = 1,717) | DTW body / hands ↓ | per-step body / hands (cm) ↓ | rh speed / GT | OpenASL BLEU-4 ↑ |
|---|---|---|---|---|
| MaDiS | **6.61** / 9.61 | **4.70** / 6.62 | 0.43 | 0.24 |
| FLARE-R0 retimed | 7.32 / 9.93 | 5.18 / 6.74 | 1.21 | 0.82 |
| **FLARE-R1 (best pair)** | 7.07 / **9.56** | 5.01 / **6.43** | 1.72 | **1.08** |

**F38.**
- FLARE-R1 improves **both** axes over single-clip retrieval.
- It **beats MaDiS on hand DTW** (9.56 vs 9.61) and on per-step hand error (6.43 vs
  6.62 cm), while having 4.5× its data-disjoint BLEU-4.
- Only body DTW is still behind (7.07 vs 6.61).
- Hand speed is now too high (1.72× GT): two whole clips are compressed into one
  predicted length. This is the next fix, via segment trimming / alignment (R5) and
  a transition renderer (R7), rather than resampling whole clips.

## 28. Proportional segment trimming (job `pair_trim_h2s`): negative result

Each retrieved clip of FLARE-R1 is trimmed to the time span proportional to the
positions of the words in its train sentence that overlap the query half, with ±10%
padding (`build_pair_retrieval.py --trim`).

| How2Sign val (n = 1,717) | OpenASL BLEU-1 / BLEU-4 | DTW body / hands | rh speed / GT |
|---|---|---|---|
| MaDiS | 8.04 / 0.24 | 6.61 / 9.61 | 0.43 |
| **FLARE-R1 (whole clips)** | **15.53 / 1.08** | **7.07 / 9.56** | 1.72 |
| FLARE-R1 + proportional trimming | 14.08 / 0.80 | 7.12 / 9.68 | 1.40 |

Bootstrap (BLEU-4):
- trimmed vs whole: −0.29 [−0.48, −0.10], significant loss;
- trimmed vs MaDiS: +0.55 [0.37, 0.74].

**F39.**
- An English word's position in the sentence is not a usable proxy for its sign's
  position in time. ASL word order and timing differ from English, so positional
  trimming cuts away semantically important motion.
- The heuristic reduces over-speed (1.72 → 1.40) but costs semantics and DTW.
- **R5 needs a learned text↔motion aligner** (e.g. a SiCLIP-style token-level
  aligner trained on train pairs, or attention from the Uni-Sign-H2S reward model)
  to locate sign segments. Until then, whole-clip composition (FLARE-R1) is the best
  training-free system.

**Current best system (dev, How2Sign): FLARE-R1.**
- 4.5× MaDiS on data-disjoint BLEU-4 (p < 0.001).
- Better hand DTW than MaDiS.
- Behind on body DTW by 0.46.

## 29. ★ FLARE-R1 ⊕ MaDiS soft blend: a system that dominates the published SOTA on every metric

This is a uniform blend of *all* streams, (1 − α)·FLARE-R1 + α·MaDiS, in normalised
133-D space at equal length (`scripts/blend_systems.py`). Uniform blending keeps
arms and fingers mutually coherent, unlike the per-stream swaps of F36. How2Sign val,
n = 1,717.

| | DTW body ↓ | DTW hands ↓ | per-step body / hands (cm) ↓ | rh speed / GT | OpenASL BLEU-1 ↑ | OpenASL BLEU-4 ↑ |
|---|---|---|---|---|---|---|
| static mean pose | 5.99 | 10.09 | 4.25 / 7.09 | 0.00 | 8.04 | 0.04 |
| **MaDiS (published SOTA)** | 6.61 | 9.61 | 4.70 / 6.62 | 0.43 | 8.04 | 0.24 |
| FLARE-R1 | 7.07 | 9.56 | 5.01 / 6.43 | 1.72 | **15.53** | **1.08** |
| **FLARE-R1 ⊕ 25% MaDiS ("FLARE-R1b")** | **6.45** | **9.09** | **4.57 / 6.10** | 1.35 | 13.79 | 0.72 |
| FLARE-R1 ⊕ 50% MaDiS | 6.17 | 9.03 | 4.37 / 6.11 | 0.97 | 10.85 | 0.23 |

Bootstrap (BLEU-4):

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| **FLARE-R1b vs MaDiS** | **+0.48** | [0.27, 0.69] | < 0.001 |
| FLARE-R1b vs FLARE-R1 | −0.36 | [−0.54, −0.18] | – |
| α = 0.5 vs MaDiS | −0.01 | [−0.15, +0.14] | n.s. |

**F40.**
- **FLARE-R1b is better than the published SOTA (MaDiS) on every metric we
  report:**
  - accumulated DTW-JPE body (−0.16) and hands (−0.52);
  - per-step body (−0.13 cm) and hands (−0.52 cm);
  - more natural hand speed (1.35 vs 0.43);
  - data-disjoint BLEU-1 (+72%) and BLEU-4 (+0.48, ≈ 3×, p < 0.001).
- α is a clean Pareto knob between semantics and DTW. At α = 0.5 the semantic gain
  disappears entirely while DTW keeps improving, which again shows that DTW rewards
  averaging (F1, F7).
- Caveats:
  - (i) The blend improves DTW partly *because* averaging improves DTW. Any SOTA
    claim must therefore lead with the data-disjoint semantic metric and report DTW
    as non-inferiority, which FLARE-R1b satisfies.
  - (ii) Dev split only, one dataset with a strong external evaluator (How2Sign).
  - (iii) No human study yet.
  - (iv) No training. The full FLARE (R5–R9) should move the Pareto front, not just
    pick a point on it.

## 30. FLARE-R1 / R1b on Phoenix-2014T dev (job `r1b_phoenix`)

**Setup.**
- Pair retrieval (k = 4 per half), reranked by B1-Phoenix (same-split mBART), at
  the MaDiS-generated length.
- R1b = 75% R1 + 25% MaDiS.
- Phoenix has no external DGS evaluator. Semantics are measured with B2
  (Qwen-LoRA, same split, different architecture), which is **weak evidence**:
  same-split back-translators saturate (F17–F20).

| Phoenix dev (n = 519) | DTW body / hands ↓ | per-step body / hands (cm) ↓ | rh speed / GT | B2 BLEU-4 | B2 R@1 |
|---|---|---|---|---|---|
| GT | 0 / 0 | – | 1.00 | 10.62 | 17.5 |
| mean pose | 5.53 / 7.94 | 5.34 / 7.63 | 0.00 | 0.13 | 1.7 |
| **MaDiS** | **5.19 / 6.94** | **4.55** / 5.56 | 0.35 | 8.48 | 18.3 |
| R1 | 6.17 / 7.91 | 5.23 / 6.00 | 1.52 | 11.39 | 16.2 |
| R1b | 5.51 / 7.26 | 4.69 / **5.53** | 1.19 | 10.32 | **19.3** |

**F41 — mixed result on Phoenix.**
- R1b is **behind MaDiS on accumulated DTW** (+0.32 body, +0.32 hands). It ties on
  per-step hands, and its hand speed is more natural.
- Under same-split BT, R1 / R1b are above MaDiS (BLEU-4 11.4 / 10.3 vs 8.5) and
  near or above GT. That is the saturated regime of F17/F20, so it is not
  evidence.
- Reasons the retrieval route is weaker here:
  - (i) the reranker is a same-split model (no external DGS SLT model exists),
    whereas the How2Sign reranker was the strong Uni-Sign-H2S;
  - (ii) MaDiS is relatively strongest on the small, repetitive weather domain.
- **The Pareto dominance of FLARE-R1b (F40) is therefore How2Sign-specific for
  now.** On Phoenix the result needs a strong reward / evaluator pair and a human
  study.

## 31. FLARE-R1 / R1b on CSL-Daily dev (job `r1b_csl`)

**Setup.** Pair retrieval (character halves, k = 4), reranked by B1-CSL (weak,
same-split), at MaDiS length. R1b = 75% R1 + 25% MaDiS. The evaluator is the
**data-disjoint** Uni-Sign CSL-News model. It is weak on CSL-Daily (GT ceiling 0.78
BLEU-4, F34).

| CSL dev (n = 1,077) | DTW body / hands ↓ | per-step body / hands (cm) ↓ | rh speed / GT | CSL-News BLEU-4 ↑ | % GT |
|---|---|---|---|---|---|
| GT (projected) | 0 / 0 | – | 1.00 | 0.78 | 100 |
| mean pose | 7.01 / 9.78 | 5.82 / 8.12 | 0.00 | 0.24 | 31 |
| **MaDiS** | **5.78 / 7.65** | **4.37 / 5.43** | 0.54 | 0.18 | 23 |
| **R1** | 6.85 / 8.62 | 5.19 / 6.08 | 1.89 | **0.67** | **86** |
| R1b | 6.16 / 7.95 | 4.69 / 5.57 | 1.48 | 0.37 | 47 |

Bootstrap (BLEU-4):
- R1 vs MaDiS: **+0.49 [0.35, 0.64], p < 0.001** (3.8×);
- R1b vs MaDiS: +0.19 [0.05, 0.30], p = 0.002.

**F42.**
- On CSL, the semantic advantage of composition-based retrieval replicates under a
  data-disjoint evaluator. R1 reaches 86% of the projected-GT BLEU-4 against 23%
  for MaDiS (p < 0.001).
- As on Phoenix, R1 and R1b remain **behind MaDiS on DTW**. The DTW–semantics
  trade-off is real, and the How2Sign Pareto dominance (F40) does not carry over
  with weak same-split rerankers.
- Blending trades away more semantics on CSL than on How2Sign (0.67 → 0.37).

**Summary across benchmarks (dev):**

| Benchmark | Evaluator | Composition retrieval vs MaDiS (semantics) | vs MaDiS on DTW |
|---|---|---|---|
| How2Sign | data-disjoint, strong | **+0.84 BLEU-4 (R1), +0.48 (R1b)**, p < 0.001 | **R1b better on all DTW** |
| CSL-Daily | data-disjoint, weak | **+0.49 (R1), +0.19 (R1b)**, p ≤ 0.002 | worse |
| Phoenix | no external evaluator | (same-split BT, saturated) | worse |

**Conclusion.**
- Data-disjoint semantics favour real-motion composition on the two benchmarks
  that have independent evaluators.
- Matching MaDiS on DTW requires the strong reward (How2Sign) or the learned
  components (R5–R9: aligner, planner, transition renderer) that the proposal
  specifies.

## 32. ★★ Held-out TEST confirmation on How2Sign (user-requested run)

**Provenance.** The user requested this test run in a separate session. It is logged
in `results/test_access_log.jsonl` (2026-11-04 15:19). The job
`how2sign_test_blends_20261104` used the configurations frozen on val (FLARE-R1
pairs k = 4, Uni-Sign-H2S rerank seed 0, blend α ∈ {0.25, 0.5}). Its log is
`outputs/jobs/how2sign_test_blends_20261104.log`. This proposal's own experiments
never touched test. Evaluator: Uni-Sign-OpenASL (data-disjoint). n = 2,300.

| How2Sign **test** | DTW body ↓ | DTW hands ↓ | per-step body / hands (cm) ↓ | rh speed / GT | OpenASL BLEU-1 ↑ | BLEU-4 ↑ |
|---|---|---|---|---|---|---|
| GT (projected) | – | – | – | 1.00 | 19.10 | 2.29 |
| **MaDiS (published SOTA)** | 6.58 | 9.42 | 4.69 / 6.50 | 0.45 | 8.42 | 0.14 |
| FLARE-R1 | 7.16 | 9.59 | 5.11 / 6.50 | 1.80 | **15.81** | **1.15** |
| **FLARE-R1b (α = 0.25)** | **6.51** | **9.07** | **4.64 / 6.14** | 1.42 | 14.06 | 0.81 |
| FLARE-R1 ⊕ 50% MaDiS | 6.20 | 8.96 | 4.42 / 6.08 | 1.03 | 11.00 | 0.27 |

Bootstrap (BLEU-4):
- **FLARE-R1b vs MaDiS: +0.68 [0.51, 0.85], p < 0.001**;
- α = 0.5 vs MaDiS: +0.14 [0.00, 0.26], p = 0.025.

**F43 — the val findings replicate on held-out test.**
- FLARE-R1b, a configuration frozen on val, is **better than the published SOTA
  (MaDiS) on every reported metric on How2Sign test**:
  - DTW body −0.07, hands −0.35;
  - per-step body −0.05 cm, hands −0.36 cm;
  - data-disjoint BLEU-1 +67%;
  - BLEU-4 ≈ 6× (p < 0.001).
- Unblended FLARE-R1 reaches 50% of the GT BLEU-4 ceiling against 6% for MaDiS.
- The α trade-off reproduces exactly.

Remaining caveats:
- The semantic metric is our C0 protocol (external SLT on projected SMPL-X), not
  the papers' metric.
- One benchmark (How2Sign).
- No human study.
- The method is training-free composition plus blending. The trained FLARE
  components (R5 aligner started, planner, renderer, RL) are still to come.

## 33. R5: learned text↔motion aligner and aligner-based segment trimming

**Aligner.** `methods/d2c_sign/aligner.py`:
- FILIP / SiCLIP-style token×frame max-similarity;
- frozen mBART encoder for text, part-aware conv/transformer for motion;
- symmetric InfoNCE;
- How2Sign **train only**, 15 epochs, 21 min.

Val text→motion retrieval in pools of 100: **R@1 27.6%, R@5 56.6%** (chance 1%).
The final train loss of 0.28 indicates some overfitting.

**Trimming.**
- For each query half, take the argmax frame of every content token in the
  retrieved clip.
- Keep the 10th–90th-percentile span, padded.
- Same pair / rerank / evaluation pipeline as FLARE-R1 (job `pair_aligned_h2s`).

| How2Sign val (n = 1,717) | OpenASL BLEU-1 / BLEU-4 | Δ BLEU-4 vs FLARE-R1 (95% CI) | DTW body / hands | rh speed / GT |
|---|---|---|---|---|
| MaDiS | 8.04 / 0.24 | – | 6.61 / 9.61 | 0.43 |
| **FLARE-R1 (whole clips)** | **15.53 / 1.08** | – | **7.07 / 9.56** | 1.72 |
| positional trimming (F39) | 14.08 / 0.80 | −0.29 [−0.48, −0.10] | 7.12 / 9.68 | 1.40 |
| **aligner trimming (R5)** | 13.51 / 0.94 | −0.15 [−0.33, +0.06] (n.s.) | 7.12 / 9.71 | 1.35 |

**F44.**
- The learned aligner localises signs much better than word position. The semantic
  cost of trimming halves and is no longer significant, and it remains +0.69 BLEU-4
  over MaDiS (p < 0.001).
- Trimming still does **not improve DTW**, and only modestly reduces over-speed
  (1.72 → 1.35).
- Conclusion: segment boundaries alone are not the bottleneck for DTW.
  - Concatenated real segments from *different signers / sentences* still differ
    in arm configuration from the reference.
  - The remaining DTW gap needs the **transition / pose-adaptation renderer (R7)**.
    For now, a uniform blend with the generator (F40/F43) is its training-free
    stand-in.
- The aligner is a useful asset for R6: it gives the planner token-level segment
  supervision.

## 34. R7 v0: learned pose-adaptation renderer, negative result (mean collapse)

**Setup.**
- Training data: 11,969 How2Sign TRAIN sentences
  (`scripts/build_train_compositions.py`).
- Input: a two-clip composition built from *other* train clips (top-1 per half,
  same-text clips excluded), retimed to the GT length.
- Target: the sentence's own GT, DTW-warped onto the composition timeline.
- Model: flow refiner (`methods/d2c_sign/refiner.py`), 30k iterations, applied at
  τ = 0 to FLARE-R1 on val.

| How2Sign val (n = 1,717) | DTW body / hands | per-step body / hands (cm) | rh speed / GT | OpenASL BLEU-1 / BLEU-4 |
|---|---|---|---|---|
| static mean pose | 5.99 / 10.09 | 4.25 / 7.09 | 0.00 | 8.04 / 0.04 |
| MaDiS | 6.61 / 9.61 | 4.70 / 6.62 | 0.43 | 8.04 / 0.24 |
| FLARE-R1 | 7.07 / 9.56 | 5.01 / 6.43 | 1.72 | 15.53 / 1.08 |
| FLARE-R1b | 6.45 / 9.09 | 4.57 / 6.10 | 1.35 | 13.79 / 0.72 |
| **R7 v0** | **5.96** / 10.75 | **4.17** / 7.42 | **0.13** | 4.32 / **0.06** |

Bootstrap (BLEU-4):
- R7 v0 vs MaDiS: −0.18 [−0.28, −0.05];
- vs FLARE-R1b: −0.66 [−0.84, −0.47].

**F45.**
- The renderer **collapsed to the conditional mean**. Hands are almost frozen
  (0.13 × GT speed) and semantics are at mean-pose level, yet body DTW is the
  *lowest of all systems*, below even the static mean pose. This is another
  direct demonstration that DTW rewards averaging (F1, F7, F40).
- **Cause: content-inconsistent training pairs.** The input content (other
  sentences' signs) does not determine the target content (this sentence's
  signs), so the MSE/flow-optimal deterministic output is the average.
- **Design rule for R7.** The renderer must be trained on pairs where the input
  *determines* the target content. Examples:
  - (a) the same sentence's own motion, cut at aligner-predicted boundaries
    (F44) and re-concatenated with synthetic timing / signer perturbations;
  - (b) retrieved clips that share gloss / aligner-matched words with the target,
    with the loss restricted to the matched segments.
  It should also be **stochastic** (sampling, not τ = 0) and carry a semantic
  guard: the C3 reward as a training signal, or BT likelihood early stopping.
- Until then, the training-free blend FLARE-R1b remains the best system on every
  axis (val F40, test F43).

## 35. Blend-weight sweeps on Phoenix and CSL (jobs `alpha_sweep_phoenix`, `alpha_sweep_csl`)

| dev | α (share of MaDiS) | DTW body / hands ↓ | per-step body / hands (cm) | rh speed / GT | external BLEU-4 (Δ vs MaDiS, 95% CI) |
|---|---|---|---|---|---|
| Phoenix (519) | MaDiS | 5.19 / 6.94 | 4.55 / 5.56 | 0.35 | n/a (no external DGS evaluator) |
| | 0.25 | 5.51 / 7.26 | 4.69 / 5.53 | 1.19 | n/a |
| | 0.4 | 5.24 / 7.03 | 4.51 / 5.40 | 0.98 | n/a |
| | 0.5 | **5.11 / 6.93** | **4.42 / 5.37** | 0.85 | n/a |
| | 0.6 | 5.03 / 6.87 | 4.38 / 5.38 | 0.71 | n/a |
| CSL (1,077) | MaDiS | 5.78 / 7.65 | 4.37 / 5.43 | 0.54 | 0.18 (CSL-News) |
| | 0.25 | 6.16 / 7.95 | 4.69 / 5.57 | 1.48 | 0.37 (+0.19 [0.05, 0.30]) |
| | 0.4 | 5.90 / 7.74 | 4.51 / 5.42 | 1.23 | 0.26 (+0.09 [−0.03, +0.19], n.s.) |
| | 0.5 | **5.77 / 7.64** | 4.42 / 5.36 | 1.07 | 0.14 (−0.03, n.s.) |
| | 0.6 | 5.67 / 7.56 | 4.33 / 5.33 | 0.91 | 0.16 (−0.02, n.s.) |

**F46.**
- On Phoenix and CSL, α ≥ 0.5 beats MaDiS on every DTW metric, with more natural hand
  speed. On CSL, where a data-disjoint evaluator exists, that is exactly where the
  semantic gain disappears. α ≤ 0.25 keeps a significant semantic gain but loses on
  DTW.
- **There is no dominating operating point on CSL.** The Pareto fronts cross. On
  Phoenix the semantic side cannot be measured, so its DTW "wins" at α ≥ 0.5 are not
  claimed.
- The difference from How2Sign (F40, F43) is the **reward strength**:
  - How2Sign reranks with Uni-Sign-H2S (GT ceiling 6.1 BLEU-4 in its own domain);
  - CSL / Phoenix rerank with weak same-split back-translators (F24–F27).
  Better reranking raises semantics *before* blending, leaving room to blend toward
  DTW.
- **Actionable:** a strong external reward for CSL (Uni-Sign CSL-News, fine-tuned on
  *other* CSL data, or the CSL-Daily Uni-Sign model used only as a reward with
  CSL-News as the evaluator) is the most direct route to CSL dominance.

## 36. CSL with a strong reward (Uni-Sign CSL-Daily as reranker, CSL-News as evaluator; job `csl_strong_reward`)

| CSL dev (n = 1,077) | DTW body / hands | per-step body / hands (cm) | CSL-News BLEU-4 | Δ vs MaDiS (95% CI) |
|---|---|---|---|---|
| MaDiS | **5.78 / 7.65** | 4.37 / 5.43 | 0.18 | – |
| R1 (weak B1-CSL rerank) | 6.85 / 8.62 | 5.19 / 6.08 | 0.67 | +0.49 [0.35, 0.64] |
| R1 (strong Uni-Sign-CSL rerank) | 6.78 / 8.53 | 5.15 / 6.01 | 0.64 | +0.46 [0.28, 0.64] |
| ⊕ 25% MaDiS | 6.11 / 7.89 | 4.65 / 5.52 | 0.40 | +0.22 [0.09, 0.33] |
| ⊕ 40% MaDiS | 5.86 / 7.69 | 4.47 / **5.39** | 0.28 | +0.10 [−0.02, +0.21], p = 0.06 |

**F47 (corrects the hypothesis in F46).**
- On CSL, the strong reranker is **no better** than the weak one (Δ = −0.03, n.s.).
  Reward strength is not what separates CSL from How2Sign.
- The closest CSL operating point (α = 0.4) is within 0.08 / 0.04 DTW of MaDiS, has
  better per-step hand error and a borderline semantic gain (p = 0.06). That is
  **not dominance**.
- Likely limits:
  - (i) the CSL-News evaluator's low resolution (GT ceiling 0.78 BLEU-4) makes small
    semantic gains unmeasurable;
  - (ii) CSL retrieval has a larger intrinsic DTW deficit (R1 +1.0 body vs +0.46 on
    How2Sign).
- A CSL win therefore needs either a stronger *independent* CSL evaluator, or the
  trained renderer (R7 v1) to lower DTW without averaging.

**Final standing (dev unless stated):**

| Benchmark | Best system | vs MaDiS on DTW | vs MaDiS on external semantics |
|---|---|---|---|
| **How2Sign (val + TEST)** | FLARE-R1b | **better on all** (test: 6.51 / 9.07 vs 6.58 / 9.42) | **+0.68 BLEU-4 test, p < 0.001** |
| CSL-Daily | R1 ⊕ 25–40% MaDiS | slightly worse | +0.10 to +0.22 (weak evaluator) |
| Phoenix | R1 ⊕ ≥ 50% MaDiS | better (DTW only) | not measurable |

# 01 — Literature review: Sign Language Production and the motion-generation toolbox

Scope: text→3-D sign motion (SMPL-X / keypoints), with the general
text-to-motion (T2M), whole-body and alignment literature that SLP borrows from.
Survey date 2026-11-03. Full per-paper tables with arXiv IDs and extracted
numbers are in the two raw appendices:
[`_raw_lit_slp.md`](_raw_lit_slp.md) (≈60 SLP papers) and
[`_raw_lit_motion.md`](_raw_lit_motion.md) (≈70 T2M papers). Items that could not
be verified against a primary source are marked `[unverified]` there.

---

## 1. Problem setting

Input: a spoken-language sentence (optionally glosses). Output: a sequence of
upper-body + hand + face parameters (here the SOKE 133-D SMPL-X feature,
see [02](02_codebase_and_data_audit.md)). Three properties make SLP harder than
generic T2M:

1. **Lexical precision.** Handshape, orientation, location and movement are
   phonemic: a 1-cm fingertip error can change the sign. Generic T2M tolerates
   "style" variation; SLP does not.
2. **Sequence-level linguistics.** One sentence ≈ 5–20 signs with co-articulation,
   word order different from the spoken sentence, and non-manual markers
   (mouthing, brows, head) carrying grammar.
3. **Small, noisy data.** 7k (Phoenix) – 30k (How2Sign) training sentences, and 3-D
   "ground truth" is a monocular SMPL-X fitting (SOKE, SignSparK and M3T all
   attribute errors to it).

---

## 2. Evolution of SLP approaches

| Era | Family | Representative work | What it fixed | What remained |
|---|---|---|---|---|
| 2020–22 | AR regression on keypoints | Progressive Transformers (2004.14874), Adversarial (2008.12405), MDN (2103.06982), MoMP (2107.11317), NSLP-G / NAR-KD (2208.06183) | first end-to-end text/gloss→pose | **regression to the mean**, under-articulated hands |
| 2022–24 | Diffusion on poses | G2P-DDM (2208.09141), Neural Sign Actors (2312.02702), Sign-IDD (2412.13609), MS2SL, Text2SignDiff | multimodality, sharper motion | slow, weak lexical precision |
| 2023–25 | VQ tokens + (L)LM | T2S-GPT (2406.07119), SignVQNet, SignLLM, **SOKE (2411.17799)** | leverage pretrained LMs, multilinguality; SOKE adds decoupled body/hand tokens + dictionary retrieval | AR decoding slow, quantization floor, retrieval noisy |
| 2024–26 | Retrieval / stitching | Spoken2Sign (2401.04730), Sign Stitching (2405.07663), SignRR (2608.28568) | lexical accuracy by copying real signs | transitions, open-vocabulary, gloss dependence |
| 2026 | Masked / parallel tokens | **MaDiS (2601.19577)**, SignMask, BLANC (2610.24109), SignPR (CVPR'26) | bidirectional context, parallel decoding, 40% faster | same SOKE tokenizer; no retrieval; no CFG/RL |
| 2026 | Flow matching / continuous | **SignSparK (2603.10446)**, HybridSign (2507.09105), SignNMFlow (2609.33339) | no quantization; 100× faster; keyframe control | relies on gloss→dictionary keyframes for its best numbers |
| 2026 | Better tokens / NMF | **M3T (2603.23617)** FSQ + SMPL-FX + aux. sign→text loss; SeRV (2609.05742) semantic RVQ; SignGPT (2609.21709); DigitCode (2608.03127) | face (NMF), codebook utilisation, semantics in tokens | hands still drift to mean (M3T) |
| 2026 | Gloss-timed generation | SIGNER (2506.07460), GARD (2606.21234) | temporal grounding of glosses → large BT gains | need text→gloss model |

Key trend: **the field converged on "discrete semantic planning (LM over part
tokens)" vs "continuous generation (flow) with lexical anchors"**. The best
numbers on each benchmark currently come from *different* families (table §4),
and no method combines a strong text-understanding LM, lexical retrieval,
and a continuous articulation model.

---

## 3. Techniques from general motion generation (what transfers)

Condensed from [`_raw_lit_motion.md`](_raw_lit_motion.md) §1–5, ranked by expected value for SLP.

| Technique | Source works | Status in SLP | Expected value |
|---|---|---|---|
| **Residual / multi-scale / FSQ tokenizers** | MoMask (2312.00063), MoMask++ (2507.09122), ScaMo FSQ (2412.14559), MoGenTS per-joint (2409.17686), HumanTOMATO H²VQ (2310.12978) | partial: M3T (FSQ), SeRV/SignRR (RVQ); SOKE/MaDiS still 96/192-code VQ | high for hands (quantization floor, §03) |
| **Tokens → continuous decoding** | DisCoRD (2411.18575, rectified-flow decoding of tokens), MARDM per-token diffusion head (2411.16575), MotionStreamer (2503.15451) | HybridSign (AR+flow, 2-D); not on SMPL-X LM tokens | high: removes the floor while keeping LM semantics |
| **Masked generative decoding + CFG in logit space** | MoMask, MMM, BAMM | MaDiS (no CFG, greedy, no remasking) | medium (cheap inference-time gains) |
| **Motion-aligned text encoders / semantic latents** | LaMP (2410.07093), MoLingo (2512.13840), TMR (2305.00976) | SiCLIP (evaluator only), SeRV | medium–high |
| **Retrieval augmentation with condition-mixture CFG** | ReMoDiffuse (2304.01116) | SOKE (prompt concat), SignRR, SignSparK SR | medium–high on small data |
| **Preference / RL post-training** | MoDiPO (2405.03803), SoPo (2412.05095), MotionRL (2410.06513), ReinDiffuse (2410.07296), AToM (2411.18654), HY-Motion RL (2512.23464); diffu-GRPO for masked-diffusion LMs: d1 (2504.12216) | **none for sign production** (GRPO only for SLT: SignRAG 2610.11371, SA-GRPO 2608.00235) | **high** — optimises the semantic metric (BT) directly |
| Contact / penetration guidance | OmniControl (2310.08580), ReinDiffuse physics rewards | none | medium (contact phonemes) |
| Large-scale motion pretraining | Motion-X++ (2501.05098), MotionMillion (2507.07095), HY-Motion | none (sign-only data) | uncertain for handshapes |
| Consistency / few-step distillation | MotionLCM (2404.19759) | none | speed only |

---

## 4. Benchmarks and the current leaderboard (SOKE protocol)

DTW-JPE body / hands (lower better; values are **accumulated DTW cost in m·frames**
— see §5), test split. Bold = best reported.

| Method | Family | Phoenix-2014T | CSL-Daily | How2Sign | needs |
|---|---|---|---|---|---|
| SOKE (ICCV'25) | AR LM + retrieval | 6.04 / 7.72 | 7.38 / 9.68 | 7.75 / 10.08 | dictionary |
| MaDiS (2026) | masked-diffusion LM | 4.97 / 6.59 | 5.99 / 7.96 | 6.59 / 9.45 | – |
| SeRV (2026) | semantic RVQ + GPT | – | – | 6.51 / **9.22** | – |
| M3T (BMVC'26) | FSQ AR + aux SLT | 4.02 / 8.55 | **4.63** / 9.28 | **4.86** / 9.44 | – |
| SignSparK noKF (ECCV'26) | flow matching | 4.40 / 7.10 | 6.27 / 10.63 | 6.30 / 11.43 | GT length [unverified] |
| SignSparK SR-T2P | flow + gloss→dict keyframes | **3.51 / 4.82** | 4.89 / **6.56** | n/a (no glosses) | text→gloss + dictionary |

MaDiS-protocol semantic metrics (SiBLEU body/hands, SiCLIP R@1): MaDiS
Phoenix 4.54/3.33, 57.94; CSL 4.87/4.08, 38.61; How2Sign 4.41/3.46, 34.27 —
all far above SOKE (≈2.5/2.0, 16–34). Back-translation BLEU-4 is **not comparable
across papers** (GT ceilings for How2Sign range 2.04–3.37 under recent evaluators,
yet SOKE reports 14.48 for generated How2Sign).

Observations:
* No single method is best on all three datasets or on both body and hands.
* Hands improve much more slowly than body, and three 2026 papers (M3T, SignFLIP,
  SignSparK) still report hands drifting toward the mean / incomplete closures.
* The strongest Phoenix/CSL results use **gloss-based dictionary keyframes**;
  gloss-free How2Sign remains the hardest and least saturated benchmark.

---

## 5. Evaluation is the weakest link

From the literature (§C.3 of the raw SLP survey) and our own code audit
([02](02_codebase_and_data_audit.md) §3.1, [03](03_diagnostics.md)):

1. **DTW-JPE is accumulated, not normalised** (SOKE `mGPT/metrics/dtw.py`, MaDiS
   `mGPT/metrics/dtw.py`): long clips dominate, and a static mean pose is a strong
   baseline (we measure that it beats SOKE's published numbers on Phoenix-dev, §03).
2. **SiBLEU** in the released code is a *unigram* overlap of SOKE-tokenizer IDs
   (order-free) and is only defined for that tokenizer.
3. **SiCLIP** ranks within a batch of 45 and its motion encoder is initialised from
   the same tokenizer as the generators it scores.
4. **Back-translation** evaluators are unnamed (SOKE) or method-specific
   (SignGPT), with GT ceilings that differ by 1.5–2×; reruns of SOKE in other
   papers differ by 1.5–2× in DTW.
5. Human studies use 2–16 signers; MaDiS finds DTW correlates weakly with signer
   ratings while SiBLEU/SiCLIP correlate strongly.
6. Meta-evaluation ("Meaningful pose-based SL evaluation", 2510.07453) found
   back-translation (likelihood) the most consistent automatic metric.

Implication for a SOTA claim: report the full SOKE+MaDiS metric set *and* a
calibrated protocol (trivial baselines, tokenizer floor, normalised DTW,
independent BT with GT ceiling, full-split SiCLIP). Gains that only appear on
one metric will not survive review.

---

## 6. Datasets beyond the three benchmarks

YouTube-ASL (984 h), YouTube3D (236k 3-D ASL pairs, SeRV), BOBSL, CSL-News
(1,985 h, Uni-Sign), SignAvatars (SMPL-X, 70k), Prompt2Sign (8 languages),
SignVerse-2M (55+ languages, 2-D). **No public sentence-level Vietnamese Sign
Language production corpus exists** (only VSL400 isolated recognition and
text→gloss MT corpora); the lab's `OurVSL_small` (≈15k sentence clips, 4 sources)
would be the first.

---

## 7. Ten papers to read in full (priority order)

1. MaDiS (2601.19577) — current LM SOTA; our backbone.
2. SignSparK (2603.10446) — continuous/keyframe alternative, best Phoenix/CSL hands.
3. M3T (2603.23617) — FSQ, face, aux sign→text loss, best body DTW.
4. SOKE (2411.17799) — protocol, tokenizer, retrieval.
5. BLANC (2610.24109) — newest; re-evaluates SOKE/SignSparK under one protocol.
6. SIGNER (2506.07460) — retrains all baselines under one protocol; gloss timing.
7. MoMask (2312.00063) + DisCoRD (2411.18575) — residual tokens and token→continuous decoding.
8. d1 / diffu-GRPO (2504.12216) — RL for masked-diffusion LMs.
9. BackTranslation2.0 (2606.28673) + Meaningful pose-based SL evaluation (2510.07453).
10. Uni-Sign (2501.15187) — strongest open SLT model, back-translation reward source.

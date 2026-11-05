# 04 — Research gaps and research questions

Each gap is tied to evidence: the literature ([01](01_literature_review.md)),
the code audit ([02](02_codebase_and_data_audit.md)), or our own measurements
([03](03_diagnostics.md)). Gaps are ordered by how much they limit the
current SOTA on the SOKE/MaDiS benchmark, not by novelty alone.

---

## G1. The quantization floor of the shared SOKE tokenizer

> Refinement (03 §3, F5): with only the fingers tokenized, the hand error is ≈1.0.
> The "hands" floor of 4.93 therefore comes mostly from **wrist/forearm orientation
> in the 96-code body stream**, not from finger codebooks. This argues for
> continuous correction (C1), not just bigger hand codebooks (M3T / SeRV-style).

**Evidence.** SOKE, MaDiS and every reproduction of them use one tokenizer:
96 body codes and 192 codes per hand, ×4 temporal down-sampling. On Phoenix-dev,
*perfect* tokens decoded by this tokenizer already incur DTW-JPE hands ≈ 4.8, against
MaDiS's 6.59 on test ([03](03_diagnostics.md) §2). SignSparK's continuous model
reaches 4.82 on test. **A token method using this tokenizer therefore cannot beat
SignSparK on hands even if the LM were perfect.** M3T and SeRV changed the
tokenizer (FSQ, semantic RVQ) but kept a single AR decoder and do not report the
floor.

**RQ1.** How much of the generation error of LM-based SLP is quantization error
and how much is planning error? Can a residual or continuous articulation stage
remove the quantization part **without** retraining the LM?

## G2. No method couples LM-level semantic planning with continuous articulation on SMPL-X

**Evidence.** The best numbers per benchmark come from different families: LMs
lead on semantics (SiBLEU/SiCLIP), flow models lead on articulation (DTW hands).
HybridSign (2-D pose) and DisCoRD (generic T2M) show that token→continuous decoding
works. We found no SMPL-X SLP method that decodes LM sign tokens with a continuous,
text-aware generative refiner.

**RQ2.** Does a token-conditioned flow-matching refiner close the gap between
masked-diffusion LMs and continuous generators on hand articulation, while
keeping the LM's semantic advantage? What training-input distribution makes it
robust to LM errors (GT tokens vs. corrupted tokens vs. self-generated tokens)?

## G3. No direct optimisation of semantic fidelity (RL / preference) for sign production

**Evidence.** Both surveys found **no** RL/DPO/GRPO work for sign *production*. GRPO
exists only for SLT (SignRAG, SA-GRPO), and RL/DPO for generic T2M (MotionRL,
SoPo, ReinDiffuse, HY-Motion). MaDiS is trained only by token cross-entropy, and the
only semantic signal in M3T is an auxiliary supervised SLT loss. Masked-diffusion
LMs now have a working policy-gradient method (diffu-GRPO, d1).

**RQ3.** Can back-translation-likelihood reward optimisation (diffu-GRPO on the
masked-diffusion sign LM) improve semantic fidelity (independent BT, SiCLIP, human
comprehension) without degrading kinematics? How do we prevent reward hacking when
reward and evaluator are both pose→text models?

## G4. Retrieval was dropped when moving to masked diffusion, and is noisy where used

**Evidence.** SOKE's dictionary prompt improves How2Sign PA-JPE from 7.91/3.10 to
6.82/2.35 ([01](01_literature_review.md) §4). MaDiS removed retrieval. SOKE's
keyword extraction is a length-sorted substring match across languages: German
`ich/aus` match English sentences (`scripts/name2kws_test.json`). SignSparK's best
results need text→gloss plus a dictionary. Phoenix ships frame-level gloss
alignments (`alignments.pami0`) that nobody exploits for a *train-split* lexicon.

**RQ4.** Does a clean, train-split-only lexicon (segments mined from aligned
training sentences), injected into a bidirectional masked-diffusion LM with
condition-mixture CFG, give SOKE-level retrieval gains without an external
dictionary or a gloss model at test time?

## G5. The evaluation protocol cannot separate good from trivial outputs

**Evidence.** Our own measurements ([03](03_diagnostics.md)) and code audit:

* Accumulated DTW-JPE: a **static mean pose** scores body ≈ 5.35, hands ≈ 7.45 on
  Phoenix-dev. That is better than SOKE's published test numbers (6.04 / 7.72) and
  better than M3T's hands (8.55).
* SiBLEU is unigram, order-free and tokenizer-specific. SiCLIP ranks within a batch
  of 45.
* Back-translation GT ceilings differ by 1.5–2× across papers. Reruns of SOKE differ
  by 1.5–2×.

* **Measured on all three benchmarks** (dev; [03](03_diagnostics.md) §2, §8, §9). The
  released MaDiS, which reproduces the paper, is compared with GT played in reverse
  and with a static mean pose:

  | Benchmark | MaDiS vs reversed GT | MaDiS vs mean pose |
  |---|---|---|
  | Phoenix | worse on body (5.19 vs 5.11) | – |
  | CSL-Daily | better on accumulated DTW, tied on per-step body | – |
  | How2Sign | worse on all DTW variants | worse on body (6.61 vs 5.99) |

**RQ5.** Which automatic metrics separate GT from trivial baselines (mean pose,
nearest-neighbour retrieval, time-reversed GT) and agree with signers? The minimum
credible protocol is a fixed evaluator suite with GT ceilings and trivial-baseline
floors that every claim must clear.

## G6. Non-manual features are generated poorly or not at all

**Evidence.** SOKE and MaDiS fold jaw plus 10 expression coefficients into the body
stream, and their metrics ignore the face entirely. M3T shows SMPL-X face codes
collapse (78.8% utilisation) and that lip articulation is the main remaining error.

**RQ6 (secondary).** Does a dedicated face stream with FSQ, conditioned on the manual
stream (EMAGE-style body→face hints), improve NMF accuracy at no manual cost?
*Out of scope for proposal 1*, apart from reporting a face DTW-VPE.

## G7. Low-resource sign languages (Vietnamese) have no production benchmark

**Evidence.** No public sentence-level VSL→3-D corpus or VSL SLP paper was found. The
lab holds `OurVSL_small`: ≈15k sentence clips, 4 sources, signer IDs, 2-D 133-kp.
SMPLer-X, HaMeR and WiLoR are available locally for fitting.

**RQ7 (follow-up).** Does multilingual pretraining (CSL + DGS + ASL) followed by
fine-tuning, plus the G4 lexicon, transfer to VSL with ≈10k pairs? This is a strong
*dataset + benchmark* contribution for a second paper. It needs SMPL-X fitting and
annotation QA first.

---

## Gap → contribution map

| Gap | Severity for SOTA | Novelty | Cost | In proposal 1? |
|---|---|---|---|---|
| G1 quantization floor | high (hands) | medium (diagnosis new; fix partly known) | low | yes, core |
| G2 LM + continuous refiner | high | medium-high on SMPL-X SLP | medium | yes, core (C1) |
| G3 RL with semantic reward | high (BT, SiCLIP) | **high (first for SLP)** | medium-high | yes, core (C3) |
| G4 train-split lexicon retrieval | medium | medium | medium | yes, C2 (ablatable) |
| G5 evaluation protocol | high (credibility) | medium-high | low | yes, C0 |
| G6 face/NMF | medium | low-medium (M3T) | medium | report only |
| G7 VSL benchmark | – (new benchmark) | high | high | follow-up paper |

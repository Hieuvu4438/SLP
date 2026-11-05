# 06 — Experiment plan, ablations, compute and timeline

Rules come from `docs/experiment_protocol.md`:
- all development happens on **val/dev**;
- test is run once per final configuration, only when the user asks;
- report ≥3 seeds (5 for the main table);
- baselines get an equal tuning budget.

## Phase 0 — Infrastructure and reproduction (week 1–2)

| # | Task | Output | Status |
|---|---|---|---|
| 0.1 | Pack Phoenix SOKE fittings for MaDiS | `scripts/madis_build_phoenix.py` → `data/processed/madis/Phoenix_2014T` | done |
| 0.2 | MaDiS env + released assets | `third_party/MaDiS/.venv`, `hf_assets/` (sha256 verified) | done |
| 0.3 | Calibration rows (oracle tokenizer, mean pose, NN text, reversed GT) | `scripts/madis_oracle_eval.py` → `outputs/diagnostics/` | done for Phoenix-dev ([03](03_diagnostics.md)) |
| 0.4 | Reproduce MaDiS SFT on **dev** (DTW, SiBLEU) for all 3 datasets | numbers vs paper | done (within ≈4%; F8 token issue) |
| 0.5 | Download and pack CSL-Daily and How2Sign SOKE fittings | `data/processed/madis/*` | done |
| 0.6 | Port DTW-JPE (accumulated + normalised) and SiBLEU (unigram + BLEU-4) into `src/slp/evaluation/metrics.py` with unit tests; register the `*_smplx` loaders | shared, tested evaluation | done (SiCLIP-full: todo) |
| 0.7 | Train back-translators B1 (mBART) and B2 (Qwen-LoRA) on GT train poses; GT ceiling; BT-retrieval | `outputs/bt/` | done for Phoenix (11.49 / 10.62 BLEU-4). **Disjoint-data evaluator: todo (F17)** |

## Phase 1 — C1 refiner (week 2–4)

1. Tokenizer pilot (running): VQ-192 vs RVQ-2 vs RVQ-4 for the right hand, trained
   on Phoenix train. This gives how much of the floor is removable at all.
2. Refiner on GT tokens → corrupted tokens → self-generated tokens with DTW-warped
   targets. Ablations:
   - input-distribution mix (a / b / c);
   - residual vs direct flow;
   - with / without the FK hand loss;
   - number of ODE steps (1, 2, 4, 8);
   - text conditioning on / off.
3. Report: DTW-JPE and PA (body, hands), normalised DTW, hand velocity / jerk
   statistics against GT, SiBLEU (unchanged), SiCLIP (re-encoded).

## Phase 2 — C2 lexicon + inference upgrades (week 4–6)

1. Training-free upgrades on MaDiS: all-part confidence, remasking, logit CFG.
   CFG needs one short SFT continuation with prompt dropout.
2. Lexicon induction on the train split (SiCLIP-style aligner). Measure lexicon
   precision on Phoenix using the HMM gloss alignments as reference.
3. Retrieval-conditioned SFT continuation plus condition-mixture CFG. Ablations:
   - no retrieval;
   - SOKE-style substring keywords;
   - induced lexicon;
   - gloss-alignment lexicon (upper bound).

## Phase 3 — C3 reward alignment (week 6–9)

1. Best-of-N (N = 1, 4, 8, 16) with B1. This is the test-time-scaling curve,
   scored with B2.
2. RAFT (top-k of N, 1–3 rounds).
3. diffu-GRPO with LoRA:
   - G = 8;
   - KL β ∈ {0.01, 0.05};
   - reward = BT-likelihood ± kinematic term.
4. Hacking checks:
   - B1 gain vs B2 gain;
   - SiCLIP;
   - DTW-PA-hands;
   - jerk;
   - 30-clip qualitative audit.

## Phase 4 — Scale out and final (week 9–12)

- CSL-Daily and How2Sign: repeat the best configuration with dataset-specific SFT,
  as MaDiS does.
- Main table: 5 seeds, all calibration rows, published baselines copied with their
  source.
- Human study.
- Test-split run **only on user request**, logged in `results/test_access_log.jsonl`.

## Ablation table (template for the paper)

| Row | C1 | C2-inf | C2-lex | C3 | DTW B/H ↓ | PA B/H ↓ | SiBLEU ↑ | SiCLIP-full R@1 ↑ | BT-B2 B4 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| mean pose | | | | | | | – | | |
| tokenizer oracle | | | | | | | 100 | | |
| MaDiS (ours, reproduced) | | | | | | | | | |
| + C1 | ✓ | | | | | | | | |
| + C2-inf | ✓ | ✓ | | | | | | | |
| + C2-lex | ✓ | ✓ | ✓ | | | | | | |
| + C3 (RAFT) | ✓ | ✓ | ✓ | ✓ | | | | | |
| + C3 (GRPO) = FLARE | ✓ | ✓ | ✓ | ✓ | | | | | |
| GT | | | | | 0 / 0 | 0 / 0 | 100 | ceiling | ceiling |

## Compute budget (one shared RTX 5880 Ada, ~20 GB usable)

| Job | Estimate |
|---|---|
| MaDiS inference on Phoenix dev (519 clips, 25 steps) | < 10 min |
| Refiner (≈60 M params) on Phoenix | ≈ 3–6 h per configuration |
| BT models B1 / B2 (≈100–600 M, LoRA) | ≈ 4–8 h each per dataset |
| RAFT round (N = 8 samples × 7k sentences + SFT) | ≈ 6–10 h |
| diffu-GRPO (LoRA, G = 8) | ≈ 1–2 days per dataset |
| CSL-Daily / How2Sign | ≈ 3× / 4× Phoenix |

If a second GPU becomes available, the dataset-specific SFT continuation in C2
is the first job to move there.

**Job policy.** The GPU is shared, so we run **at most 2–3 of our processes at a
time**, and every GPU job goes through the file-based queue `scripts/jobq.py`:
`add`, `run --workers 1`, `status`, `cancel`. Queue state and per-job logs live in
`outputs/jobs/` (`queue.jsonl`, `state.jsonl`, `<name>.log`). Interactive
evaluation runs outside the queue only when it is short and at most one at a time.

## Deliverables

1. Paper 1 (CVPR/ICCV/ECCV-style): FLARE-Sign plus the calibrated SLP evaluation
   protocol, with released code and calibration scripts.
2. Paper 2 (follow-up, dataset track): the first sentence-level 3-D Vietnamese
   SLP benchmark from `OurVSL_small`, with multilingual transfer from FLARE.

## Revision (2026-11-04): plan after the external-evaluator pilots (F21–F28)

**Primary metrics:**
- How2Sign: Uni-Sign-OpenASL BLEU-1..4, as a percentage of the projected-GT ceiling.
- CSL-Daily: Uni-Sign-CSL BLEU-1..4.
- Both reported with trivial rows and paired-bootstrap CIs (`scripts/paired_bootstrap.py`).
- DTW / SiBLEU / same-split BT are secondary.

**Baselines every variant must beat:**
- MaDiS released: How2Sign 0.24 BLEU-4 (OpenASL), CSL 3.35;
- text-NN retrieval (How2Sign 0.38);
- best-of-8 retrieval reranked by Uni-Sign-H2S (running).

| # | Step | Status / result |
|---|---|---|
| R1 | Projection pipeline SMPL-X → COCO-WB + Uni-Sign evaluators (OpenASL, H2S, CSL) | done; projection keeps 88% of real-video BLEU on CSL |
| R2 | Best-of-N over MaDiS samples (same-split and Uni-Sign rewards) | done: not significant (F24, F25, F27) |
| R3 | Token-space retrieval-init (SDEdit for masked diffusion) | done: worse than MaDiS (F28) |
| R4 | Top-8 text retrieval + Uni-Sign-H2S reward rerank (continuous, training-free) | running (`retr_rerank_h2s`) |
| R5 | **Segment lexicon** from train: word/phrase segments in continuous SMPL-X via a token-level text–motion aligner; Phoenix HMM alignments as upper bound | next |
| R6 | **Planner**: masked-diffusion LM (MaDiS init) predicts a sequence of lexicon ids, slots and durations; SFT on train alignments | after R5 |
| R7 | **Renderer**: flow matching with retrieved segments as inpainting anchors plus text; trained on train with GT segments as anchors | parallel to R6 |
| R8 | **C3 on the planner**: diffu-GRPO / RAFT with contrastive reward (H2S: Uni-Sign-H2S; CSL: B1-CSL); evaluator never in the loop | after R6, R7 |
| R9 | 3 seeds, full val, human study; test only on request | final |

**GPU note.** As of 2026-11-04 the shared GPU also runs four jobs from two other
users (≈46.6 / 49 GB in use). Our jobs stay at 1 worker. Evaluation-heavy steps
(Uni-Sign beam search on 1.7k clips) take about 30 min per row under contention.

## Status update (2026-11-04 evening)

| # | Step | Result |
|---|---|---|
| R4 | top-8 retrieval + Uni-Sign-H2S rerank (FLARE-R0) | +0.65 BLEU-4 vs MaDiS (val), stable over 3 seeds (F29, F35) |
| R5-lite | two-clip composition (FLARE-R1) | +0.84 BLEU-4 vs MaDiS (val), better hand DTW (F37, F38) |
| R7-lite | uniform blend 75/25 with MaDiS (FLARE-R1b) | **better than MaDiS on every metric, val (F40) and held-out TEST (F43, user-run)** |
| R5 | learned text↔motion aligner | R@1 27.6% / 100; better trimming than positional, but no DTW gain (F44) |
| R7 v0 | learned renderer on content-inconsistent pairs | mean collapse, negative (F45) |
| – | CSL / Phoenix replications | semantic gain replicates on CSL (weak evaluator); DTW worse on both; no Pareto dominance (F41, F42) |

**Next (in order):**
1. **R7 v1:** a renderer trained on content-consistent pairs (aligner-cut own
   segments with signer / timing perturbations), stochastic, with a semantic guard.
2. **R6:** an LM planner over aligner-derived segments (replaces TF-IDF half-split
   retrieval).
3. **C3 RL** on the planner (How2Sign: Uni-Sign-H2S reward, OpenASL evaluator).
4. A stronger CSL reward (an external CSL model never fine-tuned on CSL-Daily),
   so the How2Sign Pareto result can be reproduced there.
5. A human study (≥ 8 Deaf signers per language).

Test runs only on explicit user request.

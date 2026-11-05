# 05 — Proposed method: FLARE-Sign

**FLARE** = **F**low-refined, **L**exicon-**A**ugmented, **RE**ward-aligned masked-diffusion sign generation.

Code folder: `methods/d2c_sign/` (working name "discrete-to-continuous"; it will be
renamed to `flare_sign` once the pilot runs finish).

## 0. One-paragraph summary

Keep what the masked-diffusion LM does best, namely sentence-level semantic
planning over part-wise sign tokens (MaDiS, released checkpoints). Remove what
limits it: (i) the **quantization floor** of the 96/192-code tokenizer, via a
token-conditioned **flow-matching articulation refiner** that outputs continuous
SMPL-X; (ii) **lexical errors**, via a train-split **lexicon** induced from
parallel data and injected as retrieval conditioning with condition-mixture CFG;
(iii) the **mismatch between the training objective and semantic fidelity**, via
**back-translation-reward policy optimisation** (diffu-GRPO) of the LM. All of this
is evaluated under a calibrated protocol (C0) in which every result must clear the
trivial-baseline floors and is reported against GT ceilings.

```
text ─┬─► [C2 lexicon retrieval] ─► retrieved sign-token snippets ─┐
      │                                                             ▼
      └────────────────────────────► masked-diffusion LM (Qwen3-0.6B, MaDiS init)
                                        │  part tokens c = (c_body, c_lh, c_rh) @ T/4
                                        ▼   (+ C3: diffu-GRPO with BT reward)
                                   frozen VQ decoder D(c) → coarse motion x̃
                                        ▼
                       [C1 flow refiner R_φ(x_t, t | x̃, e(c), text)]  → continuous x̂ (133-D SMPL-X)
```

---

## C0. Calibrated evaluation protocol (prerequisite)

This is implemented once in `src/slp/evaluation/metrics.py`, with unit tests, per CLAUDE.md.

1. **Paper-compatible metrics, unchanged.** DTW-JPE and DTW-PA-JPE (body, hands),
   SiBLEU, SiCLIP R@1/5/10, all using the released code. Exception: SiBLEU reference tokens are **re-encoded from the evaluated GT poses**, because the released val/dev reference tokens do not match the GT poses ([03](03_diagnostics.md) §6, F8). Both variants are reported. Without these, the
   SOTA claim cannot be compared with prior tables.
2. **Calibration rows reported in every table:**
   - tokenizer oracle (GT → tokens → decode);
   - mean pose;
   - text nearest-neighbour train clip;
   - time-reversed GT;
   - GT itself (ceiling of back-translation and SiCLIP).
   These rows are already implemented in `scripts/madis_oracle_eval.py`.
3. **Normalised DTW** (per-path-step) next to the accumulated value.
4. **Independent back-translation (BT).**
   - Model B2 is pose→text and is *not* used for training (architecture and seed
     different from the reward model B1).
   - Report BLEU-4, chrF and BT likelihood, plus each metric's GT ceiling.
   - SignSparK-BT is reported where its pose format applies.
5. **SiCLIP over the full split** (1 vs all), in addition to the batch-45 number.
6. **Human study** with ≥8 Deaf signers per language where available, scoring
   comprehension (translate what you see) and naturalness. Small but
   pre-registered.

### C0 update after F21 (implemented)

**Primary semantic metric = a data-disjoint Uni-Sign evaluator on projected SMPL-X**
(`scripts/smplx_to_cocowb.py` + `scripts/unisign_bt_eval.py`), reported as:
- BLEU-1/4;
- % of the projected-GT ceiling;
- the trivial rows (mean pose, reversed GT).

Evaluator per benchmark:

| Benchmark | Evaluator |
|---|---|
| How2Sign | Uni-Sign fine-tuned on **OpenASL** (never saw How2Sign) |
| CSL-Daily | Uni-Sign CSL-Daily (real-video RTMPose; same sentences, different modality, never saw SMPL-X), with the real-RTMPose ceiling reported |
| Phoenix | no external DGS model; same-split B1/B2 plus a human study, flagged as weaker |

On How2Sign, MaDiS reaches 10% of the GT ceiling (BLEU-4 0.24 vs 2.50) and mean pose
2%. The metric is non-saturated and order-sensitive (reversed GT: 17%).

**C3 pairing.** The reward is a back-translator trained on How2Sign train (B1-style).
The evaluator is Uni-Sign-OpenASL. They share no data and no architecture.

### Re-prioritisation after the first C3 pilots (F24, F25)

Best-of-4 reranking of MaDiS samples gives at most +0.7 BLEU-1 under the held-out
evaluator, even with a strong, data-disjoint reward. MaDiS's samples are uniformly
wrong at the sentence level. The current order is therefore:

1. **C0** (done for How2Sign and CSL).
2. **C2 — lexical grounding / retrieval.** This is now the main lever. Real sign
   segments for content words supply the semantic content that MaDiS's samples lack.
3. **C3 as training, not reranking.** diffu-GRPO / RAFT over many rounds with a
   strong reward:
   - How2Sign: Uni-Sign-H2S reward, Uni-Sign-OpenASL evaluator;
   - CSL: a stronger CSL reward is needed (B1-CSL reaches only 4.2 BLEU-4).
4. **C1** as the fidelity stage.

### Final revision of the architecture after F26–F28: "retrieve real, plan with the LM, blend with flow"

Evidence so far:

| Evidence | Finding |
|---|---|
| text-NN retrieval of real motion > MaDiS generation | F26 |
| tokenizer round-trips destroy semantic content | F16, F28 |
| reranking MaDiS samples cannot close the gap | F24, F25, F27 |
| a continuous refiner lowers the fidelity floor | F9 |

FLARE-Sign is therefore restructured as a **hybrid continuous retrieval–generation
model**:

1. **Segment lexicon (C2).**
   - Mine word / phrase-level motion segments from the *train* split.
   - Alignment: SiCLIP-style token-level text↔motion similarity, or Phoenix HMM
     gloss alignments for an upper-bound ablation.
   - Segments are stored as **continuous SMPL-X**, not tokens.
2. **LM planner.** A masked-diffusion LM (MaDiS init) predicts a *plan*: the
   sequence of lexicon entries (or "generate" slots), their order, and durations.
   It no longer emits all motion tokens. The plan space is small and discrete,
   which suits LM and RL training.
3. **Continuous renderer (C1 extended).** A flow-matching model generates the full
   sequence conditioned on the text and on the planned segments as *inpainting
   anchors* (SignSparK-style keyframe inpainting, but anchors are retrieved real
   segments, not GT keyframes). It fills transitions and the "generate" slots.
4. **Reward alignment (C3)** is applied to the *planner*:
   - How2Sign: Uni-Sign-H2S reward, Uni-Sign-OpenASL evaluator;
   - CSL: B1-CSL reward, Uni-Sign-CSL evaluator.

**Strong, cheap baselines that every FLARE variant must beat** under the external
evaluators:
- whole-sentence text-NN retrieval (§20), already above SOTA;
- MaDiS.

## C1. Flow-matching articulation refiner (core, cheapest, largest expected DTW gain)

**Input**
- LM tokens c ∈ ({96}×{192}×{192})^{T/4};
- coarse motion x̃ = D(c) from the frozen SOKE decoders;
- token embeddings e(c) (the codebook vectors, up-sampled ×4);
- a text embedding from the frozen LM prompt hidden states.

**Output.** Continuous x̂ ∈ R^{T×133}.

**Formulation.** We use residual rectified flow on Δ = x − x̃:
x_t = (1−t)·ε + t·Δ, with ε ~ N(0, σ²I) and a small σ (the residual is small).
The network v_φ(x_t, t, x̃, e(c), text) is trained with

L = ‖v_φ − (Δ − ε)‖² + λ_fk·‖FK(x̃+Δ̂) − FK(x)‖₁ (hand joints, wrist-aligned) + λ_vel·‖∂_t(·)‖₁

At inference, 4–8 Euler/Heun steps are enough, because the residual distribution is
narrow.

**Backbone.** A 1-D temporal U-Net / DiT with AdaLN (SignSparK-like but ~10× smaller,
around 50–80 M parameters), cross-attention to text, and the decoder input
concatenated with e(c).

**Training-input distribution.** This is the crux, because the refiner must work
on *LM* tokens rather than GT tokens:
* (a) GT tokens, 30%.
* (b) Corrupted GT tokens, 40%: random code replacement at p ∈ [0, 0.3],
  sampled per part, and part-wise temporal jitter of ±1 token. This follows the
  token-corruption trick of T2M-GPT.
* (c) Self-generated, 30%: MaDiS SFT tokens decoded on *training* sentences. The
  GT target is **DTW-warped onto the generated time axis**, so the refiner learns
  to fix articulation while keeping the LM's timing. We call these
  timing-preserving targets.

**Why it should work.**
- Our measurements give a tokenizer floor (hands ≈ 4.8) close to SignSparK's test
  hands (4.82), and MaDiS adds ≈ 1.8 on top.
- The refiner attacks the first term directly. Through (b) and (c) it also reduces
  part of the second (planning noise at the articulation level).
- DisCoRD (generic T2M) and HybridSign (2-D SLP) show that token→continuous decoding
  improves fidelity.

**Pilot already running.** `methods/d2c_sign/tokenizer_pilot.py` measures how much of
the floor a residual tokenizer removes. See [03](03_diagnostics.md) §3.

**Revision after the first experiments.** Details are in [03](03_diagnostics.md) F5, F7 and §7.

- About 97% of the hand floor comes from the body tokenizer, i.e. wrist and forearm
  orientation.
- MaDiS hands move at 35% of GT speed.
- A pure sampled-residual flow (v0) restores articulation (0.87 of GT speed) but
  loses about 0.15–0.2 accumulated DTW.

C1 is therefore split into two parts:

* **C1a, conditional-mean correction of the low-frequency arm/wrist chain.** This is
  deterministic: regression, or the flow at temperature 0 on low-pass-filtered
  residuals. It targets the body-stream floor and should lower DTW.
* **C1b, sampled high-frequency articulation residual.** This is the flow at
  temperature τ on the remaining band. It targets under-articulation and should
  raise back-translation and SiCLIP, with DTW as a non-inferiority check
  (≤ +2% versus C1a).

τ and the band split are tuned on dev. Ablation rows: v0 (single flow),
C1a only, C1a + C1b.

**Second revision: re-prioritisation after F10.** C1 only removes quantization
error (≈0.4 / 0.8 of the floor). MaDiS's dominant error is planning (≈2.1 / 2.1).
Priorities are now **C3 > C2 > C1**:
- C1 is kept as the fidelity stage that lets planning gains show up below the
  tokenizer floor.
- Its next version (v1) is (i) trained on MaDiS self-generated train-set tokens
  with DTW-warped targets and (ii) cross-attends to text, to test whether it can
  also correct planning errors.

**Third revision: what C1 can and cannot claim, after F18–F20.**
- **Can claim:** C1 lowers the tokenizer floor (−12% / −17% DTW with perfect tokens).
  Given correct content, it raises sentence-specific recognisability
  (BT R@1 12 → 16–17 under two evaluators).
- **Cannot claim yet:** "better semantics than MaDiS". The +1.5 BT-BLEU on MaDiS
  outputs is in the saturated regime where even a mean-like regression output beats
  GT (F20). It must be re-measured with disjoint-data evaluators and humans (C0)
  before it enters a paper.

**Novelty versus the closest works**

| Work | Differs from C1 by |
|---|---|
| DisCoRD | generic T2M; not conditioned on text or an LM; no self-generated / DTW-warped training |
| HybridSign | 2-D pose; AR + flow; no LM tokens on SMPL-X |
| SignRR | refines *retrieved* real clips with an RVQ-VAE; no LM |
| SignSparK | keyframe inpainting; no LM tokens |

## C2. Train-split lexicon retrieval for the masked-diffusion LM

1. **Lexicon induction.** We use no external dictionary and never touch the test
   split.
   - Train a fine-grained text↔sign-token aligner on the train split only, using
     SiCLIP's token-level similarity (`get_clip_logits` in MaDiS is already
     token × pose-frame).
   - For each training sentence, take the arg-max alignment span of each content
     word and keep spans with high, consistent similarity across occurrences.
   - Phoenix also offers HMM gloss alignments (`alignments.pami0`) as an
     alternative, gloss-supervised lexicon for an upper-bound ablation.
   - The result is L: word/lemma → K prototype token snippets (medoid by DTW in
     token-latent space).
2. **Retrieval at test time.** Lemmatise the sentence and look up content words
   (exact or embedding nearest neighbour, which allows open vocabulary). Concatenate
   up to M snippets to the prompt as `[word] <snippet tokens>`, as SOKE does, but
   with MoP embeddings and a separate segment-type embedding.
3. **Training.**
   - Continue SFT from the MaDiS checkpoint with retrieval inputs.
   - Drop retrieval with p = 0.2 and the text with p = 0.1. This allows
     **condition-mixture CFG** (ReMoDiffuse) in logit space at inference:
     ℓ = ℓ(∅) + w_t·(ℓ(text) − ℓ(∅)) + w_r·(ℓ(text, ret) − ℓ(text)).
4. **Inference upgrades.** Masked diffusion gets several training-free upgrades,
   each reported separately:
   - confidence computed from **all three** part heads (MaDiS uses body only);
   - low-confidence remasking;
   - CFG.

## C3. Reward-aligned post-training (first RL for sign production)

**Reward model B1.** A pose→text back-translator trained on GT SMPL-X train poses
(a 133-D → transformer encoder feeding a Qwen3-0.6B decoder with LoRA, or an
mBART decoder). The reward is the **length-normalised log-likelihood** of the
reference sentence, which the meta-evaluation found to be the most consistent BT
signal:

r(x̂) = (1/|y|)·log p_B1(y | x̂) + β_k·r_kin(x̂)

r_kin penalises jerk above the GT 99th percentile, hand-hand inter-penetration,
and joint-limit violations, which guards against reward hacking with unnatural
motion.

**Reward revision after F14** ([03](03_diagnostics.md) §10b). With GT poses played in
reverse, B1 still recovers 80% of the GT BLEU, so the plain BT likelihood is only
weakly order-sensitive. The reward becomes **order-contrastive**:

r(x̂) = ℓ(y | x̂) − γ·max(0, ℓ(y | rev(x̂)) − ℓ(y | x̂) + m) + β_k·r_kin(x̂),  where ℓ = length-normalised log p_B1

This rewards motion whose correct-order reading is clearly more likely than its
reversed reading. It also enables cheap order augmentation for B1: reversed and
segment-shuffled GT are used as negatives with an unlikelihood loss. B2 stays
unaugmented so that it remains an independent check.

**Reward revision after F16** ([03](03_diagnostics.md) §12). MaDiS output
back-translates *better* than a perfect-token decode (9.76 vs 7.82 BLEU). A plain
in-domain BT reward is therefore hackable toward typical, generic signing. The
semantic term is replaced by the **contrastive BT likelihood**
ℓ(y | x̂) − log Σ_k exp ℓ(y_k | x̂), with batch / dev distractor sentences y_k. The
matching evaluation metric is BT-retrieval R@k (C0).

**Reward / evaluator independence after F17** ([03](03_diagnostics.md) §13). MaDiS + C1
scores *above* GT under a back-translator trained on the same train split. That
evaluator rewards the train-canonical mapping it shares with the generator. Hence:

- **Reward model B1:** trained on the generator's own train split.
- **Evaluators B2 / B3:** trained on **disjoint data**:
  - B2 is a signer-held-out back-translator. Phoenix and CSL have signer IDs; the
    generator's signers are excluded from B2's training data, or alternatively the
    generator is trained on signers A and B2 on signers B.
  - B3 is an external SLT model, e.g. Uni-Sign pose pre-trained on CSL-News, run on
    rendered 2-D keypoints.
- **Human comprehension** remains the final arbiter.
- Any **super-GT** automatic score is reported as evaluator saturation.

**Policy.** The masked-diffusion LM, which generates c. The reward is computed on
R_φ(D(c)), and R_φ is frozen during RL.

**Algorithm.** Three steps of increasing cost and risk:
1. **Best-of-N reranking** at test time with B1. This gives the test-time-scaling
   upper bound; it is reported but clearly labelled.
2. **RAFT / rejection-sampling SFT.** Keep the top-k of N samples per training
   sentence by r and continue SFT. This is stable and cheap.
3. **diffu-GRPO** (d1, 2504.12216):
   - group size G = 8 per prompt;
   - mean-field per-token log-probabilities from one forward pass with a randomly
     masked prompt;
   - KL to the SFT policy;
   - LoRA on the LM so it fits one 48 GB GPU that is shared with other jobs.

**Anti-hacking rules**
- B2 (the evaluator) ≠ B1 (the reward): different architecture, seed and
  training-data subset.
- Also report SiCLIP and human comprehension. A gain is only claimed if B2,
  SiCLIP and DTW-PA-hands do not degrade.
- We expect the reward to raise BT on B1 a lot. The scientific claim rests on B2
  and on humans.

**Novelty.** No RL/DPO/GRPO for sign *production* was found, and RL exists only for
SLT and generic T2M ([01](01_literature_review.md) §3). This is the most novel
component.

---

## Novelty search log (2026-11-03)

| Claim | Searches | Closest hits | Verdict |
|---|---|---|---|
| C3: first RL / preference optimisation for sign *production* | SLP survey agent; "sign language production reinforcement learning back-translation reward GRPO DPO 2026" (extended) | RVLF (2512.07273) and Attention-Steered VLM SLT (2608.00235): GRPO for **translation**. SignLLM "RL loss" is a sampling heuristic. M3T uses a supervised auxiliary SLT loss. UniMo (2601.12126) uses GRPO for generic T2M | not found for production → novel, to be re-checked before submission |
| C1: LM sign tokens → continuous flow refiner on SMPL-X | "sign language generation discrete tokens continuous refinement flow matching SMPL-X quantization error" (extended) | SOKE, SignMask, M3T (discrete); SignSparK, NSA (continuous); HybridSign (2-D, AR + flow); DisCoRD (generic T2M) | combination not found on SMPL-X SLP. The quantization floor has not been measured in any SLP paper |
| C0: calibrated protocol (trivial baselines, tokenizer floor) | SLP survey §C.3 | SignGPT / BLANC / SIGNER re-evaluate baselines; 2510.07453 meta-evaluates metrics; nobody reports trivial-output floors for DTW-JPE | novel as a measurement; low risk |
| C2: train-split lexicon induction for masked diffusion | SLP survey (SOKE, SignRR, Spoken2Sign, SignSparK SR) | all use external dictionaries or glosses | medium novelty; main value is removing dictionary dependence |

## Expected results and decision criteria

We use Phoenix-2014T first, because its poses are local; CSL-Daily and How2Sign
follow. Numbers are hypotheses that will be measured, not claims.

| Stage | Expected effect (DTW-JPE body/hands, Phoenix) | Go / no-go criterion on val |
|---|---|---|
| MaDiS released (reproduced on dev) | ≈ 5.0 / 6.6 (test-reported 4.97 / 6.59) | reproduction within 5% of the paper on test-like numbers; otherwise investigate before building on it |
| + C2-inference (all-part confidence, remask, CFG) | small; mainly SiBLEU / SiCLIP | at least one metric +, none −, versus 3 seeds |
| + C1 refiner | hands toward or below the tokenizer floor (≈4.8), body −0.5 or more | hands DTW below MaDiS by ≥10%, and SiBLEU unchanged by design (tokens are identical) |
| + C2 lexicon | SiCLIP / BT +, PA-hands − | gains over C1 on ≥2 of 3 semantic metrics |
| + C3 RAFT → GRPO | BT (B2) +, SiCLIP + | B2-BLEU +≥10% relative, with no kinematic regression |
| target | beat SignSparK-SR on Phoenix and CSL hands **without glosses**; beat M3T and SeRV on How2Sign | – |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| MaDiS released checkpoint does not reproduce | Fall back to our own SFT from `pretrained.ckpt`; report both |
| Refiner over-smooths back toward the mean (the classic SLP failure) | The flow, not a regression, models the residual distribution. Monitor hand-velocity statistics against GT and the mean-pose floor |
| DTW-warped self-generated targets are noisy | Ablate (a) / (b) / (c); keep (b) only if (c) hurts |
| Reward hacking of B1 | Keep B1 ≠ B2; add a kinematic reward; use KL; prefer RAFT if GRPO destabilises |
| Shared GPU (~20 GB free) | LoRA, bf16, gradient checkpointing; refiner and BT models stay under 100 M parameters; RL on Phoenix first (7k sentences) |
| Lexicon induction too noisy for How2Sign | Use ASL-Citizen / WLASL dictionary signs (local, need SMPL-X fitting) as an external-dictionary variant, flagged † like SOKE |
| Pseudo-GT SMPL-X noise caps metrics | Report PA metrics and the tokenizer floor. A refiner trained to match noisy GT reproduces the jitter, so evaluate jitter separately |

# Devil's-advocate review: Proposal 7 (SignNMFlow critique + "beat SignNMFlow" plan)

Reviewer stance: adversarial. I looked for what is wrong, overclaimed or missing, not for what is good.
Scope read: `docs/proposal7/{README,01,02,03,04}.md`, `probes/*.py|json`, `methods/signmflow/*`,
`src/slp/evaluation/{metrics,smplx_fk}.py`, the extracted paper text, and `docs/experiment_protocol.md`.
I did not read other methods or other `docs/proposal*` folders.

Independent CPU checks (val split, plus Phoenix train as a control; no GPU, no test). Scripts and outputs:

- `/tmp/claude-1007/-home-haipd-SLP/71f22ddb-3d97-4bdf-a372-fc6159e66df9/scratchpad/da_check.py`
  writes `da_check_{phoenix_smplx_val,csl_smplx_val,phoenix_smplx_train}.json` (600 random frames each).
  It measures per-sample/per-joint progress κ, the distance to the geodesic *curve*, Δlog m at equal
  progress, and per-part and per-sample shift fits.
- `/tmp/claude-1007/-home-haipd-SLP/71f22ddb-3d97-4bdf-a372-fc6159e66df9/scratchpad/da_energy.py`
  writes `da_energy_{phoenix,csl}_smplx.json`. It computes the lower bound on energy over all time
  reparametrisations of the geodesic, and the energy of SD3-shifted geodesics.

Severity: **CRITICAL** = invalidates a headline claim or the plan's ability to reach a conclusion.
**MAJOR** = wrong or unsupported as stated, or a serious design flaw. **MINOR** = fix but not fatal.

---

## 0. Verdict in five lines

1. Claim (c), "NG ≈ a time reparametrisation of the geodesic", is right, and stronger than the draft shows.
   - The best pure time reparametrisation of the geodesic reaches energy 23.66 (Phoenix) and 23.89 (CSL).
   - The learned ε reaches 23.76 and 23.75.
   - So the learned correction finds no energy saving from bending the path. This should be the headline evidence instead of cos(ε, ξ).
2. Two supporting "measurements" are artefacts of comparing points at the same t:
   - the "0.28 rad residual spatial deviation";
   - "Δlog m ≈ −0.6: the path goes through lower density".
   At equal progress the deviation is 0.07–0.08 rad and Δlog m ≈ −0.01. **CRITICAL** for the text, because both appear in the summaries.
3. "s ≈ 2.75" depends on the fitting method:
   - a direct fit on κ gives 2.55 (per body part 2.5–2.8; per sample p10–p90 1.85–3.15);
   - the energy-optimal SD3 shift lies between 2.0 and 2.75.
   - E1 tests one point estimate as if it were exact.
4. The plan cannot currently tell a winner from noise or from metric gaming:
   - a static mean pose ties or beats the model on DTW;
   - B-T is saturated on Phoenix (GT ceiling 11.49 vs model 11.34) and broken on CSL (model 6.00 > GT 4.74);
   - the noise floor will come from 1 extra seed at a different training length;
   - there is no positive control showing that NG > GE even exists in the replica.
   **CRITICAL.**
5. Several plan items break repo rules or the protocol:
   - DTW computed inside `fit` for checkpoint selection;
   - fewer than 3 seeds for CSL and How2Sign;
   - unequal tuning budget for the baseline;
   - the [Đo] metric-probe claims cite a JSON file that does not exist.

---

## 1. Central claims, checked

### 1(a) "Energy = Σ_k π_k⟨δ_k,v⟩² + ρ‖v‖²/m, so m(x) drops out"

- **Math: correct.** It follows from Eq. 8/42: (1/m)(ρ|v|² + Σ w̃_k⟨δ_k,v⟩²) with π_k = w̃_k/m.
  - P1 confirms that the density term is ~1e−5 of the energy (p95 1e−4).
  - m ≤ Σ w_k = 1, so ρ/m ≥ 1e−6. With log10 m ≥ −1.6 along the paths, ρ would need to be ≳1e−2 before the density term matters.
  - Suggestion: state the threshold ("density is inert for ρ ≲ 1e−3"). That is more useful than "never matters".
- **MAJOR (fairness): the paper itself writes this.**
  - App. C.3, Eq. 43–44: G(x) = (Σ(x) + (ρ/m) I)⁻¹, and the text says "transit through sparse or dispersed regions is expensive".
  - So this is not a hidden error. It is a *misleading interpretation* in the paper's prose: "enters only through the prefactor 1/m", "reweights … by the inverse motion density".
  - Rewrite to: "under the published ρ = 1e−6, the paper's own Eq. 44 shows that the cost depends on the normalised second moment Σ(x), not on m(x). The 'inverse-density' reading in C.3 and the density-consistency proof therefore do not describe the optimised object."
  - Do not write "the paper's density interpretation is wrong" without that qualifier. The authors will point to Eq. 44.
- **MINOR.** "Density drops out" does not mean "the data drops out". π_k still uses the frequency weights w_k (relative density), and δ_k uses prototype locations. Say "absolute mass m drops out; only relative prototype weights remain".

### 1(b) "Direction to x1 is near the most expensive eigen-direction of G⁻¹"

- **Probe code: essentially correct.**
  - `P M P` with M = S + (ρ/m)I has eigenvalue 0 in the normal direction and eig(S) + ρ/m in the tangent directions, so dropping the smallest eigenvalue is valid.
  - **MINOR numerics:** in float32, eigvalsh has absolute error of about 1e−7·‖S‖, up to about 1e−6. That is comparable to ρ/m (1e−6 to 4e−5) whenever S has tangent rank < 3. λ_min and the reported "log10 condition number" (jaw p95 4.7) are therefore unreliable. The position statistic is robust because λ_max ≫ 1e−6. Fix: use float64, or drop the eigenvector most aligned with x_t instead of the smallest eigenvalue.
- **MAJOR (interpretation): close to a tautology, and over-generalised.**
  - Far from the data, every δ_k points roughly toward the data cluster, so Σ ≈ d²uuᵀ with u the direction to the data. x1 lies in the data, so "toward x1" ≈ u for *any* metric built from a normalised second moment around x.
  - The statistic falls to median 0.60–0.79 (hands, body) for t > 0.7. Yet the summary says "đi về phía mục tiêu là hướng đắt nhất" without qualification. Restrict the claim to the noise end.
- **MAJOR: "opposite of MFM/Mahalanobis" is half right, and the conclusion drawn from it is wrong.**
  - The anisotropy is indeed inverted relative to LAND: MFM's h is the diagonal of the *unnormalised* S around x, and MFM uses G = (diag h + ε)⁻¹.
  - But lingering in expensive regions is a generic property of *energy* minimisation under any position-dependent cost. With cost a(κ)·κ'², the optimum is κ' ∝ 1/a. An MFM/LAND metric is also expensive far from data, so it would also slow the path down near the source.
  - So "the paper's metric points the wrong way, hence the time shift" is not established. The time shift would likely appear under the "right-way" metric too.
  - This weakens the motivation for E6. "Metric đúng chiều" presupposes a correct direction. In a target-prediction model the metric only matters through the induced x_t distribution (see §2.1), and no direction is "correct".

### 1(c) "ε ≈ anti-parallel to ξ, so learned path ≈ geodesic with an SD3 shift s ≈ 2.75"

What holds (my checks):

| Phoenix val / CSL val / Phoenix train | value |
|---|---|
| per-sample κ = ⟨Log_{x0}x_t, ξ⟩/‖ξ‖², median at t = .1/.3/.5/.7/.9 | .053/.170/.305/.476/.730 (identical on CSL and on train) |
| κ monotone in t (per sample, per joint) | 99.9% |
| distance from x_t^NG to the geodesic **curve** (min over s), median at t = 0.5/0.7 | 0.067/0.078 rad (p95 0.18/0.21) |
| same-t distance (what P3 calls "max deviation") at t = 0.5/0.7 | 0.24/0.28 rad |
| energy: geodesic / learned / best reparam (shared per frame) / best reparam (per joint) | Phoenix 30.53 / 23.76 / 23.92 / 23.66; CSL 30.84 / 23.75 / 24.16 / 23.89 |
| energy of SD3-shifted geodesic, s = 1.5/2/2.75/3.5/5 (Phoenix) | 26.14/24.76/24.76/25.81/29.17 |

- **Strength the draft under-uses.** The learned net *can* bend the path, but it reaches the reparametrisation lower bound and not below it. So under the paper's objective, bending buys essentially nothing; this is the optimum of Eq. 10 within the family, not an accident of our training. It is the most defensible form of the claim, and it is less exposed to "your re-implementation is wrong". Put it in `01` §A.3 and in the README.
- **CRITICAL (text): the "0.28 rad residual spatial deviation" is mostly *time lag*.**
  - `probe_theory.py` P3 computes `geodesic_distance(pg(t), pn(t))` at the same t. On a reparametrised geodesic this is non-zero even with zero spatial deviation.
  - The true deviation from the curve is about 3.5× smaller (0.07–0.08 rad median quaternion arc, i.e. about 8–9° of rotation).
  - `04` E1-b reads a negative E1 result as "the 0.28 rad spatial deviation has value". That decision rule rests on a misread statistic.
- **CRITICAL (text): "Δlog m ≈ −0.6 at t = 0.5: the NG path passes through lower density, contrary to Fig. 1" is an artefact.**
  - The lagging path is closer to the noise at equal t, so its density is trivially lower.
  - At equal progress (nearest point on the geodesic), Δlog m has median −0.004 to −0.017 (p5 −0.26, p95 +0.03) for all t, on both datasets.
  - Correct statement: "the learned path visits the same points as the geodesic (it does not bend toward high density, contrary to Fig. 1) and spends more time near the source". The contradiction with Fig. 1 survives; the "lower density" mechanism does not.
  - The "mean log m over path" numbers are time-weighted for the same reason.
- **MAJOR: "s ≈ 2.75 with RMSE 0.02" is fit-method-dependent and hides heterogeneity.**
  - P5b fits the SD3 curve to the *ratio of mean rotation angles* and assumes the geodesic's angle-to-x1 is linear in t. It is not: 65.3/126.5 = 0.517 at t = 0.5. That systematic bias (~0.017) is as large as the reported RMSE.
  - Fitting κ directly gives s = 2.55 (CSL 2.54; train 2.57).
  - Per part: body 2.63, hands 2.50–2.55, jaw 2.83.
  - Per sample and joint: p10/p50/p90 = 1.85/2.55/3.15. Global s = 2.75 leaves a per-sample κ RMSE of 0.036 (p90 0.076).
  - κ also grows slightly with ‖ξ‖ (terciles 0.298/0.299/0.311 at t = 0.5).
  - In energy terms, any s in about [2, 2.75] is equivalent (24.76 for both).
  - Fix: report s = 2.5 ± 0.3 with the per-sample spread. In E1, run a small s grid (for example {2, 2.5, 3}) rather than a single point. The inference-grid half of the grid costs nothing (see §2.2).
- **MINOR.** `probe_parallel.py` uses `arr[:2000]`, i.e. consecutive frames from the first ~15 dev sequences, not a random sample. My random-frame κ medians reproduce its `implied_kappa` to 3 decimals, so no harm done, but fix the sampling.
- **MINOR.** All probe results come from one correction seed per dataset. Phoenix and CSL agreeing (2.55 vs 2.54) suggests the warp is driven by p0 and the metric structure rather than by the data. That is itself a finding worth stating, and it predicts that σ changes s (§2.3).
- **Val vs train is a non-issue.** The train split gives the same numbers to 3 decimals.

---

## 2. Alternative explanations and overclaims

### 2.1 MAJOR: the draft misses the cleanest argument for H1, and its loophole

- In `flow.flow_loss` with `prediction=target`, the loss is ‖x̂1 − x1‖². The path velocity never enters the generator loss.
- So the path affects the generator only through (i) the distribution of (x_t, t) seen in training and (ii) the states visited by the re-planned sampler.
- A time-reparametrised geodesic is therefore *exactly* equivalent to "geodesic + non-uniform t density + relabelled t + shifted inference grid".
- The equivalence fails only through:
  - the ~0.07 rad perpendicular part;
  - **per-sample/per-joint heterogeneity of κ**: different joints of one frame sit at different effective noise levels; s ranges 1.85–3.15 across samples.
- That heterogeneity is a third candidate mechanism, distinct from the "schedule" and "spatial bend" explanations. It acts like per-token noise-level augmentation (diffusion-forcing style).
- With a single global s, E1-b removes both residuals at once. If E1-b < NG, the plan cannot tell which residual mattered.
- **Fix:** add an "NG-parallel" arm, x_t = Exp_{x0}(κ_θ(t, x0, x1) ξ), where κ_θ is the learned per-joint progress (projection of the learned ε). It reuses the existing correction net and needs no new stage. Then:
  - NG vs NG-parallel isolates the bend;
  - NG-parallel vs geoshift isolates the heterogeneity.

### 2.2 MAJOR: zero-training tests of half of H1 are missing

- With target prediction, any trained model can be sampled on any path or grid. So before spending 9 GPU-hours:
  - (i) sample the **existing NG-trained v2 model** with the re-planned sampler on a geoshift(s) path, mapping t ↦ κ_s(t) and feeding the network t = κ_s⁻¹. This tests whether the network relies on the bend or heterogeneity at inference.
  - (ii) Sample it on NG vs geoshift grids.
  - (iii) If a GE-trained checkpoint exists or is trained in E1-a, sample it with shifted grids.
- Together these split H1 into an "inference grid" part and a "training density" part. Only the latter needs training.

### 2.3 MAJOR: the paper's σ ablation is an unused falsification test

- Table 4: σ = 0.25 gives 19.41, *worse than GE (18.31)*; σ = 0.75 gives 17.58.
- If NG is a time shift, each σ must induce some s(σ), and the metric-vs-s curve must reproduce the ordering (σ = 0.5 best, σ = 0.25 worse than s = 1).
- Training ε for σ ∈ {0.25, 0.75} and fitting s is cheap: correction stage only, CPU-feasible at reduced epochs.
- If s(0.25) ≈ s(0.5) while the paper's numbers differ greatly, the time-shift story cannot explain the paper's own ablation.
- Run this check before E1; it is the cheapest test that could falsify H1.

### 2.4 MAJOR: an unmentioned alternative: NG "wins" DTW by regressing more to the mean

- With NG, the last network call (nominal t = 0.8) sees a state at effective noise ≈ 0.41, versus 0.2 for GE. The output x̂1 is therefore closer to E[x1 | text].
- The replica already shows that DTW rewards this:
  - v1: NFE 1 < NFE 5 < NFE 20 in error;
  - static mean pose: 4.70 vs the model's 5.36 JPE body.
- So part of the paper's NG gain (and any E1-d gain at s ∈ {4, 6}) may be metric gaming, not better generation.
- H2 ("push further toward noise lowers DTW-PA hand") is predicted by *both* "better schedule" and "more mean regression". As designed, the plan cannot tell them apart. See §3.3 for guards that can.

### 2.5 Overclaims to soften or remove

- **MAJOR.** "NG ≡ time-shift" / "replace K-means + ε by a scalar s" (`01` C.1, `02` §3–4, README) is stated before E1.
  - What is shown: NG's *path* is ≈ a heterogeneous time warp, and the objective gains nothing from bending.
  - What is not shown: equal *generation quality*. Use "≈" and "H1" consistently.
- **MAJOR.** "Phương pháp này gần như trùng MFM" (README).
  - Overlap: the t(1−t)φ interpolant and training by kinetic energy under a data metric. That is a legitimate missing citation.
  - Differences: the metric is inverted (normalised S vs inverse unnormalised variance); the setting is a manifold; the generator uses target prediction.
  - Write "uses MFM's interpolant and training scheme without citing it", not "nearly identical".
- **MAJOR.** Summary item 6 says GE's effect is "not separable from the noise schedule". The draft's own P5 contradicts this for the MR→GE step:
  - linear-quat-renorm and geodesic have nearly the same angular schedule (65.3° vs 65.3° at t = 0.5; 35° vs 39° at t = 0.7);
  - the paper's biggest gain is MR→GE (21.25 → 18.31).
  - So the schedule confound applies to the 6D-Gaussian row and to NG, not to GE.
  - For GE, the obvious confound is the *off-manifold norm* of linear x_t, which `flow_loss` does not renormalise for `path=linear`, not the schedule.
- **MINOR.** Table 8 (16.86 → 16.98, +0.7%, single run) is cited as a "sign of mean regression". It is not evidence; rely on the replica's v1 NFE sweep.
- **MINOR.** "Half the time budget is spent in a data-free region" (p0 at 126°). Unquantified: at t = 0.5 the geodesic is at 65°, while hand data has a p95 angle of 74–84°. Either quantify it (the fraction of t with angle-to-x1 above the data p95) or drop it.
- **MAJOR (provenance).** Summary item 5 and C.2 are tagged [Đo] and cite `probes/probe_metrics_phoenix_smplx.json` and `README.md#kết-quả-probe-metric`. The JSON does not exist yet (the probe is still running at 1240% CPU), and the README section is a placeholder `<!-- PROBE-METRICS -->`. Until it is filled, relabel these items [Suy luận] or cite the v1 README table on 100 sequences.
- **MINOR.** On "DTW: shorter predictions are hardly penalised" (`01` B.2, `02` §9): the warping path length is ≥ max(T_pred, T_gt), so every GT frame is still paid for. The precise statement is "accumulated cost has no per-frame normalisation and its minimum path length is max(T_pred, T_gt), so over-long predictions add cost and short ones do not reduce the GT-length floor".

---

## 3. Experiment plan

### 3.1 CRITICAL: no positive control. H1 can be "confirmed" by a null.

- E1's success criterion is "|E1-b − NG| < noise threshold".
- If NG ≈ GE in the replica (never tested: Table 3 has not been reproduced), then E1-b ≈ NG is guaranteed and says nothing.
- Equivalence needs: (i) a pre-specified equivalence margin, and (ii) a demonstrated NG − GE difference larger than that margin on the *same* protocol.
- The paper's NG claim is on **CSL**; E1 screens on **Phoenix** only.
- **Fix:** make E1-a (GE) vs E1-c (NG) the first, gating comparison, with ≥ 2 seeds each.
  - If NG − GE is not significant, report "NG has no measurable effect in a faithful replica". That is itself the main scientific result.
  - Then drop E1-b/E1-d as a test of the paper and keep s only as a tuning knob.
  - Run the gate on CSL if Phoenix is inconclusive.

### 3.2 CRITICAL: the metrics and guard rails have almost no power

- **DTW.** A static mean pose beats the model on JPE body and ties on PA hand (v1, 100 sequences: 4.70 vs 5.36; 1.35 vs 1.37). Any intervention that increases mean regression "wins". FK loss, larger s, an informative prior, EMA and early stopping all plausibly do.
- **B-T.**
  - Phoenix: v2 scores 11.34 against a GT ceiling of 11.49, so the metric is saturated and cannot register improvements.
  - CSL: 6.00 against a GT ceiling of 4.74, so the metric is broken; a higher score can mean *worse* (more regularised) motion.
  - Consequences: CSL confirmation "with B-T guard" is meaningless, and E4 (CFG, whose only expected benefit is B-T) cannot win by the plan's own rule.
- **Speed-ratio ≥ 0.8** is arbitrary and gameable: jitter raises speed. In v1, NFE 20 raised jitter to GT level while error rose.
- **Fix (prerequisite, CPU or cheap GPU):**
  1. Train a text–motion retrieval evaluator (contrastive, on train only) and report R-precision and a distributional score (FID-style) on dev. Calibrate it: GT ≫ shuffled-GT ≫ mean pose.
  2. Replace the speed threshold with two-sided bands on velocity *and* jerk/acceleration spectra relative to GT, for example within ±20%.
  3. Report the gap to the mean pose per column as a headline number.
  4. Pre-register a composite decision rule. Example: no significant regression on retrieval and the motion-statistics bands, then improvement on DTW-PA hand.

### 3.3 MAJOR: statistical design

- **Noise floor.** One extra 300-epoch seed gives SD_seed with 1 degree of freedom, at a training length not used for screening. "Δ > 2·SD" is then nearly arbitrary.
  - Fix: baseline × 3 seeds at the *screening* length (~27 GPU-h).
  - Use paired, per-sequence bootstrap (same dev sequences, same sampler seeds) for every comparison. It is much more powerful than seed-SD thresholds at n = 519.
- **Multiplicity.** About 15 arms, forward selection, 3 sampler seeds, NFE variants and 6 snapshots per run. Expect several false "wins" at 2σ. Pre-register primary comparisons; treat the rest as exploratory.
- **Early-selection leakage.** The checkpoint is chosen on the first 200 dev sequences and reported on all 519, which include those 200. Max-of-6 snapshot selection also favours noisy arms (winner's curse).
  - Fix: select on a fixed dev subset and report only on the disjoint remainder. Or use the same fixed epoch for all arms. Or hold out a train slice for selection.
- **Repo rule violation.** `02` §7 proposes "DTW on 200 val sequences every 25 epochs" inside `training.validation_loss`. CLAUDE.md: "Never compute reported metrics inside a method." Do selection outside the method: evaluate `generator_ep*.pt` snapshots via `scripts/run.py` with `init.generator`, and select in an analysis script.
- **Protocol §4.** ≥3 seeds are required; the plan has CSL 2, How2Sign 1.
- **Protocol §5.** Fair tuning budget: SignMFlow+ tunes s, w, λ, λ_v and σ0, while v2 gets nothing. At minimum, give the baseline the same checkpoint selection, NFE/grid tuning and EMA (these are hygiene, not method contributions), and report their search spaces.

### 3.4 MAJOR: 150-epoch screening can flip rankings

- Val loss is minimal at epoch 100 of a 300-epoch cosine schedule. A 150-epoch cosine run ends at a different point in the overfitting regime than epoch 150 of a 300-epoch run.
- Several interventions are *regularisers*, so their benefit is length-dependent: EMA, text lr, CFG dropout, FK loss, and changes of s that shift effort toward easy high-noise states. Rankings between them and the baseline can flip at 300 epochs.
- **Fix:**
  - choose the screening length after E0.1 shows where *DTW and retrieval* (not flow loss) peak;
  - run the baseline at the same length;
  - re-confirm the top 2 at the full length before composing.

### 3.5 MAJOR: interactions. Several "different" experiments turn the same knob.

- For x-prediction with x-loss, these all re-allocate training effort or inference states across effective noise levels:
  - geoshift s (E1-b, E1-d);
  - logit-normal t (E1-d);
  - w(t) = 1/(1−t)² (E3c);
  - the informative prior p0 (E5, which changes what "noise level" means);
  - the sampler grid and NFE (E0.2);
  - self-conditioning (E7).
- Training with path κ_s and uniform t equals training on the geodesic with t ~ κ_s#U. Loss weighting w(t) multiplies that density (up to gradient variance). Running them as separate sequential arms wastes runs, and forward selection will double-count.
- **Fix:** parametrise two orthogonal knobs:
  - p_train(effective noise level);
  - the inference grid.
- Run a small factorial, e.g. train-shift ∈ {1, 2.5, 4} × inference-shift ∈ {1, 2.5, 4}. That is 3 trainings and 9 cells, with the inference axis free.
- Re-tune s after any p0 change.
- Drop E3c as a separate arm; it is a p_train variant.

### 3.6 MAJOR: output length is an uncontrolled confound

- DTW is accumulated, and length comes from a head on detached text features. Every arm that changes the text encoder's trajectory changes predicted lengths, and therefore DTW, independently of motion quality: text lr, EMA, CFG null-text dropout, and even the seed.
- **Fix:**
  - freeze one shared length predictor across all arms, for example the v2 length predictions saved to a file and loaded at predict time;
  - report `dtw_jpe_norm` alongside;
  - with CFG dropout, mask the length loss on dropped samples.

### 3.7 MAJOR: feasibility of the FK loss (E2-a) is under-estimated

- `src/slp/evaluation/smplx_fk.py` builds **all 21 hand joints** by applying `orig_hand_regressor` / J_regressor rows to the **posed mesh vertices**, not from skeleton joints.
- A "skeleton-only FK (+ ~5 fingertip vertices)" therefore will not match the metric to 1e−5 m. The proposed unit test will fail by construction.
- Matching requires sparse LBS **with pose blendshapes (posedirs)** over the union of the vertex supports of those regressor rows. This is feasible but is a mesh computation, not "~150 lines".
- **MAJOR (design).** Hand JPE subtracts the wrist *position* but keeps the global wrist *orientation*. So hand JPE depends on the whole chain (spine → shoulder → elbow → wrist). A hand-local FK loss optimises the wrong quantity for JPE hand; only PA hand is rotation-invariant. Compute positions through the full chain and subtract the wrist afterwards.
- **Risk.** With x-prediction at high noise, an MSE loss in *position* space makes x̂1 the conditional mean in position space, which means more mean regression. The guard rails of §3.2 are mandatory for E2.
- **Fix:**
  - run E2-c (lever-arm weighting) first as planned;
  - for E2-a, first measure the correlation of a cheap proxy (skeleton finger joints) with the metric on GT perturbations before writing the full LBS;
  - set the test tolerance against the real metric FK.

### 3.8 MINOR to MAJOR: CFG on target prediction (E4)

- The geometry is fine: Exp_{x̂u}(w Log_{x̂u} x̂c) stays on S³. CFG on x̂1 with the re-planned sampler is equivalent to CFG on the implied velocity.
- But it doubles NFE cost (10 network calls at "NFE 5"). Report that, or compare at equal compute.
- **Expected direction:** CFG moves outputs away from the conditional mean, so DTW gets *worse* by construction, and the only metric that could reward it is broken (§3.2).
- Do not schedule E4 before a working text-faithfulness evaluator exists.
- Null-text dropout also perturbs the jointly fine-tuned mBART, which already collapses early at lr 2e-4. Run it only on the E3 base (text lr 2e-5).

### 3.9 Priority and budget

- **MAJOR: the budget ignores evaluation cost.** The SOKE-protocol DTW with pairwise Procrustes is CPU-heavy: `probe_metrics.py` on 150 sequences has been running at about 12 cores for 20+ min on a box with load average around 50.
  - The plan implies about 20 runs × 6 snapshots × 3 sampler seeds × NFE variants × 519–1077 sequences.
  - Budget CPU-hours explicitly. Evaluate snapshots on a subset; run full dev only for finalists.
- **MAJOR: the start date is unknown.** The How2Sign v2 job must finish first, with 1 process at a time on a shared GPU. "~12 days" assumes zero failures and no coding time. A realistic estimate is 3+ weeks.
- **Recommended order:**
  1. Free work:
     - the σ falsification check (§2.3, correction stage only);
     - the evaluator and guard rails (§3.2);
     - frozen lengths (§3.6);
     - inference-only tests on existing checkpoints (E0.1, E0.2, E0.4, §2.2);
     - fill in the metric probe.
  2. Baseline × 3 seeds at the screening length (the noise floor).
  3. Gate: GE vs NG (≥ 2 seeds). Only if significant, add NG-parallel and geoshift s ∈ {2, 2.5, 3}.
  4. E3 hygiene as the new base, also applied to the baseline.
  5. Noise-allocation factorial (§3.5).
  6. E2-c, then E2-a.
  7. E4 only with the evaluator.
  8. Leave E5–E9 for later.

---

## 4. What most likely makes the whole plan fail

**Most likely failure.** Every arm lands within noise on DTW. Some arms "win" by more mean regression, and the guard rails (saturated or broken B-T, a gameable speed ratio) do not catch it. E1 "confirms" H1 because neither NG nor geoshift differs from GE. Forward selection then composes noise into "SignMFlow+", which cannot survive the 3-seed test, or survives for the wrong reason.

A second, independent failure is the framing. The replica is about 0.5–1.0 behind the paper on Phoenix test DTW (4.93 vs 4.37 body; 6.54 vs 5.57 hand). No planned change is likely to close a gap of that size. "Beating SignNMFlow" can only honestly mean beating the replica; say so up front.

### Top-3 highest-value changes

1. **Make the evaluation able to decide before training anything.**
   - Add a calibrated text–motion retrieval evaluator (GT ≫ shuffled ≫ mean pose), two-sided motion-statistics bands and a frozen shared length predictor.
   - Use a 3-seed baseline at the screening length with paired per-sequence bootstrap.
   - Select checkpoints on a disjoint subset and outside the method.
   - Without this, no result in Stages 1–5 is interpretable.
2. **Turn E1 into a falsification test with a positive control:**
   - (i) the free σ check (does s(σ) reproduce Table 4's ordering?);
   - (ii) free inference-only swaps on the trained NG model (§2.2);
   - (iii) a GE vs NG gate with ≥ 2 seeds;
   - (iv) only then NG-parallel vs geoshift on a small s grid.
   - This separates bend, heterogeneity, training density and inference grid, and it makes "NG has no effect" a reportable outcome.
3. **Fix the text before it circulates.**
   - Replace "0.28 rad spatial deviation" and "Δlog m −0.6 / lower density" with the equal-progress statistics (0.07–0.08 rad; Δlog m ≈ −0.01).
   - Lead with the energy result: learned 23.76 ≈ best-reparam bound 23.66–23.92.
   - Report s as 2.5 ± 0.3 with its spread.
   - Credit Eq. 44 when criticising the density reading; restrict the "most expensive direction" claim to t < 0.3.
   - Retract "GE is schedule-confounded".
   - Downgrade "nearly identical to MFM" and the uncited [Đo] metric-probe items.

### Reproduction of my checks

```bash
cd /home/haipd/SLP
S=/tmp/claude-1007/-home-haipd-SLP/71f22ddb-3d97-4bdf-a372-fc6159e66df9/scratchpad
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src python $S/da_check.py phoenix_smplx val 600   # also csl_smplx val, phoenix_smplx train
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src python $S/da_energy.py phoenix_smplx            # also csl_smplx
```

Note: keep `OMP_NUM_THREADS` low. The machine is shared (load average around 50), and unthrottled runs take tens of minutes.

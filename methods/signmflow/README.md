# SignNMFlow — Naturalness-guided Manifold Flow Matching for SLP
status: exploring

Re-implementation of He et al., *Naturalness-Guided Manifold Flow Matching for Sign Language
Production*, arXiv:2609.33339 (paper: `docs/2609.33339v1.pdf`). The authors did not release
code. Everything below follows the paper; where the paper says nothing, the choice is listed
under **Unspecified in the paper** and can be changed in `config.yaml`.

## Paper → code

| Paper | Code |
|---|---|
| Sec. 3.1, Eq. 5: x ∈ (S³)^J, J = 41 (10 upper body + 15 + 15 hands + jaw), physical SMPL-X rotations | `representation.py` (`MotionCodec`) |
| App. C.1, Eq. 37: sign convention for quaternions | `geometry.sign_convention` |
| Eq. 6 / 47: Exp/Log on S³, tangent projection (Eq. 48) | `geometry.expmap/logmap/project` |
| App. C.2, Eq. 38–39: K-means per joint, renormalised centroids, frequency weights | `naturalness.extract_prototypes` |
| Sec. 3.2, Eq. 7–8: kernel weights, scatter S(x), mass m(x), G(x) = m(S+ρI)⁻¹ | `naturalness.NaturalnessMeasure` (uses the closed form Eq. 42, in log space) |
| Eq. 11: x_t = Exp_{x0}(t Log_{x0}x1 + t(1−t) P_{x0} ε_θ(t,x0,x1)) | `paths.path_point` |
| Alg. 1 / Eq. 12: train ε_θ by the kinetic energy; ẋ by autodiff | `paths.path_and_velocity` (forward-mode `jvp`), `training.train_correction` |
| Alg. 2 phase 1 / Eq. 13: target prediction, per-joint radial projection, loss 10·pose + 0.1·expr | `flow.flow_loss`, `training.train_generator` |
| Alg. 2 phase 2 / Eq. 53: 5-step re-planned sampler; expression e_{t+Δ} = e_t + Δ/(1−t)(ê₁ − e_t) | `flow.sample` |
| Sec. 4.1 / Table 6: ϕ = 8-layer transformer decoder (512, FF 2048, 8 heads, dropout 0.1, time emb 2048), cross-attention to mBART-large-cc25, Stylization blocks after self-attn, cross-attn and FF | `model.py` |
| Table 6: ε_θ = 3-layer MLP (hidden 512), batch 8, 100 epochs; ϕ batch 64/GPU × 4 GPUs, 300 epochs; AdamW, lr 2e-4 cosine, wd 0, fp32; K = 500, σ = 0.5, ρ = 1e-6; p0 uniform on the sign hemisphere, e0 ~ N(0, I); NFE 5 | `config.yaml` |
| SignNMFlow-M: trained on Phoenix + CSL + How2Sign, mBART source-language token per sample | `method.multilingual=true` |
| Sec. 4.1 metrics: DTW-JPE and DTW-PA-JPE (SOKE protocol) | shared: `src/slp/evaluation/metrics.py` (`soke_dtw_*`), FK in `src/slp/evaluation/smplx_fk.py` |

Ablation switches (Tables 3 and 5): `method.flow.representation` = `quat|6d`,
`method.flow.path` = `naturalness|geodesic|linear`, `method.flow.prediction` = `target|velocity`.

## Running

**All experiments (recommended): job queue** `jobs.py` — dependency graph of every run of
Tables 1-5, 7, 8 and B-T, at most 1 process at a time by default (hand-started `run.py` processes count
and are adopted), priority order (short dependents → main seed 0 → Tables 3/5 → SignNMFlow-M →
Table 4 → extra seeds 1, 2), GPU-memory gate, OOM → retry with half the generator batch (the
effective batch 256 is kept), and every retry resumes from the previous attempt's
`prototypes.pt` / `correction*.pt` / `generator.pt`. The state is the file system, so the runner
can be stopped (kill its pid) and restarted at any time.

```bash
python -m methods.signmflow.jobs status        # all jobs: done / running / ready / blocked / failed
python -m methods.signmflow.jobs plan          # launch order + resolved commands
nohup python -m methods.signmflow.jobs run > outputs/signmflow/jobs/runner.log 2>&1 &
# default: Tables 1-2, seed 0. Options: --tables 1,2,3,4,5,7,8 --extra-seeds 1,2 --max-procs 1 --min-free-gb 10
```

The CSL-Daily main run is the "full" row of Tables 3/4/5/8 (it is not trained twice).
Per-table shell scripts (sequential, same commands):

```bash
# Full model on one dataset (val split, default):
python scripts/run.py --method signmflow --data csl_smplx --name signmflow_main_csl_smplx --seeds 0,1,2
# Paper tables (scripts document seeds/overrides; SPLIT=test only for final numbers):
bash configs/experiments/signmflow/table1_2_main.sh       # Tables 1, 2 (+ SignNMFlow-M)
bash configs/experiments/signmflow/table3_5_ablation.sh   # Tables 3, 5 (CSL-Daily)
bash configs/experiments/signmflow/table4_measure.sh      # Table 4 (K, sigma)
bash configs/experiments/signmflow/table8_nfe.sh          # Table 8 (NFE)
python -m methods.signmflow.benchmark --ckpt <run>/checkpoints   # Table 7 (#params, latency)
bash configs/experiments/signmflow/bt_score.sh signmflow_main_phoenix_smplx phoenix  # B-T BLEU-4 (repo back-translator)
# Multi-GPU (paper: 4 GPUs x 64): set CUDA devices and launch run.py with torchrun.
```

Each stage checkpoints into `<run>/checkpoints/` (`prototypes.pt`, `correction.pt`,
`correction_final.pt`, `generator.pt`, `generator_ep*.pt`); interrupted stages resume from their
checkpoint when re-run with `method.init.*` or the same checkpoint files. Predictions are stored
as `<run>/predictions.npz` (`{name: [T,133]}` normalised SOKE features), which is the input
format of `scripts/score_predictions.py` and of the back-translation scorer
`scripts/train_backtranslator.py --score_npz` (B-T BLEU-4 column).

Smoke test (~15 min incl. evaluation): `python scripts/run.py --method signmflow --data phoenix_smplx --name signmflow_v2_smoke method.debug_limit=256 method.measure.K=50 method.measure.max_frames=200000 method.correction.epochs=1 method.generator.epochs=2`.

## Unspecified in the paper (our choices)

1. **Output length at inference.** The paper does not say how the number of frames is chosen.
   `predict` only sees text (repo protocol: no reference length at inference), so ϕ has a small
   length head (log-length L1 regression on the mean-pooled, *detached* mBART features; it does
   not change the flow objective). Diagnostic (v1 Phoenix model, 100 dev clips): reference
   lengths would lower DTW-JPE body/hand only from 5.36/7.42 to 5.14/7.24.
2. **"All" column of Tables 3/4/5/8.** SOKE reports body and hand only. We compute DTW over
   all 144 SMPL-X output joints (pelvis-aligned / Procrustes). "Hand" = mean of left and right.
   The paper's All values (e.g. 16.86 vs body 5.73 / hand 7.43 on CSL) are ~3x body, which our
   definition does not reproduce (ours ~1.35x); the paper's definition is unknown. Not used in
   Tables 1-2.
3. **PA-JPE body alignment.** SOKE's released code fits the Procrustes similarity on *all* joints
   and then measures the 12 upper-body joints (`soke_dtw_pa_jpe_body`); with that, PA > JPE for
   body, even for a static mean pose, whereas every published table has PA < JPE. Fitting the
   similarity on the upper-body joints (`soke_dtw_pa_jpe_body_ua`) gives PA < JPE and SOKE-level
   values, so the published numbers were most likely produced that way. Both are reported.
   Hands: similarity / wrist alignment on the 21 regressed hand joints, as in SOKE.
4. **Loss reduction.** Alg. 2 writes 10‖x1 − x̂1‖² + 0.1‖e1 − ê1‖²: the squared norm is summed
   over the 164 pose / 10 expression channels of a frame and averaged over valid frames
   (`flow.loss_reduction: sum`). (v1 averaged over channels, which shifted the pose/expression
   balance by 16.4x towards expression.)
5. **ε_θ input/initialisation.** Frame-wise MLP on [t, x0, x1] (1 + 2·164 inputs, SiLU), last
   layer zero-initialised (training starts at the geodesic; the optimum is unchanged).
6. **Text encoder checkpoint.** SOKE's reduced-vocabulary mBART-large-cc25
   (`third_party/SOKE/deps/mbart-h2s-csl-phoenix`); the encoder only is used, fine-tuned at the
   same lr. Token ids outside the reduced vocabulary map to `<unk>` as in SOKE (0% on Phoenix and
   How2Sign dev, 7% on CSL-Daily). Max 128 tokens.
7. **Batches on one GPU.** The paper uses 64 per GPU on 4 GPUs (DDP: mean of four 64-sample batch
   losses). We load 64-sample batches, process them in `micro_batch` chunks whose losses are
   normalised by the *whole* batch's valid-frame count (gradient identical to the 64-sample
   batch), and accumulate 4 batches (`effective_batch: 256`). With `torchrun` on 4 GPUs it is the
   paper setting. Only the noise draws differ.
8. **Model selection.** None reported; the final epoch is used (val flow loss is logged only).
9. **Hand mean.** SOKE's SMPL-X layer has `flat_hand_mean=False`, so the physical hand rotation is
   `pose + hands_mean`; quaternions encode the physical rotation (`add_hand_mean: true`).
10. **Back-translation.** SOKE's back-translation model is not public (not in the SOKE release);
   B-T BLEU-4 is computed with the repo's mBART back-translator per language
   (`outputs/bt/<src>_mbart`) and is only comparable across our runs (report its GT ceiling).
11. **How2Sign protocol.** The paper follows SOKE's setup, so How2Sign uses
   `how2sign_smplx_soke`: SOKE's 24-fps subsampling and SOKE's 31 failed-fit clips removed
   (the repo default `how2sign_smplx` follows MaDiS).
12. **K-means.** Lloyd iterations with k-means++ initialisation until the relative change of the
   inertia is < 1e-4 (max 300), as in Eq. 38. (v1 stopped after one iteration because of a bug.)
13. **Parameter count.** Ours: 272.4M (mBART encoder 171.7M with the reduced vocabulary, ϕ
   100.2M of which 56.7M are the Stylization blocks as in MotionDiffuse, ε_θ 0.5M). The paper
   reports 233.2M; the paper gives no further detail to locate the 39M difference.

## Results

### v2 (after the 2026-11-16 review) — Phoenix, CSL-Daily, How2Sign done (SignNMFlow-M dropped)

Changes vs v1: K-means convergence fix, sum-reduced loss (Alg. 2), exact 64-sample batches,
SOKE How2Sign protocol. Experiment names `signmflow_v2_*`, driven by `jobs.py`.
Val (dev) split, seed 0, final epoch (300). B-T: repo mBART back-translator (GT ceiling in brackets).

| Phoenix-2014T | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand | B-T |
|---|---|---|---|---|---|---|
| v2 ours (dev, 519) | 5.17 | 6.98 | 5.88 | 4.59 | 1.37 | 11.34 (11.49) |
| v1 ours (dev, 519) | 5.41 | 7.46 | 6.18 | – | 1.42 | 10.17 (11.49) |
| paper SignNMFlow (test) | 4.37 | 5.57 | 3.85 | 3.85 | 1.07 | 16.15 (SOKE BT) |
| paper SOKE (test) | 6.04 | 7.72 | 4.77 | 4.77 | 1.38 | 11.87 (SOKE BT) |

**Test split (seed 0, final checkpoint; logged in results/test_access_log.jsonl)**

| Test | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand | B-T (GT) |
|---|---|---|---|---|---|---|
| How2Sign v2 ours (2316; B-T on 2288) | 7.45 | 10.67 | 8.20 | 6.63 | 2.35 | 1.08 (1.51) |
| How2Sign paper SignNMFlow | 6.76 | 9.72 | 6.01 | 6.01 | 2.07 | 16.88 |
| How2Sign paper SOKE | 7.75 | 10.08 | 6.82 | 6.82 | 2.35 | 14.48 |
| Phoenix v2 ours (642) | 4.93 | 6.54 | 5.56 | 4.35 | 1.30 | 12.81 (–) |
| Phoenix paper SignNMFlow | 4.37 | 5.57 | 3.85 | 3.85 | 1.07 | 16.15 |
| Phoenix paper SOKE | 6.04 | 7.72 | 4.77 | 4.77 | 1.38 | 11.87 |
| CSL v2 ours (1176) | 5.93 | 7.69 | 6.77 | 5.26 | 1.38 | 6.31 (4.74) |
| CSL paper SignNMFlow | 5.73 | 7.43 | 5.10 | 5.10 | 1.30 | 15.14 |
| CSL paper SignNMFlow-M | 5.94 | 7.73 | 5.31 | 5.31 | 1.33 | 15.23 |

CSL test was run with `--allow-dirty` (uncommitted code); Phoenix test at commit bec2088 + diff.

| How2Sign (SOKE protocol) | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand | B-T |
|---|---|---|---|---|---|---|
| v2 ours (dev, 1716) | 7.50 | 10.85 | 8.34 | 6.73 | 2.35 | 0.94 (1.64) |
| paper SignNMFlow (test) | 6.76 | 9.72 | 6.01 | 6.01 | 2.07 | 16.88 (SOKE BT) |
| paper SOKE (test) | 7.75 | 10.08 | 6.82 | 6.82 | 2.35 | 14.48 (SOKE BT) |

How2Sign: 300 epochs (generator resumed at epochs 32 and 45 after pauses), val flow loss flat at
8.60 from epoch 200 on (no overfitting). The repo's How2Sign back-translator reaches only
1.64 BLEU-4 on GT, so its B-T is not informative.

| CSL-Daily | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand | B-T |
|---|---|---|---|---|---|---|
| v2 ours (dev, 1077) | 5.72 | 7.38 | 6.50 | 5.06 | 1.33 | 6.00 (4.74) |
| paper SignNMFlow (test) | 5.73 | 7.43 | 5.10 | 5.10 | 1.30 | 15.14 (SOKE BT) |
| paper SignNMFlow-M (test) | 5.94 | 7.73 | 5.31 | 5.31 | 1.33 | 15.23 (SOKE BT) |

CSL-Daily B-T: the repo's CSL back-translator is weak (GT ceiling 4.74 BLEU-4 on dev), so the
generated motion scoring above GT only says the generator output is a regularised version of GT that this
weak model reads more easily. The number is not comparable to the paper's.

v2 Phoenix: ε_θ val energy geodesic 30.76 → learned 23.94; K-means 39-69 Lloyd iterations per
joint; generator 19.5 h on the shared GPU. Val flow loss (sum-reduced) is lowest at epoch 100
(10.30) and 12.56 at epoch 300 (overfitting; the paper reports no model selection).

Note on short runs: jointly fine-tuning mBART at lr 2e-4 (as the paper and SOKE do) collapses
the text features for the first few epochs (v1 Phoenix: length loss 1.26 → 5.31 at epoch 2, back
to 0.36 at epoch 4; a 5-epoch smoke run ends collapsed: mean pooled cosine between sentences
0.999 vs 0.71 pretrained, all lengths at the 40-frame clamp). Full runs recover (v1 Phoenix final:
0.38). Smoke runs therefore say nothing about quality.

### v1 (superseded: one-iteration K-means bug, mean-reduced loss) — kept for reference

All numbers: **val (dev) split, seed 0, single run** (the paper reports the test split).
B-T = BLEU-4 of the repo's mBART back-translator (`outputs/bt/<src>_mbart`); its ground-truth
ceiling on Phoenix dev is 11.49, so it is not on the same scale as the paper's B-T (SOKE's model).

| Phoenix-2014T | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand | B-T |
|---|---|---|---|---|---|---|
| SignNMFlow, ours (dev, 519 seq) | 5.41 | 7.46 | 6.18 | – | 1.42 | 10.17 (GT 11.49) |
| paper, SignNMFlow (test) | 4.37 | 5.57 | 3.85 | 3.85 | 1.07 | 16.15 |
| paper, SOKE (test) | 6.04 | 7.72 | 4.77 | 4.77 | 1.38 | 11.87 |

Diagnostics on the first 100 dev sequences (same metric code):

| predictions | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand |
|---|---|---|---|---|---|
| ground truth | 0 | 0 | 0 | 0 | 0 |
| SignNMFlow, NFE 5 (default) | 5.36 | 7.42 | 6.12 | 4.83 | 1.37 |
| SignNMFlow, NFE 1 (x̂1 from noise) | 4.89 | 7.06 | 5.54 | – | 1.31 |
| SignNMFlow, NFE 20 | 5.64 | 8.05 | 6.62 | – | 1.60 |
| smoke model (2 epochs, 256 seq) | 5.10 | 7.53 | 6.10 | – | 1.36 |
| static mean pose | 4.70 | 7.51 | 5.46 | 4.33 | 1.35 |
| ground truth of another sentence | 6.77 | 8.74 | 7.65 | – | 1.65 |

Observations: (i) the trained model is SOKE-level on DTW but does not reach the paper's Phoenix
numbers; (ii) DTW-JPE barely separates a trained model from a 2-epoch one or from a static mean
pose (it rewards regression to the mean), while B-T does (10.17 vs 0.05 for the smoke model);
(iii) the generator is text-conditioned (val pose loss 0.0059 with the right text, 0.0072
shuffled, 0.0084 empty); (iv) error grows with NFE (1 → 5 → 20) while frame-to-frame jitter in feature space
grows to the ground-truth level (0.05 → 0.26 → 0.38 vs GT 0.37); the paper's Table 8 reports a
nearly flat curve; (v) val flow loss is lowest at epoch ~160 (0.0916) and 0.0983 at 300.
(v1 CSL-Daily and How2Sign were cancelled; see v2 above.)

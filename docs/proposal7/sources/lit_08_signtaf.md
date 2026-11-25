# Sources for `08_thiet_ke_lai_signtaf.md` (verified 2026-11-22)

Three parallel search branches. Every entry was checked on its arXiv or proceedings page.

- **[V]** = full text (or HTML) read.
- **[A]** = abstract or metadata only.
- Unverifiable items were dropped. "Not found" means a targeted search returned nothing; it is not proof of absence.

## A. SLP on the SOKE SMPL-X protocol (test; DTW-JPE body/hand, DTW-PA-JPE body/hand)

- SOKE, "Signs as Tokens", Zuo et al., ICCV 2025, arXiv 2411.17799 [V]
  - Phoenix 6.04/7.72, 4.77/1.38; CSL 7.38/9.68, 6.24/1.71; How2Sign 7.75/10.08, 6.82/2.35.
  - Word-level dictionary retrieval ablation, DTW-PA body/hand without → with: How2Sign 7.91/3.10 → 6.82/2.35; CSL 7.58/2.17 → 6.24/1.71; Phoenix 6.16/1.85 → 4.77/1.38.
  - Official metric code (`mGPT/metrics/dtw.py`): accumulated cost `D1[-1,-1]`, no window, not divided by path length.
- MaDiS, Zuo et al., arXiv 2601.19577 [V]
  - Phoenix 4.97/6.59; CSL 5.99/7.96; How2Sign 6.59/9.45 (JPE only).
- SignNMFlow, He et al., arXiv 2609.33339 [V]
  - Phoenix 4.37/5.57, 3.85/1.07; CSL 5.73/7.43, 5.10/1.30; How2Sign 6.76/9.72, 6.01/2.07.
  - Output-length rule not stated (full text searched).
  - Zero citations on Semantic Scholar as of 2026-11-22.
- SignSparK, Low et al., ECCV 2026, arXiv 2603.10446 [V]
  - Uses its own pseudo-GT (NLF, WiLoR, TEASER).
  - Gloss-free: Phoenix 4.40/7.10, 5.24/1.52.
  - SR-T2P (predicted glosses + dictionary keyframes injected by mask replacement in the FM path): Phoenix 3.51/4.82, 3.68/1.18; CSL 4.89/6.56, 4.87/1.37.
- M3T, Symeonidis-Herzig et al., BMVC 2026, arXiv 2603.23617 [V]
  - Own SMPL-FX pseudo-GT; not comparable.
- GARD, Sung et al., arXiv 2606.21234 [V]
  - Gloss input. Phoenix 6.69/6.90, 5.03/1.21; CSL 7.73/9.32, 6.98/1.56.
- SRVQ, Liu et al., WACV 2027, arXiv 2611.06781 [V]
  - Glosses used in training only. Phoenix 4.70/6.41, 5.32/1.26; CSL 6.30/8.38, 7.11/1.51.
- BLANC (2610.24109), MV-Sign (2611.07811), SignTok (2611.23944) [V]: own pseudo-GT or partial protocols.
- SignFlow, Khan et al., ICMI 2025 [A]: numbers only as copied by SignNMFlow.
- SemRAG, Liu et al., arXiv 2611.18189 [V]
  - Sentence-template retrieval with gated cross-attention in an FM generator (2-D keypoints).
  - States explicitly that it uses GT length.
- SignRR, Tito Cruz et al., BMVC 2026, arXiv 2608.28568 [V]
  - Gloss dictionary built from training segments, plus an RVQ-VAE refiner.
- Spoken2Sign, Zuo et al., ECCV 2024, arXiv 2401.04730 [V]
  - Text → gloss → dictionary 3-D signs, plus a learned connector.
- Hybrid AR-Diffusion SLP ("HybridSign"), Ye et al., ACL 2026, arXiv 2507.09105 [V]
  - soft-DTW term on 2-D pose descriptors.
  - Claims "DTW score reduced by approximately 20%" (v1 said 80%); no ablation table; γ not given.
- SLRTP 2025 Challenge report, Walsh et al., arXiv 2508.06951 [V]
- Conditional Collapse in SLP, Hong & Košecká, arXiv 2606.01643 [V]

## B. Alignment-invariant losses and over-smoothing

- Soft-DTW, Cuturi & Blondel, ICML 2017, arXiv 1703.01541 [V]
  - Barycenters: soft-DTW at γ = 0.01 beats DBA on all UCR datasets in DTW loss. Low γ gives "spurious" barycenters; γ = 1 is much smoother.
  - Forecasting ranks: 1.87 vs 3.46 under DTW evaluation; 3.99 vs 1.05 under Euclidean evaluation.
  - App. G (DTW loss of the barycenter): Euclidean mean 28.24 vs soft-DTW 11.95 (CBF).
- Differentiable divergences between time series, Blondel, Mensch & Vert, AISTATS 2021, arXiv 2010.08354 [V]
  - Soft-DTW can be negative and is not minimised at X = Y.
  - The divergence form is proven non-negative only for special costs.
- DILATE, Le Guen & Thome, NeurIPS 2019, arXiv 1909.09020 [V]; TPAMI extension, arXiv 2104.04610 [V]
- Keogh & Pazzani, Derivative DTW, SDM 2001 [V]: singularities.
- Petitjean et al., DBA, Pattern Recognition 2011 [A]
- Revisiting Over-Smoothness in TTS, Ren et al., ACL 2022, arXiv 2202.13066 [V]
  - Sharpness (VarL) on LJSpeech: GT 0.367; MAE 0.072; FastSpeech 2 (duration, pitch and energy conditioning) 0.184; Glow 0.310.
- EATS, Donahue et al., ICLR 2021, arXiv 2006.03575 [V]: removing soft-DTW drops MOS from 4.08 to 3.56.
- Parallel Tacotron 2, Elias et al., arXiv 2103.14574 [V]: soft-DTW with band 60, warp penalty 128, γ = 0.05.
- DeepNAG, Maghoumi et al., arXiv 2011.09149 [V]
- Ginosar et al., CVPR 2019 [V]; Saunders et al., arXiv 2008.12405 [A]
- EDM, Karras et al., arXiv 2206.00364 [V]: the denoiser optimum is a per-x weighted average (App. B.3).
- Flow Matching, Lipman et al., arXiv 2210.02747 [V]
- Banerjee, Guo & Wang, IEEE TIT 2005 [A]: the conditional mean is optimal exactly for Bregman losses.
- Diffusion with perceptual loss, Lin & Yang, arXiv 2401.00110 [V]
- Improved consistency training, Song & Dhariwal, arXiv 2310.14189 [V]
- Perception–Distortion Tradeoff, Blau & Michaeli, CVPR 2018, arXiv 1711.06077 [A]

## C. Priors and bridges, retrieval, MBR, RL, pullback metrics

- Grad-TTS, Popov et al., ICML 2021, arXiv 2105.06337 [V]
  - N(μ, I) prior beats N(0, I) at 10 steps (preferred in 93.8% of pairwise tests).
- PriorGrad, Lee et al., ICLR 2022, arXiv 2106.06406 [V]
- DiffSinger, Liu et al., AAAI 2022, arXiv 2105.02446 [V]
- Matcha-TTS, Mehta et al., ICASSP 2024, arXiv 2309.03199 [V]: N(0, I) source; μ used only as a condition; no ablation.
- Bridge-TTS, Chen et al., arXiv 2312.03491 [V]: data prediction works; noise and velocity prediction do poorly.
- DDBM, Zhou et al., ICLR 2024, arXiv 2309.16948 [V]: pure ODE sampler is blurry; needs stochasticity.
- CorrDiff, Mardani et al., 2025, arXiv 2309.15214 [V]
- Stochastic interpolants with data-dependent couplings, Albergo et al., ICML 2024, arXiv 2310.03725 [V]
- Conditional prior for flows, Issachar et al., 2025, arXiv 2502.09611 [V]: σ = 0.2 gives FID 23.55 vs 7.55 at σ = 0.7.
- FoldFlow, Bose et al., ICLR 2024, arXiv 2310.02391 [V]
- RFM, Chen & Lipman, ICLR 2024, arXiv 2302.03660 [V]
- ReMoDiffuse, ICCV 2023, arXiv 2304.01116 [V]; ReMoGPT, AAAI 2025 [V]; MoRAG, arXiv 2409.12140 [V]; ReMoMask, arXiv 2508.02605 [V]
- MBR with neural metrics, Freitag et al., TACL 2022, arXiv 2111.09388 [V]
  - The utility metric inflates: BLEURT 26.8 → 50.0 while BLEU 34.3 → 29.2.
- DiffuSeq, ICLR 2023, arXiv 2210.08933 [V]; Diffusion-LM, arXiv 2205.14217 [V]
- DDPO, arXiv 2305.13301 [V]; DRaFT, ICLR 2024, arXiv 2309.17400 [V]; Flow-GRPO, arXiv 2505.05470 [V]; DanceGRPO, arXiv 2505.07818 [V]; ReinFlow, NeurIPS 2025, arXiv 2505.22094 [V]; Pref-GRPO, arXiv 2508.20751 [V]
- ReinDiffuse, WACV 2025, arXiv 2410.07296 [V]; HY-Motion 1.0, arXiv 2512.23464 [V]; DiMo, arXiv 2602.04188 [V]
- RMPflow, arXiv 1811.07049 [V]; MorphIK, arXiv 2609.29908 [V]; QuaterNet, arXiv 1805.06485 [V]; MDM, arXiv 2209.14916 [V]; PoseRieFlow, arXiv 2611.09692 [V]
- MeanFlow, arXiv 2505.13447 [V]; Riemannian MeanFlow, ICML 2026, arXiv 2603.10718 [A]; Autoguidance, NeurIPS 2024, arXiv 2406.02507 [V]

## D. Added after the independent review (`09`)

Classic references supplied by the reviewer. They are widely known, and I did not re-fetch them:

- Kneip & Gasser 1992, Annals of Statistics (structural vs cross-sectional mean).
- Wang & Gasser 1997, "Alignment of curves by dynamic time warping", Annals of Statistics.
- Marron et al. 2015, "Functional data analysis of amplitude and phase variation", Statistical Science.
- Su, Kurtek, Klassen & Srivastava 2014, rate-invariant analysis of trajectories on Riemannian manifolds (TSRVF), Annals of Applied Statistics.
- Jayaraman et al., "Time-Agnostic Prediction", ICLR 2019, arXiv 1808.07784.
- Bruderlin & Williams 1995, "Motion signal processing", SIGGRAPH.
- Kovar & Gleicher 2003, "Flexible automatic motion blending with registration curves", SCA.
- Glow-TTS (Kim et al., NeurIPS 2020; monotonic alignment search).
- NAT-EA (Huang et al., ACM MM 2021; SLP with duration prediction and an external aligner).
- Signing at Scale / FS-Net (Saunders et al., CVPR 2022).
- Sign segmentation: Renz et al. 2021 (ICASSP); Moryossef et al. 2023 (EMNLP Findings).
- Ham2Pose (Arkushin et al., CVPR 2023; DTW-MJE pathologies).

Found after the review (search on 2026-11-22):

- Elastic Shape VAE for Skeleton Pose Trajectories, Rahman, Kumar, Barnes & Srivastava, arXiv 2605.09231 (preprint) [A].
  - Unconditional Riemannian VAE.
  - TSRVF on Kendall's shape manifold removes rate variability.
- Generative Motion Infilling From Imprecisely Timed Keyframes, Goel, Zhang, Liu & Fatahalian, Eurographics 2025, arXiv 2503.01016 [A].
  - Outputs a time-warping function plus spatial residuals.
  - Conditioned on keyframes, not text.

Search for a text-conditioned phase–amplitude (elastic) generative flow for motion or SLP: **not found**.

## Not found (targeted searches)

- A controlled study attributing motion under-articulation to timing variability.
- A diffusion or flow SMPL-X SLP model trained with soft-DTW on its output.
- A text-conditioned prior or bridge on (S³)^J or SO(3)^J for body motion or SLP.
- FM on joint rotations with an FK-pullback metric for the path or loss.
- MBR or medoid selection in motion generation, TTS or SLP.
- RL with a DTW-to-GT reward, or any RL for SLP.
- A paper after September 2026 that compares against SignNMFlow.

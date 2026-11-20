# Literature review: SLP context for SignNMFlow (compiled 2026-11-19)

Scope: text-to-sign 3D (SMPL-X) production, sign-language structure, evaluation, data.
Every entry below was confirmed by fetching the listed URL: an arXiv abs page, a proceedings or anthology page, OpenAlex, Crossref or PMC.
Venue notes in brackets say which page confirmed the venue. Papers I could not confirm by fetch were dropped. The one dropped is Kipp et al., ASSETS 2011, a Deaf-user avatar study: its pages returned 403 or refused the connection.

Target paper (for reference): J. He, S. Tang, S. You, Y. Hao, L. Cheng, R. Hong. "Naturalness-guided Manifold Flow Matching for Sign Language Production" (SignNMFlow). arXiv:2609.33339, 27 Sep 2026. https://arxiv.org/abs/2609.33339 [verified: fetched abs + full PDF].
Facts read from the PDF:
- **Representation.** J=41 unit quaternions plus a 10-d Euclidean expression channel. Source distribution: uniform on a hemisphere.
- **Model.** Target (x1) prediction with an MSE loss. 5-step Euler sampling. mBART-large-cc25 text encoder.
- **Naturalness measure.** A KDE over K=500 prototypes per joint (sigma=0.5).
- **Pseudo-ground truth and back-translation.** SMPL-X pseudo-GT comes from SOKE (Phoenix, CSL) and NSA (How2Sign). The back-translator is "the same as SOKE". No ground-truth-motion BLEU row is reported.
- **User study.** 5 CSL signers, 18 instances, forced choice: 81.1% preferred SignNMFlow.
- **Reporting.** No seeds or confidence intervals. NFE=5 scores slightly better than NFE=20 (CSL DTW-JPE All 16.86 vs 16.98).
- **Missing citation.** It does not cite Metric Flow Matching (Kapusniak et al., NeurIPS 2024), the closest prior art.

---

## 1. SLP state of the art (2020-2026)

### 1.1 Neural Sign Actors (NSA)
V. Baltatzis, R. A. Potamias, E. Ververas, G. Sun, J. Deng, S. Zafeiriou. "Neural Sign Actors: A Diffusion Model for 3D Sign Language Production from Text." CVPR 2024. https://arxiv.org/abs/2312.02702 [verified: fetched abs, comments "Accepted at CVPR 2024"; HTML v2 fetched]
- **What it is.** A diffusion model over SMPL-X built on an anatomically informed graph network over the skeleton.
- **How2Sign pseudo-GT (used by SOKE and SignNMFlow).** Built by OSX initialisation, then refinement against MediaPipe 2D reprojection. Only arm and hand joints are optimised, under a PCA prior from AMASS and ARCTIC plus a temporal-smoothness term. The authors acknowledge jitter on blurry frames and ambiguous hands.
- **Results.** How2Sign DTW 7.83 and back-translation BLEU-4 13.12. Human study with 15 ASL-fluent participants: semantic alignment 5.8/10, versus 8.7 for ground truth and 8.1 for the fittings.
- **Implication for SignNMFlow.** The How2Sign "ground truth" is itself an optimised fit with smoothness and PCA priors. A model that regresses toward smooth, prior-like poses is partly rewarded for matching the fitter's own bias, not the signer.

### 1.2 SOKE (Signs as Tokens)
R. Zuo, R. A. Potamias, E. Ververas, J. Deng, S. Zafeiriou. "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator." ICCV 2025. https://arxiv.org/abs/2411.17799 [verified: abs "Accepted by ICCV 2025"; HTML v3 and PDF/supplement read]
- **Method.** A decoupled VQ tokenizer (body codebook 96, 192 per hand) feeds mBART-large-cc25 with multi-head decoding. It also retrieves word-level signs from external dictionaries.
- **Benchmark it defines.** Body DTW-JPE and DTW-PA-JPE, hand DTW-JPE and DTW-PA-JPE, and B-T BLEU-4. Units are not stated and the back-translator is not described; the paper says only "following [3]" (NSA).
- **Fitting pipeline for Phoenix and CSL-Daily.** OSX body, WiLoR hands, then MediaPipe-based refinement. On the SGNify mocap set (57 isolated signs) the per-vertex error is body 46.73 mm and hands 10.55 / 8.94 mm (Table S5).
- **User study.** 5 ASL and 4 CSL signers rated 1-10 on 15 signs. SOKE scored 5.32 / 6.45 versus 1.36 / 2.70 for S-MotionGPT.
- **Stated limitation.** The pseudo-GT "inevitably contain[s]" reconstruction error.
- **Implication for SignNMFlow.**
  - Every number SignNMFlow reports is computed against these fits. SignNMFlow's hand DTW-PA-JPE gains (1.38 to 1.07 on Phoenix) are similar in magnitude to the fitter's own hand error, if the metric is in cm, which is unconfirmed.
  - Adopt SOKE's no-dictionary variant as the fair comparison for a retrieval-free model.

### 1.3 SignSparK
J. Low, A. Symeonidis-Herzig, M. Ivashechkin, O. Mercanoglu Sincan, R. Bowden. "SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning." ECCV 2026. https://arxiv.org/abs/2603.10446 [verified: abs "Accepted at ECCV 2026"; HTML read]
- **Method.** The FAST segmenter (BIO + CTC, WiLoR features) gives onset, mid and offset keyframes per sign. A conditional flow matching model in 6D rotations infills dense motion with CFG. Losses: CFM + reconstruction + velocity (2:1:1).
- **Data.** Its own pseudo-GT (NLF body, WiLoR hands, TEASER face with 50 expression coefficients). Four languages, including 10% of BOBSL.
- **Results.**
  - Gloss-free text-to-pose (T2P): Phoenix 5.24 / 1.52 (PA) and 4.40 / 7.10.
  - Sign-retrieval T2P: Phoenix 3.68 / 1.18 / 3.51 / 4.82; CSL 4.87 / 1.37 / 4.89 / 6.56.
  - No B-T for T2P, because "ground-truth baselines are publicly unavailable".
  - The paper argues that direct text-to-pose models give averaged, "washed-out" motion.
  - Deaf/BSL user study (16 signers): 86.7% preferred SignSparK over SOKE/S-MotionGPT.
- **Implication for SignNMFlow.**
  - SignNMFlow lists only the gloss-free SignSparK row. The retrieval row beats SignNMFlow on Phoenix body PA (3.68 vs 3.85), body JPE (3.51 vs 4.37) and hand JPE (4.82 vs 5.57), and on 3 of 4 CSL columns.
  - SignSparK's pseudo-GT differs from SOKE's. Its numbers are not strictly comparable in either direction, and this should be stated.
  - Its keyframe anchors and velocity loss are adoptable techniques.

### 1.4 SignFlow
N. Khan, B. Wu, S. Tan, C. T. Ishi, K. Nakadai. "SignFlow: End-to-End Sign Language Generation for One-to-Many Modeling using Conditional Flow Matching." Proc. 27th ACM ICMI 2025, pp. 173-180. DOI 10.1145/3716553.3750765. https://api.openalex.org/works/doi:10.1145/3716553.3750765 [verified: OpenAlex record; no abstract available]
- **Method.** Euclidean conditional flow matching with OT coupling, framed as one-to-many generation. The method description here comes from SignNMFlow's related work, because the abstract is unavailable.
- **Numbers.** SignNMFlow reports SignFlow on the SOKE benchmark (e.g., Phoenix PA hand 1.20) with no B-T. It is unclear whether SignFlow itself used SOKE's SMPL-X, or whether SignNMFlow re-ran it.
- **Implication for SignNMFlow.** SignFlow is the direct ablation partner (linear versus geodesic paths). Ask SignNMFlow to state the provenance of the SignFlow numbers. SignFlow's "one-to-many" framing also implies diversity should be measured, which SignNMFlow does not do.

### 1.5 Sign-IDD
S. Tang, J. He, D. Guo, Y. Wei, F. Li, R. Hong. "Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production." AAAI 2025. https://arxiv.org/abs/2412.13609 [verified: abs "Accepted by AAAI 2025"]
- **Method.** Same group as SignNMFlow. Gloss-to-pose diffusion that re-parameterises joints as 4D bones (3D direction plus length), with attribute-controllable diffusion. Evaluated on PHOENIX14T and USTC-CSL with skeletons, not on the SOKE benchmark.
- **Implication for SignNMFlow.** Bone-length and direction disentanglement is a cheaper structural prior than a manifold. A skeleton-direction baseline would isolate whether the quaternion manifold matters.

### 1.6 GARD
J. Sung, B. Kim, C. Xin, H. J. Chang, C. Kim, S.-I. Choi, Y. Choi. "Context-Aware Autoregressive Diffusion for Gloss-Wise Sign Language Production." arXiv:2606.21234, Jun 2026. https://arxiv.org/abs/2606.21234 [verified: abs + HTML; no peer-reviewed venue listed]
- **Method.** Gloss-by-gloss autoregressive diffusion on SOKE SMPL-X (6D rotations plus positions). It conditions on the previous gloss's text and motion. Inter-Gloss Transition Guidance uses geodesic rotation distance at boundaries, followed by a Global Motion Harmonizer.
- **Results.** Phoenix B4 13.16, PA 5.03 / 1.21, JPE 6.69 / 6.90. CSL B4 12.73, PA 6.98 / 1.56. Its re-run of SOKE gives Phoenix 4.72 / 1.39 / 6.12 / 7.80 against SOKE's reported 4.77 / 1.38 / 6.04 / 7.72.
- **Implication for SignNMFlow.**
  - SignNMFlow cites GARD but leaves it out of its tables.
  - The SOKE re-run differs by 0.01-0.08 from SOKE's reported values. That gives a scale for reproduction noise, which SignNMFlow should report with seeds and CIs.
  - Gloss-boundary coarticulation is a structure SignNMFlow ignores.

### 1.7 T2S-GPT
A. Yin, H. Li, K. Shen, S. Tang, Y. Zhuang. "T2S-GPT: Dynamic Vector Quantization for Autoregressive Sign Language Production from Text." ACL 2024. https://arxiv.org/abs/2406.07119 [verified: abs "Accepted by ACL 2024"]
- **Method.** DVA-VAE gives variable-rate codes according to information density, and a GPT predicts codes plus durations. The paper also introduces PHOENIX-News (486 h).
- **Implication for SignNMFlow.** Information density varies within a sentence (holds versus transitions). A uniform-time flow ignores this; duration and length modelling is an under-addressed axis.

### 1.8 MotionGPT (the "S-MGPT" baseline)
B. Jiang, X. Chen, W. Liu, J. Yu, G. Yu, T. Chen. "MotionGPT: Human Motion as a Foreign Language." arXiv:2306.14795, 2023. https://arxiv.org/abs/2306.14795 [verified: abs; the venue is not shown on the fetched page]
- **Summary.** A generic VQ-token motion LM. In the SOKE benchmark it is reimplemented as S-MotionGPT.
- **Implication for SignNMFlow.** It is already a baseline. Its hand PA (3.41 on Phoenix) shows that generic tokenizers lose hands, which motivates hand-specific structure.

### 1.9 SignAvatar
L. Dong, L. Chaudhary, F. Xu, X. Wang, M. Lary, I. Nwogu. "SignAvatar: Sign Language 3D Motion Reconstruction and Generation." IEEE FG 2024. https://arxiv.org/abs/2405.07974 [verified: abs comments FG 2024]
- **Summary.** A transformer CVAE with curriculum learning that generates word-level SMPL-X motion. Releases ASL3DWord (built from WLASL).
- **Implication for SignNMFlow.** Isolated-sign SMPL-X data (ASL3DWord) could be used to test lexical distinguishability of generated signs, which sentence-level DTW cannot measure.

### 1.10 Spoken2Sign
R. Zuo, F. Wei, Z. Chen, B. Mak, J. Yang, X. Tong. "A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars." ECCV 2024. https://arxiv.org/abs/2401.04730 [verified: abs "Accepted by ECCV 2024"]
- **Pipeline.** Text-to-gloss, then retrieval of SMPLSign-X-fitted dictionary signs, then a learned sign connector that predicts coarticulation durations.
- **Implication for SignNMFlow.** A retrieve-and-connect baseline is strong on lexical precision. SignNMFlow should report against a retrieval baseline, or at least a nearest-training-sentence retrieval baseline, to show generation beats lookup.

### 1.11 SignLLM
S. Fang, C. Chen, L. Wang, C. Zheng, C. Sui, Y. Tian. "SignLLM: Sign Language Production Large Language Models." arXiv:2405.10718 (v3 Apr 2025). https://arxiv.org/abs/2405.10718 [verified: abs; no venue listed]
- **Summary.** A multilingual (8 languages) pose-producing LLM trained on the Prompt2Sign dataset (2D/upper-body compressed pose), with RL-style "Priority Learning Channel" sampling.
- **Implication for SignNMFlow.** It shows the multilingual direction (SignNMFlow-M covers only 3 languages). It is not on SMPL-X and has no SOKE-benchmark numbers.

### 1.12 SignDiff
S. Fang, C. Sui, Y. Zhou, X. Zhang, H. Zhong, Y. Tian, C. Chen. "SignDiff: Diffusion Model for American Sign Language Production." arXiv:2308.16082 (v4 Apr 2025). https://arxiv.org/abs/2308.16082 [verified: abs; "Camera-Ready Version", venue not named]
- **Summary.** Pose-to-video dual-condition diffusion with FR-Net (which reduces extra fingers), plus a text-to-pose stage. Reports How2Sign B-T BLEU-4 17.19 on dev and 12.85 on test with 2D poses.
- **Implication for SignNMFlow.** High back-translation BLEU on How2Sign has appeared before with different back-translators. This again shows B-T numbers are only comparable under one fixed, released back-translator.

### 1.13 G2P-DDM
P. Xie, Q. Zhang, T. Peng, H. Tang, Y. Du, Z. Li. "G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model." AAAI 2024. https://arxiv.org/abs/2208.09141 [verified: abs "Accepted by AAAI 2024"]
- **Summary.** Pose-VQVAE plus discrete diffusion (CodeUnet), with heuristic length prediction. Gloss-to-pose on 2D/3D keypoints.
- **Implication for SignNMFlow.** Length prediction is a known weak point. SignNMFlow does not describe how output length is set at inference, and DTW hides length errors.

### 1.14 MS2SL
J. Ma, W. Wang, Y. Yang, F. Zheng. "MS2SL: Multimodal Spoken Data-Driven Continuous Sign Language Production." Findings of ACL 2024, pp. 7241-7254. https://aclanthology.org/2024.findings-acl.432 [verified: ACL Anthology]
- **Summary.** A sequence diffusion model conditioned on text or speech, with a joint text-audio-sign embedding used as training feedback. Evaluated on How2Sign and PHOENIX14T.
- **Implication for SignNMFlow.** A joint embedding used as a semantic consistency loss is a cheap addition that targets semantics directly rather than joint error.

### 1.15 Text2SignDiff
L. Feng, L. Wang, K. Hu, D. Kong, Z. Wang. "Text2Sign Diffusion: A Generative Approach for Gloss-Free Sign Language Production." arXiv:2509.10845, Sep 2025. https://arxiv.org/abs/2509.10845 [verified: abs; no venue]
- **Summary.** Gloss-free latent diffusion with a cross-modal signing aligner (shared sign/text latent). Evaluated on PHOENIX14T and How2Sign skeletons.
- **Implication for SignNMFlow.** Another argument for a learned sign-text alignment space. It could double as an evaluation embedding.

### 1.16 M3T
A. Symeonidis-Herzig, J. Low, O. Mercanoglu Sincan, R. Bowden. "M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production." BMVC 2026. https://arxiv.org/abs/2603.23617 [verified: abs "Accepted at BMVC 2026"; HTML read]
- **Method.** SMPL-FX: SMPL-X plus a 100-d FLAME expression from Pixel3DMM. FSQ tokenizers per modality (face codebook use rises from 78.8% to 99.0%), an autoregressive transformer and an auxiliary sign-to-text objective.
- **Results.** Against SOKE: How2Sign PA 4.28 / 2.28 vs 6.82 / 2.35; Phoenix hand PA 1.46 vs 1.38 (worse; attributed to noisy hand pseudo-GT).
- **Critiques and controls.**
  - "Each system is evaluated against its own pseudo-GT", so the comparison is uncontrolled.
  - SOKE faces vary mostly in jaw aperture, because the expression channel is 10-d.
  - Back-translation BLEU-4 upper bounds on ground-truth 3DMM motion are 2.04 (How2Sign), 7.11 (CSL) and 11.84 (Phoenix).
  - NMFs-CSL probe: 58.3% vs 49.0%. BSL-signer study: 64.6% overall preference.
- **Implication for SignNMFlow.**
  - SignNMFlow inherits SOKE's 10-d Euclidean face channel, which is unexamined.
  - Report a ground-truth-motion back-translation ceiling. M3T's ceilings are far below the 15-17 BLEU-4 that SignNMFlow reports on How2Sign, a gap that needs explaining.

### 1.17 SignMask
Y. Xia, Q. Zhan, X. Luo, X. Shi, Y. Wang. "SignMask: Structure-aware Masked Modeling for Holistic 3D Sign Language Production." ACM TOMM 22(1), Jan 2026. DOI 10.1145/3776750. https://api.crossref.org/works/10.1145%2F3776750 [verified: Crossref record; ACM page 403]
- **Summary.** Hierarchical body/hand SMPL-X tokenizer with part-weighted grouping, plus a masked-token transformer with parallel iterative decoding.
- **Implication for SignNMFlow.** Another SMPL-X SOTA claimant that SignNMFlow does not compare against. Its part-weighting is a structural prior.

### 1.18 PIDiffSign
E. Colonna, M. Diaz, G. Vessio, M. A. Ferrer, G. Castellano. "A Diffusion-Based Kinematics and Dynamics Model for 3D Sign Language Production." arXiv:2607.14836 (v2 Nov 2026). https://arxiv.org/abs/2607.14836 [verified: abs; no venue]
- **Method.** A forward-kinematics head (bone directions and lengths) plus rigid-body dynamics priors. On PHOENIX14T and CSL-Daily it cuts back-translation WER by about 5 points; geometric penalties had no measurable effect.
- **Key finding.** Generated signing "has noticeably less kinetic energy" than real signing.
- **Implication for SignNMFlow.**
  - Report motion energy and velocity statistics against ground truth. A naturalness KDE over prototypes plus x1-MSE is exactly the kind of setup that damps energy.
  - PIDiffSign's null result for geometric penalties suggests manifold validity alone may not drive semantics.

### 1.19 DARSLP and A2V-SLP (Hacettepe)
S. M. Tasyurek, T. Kiziltepe, H. Y. Keles. "Disentangle and Regularize: Sign Language Production with Articulator-Based Disentanglement and Channel-Aware Regularization." arXiv:2504.06610 (v3 Sep 2025). https://arxiv.org/abs/2504.06610 [verified: abs]
S. M. Taşyürek, E. M. İskender, H. Y. Keles. "A²V-SLP: Alignment-Aware Variational Modeling for Disentangled Sign Language Production." arXiv:2602.11861, Feb 2026. https://arxiv.org/abs/2602.11861 [verified: abs]
- **Summary.** Per-articulator (face, each hand, body) latent spaces with channel-weighted KL. A²V-SLP predicts distributions rather than points, which reduces regression to the mean.
- **Implication for SignNMFlow.** Per-articulator factorisation (hands separate from body) and distributional prediction are both direct responses to averaging. SignNMFlow treats all 41 joints in one product manifold with one loss weight.

### 1.20 HybridSign
M. Ye, X. Ye, S. Manoharan. "Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production." ACL 2026 (Long), pp. 750-763. https://aclanthology.org/2026.acl-long.31/ [verified: ACL Anthology]
- **Summary.** Causal frame-wise generation plus flow-based refinement, with confidence-aware attention for noisy pose labels.
- **Implication for SignNMFlow.** Joint-confidence weighting is a direct remedy for noisy pseudo-GT (e.g., WiLoR or MediaPipe confidences). SignNMFlow weights all joints equally.

### 1.21 TEAM-Sign
Z. An, R. Kawakami. "Teach Me Sign: Stepwise Prompting LLM for Sign Language Production." IEEE ICIP 2025. https://arxiv.org/abs/2507.10972 [verified: abs "Accepted by IEEE ICIP 2025"]
- **Summary.** A fine-tuned off-the-shelf LLM with stepwise prompting to bridge grammar differences.
- **Implication for SignNMFlow.** The text side (mBART) could be swapped for an LLM. This is orthogonal to the flow, so it is low priority.

### 1.22 Progressive Transformers and the MDN extension (origin of B-T evaluation)
B. Saunders, N. C. Camgoz, R. Bowden. "Progressive Transformers for End-to-End Sign Language Production." ECCV 2020. https://arxiv.org/abs/2004.14874 ; venue: https://mlanthology.org/eccv/2020/saunders2020eccv-progressive [verified]
B. Saunders, N. C. Camgoz, R. Bowden. "Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and Mixture Density Networks." arXiv:2103.06982, 2021. https://arxiv.org/abs/2103.06982 [verified: abs; journal venue not shown on page]
- **Summary.** Introduced counter decoding and back-translation evaluation for SLP. The 2021 paper adds adversarial training and an MDN against regression to the mean, plus a Deaf user evaluation.
- **Implication for SignNMFlow.** Regression to the mean has been documented since 2020. SignNMFlow's x1-prediction MSE objective is the classic setup that produces it, so SignNMFlow needs evidence against it.

### 1.23 Text2Mesh (T2Mesh)
S. Stoll, A. Mustafa, J.-Y. Guillemaut. "There and Back Again: 3D Sign Language Generation from Text Using Back-Translation." 3DV 2022, pp. 187-196. DOI 10.1109/3DV57658.2022.00031. Surrey record: http://openresearch.surrey.ac.uk/discovery/fulldisplay?&context=L&vid=44SUR_INST:ResearchRepository&docid=alma99783722402346 [verified]
- **Summary.** Back-translation used as supervision, then Pose2Mesh to produce meshes. It is a baseline in the SOKE benchmark (reimplemented).
- **Implication for SignNMFlow.** Using the back-translator as a training signal and as the metric at once is risky. Keep the evaluation back-translator independent of any training loss.

### 1.24 Data-driven VQ representation and sign stitching
H. Walsh, A. Ravanshad, M. Rahmani, R. Bowden. "A Data-Driven Representation for Sign Language Production." IEEE FG 2024. https://arxiv.org/abs/2404.11499 [verified: abs comments FG 2024]
- **Summary.** A learned VQ codebook of short motion tokens, used as a lexicon, plus sign stitching.
- **Implication for SignNMFlow.** A learned motion codebook is an alternative way to encode "naturalness" (discrete prototypes) and is directly comparable to SignNMFlow's K=500 prototypes.

### 1.25 SignAvatars dataset
Z. Yu, S. Huang, Y. Cheng, T. Birdal. "SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark." ECCV 2024. https://arxiv.org/abs/2310.20436 [verified: abs "ECCV2024"]
- **Summary.** About 70K videos, 153 signers, 8.34M frames of SMPL-X/MANO annotations, with HamNoSys, word and sentence prompts.
- **Implication for SignNMFlow.** A larger, more diverse SMPL-X source for building prototypes or pretraining, and a second benchmark that avoids the single SOKE fitter.

---

## 2. Sign-language structure a generator could exploit

### 2.1 SGNify (linguistic priors for hands)
M.-P. Forte, P. Kulits, C.-H. P. Huang, V. Choutas, D. Tzionas, K. J. Kuchenbecker, M. J. Black. "Reconstructing Signing Avatars From Video Using Linguistic Priors." CVPR 2023, pp. 12791-12801. https://arxiv.org/abs/2304.10482 ; https://mlanthology.org/cvpr/2023/forte2023cvpr-reconstructing [verified]
- **Priors.** Battison's symmetry and dominance conditions become 4 classes × {static, transitioning}.
  - A hand-pose symmetry loss: mirrored left/right finger poses for symmetric classes.
  - Hand-pose invariance: a constant handshape, or a SLERP between start and end handshapes, within a sign.
- **Off-the-shelf fitting error.** Per-vertex hand error against mocap is about 19-25 mm.
- **Implication for SignNMFlow.**
  - Concrete, cheap priors exist. Add a left/right mirror-symmetry term on quaternions for symmetric segments, and a handshape-constancy (low angular velocity) term within sign segments.
  - On SO(3) these are natural geodesic penalties, which fits SignNMFlow's own machinery.

### 2.2 ASL-LEX 2.0
Z. S. Sehyr, N. Caselli, A. M. Cohen-Goldberg, K. Emmorey. "The ASL-LEX 2.0 Project: A Database of Lexical and Phonological Properties for 2,723 Signs in American Sign Language." J. Deaf Studies and Deaf Education 26(2):263-277, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685 [verified]
- **Coded properties.** Sign type and symmetry (Battison), major and minor location, contact, selected fingers, flexion, spread, thumb, handshape (dominant and non-dominant), path movement, internal movement, and repetition.
- **Implication for SignNMFlow.** Ready-made phonological labels could serve as (a) prototype clusters (handshape classes instead of K-means on joints) or (b) a phonological-accuracy evaluation probe on How2Sign (ASL).

### 2.3 Phonology-guided motion generation
R. Hong, J. Košecká. "Toward Phonology-Guided Sign Language Motion Generation: A Diffusion Baseline and Conditioning Analysis." arXiv:2603.17388, Mar 2026. https://arxiv.org/abs/2603.17388 [verified: abs]
- **Summary.** An MDM-style SMPL-X diffusion model conditioned on ASL-LEX attributes. Converting symbolic notation into natural language is necessary for CLIP conditioning. It beats SignAvatar on gloss discriminability.
- **Implication for SignNMFlow.** Phonological conditioning measurably improves lexical distinguishability. "Gloss discriminability" is an evaluation SignNMFlow could add.

### 2.4 HandMDM
L. Bensabath, M. Petrovich, G. Varol. "Text-Driven 3D Hand Motion Generation from Sign Language Data." arXiv:2508.15902, Aug 2025. https://arxiv.org/abs/2508.15902 [verified: abs]
- **Summary.** An LLM converts dictionary sign attributes (handshape, location, movement) into text descriptions, and a hand-motion diffusion model is trained on them. It generalises to unseen signs and languages.
- **Implication for SignNMFlow.** It is a candidate pretrained hand prior or hand-motion evaluator. Hands are where SignNMFlow's claims concentrate.

### 2.5 Ham2Pose (HamNoSys to pose, nDTW)
R. Shalev-Arkushin, A. Moryossef, O. Fried. "Ham2Pose: Animating Sign Language Notation into Pose Sequences." CVPR 2023, pp. 21046-21056. https://arxiv.org/abs/2211.13613 ; https://mlanthology.org/cvpr/2023/arkushin2023cvpr-ham2pose [verified]
- **Summary.** HamNoSys-conditioned pose generation with weak supervision. Introduces nDTW / DTW-MJE with missing-keypoint handling.
- **Implication for SignNMFlow.** HamNoSys is a language-independent phonological intermediate, a possible conditioning signal for multilingual SignNMFlow-M. It is also the origin of DTW-based pose metrics, whose tuning matters (see 3.1).

### 2.6 Linguistically motivated segmentation
A. Moryossef, Z. Jiang, M. Müller, S. Ebling, Y. Goldberg. "Linguistically Motivated Sign Language Segmentation." Findings of EMNLP 2023, pp. 12703-12724. https://aclanthology.org/2023.findings-emnlp.846 [verified]
- **Summary.** BIO tagging of signs and phrases. Phrase boundaries come from prosody (optical flow). Hand shape and 3D hand normalisation are analysed as cues for sign boundaries.
- **Implication for SignNMFlow.** Off-the-shelf sign and phrase boundaries make segment-aware priors possible (handshape constancy within a sign, transitions between signs). They also allow boundary-aware evaluation, such as per-segment DTW.

### 2.7 Spoken2Sign, SignSparK and GARD (cross-reference)
- **Spoken2Sign (1.10).** Coarticulation durations between signs are learnable.
- **SignSparK (1.3).** Onset, mid and offset keyframes per sign.
- **GARD (1.6).** Geodesic transition guidance at gloss boundaries.
- **Implication for SignNMFlow.** Keyframe or boundary conditioning is the strongest documented lever against averaged motion, and it composes naturally with geodesic paths. A SLERP between keyframes is SignNMFlow's geodesic at t-space.

### 2.8 Non-manual markers (cross-reference)
- **M3T (1.16).** SMPL-FX, NMFs-CSL probe.
- **SignSparK (1.3).** TEASER face, 50 expression coefficients.
- **Implication for SignNMFlow.** Mouthing and eyebrows are grammatical. SignNMFlow's face is a 10-d Euclidean side channel, weighted 0.1 and not evaluated at all.

---

## 3. Evaluation critique

### 3.1 Meaningful pose-based evaluation
Z. Jiang, C. Leong, A. Moryossef, A. Göhring, A. Rios, O. Cory, M. Ivashechkin, N. Tarigopula, B. Zhang, R. Sennrich, S. Ebling. "Meaningful Pose-Based Sign Language Evaluation." WMT 2025. https://arxiv.org/abs/2510.07453 [verified: abs "Accepted at WMT 2025"; HTML read]
- **DTW metrics.** Results hinge on keypoint selection, normalisation, trimming and masking. Untuned nAPE and nDTW correlate negatively with humans (about -0.41 and -0.10). Tuned DTWpp reaches about 0.76 segment-level correlation.
- **Embedding and back-translation metrics.** SignCLIP embeddings excel at the sign level but struggle at the sentence level. Back-translation likelihood is the most consistent (about 0.42), ahead of BLEU, chrF and BLEURT.
- **Toolkit.** github.com/sign-language-processing/pose-evaluation.
- **Implication for SignNMFlow.** DTW-JPE as used by SOKE has never been validated against human judgement on SMPL-X. Adopt back-translation likelihood or negative log-likelihood, not only BLEU-4, and report tuned DTW variants from the toolkit.

### 3.2 SLRTP 2025 SLP Challenge
H. Walsh, E. Fish, O. Mercanoglu Sincan, ..., S. Tang, J. He, ..., H. Y. Keles (22 authors). "SLRTP2025 Sign Language Production Challenge: Methodology, Results, and Future Work." CVPR 2025 SLRTP Workshop. https://arxiv.org/abs/2508.06951 [verified: abs + HTML]
- **Total distance.** Hand path length normalised by ground truth (1.0 is optimal), introduced to capture expressiveness. The Progressive Transformer baseline covered only about 25% of ground-truth hand travel.
- **DTW-MJE bias.** It "tended to favor less articulated or longer sequences". The baseline scored competitively on DTW-MJE (0.0467) despite very low total distance.
- **Winner.** A retrieval-based system (BLEU-1 31.40).
- **SignNMFlow authors' entry.** hfut-lmc (Tang and He, SignNMFlow authors) placed 2nd with a text-conditional diffusion model whose outputs on PHOENIX14T were about 2.5× ground-truth length.
- **Implication for SignNMFlow.**
  - This is direct evidence that DTW-style joint errors reward under-articulated, averaged motion. Report a total-distance / energy ratio and a velocity-spectrum match alongside DTW-JPE.
  - Report predicted versus ground-truth length.

### 3.3 Content-free baselines audit
A. Pelykh, O. Mercanoglu Sincan, R. Bowden. "Auditing the Evaluation of Sign Language Production with Content-Free Baselines." NeurIPS 2026 Workshop (Foundation Models for Temporal Systems). https://neurips.cc/virtual/2026/175052 [verified: NeurIPS page; OpenReview PDF blocked]
- **Summary.** Content-free signals score well on both distance and back-translation metrics. Two SOTA SLP models are shown to score well "without learning to sign". The authors propose measured floors and ceilings and a shared translation model.
- **Implication for SignNMFlow.** Add content-free baselines on the SOKE benchmark: the dataset mean pose, a per-dataset mean sequence, a random training sequence, and the nearest-neighbour training sentence. Report SignNMFlow's numbers as a position between floor and ceiling (ground-truth motion).

### 3.4 Conditional collapse
R. Hong, J. Košecká. "Conditional Collapse in Sign Language Production: A Diagnostic and a Coverage Argument." arXiv:2606.01643 (v2 Nov 2026). https://arxiv.org/abs/2606.01643 [verified: abs + HTML]
- **Faithfulness diagnostic.** A three-level diagnostic (initial-pose conditioning, diversity, target faithfulness) uses latent pairwise-distance ratios from a frozen motion autoencoder. On How2Sign no model reaches faithfulness, and FID varies widely without tracking it.
- **Human-calibrated DTW.** DTW/random ratio for two humans signing the same sentence is 0.74 (Phoenix) and 0.59 (CSL); for models it is 0.86 and 0.76.
- **Back-translation ceiling.** A How2Sign-only back-translator plateaus at about 2.5 BLEU-4 even on ground-truth motion. Non-zero BLEU arises from "high-frequency vocabulary" n-grams.
- **Coverage.** Faithfulness tracks how densely training sentences cover the test sentences.
- **Implication for SignNMFlow.**
  - Calibrate DTW against human-human repeats (Phoenix and CSL have repeated sentences).
  - Test whether outputs change with the text: swap the conditioning text and measure distance.
  - Report results stratified by nearest-training-sentence similarity. Phoenix weather sentences are highly repetitive (see SignNMFlow's own Fig. 2 example), so a low DTW may partly reflect memorisation.

### 3.5 BackTranslation2.0
O. Cory, M. Ivashechkin, K. Sahin, O. Ranum, J. Low, E. Fish, A. Pelykh, O. Mercanoglu Sincan, R. Bowden. "BackTranslation2.0 -- A Linguistically Motivated Metric to Assess Sign Language Production." ECCV 2026. https://arxiv.org/abs/2606.28673 [verified: abs "Accepted at ECCV 2026"]
- **Summary.** An agentic, tool-based metric scoring grammar, phonological accuracy, motion fluency and fidelity, with deterministic weighting. Validated against BSL human ratings (protocol co-designed with Deaf experts) and compared with 6 baseline metrics.
- **Implication for SignNMFlow.** The emerging standard explicitly separates phonological accuracy from fluency. SignNMFlow's naturalness claim is a fluency claim and should be evaluated as one, separately from semantics.

### 3.6 SignCLIP
Z. Jiang, G. Sant, A. Moryossef, M. Müller, R. Sennrich, S. Ebling. "SignCLIP: Connecting Text and Sign Language by Contrastive Learning." EMNLP 2024 (Main). https://arxiv.org/abs/2407.01264 [verified: abs]
- **Summary.** CLIP-style text-to-sign-pose embedding pretrained on Spreadthesign (about 500K clips, up to 44 languages).
- **Implication for SignNMFlow.** It offers a text-motion retrieval score (R-precision / MM-Dist analogue) that is independent of the SOKE back-translator. It needs SMPL-X-to-keypoint conversion and in-domain fine-tuning.

### 3.7 SiLVERScore
S. Imai, M. Inan, A. B. Sicilia, M. Alikhani. "SiLVERScore: Semantically-Aware Embeddings for Sign Language Generation Evaluation." RANLP 2025, pp. 452-461. https://arxiv.org/abs/2509.03791 ; https://aclanthology.org/2025.ranlp-1.54/ [verified]
- **Summary.** A joint-embedding metric (built on SignCLIP) that avoids back-translation. ROC AUC 0.99 separating correct from random pairs on PHOENIX-14T and CSL-Daily.
- **Implication for SignNMFlow.** It can be computed on exactly SignNMFlow's datasets, which makes it a ready second semantic metric.

### 3.8 Deaf-led critique
A. Desai, M. De Meulder, J. A. Hochgesang, A. Kocab, A. X. Lu. "Systemic Biases in Sign Language AI Research: A Deaf-Led Call to Reevaluate Research Agendas." arXiv:2403.02563, 2024. https://arxiv.org/abs/2403.02563 [verified: abs]
- **Summary.** A review of 101 papers. It flags unrepresentative data, annotations without linguistic grounding, and a lack of Deaf leadership.
- **Implication for SignNMFlow.** The 5-signer CSL-only forced-choice study (no ASL or DGS raters, no comprehension task, no Deaf-status report) is thin. Pre-register a comprehension-based study with Deaf native signers per language.

---

## 4. Data issues

### 4.1 How2Sign
A. Duarte, S. Palaskar, L. Ventura, D. Ghadiyaram, K. DeHaan, F. Metze, J. Torres, X. Giro-i-Nieto. "How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language." CVPR 2021. https://arxiv.org/abs/2008.08143 [verified: abs "Accepted at CVPR 2021"]
- **Summary.** More than 80 h of ASL with English transcripts and 11 signers. A 3 h Panoptic subset has better 3D. The project page notes the ASL and English may not align perfectly, and offers re-aligned timestamps.
- **Implication for SignNMFlow.** Sentence-level text-to-motion targets on How2Sign are noisy (see 4.4). The SMPL-X comes from the NSA fitter, not from the Panoptic 3D.

### 4.2 CSL-Daily
H. Zhou, W. Zhou, W. Qi, J. Pu, H. Li. "Improving Sign Language Translation with Monolingual Data by Sign Back-Translation." CVPR 2021. https://arxiv.org/abs/2105.12397 [verified: abs]
- **Size.** About 20K samples. SignNMFlow states 20,654 clips and 10 signers.

### 4.3 RWTH-PHOENIX-Weather 2014T
Introduced in Camgoz et al., "Neural Sign Language Translation", CVPR 2018. Dataset page: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ [verified: dataset page]
- **Format.** 25 fps, 210×260 px interpreter crop. SignNMFlow states 8,257 sequences and 9 signers.
- **Implication for SignNMFlow.** At 210×260, hands are tiny and blurred, so hand pseudo-GT (WiLoR) is noisiest here. M3T attributes its Phoenix hand deficit to exactly this. Phoenix hand DTW-PA-JPE improvements (1.38 to 1.07) need a pseudo-GT noise floor: fit twice with perturbed initialisation, or compare with a second fitter.

### 4.4 Sentence-level alignment problems
G. Tanzer, M. Shengelia, K. Harrenstien, D. Uthus. "Reconsidering Sentence-Level Sign Language Translation." arXiv:2406.11049, 2024. https://arxiv.org/abs/2406.11049 [verified: abs]
- **Summary.** A fluent Deaf signer could understand key parts of about one third of isolated How2Sign clips only when given discourse context.
- **Implication for SignNMFlow.** How2Sign sentence-to-motion pairs are partly unpredictable from text alone. This puts a ceiling on both DTW and B-T, and makes very high How2Sign B-T scores suspicious.

### 4.5 Reference points for How2Sign back-translation plausibility
D. Uthus, G. Tanzer, M. Georg. "YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus." arXiv:2306.15162, 2023 (NeurIPS 2023 per proceedings listing). https://arxiv.org/abs/2306.15162 [verified: abs]
P. Rust, B. Shi, S. Wang, N. C. Camgöz, J. Maillard. "Towards Privacy-Aware Sign Language Translation at Scale." ACL 2024. https://arxiv.org/abs/2402.09611 [verified: abs]
- **Numbers.**
  - Video-based ASL-to-English translation on How2Sign: 12.39 BLEU fine-tuned with about 1000 h of extra data (YouTube-ASL).
  - SSVP-SLT reports "over 3 BLEU-4" above the strongest baselines.
- **Implication for SignNMFlow.** SignNMFlow-M's How2Sign back-translation BLEU-4 of 17.49 (and SOKE's 14.48) comes from generated SMPL-X. It exceeds what translation from real video achieved with large-scale pretraining. M3T's ground-truth 3DMM ceiling is 2.04 and Conditional Collapse's is about 2.5.
- **Likely causes.**
  - The SOKE back-translator was trained or evaluated in a way that inflates scores, such as tokenisation or lowercasing, BLEU implementation, or overlap between back-translator training data and the test set.
  - The scores reflect high-frequency n-grams.
- **Required check.** Report B-T on ground-truth SMPL-X with the same back-translator, together with the BLEU tool and its settings.

### 4.6 Frame rate, length and size notes
- **Frame rate.** Neither SOKE nor NSA states the fps or downsampling used for SMPL-X sequences. This was checked in the SOKE main text, its supplement and the NSA HTML.
- **Sequence length and DTW.** SOKE handles length mismatches only through DTW, which hides timing and duration errors.
- **Sample counts.** Stated counts differ: SOKE gives How2Sign 35K, CSL 20K, Phoenix 8K; SignNMFlow gives How2Sign 31K+, CSL 20,654, Phoenix 8,257.
- **Implication for SignNMFlow.** Report fps, maximum length, how length is chosen at inference, and a length-error metric.

---

## 5. Flow-matching prior art directly relevant to SignNMFlow's novelty

### 5.1 Riemannian Flow Matching
R. T. Q. Chen, Y. Lipman. "Flow Matching on General Geometries." ICLR 2024. https://arxiv.org/abs/2302.03660 [verified: abs, journal-ref ICLR 2024]
- **Summary.** Geodesic conditional paths on manifolds, simulation-free on simple geometries such as spheres.
- **Implication for SignNMFlow.** The geodesic part of SignNMFlow is RFM on a product of S³ (a quaternion hemisphere). SignNMFlow cites it.

### 5.2 Metric Flow Matching
K. Kapuśniak, P. Potaptchik, T. Reu, L. Zhang, A. Tong, M. Bronstein, A. J. Bose, F. Di Giovanni. "Metric Flow Matching for Smooth Interpolations on the Data Manifold." NeurIPS 2024. https://arxiv.org/abs/2405.14780 ; https://proceedings.neurips.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract.html [verified]
- **Summary.** Learns interpolants that approximate geodesics by minimising kinetic energy under a data-induced Riemannian metric (a learned correction network around the straight path).
- **Implication for SignNMFlow.** SignNMFlow's "naturalness-guided interpolation" uses the same recipe: a learned deviation from the geodesic, a 3-layer MLP trained by kinetic energy under a data-dependent metric G(x) built from prototypes. Its own appendix says the minimiser is "the geodesic of the data-dependent Riemannian metric".
  - SignNMFlow does not cite MFM. This is a novelty and related-work gap.
  - The fair ablation is MFM, with its Euclidean base, against the sphere-based version.

---

## Table: published numbers on the SOKE SMPL-X benchmark

DTW metrics are lower-is-better. Columns are PA-Body / PA-Hand / JPE-Body / JPE-Hand, and B-T is BLEU-4. Units are unreported in all sources. "–" means not reported.

**Caveats that apply to every row.**
- SignSparK and M3T use their own pseudo-GT, so they are not strictly comparable with the SOKE-GT rows.
- No paper reports seeds or confidence intervals.
- B-T back-translators differ: SOKE/SignNMFlow/GARD use the SOKE (NSA-style) back-translator, and M3T uses its own.

| Method | Dataset | PA-B | PA-H | JPE-B | JPE-H | B-T | Source |
|---|---|---|---|---|---|---|---|
| Prog. Trans.* | H2S | 14.15 | 11.57 | 14.74 | 30.17 | 2.75 | SOKE T1 |
| Text2Mesh* | H2S | 13.99 | 13.47 | 15.50 | 32.97 | 7.51 | SOKE T1 |
| T2S-GPT* | H2S | 11.48 | 6.39 | 12.65 | 18.44 | 11.20 | SOKE T1 |
| NSA | H2S | 7.83 | 7.33 | – | – | 13.12 | SOKE T1 |
| S-MotionGPT* | H2S | 11.23 | 4.39 | 12.41 | 13.74 | 11.45 | SOKE T1 |
| SOKE (no dict.) | H2S | 7.91 | 3.10 | – | – | – | SignSparK T3 |
| SOKE | H2S | 6.82 | 2.35 | 7.75 | 10.08 | 14.48 | SOKE T1 |
| SignFlow | H2S | 6.92 | 2.27 | 7.98 | 10.52 | – | SignNMFlow T2 |
| SignSparK (GF, own GT) | H2S | 7.26 | 2.72 | 6.30 | 11.43 | – | SignSparK T3 |
| M3T w/ SLT (own GT) | H2S | 4.28 | 2.28 | 4.86 | 9.44 | 1.72 (own BT; GT ceiling 2.04) | M3T T1/T2 |
| SignNMFlow | H2S | 6.01 | 2.07 | 6.76 | 9.72 | 16.88 | SignNMFlow T2 |
| SignNMFlow-M | H2S | 5.99 | 2.06 | 6.74 | 9.77 | 17.49 | SignNMFlow T2 |
| Prog. Trans.* | CSL | 15.98 | 12.91 | 16.30 | 32.63 | 3.07 | SOKE T1 |
| Text2Mesh* | CSL | 13.47 | 12.10 | 13.76 | 30.37 | 5.11 | SOKE T1 |
| T2S-GPT* | CSL | 11.94 | 5.93 | 12.32 | 15.43 | 8.94 | SOKE T1 |
| S-MotionGPT* | CSL | 10.81 | 3.78 | 11.58 | 11.31 | 8.82 | SOKE T1 |
| SOKE (no dict.) | CSL | 7.58 | 2.17 | – | – | – | SignSparK T3 |
| SOKE | CSL | 6.24 | 1.71 | 7.38 | 9.68 | 11.30 | SOKE T1 |
| SOKE (re-run by GARD) | CSL | 6.32 | 1.83 | 7.41 | 9.79 | 11.21 | GARD T1 |
| SignFlow | CSL | 6.43 | 1.65 | 7.56 | 8.82 | – | SignNMFlow T1 |
| SignSparK (GF, own GT) | CSL | 7.26 | 2.00 | 6.27 | 10.63 | – | SignSparK T3 |
| SignSparK (SR, retrieval, own GT) | CSL | 4.87 | 1.37 | 4.89 | 6.56 | – | SignSparK T3 |
| GARD (gloss-wise) | CSL | 6.98 | 1.56 | 7.73 | 9.32 | 12.73 | GARD T1 |
| M3T w/ SLT (own GT) | CSL | 3.97 | 1.51 | 4.63 | 9.28 | 5.80 (own BT; GT ceiling 7.11) | M3T |
| SignNMFlow | CSL | 5.10 | 1.30 | 5.73 | 7.43 | 15.14 | SignNMFlow T1 |
| SignNMFlow-M | CSL | 5.31 | 1.33 | 5.94 | 7.73 | 15.23 | SignNMFlow T1 |
| Prog. Trans.* | Phoenix | 13.67 | 11.95 | 15.01 | 31.77 | 4.94 | SOKE T1 |
| Text2Mesh* | Phoenix | 13.48 | 12.06 | 14.04 | 31.64 | 5.81 | SOKE T1 |
| T2S-GPT* | Phoenix | 10.38 | 6.47 | 11.65 | 19.09 | 9.06 | SOKE T1 |
| S-MotionGPT* | Phoenix | 9.45 | 3.41 | 10.42 | 9.08 | 9.68 | SOKE T1 |
| SOKE (no dict.) | Phoenix | 6.16 | 1.85 | – | – | – | SignSparK T3 |
| SOKE | Phoenix | 4.77 | 1.38 | 6.04 | 7.72 | 11.87 | SOKE T1 |
| SOKE (re-run by GARD) | Phoenix | 4.72 | 1.39 | 6.12 | 7.80 | 11.84 | GARD T1 |
| SignFlow | Phoenix | 4.56 | 1.20 | 5.40 | 6.21 | – | SignNMFlow T1 |
| SignSparK (GF, own GT) | Phoenix | 5.24 | 1.52 | 4.40 | 7.10 | – | SignSparK T3 |
| SignSparK (SR, retrieval, own GT) | Phoenix | 3.68 | 1.18 | 3.51 | 4.82 | – | SignSparK T3 |
| GARD (gloss-wise) | Phoenix | 5.03 | 1.21 | 6.69 | 6.90 | 13.16 | GARD T1 |
| M3T w/ SLT (own GT) | Phoenix | 3.41 | 1.46 | 4.02 | 8.55 | 10.58 (own BT; GT ceiling 11.84) | M3T |
| SignNMFlow | Phoenix | 3.85 | 1.07 | 4.37 | 5.57 | 16.15 | SignNMFlow T1 |
| SignNMFlow-M | Phoenix | 3.91 | 1.08 | 4.42 | 5.75 | 14.45 | SignNMFlow T1 |

\* Reimplemented by SOKE. Other numbers on the benchmark:
- SOKE's own scalability ablation reports average DTW-PA-JPE of 3.34 (How2Sign), 2.72 (CSL) and 2.13 (Phoenix).
- SignNMFlow's CSL ablation reports DTW-JPE All 16.86 and DTW-PA-JPE All 8.92.
- Latency in SignNMFlow T7 (H100): SignNMFlow 0.0103 s, SignFlow 0.086 s, SOKE 0.269 s. SignSparK reports 0.01 s per video against 1.55 s for SOKE.

---

## Synthesis (10 bullets)

1. **SignNMFlow's tables are selective.** They leave out GARD (cited but not tabulated), SignSparK's retrieval variant (which beats SignNMFlow on body and JPE metrics on Phoenix and CSL), M3T and SignMask. The fair retrieval-free SOKE comparison is SOKE without dictionary (Phoenix 6.16 / 1.85), not SOKE with retrieval.
2. **Pseudo-GT comparability is the biggest validity issue.**
   - Phoenix/CSL "ground truth" is the SOKE fitter: OSX + WiLoR + MediaPipe refinement, with about 9-11 mm per-vertex hand error even on clean mocap. How2Sign uses the NSA fitter with smoothness and PCA priors.
   - Hand DTW-PA-JPE differences of 0.1-0.3 likely sit inside the fitter noise floor, and possibly inside run-to-run noise (GARD's re-run of SOKE differs by up to 0.08).
   - Report seeds or CIs and a pseudo-GT noise floor.
3. **DTW joint-error metrics reward under-articulated, averaged motion.**
   - The SLRTP 2025 challenge (where SignNMFlow's authors placed 2nd) found this explicitly. PIDiffSign finds generated signing has lower kinetic energy, and Jiang et al. show untuned DTW can correlate negatively with humans.
   - SignNMFlow's ingredients all pull toward the mean: x1-MSE, a prototype-KDE "naturalness" measure, and 5-step sampling where more steps slightly hurt.
   - Adopt SLRTP's total-distance ratio, velocity and energy ratios, and a mean-pose floor.
4. **Back-translation BLEU-4 on How2Sign is implausibly high.**
   - SignNMFlow-M reports 17.49 and SOKE 14.48. Ground-truth-motion ceilings are about 2-2.5 in M3T and Conditional Collapse, and video SLT state of the art is about 12-16 BLEU.
   - Without a ground-truth-motion B-T row and a released back-translator, B-T gains are uninterpretable. Report the ground-truth ceiling, content-free floors (Pelykh et al.), back-translation likelihood (Jiang et al.) and BLEU tool settings.
5. **Content-free baselines are now the expected sanity check:** mean pose, mean sequence, random training sequence and nearest training sentence (Pelykh et al., NeurIPS 2026 workshop). So is a conditioning test: swap the text and measure output change (Conditional Collapse). Phoenix's repetitive weather sentences make nearest-neighbour retrieval a strong floor.
6. **SignNMFlow's interpolation is close prior art to Metric Flow Matching** (NeurIPS 2024): a kinetic-energy-trained correction under a data-induced metric. It is uncited, so related work must position against MFM and RFM, and an MFM-style Euclidean baseline is the right ablation.
7. **Sign phonology offers cheap, testable priors that fit the quaternion framework:**
   - left/right mirror symmetry for symmetric two-handed signs (SGNify, Battison classes);
   - handshape constancy or SLERP within a sign;
   - dominant/non-dominant hand roles (ASL-LEX coding);
   - keyframe anchors at sign onset, mid and offset (SignSparK);
   - geodesic gloss-boundary guidance (GARD).
   FAST or the Moryossef segmenter can supply boundaries without gloss labels.
8. **Non-manual channels are neglected.** SignNMFlow keeps SOKE's 10-d Euclidean face channel (loss weight 0.1, never evaluated). M3T shows this face is nearly static apart from jaw aperture, and that 100-d FLAME with FSQ is feasible. At minimum, report face DTW-VPE; better, add an NMF probe.
9. **Semantic evaluation should not rely on a single opaque back-translator.** Add embedding metrics (SignCLIP, SiLVERScore, which is validated on Phoenix and CSL), lexical distinguishability (ASL-LEX / ASL3DWord gloss discriminability), and stratify results by train-test sentence coverage (Conditional Collapse).
10. **The human evaluation is underpowered.** Five CSL signers, 18 items, forced choice, no ASL or DGS raters and no comprehension measure. Current practice (SignSparK: 16 BSL signers including 6 Deaf; BackTranslation2.0: a protocol co-designed with Deaf experts) and the Deaf-led critique (Desai et al.) call for Deaf native raters per language, comprehension tasks and separate ratings for semantics versus naturalness.

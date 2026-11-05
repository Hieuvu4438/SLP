# Raw literature survey: Sign Language Production (text-to-sign 3D pose/avatar)

Compiled 2026-11-03 for proposal 1. Sources: arXiv abstract pages and HTML full texts, the arXiv API, the Semantic Scholar citation graph for SOKE and MaDiS, and CVPR/NeurIPS/ECVA pages.

How to read this:
- A number is "verified" when I read it from the paper's own HTML table or abstract.
- **[unverified]** marks things I recall or saw only second-hand.
- "(X-reported)" means paper X reports the number for another method, often after retraining it.

Anchor papers: SOKE (2411.17799, ICCV 2025), MaDiS (2601.19577) and SignSparK (2603.10446, ECCV 2026). The other papers are positioned relative to these three.

---

## (A) Taxonomy of approaches

### A.1 Families

1. **Regression / autoregressive continuous decoders (2020-2022).** Examples are Progressive Transformers, Adversarial training, MDN and Mixed SIGNals. They are trained with L1/L2 losses on 2D-lifted skeletons from Phoenix-2014T. Their known failure mode is *regression to the mean*: under-articulated hands. Later families mostly exist to fix this.
2. **Non-autoregressive / length-regulated.** Examples are NSLP-G and NAR-KD (2208.06183), plus NAT-AT/NAT-EA from MM 2021. They address error propagation and false decoding starts.
3. **Diffusion on poses (gloss- or text-conditioned).**
   - G2P-DDM: discrete diffusion over VQ pose codes.
   - Sign-IDD: 4D bone direction and length representation.
   - Neural Sign Actors: graph diffusion on SMPL-X.
   - MS2SL: text or speech input.
   - Text2SignDiff: latent diffusion, gloss-free.
   - GARD: gloss-wise autoregressive diffusion.
   - Sign-D2C: diffusion that infills transitions between signs.
   - Phonology-guided MDM (2603.17388).
4. **Discrete tokens + language model (the "signs as tokens" family).**
   - T2S-GPT: dynamic-length VQ.
   - SOKE: decoupled body/hand VQ, mBART, multi-head decoding, dictionary retrieval.
   - M3T: FSQ tokens for face, body and hands (SMPL-FX), plus an auxiliary SLT loss.
   - SeRV: semantic-aligned RVQ with a hierarchical GPT.
   - SignGPT and SignFLIP: unified SLT+SLG in one LLM.
   - TEAM-Sign: stepwise prompting of an LLM.
   - SignLLM: multilingual, trained on Prompt2Sign.
5. **Masked / parallel token generation.**
   - MaDiS: masked-diffusion LM on Qwen3-0.6B.
   - SignMask (ACM TOMM 2026): Sign-M masked transformer on SMPL-X tokens.
   - BLANC: block-wise causal, parallel denoising within each block, multi-rate VQ.
   - SignPR (CVPR 2026): block-wise causal VQ-diffusion.
6. **Flow matching.**
   - SignSparK: conditional flow matching (CFM) from sparse keyframes.
   - HybridSign (ACL 2026): causal autoregressive (AR) generation plus flow refinement, real-time.
   - SignNMFlow: manifold or geodesic flow matching.
   - SignFlow (Khan et al., ACM 2025, DOI 10.1145/3716553.3750765) [details unverified].
7. **Retrieval / stitching / hybrid retrieve-and-refine.**
   - Spoken2Sign: Text2Gloss, then a gloss-3D dictionary, then a sign connector.
   - Sign Stitching: a 7-step normalise/stitch/filter pipeline.
   - The SOKE retrieval module.
   - SignRR: retrieve, then refine with a part-aware RVQ-VAE.
   - The SLRTP-2025 challenge winner, which used retrieval plus a pretrained LM.
8. **Video / photorealistic rendering (pose-to-video or text-to-video).**
   - SignGAN (Signing at Scale), SignGen (ECCV 2024) and PGMM (ECCV 2024).
   - SignViP (NeurIPS 2025): FSQ multi-condition tokens.
   - SignAligner: the PHOENIX14T+ dataset.
   - SignSplat and Stable Signer V3: Gaussian or primitive splatting.
   - SignRefine: adapts a video foundation model (DiT) with hand and face adapters.
   - Diverse Signer Avatars: latent diffusion model (LDM) with manual and non-manual aggregation.
   - Stable Signer.
   - The Apple CHI'25 non-manual-marker system.
9. **Evaluation-focused work.**
   - SiBLEU and SiCLIP (from MaDiS).
   - BackTranslation2.0 (ECCV 2026): an agentic, linguistically grounded metric.
   - "Beyond BLEU" (2609.03734): an LLM-QA metric for SLT.
   - The SLRTP-2025 standardized evaluator.
   - SignCLIP (EMNLP 2024).

**RL and preference optimization.** I found **no** paper that applies RL or preference optimization (DPO/GRPO with a back-translation or CLIP reward) to SLP. The nearest work is:
- SignLLM's "RL loss / Priority Learning Channel", which is really a data-sampling heuristic, and its authors say the gain was not statistically significant.
- GRPO used in *translation*: SignRAG 2610.11371 and the attention-steered VLM SLT paper 2608.00235.
- GRPO for general text-to-motion: UniMo 2601.12126.

This gap looks open.

### A.2 Paper table

Representations: Skel = 2D/3D keypoints; SMPL-X = parametric body model; FLAME = face model. "Verified" means the arXiv ID and venue come from arXiv comments or proceedings pages.

| # | Title (short) | Venue / year | arXiv | Repr. | Generator | Key idea |
|---|---|---|---|---|---|---|
| 1 | Progressive Transformers (Saunders) | ECCV 2020 | 2004.14874 | 3D skel (lifted) | AR transformer, regression | Counter decoding; gloss or text to pose |
| 2 | Adversarial Training for Multi-Channel SLP | BMVC 2020 | 2008.12405 | 3D skel | AR + conditional discriminator | Fights regression to the mean |
| 3 | Continuous 3D Multi-Channel SLP via PT + MDN | IJCV 2021 [venue unverified] | 2103.06982 | 3D skel | AR + mixture density | Multimodal output distribution |
| 4 | Mixed SIGNals (MoMP) | ICCV 2021 [venue unverified] | 2107.11317 | skel | Mixture of motion primitives | Sign split into reusable primitives |
| 5 | Signing at Scale (SignGAN) | CVPR 2022 [venue unverified] | 2203.15354 | skel to video | FS-Net + GAN | Learns co-articulation; photoreal |
| 6 | Text2Sign (Stoll) | IJCV 2020 | none found | skel to video | NMT + GAN | Text, gloss, pose, video pipeline |
| 7 | NSLP-G (Hwang) | BMVC 2021 [unverified] | none found | skel | NAR, Gaussian latent | Non-autoregressive |
| 8 | NAR-SLP via Knowledge Distillation | preprint (ECCV'23 submission) | 2208.06183 | skel | NAR + length regulator | Fixes NSLP-G false decoding initiation |
| 9 | Ham2Pose | CVPR 2023 [venue unverified] | 2211.13613 | 2D skel | Iterative refinement | HamNoSys to pose |
| 10 | G2P-DDM | AAAI 2024 | 2208.09141 | skel, VQ | Discrete diffusion | Pose VQ codes plus diffusion |
| 11 | SignDiff (Fang) | [venue unverified] | 2308.16082 | skel to video | Diffusion + FR-Net | First ASL baseline: How2Sign B-T BLEU-4 17.19 / 12.85 (dev/test) |
| 12 | Gloss-free SLP with discrete representation (SignVQNet) | FG 2024 [unverified] | 2309.12179 | skel, VQ | AR transformer | Gloss-free VQ |
| 13 | SignAvatars dataset/benchmark | ECCV 2024 | 2310.20436 | SMPL-X | Benchmark + baseline | 70K videos, 153 signers, 8.34M frames |
| 14 | Neural Sign Actors (NSA) | CVPR 2024 | 2312.02702 | SMPL-X (4D) | Graph-NN diffusion | Anatomical GNN diffusion; How2Sign |
| 15 | Spoken2Sign: simple baseline with 3D avatars (Zuo) | ECCV 2024 | 2401.04730 | SMPL-X | T2G + dictionary + connector | Gloss-3D dictionary retrieval (SOKE precursor) |
| 16 | A Data-Driven Representation for SLP (Walsh) | FG 2024 | 2404.11499 | skel, VQ | Transformer over codebook | Learned sign "phonemes" |
| 17 | SignAvatar (Dong) | FG 2024 | 2405.07974 | SMPL-X | CVAE | 3D reconstruction and generation |
| 18 | Sign Stitching (Walsh) | BMVC 2024 | 2405.07663 | skel to video | Dictionary + 7-step stitching | Retrieval answer to regression to the mean |
| 19 | SignLLM / Prompt2Sign | arXiv 2024 (v3) | 2405.10718 | compressed 3D pose | Multilingual transformer | 8 sign languages; ASL BLEU-4 15.03 (self-reported) |
| 20 | T2S-GPT | ACL 2024 | 2406.07119 | skel, DVQ | GPT | Dynamic-length VQ plus duration |
| 21 | MS2SL | ACL 2024 Findings | 2407.12842 | skel | Sequence diffusion | Text or speech input; joint embedding |
| 22 | SignGen (Qi) | ECCV 2024 | [no arXiv confirmed] | video | Latent diffusion | Train/inference condition mismatch (per SignViP) |
| 23 | Pose-Guided Fine-Grained SL Video Gen. (PGMM) | ECCV 2024 | 2409.16709 | pose to video | Flow warp + fusion | Pose2Vid |
| 24 | Discrete to Continuous (Sign-D2C) | arXiv 2024 | 2411.16810 | skel | Conditional diffusion | Transition-frame infilling |
| 25 | **SOKE**, Signs as Tokens | ICCV 2025 | 2411.17799 | SMPL-X, decoupled VQ | mBART + multi-head decoding | Multilingual (ASL/CSL/DGS); retrieval |
| 26 | Sign-IDD | AAAI 2025 | 2412.13609 | 4D bone (direction + length) | Attribute-controllable diffusion | Iconicity disentanglement |
| 27 | SignSplat | arXiv 2025 | 2505.02108 | SMPL-X + 3DGS | Rendering | Gaussian-splat signer |
| 28 | SIGNER | ECCV 2026 | 2506.07460 | 3D joints + face (529-d) | PVQ-VAE + denoiser | Time-resolved gloss conditioning (local temporal fusion, LTF) |
| 29 | SLP as data augmentation for SLT | arXiv 2025 | 2506.09643 | skel / SignGAN / SignSplat | n/a | Up to 19% SLT gain |
| 30 | SignAligner / PHOENIX14T+ | arXiv 2025 | 2506.11621 | Pose + HaMeR + SMPLer-X | Joint co-generation + video | Multi-representation co-generation |
| 31 | SignViP | NeurIPS 2025 | 2506.15980 | Multi-condition FSQ tokens to video | Video diffusion + token translator | Fine-grained pose and 3D-hand conditions |
| 32 | HybridSign | ACL 2026 | 2507.09105 | 2D pose | Causal AR + flow refinement | Real-time; How2Sign BLEU-1/4 30.12 / 6.48, 10.17 FPS |
| 33 | TEAM-Sign (Teach Me Sign) | ICIP 2025 | 2507.10972 | pose tokens | Fine-tuned LLM | Stepwise prompting |
| 34 | SLRTP-2025 SLP Challenge | CVPRW 2025 | 2508.06951 | skel | n/a | Winner: BLEU-1 31.40, DTW-MJE 0.0574 (retrieval + pretrained LM) |
| 35 | Quaternion + contrastive PT | arXiv 2025 | 2508.14574 | bone quaternions | PT + contrastive loss | Reduces signer noise |
| 36 | HandMDM: text-driven 3D hand motion | CVPR 2026 | 2508.15902 | MANO | Diffusion | Hand-motion descriptions generated by an LLM; BOBSL3DT |
| 37 | Diverse Signer Avatars (manual + non-manual) | arXiv 2025 | 2508.15988 | video | LDM | Signer diversity; YouTube-SL-25 |
| 38 | Text2SignDiff | arXiv 2025 | 2509.10845 | latent pose (SignVAE) | Latent diffusion | Gloss-free; PHOENIX14T and How2Sign |
| 39 | Stable Signer | [venue unverified: S2 says ACL 2025] | 2512.04048 | video | SLUL + SLP-MoE | Hierarchical end-to-end video |
| 40 | SignMask | ACM TOMM 2026 [no arXiv found] | n/a | SMPL-X tokens | Masked transformer | Structure-aware tokenizer |
| 41 | **MaDiS** | arXiv 2026 (venue not stated) | 2601.19577 | SMPL-X tokens | Masked-diffusion LM (Qwen3-0.6B) | Tri-level pretraining; SiBLEU and SiCLIP |
| 42 | **SignSparK** (+ FAST segmentor) | ECCV 2026 | 2603.10446 | SMPL-X + MANO + FLAME | Conditional flow matching | Sparse keyframes; ~100x faster than SOKE |
| 43 | Phonology-guided SL motion generation | arXiv 2026 | 2603.17388 | SMPL-X | MDM | ASL-LEX attribute conditioning |
| 44 | **M3T** | BMVC 2026 | 2603.23617 | SMPL-FX (FLAME expression), FSQ | AR transformer + SLT auxiliary | Non-manual features; face codebook use 78.8% to 99.0% |
| 45 | SignVerse-2M | arXiv 2026 | 2605.01720 | DWPose | SignDW transformer | 2M clips, 55+ sign languages |
| 46 | GARD | arXiv 2026 | 2606.21234 | 3D joints | Gloss-wise AR diffusion | Inter-gloss guidance + global harmonizer |
| 47 | VAE design for latent-diffusion SLP | arXiv 2026 | 2606.22959 | latent | LDM | Latent properties predict B-T better than reconstruction |
| 48 | BackTranslation2.0 | ECCV 2026 | 2606.28673 | n/a | Metric | Agentic, linguistically grounded metric (BSL) |
| 49 | Text2Sign single-GPU video baseline | arXiv 2026 | 2607.13164 | video | Diffusion | Low-compute baseline |
| 50 | SLG to humanoid retargeting | arXiv 2026 | 2607.17769 | SMPL-X | n/a | Collision mitigation; VLM critic |
| 51 | SignRR | BMVC 2026 | 2608.28568 | skel, RVQ | Retrieve + refine | Real dictionary motion refined globally |
| 52 | SeRV | arXiv 2026 | 2609.05742 | SMPL-X, RVQ | Hierarchical GPT | Semantic-aligned RVQ; YouTube3D (236K pairs) |
| 53 | SignRefine / NVSign | ECCV 2026 (per S2) | 2609.08496 | 2D keypoints to video | Video DiT + adapters | Up to 30% hand-precision gain |
| 54 | SignGPT | arXiv 2026 | 2609.21709 | part-hierarchical pose tokens | Shared LM, multi-token prediction | Unified SLT+SLG; 12 Deaf signers |
| 55 | SignFLIP | EMNLP 2026 Findings | 2609.35225 | 2D pose | mT5 symmetric unified model | Stage-wise alignment; YouTube-ASL + CSL-News |
| 56 | SignNMFlow | arXiv 2026 | 2609.33339 | rotations on manifold | Manifold flow matching | Geodesic plus learnable deviation |
| 57 | BLANC (MRVQ + Latent Articulatory Clock) | arXiv 2026 | 2610.24109 | SMPL-X, multi-rate VQ | Block-wise adaptive diffusion | Variable block durations |
| 58 | Stable Signer V3 | arXiv 2026 | 2610.24926 | 4D primitive splatting | Rendering | Face and body splatting avatar |
| 59 | SignPR | CVPR 2026 | [arXiv not found] | region VQ | Progressive VQ-diffusion | Block-wise causal |
| 60 | Focal-General Diffusion with Semantic Consistent Guidance | CVPR 2026 | [unverified] | n/a | Diffusion | Title seen on the CVPR schedule only |

**Papers citing SOKE (from Semantic Scholar, 30 indexed; the list is incomplete).** All are in the table above except:
- AHA-JEPA (IEEE Access 2026)
- DigitCode (2608.03127)
- HandX (2603.28766)
- SignAgent (2603.19059)
- Direct and Simultaneous Translation between Sign Languages (2605.20588, 2609.35608)
- RoboSTAR (2609.32250)
- SignBot (ICRA 2025, 2505.24266)
- Q-BridgeNet (ECCV 2026, 2607.11215)
- How2Sign-Synth3D (CVPR 2026 Findings)

**Papers citing MaDiS (Semantic Scholar):** SeRV, BLANC, BackTranslation2.0, M3T, and Kinematics-Centric Continuous Sign Retrieval (2610.05306).

---

## (B) Benchmarks, metrics and SOTA numbers

### B.1 Datasets

| Dataset | Language | Size | 3D? | Notes |
|---|---|---|---|---|
| Phoenix-2014T | DGS | ~8K sentences; 1,066 glosses (GARD: 7,096 / 519 / 642 split) | no; SMPL-X fitted by SOKE | Weather domain; 9 signers [unverified] |
| CSL-Daily (2105.12397, CVPR 2021) | CSL | ~20K sentences, ~2K glosses | no; SMPL-X fitted by SOKE | Interpreters, not Deaf signers (SignFLIP) |
| How2Sign (2008.08143, CVPR 2021) | ASL | ~35K pairs (SOKE); ~80 h [unverified] | 3D only for a Panoptic subset | Instructional domain |
| YouTube-ASL (2306.15162, NeurIPS 2023 D&B) | ASL | ~610K clips, 984 h (SignFLIP) | no; SeRV recovered "YouTube3D", 236,537 pairs, 295 h | Open domain |
| BOBSL (2111.03635) | BSL | ~1,400 h [unverified] | no | Weakly aligned subtitles; SignSparK trains on 10% |
| CSL-News (Uni-Sign, 2501.15187) | CSL | 1,985 h; 751,320 clips (722,711 used) | no | TV news |
| SignAvatars (2310.20436, ECCV 2024) | ASL, plus HamNoSys and word-level sources | 70K videos, 153 signers, 8.34M frames | SMPL-X | Multi-prompt: text, HamNoSys, gloss [source composition partly unverified] |
| Prompt2Sign (2405.10718) | 8 languages: ASL, DGS, DSGS, LSF-CH, LIS-CH, LSA, KSL, TSL | ~200 h, 40 signers | OpenPose, compressed 3D | Prompt templates |
| SignVerse-2M (2605.01720) | 55+ languages | ~2M DWPose clips | 2D | Pose-native |
| OpenASL | ASL | 97K (SignFLIP) | no | |
| NMFs-CSL | CSL (isolated) | n/a | no | Signs that differ only by non-manual features; used by M3T |
| Vietnamese SL | VSL | VSL400: 400 glosses, 74,259 clips, 28 signers, 3 views (Authorea preprint 2026, isolated recognition only) | no | **No public continuous text-to-VSL dataset found.** Only small NMT corpora (10K to 60K text-to-VSL-gloss pairs, JCC 2023) and HamNoSys/SiGML avatar work (2018) |

### B.2 Metrics

**Back-translation (B-T) BLEU-1/4, ROUGE-L and WER.** A sign-to-text (or sign-to-gloss) model is trained on ground-truth poses and then run on generated poses.
- Sign-IDD uses NSLT (Camgoz 2018) to gloss.
- SignSparK and M3T use an SL-Transformer trained on GT 3DMM parameters.
- SignGPT uses its own model as the evaluator.
- SOKE does **not** name its back-translator.

**DTW-JPE and DTW-PA-JPE (the SOKE protocol).** DTW-aligned joint position error, reported separately for body and hands. The PA variant is Procrustes-aligned. MaDiS reports DTW-JPE only.

**DTW-MJE.** Used in the SLRTP challenge and SignFLIP (normalized keypoints, about 1e-4 scale).

**MPJPE / MPJAE / FID.** Sign-IDD and SignRR report MPJPE and FID on poses. For video work (SignViP), the metrics are FID, CLIP-FID, FVD, SSIM, PSNR, LPIPS and Hand-SSIM.

**SiBLEU (MaDiS).** BLEU-4 between generated and GT *sign token* sequences, computed per body part.

**SiCLIP (MaDiS).** A CLIP-style text-sign retriever with decoupled encoders and a CiCo-style fine-grained loss. The score is sign-to-text R@1. Retriever quality, R@1/R@5:
- CSL-Daily: 59.10 / 95.66
- Phoenix: 68.85 / 95.64
- How2Sign: 41.33 / 88.43

MaDiS reports that SiBLEU and SiCLIP correlate strongly with signer ratings and that DTW correlates weakly (Spearman; exact values not extracted).

**Newer metrics.** NMFs-CSL accuracy (M3T), facial PA-VPE (SignSparK), DTW-VPE for the face (M3T), and peak velocity and jerk (SIGNER). BackTranslation2.0 scores grammar, phonology, fluency and fidelity.

### B.3 SOTA under the SOKE protocol (SMPL-X fitted by SOKE; DTW in SOKE units)

Values are Body / Hand. Lower is better for DTW. Higher is better for B-T BLEU-4.

| Method | How2Sign DTW-PA-JPE | How2Sign DTW-JPE | How2Sign B-T B4 | CSL-Daily DTW-PA-JPE | CSL-Daily DTW-JPE | CSL-Daily B-T B4 | Phoenix DTW-PA-JPE | Phoenix DTW-JPE | Phoenix B-T B4 | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Prog. Trans.* | 14.15 / 11.57 | 14.74 / 30.17 | 2.75 | 15.98 / 12.91 | 16.30 / 32.63 | 3.07 | 13.67 / 11.95 | 15.01 / 31.77 | 4.94 | SOKE Tab. 1 |
| Text2Mesh* | 13.99 / 13.47 | 15.50 / 32.97 | 7.51 | 13.47 / 12.10 | 13.76 / 30.37 | 5.11 | 13.48 / 12.06 | 14.04 / 31.64 | 5.81 | SOKE |
| NAR | 13.94 / 11.80 | n/a | 5.75 | n/a | n/a | n/a | n/a | n/a | n/a | SOKE |
| Adv. Train. | 13.78 / 11.17 | n/a | 6.21 | n/a | n/a | n/a | n/a | n/a | n/a | SOKE |
| T2S-GPT* | 11.48 / 6.39 | 12.65 / 18.44 | 11.20 | 11.94 / 5.93 | 12.32 / 15.43 | 8.94 | 10.38 / 6.47 | 11.65 / 19.09 | 9.06 | SOKE |
| NSA | 7.83 / 7.33 | n/a | 13.12 | n/a | n/a | n/a | n/a | n/a | n/a | SOKE |
| S-MotionGPT* | 11.23 / 4.39 | 12.41 / 13.74 | 11.45 | 10.81 / 3.78 | 11.58 / 11.31 | 8.82 | 9.45 / 3.41 | 10.42 / 9.08 | 9.68 | SOKE |
| SOKE (no dict.) | 7.91 / 3.10 | n/a | n/a | 7.58 / 2.17 | n/a | n/a | 6.16 / 1.85 | n/a | n/a | SignSparK Tab. 3 |
| **SOKE** (with retrieval) | 6.82 / 2.35 | 7.75 / 10.08 | 14.48 | 6.24 / 1.71 | 7.38 / 9.68 | 11.30 | 4.77 / 1.38 | 6.04 / 7.72 | 11.87 | SOKE |
| SignSparK (no keyframes) | 7.26 / 2.72 | 6.30 / 11.43 | n/a | 7.26 / 2.00 | 6.27 / 10.63 | n/a | 5.24 / 1.52 | 4.40 / 7.10 | n/a | SignSparK |
| SignSparK (with retrieval) | n/a | n/a | n/a | 4.87 / 1.37 | 4.89 / 6.56 | n/a | 3.68 / 1.18 | 3.51 / 4.82 | n/a | SignSparK |
| M3T w/o SLT | 4.52 / 2.42 | 5.02 / 8.94 | 1.58 | 4.13 / 1.59 | 4.77 / 9.94 | 4.74 | 3.54 / 1.54 | 4.17 / 9.14 | 10.16 | M3T Tab. 1-2 |
| **M3T w/ SLT** | 4.28 / 2.28 | 4.86 / 9.44 | 1.72 (GT 2.04) | 3.97 / 1.51 | 4.63 / 9.28 | 5.80 (GT 7.11) | 3.41 / 1.46 | 4.02 / 8.55 | 10.58 (GT 11.84) | M3T |
| GARD (gloss-based) | n/a | n/a | n/a | 6.98 / 1.56 | 7.73 / 9.32 | 12.73 (ROUGE 36.76) | 5.03 / 1.21 | 6.69 / 6.90 | 13.16 (ROUGE 37.47) | GARD Tab. 1 |

\* = retrained by the SOKE authors.

SignSparK's B-T GT ceilings under its evaluator: How2Sign 3.37, CSL-Daily 6.83, Phoenix 14.77. Its keyframe-to-pose (KF2P) mode reaches 2.99, 5.85 and 11.70.

GARD's own SOKE\* row: Phoenix 11.84 B4, PA 4.72 / 1.39; CSL-Daily 11.21 B4.

**MaDiS protocol (DTW-JPE Body / Hand; SiBLEU Body / Hand; SiCLIP R@1).** From MaDiS Table 1. † = uses an external dictionary.

| Method | CSL-Daily | Phoenix-2014T | How2Sign |
|---|---|---|---|
| SOKE (mBART)† | 7.38 / 9.68; 2.59 / 2.14; 23.64 | 6.04 / 7.72; 2.71 / 1.85; 34.30 | 7.75 / 10.08; 2.52 / 1.74; 16.12 |
| SOKE (Qwen3)† | 7.29 / 9.59; 2.71 / 2.27; 25.05 | 6.09 / 7.65; 2.38 / 1.98; 38.53 | 7.18 / 10.09; 2.57 / 1.97; 18.48 |
| MoMask++ | 8.93 / 10.57; 1.53; 22.62 | 8.16 / 9.96; 1.49; 37.88 | 9.06 / 11.34; 1.58; 20.39 |
| MDLM baseline | 7.12 / 9.57; 3.17 / 2.55; 29.17 | 5.98 / 7.58; 2.36 / 2.14; 44.25 | 7.02 / 10.16; 2.73 / 1.79; 25.78 |
| **MaDiS** | **5.99 / 7.96; 4.87 / 4.08; 38.61** | **4.97 / 6.59; 4.54 / 3.33; 57.94** | **6.59 / 9.45; 4.41 / 3.46; 34.27** |
| SeRV (SeRV-reported) | n/a | n/a | 6.51 / 9.22; SiBLEU-4 4.12 |

Other MaDiS figures:
- Throughput is 132 tok/s, about 40% above SOKE with KV cache. A 400-frame video takes about 0.76 s.
- User study (5 CSL signers): MaDiS 6.9, SOKE 4.6, GT 8.4.
- On YouTube3D, SeRV reports SOKE at 5.42 / 7.44 and SeRV at 4.98 / 6.70.

**Inference speed (from SOKE and SignSparK).**
- SOKE: 1.55 s per video with multi-head decoding and retrieval (sequential decoding: 3.26 s).
- SignSparK: about 0.01 s per video with 10 steps (about 100x faster).
- HybridSign: 10.17 FPS; time to first frame 5.90 s.

**Other SOKE-protocol-like re-evaluations that disagree with the numbers above:**
- **BLANC** reports SOKE (no dict.) on Phoenix at PA 8.24 / 1.64, B4 5.13. BLANC itself scores PA 7.58 / 1.48, B4 9.85, ROUGE 25.36. On CSL-Daily, BLANC scores B4 2.13 against SOKE's 2.66.
- **SignGPT** reports SOKE on Phoenix at DTW-JPE 2.58 / 5.89.
- **SIGNER** retrained everyone on a 529-d joint+face representation. Under that setup SOKE gets B4 4.09 on CSL-Daily and 5.52 on Phoenix, while SIGNER gets 15.60 and 11.46.

These numbers are not mutually comparable across papers (see C.3).

### B.4 Gloss/keypoint Phoenix-2014T protocol (NSLT back-translation, test set)

Columns: BLEU-1 / BLEU-4 / ROUGE / WER.

| Method | B1 / B4 / ROUGE / WER | Source |
|---|---|---|
| GT | 29.76 / 11.93 / 28.98 / 71.94 | Sign-IDD |
| PT (original, G2P / T2P) | B4 10.43 / 10.51 | Adversarial paper (2008.12405) |
| Adversarial (conditional, G2P) | B4 11.70, ROUGE 33.64 | 2008.12405 |
| PT-base (retested) | 9.47 / 0.59 / 8.88 / 98.36 | Sign-IDD |
| NAT-EA | 15.12 / 6.66 / 19.43 / 82.01 | Sign-IDD |
| GEN-OBT | 23.08 / 8.01 / 23.49 / 81.78 | Sign-IDD |
| G2P-DDM | 16.11 / 7.50 / n/a / 77.26 | Sign-IDD |
| GCDM | 22.03 / 7.91 / 23.20 / 81.94 | Sign-IDD |
| Sign-IDD | 24.80 / 9.08 / 26.58 / 76.66 (FID 2.46, MPJPE 47.19) | Sign-IDD |
| SignRR | 28.78 / 10.96 / 28.15 / 80.98 (GT 13.43 B4 under the SignRR evaluator) | SignRR |
| SLRTP-25 winner | B1 31.40, DTW-MJE 0.0574 | 2508.06951 |

SignRR on CSL-Daily: B4 1.57 against a GT ceiling of 5.06.

**Video SLP (SignViP, NeurIPS 2025), video B-T BLEU-4.**
- RWTH: GT 11.90; SignViP 8.65; SignGen 0.51; SignGAN 3.16.
- How2Sign: SignViP 5.04.
- FVD (RWTH): SignViP 1025 against SignGAN 1431.

### B.5 SLT (sign-to-text) SOTA relevant as evaluators or pretraining (test BLEU-4)

| Method | Venue | arXiv | Phoenix-2014T | CSL-Daily | How2Sign |
|---|---|---|---|---|---|
| GFSLT-VLP | ICCV 2023 | 2307.14768 | 21.44 | 11.00 | n/a |
| Sign2GPT | ICLR 2024 | 2405.04164 | 22.52 | 15.40 (Uni-Sign table; SignFLIP's table gives 22.52 for CSL-Daily, which looks like a copy error) | n/a |
| SignLLM (Gong et al., "LLMs are Good SL Translators") | CVPR 2024 | 2404.00925 | 23.40 | 15.75 | n/a |
| SpaMo | arXiv 2024 [venue unverified] | 2408.10593 | 24.32 | 20.55 | n/a |
| YouTube-ASL baseline (T5) | NeurIPS 2023 D&B | 2306.15162 | n/a | n/a | 12.4 |
| SSVP-SLT | ACL 2024 | 2402.09611 | n/a | n/a | 15.5 |
| Uni-Sign (pose / pose+RGB) | ICLR 2025 | 2501.15187 | not reported | 25.61 / 26.36 | 14.5 / 14.9 |
| Geo-Sign (SignFLIP-reported) | [unverified] | [unverified] | n/a | 27.42 | 15.1 |
| ShuBERT (SignFLIP-reported) | [unverified] | [unverified] | n/a | n/a | 16.2 |
| SignGPT (unified) | arXiv 2026 | 2609.21709 | 22.95 | n/a | 16.42 (with TSA: 16.77) |
| SignFLIP (unified) | EMNLP 2026 Findings | 2609.35225 | n/a | 26.10 | 15.1 |

Other useful evaluator and pretraining assets:
- SignCLIP (2407.01264, EMNLP 2024)
- Scaling SLT (2407.11855, Google)
- "Lost in Translation, Found in Context" (2501.09754, CVPR 2025, BOBSL)
- Uni-Sign's CSL-News
- "Beyond BLEU" (2609.03734): an LLM-QA SLT metric that is 6 to 7 times more paraphrase-invariant than BLEU-4.

---

## (C) Limitations and open problems (with citations)

### C.1 Non-manual features (face, mouthing, gaze) are mostly missing
- SOKE, MaDiS, BLANC and SignRR generate body and hands only. SignRR explicitly says it does not capture mouthings or eye movements (2608.28568).
- M3T (2603.23617) argues that SMPL-X's face space is insufficient. It introduces SMPL-FX and fixes codebook collapse (face code usage 78.8% to 99.0%). It still lists lip articulation as the most common residual error. It also says a face comparison against prior work is "undefined" because no baseline produces comparable face parameters.
- SignSparK (2603.10446) reports facial PA-VPE only as a stand-alone reference.
- SeRV (2609.05742) says facial expression and emotion are under-modelled.
- Apple's CHI'25 paper (2502.05661) found user dissatisfaction due to the "absence of facial cues and body language."

### C.2 Regression to the mean and under-articulation
- The problem was first named by Saunders (2008.12405, 2103.06982) and NAR-KD (2208.06183).
- It is still present: SignSparK's motivation is the trade-off between regression to the mean and the disjointed transitions of retrieval.
- SignFLIP (2609.35225) notes generated hands have "limited amplitude and tend toward mean poses."
- M3T reports hands drifting to the body centre, compressing the signing space.

### C.3 Evaluation is unreliable and not comparable
- **Back-translators are different and often unnamed** (SOKE). The GT ceilings differ wildly:
  - How2Sign B-T BLEU-4 GT is 3.37 under SignSparK and 2.04 under M3T. Yet SOKE reports 14.48 for its *generated* How2Sign output.
  - Phoenix GT is 11.93 (Sign-IDD), 13.43 (SignRR), 14.77 (SignSparK) and 11.84 (M3T).
- **Re-evaluations of SOKE disagree by a factor of 1.5 to 2 in DTW.** Phoenix PA-body is 4.77 (SOKE), 6.16 (SignSparK no dict.), 8.24 (BLANC no dict.) and 2.58 DTW-JPE body (SignGPT).
- M3T notes "each method is evaluated against its own pseudo-GT."
- MaDiS finds DTW correlates weakly with signer ratings and is computationally costly. This motivates SiBLEU and SiCLIP.
- BackTranslation2.0 (2606.28673) says existing metrics are "simplistic and poorly aligned with human judgements."
- "Beyond BLEU" (2609.03734) shows BLEU-4 gains in SLT do not imply better sign understanding. This undermines B-T as an SLP metric too.
- SLRTP-2025 (2508.06951) states that "the lack of standardized evaluation metrics … hampers meaningful comparisons."
- SignGPT says "smooth and visually plausible motion does not necessarily guarantee semantic accuracy," and notes its own evaluator is not architecture-independent.
- User studies are tiny:
  - SOKE: 5 ASL and 4 CSL signers, no DGS signers.
  - MaDiS: 5 CSL signers.
  - SignFLIP: 2 signers.
  - M3T: non-fluent raters only.
  - BLANC: none.
  - SignSparK, with 16 signers, is an exception.

### C.4 3D ground truth is pseudo and noisy
- SOKE: CSL-Daily and Phoenix have no 3D, so SMPL-X is fitted and "some reconstruction errors are inevitable."
- SignSparK attributes incomplete hand closure to imperfect single-view reconstruction.
- BLANC and M3T rely on reconstructed motion. M3T reconstructions degrade most on Phoenix.
- Sign-IDD lifts OpenPose 2D to 3D.

### C.5 Data scarcity and low-resource languages
- Everything is benchmarked on three languages (ASL, CSL, DGS), with BSL in SignSparK and M3T-adjacent work.
- SOKE plans to extend to BSL and ISL.
- SignLLM says low-resource languages "make training harder." KSL has only 700 training clips in Prompt2Sign.
- SignFLIP notes most data is not produced by Deaf signers.
- **Vietnamese Sign Language** has no public continuous sentence-level text-to-sign corpus with 3D annotation, and no SLP paper in the arXiv/S2 results. VSL400 is isolated recognition only, and VSL machine-translation work stops at the gloss level.
- Scale-oriented resources could seed low-resource transfer:
  - SignVerse-2M: 55+ languages, 2D only.
  - YouTube3D (SeRV): 236K ASL pairs.
  - CSL-News.

### C.6 Multilinguality
- SOKE and SignSparK are multilingual. SignSparK found that without a language token the model confuses ASL and BSL, since both use English text.
- Cross-sign-language transfer and sign-to-sign translation are new directions (2605.20588, 2609.35608).

### C.7 Dictionary or retrieval dependence
- SOKE's best numbers need external isolated-sign dictionaries. Without them, How2Sign PA rises from 6.82 / 2.35 to 7.91 / 3.10.
- MaDiS marks SOKE † for dictionary use.
- SignSparK has no retrieval results on How2Sign.
- SignRR: unseen glosses are mapped by edit distance, which "limits open-vocabulary use."
- SIGNER and GARD need a text-to-gloss model and gloss annotations. SIGNER says adding a new language requires training text-to-gloss.

### C.8 Long-sequence coherence, co-articulation and timing
- GARD says whole-sentence methods "drift over time, blur hand motion" as sentences get longer.
- SIGNER finds globally fused conditioning causes wrong lexical order.
- BLANC targets variable durations and the different temporal rates of hands and body.
- Sign Stitching and Sign-D2C target transitions.
- M3T's fixed 4x temporal downsampling hurts fast fingerspelling.

### C.9 Inference speed
- SOKE AR decoding: 1.55 to 3.26 s per video.
- MaDiS has a speed/quality trade-off: 10 steps exceed 300 tok/s at lower quality.
- GARD: "inference speed remains a challenge for real-time interaction" (1,000 diffusion steps).
- SeRV: AR limits parallelism.
- SignSparK (0.01 s) and HybridSign (real-time) address this.

### C.10 Representation and tokenizer issues
- MaDiS says MoMask++ multi-scale tokenization accumulates errors.
- The VAE-design study (2606.22959) shows reconstruction accuracy does not predict generative B-T. Latent structure matters more.
- DigitCode (2608.03127) argues for anatomical-unit hand tokens.
- SeRV notes no explicit supervision of handshape, palm orientation or phonological categories. Phonology conditioning (2603.17388) is an early answer.

### C.11 Physical plausibility and embodiment
- Generated SMPL-X motion has hand-hand and hand-torso penetrations, which become critical for humanoid retargeting (2607.17769).

### C.12 Not yet explored (gap)
- RL or preference optimization of SLP generators with learned rewards, such as a back-translator, SiCLIP or BackTranslation2.0.
- Test-time guidance with those rewards.
- I found no paper that does this (see A.1).

---

## (D) Ten papers to read in full

1. **MaDiS (2601.19577)**: the current token-LM SOTA under the SOKE protocol. It defines SiBLEU and SiCLIP and shows how a pretrained LLM (Qwen3) plus masked diffusion beats AR.
2. **SignSparK (2603.10446, ECCV 2026)**: a flow-matching alternative with about 100x speedup and keyframe editing. It also provides the clearest evidence of evaluator inconsistency (its GT B-T ceilings).
3. **M3T (2603.23617, BMVC 2026)**: the only strong SOKE-protocol work with faces (SMPL-FX, FSQ). It introduces the NMFs-CSL test and an auxiliary SLT loss. It has the best DTW numbers so far on all three datasets.
4. **SOKE (2411.17799, ICCV 2025)**: the reference protocol and the SMPL-X pseudo-GT our repository builds on. Read it for the tokenizer, multi-head decoding and retrieval details.
5. **BLANC (2610.24109)**: the newest work. Its multi-rate per-articulator tokens and block-wise adaptive diffusion directly address timing. Its re-evaluation of SOKE, SignSparK and SignPR shows how much numbers change under a unified protocol.
6. **SIGNER (2506.07460, ECCV 2026)**: retrains every baseline under one representation and evaluator. It makes a strong case that temporal gloss grounding is the main lever for B-T.
7. **BackTranslation2.0 (2606.28673, ECCV 2026)**: the state of the art in SLP evaluation with human-correlated, multi-dimensional scoring. It is a candidate evaluator or reward.
8. **SeRV (2609.05742)**: semantic-aligned RVQ tokenizer and YouTube3D, a 236K-pair 3D ASL corpus. Relevant to scaling data and to tokenizer semantics.
9. **Uni-Sign (2501.15187, ICLR 2025)**: the strongest open SLT model and CSL-News. It is the natural back-translation or SiCLIP-style evaluator and pretraining source for CSL and ASL.
10. **SignFLIP (2609.35225, EMNLP 2026 Findings)** or **SignGPT (2609.21709)**: unified SLT+SLG in one LLM. They show where bidirectional training helps or hurts, and that hands still regress to the mean.

Honorable mentions:
- SignRR (retrieve and refine)
- GARD (gloss-wise AR diffusion)
- Sign-IDD (bone representation)
- SignViP (video tokens)
- Neural Sign Actors
- The SLRTP-2025 challenge report
- "Beyond BLEU" (2609.03734)

---

### Verification notes
- **Venues not confirmed:** SignDiff, NSLP-G, Mixed SIGNals (ICCV 2021?), Signing at Scale (CVPR 2022?), MDN (IJCV?), Ham2Pose (CVPR 2023?) and Stable Signer (S2 lists ACL 2025, but arXiv v1 is Dec 2025). MaDiS lists no venue on arXiv.
- **Not found on arXiv:** SignGen, SignMask, SignPR, SignFlow (Khan), Text2Sign (Stoll, IJCV) and NSLP-G. "DiffSLP" was not found under that name.
- **Not found at all:** "GReg", "SALAD", "VMarker-Pro", "DexAvatar" and "PEAR" appear in a GitHub awesome-list PR, but I could not locate the papers.
- **Possible copy error:** the CSL-Daily Sign2GPT value in SignFLIP's table (22.52) matches Sign2GPT's Phoenix score. Uni-Sign gives 15.40.
- **Possible discrepancy:** SeRV reports MaDiS How2Sign SiBLEU-4 as 4.47, while MaDiS's own table gives body 4.41 and hands 3.46. SeRV may use a different aggregation [unverified].
- **Prog. Trans. / Text2Mesh rows in MaDiS Table 1:** these did not extract cleanly from the HTML, so they are omitted. Their DTW values match SOKE's Table 1.

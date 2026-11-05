# Sources và reading ledger

Ngày truy cập: 04/11/2026. **39 full-text snapshots**, không đồng nghĩa 39 paper được đọc trọn vẹn. D = đọc sâu phần method/experiment liên quan; M = đọc chọn lọc method; S = screening. Không có human-read attestation. Các locator dưới đây mô tả phần thực sự dùng; không xác nhận supplement/proofs khi chưa đọc.

Bibliography rút gọn dùng first author, năm arXiv, title và identifier; năm proceedings có thể khác năm preprint. Venue chỉ nhận khi có nguồn chính thức ở phần cuối. Các summary và implications nằm ở files 01/02/04; ledger này phục vụ tra nguồn, không thay cho synthesis.

| Key / trích dẫn rút gọn | Full text và phiên bản snapshot | Mức đọc / phần dùng |
|---|---|---|
| `soke` — Zuo et al. (2024) | [Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator](https://arxiv.org/html/2411.17799v3) · `2411.17799v3` | **D** — §3–4, Tables 1–3; code DTW |
| `madis` — Zuo et al. (2026) | [MaDiS: Taming Masked Diffusion Language Models for Sign Language Generation](https://arxiv.org/html/2601.19577v2) · `2601.19577v2` | **D** — §3–4, Tables 1–6; code masking và metrics |
| `signspark` — Low et al. (2026) | [SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning](https://arxiv.org/html/2603.10446v4) · `2603.10446v4` | **D** — §3–4, Tables 2–8; code flow và release README |
| `momask` — Guo et al. (2023) | [MoMask: Generative Masked Modeling of 3D Human Motions](https://arxiv.org/html/2312.00063v1) · `2312.00063v1` | **M** — §3: RVQ, masking, residual transformer; code generation |
| `mdm` — Tevet et al. (2022) | [Human Motion Diffusion Model](https://arxiv.org/html/2209.14916v2) · `2209.14916v2` | **M** — §3: clean prediction, FK/contact/velocity, CFG/inpainting |
| `mld` — Chen et al. (2022) | [Executing your Commands via Motion Diffusion in Latent Space](https://arxiv.org/html/2212.04048v3) · `2212.04048v3` | **M** — §3: VAE compression và latent denoising |
| `t2mgpt` — Zhang et al. (2023) | [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations](https://arxiv.org/html/2301.06052v4) · `2301.06052v4` | **M** — §3: VQ training, motion generation recipe |
| `motiongpt` — Jiang et al. (2023) | [MotionGPT: Human Motion as a Foreign Language](https://arxiv.org/html/2306.14795v2) · `2306.14795v2` | **M** — §3.3: tokenizer, motion-language pretraining, instruction tuning |
| `motionlcm` — Dai et al. (2024) | [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](https://arxiv.org/html/2404.19759v3) · `2404.19759v3` | **M** — §3: consistency distillation và decoded control supervision |
| `flowmdm` — Barquero et al. (2024) | [Seamless Human Motion Composition with Blended Positional Encodings](https://arxiv.org/html/2402.15509v1) · `2402.15509v1` | **M** — §3.1–3.2: bidirectional denoising, blended position encoding |
| `dart` — Zhao et al. (2024) | [DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control](https://arxiv.org/html/2410.05260v3) · `2410.05260v3` | **M** — §3: primitives, history conditioning, VAE/diffusion |
| `hymotion` — Wen et al. (2025) | [HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation](https://arxiv.org/html/2512.23464v1) · `2512.23464v1` | **M** — §3 và §4.2 chọn lọc: flow DiT, duration/rewrite, HQ fine-tuning |
| `dcmotion` — Wang et al. (2026) | [DC-Motion: Decoupling Structure and Details via Discrete-Continuous Tokens for Human Motion Generation](https://arxiv.org/html/2606.14721v2) · `2606.14721v2` | **M** — §III-B–E: paired discrete-continuous representation và two-stage inference |
| `m3t` — Symeonidis-Herzig et al. (2026) | [M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production](https://arxiv.org/html/2603.23617v2) · `2603.23617v2` | **M** — §3.1–3.2 và overview: SMPL-FX, FSQ, translation task |
| `mardm` — Meng et al. (2024) | [Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression](https://arxiv.org/html/2411.16575v2) · `2411.16575v2` | **M** — §2–3 chọn lọc: redundant features, continuous AE, masked AR diffusion |
| `humantomato` — Lu et al. (2023) | [HumanTOMATO: Text-aligned Whole-body Motion Generation](https://arxiv.org/html/2310.12978v1) · `2310.12978v1` | **M** — §2.1–2.2: hierarchical body/hand VQ; retrieval-prior overview |
| `emage` — Liu et al. (2024) | [EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling](https://arxiv.org/html/2401.00374v5) · `2401.00374v5` | **M** — §4.1–4.4: compositional VQ, content/rhythm, masked gesture paths |
| `motionstreamer` — Xiao et al. (2025) | [MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](https://arxiv.org/html/2503.15451v3) · `2503.15451v3` | **M** — §3.1–3.2, introduction: causal AE và exposure-bias training |
| `remodiffuse` — Zhang et al. (2023) | [ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model](https://arxiv.org/html/2304.01116v1) · `2304.01116v1` | **S** — Abstract/official proceedings: hybrid retrieval, selective fusion |
| `nsa` — Baltatzis et al. (2023) | [Neural Sign Actors: A diffusion model for 3D sign language production from text](https://arxiv.org/html/2312.02702v2) · `2312.02702v2` | **S** — Abstract và đoạn §3–4 từ paper CVF: fitting, anatomical diffusion |
| `signrr` — Tito Cruz et al. (2026) | [SignRR: Retrieve and Refine Real Motion for Sign Language Production](https://arxiv.org/html/2608.28568v3) · `2608.28568v3` | **M** — §3.1–3.3: dictionary, RVQ, length alignment, joint refinement |
| `collapse` — Hong et al. (2026) | [Conditional Collapse in Sign Language Production: A Diagnostic and a Coverage Argument](https://arxiv.org/html/2606.01643v2) · `2606.01643v2` | **M** — §3, các mục §5–6 được định vị: diagnostic ratios và giới hạn reference |
| `talkflow` — Wang et al. (2026) | [TalkFlow: Structured Latent Flow Matching for Speech Driven Holistic Motion Generation](https://arxiv.org/html/2610.40972v1) · `2610.40972v1` | **M** — §3.1–3.4: structured latent streams và FiLM-gated interactions |
| `phase` — Dai et al. (2026) | [Controllable Text-to-Motion Generation via Modular Body-Part Phase Control](https://arxiv.org/html/2603.19795v1) · `2603.19795v1` | **M** — §4: sinusoidal phase, modular ControlNet, training/inference |
| `unimoflow` — Hua et al. (2026) | [UniMoFlow: Grounding Instruction-Driven 3D Human Motion Editing in Generation](https://arxiv.org/html/2608.09143v1) · `2608.09143v1` | **M** — The UniMoFlow Model: joint tokens, generation/editing, time reparameterization |
| `f5tts` — Chen et al. (2024) | [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/html/2410.06885v3) · `2410.06885v3` | **M** — §3.1–3.2: infilling, duration assumption, text processing |
| `diffusionpolicy` — Chi et al. (2023) | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/html/2303.04137v5) · `2303.04137v5` | **M** — §2: conditional action diffusion, horizons, observations |
| `rdt` — Liu et al. (2024) | [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/html/2410.07864v2) · `2410.07864v2` | **M** — §4.1: bimanual chunks, modality encoding/masking |
| `maskgit` — Chang et al. (2022) | [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/html/2202.04200v1) · `2202.04200v1` | **M** — §3: masked objective, iterative decode, mask scheduling |
| `mar` — Li et al. (2024) | [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/html/2406.11838v3) · `2406.11838v3` | **M** — §3.1–3.2: continuous tokens và diffusion loss |
| `fsq` — Mentzer et al. (2023) | [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/html/2309.15505v2) · `2309.15505v2` | **M** — §3: scalar grid, rounding/STE, codebook size |
| `flowmatching` — Lipman et al. (2022) | [Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2) · `2210.02747v2` | **M** — §3.1–3.2: marginal/conditional paths và gradient equivalence |
| `meanflow` — Geng et al. (2025) | [Mean Flows for One-step Generative Modeling](https://arxiv.org/html/2505.13447v1) · `2505.13447v1` | **M** — §4.1: average velocity identity và JVP; không kiểm chứng toàn bộ proofs |
| `tmr` — Petrovich et al. (2023) | [TMR : Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis](https://arxiv.org/html/2305.00976v2) · `2305.00976v2` | **S** — Abstract và headings; không coi là đã đọc toàn bộ §3 |
| `signer` — Lee et al. (2025) | [SIGNER: Temporally Grounded Sign Language Generation via Time-Resolved Conditioning](https://arxiv.org/html/2506.07460v2) · `2506.07460v2` | **M** — §4.1–4.3: PVQ, durations, IPA và local temporal fusion |
| `snapmogen` — Guo et al. (2025) | [SnapMoGen: Human Motion Generation from Expressive Texts](https://arxiv.org/html/2507.09122v2) · `2507.09122v2` | **M** — §4.1–4.2, Table 2: multi-scale quantization và single masked generator |
| `uaflow` — Han et al. (2026) | [Flow Matching with Uncertainty Quantification and Guidance](https://arxiv.org/html/2602.10326v3) · `2602.10326v3` | **M** — §3.1 và phần đầu §3.2: velocity NLL, conditional correction, uncertainty propagation |
| `sureflow` — Islam et al. (2026) | [SUREFlow: State-space Uncertainty-aware REsidual Flow Matching for Robust Robot Manipulation](https://arxiv.org/html/2607.10504v1) · `2607.10504v1` | **M** — §IV-A–B và overview: state-space velocity/uncertainty heads |
| `anchorroute` — Fang et al. (2026) | [SCAMP: Sparse-Anchor Control Is One Small Projection](https://arxiv.org/html/2605.14716v3) · `2605.14716v3` | **M** — §3.1–3.5: SCAMP projection, damping và temporal support |

## Phiên bản thay đổi và chất lượng nguồn

- SOKE: dùng v3; [ICCV 2025 proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html).
- MoMask: [CVPR 2024 proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.html). Bản HTML snapshot có thể là v1; không hợp nhất mọi số liệu khác phiên bản.
- MARDM: [CVPR 2025 proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.html).
- MotionStreamer: [ICCV 2025 proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_MotionStreamer_Streaming_Motion_Generation_via_Diffusion-based_Autoregressive_Model_in_Causal_ICCV_2025_paper.html).
- HumanTOMATO: [ICML 2024 / PMLR](https://proceedings.mlr.press/v235/lu24b.html). EMAGE: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_CVPR_2024_paper.html).
- ReMoDiffuse: [ICCV 2023 proceedings](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.html).
- MotionLCM: [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02504.pdf). F5-TTS: [ACL 2025 paper](https://aclanthology.org/2025.acl-long.313.pdf). Metadata/proceedings đã dùng để xác minh; nội dung đọc là HTML snapshot trong bảng.
- DART: [trang tác giả xác nhận ICLR 2025](https://zkf1997.github.io/DART/). MAR: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/66e226469f20625aaebddbe47f0ca997-Abstract-Conference.html). Diffusion Policy: [RSS 2023 proceedings](https://www.roboticsproceedings.org/rss19/p026.html).
- SignSparK và M3T: trang tác giả/repo xác nhận ECCV 2026 và BMVC 2026; [SignSparK](https://github.com/JianHe0628/SignSparK), [M3T](https://cogvis-cvssp.github.io/papers/m3t/). M3T code trang project vẫn ghi coming soon; không nhận đã audit code M3T.
- SIGNER là v2 của arXiv từng mang tên GLOS; SCAMP là v3 của arXiv từng mang tên AnchorRoute. Dùng full text được pin ở bảng, không chỉ search snippets cũ.
- MaDiS, DC-Motion, HY-Motion, TalkFlow, UA-Flow, SUREFlow và các preprints mới khác: trong tài liệu này không nâng hạng thành peer reviewed nếu chưa có xác minh venue riêng. Tính liên quan không thay thế thẩm định độc lập.

## Nguồn screening bổ sung ngoài 39 snapshots

- Khan et al. (2025), **SignFlow: End-to-End Sign Language Generation for One-to-Many Modeling using Conditional Flow Matching**. [Abstract chính thức ICMI](https://icmi.acm.org/2025/proceedings/), [DOI](https://doi.org/10.1145/3716553.3750765). Mức S; chỉ xác nhận prior art CFM cho SLG, không lấy điểm số.
- FineMotion: [ICCV 2025 proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_FineMotion_A_Dataset_and_Benchmark_with_both_Spatial_and_Temporal_ICCV_2025_paper.html). Mức S; gợi ý vai trò annotation theo part/time, chưa phân tích method sâu.
- Unified Number-Free Text-to-Motion Generation Via Flow Matching: [CVPR 2026 proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Unified_Number-Free_Text-to-Motion_Generation_Via_Flow_Matching_CVPR_2026_paper.html). Mức S; multi-agent setting khác single-signer benchmark, không dùng làm baseline SOTA trực tiếp.
- SignFML: tìm được thông báo tác giả, chưa xác minh full paper; không đưa nội dung/score chưa đọc vào proposal. Đây là khoảng trống cần cập nhật trước submission, không phải bằng chứng không có prior art.

## Snapshot integrity và tái sử dụng

Nguồn nằm tại `third_party/proposal2_sources/<key>.html` và `<key>.txt`. SHA256 dưới đây áp cho bytes HTML nguyên bản; parser BeautifulSoup trích phần `article`. File `manifest.json` còn có URL, ngày truy cập và kích thước text. Các URL không version trong prose thuận tiện đọc; khi tái lập dùng version/hash ở ledger này.

| Key | SHA256 HTML |
|---|---|
| `soke` | `68601100dfa348168d397aaaaf2594e159320980f333a5ddb33e286afe44ecf5` |
| `madis` | `408787a27b17b690e02f338ed1b2f7891d9d9f8df5aad2f85103830c5ce1c79a` |
| `signspark` | `59297db4875ace71946ed2e3c99c01852125872e733c02436a4bdf490c2ccf5f` |
| `momask` | `bb8d6cb4e1ac889783e29d25376cd4b0575753f76028027175c2ad2d2cdc0888` |
| `mdm` | `3d62cb2de280cec523d8e1a6fc48492be14273e922ffccd6c3a8cc389dd5fac5` |
| `mld` | `ec49524b595ae01bc0223e4cc9ab5d55ebd02304a24d873d68c3a81c8348a768` |
| `t2mgpt` | `10183709ae6b4ca78956720525c0ad4691ef97396b6c19d1358114248ff456b4` |
| `motiongpt` | `e26a1d8c83b43edb686c8bad11ebf54bac5dc3fd39fd63e618559eeda0449e7b` |
| `motionlcm` | `5ae16343bea07145007d6f9a9d4d9c39f6ba087cb0087bbc4fbb3882ece0b04a` |
| `flowmdm` | `16084eeb79baa4c7ba22fedce011143474068d0201dac464a647bc53ef899f4f` |
| `dart` | `ae430299ffebeca59ada09be365dc7a90a147c9b34517d640e103d8be36b6e75` |
| `hymotion` | `2a7319d2370c27ac158a922340edf205a3c3133b34a65a69f05c778c57b7a7d0` |
| `dcmotion` | `65cc059e4bb7252fefea538c724541a5b5862ff8d06a715a9d19223c903b01aa` |
| `m3t` | `e08cabea86364ea9665e6fe2ae7c131964b03543639e2c52cf0db6a3246e980f` |
| `mardm` | `c1d1adee9319be675ce42bf7b6c540cd27146672f0d524b1262ec5211ed0477a` |
| `humantomato` | `0ba1dc7bb0aba4ab48b0067bdad8aa23d80c5cb0964153e3a202ff6866716d93` |
| `emage` | `7ad718d46c97a54a1342d6ece0513a8447a92a2d3a62eb36a73f5d5cfd042ab6` |
| `motionstreamer` | `a86683af45f5339ed13061df03632a3980aca173f25c5e42a9878f5e25f42cc0` |
| `remodiffuse` | `93f0d9f2f41bbd772f838f7a99deca3800bf43f60c9f5d7881d7fc98f7e4f1a8` |
| `nsa` | `211d70cbba5dd9176a9aac9f6b23de9453e6effb0c02cbcb16f95fb957767209` |
| `signrr` | `8dd33513dac6d2e9cdd10de941525bbbd7c94c77a97dfa968f33bc929e85aad2` |
| `collapse` | `e01adf5ecc14972d0dcbd95566a4f543abc3e3efa5c9231f0f346f92c4500dd5` |
| `talkflow` | `605fcc51ef7006a224bc22c45e9c6bf386e0c8c4bf199f5ed00beb53b3de3608` |
| `phase` | `d302f7d10304cd22d306de4488604ed9a4858b701e1dcedc43d781b8a4958507` |
| `unimoflow` | `52944f868256523a52664cc5855aaf39ddca6a6b244b16496be43a31f12dd0d2` |
| `f5tts` | `a55f9ace2221d4d557478183fc14fbd19c1cea40f682396c08a4a5abaaa4f96c` |
| `diffusionpolicy` | `e6dfaa427bc849e35261602e56fba558bb2d1f14a7bfc15e3e42d259de72d065` |
| `rdt` | `98f9baf35dfc5888fe7176fb16fdad64a3be9e84f05ecc91514fdc45b37fab75` |
| `maskgit` | `bd4edadef7fa8eea1ce519d080da79c2b1e7e20f0b01c19e9592f72015b2289f` |
| `mar` | `7048ab0745822792003007ab62d6c1e826dcde863ebb12d6a0796a86e1236cfd` |
| `fsq` | `d289a0f9eebf8766de11c6d9a3812f9291bc53880a7f1c5677cad601b0c429a2` |
| `flowmatching` | `0a2d1352d1fcc29dbcb12e189938033ae77c971879a61b4cd8318c54ea09b024` |
| `meanflow` | `5bb377b5be4dc4c6603dad56bfb2b8ff2205caa828dfd49aa66d42923a0a9285` |
| `tmr` | `f5699b6bdda32c3261569bfd5180b904a56204e7eb20b3f5f6852611169fb6b4` |
| `signer` | `733e4e80288dc8b47ea2837faadb6ff44122023a3f6cf7f54934e5f2df9782a3` |
| `snapmogen` | `cd5c0f897a13c555181c53a9f5af8d48faf4d46816bf0dd39d93139768746d02` |
| `uaflow` | `a686342ab427102692c0e0fb1dd2574865d2328ec8a14493e182fa497a9345f4` |
| `sureflow` | `57b440d36f0103f4a4a469551f2b8e89a98cf84c4c570eec5295ccae462dfb58` |
| `anchorroute` | `8cd59c540ea13f6770722c7bca663aa19bc2a22c48755761df46a668763c2983` |

Kiểm tra local: hash phải khớp, mọi HTML có `article`, title không rỗng, các snapshot không có error trong manifest. Đây là kiểm tra tính toàn vẹn nguồn, không phải kiểm tra đúng khoa học của paper. Hash giúp phát hiện thay đổi; không chứng minh model đã đọc hết mọi ký tự.

Một số URL project như MARDM trả 404 khi mở trong đợt này; full text arXiv/CVF vẫn truy cập được. Không clone repo dựa trên một tên GitHub đoán. MoMask clone chính thức đã ghi ở [07_code_audit.md](07_code_audit.md).

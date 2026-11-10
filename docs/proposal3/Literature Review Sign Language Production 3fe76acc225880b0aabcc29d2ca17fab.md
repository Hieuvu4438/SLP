# Literature Review Sign Language Production

Classify: Projects (https://app.notion.com/p/Projects-e0976acc225883308f31816296f03bf3?pvs=21)
Done: Yes
Project Category: https://app.notion.com/87876acc225882abb47d010e67d9f659
Project: Sign Language Production (https://app.notion.com/p/Sign-Language-Production-3fe76acc2258808697dbc56a4f9513d8?pvs=21)

Bạn nên đi theo lộ trình **biểu diễn chuyển động người → text-to-motion → các kỹ thuật sinh hiện đại → chuyển động toàn thân liên tục → sign language production (SLP)**. Với mục tiêu cuối cùng là SLP, tôi sẽ ưu tiên những kiến thức có thể chuyển sang bài toán bàn tay, nét mặt, thời lượng và ngữ nghĩa ký hiệu.

Dưới đây là lộ trình khoảng **24–28 tuần, với 10–12 giờ/tuần**, giả định bạn đã biết PyTorch, backpropagation, xác suất và đại số tuyến tính cơ bản. Các paper mở rộng là tài liệu tra cứu; không cần đọc sâu toàn bộ ngay lượt đầu.

Danh mục được kiểm tra nguồn đến **29/10/2026**. Tôi ưu tiên proceedings chính thức, trang tác giả và bản paper gốc; ghi rõ workshop, arXiv và các bài mới được thông báo accepted.

Quy ước:

- **★ Cốt lõi:** nên đọc kỹ phương pháp, loss, thí nghiệm và hạn chế.
- **○ Mở rộng:** đọc abstract, hình kiến trúc và kết luận trước; quay lại khi chọn hướng nghiên cứu.
- Tên paper đều dẫn đến nguồn gốc hoặc trang chính thức có PDF.

**Trước hết, cần xác định điều gì chuyển được từ human motion sang SLP.**

| Khía cạnh | Text-to-human-motion tổng quát | Sign language production |
| --- | --- | --- |
| Quan hệ giữa text và motion | Mô tả hành động, chẳng hạn “một người đi rồi ngồi xuống” | Dịch nội dung từ ngôn ngữ viết/nói sang một ngôn ngữ ký hiệu |
| Chi tiết cần giữ | Tư thế, quỹ đạo, động tác, tiếp xúc | Thêm hình dạng bàn tay, hướng tay, vị trí ký hiệu, nét mặt, ánh mắt, đầu và thân |
| Thời gian | Độ dài và chuyển tiếp giữa các hành động | Thêm thời lượng ký hiệu, nhịp, đồng thời giữa các bộ phận và co-articulation |
| Điều kiện thành công | Motion tự nhiên và phù hợp mô tả | Motion còn phải truyền đạt đúng nội dung ngôn ngữ |

Ngôn ngữ ký hiệu có ngữ pháp riêng; không thể mặc định ánh xạ từng từ của câu đầu vào thành một động tác tay. Đây là lý do lộ trình cần một phần ngôn ngữ học và đánh giá riêng sau phần human motion. [openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html?utm_source=chatgpt.com)

---

**Chặng 1, tuần 1–3: nắm các cơ chế sinh mà paper motion đang sử dụng.**

Bạn cần hiểu sáu nền tảng sau. Nếu đã vững một phần, có thể chuyển thẳng sang paper motion tương ứng.

| ID | Paper và nguồn | Nội dung cần nắm |
| --- | --- | --- |
| F1 ★ | Kingma & Welling — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114?utm_source=chatgpt.com), **ICLR 2014** | ELBO, reparameterization, posterior/prior; vì sao một điều kiện có thể sinh nhiều motion khác nhau. |
| F2 ★ | Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com), **NeurIPS 2017** | Self-attention, cross-attention, positional encoding, causal mask; phân biệt encoder và decoder. |
| F3 ★ | van den Oord et al. — [Neural Discrete Representation Learning](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html?utm_source=chatgpt.com), **NeurIPS 2017** | VQ-VAE, codebook, commitment loss, straight-through estimator; nền tảng motion tokenization. |
| F4 ★ | Ho et al. — [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html?utm_source=chatgpt.com), **NeurIPS 2020** | Forward/reverse process, noise schedule, denoising objective; phân biệt dự đoán noise và dữ liệu sạch. |
| F5 ★ | Ho & Salimans — [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598?utm_source=chatgpt.com), **arXiv 2022; bản ngắn tại NeurIPS Workshop 2021** | Condition dropout, conditional/unconditional prediction; đánh đổi giữa bám điều kiện và độ đa dạng. |
| F6 ★ | Lipman et al. — [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com), **ICLR 2023** | Probability path, velocity field, conditional flow matching và ODE sampling. |

Đọc bổ sung **đúng lúc cần dùng**, thay vì học tất cả trước khi bắt đầu motion:

| Paper | Đọc trước nhánh nào? |
| --- | --- |
| Radford et al. — [Learning Transferable Visual Models From Natural Language Supervision — CLIP](https://arxiv.org/abs/2103.00020?utm_source=chatgpt.com), **ICML 2021** | MotionCLIP, text conditioning và contrastive alignment. |
| Rombach et al. — [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752?utm_source=chatgpt.com), **CVPR 2022** | Motion Latent Diffusion: hiểu cách tách autoencoder khỏi generative prior. |
| Chang et al. — [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200?utm_source=chatgpt.com), **CVPR 2022** | MoMask: masked-token prediction và iterative refinement. |
| Peebles & Xie — [Scalable Diffusion Models with Transformers — DiT](https://arxiv.org/abs/2212.09748?utm_source=chatgpt.com), **ICCV 2023** | Các mô hình diffusion/flow dùng Transformer, đặc biệt nhánh mở rộng quy mô. |
| Mentzer et al. — [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505?utm_source=chatgpt.com), **ICLR 2024** | Tokenization bằng FSQ; hữu ích khi đọc M3T ở phần SLP. |

**Đầu ra của chặng này:** tự giải thích được ba pipeline: **VAE + latent prior**, **motion tokenizer + token generator**, và **continuous diffusion/flow**. Phải phân biệt được *cách biểu diễn motion* với *mô hình sinh trên biểu diễn đó*.

---

**Chặng 2, tuần 4–5: học hình học, biểu diễn motion và cấu trúc dữ liệu.**

Đây là phần rất dễ bị bỏ qua nhưng ảnh hưởng trực tiếp đến chất lượng nghiên cứu. Một mô hình tốt vẫn có thể cho kết quả sai nếu hệ tọa độ, root motion, skeleton hoặc chuẩn hóa dữ liệu không nhất quán.

| ID | Paper và nguồn | Nội dung cần đọc kỹ |
| --- | --- | --- |
| R1 ★ | Loper et al. — [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/?utm_source=chatgpt.com), **ACM TOG / SIGGRAPH Asia 2015** | Pose, shape, skeletal hierarchy, linear blend skinning; phân biệt joint coordinates với tham số body model. |
| R2 ★ | Zhou et al. — [On the Continuity of Rotation Representations in Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html?utm_source=chatgpt.com), **CVPR 2019** | Euler, quaternion, axis-angle và biểu diễn rotation 6D; vì sao lựa chọn representation ảnh hưởng việc học. |
| R3 ★ | Romero et al. — [Embodied Hands: Modeling and Capturing Hands and Bodies Together — MANO / SMPL+H](https://mano.is.tuebingen.mpg.de/?utm_source=chatgpt.com), **SIGGRAPH Asia 2017** | Biểu diễn bàn tay có khớp; quan hệ giữa hand pose và body pose. |
| R4 ★ | Pavlakos et al. — [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image — SMPL-X](https://smpl-x.is.tue.mpg.de/?utm_source=chatgpt.com), **CVPR 2019** | Biểu diễn thống nhất thân, tay và mặt; nền tảng quan trọng cho avatar ký hiệu. |
| R5 ★ | Mahmood et al. — [AMASS: Archive of Motion Capture As Surface Shapes](https://amass.is.tue.mpg.de/?utm_source=chatgpt.com), **ICCV 2019** | Hợp nhất nhiều nguồn mocap; nguồn gốc dữ liệu, tham số hóa và preprocessing. |
| R6 ○ | Plappert et al. — [The KIT Motion-Language Dataset](https://arxiv.org/abs/1607.03827?utm_source=chatgpt.com), **Big Data 2016** | Motion–text pairing, annotation và đặc điểm benchmark KIT-ML. |
| R7 ○ | Punnakkal et al. — [BABEL: Bodies, Action and Behavior With English Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Punnakkal_BABEL_Bodies_Action_and_Behavior_With_English_Labels_CVPR_2021_paper.html?utm_source=chatgpt.com), **CVPR 2021** | Nhãn hành động có căn chỉnh thời gian; hữu ích cho motion dài và composition. |

SMPL-X bổ sung bàn tay và mặt biểu cảm so với SMPL cơ bản. Điều này giúp giải thích vì sao biểu diễn phục vụ sinh chuyển động thân người chưa đủ để mô tả chi tiết signing. [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/?utm_source=chatgpt.com)

Hai paper lịch sử nên đọc lướt để hiểu cách lĩnh vực hình thành:

- Holden, Saito & Komura — [A Deep Learning Framework for Character Motion Synthesis and Editing](https://www.ipab.inf.ed.ac.uk/cgvu/motionsynthesis.pdf?utm_source=chatgpt.com), **ACM TOG / SIGGRAPH 2016**. Học ý tưởng *motion manifold*, autoencoder và chỉnh sửa motion trong latent space; bài này không phải text-to-motion.
- Plappert, Mandery & Asfour — [Learning a Bidirectional Mapping between Human Whole-Body Motion and Natural Language using Deep Recurrent Neural Networks](https://arxiv.org/abs/1705.06400?utm_source=chatgpt.com), **Robotics and Autonomous Systems 2018**. Xem một cách tiếp cận text–motion bằng RNN/seq2seq trước thời Transformer và diffusion.

**Bài thực hành:** tải một đoạn motion, dựng skeleton animation, thực hiện forward kinematics, đổi local/global coordinates và kiểm tra rotation conversion. Với SMPL-X, xác định rõ những tham số nào điều khiển thân, hai tay và mặt.

---

**Chặng 3, tuần 6–7: hiểu bài toán text-to-motion và cách căn chỉnh text–motion.**

Trước khi đọc hàng loạt mô hình mới, đọc lướt [Motion Generation: A Survey of Generative Approaches and Benchmarks](https://arxiv.org/abs/2507.05419?utm_source=chatgpt.com), **Khani et al., arXiv 2025**. Dùng bài này làm bản đồ phân loại và tra cứu benchmark, không thay thế việc đọc paper phương pháp. [arxiv.org](https://arxiv.org/abs/2507.05419?utm_source=chatgpt.com)

Sau đó đọc theo thứ tự:

| ID | Paper và nguồn | Vai trò trong lộ trình |
| --- | --- | --- |
| H1 ○ | Petrovich et al. — [Action-Conditioned 3D Human Motion Synthesis with Transformer VAE — ACTOR](https://openaccess.thecvf.com/content/ICCV2021/html/Petrovich_Action-Conditioned_3D_Human_Motion_Synthesis_With_Transformer_VAE_ICCV_2021_paper.html?utm_source=chatgpt.com), **ICCV 2021** | Bước đệm từ VAE sang motion sequence. Điều kiện ở đây là nhãn hành động, chưa phải mô tả text tự do. |
| H2 ★ | Guo et al. — [Generating Diverse and Natural 3D Human Motions From Text](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html?utm_source=chatgpt.com), **CVPR 2022** | Paper giới thiệu **HumanML3D**; đọc representation, text-to-length, motion snippets và bộ đánh giá. |
| H3 ★ | Petrovich et al. — [TEMOS: Generating Diverse Human Motions from Textual Descriptions](https://mathis.petrovich.fr/temos/?utm_source=chatgpt.com), **ECCV 2022** | Học latent distribution tương thích giữa text và motion; hiểu bản chất one-to-many. |
| H4 ○ | Tevet et al. — [MotionCLIP: Exposing Human Motion Generation to CLIP Space](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4022_ECCV_2022_paper.php?utm_source=chatgpt.com), **ECCV 2022** | Cách tận dụng không gian ngữ nghĩa pretrained để điều kiện hóa motion. |
| H5 ★ | Petrovich et al. — [TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Petrovich_TMR_Text-to-Motion_Retrieval_Using_Contrastive_3D_Human_Motion_Synthesis_ICCV_2023_paper.html?utm_source=chatgpt.com), **ICCV 2023** | Contrastive text–motion alignment; nền tảng để hiểu retrieval và semantic evaluation. |

**Đặc biệt ở H2:** đọc cả representation và evaluator, không chỉ kiến trúc. HumanML3D sử dụng skeleton thân người 22 joints trong thiết lập của paper; biểu diễn này không cung cấp đầy đủ khớp ngón tay và nét mặt để chuyển nguyên trạng sang SLP. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf?utm_source=chatgpt.com)

**Đầu ra:** giải thích được vì sao một câu có nhiều motion hợp lệ, thời lượng được xác định như thế nào, và “motion tự nhiên” khác “motion đúng text” ở đâu.

---

**Chặng 4, tuần 8–11: đọc sâu hai nhánh mô hình sinh chính.**

Nên đọc cả nhánh liên tục và nhánh token. Đây là hai nền tảng hữu ích trực tiếp khi bước sang SLP.

| ID | Paper và nguồn | Điểm cần nắm và câu hỏi khi đọc |
| --- | --- | --- |
| H6 ★ | Tevet et al. — [Human Motion Diffusion Model — MDM](https://arxiv.org/abs/2209.14916?utm_source=chatgpt.com), **ICLR 2023** | Diffusion trực tiếp trên motion; dự đoán dữ liệu sạch để dùng geometric losses. Những loss nào phụ thuộc đặc điểm motion? |
| H7 ★ | Chen et al. — [Executing Your Commands via Motion Diffusion in Latent Space — MLD](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.html?utm_source=chatgpt.com), **CVPR 2023** | Tách motion VAE và latent diffusion. Khi nén mạnh hơn, mất chi tiết gì? |
| H8 ★ | Zhang et al. — [Generating Human Motion From Textual Descriptions With Discrete Representations — T2M-GPT](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html?utm_source=chatgpt.com), **CVPR 2023** | Motion VQ-VAE + autoregressive Transformer. Phân biệt lỗi tokenizer và lỗi sinh token. |
| H9 ★ | Guo et al. — [MoMask: Generative Masked Modeling of 3D Human Motions](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.html?utm_source=chatgpt.com), **CVPR 2024** | Residual quantization và masked generation; so sánh với sinh trái sang phải của T2M-GPT. |
| H10 ○ | Jiang et al. — [MotionGPT: Human Motion as a Foreign Language](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract.html?utm_source=chatgpt.com), **NeurIPS 2023** | Motion vocabulary, motion–language pretraining, instruction tuning và nhiều tác vụ chung một mô hình. |
| H11 ★ | Lu et al. — [HumanTOMATO: Text-aligned Whole-body Motion Generation](https://proceedings.mlr.press/v235/lu24b.html?utm_source=chatgpt.com), **ICML 2024** | Whole-body generation và hierarchical tokenization; cầu nối quan trọng sang bài toán cần thân, tay và mặt. |

MDM, MLD và T2M-GPT cho bạn ba cách đặt bài toán khác nhau: sinh trong motion space, sinh trong continuous latent space, và sinh chuỗi discrete tokens. HumanTOMATO đưa yêu cầu whole-body vào trung tâm thiết kế. [arxiv.org](https://arxiv.org/abs/2209.14916?utm_source=chatgpt.com)

**Bài thực hành nên làm:** chạy checkpoint của MDM và T2M-GPT hoặc MoMask trên cùng một tập prompt nhỏ. So sánh lỗi ngữ nghĩa, độ đa dạng và chi tiết chuyển động. Với tokenizer, dựng lại motion từ token trước khi đánh giá khả năng sinh từ text.

---

**Chặng 5, tuần 12–14: học motion dài, điều khiển, tốc độ và các hướng hiện đại đến 2026.**

Phần này giúp bạn thoát khỏi thiết lập “một câu → một clip ngắn”, đồng thời hiểu các kỹ thuật đang được phát triển.

| ID | Paper và nguồn | Giá trị đối với mục tiêu SLP |
| --- | --- | --- |
| H12 ○ | Zhang et al. — [ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.html?utm_source=chatgpt.com), **ICCV 2023** | Kết hợp motion được truy hồi với generative model; chuẩn bị cho nhánh sign dictionary/retrieval. |
| H13 ★ | Barquero et al. — [Seamless Human Motion Composition with Blended Positional Encodings — FlowMDM](https://openaccess.thecvf.com/content/CVPR2024/html/Barquero_Seamless_Human_Motion_Composition_with_Blended_Positional_Encodings_CVPR_2024_paper.html?utm_source=chatgpt.com), **CVPR 2024** | Motion dài, nhiều mô tả nối tiếp và chất lượng chuyển tiếp. **FlowMDM là mô hình diffusion**, không phải flow matching. |
| H14 ○ | Dai et al. — [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2504_ECCV_2024_paper.php?utm_source=chatgpt.com), **ECCV 2024** | Consistency distillation, ít bước sinh và điều khiển trong latent space. |
| H15 ★ | Zhao et al. — [DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control](https://zkf1997.github.io/DART/index.html?utm_source=chatgpt.com), **ICLR 2025** | Sinh motion primitives liên tiếp dựa trên lịch sử và text; hiểu hybrid autoregressive–diffusion. |
| H16 ○ | Xiao et al. — [MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_MotionStreamer_Streaming_Motion_Generation_via_Diffusion-based_Autoregressive_Model_in_Causal_ICCV_2025_paper.html?utm_source=chatgpt.com), **ICCV 2025** | Causal latent space và streaming; hữu ích khi nghiên cứu SLP online. |
| H17 ○ | Guo et al. — [MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_MotionLab_Unified_Human_Motion_Generation_and_Editing_via_the_Motion-Condition-Motion_ICCV_2025_paper.html?utm_source=chatgpt.com), **ICCV 2025** | Unified generation/editing bằng rectified flow; cách biểu diễn nhiều tác vụ trong cùng một framework. |
| H18 ★ | Meng et al. — [Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf?utm_source=chatgpt.com), **CVPR 2025** | Đọc để phản biện representation và evaluator; kết quả benchmark không chỉ đến từ kiến trúc. |
| H19 ○ | Wen et al. — [HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation](https://arxiv.org/abs/2512.23464?utm_source=chatgpt.com), **báo cáo arXiv, 12/2025** | Scaling, data pipeline, DiT-based flow matching và post-training. Đọc về chiến lược xây mô hình lớn. |
| H20 ★ | Li et al. — [MotionHiFlow: Text-to-Motion via Hierarchical Flow Matching](https://arxiv.org/abs/2604.23264?utm_source=chatgpt.com), **CVPR 2026** | Hierarchical temporal scales, topology-aware motion VAE và coarse-to-fine flow generation. |
| H21 ○ | Weng et al. — [BiMoGen: Bidirectional Motion-Text Generation via Unified Masked Discrete Diffusion](https://arxiv.org/abs/2609.35407?utm_source=chatgpt.com), **bản arXiv cập nhật 21/10/2026; tác giả ghi accepted NeurIPS 2026** | Unified motion↔text, masked discrete diffusion và self-correction; đọc sau MoMask và MotionGPT. |

Các bài mới nên được dùng để cập nhật lựa chọn kỹ thuật, không cần trở thành điều kiện để bạn bắt đầu thực nghiệm. MotionHiFlow đã có thông tin CVPR 2026; HY-Motion được liên kết ở dạng báo cáo arXiv; trạng thái accepted của BiMoGen được ghi trên bản tác giả. [arxiv.org](https://arxiv.org/abs/2604.23264?utm_source=chatgpt.com)

**Chỉ chuyển sang phần SLP khi bạn có thể trả lời được các câu sau:**

1. Vì sao chọn joint positions, rotations hay tham số SMPL-X?
2. Tokenizer làm mất thông tin gì và đo sự mất mát đó thế nào?
3. Diffusion trong motion space khác latent diffusion ở đâu?
4. Autoregressive, masked generation và diffusion có những đánh đổi nào?
5. Mô hình biết khi nào kết thúc motion bằng cách nào?
6. Đánh giá độ tự nhiên, đúng text và đa dạng khác nhau ra sao?

Bạn không cần thành chuyên gia ở mọi bài toán human motion. Các phần scene interaction, locomotion physics hay multi-person interaction có thể để sau nếu đề tài SLP của bạn chưa sử dụng chúng.

---

**Chặng 6, tuần 15–16: bắt đầu domain sign language bằng ngôn ngữ, dữ liệu và định nghĩa tác vụ.**

Đọc các bài sau **trước khi đi vào kiến trúc SLP**:

| ID | Paper và nguồn | Mục đích đọc |
| --- | --- | --- |
| L1 ★ | Bragg et al. — [Sign Language Recognition, Generation, and Translation: An Interdisciplinary Perspective](https://arxiv.org/abs/1908.08597?utm_source=chatgpt.com), **ASSETS 2019** | Hiểu ngôn ngữ ký hiệu, nhu cầu người dùng, vai trò chuyên gia ngôn ngữ và cộng đồng người Điếc trong nghiên cứu. |
| L2 ★ | Yin et al. — [Including Signed Languages in Natural Language Processing](https://aclanthology.org/2021.acl-long.570/?utm_source=chatgpt.com), **ACL-IJCNLP 2021** | Linguistic structure, tokenization, hạn chế của gloss và những gì mô hình thường bỏ sót. |
| L3 ○ | Rastgoo et al. — [Sign Language Production: A Review](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Rastgoo_Sign_Language_Production_A_Review_CVPRW_2021_paper.html?utm_source=chatgpt.com), **CVPR Workshops 2021** | Bản đồ lịch sử SLP; dùng cho bối cảnh và related work, bổ sung bằng các bài mới bên dưới. |

Trong ghi chú, luôn phân biệt:

| Tác vụ | Đầu vào → đầu ra |
| --- | --- |
| Sign recognition | Video/pose → nhãn ký hiệu hoặc gloss |
| Sign language translation, trong thiết lập thường gặp | Video/pose ký hiệu → text |
| Gloss-to-pose | Gloss → pose sequence |
| Text-to-pose / text-to-sign motion | Text ngôn ngữ viết → motion ký hiệu |
| Pose-to-video | Motion/pose → hình ảnh người ký hiệu |
| Text-to-sign video | Toàn bộ chuỗi từ text đến video ký hiệu |

**Một paper gloss-to-pose chưa giải quyết toàn bộ text-to-sign.** Khi so sánh kết quả, phải giữ đúng đầu vào và mức giám sát.

Tiếp theo đọc bốn nguồn dữ liệu này:

| ID | Paper / nguồn chính thức | Điều cần hiểu |
| --- | --- | --- |
| D1 ★ | Camgoz et al. — [Neural Sign Language Translation](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html?utm_source=chatgpt.com), **CVPR 2018** | Giới thiệu **RWTH-PHOENIX-Weather 2014T**: video DGS, text tiếng Đức và gloss; domain thời tiết. Đây là paper SLT, được đọc ở đây vì dataset và protocol. |
| D2 ★ | Duarte et al. — [How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html?utm_source=chatgpt.com), **CVPR 2021** | ASL, English transcripts, nhiều modality/view; đọc kỹ cách căn chỉnh và phạm vi dữ liệu 3D. |
| D3 ★ | Zhou et al. — [CSL-Daily: nguồn dataset và paper Improving Sign Language Translation with Monolingual Data by Sign Back-Translation](https://ustc-slr.github.io/datasets/2021_csl_daily/?utm_source=chatgpt.com), **CVPR 2021** | Chinese Sign Language với text và gloss, nội dung đời sống; thêm một môi trường kiểm tra ngoài PHOENIX. |
| D4 ★ | Yu et al. — [SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark](https://signavatars.github.io/?utm_source=chatgpt.com), **ECCV 2024** | Whole-body SMPL-X/MANO annotations, pipeline dựng annotation 3D và benchmark production. |

PHOENIX14T, How2Sign và SignAvatars cung cấp những loại dữ liệu khác nhau. Cần kiểm tra pose là mocap, kết quả ước lượng từ video hay fitting body model; không mặc định mọi nhãn “3D” đều là ground truth chính xác. [openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html?utm_source=chatgpt.com)

**Đầu ra:** chọn được một ngôn ngữ ký hiệu đích, một dataset chính và một output representation. Nếu đích cuối là ngôn ngữ ký hiệu Việt Nam, hãy coi ASL/DGS/CSL là môi trường học phương pháp; cần xác định riêng dữ liệu và cách đánh giá cho ngôn ngữ đích.

---

**Chặng 7, tuần 17–20: đi từ nền tảng SLP đến tokenization, diffusion và retrieval.**

Đọc theo thứ tự trong bảng. Các bài mở rộng chỉ cần đào sâu nếu phù hợp hướng bạn chọn.

| ID | Paper và nguồn | Điều cần rút ra |
| --- | --- | --- |
| S1 ★ | Stoll et al. — [Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks](https://link.springer.com/article/10.1007/s11263-019-01281-2?utm_source=chatgpt.com), **IJCV 2020** | Hiểu pipeline translation → motion → rendering và cách tách lỗi giữa các tầng. |
| S2 ★ | Saunders et al. — [Progressive Transformers for End-to-End Sign Language Production](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1430_ECCV_2020_paper.php?utm_source=chatgpt.com), **ECCV 2020** | Baseline quan trọng cho continuous pose generation; counter decoding, stopping và error accumulation. |
| S3 ★ | Saunders et al. — [Mixed SIGNals: Sign Language Production via a Mixture of Motion Primitives](https://openaccess.thecvf.com/content/ICCV2021/papers/Saunders_Mixed_SIGNals_Sign_Language_Production_via_a_Mixture_of_Motion_ICCV_2021_paper.pdf?utm_source=chatgpt.com), **ICCV 2021** | Tách translation khỏi animation thông qua latent representation và motion primitives. |
| S4 ○ | Saunders et al. — [Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic Sign Language Production](https://openaccess.thecvf.com/content/CVPR2022/html/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.html?utm_source=chatgpt.com), **CVPR 2022** | Dictionary signs, temporal alignment, co-articulation và SignGAN rendering. |
| S5 ○ | Shalev-Arkushin et al. — [Ham2Pose: Animating Sign Language Notation Into Pose Sequences](https://openaccess.thecvf.com/content/CVPR2023/html/Arkushin_Ham2Pose_Animating_Sign_Language_Notation_Into_Pose_Sequences_CVPR_2023_paper.html?utm_source=chatgpt.com), **CVPR 2023** | Nhánh biểu diễn ngôn ngữ có cấu trúc: **HamNoSys → pose**, khác với đầu vào câu ngôn ngữ viết thông thường. |
| S6 ★ | Xie et al. — [G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model](https://ojs.aaai.org/index.php/AAAI/article/view/28441?utm_source=chatgpt.com), **AAAI 2024** | Pose-VQVAE, discrete diffusion, CodeUnet và length prediction. Đây là **gloss-to-pose**. |
| S7 ★ | Yin et al. — [T2S-GPT: Dynamic Vector Quantization for Autoregressive Sign Language Production from Text](https://aclanthology.org/2024.acl-long.183/?utm_source=chatgpt.com), **ACL 2024** | Dynamic tokenization và duration; đọc đối chiếu T2M-GPT để thấy yêu cầu riêng của signing. Paper còn giới thiệu PHOENIX-News. |
| S8 ★ | Baltatzis et al. — [Neural Sign Actors: A Diffusion Model for 3D Sign Language Production from Text](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html?utm_source=chatgpt.com), **CVPR 2024** | Text-conditioned 3D signing với SMPL-X và kiến trúc khai thác cấu trúc giải phẫu. |
| S9 ○ | Ma et al. — [MS2SL: Multimodal Spoken Data-Driven Continuous Sign Language Production](https://aclanthology.org/2024.findings-acl.432/?utm_source=chatgpt.com), **Findings of ACL 2024** | Sequence diffusion và semantic consistency giữa text, audio và sign. |
| S10 ★ | Tang et al. — [Discrete to Continuous: Generating Smooth Transition Poses from Sign Language Observations](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Discrete_to_Continuous_Generating_Smooth_Transition_Poses_from_Sign_Language_CVPR_2025_paper.html?utm_source=chatgpt.com), **CVPR 2025** | Sign-D2C: tạo transition từ ngữ cảnh motion; đọc như bài toán completion/co-articulation, không đồng nhất với toàn bộ text-to-sign. |
| S11 ★ | Zuo et al. — [Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator — SOKE](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html?utm_source=chatgpt.com), **ICCV 2025** | Part-wise tokenizer, multi-head decoding, pretrained LM và sign dictionary retrieval. |
| S12 ○ | Fang et al. — [SignLLM: Sign Language Production Large Language Models](https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/html/Fang_SignLLM_Sign_Language_Production_Large_Language_Models_ICCVW_2025_paper.html?utm_source=chatgpt.com), **ICCV Workshops 2025** | Nhánh multilingual SLP và Prompt2Sign; đối chiếu preprocessing, mức giám sát và đánh giá giữa các ngôn ngữ. |

Ba cặp paper đặc biệt nên đọc cạnh nhau:

- **T2M-GPT → T2S-GPT:** tokenization cho motion tổng quát chuyển sang signing như thế nào; vì sao thời lượng và mật độ thông tin quan trọng.
- **MDM/MLD → Neural Sign Actors:** diffusion chuyển sang 3D signing cần thay đổi representation và modeling ra sao.
- **TMR/ReMoDiffuse → SOKE:** retrieval dùng cho semantic matching khác gì retrieval ký hiệu từ dictionary.

T2S-GPT thiết kế mã hóa động theo mật độ thông tin, còn SOKE sử dụng tokenizer theo bộ phận và retrieval ký hiệu. Đây là những ví dụ cụ thể cho thấy SLP cần điều chỉnh cả representation lẫn generation. [ACL Anthology](https://aclanthology.org/2024.acl-long.183/?utm_source=chatgpt.com)

---

**Chặng 8, tuần 21–22: cập nhật các kỹ thuật SLP năm 2026.**

Sau khi nắm các baseline ở trên, đọc phần này để xác định hướng nghiên cứu hiện tại.

| ID | Paper và nguồn | Điểm đáng nghiên cứu |
| --- | --- | --- |
| S13 ★ | Taşyürek et al. — [Disentangle and Regularize: Sign Language Production with Articulator-Based Disentanglement and Channel-Aware Regularization](https://arxiv.org/abs/2504.06610?utm_source=chatgpt.com), **WACV 2026; bản arXiv bắt đầu 2025** | Tách biểu diễn theo articulator và regularization theo kênh; hữu ích khi motion bàn tay/mặt có đặc tính khác thân. |
| S14 ○ | Yu et al. — [Focal-General Diffusion Model with Semantic Consistent Guidance for Sign Language Production](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Focal-General_Diffusion_Model_with_Semantic_Consistent_Guidance_for_Sign_Language_CVPR_2026_paper.html?utm_source=chatgpt.com), **CVPR 2026** | Kết hợp local joint dependencies với global sequence coherence; chú ý thiết lập **gloss-to-pose**. |
| S15 ★ | Ye et al. — [Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](https://aclanthology.org/2026.acl-long.31/?utm_source=chatgpt.com), **ACL 2026** | Hybrid AR–diffusion, multi-scale pose representation và confidence-aware causal attention. Đọc sau DART/MotionStreamer. |
| S16 ★ | Low et al. — [SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning](https://arxiv.org/abs/2603.10446?utm_source=chatgpt.com), **accepted ECCV 2026 theo bản tác giả và trang dự án** | Conditional flow matching, sparse keyframes, temporal anchors và điều khiển signing. |
| S17 ★ | Symeonidis-Herzig et al. — [M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production](https://arxiv.org/abs/2603.23617?utm_source=chatgpt.com), **accepted BMVC 2026; bản arXiv cập nhật 09/2026** | Body–hands–face tokenization, FSQ, biểu diễn mặt phong phú hơn và auxiliary sign-to-text objective. |
| S18 ○ | Fauré et al. — [The Impact of VAE Design on Latent Pose Representations for Diffusion-based Sign Language Production](https://openaccess.thecvf.com/content/CVPR2026W/GenSign/html/Faure_The_Impact_of_VAE_Design_on_Latent_Pose_Representations_for_CVPRW_2026_paper.html?utm_source=chatgpt.com), **CVPR Workshops 2026** | Đánh giá latent space vượt ra ngoài reconstruction error; hữu ích khi chọn hoặc thiết kế motion autoencoder. |

Ở thời điểm kiểm tra, M3T đã có bản tác giả ghi accepted BMVC 2026; hội nghị diễn ra sau ngày cập nhật lộ trình. SignSparK ghi accepted ECCV 2026 và sử dụng conditional flow matching. Không nên gộp hai bài này với trạng thái của một preprint chưa có thông tin chấp nhận. [arxiv.org](https://arxiv.org/abs/2603.23617?utm_source=chatgpt.com)

**Các kỹ thuật cần nối được sau khi đọc xong:**

| Nhánh kỹ thuật | Human motion nên đọc | SLP đối chiếu |
| --- | --- | --- |
| Continuous diffusion / latent diffusion | MDM, MLD | Neural Sign Actors, Hybrid AR–Diffusion |
| Discrete tokens + autoregressive generation | T2M-GPT, MotionGPT | T2S-GPT, SOKE, M3T |
| Masked generation / discrete diffusion | MoMask, BiMoGen | G2P-DDM |
| Whole-body / biểu diễn theo bộ phận | SMPL-X, HumanTOMATO | SignAvatars, SOKE, Disentangle and Regularize, M3T |
| Retrieval augmentation | TMR, ReMoDiffuse | Signing at Scale, SOKE |
| Motion dài, transition và streaming | FlowMDM, DART, MotionStreamer | Sign-D2C, Hybrid AR–Diffusion |
| Flow matching và temporal hierarchy | Flow Matching, MotionLab, MotionHiFlow | SignSparK |
| Duration và variable-rate representation | HumanML3D text-to-length; motion tokenization | T2S-GPT, G2P-DDM |

Bảng này thể hiện **quan hệ kiến thức để đọc và so sánh**, không khẳng định mọi paper SLP trực tiếp kế thừa từng paper human motion tương ứng.

---

**Chặng 9, tuần 23–24: tập trung đánh giá và tái lập một baseline.**

Paper bắt buộc ở đây là Walsh et al. — [SLRTP2025 Sign Language Production Challenge: Methodology, Results and Future Work](https://openaccess.thecvf.com/content/CVPR2025W/SLRTP/html/Walsh_SLRTP2025_Sign_Language_Production_Challenge_Methodology_Results_and_Future_Work_CVPRW_2025_paper.html?utm_source=chatgpt.com), **CVPR Workshops 2025**.

Bài challenge này quan trọng vì cung cấp protocol, evaluation network và benchmark chuẩn hóa cho text-to-pose. Nó cũng cho thấy retrieval-based methods là baseline cần xem xét nghiêm túc. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025W/SLRTP/html/Walsh_SLRTP2025_Sign_Language_Production_Challenge_Methodology_Results_and_Future_Work_CVPRW_2025_paper.html?utm_source=chatgpt.com)

Bạn nên xây bảng đánh giá theo nhiều tầng:

| Tầng đánh giá | Chỉ số / cách kiểm tra | Cách diễn giải |
| --- | --- | --- |
| Motion reconstruction | Joint error, rotation error; tách body/hands/face | Tokenizer/autoencoder có giữ được chi tiết trước khi sinh hay không? |
| Phân phối motion tổng quát | FID trong motion embedding space | Phụ thuộc feature extractor và protocol; không trực tiếp chứng minh ký hiệu đúng nghĩa. |
| Text–motion alignment | R-Precision, matching distance | Có truy hồi đúng cặp text–motion trong không gian evaluator hay không? |
| Đa dạng | Diversity và multimodality | Phân biệt đa dạng toàn tập với nhiều cách thể hiện cho cùng một điều kiện. |
| Sign pose và thời gian | DTW-based joint error, duration error, kiểm tra chuyển tiếp | DTW cho phép căn chỉnh thời gian, vì vậy cần đo nhịp và độ dài riêng. |
| Nội dung ký hiệu | Back-translation qua mô hình sign-to-text | Là tín hiệu gián tiếp, chịu ảnh hưởng từ chất lượng và thiên lệch của evaluator. |
| Mức dễ hiểu | Comprehension task với người thông thạo ngôn ngữ ký hiệu đích | Kiểm tra nội dung người xem thực sự hiểu được. |

H18 và S18 là hai bài đáng đọc lại tại đây: một bài đặt vấn đề representation/evaluation trong human motion, bài còn lại cho thấy chất lượng reconstruction của VAE chưa giải thích đầy đủ chất lượng sinh SLP phía sau. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf?utm_source=chatgpt.com)

Khi ghi lại một kết quả, hãy lưu cùng:

- Dataset, split và phiên bản pose extraction.
- Input là text, gloss hay text kèm dictionary/keyframes.
- Motion length được dự đoán hay lấy từ ground truth.
- Body model, skeleton, normalization và fps.
- Evaluator checkpoint, số lần sampling và uncertainty nếu protocol yêu cầu.
- Thời gian inference trên phần cứng và độ dài chuỗi cụ thể.

**Thực nghiệm đầu tiên nên nhỏ nhưng chẩn đoán được lỗi:** chọn một baseline phù hợp representation và dữ liệu bạn có; chạy inference, kiểm tra reconstruction rồi mới fine-tune. Sau đó thực hiện một ablation rõ ràng, thay vì đổi đồng thời tokenizer, backbone, loss và dữ liệu.

---

**Bạn có thể triển khai lịch đọc như sau.**

| Tuần | Trọng tâm | Đầu ra cần có |
| --- | --- | --- |
| 1–3 | F1–F6; bổ sung nền tảng còn thiếu | Giải thích được VAE, VQ, AR, diffusion, flow; phân biệt representation và prior. |
| 4–5 | R1–R7 | Notebook hiển thị motion, kiểm tra hệ tọa độ và forward kinematics. |
| 6–7 | H1–H5 | Ghi chú về HumanML3D, text–motion alignment và evaluator. |
| 8–9 | H6–H8 | Chạy checkpoint continuous và token-based generation. |
| 10–11 | H9–H11 | So sánh tokenizer, masked generation và whole-body modeling. |
| 12–14 | H12–H21 theo mức ưu tiên | Bảng so sánh generation, control, streaming và flow; hoàn tất nền human motion. |
| 15–16 | L1–L3, D1–D4 | Xác định tác vụ SLP, ngôn ngữ đích, dataset và representation. |
| 17–18 | S1–S8 | So sánh pipeline cổ điển, AR, tokenization và diffusion cho SLP. |
| 19–20 | S9–S12; đọc lại S4 nếu theo retrieval | Bản đồ duration, co-articulation, retrieval và multilingual generation. |
| 21–22 | S13–S18 | Chọn một hướng kỹ thuật và hai baseline gần nhất. |
| 23–24 | Challenge, evaluator và tái lập baseline | Baseline chạy được, error analysis và một ablation có kết luận. |
| 25–28 nếu cần | Thực nghiệm sâu hơn | Research proposal ngắn, kết quả sơ bộ và kế hoạch thí nghiệm tiếp theo. |

Với mỗi paper cốt lõi, nên viết **một trang ghi chú** gồm: bài toán; input/output; representation; kiến trúc; loss; điều kiện lúc inference; dữ liệu/evaluator; hạn chế; và một ý tưởng có thể kiểm chứng. Luôn đọc phần supplementary về preprocessing và evaluation của những baseline bạn định tái lập.

**Sau lộ trình, tôi ưu tiên bốn hướng thực nghiệm sau cho mục tiêu của bạn.** Đây là các giả thuyết để kiểm tra, chưa phải khẳng định tính mới:

| Hướng | Nguồn đọc chính | Thí nghiệm khởi đầu hợp lý |
| --- | --- | --- |
| **Giữ chi tiết bàn tay và non-manual features khi nén motion** | HumanTOMATO, T2S-GPT, SOKE, M3T, FSQ | So sánh tokenizer chung và tokenizer theo body/hands/face ở cùng mức nén; đo riêng từng bộ phận và mức dễ hiểu. |
| **Duration và tokenization thích ứng theo thông tin** | HumanML3D, G2P-DDM, T2S-GPT | So sánh fixed-rate và variable-rate encoding; kiểm tra lỗi handshape, timing và độ dài, không chỉ reconstruction trung bình. |
| **Retrieval kết hợp sinh chuyển tiếp có điều kiện** | ReMoDiffuse, Signing at Scale, SOKE, Sign-D2C, SignSparK | Đối chiếu retrieval, generation và hybrid; tách chất lượng ký hiệu khỏi chất lượng transition. |
| **SLP liên tục với độ trễ thấp** | MotionLCM, DART, MotionStreamer, Hybrid AR–Diffusion, SignSparK | Đo trade-off giữa latency, temporal context, semantic accuracy và chi tiết ngón tay khi sinh theo đoạn. |

Nếu cần một **trục đọc cốt lõi để bắt đầu ngay**, hãy lấy:

**SMPL/6D rotations → SMPL-X/AMASS → HumanML3D → TEMOS → TMR → MDM → MLD → T2M-GPT → MoMask → HumanTOMATO → FlowMDM/DART → MotionHiFlow → Bragg/Yin → dữ liệu SLP → Progressive Transformers → G2P-DDM → T2S-GPT → Neural Sign Actors → SOKE → SignSparK/M3T → SLRTP evaluation.**

Các paper nền tảng F1–F6 được học trước hoặc bổ sung theo đúng điểm bạn còn thiếu trong trục này.

# Literature Review: Text-to-Motion → Sign Language Generation

Research landscape 2018–2026 | Cập nhật đến 19/10/2026 | Computer Vision · Generative AI · Human Motion · Sign Language Production

## 1. Tổng quan và định hướng nghiên cứu

Với mục tiêu của bạn là nghiên cứu Text-to-Motion (T2M) ở phạm vi rộng, sau đó chuyên sâu vào Sign Language Generation (SLG), mình đề xuất tiếp cận theo hướng:

Human Motion Generation → Text-conditioned Motion → Whole-body Motion → Sign Language Motion → Linguistically Correct Sign Generation.

Đây là một hướng hợp lý bởi Sign Language Generation có thể kế thừa nhiều công nghệ từ Text-to-Motion, nhưng đồng thời có những yêu cầu ngôn ngữ học mà các mô hình sinh chuyển động thông thường chưa giải quyết tốt.

Research landscape

Human Motion Generation

VAE · VQ-VAE · Transformer · Diffusion · Flow Matching

Text-to-Motion Generation

Natural language → 3D skeletal motion

Sign Language Production

Text / Gloss → Body + Hands + Face motion

Text-to-Sign Pose

2D/3D keypoints · SMPL-X

Sign Video / Avatar

Video generation · 3D rendering

### Phân biệt các bài toán ngay từ đầu

| Task | Input → Output | Trọng tâm |
| --- | --- | --- |
| Text-to-Motion | Text → 3D motion | Motion realism, semantic alignment |
| Motion-to-Text | Motion → Description | Motion understanding/captioning |
| Sign Language Recognition | Sign → Gloss | Nhận dạng ký hiệu |
| Sign Language Translation | Sign → Spoken text | Dịch ngôn ngữ ký hiệu sang văn bản |
| Sign Language Production | Text/Gloss → Sign motion/video | Sinh ngôn ngữ ký hiệu |
| Sign Video Generation | Pose/Condition → RGB video | Hình ảnh signer chân thực |

Một điểm rất quan trọng: sign language không đơn giản là gesture generation hoặc diễn tả từng từ bằng cử chỉ tay. Nó là ngôn ngữ tự nhiên với ngữ pháp, cú pháp, cấu hình bàn tay, vị trí, chuyển động, biểu cảm khuôn mặt và các tín hiệu phi thủ công (non-manual markers).

Do đó, một mô hình có FID rất tốt trên HumanML3D chưa chắc tạo được một chuỗi ký hiệu có thể hiểu chính xác.

### Những xu hướng nghiên cứu nổi bật tính đến 2026

Qua các công trình tại CVPR, ICCV, ECCV và ACL, có thể nhận thấy:

- 2022–2023: Motion generation chuyển mạnh từ VAE sang Diffusion và discrete motion tokenization.
- 2023–2024: Latent diffusion, motion-language models và masked modeling trở thành những hướng quan trọng.
- 2024–2025: Sign Language Production phát triển các mô hình 3D toàn thân, diffusion, text-to-sign video và motion tokenization.
- 2025–2026: Nghiên cứu tập trung hơn vào lexical correctness, temporal grounding, fine-grained hand motion, multilingual generation, dữ liệu quy mô lớn và real-time generation.

Các hướng mới được thể hiện rõ trong những công trình như Signs as Tokens (ICCV 2025), FGDM (CVPR 2026) và Hybrid Autoregressive-Diffusion Model (ACL 2026).

[](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+3

## 2. Thống kê và danh mục paper

Mình đã tổng hợp 52 công trình được tuyển chọn, ưu tiên các hội nghị và tạp chí có phản biện học thuật như CVPR, ICCV, ECCV, NeurIPS, ICLR, ACL, AAAI và TPAMI.

# 52

Tổng công trình

# 23

Text-to-Motion

# 26

Sign Language

Ngoài ra có 3 bài survey. Đây là thống kê trên tập paper được chọn lọc, không phải số lượng toàn bộ công bố của lĩnh vực.

Số paper theo năm trong danh mục tuyển chọn

Số công trình được lựa chọn, không phải xu hướng sản lượng nghiên cứu toàn ngành.

[Chart](data:image/png;base64,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)

Literature Review — 52 Papers (Excel)

4 sheets: Overview, Paper Library, Datasets, Reading Roadmap.

Tải bảng Literature Review 2026 (.xlsx)

Gồm năm, venue, tên đầy đủ, phương pháp, input/output, benchmark, mức ưu tiên, lý do nên đọc và link paper gốc.

Quy ước ưu tiên đọc: A = cốt lõi, B = mở rộng có liên quan, C = chuyên biệt. Mức ưu tiên là đánh giá cho mục tiêu nghiên cứu Text-to-Motion → Sign Language của bạn, không phải xếp hạng chất lượng học thuật tuyệt đối.

Theo survey tại ACL 2026, lĩnh vực sign-language hiện đã có ít nhất 120 tài nguyên dữ liệu được khảo sát thuộc 35 ngôn ngữ ký hiệu. Con số này cho thấy sự phong phú về nguồn dữ liệu nhưng cũng nhấn mạnh vấn đề thiếu chuẩn hóa annotation và benchmark.

[](https://www.google.com/s2/favicons?domain=https://aclanthology.org&sz=32)

ACL Anthology

## 3. Text-to-Motion: Những paper cần đọc để xây dựng nền tảng

Mình chia thành bốn nhánh nghiên cứu. Quan trọng nhất là hiểu motion representation, conditional generation, temporal coherence và semantic alignment, thay vì chỉ học kiến trúc của một mô hình.

### 3.1. Foundation: VAE, Transformer và Motion Diffusion

| Paper | Venue | Điểm quan trọng cần học |
| --- | --- | --- |
| [Generating Diverse and Natural 3D Human Motions From Text](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html?utm_source=chatgpt.com) | CVPR 2022 | HumanML3D; text-conditioned motion generation và evaluation protocol |
| [TEMOS](https://is.mpg.de/ps/publications/temos-2022?utm_source=chatgpt.com) | ECCV 2022 | Text–motion latent space, VAE và diversity |
| [MotionDiffuse](https://pubmed.ncbi.nlm.nih.gov/38285589/?utm_source=chatgpt.com) | TPAMI 2024 | Diffusion cho motion generation; khởi nguồn preprint từ 2022 |
| [Human Motion Diffusion Model — MDM](https://mlanthology.org/iclr/2023/tevet2023iclr-human/?utm_source=chatgpt.com) | ICLR 2023 | Diffusion trực tiếp trên motion, classifier-free guidance |
| [Motion Latent Diffusion — MLD](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.html?utm_source=chatgpt.com) | CVPR 2023 | Nén motion vào latent space rồi diffusion |
| [T2M-GPT](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html?utm_source=chatgpt.com) | CVPR 2023 | VQ-VAE + autoregressive Transformer sinh motion token |
| [MotionGPT: Human Motion as a Foreign Language](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract.html?utm_source=chatgpt.com) | NeurIPS 2023 | Xem motion như chuỗi token để mô hình ngôn ngữ xử lý |
| [MoMask](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.html?utm_source=chatgpt.com) | CVPR 2024 | Residual VQ và masked motion modeling |

Các paper này là phần nền quan trọng nhất. Đặc biệt, MDM, MLD, T2M-GPT và MoMask đại diện cho những lựa chọn thiết kế khác nhau: continuous diffusion, latent diffusion, autoregressive discrete tokens và masked discrete generation.

[](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+2

### 3.2. Temporal modeling, retrieval và whole-body motion

| Paper | Venue | Vì sao liên quan đến sign language? |
| --- | --- | --- |
| [FineMoGen](https://papers.neurips.cc/paper_files/paper/2023/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html?utm_source=chatgpt.com) | NeurIPS 2023 | Điều khiển chuyển động ở mức chi tiết không gian–thời gian |
| [ReMoDiffuse](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.html?utm_source=chatgpt.com) | ICCV 2023 | Retrieval-augmented generation, hữu ích với sign dictionary |
| [Motion-X](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f8e27f6036c1d8b4a66b5b3a947dd7b-Abstract-Datasets_and_Benchmarks.html?utm_source=chatgpt.com) | NeurIPS 2023 | Whole-body SMPL-X: body, hands, face |
| [FlowMDM](https://openaccess.thecvf.com/content/CVPR2024/html/Barquero_Seamless_Human_Motion_Composition_with_Blended_Positional_Encodings_CVPR_2024_paper.html?utm_source=chatgpt.com) | CVPR 2024 | Ghép nhiều motion segments, xử lý transition |
| [MotionLCM](https://mlanthology.org/eccv/2024/dai2024eccv-motionlcm/?utm_source=chatgpt.com) | ECCV 2024 | Latent consistency, giảm latency khi sinh motion |
| [Motion Mamba](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/100_ECCV_2024_paper.php?utm_source=chatgpt.com) | ECCV 2024 | Long-sequence generation với kiến trúc hiệu quả hơn |
| [SALAD](https://openaccess.thecvf.com/content/CVPR2025/html/Hong_SALAD_Skeleton-aware_Latent_Diffusion_for_Text-driven_Motion_Generation_and_Editing_CVPR_2025_paper.html?utm_source=chatgpt.com) | CVPR 2025 | Skeleton-aware latent diffusion và editing |
| [PersonaBooth](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_PersonaBooth_Personalized_Text-to-Motion_Generation_CVPR_2025_paper.html?utm_source=chatgpt.com) | CVPR 2025 | Điều khiển motion theo identity hoặc style cá nhân |

Điểm cần chú ý: phần lớn benchmark Text-to-Motion cổ điển tập trung vào body locomotion và các hành động thông thường. Khi nghiên cứu sign language, cần chuyển sang expressive whole-body representation, trong đó hai bàn tay và khuôn mặt phải được mô hình hóa đủ chi tiết.

### 3.3. Các paper Text-to-Motion mới năm 2026

| Paper | Venue | Research direction |
| --- | --- | --- |
| [OpenT2M](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_OpenT2M_No-frill_Motion_Generation_with_Open-source_Large-scale_High-quality_Data_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Quy mô dữ liệu lớn, motion tokenizer và khả năng generalization |
| [MotionMaster](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Text-guided generation và editing |
| [Open the Motion Door](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Open_the_Motion_Door_Atomic_Motion_Decomposition_and_Recomposition_for_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Phân rã và tổ hợp atomic motion |
| [LLaMo](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LLaMo_Scaling_Pretrained_Language_Models_for_Unified_Motion_Understanding_and_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Unified motion understanding và generation bằng LM |
| [Multi-level Causal LLM-based Text-to-Motion Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Multi-level_Causal_LLM-based_Text-to-Motion_Generation_with_Human_Alignment_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Causal modeling và human alignment |
| [Hierarchical Enhancement of Semantic Priors](https://openaccess.thecvf.com/content/CVPR2026/html/Lv_Hierarchical_Enhancement_of_Semantic_Priors_for_Disentangled_Text-Driven_Motion_Generation_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Hierarchical semantics và disentangled generation |

OpenT2M đáng chú ý vì giới thiệu tập dữ liệu hơn 2.800 giờ motion, cùng một motion tokenizer khai thác sự phân chia các bộ phận cơ thể. Đây là hướng có thể liên hệ trực tiếp với việc mã hóa riêng hand, body và face cho sign language.

[](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

## 4. Sign Language Production: Từ các phương pháp nền tảng đến 2026

Đây là phần bạn nên dành nhiều thời gian hơn sau khi đã đọc các mô hình Text-to-Motion chính.

### 4.1. Những công trình nền tảng

| Paper | Venue | Đóng góp và ý nghĩa |
| --- | --- | --- |
| [Sign Language Production using NMT and GANs](https://github.com/VIPL-SLP/Awesome-Sign-Language-Processing/blob/master/complete_list/sign_language_production.md?utm_source=chatgpt.com) | BMVC 2018 | Một trong các hướng Neural Sign Production sớm |
| [Progressive Transformers for End-to-End Sign Language Production](https://arxiv.org/abs/2004.14874?utm_source=chatgpt.com) | ECCV 2020 | Sinh chuỗi sign pose từ ngôn ngữ bằng Transformer |
| [Mixed SIGNals](https://openaccess.thecvf.com/content/ICCV2021/html/Saunders_Mixed_SIGNals_Sign_Language_Production_via_a_Mixture_of_Motion_ICCV_2021_paper.html?utm_source=chatgpt.com) | ICCV 2021 | Mixture of motion primitives để sinh sign |
| [How2Sign](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html?utm_source=chatgpt.com) | CVPR 2021 | Dataset ASL đa phương thức |
| [Signing at Scale](https://openaccess.thecvf.com/content/CVPR2022/html/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.html?utm_source=chatgpt.com) | CVPR 2022 | Co-articulation và photorealistic signing |

Từ nhóm này, vấn đề bạn cần nắm là co-articulation: khi ký hiệu liên tục, các dấu hiệu không chỉ được thực hiện độc lập rồi nối lại. Sự chuyển tiếp và bối cảnh xung quanh ảnh hưởng đến cách thực hiện một ký hiệu.

### 4.2. Giai đoạn 2024: Diffusion, VQ và 3D avatars

| Paper | Venue | Vì sao quan trọng? |
| --- | --- | --- |
| [G2P-DDM](https://ojs.aaai.org/index.php/AAAI/article/view/28441?utm_source=chatgpt.com) | AAAI 2024 | Gloss-to-pose qua discrete diffusion |
| [Neural Sign Actors](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html?utm_source=chatgpt.com) | CVPR 2024 | Text-to-3D holistic sign motion bằng diffusion |
| [SignAvatars](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/653_ECCV_2024_paper.php?utm_source=chatgpt.com) | ECCV 2024 | Dataset và benchmark 3D signing toàn thân |
| [A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](https://mlanthology.org/eccv/2024/zuo2024eccv-simple/?utm_source=chatgpt.com) | ECCV 2024 | Text/gloss → sign dictionary → smoothing → avatar |
| [SignGen](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6988_ECCV_2024_paper.php?utm_source=chatgpt.com) | ECCV 2024 | End-to-end sign video generation qua latent diffusion |
| [T2S-GPT](https://aclanthology.org/2024.acl-long.183/?utm_source=chatgpt.com) | ACL 2024 | Dynamic vector quantization và autoregressive generation |
| [MS2SL](https://aclanthology.org/2024.findings-acl.432/?utm_source=chatgpt.com) | Findings ACL 2024 | Text/audio-conditioned continuous sign production |
| [Word-Conditioned 3D ASL Motion Generation](https://aclanthology.org/2024.findings-emnlp.584/?utm_source=chatgpt.com) | Findings EMNLP 2024 | Sinh sign motion ở mức từ, không phải cả câu |

Hai paper nên so sánh rất kỹ là Neural Sign Actors và T2S-GPT. Chúng thể hiện hai triết lý: generative diffusion trên 3D signing so với tokenization và autoregressive modeling.

T2S-GPT còn giới thiệu PHOENIX-News với 486 giờ video ký hiệu, và thiết kế codebook có độ dài linh hoạt theo mật độ thông tin trong chuỗi.

[](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+1

### 4.3. Các hướng nổi bật 2025–2026

| Paper | Venue | Vấn đề giải quyết |
| --- | --- | --- |
| [SLRTP2025 Sign Language Production Challenge](https://openaccess.thecvf.com/content/CVPR2025W/SLRTP/html/Walsh_SLRTP2025_Sign_Language_Production_Challenge_Methodology_Results_and_Future_Work_CVPRW_2025_paper.html?utm_source=chatgpt.com) | CVPR Workshop 2025 | Benchmark, đánh giá và giới hạn của phương pháp hiện tại |
| [Discrete to Continuous](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Discrete_to_Continuous_Generating_Smooth_Transition_Poses_from_Sign_Language_CVPR_2025_paper.html?utm_source=chatgpt.com) | CVPR 2025 | Sinh transition pose giữa các ký hiệu |
| [GReg](https://openaccess.thecvf.com/content/ICCV2025/html/Shi_GReg_Geometry-Aware_Region_Refinement_for_Sign_Language_Video_Generation_ICCV_2025_paper.html?utm_source=chatgpt.com) | ICCV 2025 | Cải thiện các vùng hình ảnh quan trọng khi sinh sign video |
| [Signs as Tokens — SOKE](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html?utm_source=chatgpt.com) | ICCV 2025 | Multilingual 3D sign generation, part-based tokenization và retrieval |
| [SignPR](https://dblp.org/rec/conf/cvpr/LiuGYGJMXL26?utm_source=chatgpt.com) | CVPR 2026 | Progressive vector-quantized diffusion |
| [FGDM](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Focal-General_Diffusion_Model_with_Semantic_Consistent_Guidance_for_Sign_Language_CVPR_2026_paper.html?utm_source=chatgpt.com) | CVPR 2026 | Local joint dependency + global motion coherence |
| [Hybrid Autoregressive-Diffusion Model](https://aclanthology.org/2026.acl-long.31/?utm_source=chatgpt.com) | ACL 2026 | Real-time sign production |
| [Stable Signer](https://aclanthology.org/2026.acl-long.659/?utm_source=chatgpt.com) | ACL 2026 | Hierarchical sign-language generation |
| [SIGNER](https://arxiv.org/abs/2506.07460?utm_source=chatgpt.com) | ECCV 2026 | Time-resolved conditioning và temporal grounding |
| [SignSparK](https://cogvis-cvssp.github.io/papers/signspark/?utm_source=chatgpt.com) | ECCV 2026 | Sparse keyframe learning, efficient multilingual generation |
| [DexAvatar](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html?utm_source=chatgpt.com) | WACV 2026 | 3D reconstruction, hỗ trợ xây dữ liệu motion |
| [Toward Phonology-Guided Sign Language Motion Generation](https://openaccess.thecvf.com/content/CVPR2026W/GenSign/html/Hong_Toward_Phonology-Guided_Sign_Language_Motion_Generation_A_Diffusion_Baseline_and_CVPRW_2026_paper.html?utm_source=chatgpt.com) | CVPR Workshop 2026 | Handshape, location và phonological conditioning |
| [The Impact of VAE Design on Latent Pose Representations](https://openaccess.thecvf.com/content/CVPR2026W/GenSign/html/Faure_The_Impact_of_VAE_Design_on_Latent_Pose_Representations_for_CVPRW_2026_paper.html?utm_source=chatgpt.com) | CVPR Workshop 2026 | Thiết kế latent representation cho diffusion |

Một số điểm mới đáng đọc kỹ:

- SOKE: biến ký hiệu thành token theo từng bộ phận cơ thể, kết hợp retrieval để sinh sign avatar 3D đa ngôn ngữ.
    
    [](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)
    
    openaccess.thecvf.com
    
- FGDM: sử dụng hai giai đoạn diffusion, từ chi tiết khớp xương cục bộ đến tính nhất quán trên toàn chuỗi.
    
    [](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)
    
    openaccess.thecvf.com
    
- Hybrid Autoregressive-Diffusion (ACL 2026): kết hợp modeling tuần tự và refinement, nhằm giảm đánh đổi giữa chất lượng và latency.
    
    [](https://www.google.com/s2/favicons?domain=https://aclanthology.org&sz=32)
    
    ACL Anthology
    

Lưu ý về mức độ chứng cứ: Các paper tại main conference, workshop và preprint không nên xem là có cùng trọng số. Đặc biệt, kết quả SOTA do tác giả công bố phải được đối chiếu cùng dataset, split, preprocessing và evaluation protocol; không nên gom điểm số từ các paper khác nhau thành một leaderboard duy nhất.

## 5. Dataset: Những số liệu cần ghi nhớ

Đây là bảng cực kỳ quan trọng nếu bạn định chuyển từ đọc paper sang thực nghiệm.

| Dataset | Quy mô công bố | Ngôn ngữ / representation | Ứng dụng |
| --- | --- | --- | --- |
| [KIT Motion-Language](https://motion-annotation.humanoids.kit.edu/dataset/?utm_source=chatgpt.com) | ~3.911 motion samples, ~6.353 annotations | 3D body motion + text | T2M nền tảng |
| [HumanML3D](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html?utm_source=chatgpt.com) | 14.616 motions, 44.970 captions | 3D body motion | Benchmark T2M phổ biến |
| [Motion-X](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f8e27f6036c1d8b4a66b5b3a947dd7b-Abstract-Datasets_and_Benchmarks.html?utm_source=chatgpt.com) | 81.100 sequences, 15,6 triệu frames | SMPL-X: body, hands, face | Whole-body pretraining |
| [OpenT2M](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_OpenT2M_No-frill_Motion_Generation_with_Open-source_Large-scale_High-quality_Data_CVPR_2026_paper.html?utm_source=chatgpt.com) | Hơn 2.800 giờ | Human motion + text | Large-scale T2M |
| [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/?utm_source=chatgpt.com) | ~8.257 clips theo split phổ biến | German Sign Language | SLP, SLT, gloss-to-pose |
| [How2Sign](https://how2sign.github.io/?utm_source=chatgpt.com) | Hơn 80 giờ | ASL video, speech, text, depth | Text-to-sign, video generation |
| [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/?utm_source=chatgpt.com) | ~20.654 samples | Chinese Sign Language | Sign translation, production |
| [SignAvatars](https://signavatars.github.io/?utm_source=chatgpt.com) | 70.000 videos, 153 signers, 8,34 triệu frames | Holistic 3D, SMPL-X | 3D sign language generation |
| [PHOENIX-News](https://aclanthology.org/2024.acl-long.183/?utm_source=chatgpt.com) | 486 giờ | German Sign Language | Large-scale sign modeling |
| [BOBSL](https://www.robots.ox.ac.uk/~vgg/data/bobsl/?utm_source=chatgpt.com) | Hơn 1.400 giờ, 1,2 triệu subtitle sentences, 37 signers | British Sign Language | Weakly aligned sign-language learning |

Các số liệu HumanML3D, Motion-X, How2Sign, SignAvatars và OpenT2M được đối chiếu từ bài báo gốc hoặc trang dataset chính thức.

[](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+5

### Nên dùng dataset nào khi bắt đầu nghiên cứu?

| Mục tiêu | Dataset nên ưu tiên | Hạn chế |
| --- | --- | --- |
| Tái hiện mô hình T2M | HumanML3D | Chủ yếu body motion, không đủ chi tiết cho sign |
| Pretrain whole-body motion encoder | Motion-X | Không phải dữ liệu ngôn ngữ ký hiệu chuyên biệt |
| Gloss-to-sign pose | PHOENIX-2014T | Domain thời tiết, vocab và style hạn chế |
| Text-to-ASL / video | How2Sign | Transcript–sign có thể không căn chỉnh hoàn hảo |
| 3D holistic sign motion | SignAvatars | Chất lượng reconstruction và kiểu annotation cần được kiểm tra |
| Large-scale sign pretraining | PHOENIX-News, BOBSL | Licensing, alignment và annotation không đồng nhất |

Đề xuất thực nghiệm: Nếu tài nguyên GPU và thời gian còn hạn chế, hãy bắt đầu với PHOENIX-2014T để tái hiện một sign-generation baseline. Nếu mục tiêu là hands + face + whole-body 3D, chuyển sang SignAvatars và đối chiếu với How2Sign.

## 6. Evaluation: Vấn đề rất dễ gây sai lệch khi đọc paper

Một trong những khó khăn lớn nhất là các mô hình có thể tạo motion đẹp nhưng nội dung ngôn ngữ ký hiệu không chính xác.

| Nhóm metric | Text-to-Motion | Sign Language Production |
| --- | --- | --- |
| Visual/motion realism | FID ↓, Diversity ↑ | Motion FID ↓, video quality, naturalness |
| Semantic alignment | R-Precision ↑, Matching Score ↓ | Gloss recognition, back-translation, comprehension |
| Pose accuracy | MPJPE ↓, reconstruction error | DTW-JPE ↓, PCK ↑, joint-level errors |
| Temporal coherence | Foot sliding, velocity, acceleration | Sign transitions, co-articulation, jerk |
| Fine-grained articulation | Whole-body consistency | Handshape, orientation, location, facial markers |
| Human evaluation | Action recognizability | Sign intelligibility bởi người dùng ngôn ngữ ký hiệu |
| Efficiency | Sampling time, FPS | Real-time latency, streaming quality |

Không nên xem BLEU của back-translation là thước đo duy nhất. Nếu dùng một sign-to-text model để đánh giá motion do text-to-sign model tạo ra, kết quả có thể bị ảnh hưởng bởi chất lượng và bias của chính mô hình nhận dạng.

Mình khuyến nghị khi đọc paper luôn ghi lại bốn thông tin:

1. Input thật sự là gì? Raw text, gloss, isolated word hay phonological annotation?
2. Output là gì? 2D keypoints, 3D joints, SMPL-X hay RGB video?
3. Mô hình có sinh được unseen sign và câu dài không?
4. Đã có human evaluation với người sử dụng ngôn ngữ ký hiệu chưa?

Chỉ so sánh kết quả giữa các paper khi chúng có cùng điều kiện benchmark.

## 7. Những research gaps đáng cân nhắc

Dưới đây là đánh giá của mình về các hướng có khả năng phát triển thành đề tài nghiên cứu. Đây là giả thuyết và đề xuất, không phải khẳng định các vấn đề chưa từng được nghiên cứu.

A. Phonology-aware 3D Sign Generation

Tách handshape, orientation, location, trajectory và non-manual markers trong conditioning. Kết hợp structured linguistic features với latent diffusion hoặc token-based modeling.

Đọc: Neural Sign Actors → SOKE → FGDM → Phonology-Guided Generation (2026).

B. Temporally Consistent Sign Motion

Tối ưu transition giữa các ký hiệu, temporal grounding và long-context generation, không chỉ smoothing các keypoints.

Đọc: FlowMDM → Discrete to Continuous → SIGNER → SignSparK.

C. Retrieval-Augmented Text-to-Sign

Kết hợp sign dictionary với generative model để giảm lỗi đối với ký hiệu hiếm, từ mới và những chuỗi chưa gặp trong tập train.

Đọc: ReMoDiffuse → A Simple Baseline (ECCV 2024) → SOKE.

D. Efficient Sign Language Generation

Few-step diffusion, hybrid autoregressive-diffusion hoặc sparse keyframes cho sinh ngôn ngữ ký hiệu thời gian thực.

Đọc: MotionLCM → Hybrid Autoregressive-Diffusion (ACL 2026) → SignSparK.

E. Low-resource / Multilingual Sign Language

Transfer learning hoặc multilingual modeling với dữ liệu ký hiệu hạn chế, có thể hướng đến các biến thể ngôn ngữ ký hiệu tại Việt Nam.

Cần hợp tác chuyên gia và cộng đồng người Điếc, cũng như thiết kế annotation đáng tin cậy.

### Đề tài mình ưu tiên

Linguistically Grounded Whole-Body Sign Language Generation with Part-Aware Motion Tokenization

Mục tiêu là cải thiện semantic correctness của sign motion bằng cách kết hợp kiến thức ngôn ngữ ký hiệu với kỹ thuật tokenization và generative modeling của Text-to-Motion.

Một kiến trúc giả thuyết có thể thử nghiệm

Spoken Text / Gloss / Linguistic Features

Semantic + Temporal Encoder

Hand Tokens

Body Tokens

Face Tokens

Diffusion / Masked Token Decoder

Holistic 3D Sign Motion + Linguistic Evaluation

Giả thuyết có thể kiểm chứng: Part-aware motion tokenization cộng với phonological/temporal conditioning sẽ cải thiện độ chính xác của handshape, orientation và sự liên kết giữa các sign, trong khi vẫn duy trì naturalness của chuyển động.

Để kiểm chứng nghiêm túc, cần so sánh với baseline diffusion/token models, ablation theo bộ phận cơ thể và đánh giá semantic correctness, thay vì chỉ báo cáo giảm FID.

## 8. Reading Roadmap: 6 tuần

| Tuần | Đọc các paper | Kết quả cần đạt |
| --- | --- | --- |
| 1 — Foundation | HumanML3D, TEMOS, MDM, MLD | Hiểu motion representation, pipeline và T2M metrics |
| 2 — Motion tokens | T2M-GPT, MotionGPT, MoMask, Motion-X | Hiểu VQ, autoregressive modeling, holistic motion |
| 3 — Sign foundations | Progressive Transformers, Mixed SIGNals, Signing at Scale, How2Sign | Hiểu SLP và co-articulation |
| 4 — Modern SLP | Neural Sign Actors, T2S-GPT, SignAvatars, G2P-DDM | So sánh diffusion với token-based SLP |
| 5 — Recent SOTA | SOKE, Sign-D2C, SignPR, FGDM, SIGNER, Hybrid AR-Diffusion | Xác định 2–3 research gaps |
| 6 — Research design | Survey, evaluation challenge, phonology-guided papers | Chọn dataset, baseline, metric và thiết kế ablation |

Không nhất thiết phải đọc tuần tự toàn bộ 52 paper. Nên đọc kỹ khoảng 12–15 paper lõi, còn các paper khác dùng để mở rộng related work và kiểm tra novelty.

### Ba bài survey nên đọc song song

- [A Survey on Recent Advances in Sign Language Production](https://www.sciencedirect.com/science/article/pii/S0957417423033481?utm_source=chatgpt.com) — Expert Systems with Applications, 2024.
- [Deep Learning Pathways for Automatic Sign Language Processing](https://www.sciencedirect.com/science/article/pii/S0031320325001359?utm_source=chatgpt.com) — Pattern Recognition, 2025.
- [Sign-Language Datasets at Scale](https://aclanthology.org/2026.acl-long.1928/?utm_source=chatgpt.com) — ACL, 2026.

## 9. Kết luận và hướng triển khai

Nếu mục tiêu cuối cùng là publish một paper về Text-to-Sign Motion Generation, mình khuyến nghị không dành quá nhiều thời gian để cải thiện trực tiếp FID trên HumanML3D. Thay vào đó:

Bước nền tảng: Tái hiện một baseline T2M (MDM, T2M-GPT hoặc MoMask), để nắm được representation, training và evaluation.

Bước chuyển miền: Tìm hiểu SMPL-X và 3D holistic motion bằng Motion-X, SignAvatars; sau đó tái hiện một SLP baseline như Neural Sign Actors, T2S-GPT hoặc FGDM.

Bước đóng góp nghiên cứu: Tập trung vào một vấn đề cụ thể như handshape correctness, phonological conditioning, temporal transitions hoặc rare-sign generalization.

Câu hỏi nghiên cứu mình đánh giá có tiềm năng nhất:

How can part-aware motion representations and linguistically grounded conditioning improve both semantic correctness and temporal coherence in 3D sign language generation?

Điểm mạnh của hướng này là nối được hai dòng nghiên cứu: generative motion modeling từ Computer Vision và linguistic validity từ Sign Language Processing.

Trong file Excel kèm theo, bạn đã có danh mục đầy đủ 52 paper, gồm paper gốc, venue, phương pháp, dataset và thứ tự đọc — có thể sử dụng ngay để lập bảng Related Work cho thesis hoặc proposal nghiên cứu.
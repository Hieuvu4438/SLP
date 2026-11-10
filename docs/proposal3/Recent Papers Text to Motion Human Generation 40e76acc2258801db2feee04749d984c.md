# Recent Papers Text to Motion / Human Generation

Classify: Projects (https://app.notion.com/p/Projects-e0976acc225883308f31816296f03bf3?pvs=21)
Done: Yes
Project Category: https://app.notion.com/87876acc225882abb47d010e67d9f659
Project: Sign Language Production (https://app.notion.com/p/Sign-Language-Production-3fe76acc2258808697dbc56a4f9513d8?pvs=21)

Tôi đã tổng hợp **63 tài liệu**, ưu tiên các paper về **sinh chuyển động người 3D từ văn bản**, cập nhật đến **04/11/2026**:

- **46 paper cốt lõi** về text-to-motion, motion editing, control và composition.
- **10 paper mở rộng** về vật lý, whole-body và tương tác.
- **4 tài liệu dữ liệu nền** và **3 survey**.
- Trong 56 paper phương pháp/benchmark, **27 paper thuộc 2025–2026**.

Bản đầy đủ có tên paper, đường dẫn nguồn, bảng dataset, metric, phân tích kỹ thuật và hướng nghiên cứu:

[text_to_motion_literature_review_2026_vi.md](https://chatgpt.com/library?file_id=libfile_80e0a1546ec08191bcdf0bb3f06ee3f1&account_id=personal)

Phạm vi chính là **chuỗi chuyển động 3D** như skeleton, joint rotations, root trajectory và SMPL/SMPL-X. Đây là danh mục mở rộng có chọn lọc; chưa phải systematic review bao phủ mọi cơ sở dữ liệu. Năm dưới đây là **năm công bố chính thức**, có phân biệt workshop và preprint.

**Bức tranh kỹ thuật** có thể chia thành các hướng sau. Đây là cách phân nhóm tổng hợp; một paper có thể thuộc nhiều hướng.

| Hướng kỹ thuật | Ý tưởng chính | Paper tiêu biểu |
| --- | --- | --- |
| VAE và shared latent | Học không gian chung giữa ngôn ngữ và chuyển động | TEMOS, MotionCLIP |
| Diffusion và latent diffusion | Sinh motion qua khử nhiễu, trực tiếp hoặc trong latent | MDM, MLD, SALAD, RAM |
| Discrete autoregression | Token hóa motion rồi sinh bằng next-token prediction | T2M-GPT, MotionGPT, ScaMo |
| Generative masked modeling | Dự đoán song song các motion token bị che | MoMask, MMM, BAMM |
| Hybrid AR–diffusion | Kết hợp sinh nối tiếp với mô hình liên tục | DART, MotionStreamer, LLaMo |
| Consistency và flow matching | Giảm chi phí lấy mẫu, hướng tới sinh nhanh | MotionLCM, FlowMotion, UMF |
| Fine-grained alignment và preference optimization | Cải thiện đúng hành động, thứ tự, tần suất và sở thích | AToM, SoPo, MATE, MoTiGA |
| Dữ liệu lớn và pretrained models | Mở rộng khả năng generalization | Go to Zero, OpenT2M, MotionMaster |

Các bảng tiếp theo liệt kê paper theo thời gian; tên đầy đủ và nguồn gốc được lưu trong bản tải xuống.

**Các mốc nền tảng và phương pháp quan trọng giai đoạn 2022–2024:**

| Paper | Năm / nơi công bố | Kỹ thuật chính | Đóng góp đáng chú ý |
| --- | --- | --- | --- |
| **T2M / HumanML3D** | 2022 · CVPR | Temporal VAE, motion snippet, text2length | Đưa ra HumanML3D và pipeline sinh motion đa dạng với độ dài thay đổi. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html?utm_source=chatgpt.com) |
| **TEMOS** | 2022 · ECCV | VAE, latent tương thích text–motion | Sinh nhiều chuyển động từ cùng mô tả; hỗ trợ skeleton và SMPL. [mathis.petrovich.fr](https://mathis.petrovich.fr/temos/?utm_source=chatgpt.com) |
| **MotionCLIP** | 2022 · ECCV | Motion autoencoder căn chỉnh với CLIP | Khai thác tri thức text/image pretrained cho generation và editing. [European Computer Vision Association](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4022_ECCV_2022_paper.php?utm_source=chatgpt.com) |
| **TEACH** | 2022 · 3DV | Transformer phân cấp | Sinh chuỗi hành động theo thứ tự các mô tả ngôn ngữ; mốc sớm về temporal composition. [Perzeptive Systeme – MPI-IS](https://is.mpg.de/ps/de/publications/teach-2022?utm_source=chatgpt.com) |
| **MDM — Human Motion Diffusion Model** | 2023 · ICLR | Classifier-free diffusion, Transformer | Dự đoán motion sạch để tích hợp loss hình học, vận tốc và tiếp xúc chân. [guytevet.github.io](https://guytevet.github.io/mdm-page/?trk=public_post_main-feed-card_reshare-text&utm_source=chatgpt.com) |
| **MLD** | 2023 · CVPR | Motion VAE + latent diffusion | Chuyển quá trình diffusion sang latent để giảm chi phí sinh motion. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.html?utm_source=chatgpt.com) |
| **T2M-GPT** | 2023 · CVPR | VQ-VAE + autoregressive GPT | Baseline quan trọng cho sinh motion bằng token rời rạc. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html?utm_source=chatgpt.com) |
| **MotionGPT** | 2023 · NeurIPS | Motion vocabulary + language modeling | Thống nhất generation, captioning, prediction và in-betweening. [papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html?utm_source=chatgpt.com) |
| **ReMoDiffuse** | 2023 · ICCV | Retrieval-augmented diffusion | Truy xuất motion tham chiếu theo ngữ nghĩa và động học để hỗ trợ denoising. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.html?utm_source=chatgpt.com) |
| **Fg-T2M** | 2023 · ICCV | Diffusion, cấu trúc ngôn ngữ, GNN | Khai thác đặc trưng ngôn ngữ chi tiết qua suy luận ngữ cảnh nhiều mức. [durham-repository.worktribe.com](https://durham-repository.worktribe.com/output/1735623/fg-t2m-fine-grained-text-driven-human-motion-generation-via-diffusion-model?utm_source=chatgpt.com) |
| **MotionDiffuse** | 2024 · IEEE TPAMI | Diffusion, điều khiển bộ phận/thời gian | Một hướng diffusion sớm cho T2M; preprint xuất hiện năm 2022. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38285589/?utm_source=chatgpt.com) |
| **MoMask** | 2024 · CVPR | Residual VQ, masked/residual Transformers | Motion token nhiều tầng để giữ chi tiết; hỗ trợ temporal inpainting. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.html?utm_source=chatgpt.com) |
| **MMM** | 2024 · CVPR | Tokenizer + bidirectional masked Transformer | Giải mã nhiều token song song; hỗ trợ editing, in-betweening và nối chuỗi. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Pinyoanuntapong_MMM_Generative_Masked_Motion_Model_CVPR_2024_paper.html?utm_source=chatgpt.com) |
| **BAMM** | 2024 · ECCV | Bidirectional autoregression + masking | Kết hợp editing với khả năng tự điều chỉnh độ dài motion. [European Computer Vision Association](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2316_ECCV_2024_paper.php?utm_source=chatgpt.com) |
| **MotionLCM** | 2024 · ECCV | Latent consistency, motion ControlNet | Sinh một hoặc ít bước, đồng thời nhận tín hiệu điều khiển. [European Computer Vision Association](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2504_ECCV_2024_paper.php?utm_source=chatgpt.com) |
| **Motion Mamba** | 2024 · ECCV | Temporal/spatial state-space models | Thiết kế Mamba chuyên cho tính nhất quán thời gian và cấu trúc không gian. [European Computer Vision Association](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/100_ECCV_2024_paper.php?utm_source=chatgpt.com) |
| **FlowMDM** | 2024 · CVPR | Diffusion, blended positional encodings | Ghép nhiều đoạn motion với chuyển tiếp mượt. **Đây không phải flow matching.** [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Barquero_Seamless_Human_Motion_Composition_with_Blended_Positional_Encodings_CVPR_2024_paper.html?utm_source=chatgpt.com) |
| **OmniControl** | 2024 · ICLR | Spatial guidance + realism guidance | Điều khiển nhiều khớp ở các thời điểm khác nhau bằng một mô hình. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7827290f07f63485b849b66cfa3e5dd0-Abstract-Conference.html?utm_source=chatgpt.com) |
| **PriorMDM** | 2024 · ICLR | Composition của diffusion priors | DoubleTake, ComMDM và DiffusionBlending cho chuỗi dài, tương tác, điều khiển. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/24161b531db5b1483e1b3842a204860f-Abstract-Conference.html?utm_source=chatgpt.com) |
| **STMC** | 2024 · **CVPR Workshop** | Spatio-Temporal Motion Collage | Điều khiển nhiều prompt theo timeline và bộ phận cơ thể tại inference. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/html/Petrovich_Multi-Track_Timeline_Control_for_Text-Driven_3D_Human_Motion_Generation_CVPRW_2024_paper.html?utm_source=chatgpt.com) |
| **MotionFix / TMED** | 2024 · SIGGRAPH Asia | Conditional diffusion, editing triplets | Dataset source–target–edit-text và benchmark riêng cho motion editing. [SIGGRAPH Asia 2024](https://asia.siggraph.org/2024/presentation/?id=papers_114&sess=sess117&utm_source=chatgpt.com) |

**Các paper năm 2025** đáng chú ý về phương pháp, dữ liệu và đánh giá:

| Paper | Nơi công bố | Kỹ thuật chính | Đóng góp đáng chú ý |
| --- | --- | --- | --- |
| **SALAD** | CVPR | Skeleton-aware latent diffusion | Mô hình hóa quan hệ joint–frame–word; editing bằng attention không cần fine-tune. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Hong_SALAD_Skeleton-aware_Latent_Diffusion_for_Text-driven_Motion_Generation_and_Editing_CVPR_2025_paper.html?utm_source=chatgpt.com) |
| **AToM** | CVPR | RL với GPT-4Vision reward | Căn chỉnh theo event, gồm độ đầy đủ, quan hệ thời gian và tần suất hành động. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Han_AToM_Aligning_Text-to-Motion_Model_at_Event-Level_with_GPT-4Vision_Reward_CVPR_2025_paper.html?utm_source=chatgpt.com) |
| **ScaMo** | CVPR | FSQ-VAE + prefix autoregressive Transformer | Nghiên cứu scaling law theo compute, model size, vocabulary và dữ liệu. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_ScaMo_Exploring_the_Scaling_Law_in_Autoregressive_Motion_Generation_Model_CVPR_2025_paper.html?utm_source=chatgpt.com) |
| **Rethinking Diffusion for Text-Driven Human Motion Generation** | CVPR | Continuous diffusion + masked autoregression | Phân tích representation, distribution và những vấn đề của cách đánh giá hiện hành. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.html?utm_source=chatgpt.com) |
| **MG-MotionLLM** | CVPR | Motion–language modeling đa mức | Kết hợp hiểu/sinh motion ở mức chuỗi, đoạn và bộ phận. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_MG-MotionLLM_A_Unified_Framework_for_Motion_Comprehension_and_Generation_across_CVPR_2025_paper.html?utm_source=chatgpt.com) |
| **Motion-Agent** | ICLR | Motion tokens, adapter-tuned LLM | Generation, editing và understanding qua hội thoại nhiều lượt. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/77c6ccacfd9962e2307fc64680fc5ace-Abstract-Conference.html?utm_source=chatgpt.com) |
| **DartControl / DART** | ICLR | Latent diffusion primitives + autoregression | Sinh motion online từ luồng text; điều khiển không gian bằng optimization hoặc RL. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3b057de5a2e38bd8fa10201866c20dbf-Abstract-Conference.html?utm_source=chatgpt.com) |
| **MotionStreamer** | ICCV | Continuous causal latent + diffusion AR | Streaming motion theo lịch sử chuyển động và prompt mới. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_MotionStreamer_Streaming_Motion_Generation_via_Diffusion-based_Autoregressive_Model_in_Causal_ICCV_2025_paper.html?utm_source=chatgpt.com) |
| **MaskControl** | ICCV | Logit regularization/optimization | Đưa điều khiển khớp–thời gian vào generative masked motion models. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/html/Pinyoanuntapong_MaskControl_Spatio-Temporal_Control_for_Masked_Motion_Synthesis_ICCV_2025_paper.html?utm_source=chatgpt.com) |
| **Go to Zero / MotionMillion** | ICCV | Dữ liệu triệu mẫu, scaling tới 7B | Dataset và benchmark cho zero-shot, ngoài miền và chuyển động tổ hợp. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Go_to_Zero_Towards_Zero-shot_Motion_Generation_with_Million-scale_Data_ICCV_2025_paper.html?utm_source=chatgpt.com) |
| **SnapMoGen / MoMask++** | NeurIPS | Multi-scale tokens + masked Transformer | Dữ liệu text giàu chi tiết, giữ temporal continuity; mô hình MoMask++. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9028b8a3ca98f58e373f0c1497a17448-Abstract-Conference.html?utm_source=chatgpt.com) |
| **SoPo** | NeurIPS | Semi-online preference optimization | Kết hợp mẫu không được ưu tiên online với mẫu được ưu tiên offline. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/577cd5863ec73be4e6871340be0936ae-Abstract-Conference.html?utm_source=chatgpt.com) |
| **FlowMotion** | Computers & Graphics | Target-predictive conditional flow matching | Nhắm tới motion mượt, giảm jitter và lấy mẫu hiệu quả. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0097849325002158?utm_source=chatgpt.com) |

**Các paper năm 2026** đã xác minh qua proceedings, nguồn tác giả và danh sách hội nghị:

| Paper | Nơi công bố | Kỹ thuật chính | Đóng góp đáng chú ý |
| --- | --- | --- | --- |
| **Motion-R1** | ICLR | Decomposed CoT + RL Binding | Tăng khả năng xử lý quan hệ thời gian, nhân quả và text–motion alignment. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7a4cf58610e1c911e06970c8d75b86dc-Abstract-Conference.html?utm_source=chatgpt.com) |
| **MATE** | ICLR | Motion localization, kinematic prototypes | Căn chỉnh ngữ nghĩa motion ngay tại word embedding của text encoder. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2fbfefa47aadd5d583439590693b2dcc-Abstract-Conference.html?utm_source=chatgpt.com) |
| **Event-T2M** | ICLR | Event decomposition + event cross-attention | Xử lý prompt nhiều hành động; HumanML3D-E đánh giá theo số event. [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3ea322e6eb11d3852bdb40263547ae78-Abstract-Conference.html?utm_source=chatgpt.com) |
| **MoTiGA** | CVPR | Causal RVQ-VAE, causal prediction, MHPO | Token nhân quả nhiều tầng và human preference alignment; HumanML3D-R. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Multi-level_Causal_LLM-based_Text-to-Motion_Generation_with_Human_Alignment_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **UMF** | CVPR | Pyramid/Semi-Noise Motion Flow | Flow matching cho text-to-motion với số lượng người thay đổi. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Unified_Number-Free_Text-to-Motion_Generation_Via_Flow_Matching_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **OpenT2M** | CVPR | Dữ liệu lớn, tokenizer 2D-PRQ | Filtering chất lượng vật lý, annotation theo giây và hỗ trợ chuỗi dài. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_OpenT2M_No-frill_Motion_Generation_with_Open-source_Large-scale_High-quality_Data_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **Open the Motion Door / Atomic Motion** | CVPR | Atomic decomposition và recomposition | Phân rã mô tả ngoài miền thành motion nguyên tử rồi tái tổ hợp. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Open_the_Motion_Door_Atomic_Motion_Decomposition_and_Recomposition_for_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **LLaMo** | CVPR | Mixture-of-Transformers, continuous AR, flow head | Thống nhất motion understanding/generation, tận dụng pretrained LLM và streaming. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LLaMo_Scaling_Pretrained_Language_Models_for_Unified_Motion_Understanding_and_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **MotionMaster** | CVPR | MLLM fine-tuning, FSQ tokenizer, MotionGB | Unified generation/editing, đa hành động và generalization. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **RAM** | ECCV | Motion-centric alignment + reconstruction guidance | Dùng nhánh tái dựng để căn chỉnh và sửa sai trong denoising. [Reconstruction-Anchored Diffusion Model](https://feifeifeiliu.github.io/RAM/?utm_source=chatgpt.com) |
| **IRG-MotionLLM** | ECCV | Interleaved generation–assessment–refinement | Kết nối hiểu và sinh motion bằng vòng đánh giá, chỉnh sửa. [GitHub](https://github.com/HumanMLLM/IRG-MotionLLM?utm_source=chatgpt.com) |
| **ReMoMask** | ECCV | Retrieval, RVQ-VAE, spatiotemporal attention | Kết hợp motion retrieval với masked generation; preprint từ 2025. [aigeeksgroup.github.io](https://aigeeksgroup.github.io/ReMoMask/?utm_source=chatgpt.com) |

Nếu mở rộng từ text-to-motion sang **human motion generation trong môi trường thực tế**, nên đưa thêm các paper sau vào literature review:

| Paper | Năm / nơi công bố | Nhánh | Đóng góp chính |
| --- | --- | --- | --- |
| **PhysDiff** | 2023 · ICCV | Physics-aware generation | Physics-based projection trong denoising để giảm floating, foot sliding và penetration. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PhysDiff_Physics-Guided_Human_Motion_Diffusion_Model_ICCV_2023_paper.html?utm_source=chatgpt.com) |
| **TMR** | 2023 · ICCV | Text–motion retrieval | Joint embedding bằng contrastive learning và synthesis; hữu ích cho retrieval/evaluation. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2023/html/Petrovich_TMR_Text-to-Motion_Retrieval_Using_Contrastive_3D_Human_Motion_Synthesis_ICCV_2023_paper.html?utm_source=chatgpt.com) |
| **HumanTOMATO** | 2024 · ICML | Whole-body generation | Hierarchical VQ-VAE/GPT cho thân, tay và biểu cảm mặt. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/humantomato-text-aligned-whole-body-motion-generation/?utm_source=chatgpt.com) |
| **InterGen** | 2024 · IJCV | Human–human interaction | Diffusion cho chuyển động phối hợp hai người; giới thiệu InterHuman. [tr3e.github.io](https://tr3e.github.io/intergen-page/?utm_source=chatgpt.com) |
| **Inter-X** | 2024 · CVPR | Interaction dataset/benchmark | Chuyển động thân–tay với mô tả chi tiết, thứ tự và quan hệ tương tác. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Inter-X_Towards_Versatile_Human-Human_Interaction_Analysis_CVPR_2024_paper.html?utm_source=chatgpt.com) |
| **CHOIS** | 2024 · ECCV | Human–object interaction | Sinh đồng bộ người–vật từ text, trạng thái ban đầu và object waypoints. [lijiaman.github.io](https://lijiaman.github.io/projects/chois/?utm_source=chatgpt.com) |
| **TRUMANS** | 2024 · CVPR | Human–scene interaction | Dataset cảnh động và autoregressive diffusion cho tương tác chuỗi dài. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Scaling_Up_Dynamic_Human-Scene_Interaction_Modeling_CVPR_2024_paper.html?utm_source=chatgpt.com) |
| **UniHSI** | 2024 · ICLR | Language-guided interaction | LLM chuyển ngôn ngữ thành Chain of Contacts cho controller thực thi. [GitHub](https://github.com/OpenRobotLab/UniHSI?utm_source=chatgpt.com) |
| **TIMotion** | 2025 · CVPR | Human–human generation | Mô hình hóa thời gian, tương tác và thay đổi vai chủ động–bị động. [openaccess.thecvf.com](https://www.openaccess.thecvf.com/content/CVPR2025/html/Wang_TIMotion_Temporal_and_Interactive_Framework_for_Efficient_Human-Human_Motion_Generation_CVPR_2025_paper.html?utm_source=chatgpt.com) |
| **MoLaM** | 2025 · ICCV | Interactive motion–language model | Reasoning, generation, reaction và editing qua hội thoại nhiều lượt. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/html/Park_A_Unified_Framework_for_Motion_Reasoning_and_Generation_in_Human_ICCV_2025_paper.html?utm_source=chatgpt.com) |

**Dataset cần theo dõi** khi chọn bài toán và thiết kế thực nghiệm:

| Dataset | Quy mô hoặc đặc điểm được công bố | Phù hợp với |
| --- | --- | --- |
| **KIT-ML** | 3.911 motion; bản 2017 có 6.353 annotation | T2M và retrieval; cần ghi rõ phiên bản. [motion-annotation.humanoids.kit.edu](https://motion-annotation.humanoids.kit.edu/dataset/?utm_source=chatgpt.com) |
| **AMASS** | Bản ICCV 2019: hơn 40 giờ, hơn 300 người, hơn 11.000 motion | Pretraining và motion prior; dữ liệu nền. [openaccess.thecvf.com](https://openaccess.thecvf.com/content_ICCV_2019/html/Mahmood_AMASS_Archive_of_Motion_Capture_As_Surface_Shapes_ICCV_2019_paper.html?utm_source=chatgpt.com) |
| **BABEL** | Hơn 43 giờ, hơn 250 action, hơn 66.000 action segments | Temporal composition và hành động nối tiếp/chồng lấp. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2021/html/Punnakkal_BABEL_Bodies_Action_and_Behavior_With_English_Labels_CVPR_2021_paper.html?utm_source=chatgpt.com) |
| **HumanML3D** | 14.616 motion clips, 44.970 descriptions | Benchmark T2M một người. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html?utm_source=chatgpt.com) |
| **Motion-X** | 81,1K chuỗi, 15,6M whole-body poses | SMPL-X, body–hand–face generation. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f8e27f6036c1d8b4a66b5b3a947dd7b-Abstract-Datasets_and_Benchmarks.html?utm_source=chatgpt.com) |
| **Inter-X** | Khoảng 11K tương tác, hơn 8,1M frame | Human–human interaction chi tiết. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Inter-X_Towards_Versatile_Human-Human_Interaction_Analysis_CVPR_2024_paper.html?utm_source=chatgpt.com) |
| **MotionMillion** | Khoảng 2 triệu chuỗi, hơn 2.000 giờ | Scaling, zero-shot và OOD evaluation. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Go_to_Zero_Towards_Zero-shot_Motion_Generation_with_Million-scale_Data_ICCV_2025_paper.html?utm_source=chatgpt.com) |
| **OpenT2M** | Million-level, hơn 2.800 giờ | Generalization và long-horizon generation. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_OpenT2M_No-frill_Motion_Generation_with_Open-source_Large-scale_High-quality_Data_CVPR_2026_paper.html?utm_source=chatgpt.com) |
| **MotionGB** | 10.000 giờ **sau augmentation**, từ 400 giờ verified mocap | Multi-action generation/editing. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.html?utm_source=chatgpt.com) |

Không nên so trực tiếp số giờ dữ liệu đã augmentation với số giờ mocap độc lập. Các dataset được nhắc lại trong bảng này không được tính thêm thành paper mới.

**Ba survey nên đọc trước** để xây dựng phần Related Work:

| Survey | Công bố | Vai trò |
| --- | --- | --- |
| **Human Motion Generation: A Survey** | IEEE TPAMI, 2024 | Nền tảng về text-, audio-, scene-conditioned generation, dataset và metric. [ir.pku.edu.cn](https://ir.pku.edu.cn/handle/20.500.11897/707972?mode=full&utm_source=chatgpt.com) |
| **Motion Generation: A Survey of Generative Approaches and Benchmarks** | arXiv:2507.05419, 2025 — **preprint** | Phân loại theo AE/VAE/VQ-VAE, autoregression và diffusion. [arxiv.org](https://arxiv.org/abs/2507.05419?utm_source=chatgpt.com) |
| **Multimodal Generative AI for Human Motion Understanding and Generation: A Survey and Way Forward** | Information Fusion, 2026 | Tổng hợp LLM, diffusion, representation và multimodal fusion cho text–motion. [researchonline.jcu.edu.au](https://researchonline.jcu.edu.au/93557/?utm_source=chatgpt.com) |

Khi viết phần **so sánh phương pháp**, nên tổ chức theo vấn đề nghiên cứu thay vì chỉ liệt kê theo năm:

| Vấn đề | Nhóm paper nên đối chiếu | Cần đo thêm |
| --- | --- | --- |
| Chất lượng và hiệu quả sinh | MDM, MLD, MoMask, MMM, MotionLCM, FlowMotion | FID, alignment, latency, sampling steps/NFE |
| Prompt phức tạp và nhiều hành động | TEACH, FlowMDM, STMC, Event-T2M, Atomic Motion | Đủ hành động, đúng thứ tự, đúng số lần lặp, mượt chuyển tiếp |
| Streaming và tương tác online | DART, MotionStreamer, LLaMo | First-response latency, drift, phản ứng khi đổi prompt |
| Căn chỉnh ngôn ngữ chi tiết | SALAD, AToM, SoPo, MATE, MoTiGA | Left/right, before/after, tốc độ, tần suất và human preference |
| Generalization | ScaMo, Go to Zero, OpenT2M, MotionMaster | Unseen action, unseen composition, cross-dataset, kiểm soát trùng dữ liệu |
| Tính khả thi vật lý | PhysDiff, CHOIS, UniHSI, TIMotion, UMF | Foot sliding, penetration, contact, constraint satisfaction |

Bảng trên là **đề xuất thiết kế so sánh**, chưa phải kết quả thực nghiệm. Đặc biệt, không nên gộp FID từ nhiều paper thành một leaderboard chung khi evaluator, split, representation và sampling protocol khác nhau; vấn đề đánh giá này cũng được phân tích trong *Rethinking Diffusion*. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.html?utm_source=chatgpt.com)

Nếu chọn đề tài mới, ba hướng đáng kiểm chứng là **multi-event generation có điều khiển**, **streaming có ràng buộc vật lý**, và **fine-grained alignment được đánh giá độc lập bằng con người**. Bản đầy đủ có câu hỏi thực nghiệm và nhóm baseline tương ứng cho từng hướng.
# Tổng hợp tài liệu và quyết định chuyển giao sang signing

Phần synthesis ban đầu; các trạng thái đọc và diễn giải đã được cập nhật bởi [external evidence review](06_external_evidence_review.md) và [proposal v2](05_revised_proposal.md). Các nguồn chỉ có trong danh mục không được dùng như đã xác minh full text.

Đây là synthesis có chọn lọc từ hai danh mục đầu vào, không phải systematic review đã hoàn tất. Các cơ chế từ paper được tách khỏi suy luận chuyển giao do proposal đề xuất. Nguồn của từng hàng là paper/project gốc hoặc snapshot primary-source local; mức đọc ghi rõ trong `research/reading_status.md`.

## Các họ phương pháp và cơ chế có thể chuyển

| Họ / tài liệu | Cơ chế liên quan | Chuyển sang SLP thế nào | Vì sao không dùng nguyên xi |
|---|---|---|---|
| [MDM](https://arxiv.org/abs/2209.14916) | Dự đoán clean motion để tính geometry/velocity loss | Supervise FK ở endpoint refiner | Foot-contact regularizer không đáp ứng handshape/location; phạt chuyển động mạnh dễ xóa articulation |
| [MLD](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.html) | VAE nén motion trước diffusion | Chẩn đoán reconstruction rồi mới quyết định latent compression | Latent nhỏ tối ưu body không bảo đảm giữ fingers và timing |
| [T2M-GPT](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html) | VQ token rồi AR LM | Tách lỗi representation/generator khi audit SOKE | Thay vocabulary kéo theo thay embedding/LM; cần cùng data và rate |
| [MotionGPT](https://arxiv.org/abs/2306.14795) | Motion như vocabulary và multi-task LM | Auxiliary understanding có thể hỗ trợ semantics | Motion-caption alignment không tương đương dịch ngôn ngữ ký hiệu |
| [MoMask](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.html) | RVQ, masked base tokens, residual stages | Bidirectional sửa motion tokens; phân biệt semantic coarse và detail | Cần xử lý length; confidence không mặc định phát hiện confidently-wrong signs |
| [ReMoDiffuse](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.html) | Retrieval theo semantic/kinematic, selective attention | Coi dictionary là condition mềm | Dictionary không bao phủ word senses/inflection; retrieval nhầm có thể bị copy |
| [HumanTOMATO](https://proceedings.mlr.press/v235/lu24b.html) | Whole-body hierarchical modeling | Duy trì khác biệt dynamics body/hands/face | Whole-body naturalness không chứng minh truyền tải đúng signed grammar |
| [FlowMDM](https://openaccess.thecvf.com/content/CVPR2024/html/Barquero_Seamless_Human_Motion_Composition_with_Blended_Positional_Encodings_CVPR_2024_paper.html) | Blended positional encodings, coherent transitions | Transition metrics và local context giữa signs | Là diffusion, không phải flow matching; smoothing không thay thế nội dung |
| [MotionLCM](https://arxiv.org/abs/2404.19756) | Distill latent consistency, few-step sampling | Distill refiner sau khi teacher chính xác | Teacher sai ngữ nghĩa sẽ truyền lỗi; ít NFE chưa chắc end-to-end nhanh |
| [Rethinking Diffusion / MARDM](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.html) | Representation/distribution ảnh hưởng generation và evaluator | Giữ evaluator/representation controls | Bỏ rotational channels hữu ích trong HumanML3D không có nghĩa nên bỏ finger rotations |
| [MotionHiFlow](https://arxiv.org/abs/2604.23264) | Coarse-to-fine temporal flow, topology-aware VAE | Ưu tiên semantic scaffold và hand detail riêng | Coarse temporal averaging có thể mất dấu rất ngắn; chain nhiều scale thêm compute |
| [SALAD](https://arxiv.org/abs/2503.13836) | Joint–frame–word interactions, attention editing | Token-level text và joint-aware motion là donor hợp lý | Không ép word timeline của spoken text lên sign order; cần đọc method/code sâu thêm |
| [MATE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2fbfefa47aadd5d583439590693b2dcc-Abstract-Conference.html) | Localized text-motion correspondence, kinematic prototypes | Semantic encoder pretraining là alternative nếu base sai nội dung | Prototype từ action words không sẵn có cho lexical signs và non-manual syntax; hiện mới đối chiếu abstract |
| [RAM](https://feifeifeiliu.github.io/RAM/) | Reconstruction branch và residual guidance sửa lỗi denoising | Đối thủ gần của correction; cần REG control | RACE không được claim correction là mới; RAM có guidance từ reconstruction, khác predicted rollout risk. Hiện đọc project/abstract, cần full text và supplement |

Các hàng trên cho thấy một thiết kế tốt cần giữ đồng thời semantics và articulatory detail, nhưng không có phép cộng kết quả benchmark để dự đoán hiệu quả của tổ hợp. Chọn refiner nhỏ chỉ có căn cứ sau gate lỗi; nếu semantic omission chi phối, MATE/MaDiS/time grounding là hướng mạnh hơn.

## Các paper ký hiệu quyết định thiết kế

| Paper | Điều đã xác minh từ primary source | Hệ quả cho proposal |
|---|---|---|
| [SOKE](https://arxiv.org/html/2411.17799) | Part-wise VQ, multi-head mBART, dictionary làm auxiliary condition; Tables 1–2 | Baseline có resource retrieval riêng; cố định tokenization để đo correction |
| [SignSparK](https://arxiv.org/html/2603.10446v4) | FAST onset/mid/offset, separate streams, continuous flow; GF/SR/KF2P riêng | Không đưa GT keyframe vào text-only; tránh nhầm known-length với native-length |
| [MaDiS](https://2000zrl.github.io/madis/) | Token/latent/physical pretraining, temporal checkpoints, mixture-of-parts; metric mới | Chỉ thêm geometry+gate chưa mới; cần run cùng code/eval khi so |
| [SIGNER](https://arxiv.org/abs/2506.07460) | Text→predicted gloss→duration repetition; local temporal fusion và inter-part attention | Temporal grounding/part interaction đã có; GT gloss/duration phải tách oracle |
| [BLANC](https://arxiv.org/html/2610.24109v1) | Multi-rate VQ, motion teacher, causal duration, hand refinement; pose fit/eval riêng | Adaptive clock không phải khoảng trống chưa ai làm; refinement phải so trực tiếp |
| [Neural Sign Actors](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html) | Text-conditioned 3D sign diffusion | Continuous alternative, cần giữ anatomical expressiveness; chưa so số khác evaluator |
| [Sign-D2C](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Discrete_to_Continuous_Generating_Smooth_Transition_Poses_from_Sign_Language_CVPR_2025_paper.html) | Mask observed signing segments để học transition completion | Generated draft khác observed correct sign context; train bằng rollout là câu hỏi cần thử |
| [HandMDM](https://arxiv.org/abs/2508.15902) | Source local abstract nói handshape/location descriptions, dictionary và noisy pseudo-labeling | Attribute text là supervision tiềm năng nhưng dataset/sign-language transfer phải kiểm tra; chưa dùng nó làm thành phần bắt buộc |

HandMDM đã xác minh title/authors qua arXiv:2508.15902; phạm vi đọc vẫn là abstract/intro, không dùng làm căn cứ cho số liệu benchmark.

## Những nhánh được cân nhắc nhưng không đưa vào MVP

- Streaming DART/MotionStreamer/LLaMo: hữu ích cho latency và history, nhưng chưa chứng minh cần thiết cho offline text-to-sign. Causal rollout làm tăng exposure bias; không thêm streaming chỉ vì mới.
- FSQ/ScaMo/M3T/OpenT2M: quantization và scaling có thể giúp reconstruction/data coverage. Đổi tokenizer đồng thời đổi generator+refiner làm mất khả năng quy kết. M3T đặc biệt cần đọc kỹ face và sign-to-text objective trước full-face extension.
- AToM/SoPo/MoTiGA: reward/preferences là hướng post-training; text/video evaluator chung chưa được chứng minh phân biệt ký hiệu gần nhau. Cần fluent-signers preference và evaluator độc lập, không lấy VLM aesthetics làm intelligibility.
- PhysDiff/CHOIS/UniHSI/InterGen/Inter-X/TIMotion/UMF: contact/interaction đáng học về constraint modeling, nhưng object/locomotion/multi-person task không khớp benchmark một signer. Không mang blanket non-contact constraints sang signs có hand-to-hand/face contact.
- SignGen/GReg/Stable Signer: video fidelity và rendering có mục tiêu khác pose benchmark. Cải thiện texture/mouth appearance không chứng minh motion semantics; không lấy video-only scores để so JPE.
- Ham2Pose/G2P-DDM/FGDM: notation hoặc gloss là input khác. Chỉ so riêng task hoặc dùng predicted gloss và tính chi phí/lỗi text→gloss.

Các nhánh này đang được **screen theo danh mục**, chưa được khẳng định đã full-text/code review. Candidate bị loại khỏi MVP vì mismatch nghiên cứu không đồng nghĩa paper kém.

## Tension inventory có giới hạn

| Cặp | Căng thẳng | Đánh giá |
|---|---|---|
| SOKE ↔ SignSparK | Token precision/retrieval vs continuous mean-regression/transition | Khác conditional regime và data; matched reproduction trước khi suy ra kiến trúc tốt hơn |
| MaDiS ↔ SIGNER | Global bidirectional context vs local time-resolved gloss | Không mâu thuẫn trực tiếp; local condition có thêm alignment và gloss estimator |
| MoMask ↔ MotionHiFlow | Discrete residual detail vs continuous temporal hierarchy | Cả hai cần kiểm tra reconstruction; body-level score không chứng minh handshape |
| BLANC ↔ proposal | Adaptive timing và hand refinement đã có | Novelty risk lớn; đổi sang risk/relations chỉ là giả thuyết khác biệt, cần ablation |
| RAM ↔ proposal | Residual self-correction đã có trong T2M | Phải bổ sung REG baseline và full-text audit; claim generic correction bị bác bỏ |
| Rethinking Diffusion ↔ benchmark reuse | Evaluator có thể làm thay đổi ranking | Giữ exact legacy track, thêm diagnostic độc lập; không thay metric rồi giữ tên cũ |

Scholar confirmation: **pending** cho mọi cặp. Đây là scan theo các cặp cùng cơ chế, không exhaustive pairwise contradiction review. Không có reviewer thứ hai hay cross-model review trong lượt này.

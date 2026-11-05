# Chuyển giao kỹ thuật và kiểm tra tính mới

## 1. Ma trận lựa chọn

Các quyết định dưới đây là [P]; cột nguồn ghi bằng chứng [E] về cơ chế đã tồn tại, không chứng minh hiệu quả trên SLG.

| Kỹ thuật / nguồn | Giả định cần sửa khi sang SLG | Quyết định | Thí nghiệm phân biệt |
|---|---|---|---|
| [MaDiS](https://arxiv.org/html/2601.19577v2) masked prior | EOS, 400-frame capacity và confidence có thể sai | Giữ làm prior đầu tiên | Prior cố định, so refiner với raw output |
| [MoMask](https://arxiv.org/html/2312.00063), [MoMask++](https://arxiv.org/html/2507.09122v2) | Nhiều token hơn có thể khó dự đoán trên paired sign data ít | Nhánh tokenizer sau reconstruction audit | Cùng token budget; reconstruction và generation đo riêng |
| [FSQ](https://arxiv.org/html/2309.15505), [M3T](https://arxiv.org/html/2603.23617) | Grid không tự bảo đảm giữ lexical handshape | Chỉ ưu tiên khi collapse đo được | Utilization, held-out reconstruction, semantic minimal pairs |
| [MDM](https://arxiv.org/html/2209.14916) geometry loss | Không thay hand contact bằng foot sliding penalty | Dùng FK và quan hệ tay–tay/tay–thân | Local articulation và global placement riêng |
| [MLD](https://arxiv.org/html/2212.04048), [MARDM](https://arxiv.org/html/2411.16575), [DC-Motion](https://arxiv.org/html/2606.14721) | Compression hoặc latent residual có thể mang semantics | Continuous correction là baseline mạnh bắt buộc | Pose-space flow vs latent flow, cùng compute |
| [SignSparK](https://arxiv.org/html/2603.10446) sparse controls | Train anchors từ GT khác anchors dự đoán | Soft evidence, huấn luyện với prior errors | Oracle/predicted/no anchor, hard/soft adherence |
| [SIGNER](https://arxiv.org/html/2506.07460v2) local alignment | Gloss và duration supervision không có đều ở ba dataset | Đối chứng/nhánh gloss-supervised | Predicted vs GT gloss; không trộn vào gloss-free track |
| [HumanTOMATO](https://arxiv.org/html/2310.12978), [EMAGE](https://arxiv.org/html/2401.00374), [TalkFlow](https://arxiv.org/html/2610.40972) | Gesture synchrony khác linguistic simultaneity | Giữ stream, trao đổi thông tin giữa vùng | Independent heads vs same-frame cross-part blocks |
| [FlowMDM](https://arxiv.org/html/2402.15509), [MotionStreamer](https://arxiv.org/html/2503.15451) | Offline full-sentence và online prompt khác nhau | Học exposure-bias recipe; hoãn streaming | Train chỉ GT condition vs generated condition |
| [ReMoDiffuse](https://arxiv.org/html/2304.01116), [SignRR](https://arxiv.org/html/2608.28568) | Exemplar có thể sai sense, signer hoặc context | Retrieval là track riêng | Same dictionary, random/wrong/missing retrieval |
| [F5-TTS](https://arxiv.org/html/2410.06885) infilling/text conditioning | Text order không tương đương sign order; không có audio reference | Chuyển conditioning recipe, không chuyển alignment nguyên xi | Token-level multilingual text vs pooled sentence |
| [Diffusion Policy](https://arxiv.org/html/2303.04137), [RDT-1B](https://arxiv.org/html/2410.07864) | Không có environment feedback khi sinh sign | Chuyển joint action modeling | Coupled bimanual geometry vs hai tay độc lập |
| [MotionLCM](https://arxiv.org/html/2404.19759), [MeanFlow](https://arxiv.org/html/2505.13447) | Một bước có thể mất chi tiết ngón tay | Distill sau teacher | 25/10/4/2/1 NFE và semantic guardrails |
| [HY-Motion](https://arxiv.org/html/2512.23464) scaling/reward | Dữ liệu/sign reward chưa đủ tin cậy | Học data pipeline; hoãn RL và billion-scale training | Data-quality ablation trước capacity scaling |

## 2. Negative search mở rộng: uncertainty và anchors cũng có prior art

**[E]** [UA-Flow, §3](https://arxiv.org/html/2602.10326) đã ước lượng velocity uncertainty và lan truyền uncertainty để guidance trong image generation. [SUREFlow, §IV](https://arxiv.org/html/2607.10504) đã dùng input-dependent uncertainty cho residual/action refinement trong robotics. Vì vậy không claim “uncertainty-aware flow” hoặc “sửa chiều không chắc chắn” là nguyên lý mới của PACT-Sign.

**[E]** Kết quả tìm kiếm còn tên AnchorRoute, nhưng full text hiện tại của `2605.14716` là **SCAMP: Sparse-Anchor Control Is One Small Projection**. §3 mô tả sparse-anchor correction qua decoder khả vi bằng damped Gauss–Newton và phân tích temporal support. [SCAMP](https://arxiv.org/html/2605.14716). **[P]** Cần đối chứng geometric projection/smoothing đơn giản trước khi quy lợi ích cho generative refiner. Không áp solver lên anchor từ model như thể đó là measurement đúng.

**[I]** Điểm khác có thể nghiên cứu là *đánh giá và học độ tin cậy của evidence do sign prior sinh theo articulator*, dùng nó cho correction trong khi bảo toàn quan hệ bimanual trên cùng benchmark. Đây là **search-bounded candidate contribution**, không có bảo đảm “đầu tiên”. Cần đặt cạnh SUREFlow, SCAMP và SIGNER trong related work, không chỉ cạnh MaDiS.

## 3. Claim nào không được dùng

| Claim dễ viết | Tình trạng sau kiểm tra |
|---|---|
| “Lần đầu sinh sign bằng flow matching” | Đã có SignFlow, SignSparK |
| “Lần đầu discrete semantics + continuous details” | Đã có DC-Motion và các hướng continuous-token AR |
| “Lần đầu body-part interaction/temporal grounding” | Đã có HumanTOMATO, SIGNER, TalkFlow |
| “Lần đầu retrieve rồi refine sign motion” | Đã có SignRR |
| “FSQ bảo đảm mọi code được dùng, nên hết lỗi mặt” | Không suy ra từ định nghĩa FSQ; cần đo representation/fitting |
| “Confidence gate chứng minh sign đúng nghĩa” | Confidence được giám sát theo hình học, không có chứng nhận ngữ nghĩa |
| “SOTA trên SOKE đồng nghĩa intelligible trên mọi sign language” | Benchmark và user study chỉ cho bằng chứng trong phạm vi đã đo |

## 4. So sánh các hướng proposal

**A — PACT-Sign, ưu tiên chính.** Giữ prior mạnh, học correction dưới predicted conditioning, reliability theo vùng và coupled geometry. Ưu thế [I]: có thể phân rã giả thuyết với baseline nhỏ, không phụ thuộc gloss cho How2Sign. Rủi ro: prior đã đủ tốt ở hình học hoặc prior sai ngữ nghĩa quá nặng; novelty có thể chỉ ở adaptation/empirical study.

**B — SIGNER-style duration/gloss planner + coupled flow.** Ưu thế: lexical order rõ hơn trên CSL/PHOENIX. Rủi ro: supervision khác Track A, T2G lỗi, đã rất gần SIGNER/SignSparK. Chọn nếu diagnostics chứng minh ordering là lỗi chủ đạo và có gloss resources hợp lệ. Đây là hướng thay thế đầy đủ, không gắn planner vào A trước khi có bằng chứng.

**C — part-wise RVQ/FSQ masked model.** Ưu thế: cải thiện representation, pipeline đơn giản hơn two-stage refinement. Rủi ro: MoMask++, M3T, SignRR đã khai thác mạnh. Chọn nếu reconstruction ceiling là bottleneck; không tiếp tục A chỉ vì đã đặt tên method.

**D — scale data/pretraining + independent semantic supervision.** Ưu thế: giải coverage gap có căn cứ từ HY-Motion/Conditional Collapse. Rủi ro: chi phí fitting, data access, supervision và contamination lớn hơn. Triển khai thành nhánh dữ liệu, ghi nguồn và so cùng data; không che data gain dưới tên architecture.

**Quyết định:** bắt đầu A với strong baseline controls và các tiêu chí bác bỏ ở [06](06_experiment_plan.md). Nếu A thất bại, diagnostics dẫn tới B/C/D; không giảm mục tiêu thành thắng một metric dễ hơn.

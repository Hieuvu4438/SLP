# Proposal 2 — từ motion generation đến sign language generation

Ngày chốt khảo sát: **04/11/2026**. Ngôn ngữ đích: ASL, CSL và DGS trên How2Sign, CSL-Daily và PHOENIX-2014T. Đây là **literature review có định hướng thiết kế phương pháp**, độc lập với các hướng thực nghiệm đang chạy trong SLP.

**Khuyến nghị:** nghiên cứu **PACT-Sign — Part-coupled Articulatory Correction with conditional Transport**: một semantic prior kiểu MaDiS sinh chuyển động thô; một conditional flow sửa chuyển động liên tục, được điều kiện hóa bằng độ tin cậy của từng vùng và quan hệ hình học giữa hai tay–thân. Điểm cần kiểm nghiệm là **khi nào nên giữ hoặc sửa một anchor do mô hình dự đoán**, chứ không phải tính mới của việc ghép token với diffusion. Đặc tả đầy đủ trong [05_method_pact_sign.md](05_method_pact_sign.md).

Ba kết luận quyết định thiết kế:

1. SOKE cung cấp benchmark và tokenizer tốt; MaDiS đã khai thác bidirectional masked generation, pretraining nhiều tầng và fusion theo bộ phận. Chỉ thay AR bằng masked diffusion không còn là đóng góp mới. [SOKE, §3–4](https://arxiv.org/html/2411.17799v3), [MaDiS, §3–4](https://arxiv.org/html/2601.19577v2).
2. Continuous refinement, temporal grounding, inter-part attention và retrieval-refinement đều có tiền lệ mạnh. DC-Motion, SIGNER, SignSparK và SignRR phải được đưa vào kiểm tra tính mới và đối chứng; không thể chỉ so với SOKE. [DC-Motion, §III](https://arxiv.org/html/2606.14721), [SIGNER, §4](https://arxiv.org/html/2506.07460v2), [SignSparK, §3–4](https://arxiv.org/html/2603.10446), [SignRR, §3](https://arxiv.org/html/2608.28568).
3. Chất lượng pseudo-GT, nguồn gloss/keyframe, cách tính DTW và định nghĩa evaluator có thể làm thay đổi thứ hạng. Cần khóa protocol trước khi huấn luyện mô hình mới. [03_benchmark_contract.md](03_benchmark_contract.md), [07_code_audit.md](07_code_audit.md).

## Cách đọc

| File | Nội dung |
|---|---|
| [00_scope_and_search.md](00_scope_and_search.md) | RQ, phạm vi, truy vấn, tiêu chí chọn và giới hạn bằng chứng |
| [01_motion_generation_review.md](01_motion_generation_review.md) | Tổng hợp các họ phương pháp motion, gesture, speech, robotics và image generation |
| [02_sign_language_deep_review.md](02_sign_language_deep_review.md) | Đọc sâu SLG; phân biệt đóng góp, supervision và hạn chế |
| [03_benchmark_contract.md](03_benchmark_contract.md) | Các mốc số liệu đã đối chiếu và điều kiện so sánh |
| [04_transfer_and_novelty.md](04_transfer_and_novelty.md) | Chọn kỹ thuật, các hướng thay thế và nguy cơ trùng ý tưởng |
| [05_method_pact_sign.md](05_method_pact_sign.md) | Kiến trúc, biểu diễn, loss, training/inference và failure modes |
| [06_experiment_plan.md](06_experiment_plan.md) | Đối chứng, ablation, tiêu chí dừng và lộ trình kiểm chứng SOTA |
| [07_code_audit.md](07_code_audit.md) | Commit, điểm đọc mã nguồn, sai khác paper–code |
| [08_sources_and_reading_ledger.md](08_sources_and_reading_ledger.md) | Danh mục nguồn, mức độ đọc và dấu vết snapshot |
| [09_critical_review_and_audit.md](09_critical_review_and_audit.md) | Phản biện phương pháp và audit deliverable |

## Trạng thái và cách hiểu kết luận

**Đã hoàn thành phần nghiên cứu và proposal; chưa triển khai hay chạy thực nghiệm PACT-Sign.** Không có số liệu tự tạo, không có kết luận đã vượt SOTA. SOTA là mục tiêu thực nghiệm, không phải kết quả có thể suy ra từ literature review.

Corpus có 39 bản full text HTML được lưu để kiểm tra lại trong `third_party/proposal2_sources/`; mức độ đọc từng paper được ghi riêng, không coi tải file là đã đọc kỹ. MoMask được clone thêm vào `third_party/MoMask/`; SOKE, MaDiS và SignSparK được đọc từ các checkout sẵn có. Không dùng kết quả thực nghiệm nội bộ của SLP để chọn hướng này.

Nhãn sử dụng trong tài liệu: **[E]** bằng chứng từ nguồn; **[I]** suy luận tổng hợp; **[P]** lựa chọn/giả thuyết đề xuất; **[U]** điều chưa xác minh. Các phần đặc tả phương pháp và kế hoạch thực nghiệm mặc định là [P].

Disclosure: tài liệu được tổng hợp với công cụ AI hỗ trợ tìm kiếm, đọc paper và kiểm tra source code. Không có xác nhận rằng người dùng đã đọc các nguồn; không có đánh giá độc lập bởi chuyên gia hay người ký hiệu trong quá trình soạn proposal này.

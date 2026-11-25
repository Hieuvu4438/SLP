# Proposal 9 — SignHiFlow: Estimate-then-Transport với flow phân cấp kiểu MotionHiFlow cho SLP

**Trạng thái:** đề xuất v2 (đã sửa sau một vòng phản biện độc lập). Chưa cài đặt, chưa chạy thực nghiệm.
Ngày lập: 2026-11-23.

**Đầu vào:**

- MotionHiFlow (CVPR 2026, arXiv 2604.23264), code ở `third_party/MotionHiFlow`;
- bản port `methods/motionhiflow_slp`;
- đối thủ cần vượt: SignNMFlow (arXiv 2609.33339).

Theo yêu cầu, đề xuất **không dựa** vào các proposal hay method khác trong repo. Chỉ hai kết quả đo được
dùng làm bằng chứng nền: MotionHiFlow-SLP và SignNMFlow-reimpl.

## TL;DR

1. **Chẩn đoán (03).** Metric SOKE (DTW-JPE tích luỹ, DTW-PA-JPE) chấm theo **một** reference. Nó thưởng
   cho ước lượng "trung tâm" hơn là cho mẫu chân thực:
   - pose trung bình tĩnh đạt 4.70 body trên dev;
   - với SignNMFlow-reimpl, NFE 1 cho DTW tốt hơn NFE 5, và NFE 5 tốt hơn NFE 20.

   MotionHiFlow-SLP sample toàn bộ chuyển động (v-pred, 12 bước, CFG), trên latent VAE, với text encoder
   bị đóng băng. Sửa từng lỗi đó chỉ cho ra "MotionHiFlow + bản vá", **không đủ novelty**.
2. **Ý tưởng (04, 05): Estimate-then-Transport.**
   - **μ-net:** MMDiT kiểu MotionHiFlow có điều kiện text, dự đoán **tất định** motion "trung tâm" μ.
     Loss nằm trong đúng không gian FK mà metric đo.
   - **T-net:** chính flow phân cấp của MotionHiFlow, nhưng dùng làm **transport không nhận text** kiểu
     PMRF, đi từ μ tới phân phối motion thật. Mỗi stage chỉ sinh **một band DCT trực chuẩn mới**, các
     band đã sinh được giữ nguyên.
   - **Kết hợp:** $X=\mu+\Lambda(G-\mu)$, với $\Lambda$ tách theo articulator.
   - **Mệnh đề 1** (05-T2, dựa trên Freirich et al. 2021) cho rủi ro chính xác
     $\mathrm{MMSE}+\sum_p\lambda_p^2\mathbb E\|G_p-\mu_p\|^2$. Từ đó, ở mọi $\Lambda$, transport tốt hơn
     sampling hoặc shrink-to-mean.
   - **Cross-fitting** cho μ tránh việc anchor bị overfit trên tập train nhỏ.
3. **Đánh giá trung thực (06).**
   - Điểm vận hành chính được **đăng ký trước**: matched kinematics, tức tốc độ và gia tốc khớp GT.
     Mọi baseline được đưa về cùng điều kiện bằng shrink-to-mean.
   - Báo cáo frontier nDTW theo SiCLIP và theo FD (dùng evaluator bên ngoài), cùng DTW theo band.
   - ≥ 3 seed, bootstrap CI.
   - User study với người Điếc là bắt buộc trước mọi tuyên bố về articulation.
4. **Gate quyết định.**
   - **G0:** chẩn đoán không cần train. Gồm E0.8 (thử ý tưởng cũ trên checkpoint đã có) và **E0.9 Gelbrich
     headroom**, chạy trên CPU, trả lời câu hỏi transport có chỗ để thắng sampling hay không.
   - **G1:** μ-net phải tốt hơn mean của SignNMFlow-reimpl.
   - **G1.5:** pilot stage 1 phải cho $\rho=\mathbb E\|G-\mu\|^2/\mathbb E\|Y-\mu\|^2<0.8$. Đây là điều kiện
     tiên quyết trước khi train T-net đầy đủ.
   - **G2:** SignHiFlow phải trội frontier so với flat regressor, CorrDiff-style, PMRF phẳng,
     MotionHiFlow-SLP và SignNMFlow-reimpl, ở cùng ngân sách, với cùng $\Lambda$ theo articulator.

   Nếu một gate thất bại, xem kế hoạch dự phòng ở 07 §4.
5. **Mục tiêu Phoenix test, không dùng retrieval:** JPE body ≤ 4.35, hand ≤ 5.55, PA-body ≤ 3.85,
   PA-hand ≤ 1.05. Đây là đích để nhắm tới, **không phải dự báo**.
6. **Bài học từ hai vòng phản biện độc lập** (07 §5, §5b).
   - **Vòng 1:** bản v1, với các stage mịn "anchored" có nhận text, **sai ở nghiệm tối ưu**. Anchor chỉ là
     một offset, và model vẫn sample lại toàn bộ. Từ đó ra thiết kế v2.
   - **Vòng 2:** mỗi câu chỉ có một μ, nên $p_\mu$ rời rạc. Ở những hướng mà μ ít khác nhau giữa các câu,
     transport không nhận text cũng chỉ là sampler. Lợi ích của v2 vì vậy phải được **đo trước** bằng
     E0.9 và G1.5.
   - Ngoài ra đã sửa thêm: bias của $\hat\mu$, mismatch do cross-fit, anchor lấy từ rollout, slerp thay
     cho trộn tuyến tính, và cách giải matched kinematics.
   - Các tuyên bố novelty đã được thu hẹp lại.

## Mục lục

| File | Nội dung |
|---|---|
| [01_research_question_and_scope.md](01_research_question_and_scope.md) | Câu hỏi nghiên cứu, FINER, phạm vi, tiêu chí thành công, quy trình |
| [02_literature_review.md](02_literature_review.md) | Tổng quan: SLP (bảng SOTA theo protocol SOKE), MotionHiFlow và hierarchy, T2M, flow matching, perception–distortion, biểu diễn, TTS, ngôn ngữ học ký hiệu |
| [03_diagnosis.md](03_diagnosis.md) | Vì sao MotionHiFlow-SLP thua; SignNMFlow mạnh nhờ gì; metric thưởng cho cái gì; so sánh hai tầng; mục tiêu |
| [04_method_signhiflow.md](04_method_signhiflow.md) | **Đặc tả end-to-end (v2):** biểu diễn, toán tử band, μ-net, T-net, PES chính xác, $\Lambda$, paradigm huấn luyện, inference, ablation |
| [05_theory.md](05_theory.md) | Toán tử trực chuẩn, Mệnh đề 1 (rủi ro nội suy), transport không text và vì sao, hierarchy có ràng buộc band, ghi chú renoise, DTW/thời lượng, hình học quay, MBR |
| [06_experiment_plan.md](06_experiment_plan.md) | E0–E7, giao thức matched kinematics, các gate, baseline quy thuộc, ưu tiên, chi phí, bảng và hình cho paper |
| [07_devils_advocate_and_novelty.md](07_devils_advocate_and_novelty.md) | Định vị novelty, phản biện độc lập và phản hồi từng điểm, rủi ro, kế hoạch dự phòng |
| [08_implementation_blueprint.md](08_implementation_blueprint.md) | Cấu trúc code theo quy ước của repo, config, test bắt buộc, các mốc |
| [references.md](references.md) | Tài liệu tham khảo đã kiểm tra |

**Đọc nhanh:** README → 03 §1–2 → 04 §0 → 05-T2/T3 → 06 §7 → 07 §5.

## Bước tiếp theo

Chạy **E0** (06 §2). Phần lớn chỉ là chấm điểm hoặc inference trên checkpoint đã có, khoảng 1–2 ngày,
chưa cần viết method mới. Đặc biệt:

- **E0.8:** gate không cần train cho ý tưởng anchored cũ;
- **E0.6:** bảng độ nhạy metric.

Sau đó cài các mốc M1–M3 ở 08.

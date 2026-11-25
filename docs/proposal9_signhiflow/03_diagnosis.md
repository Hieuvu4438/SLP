# 03 — Chẩn đoán: vì sao MotionHiFlow-SLP thua SignNMFlow, và metric thực sự thưởng cho điều gì

Tài liệu này trả lời hai câu hỏi trước khi bàn đến thiết kế:

1. Khoảng cách giữa `methods/motionhiflow_slp` (MotionHiFlow chuyển nguyên dạng sang SLP) và
   SignNMFlow đến từ đâu?
2. Muốn vượt SignNMFlow thì phải thay đổi điều gì về **bản chất**, và điều gì chỉ là sửa lỗi kỹ thuật?

Những điều chưa chắc chắn được nêu kèm một phép thử rẻ để kiểm chứng, đánh mã E0.x và trình bày chi
tiết ở 06.

---

## 1. Số liệu đã có trong repo (Phoenix-2014T, seed 0)

| Run | Split | JPE body | JPE hand | PA body (code) | PA body (ua) | PA hand |
|---|---|---|---|---|---|---|
| MotionHiFlow-SLP `motionhiflow_phoenix_opt` | dev | 5.75 | 7.46 | 6.46 | 5.07 | 1.47 |
| MotionHiFlow-SLP `motionhiflow_phoenix_improve_three` | dev | 5.76 | 7.43 | 6.47 | 5.09 | 1.47 |
| SignNMFlow reimpl v2 | dev | 5.17 | 6.98 | 5.88 | 4.59 | 1.37 |
| SignNMFlow reimpl v2 | test | 4.93 | 6.54 | 5.56 | 4.35 | 1.30 |
| **SignNMFlow paper** | test | **4.37** | **5.57** | 3.85 | (3.85) | **1.07** |
| SOKE paper | test | 6.04 | 7.72 | 4.77 | (4.77) | 1.38 |
| Pose trung bình tĩnh (memory dự án, v1) | dev | 4.70 | 7.51 | – | – | – |
| SignNMFlow reimpl v1, NFE 1 / 5 / 20 (100 clip dev) | dev | 4.89 / 5.36 / 5.64 | 7.06 / 7.42 / 8.05 | – | – | 1.31 / 1.37 / 1.60 |

Nguồn: `outputs/runs/motionhiflow_phoenix_*/…/metrics.json`, `methods/signmflow/README.md` và
`results/test_access_log.jsonl`. Người dùng cho biết MotionHiFlow-SLP trên Phoenix test tốt hơn SOKE
và SignSparK, nhưng thua SignNMFlow ở mọi metric. Tài liệu này không có số test của MotionHiFlow-SLP;
các dòng dev ở trên là số được dùng để phân tích.

**Ba nhận xét định lượng:**

1. **Cùng evaluator, trên dev:** MotionHiFlow-SLP thua SignNMFlow-reimpl khoảng +0.58 body (11%),
   +0.45 hand (6%), +0.48 PA-body(ua) và +0.10 PA-hand (7%).
2. **Bản reimpl SignNMFlow tự nó cũng thua paper khá xa trên test:** +0.56 body (13%), +0.97 hand
   (17%), +0.23 PA-hand (21%). Ít nhất một phần con số trong paper không tái lập được từ những gì
   paper mô tả (methods/signmflow README liệt kê 13 điểm paper không nói rõ). Muốn vượt *con số báo
   cáo*, ta phải đạt một mô hình tốt hơn đáng kể so với bản reimpl tốt nhất đang có trong repo.
3. **Metric phân biệt yếu ở phần body.** Một pose tĩnh đạt 4.70 body (dev), tức chỉ cách SignNMFlow
   paper (test) 0.33. Ở phần tay, pose tĩnh đạt 7.51 so với 5.57, nên đây mới là chỗ metric thật sự
   phân biệt. Ngoài ra **càng nhiều NFE thì DTW càng tệ**. Mẫu "đúng phân phối" hơn bị phạt; output
   gần posterior mean được thưởng.

---

## 2. Phân rã khoảng cách MotionHiFlow-SLP → SignNMFlow: giả thuyết và phép thử

| # | Khác biệt | MotionHiFlow-SLP | SignNMFlow | Bằng chứng hướng ảnh hưởng | Phép thử rẻ |
|---|---|---|---|---|---|
| H1 | Target dự đoán | velocity | target ($x_1$) | SignNMFlow Tab. 5 (CSL): velocity → target cho body 6.44 → 5.73 trên path của họ; 7.43 → 6.70 trên CFM | E1-a: cùng model, chỉ đổi sang x-pred |
| H2 | Chế độ sampling | 12 bước Euler, CFG 2.0, gần với sample | 5 bước, mỗi bước dự đoán lại $\hat x_1$, gần với mean | Diagnostics NFE 1/5/20 ở trên; 05-T5 | E0.2: chạy lại checkpoint MotionHiFlow-SLP với steps ∈ {1, 3, 6, 12}, CFG ∈ {1, 1.5, 2} |
| H3 | Trần của codec | VAE 4 part × 32-d, stride 2, L1 trên feature chuẩn hoá | không có codec (không gian thô) | Metric độ chính xác bị chặn dưới bởi lỗi tái tạo (InterCMDM, ACMDM) | E0.1: chấm `codec.decode(codec.encode(GT))` bằng evaluator chung. Nếu PA-hand tái tạo > 0.3 thì codec là nút thắt |
| H4 | Không gian của loss | L2 trên latent | L2 trên quaternion theo khớp (tổng theo kênh) | Cả hai đều không khớp metric vị trí (PIDiffSign, vMDM) | (đưa vào E1) |
| H5 | Text encoder | mBART đóng băng | mBART fine-tune | SOKE và SignNMFlow đều fine-tune; repo ghi nhận collapse nếu LR cao | E1-b |
| H6 | Dung lượng và ngân sách huấn luyện | width 384, 9 block (~20–25M); 40k update × batch 32 | decoder 8 lớp, 100M; 300 epoch × batch 256 | MotionHiFlow Tab. 3: kiến trúc ảnh hưởng nhiều hơn hierarchy | E1-c: scale lên 512 / 14 block |
| H7 | Biểu diễn rotation | latent từ axis-angle | quaternion + geodesic | SignNMFlow Tab. 3: GE −2.9 All. Baseline 6D của họ không có FK loss | A3 |
| H8 | Hierarchy | 3 scale | không có | MotionHiFlow: lợi ích nhỏ (FID .051 → .032) | `motionhiflow_phoenix_improve_single` (đã có recipe) |

**Đánh giá (suy luận, cần kiểm chứng).** H1, H2, H3 và H5 nhiều khả năng giải thích phần lớn khoảng
cách. Đây đều là **lựa chọn kỹ thuật**, không phải hạn chế cốt lõi của ý tưởng hierarchical flow. Hệ
quả quan trọng cho bài báo: nếu chỉ sửa H1–H6 thì ta có một "MotionHiFlow + vá lỗi", **không đủ
novelty** (đúng như người dùng lo ngại). Novelty phải đến từ một thay đổi có cơ sở trong *vai trò của
hierarchy* đối với SLP. 04/05 đề xuất thay đổi đó: hierarchy không còn là sampler từ nhiễu mà trở
thành **transport từ một ước lượng** (Estimate-then-Transport).

---

## 3. SignNMFlow mạnh vì điều gì? Phản biện

SignNMFlow lý giải lợi thế của mình bằng hình học: các path trên manifold. Có ba cách đọc khác cũng
nhất quán với dữ liệu của họ.

1. **Target-prediction cộng ít NFE cộng re-planning gần với ước lượng posterior mean.** Mỗi bước sampler
   dự đoán lại $\hat x_1$ rồi tiến theo path. Với NFE = 5, output nằm gần mean hơn so với một sampler
   20 bước. Trong repo, NFE 1 tốt hơn NFE 5. Paper không báo cáo NFE = 1, nên không biết họ đã chọn
   NFE 5 vì tốt nhất trên metric hay vì lý do khác (Table 8 trong paper là về NFE, repo chưa chạy).
2. **Một phần lợi thế của "geometry" có thể đến từ việc thiếu FK loss ở baseline.** Baseline 6D trong
   Table 3 dùng linear CFM với L2 trên feature. Nó không được giám sát trong không gian vị trí. Nếu
   baseline 6D có FK loss, khoảng 6D → quaternion+GE có thể thu hẹp. Đây là giả thuyết, kiểm chứng
   bằng A3 ở 06.
3. **Một số điểm không xác định được trong paper có thể làm lệch con số:**
   - Cách chọn thời lượng không được mô tả. Nếu họ dùng độ dài GT thì metric DTW tích luỹ có lợi; theo
     repo, độ dài GT cải thiện khoảng 0.2 / 0.2.
   - Cách chọn model không được báo cáo.
   - Không rõ PA-body căn chỉnh theo upper-body hay toàn thân. Repo cho thấy biến thể "ua" mới cho
     PA < JPE như mọi bảng đã công bố.
   - Back-translator không công khai.

**Kết luận chiến lược.** Ta không cần đánh bại SignNMFlow *bằng hình học*. Cần làm ba việc:

- **(a)** Đạt chất lượng ước lượng vượt trội ở tầng thô.
- **(b)** Giữ articulation sắc ở tầng mịn để không mất B-T.
- **(c)** Giám sát đúng không gian mà metric đo.

Hình học quay được giải quyết ở biểu diễn (6D hoặc exp-coords) cộng FK loss (05-T7). Lập luận này rẻ
hơn và có thể kiểm chứng.

---

## 4. Metric thưởng cho điều gì, và hệ quả cho thiết kế

1. **Chi phí tích luỹ** (05-T6). Output ngắn hơn hoặc tĩnh hơn được lợi. Thiết kế phải được đánh giá
   thêm bằng nDTW, tỉ lệ tốc độ và gia tốc, và B-T, để chứng minh lợi ích không đến từ việc "co lại".
2. **Một reference duy nhất** (05-T5). Ước lượng tốt hơn sampling ở những phần dự đoán được. Sampling
   chỉ có lợi ở những phần mà mean bị nhoè và người xem hoặc back-translator cần độ sắc. Đây là cơ sở
   cho Estimate-then-Transport, và cho việc chọn điểm vận hành $\Lambda$ một cách tường minh.
3. **Căn chỉnh của từng phần:**
   - JPE-hand trừ vị trí cổ tay nhưng **giữ hướng thế giới**, nên sai lệch xoay của chuỗi
     vai → khuỷu → cổ tay bị tính vào lỗi tay.
   - PA-hand dùng Procrustes theo từng cặp frame, nên **chỉ đo hình dạng bàn tay**.
   - JPE-body căn theo pelvis, gồm cả landmark mặt (tai, mắt, mũi), nên hướng đầu cũng được tính.

   Đây chính là các thành phần của PES (04 §5.2).
4. **DTW tha thứ cho lệch nhịp.** Mean theo từng frame không phải ước lượng tối ưu; DTW-barycenter
   hoặc medoid tốt hơn. Từ đó có tuỳ chọn soft-DTW cho μ-net (A9). Lưu ý thêm: metric lấy khoảng cách
   **không bình phương**, nên ước lượng tối ưu theo frame là geometric median (05-T2.2).

---

## 5. Hai tầng so sánh (bắt buộc trong báo cáo)

- **Tier A, có kiểm soát** (khẳng định khoa học chính). SignHiFlow so với SignNMFlow-reimpl v2, so với
  MotionHiFlow-SLP, so với baseline phẳng E1, với:
  - cùng dữ liệu và evaluator (`src/slp/evaluation`);
  - cùng cách chọn thời lượng: độ dài dự đoán, kèm một dòng chẩn đoán dùng độ dài GT;
  - cùng ngân sách tuning;
  - ≥ 3 seed, paired bootstrap CI (`metrics.paired_bootstrap_ci`).
- **Tier B, so với số đã công bố.** Bảng giống Table 1/2 của SignNMFlow, ghi rõ:
  - con số của người khác là tự báo cáo, đo bằng code có thể khác;
  - B-T của người khác dùng back-translator không công khai, còn B-T của ta dùng back-translator
    trong repo, có kèm trần GT;
  - dòng nào có retrieval (Track R) thì ghi riêng.

**Mục tiêu định lượng** là đích hướng tới, **không phải dự báo**. Lý do cho từng con số ở 06 §6.

| Phoenix test | JPE body | JPE hand | PA body | PA hand |
|---|---|---|---|---|
| SignNMFlow-reimpl v2 (đã đo) | 4.93 | 6.54 | 4.35 (ua) | 1.30 |
| SignNMFlow paper | 4.37 | 5.57 | 3.85 | 1.07 |
| **Mục tiêu SignHiFlow (không retrieval)** | ≤ 4.35 | ≤ 5.55 | ≤ 3.85 | ≤ 1.05 |
| Mục tiêu SignHiFlow-R (có retrieval) | ≤ 3.9 | ≤ 5.0 | ≤ 3.6 | ≤ 1.05 |

Muốn đạt mục tiêu không retrieval, so với bản reimpl tốt nhất của repo cần giảm hand JPE khoảng 15%
và PA-hand khoảng 19%. Mức này tương đương bước nhảy SOKE → SignNMFlow, nên là tham vọng. Nếu sau
E2 (06) vẫn không thấy tiến tới mục tiêu, kế hoạch dự phòng là Track R. Bằng chứng: SignSparK
GF → SR giảm hand 32% trên Phoenix.

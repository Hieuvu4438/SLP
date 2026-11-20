# 01 — Phản biện SignNMFlow (arXiv 2609.33339v1)

Tài liệu này đóng vai phản biện (reviewer) độc lập cho paper *Naturalness-guided Manifold Flow Matching for Sign Language Production* (He và cộng sự, 27/09/2026). Mỗi nhận định được gắn nhãn:

- **[Đo]**: đã kiểm chứng bằng probe chạy trên checkpoint v2 của bản tái lập (`probes/`).
- **[Toán]**: suy ra trực tiếp từ công thức trong paper.
- **[Văn bản]**: dựa trên những gì paper viết hoặc không viết; có nêu mục đã đọc.
- **[Suy luận]**: giả thuyết hợp lý, chưa kiểm chứng.

Probe dùng checkpoint v2 của Phoenix và CSL: seed 0, prototype K = 500, σ = 0.5, correction đã huấn luyện 100 epoch theo đúng paper. Probe chỉ dùng split val, và train làm đối chứng ở một chỗ.

> **Lịch sử chỉnh sửa.** Bản nháp đầu có hai con số sai cách hiểu: "lệch không gian 0.28 rad" và "đi qua vùng mật độ thấp hơn, Δlog m ≈ −0.6". Cả hai so sánh hai đường *tại cùng t*, nên chủ yếu đo độ trễ thời gian. Phản biện nội bộ (`05`) đã chỉ ra lỗi này, và các con số đã được thay bằng phép đo tại cùng tiến độ. s cũng được sửa từ 2.75 thành 2.5 ± 0.3.

---

## Tóm tắt

1. **Dưới chính mục tiêu Eq. 10 của paper, "naturalness-guided interpolation" không bẻ cong đường đi. Nó chỉ đổi tốc độ đi trên geodesic.** [Đo]
   - Năng lượng của đường học được là 23.76 (Phoenix) và 23.75 (CSL). Con số này bằng cận dưới của *mọi phép tái tham số hoá thời gian thuần tuý* của geodesic: 23.66–23.92 trên Phoenix, 23.89–24.16 trên CSL.
   - Vậy bẻ cong đường không đem lại lợi ích nào về năng lượng. Đây là tối ưu của họ hàm, không phải tình cờ do huấn luyện.
   - Khoảng cách từ đường học được tới *đường cong* geodesic (min theo tiến độ) chỉ 0.04–0.08 rad cung quaternion (trung vị, t = 0.3–0.7), tức khoảng 5–9° góc quay.
   - Tiến độ κ(t) khớp với time-shift kiểu SD3 với s ≈ 2.5 ± 0.3. Fit trực tiếp trên κ cho 2.55 (Phoenix) và 2.54 (CSL). Theo bộ phận: body 2.63, tay 2.50–2.55, hàm 2.83. Theo từng mẫu và joint, p10–p90 là 1.85–3.15.
   - Về năng lượng, mọi s trong khoảng [2, 2.75] là tương đương.
2. **m(x) bị triệt tiêu, chỉ còn moment bậc hai chuẩn hoá Σ(x).** [Toán + Đo]
   - Chính Eq. 43–44 của paper viết G⁻¹ = Σ(x) + (ρ/m)·I. Với ρ = 1e−6 mà paper công bố, số hạng ρ/m chiếm dưới 1e−4 năng lượng (p95).
   - Dọc đường đi, log10 m có p5 ≈ −1.6 (m ≈ 0.025). Với ρ ≲ 1e−3, mật độ trơ hoàn toàn với mục tiêu.
   - Do đó cách đọc "reweights the kinetic energy by the inverse motion density" trong App. C.3, và phép chứng minh "density consistency", không mô tả đúng vật thể được tối ưu.
   - Chỉ khối lượng tuyệt đối m bị triệt tiêu; trọng số tương đối của prototype (π_k) và vị trí prototype vẫn còn.
3. **Đường học được *không* đi về vùng dày đặc như Fig. 1 vẽ.** [Đo] Tại cùng tiến độ, Δlog m (learned − geodesic) có trung vị −0.004 đến −0.017. Nó chỉ đi chậm hơn ở gần nguồn nhiễu.
4. **Trùng lặp với Metric Flow Matching (Kapuśniak và cộng sự, NeurIPS 2024) mà không trích dẫn.** [Văn bản]
   - MFM đã dùng đúng interpolant (đường cơ sở) + t(1−t)·φ (Eq. 4) và huấn luyện φ bằng kinetic energy dưới metric phụ thuộc dữ liệu (Eq. 6). Metric của MFM xây từ LAND (phương sai cục bộ có trọng số kernel) hoặc RBF tại tâm K-means.
   - SignNMFlow khác ở ba điểm:
     - đường cơ sở là geodesic trên (S³)^J;
     - metric dùng moment bậc hai *chuẩn hoá*, chiều ngược với MFM vốn nghịch đảo phương sai *không chuẩn hoá*;
     - bối cảnh là target-prediction cho SLP.
   - Đã kiểm tra: danh mục tài liệu tham khảo, Sec. 2.2 và App. A.2 không nhắc MFM.
5. **Đánh giá thiếu baseline tầm thường, phương sai và thống kê động học. Họ metric DTW đã được chính báo cáo SLRTP 2025 (nhóm tác giả đồng tác giả) ghi nhận là "favor less articulated or longer sequences".** [Văn bản + Đo]
   - Trên v2 (Phoenix dev, 150 chuỗi), mô hình thắng tư thế trung bình ở cả 4 cột. Điều này khác với v1, nơi mean pose thắng ở JPE body.
   - Nhưng các phép biến đổi không dùng văn bản lại cải thiện metric. Co 25% về tư thế trung bình thắng dự đoán ở cả 4 cột. Làm mượt σ = 3 frame xoá khoảng 75–80% tốc độ mà vẫn thắng 3/4 cột (mục C.2).
6. **Ablation có confound.** [Toán + Đo]
   - Bước 6D → MR đổi đồng thời biểu diễn, nguồn p0 (Gaussian so với uniform-hemisphere) và lịch nhiễu (P5: tại t = 0.7, 6D-Gaussian còn 42° tới x1 so với 36° của quaternion tuyến tính).
   - Bước MR → GE *không* bị nhiễu bởi lịch (quaternion tuyến tính renorm và geodesic có lịch góc gần như trùng nhau). Confound ở đây là chuẩn ngoài manifold của x_t tuyến tính, mà generator nhận như đầu vào.
   - Bước GE → NG là đổi lịch nhiễu (mục 1).

---

## A. Phản biện lý thuyết

### A.1 "Property 1" đúng nhưng không liên quan trực tiếp tới thiết lập target-prediction [Toán]

- Property 1 (Eq. 1–2, App. B) chứng minh nội suy tuyến tính rời khỏi SO(3) và S³. Điều này hiển nhiên đúng.
- Nhưng với `prediction=target`, loss là ‖x̂1 − x1‖² (Eq. 13). Vận tốc của path *không bao giờ* đi vào loss của generator.
- Path chỉ tác động lên generator qua hai kênh:
  - (i) phân phối các cặp (x_t, t) thấy khi huấn luyện;
  - (ii) các trạng thái mà re-planned sampler đi qua.
- x_t chỉ là *đầu vào* của mạng. "x_t off-manifold" vì vậy không tự động làm giảm chất lượng; điều quan trọng là phân phối x_t, tức tỉ lệ tín hiệu/nhiễu theo t.
- Paper không có thí nghiệm tách hai yếu tố "hợp lệ hình học" và "phân phối mức nhiễu". Đã kiểm tra Sec. 4.3, Tables 3 và 5.

### A.2 Năng lượng thực sự được tối ưu [Toán + Đo]

Từ Eq. 8, 42 và 44, với w̃_k = m·π_k:

```
v^T G(x)^-1 v = Σ_k π_k(x) ⟨δ_k, v⟩²  +  (ρ/m(x)) ‖v‖²         (= v^T [Σ(x) + (ρ/m) I] v, Eq. 44)
```

- **Paper đã viết đúng công thức này (Eq. 44).** Phê bình ở đây nhắm vào *diễn giải* trong App. C.3 ("enters only through the prefactor 1/m", "inverse motion density"), không nhắm vào toán.
- Với ρ = 1e−6, số hạng thứ hai không đáng kể [Đo, P1]: tỉ lệ năng lượng có p95 ≈ 1e−4, log10 m ≥ −1.6 dọc đường đi.
- **Dị hướng [Đo, P2].**
  - Gần nguồn nhiễu (t < 0.3), mọi δ_k cùng chỉ về phía cụm dữ liệu, nên Σ ≈ d²uuᵀ. Hướng x_t → x1 nằm ở vị trí trung vị 0.97–0.99 trong khoảng [λ_min, λ_max].
  - Với t > 0.7 con số này giảm còn 0.60–0.79 (tay–body).
  - Đây gần như là tính chất chung của *mọi* metric dựng từ moment bậc hai chuẩn hoá quanh x khi x ở xa dữ liệu. Nó không phải một "lỗi chiều" riêng của paper.
- **Hệ quả.** Chi phí trên một đơn vị tốc độ cao ở gần nguồn và thấp ở gần dữ liệu. Với chi phí dạng a(κ)·κ'², nghiệm tối ưu có κ' ∝ 1/a, tức đi chậm ở đầu và nhanh ở cuối. Đây là một phép tái tham số hoá thời gian.
  - Một metric kiểu MFM/LAND cũng đắt ở xa dữ liệu, nên rất có thể cũng sinh ra time-shift tương tự. Vì vậy câu hỏi "chiều metric nào đúng" ít quan trọng hơn câu hỏi "lịch nhiễu nào tốt".
- **Định nghĩa chuẩn bất thường** [Văn bản]. Eq. 9 định nghĩa ‖v‖_G := sqrt(vᵀG⁻¹v), trong khi quy ước Riemann thông thường là vᵀGv. Paper không biện luận lựa chọn này (đã kiểm tra Sec. 3.2 và App. C.3).

### A.3 Điều mà ε_θ thực sự học [Đo]

Các số liệu dưới đây được đo trên 600–2000 frame ngẫu nhiên của val. Trên Phoenix train, kết quả giống val tới 3 chữ số thập phân.

| Phép đo | Phoenix | CSL |
|---|---|---|
| Năng lượng: geodesic / learned / cận dưới tái tham số hoá (chung frame – theo joint) | 30.53 / **23.76** / 23.92–23.66 | 30.84 / **23.75** / 24.16–23.89 |
| Năng lượng geodesic + SD3 shift, s = 1.5 / 2 / 2.75 / 3.5 / 5 | 26.14 / 24.76 / 24.76 / 25.81 / 29.17 | 26.40 / 25.00 / 24.99 / 26.05 / 29.44 |
| κ(t) trung vị tại t = 0.1/0.3/0.5/0.7/0.9 | .053/.170/.305/.476/.730 | .054/.172/.308/.478/.728 |
| κ đơn điệu theo t (theo mẫu, theo joint) | 99.9% | — |
| Khoảng cách tới đường cong geodesic, trung vị (t = 0.3/0.5/0.7) | 0.042/0.067/0.078 rad | 0.041/0.068/0.083 rad |
| (Để so sánh) khoảng cách tại cùng t | 0.16/0.24/0.28 rad | 0.16/0.24/0.28 rad |
| Δlog m tại cùng tiến độ, trung vị | −0.004 … −0.017 | −0.003 … −0.016 |
| cos(ε, Log_{x0}x1) trung vị | −0.94 … −0.97 | −0.92 … −0.97 |
| s fit trên κ: toàn bộ / body / tay / hàm; p10–p90 theo mẫu | 2.56 / 2.63 / 2.50–2.55 / 2.83; 1.85–3.15 | 2.54 / 2.61 / 2.47–2.54 / 2.82 |

**Kết luận.**

- Đường "naturalness" ≈ Exp_{x0}(κ(t)·Log_{x0}x1), với κ(t) ≈ 1 − n_s(t), n_s(t) = s(1−t)/(1 + (s−1)(1−t)), s ≈ 2.5. Đây là phép dịch timestep của SD3 (Esser và cộng sự 2024).
- Ngoài ra có hai phần dư:
  - (a) lệch vuông góc khoảng 0.07 rad;
  - (b) κ không đồng nhất giữa các mẫu và joint. Các joint trong cùng một frame nằm ở mức nhiễu khác nhau, giống một kiểu tăng cường "mức nhiễu theo token".
- Thí nghiệm E1 trong `04` tách ba thành phần: lịch toàn cục, tính không đồng nhất, và phần bẻ cong.

### A.4 Các điểm lý thuyết khác

- **[Toán] Prototype và kernel làm việc theo từng joint độc lập** (App. C.2). "Motion distribution" chỉ là phân phối biên của từng khớp tại một frame. Không có hiệp đồng ngón tay, quan hệ hai tay hay thời gian.
- **[Toán] ε_θ là MLP theo từng frame trên (t, x0, x1).** x0 là nhiễu độc lập theo frame, nên đường đi không có thông tin chuỗi hay văn bản.
- **[Đo] Nguồn nhiễu p0 rất xa dữ liệu.**
  - Góc quay trung bình p0 → x1 là 126.5°. Tư thế trung bình chỉ cách x1 21°. Wrapped normal σ = 0.25 cách 51°.
  - Dữ liệu tay có góc quay p95 là 74–84° quanh tư thế gốc.
  - Phần thời gian mà path ở xa dữ liệu hơn mức này phụ thuộc lịch nhiễu. Với naturalness, x_t còn cách x1 trên 80° cho tới t ≈ 0.6 (P5).
- **[Văn bản] Năng lượng của path có điều kiện ≠ năng lượng của path biên** (Shaul và cộng sự 2023). Paper không đo độ cong của trường vận tốc biên hay lỗi khi lấy mẫu ít bước.

---

## B. Phản biện thực nghiệm

### B.1 Ablation và đối chứng

| Vấn đề | Chi tiết | Hệ quả |
|---|---|---|
| Table 3, bước 6D → MR | Đổi biểu diễn, p0 và lịch nhiễu cùng lúc. | Không quy được cho biểu diễn. |
| Table 3, bước MR → GE | Lịch góc gần như trùng nhau, nhưng x_t tuyến tính *không* được renormalize, nên mạng nhận chuẩn ‖x_t‖ < 1 như đầu vào. | Cần đối chứng "linear + renorm" để xác định lợi ích GE đến từ hình học hay từ đặc trưng đầu vào. |
| Table 3, bước GE → NG | Xem A.3: tương đương đổi lịch nhiễu. | Thiếu đối chứng "geodesic + time-shift / logit-normal" (thay đổi một dòng code; SD3, Chen 2023). |
| Table 4 (σ) là một phép thử có thể bác bỏ mà chưa dùng | σ = 0.25 cho 19.41 (*tệ hơn GE* 18.31); σ = 0.75 cho 17.58. Nếu NG chỉ là time-shift, mỗi σ phải sinh ra một s(σ), và thứ tự metric phải giải thích được bằng s. | Kiểm tra rẻ (chỉ cần stage correction). Ghi là E1-0 trong `04`. |
| Ablation chỉ trên CSL | Sec. 4.3. | Không biết hiệu ứng có tổng quát không. |
| Không có seed hay khoảng tin cậy | Table 4: K = 250/500/750 cho hand PA 1.32/1.30/1.29. Lần chạy lại SOKE của GARD lệch 0.01–0.08 so với số SOKE tự báo. | Hiệu ứng 0.1–0.3 có thể nằm trong nhiễu. |
| Table 8 | Tăng NFE thì lỗi tăng nhẹ (một run, +0.7%). Bản thân không đủ làm bằng chứng. Nhưng bản tái lập v1 cho NFE 1 → 5 → 20 ứng với DTW-JPE body 4.89 → 5.36 → 5.64. | Nhất quán với việc metric ưu tiên ước lượng gần trung bình có điều kiện. |
| Bảng so sánh chọn lọc | Chỉ có SignSparK bản gloss-free. Bản retrieval tốt hơn SignNMFlow ở 3/4 cột trên Phoenix (pseudo-GT khác). GARD được trích dẫn nhưng không có trong bảng. M3T và SignMask vắng mặt. | Vị trí SOTA bị phóng đại. |
| Cột "All" | Không định nghĩa; bản tái lập không tái tạo được. | Không kiểm chứng được Tables 3–5 và 8. |

### B.2 Đánh giá

- **DTW tích luỹ, không chuẩn hoá theo độ dài** (`metrics.dtw_cost`, giao thức SOKE).
  - Đường warping dài ít nhất max(T_pred, T_gt). Dự đoán quá dài làm tăng chi phí; dự đoán ngắn không làm giảm được "sàn" theo độ dài GT.
  - Paper không nói độ dài đầu ra được chọn thế nào (đã kiểm tra Sec. 4, App. C.5 và D.1).
- **B-T trên How2Sign khó tin** [Văn bản + tài liệu].
  - SignNMFlow báo 16.88 và SignNMFlow-M báo 17.49, tính từ chuyển động *sinh ra*.
  - Trần GT đo bằng các back-translator khác chỉ khoảng 2–2.5 (M3T; Hong & Košecká 2026). Dịch từ video thật với tiền huấn luyện lớn đạt khoảng 12–16.
  - Không có dòng GT, và back-translator của SOKE không công bố.
- **Bối cảnh từ chính nhóm tác giả.** Báo cáo SLRTP 2025 (nhóm tác giả SignNMFlow đồng tác giả; đội hfut-lmc hạng 2) kết luận "DTW-MJE tended to favor methods that produced less articulated or longer sequences". Paper SignNMFlow không thảo luận rủi ro này.
- **User study.** 5 người ký CSL × 18 mẫu, lựa chọn bắt buộc. Không mô tả cách chọn mẫu, không có kiểm định thống kê, không có người ký ASL hay DGS.
- **Hiệu năng (Table 7).** Không rõ độ trễ có tính mBART và length head hay không. Số tham số 233.2M không tái tạo được (bản tái lập có 272.4M).

### B.3 Những gì paper không nói nhưng ảnh hưởng metric (mục đã đọc: Sec. 4, App. C.5, D.1, Table 6)

Paper không nêu: cách chọn độ dài đầu ra, quy tắc chọn checkpoint, EMA, learning rate của text encoder, cách reduce loss theo kênh, và định nghĩa PA-body.

Trên bản tái lập, val flow loss nhỏ nhất ở epoch 100 rồi tăng tới epoch 300:

| Dataset | Epoch 100 | Epoch 300 |
|---|---|---|
| Phoenix | 10.30 | 12.56 |
| CSL | 8.57 | 11.26 |

Quy tắc chọn checkpoint có thể tạo khác biệt cỡ hiệu ứng NG. Điều này chưa được đo trên DTW; đó là thí nghiệm E0.1.

---

## C. Điều gì vẫn đứng vững

### C.1 Ghi nhận và hướng đi

- **Target-prediction kết hợp re-planned sampler** có cơ sở tốt.
  - Cấu trúc giống FrameFlow; phù hợp lập luận của JiT (Li & He 2025).
  - Table 5 của paper ủng hộ điểm này.
- **Biểu diễn trên (S³)^J với Exp/Log là đúng về toán.** Bản tái lập có test so `velocity_chain_rule` với autodiff.
- **Bản tái lập v2 là baseline mạnh.** Phoenix dev đạt JPE 5.17/6.98, PA-hand 1.37, tốt hơn số SOKE paper báo trên test (6.04/7.72/1.38).
- **Hệ quả cho cải tiến.** Pipeline gồm transformer, mBART và target-prediction là nền tốt. Thành phần "naturalness" nhiều khả năng tương đương một lựa chọn lịch nhiễu (H1, chưa kiểm chứng trên chất lượng sinh).
- **Hướng dồn ngân sách.** Những yếu tố mà tài liệu cho thấy tác động lớn và paper bỏ qua:
  - loss trong không gian vị trí;
  - phân phối mức nhiễu khi huấn luyện và lưới suy diễn;
  - chọn checkpoint và EMA;
  - CFG (khi đã có thước đo ngữ nghĩa đáng tin);
  - prior có thông tin.

### C.2 Probe độ nhạy metric (Phoenix dev, 150 chuỗi đầu, v2 seed 0)

Phoenix dev, 150 chuỗi đầu, checkpoint v2 seed 0, giao thức SOKE: body = 12 khớp thân trên, căn pelvis; tay = 21 joint, căn cổ tay; PA = Procrustes; DTW tích luỹ. Nguồn: `probes/probe_metrics_phoenix_smplx.json`. Mọi biến thể ngoài `GT câu khác` đều **không dùng văn bản**.

| Biến thể | JPE body | JPE hand | PA body (ua) | PA hand | Tốc độ body / ngón (mm/frame) |
|---|---|---|---|---|---|
| Dự đoán v2 | 5.04 | 6.87 | 4.47 | 1.334 | 4.45 / 23.5 |
| Mean pose tĩnh | 5.53 | 7.95 | 4.87 | 1.542 | 0 / 0 |
| Co 25% về mean pose | **4.77** | **6.70** | **4.24** | **1.290** | — |
| Co 50% về mean pose | **4.77** | 6.92 | **4.24** | **1.323** | — |
| Làm mượt Gaussian σ = 1 frame | **5.01** | **6.86** | **4.45** | **1.317** | — |
| Làm mượt Gaussian σ = 3 frame | **4.96** | 6.91 | **4.41** | **1.315** | **1.22 / 5.6** |
| GT của câu khác | 6.82 | 8.81 | 6.11 | 1.697 | — |
| (GT) | 0 | 0 | 0 | 0 | 4.86 / 26.1 |

Số **đậm** là tốt hơn dự đoán gốc.

**Đọc bảng.**

1. Metric *có* phân biệt đúng câu với sai câu: GT câu khác kém xa. Mô hình v2 cũng thắng mean pose tĩnh ở cả 4 cột.
2. Nhưng các phép biến đổi không chứa nội dung, kéo dự đoán về trung bình hoặc làm mất chuyển động, lại *cải thiện* metric:
   - co 25% về mean pose thắng dự đoán gốc ở cả 4 cột;
   - làm mượt σ = 3 xoá khoảng 75–80% tốc độ mà vẫn thắng 3/4 cột.

   Đây là cùng hiện tượng mà báo cáo SLRTP 2025 mô tả. Mọi "cải thiện" DTW cỡ 0.05–0.3 vì vậy cần rào chắn động học và ngữ nghĩa trước khi được tin.
3. Tốc độ ngón tay của GT (26 mm/frame) phần lớn là dao động tần số cao. Làm mượt σ = 3 giảm nó xuống 5.6, gợi ý rằng pseudo-GT có nhiều giật do fitting.
   - Tỉ lệ tốc độ thô (0.90–0.92 cho v2) vì vậy là thước đo lẫn lộn.
   - Rào chắn M2 trong `04` phải so sánh trên tín hiệu đã lọc thông thấp, hoặc theo dải tần.
4. Sai số độ dài trung bình là 16.2 frame trên chuỗi dài khoảng 100 frame (dự đoán 99.4, GT 100.6).
   - Độ dài là một nguồn phương sai DTW đáng kể, nên các arm cần dùng chung độ dài (`04` §1.2).


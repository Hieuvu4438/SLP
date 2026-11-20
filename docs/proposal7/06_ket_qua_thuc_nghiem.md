# 06 — Kết quả thực nghiệm vòng 1: các chỉnh sửa chỉ ở khâu suy diễn (E0) và phép thử σ (E1-0)

**Câu hỏi.** Áp dụng các chỉnh sửa lên hàm của `signmflow` mà *không cần huấn luyện lại generator* thì có vượt được SignNMFlow v2 không? Các chỉnh sửa gồm:

- lưới thời gian của sampler;
- số bước NFE;
- thay path khi suy diễn;
- chọn checkpoint.

**Thiết lập.**

| Mục | Chi tiết |
|---|---|
| Dữ liệu | Phoenix-2014T dev, toàn bộ 519 chuỗi (split val; không chạm test) |
| Mô hình | Checkpoint v2 seed 0 (`outputs/runs/signmflow_v2_main_phoenix_smplx/.../checkpoints`), các snapshot `generator_ep{50..300}.pt` |
| Thiết bị | Suy diễn trên CPU, để không đụng job How2Sign đang chạy trên GPU |
| Script | `probes/probe_inference.py` → `probes/probe_inference_phoenix_smplx.json` (điểm theo từng chuỗi + bootstrap) |

Sampler trong script là `flow.sample` (Alg. 2) tổng quát hoá cho lưới t, NFE và path tuỳ ý. Kiểm chứng: biến thể `base_s0` tái tạo đúng số v2 trên dev (5.17 / 6.98 / 1.37).

**Kiểm soát.** Thiết kế theo `04` v2 §1:

- **Độ dài đầu ra cố định** cho mọi biến thể, lấy từ dự đoán đã lưu của v2.
- **Cùng nhiễu sampler** cho mọi biến thể (thiết kế cặp).
- **Bootstrap cặp theo chuỗi** (4000 mẫu) so với `base_s0`.
- **Rào chắn M2:** tốc độ khớp sau lọc thông thấp Gaussian σ = 2 frame. Body tính tương đối pelvis, ngón tay tương đối cổ tay. GT có tốc độ 2.22 / 8.35 mm/frame.
- **Rào chắn M3:** khoảng cách tới mean pose.
- **Đối chứng "co về trung bình" không dùng văn bản:** nhân dự đoán của base với (1−a).

Đọc kết quả: DTW thấp hơn là tốt hơn. Δ là biến thể trừ `base_s0`; khoảng tin cậy 95% nằm trong ngoặc.

## 1. Bảng kết quả

| Biến thể | JPE body | JPE hand | PA body (ua) | PA hand | Tốc độ body/ngón (mm/frame) | Tỉ lệ tốc độ so với GT |
|---|---|---|---|---|---|---|
| **base_s0** (v2, epoch 300, NFE 5, lưới đều) | 5.170 | 6.979 | 4.584 | 1.3685 | 1.45 / 6.38 | 0.65 / 0.76 |
| base_s1, s2 (seed sampler khác) | 5.185, 5.188 | 6.991, 7.009 | 4.603, 4.612 | 1.3689, 1.3669 | 1.47 / 6.44 | 0.66 / 0.77 |
| ep50 / ep100 / ep150 / ep200 | 5.479 / 5.373 / 5.274 / 5.199 | 7.937 / 7.348 / 7.241 / 7.048 | 4.917 / 4.789 / 4.718 / 4.622 | 1.555 / 1.430 / 1.418 / 1.391 | 1.01–1.42 / 4.8–6.2 | 0.46–0.64 / 0.57–0.75 |
| ep250 | 5.159 | 6.976 | 4.574 | 1.3632 | 1.43 / 6.33 | 0.65 / 0.76 |
| nfe1 | 4.935 | 6.849 | 4.383 | 1.3173 | 0.94 / 4.05 | 0.42 / 0.48 |
| nfe2 | 5.008 | 6.853 | 4.445 | 1.3151 | 1.17 / 5.02 | 0.53 / 0.60 |
| nfe10 | 5.308 | 7.247 | 4.719 | 1.4392 | 1.63 / 7.33 | 0.73 / 0.88 |
| geoshift (thay path NG bằng geodesic + shift s = 2.55 khi suy diễn) | 5.107 | 7.089 | 4.528 | 1.4191 | 1.44 / 6.84 | 0.65 / 0.82 |
| lưới t_k = (k/5)^0.7 (ít bước ở vùng nhiễu) | 5.190 | 7.051 | 4.610 | 1.3895 | 1.50 / 6.65 | 0.67 / 0.80 |
| lưới p1.5 | 5.142 | 6.912 | 4.555 | 1.3449 | 1.38 / 6.01 | 0.62 / 0.72 |
| lưới p2 | 5.110 | 6.877 | 4.525 | 1.3321 | 1.32 / 5.72 | 0.60 / 0.69 |
| lưới p3 | 5.066 | 6.847 | 4.487 | 1.3180 | 1.24 / 5.29 | 0.56 / 0.63 |
| ep250 + lưới p2 | 5.097 | 6.868 | 4.513 | 1.3289 | 1.30 / 5.63 | 0.59 / 0.67 |
| *Đối chứng:* co 5% về mean pose | 5.099 | 6.896 | 4.522 | 1.3511 | 1.38 / 6.09 | 0.62 / 0.73 |
| *Đối chứng:* co 10% | 5.039 | 6.835 | 4.470 | 1.3383 | 1.32 / 5.80 | 0.59 / 0.70 |
| *Đối chứng:* co 15% | 4.991 | 6.796 | 4.428 | 1.3300 | 1.25 / 5.51 | 0.56 / 0.66 |
| *Sàn:* mean pose tĩnh | 5.796 | 8.134 | 5.093 | 1.614 | 0 / 0 | 0 / 0 |
| *(Paper, test — không cùng split)* | 4.37 | 5.57 | 3.85 | 1.07 | — | — |

Δ so với `base_s0` (bootstrap cặp, 519 chuỗi), cho các biến thể chính:

| Biến thể | Δ JPE body | Δ JPE hand | Δ PA body | Δ PA hand |
|---|---|---|---|---|
| base_s1 (nhiễu sampler) | +0.015 [−0.020, 0.050] | +0.012 [−0.036, 0.059] | +0.018 [−0.015, 0.052] | 0.000 [−0.006, 0.007] |
| ep100 | +0.203 [0.134, 0.276] | +0.369 [0.284, 0.457] | +0.204 [0.145, 0.268] | +0.062 [0.049, 0.075] |
| ep250 | −0.011 [−0.024, 0.002] | −0.003 [−0.020, 0.014] | −0.010 [−0.021, 0.000] | −0.005 [−0.008, −0.003] |
| nfe1 | −0.235 [−0.270, −0.201] | −0.130 [−0.177, −0.085] | −0.201 [−0.232, −0.172] | −0.051 [−0.059, −0.044] |
| nfe10 | +0.138 [0.121, 0.155] | +0.268 [0.243, 0.294] | +0.134 [0.120, 0.150] | +0.071 [0.066, 0.075] |
| geoshift | −0.064 [−0.092, −0.034] | +0.110 [0.071, 0.150] | −0.057 [−0.083, −0.031] | +0.051 [0.044, 0.057] |
| lưới p2 | −0.060 [−0.072, −0.048] | −0.102 [−0.119, −0.086] | −0.059 [−0.069, −0.050] | −0.036 [−0.039, −0.034] |
| ep250 + lưới p2 | −0.073 [−0.090, −0.057] | −0.111 [−0.134, −0.088] | −0.072 [−0.085, −0.058] | −0.040 [−0.043, −0.036] |
| co 10% (không dùng văn bản) | −0.131 [−0.144, −0.120] | −0.144 [−0.158, −0.129] | −0.114 [−0.124, −0.105] | −0.030 [−0.032, −0.028] |

## 2. Diễn giải

1. **Có thể "vượt" v2 trên DTW chỉ bằng sửa sampler. Nhưng ở cùng mức chuyển động, không biến thể nào vượt được phép co không dùng văn bản.**
   - Lưới p2 (dồn bước suy diễn về vùng nhiễu) giảm cả 4 cột với khoảng tin cậy loại trừ 0. PA hand 1.369 → 1.332 (−2.7%); kết hợp với ep250 là 1.329.
   - Nhưng tốc độ chuyển động giảm 9–10%. Ở *cùng* tốc độ (co 10%: 1.32 / 5.80 so với lưới p2: 1.32 / 5.72), phép co tốt hơn ở JPE body, JPE hand và PA body. Lưới p2 chỉ hơn ở PA hand, khoảng 0.006.
   - Cặp lưới p3 và co 15% cho cùng kết luận (PA hand hơn 0.012).
   - **Kết luận:** gần như toàn bộ "cải thiện" DTW từ sampler là trung bình hoá. Phần dư thật sự, nếu có, chỉ ở PA hand và nhỏ hơn 0.015.
2. **Trục NFE cho thấy metric ngược chiều với độ sống động.**
   - NFE 1 có DTW tốt nhất nhưng tốc độ chỉ còn 0.42–0.48 so với GT.
   - NFE 10 có DTW kém nhất nhưng tốc độ gần GT nhất (0.73 / 0.88).
   - Điều này khớp với báo cáo SLRTP 2025 và với Table 8 của paper (NFE tăng thì DTW xấu đi).
   - **Hệ quả:** việc paper chọn NFE = 5 là một điểm trên đường đánh đổi giữa DTW và độ sống động, không phải "bão hoà chất lượng" như App. D.2 viết.
3. **Chọn checkpoint sớm làm xấu đi, ngược với kỳ vọng ban đầu (E0.1).**
   - Dù val flow loss nhỏ nhất ở epoch 100, DTW và tốc độ đều tốt dần theo epoch. Epoch 250 nhỉnh hơn 300 rất ít (PA hand −0.005, có ý nghĩa nhưng không đáng kể).
   - **Hệ quả:** val flow loss *không* phải proxy để chọn mô hình. Quy tắc "early selection" trong `04` E3 bị loại. Độ dài sàng lọc phải gần độ dài đầy đủ (≥ 250 epoch), nên ngân sách mỗi run sàng lọc khoảng 15–18 GPU-h thay vì 9.
4. **Đổi path khi suy diễn (E0.3): mô hình NG phụ thuộc vào path đúng của nó, nhất là ở tay.**
   - Thay path naturalness bằng geodesic + shift s = 2.55 (khớp tiến độ trung vị) cải thiện body (−0.06), nhưng làm *xấu* JPE hand (+0.11) và PA hand (+0.05). Tốc độ ngón tăng.
   - Vậy ở khâu suy diễn, cấu trúc của path NG không trung tính đối với mạng đã học theo nó. Đó là tính không đồng nhất giữa joint và mẫu (s theo mẫu p10–p90 = 1.85–3.15), cộng phần lệch khoảng 0.07 rad.
   - **H1 dạng mạnh** ("NG ≡ một time-shift vô hướng") bị bác bỏ ở *khâu suy diễn*. Câu hỏi ở *khâu huấn luyện* (geoshift huấn luyện từ đầu có bằng NG không) vẫn mở và cần E1-gate/E1-H1.
5. **Nhiễu do sampler nhỏ.** Giữa các seed, Δ ≤ 0.03 và khoảng tin cậy chứa 0. PA hand gần như không đổi (±0.007). Thiết kế cặp đủ mạnh để phát hiện hiệu ứng khoảng 0.01 ở PA hand.
6. **Bản thân v2 đã "kém sống động".** Tốc độ sau lọc thông thấp chỉ bằng 65% (body) và 76% (ngón) so với GT. Một cải tiến *thật* cần giảm DTW trong khi giữ nguyên hoặc tăng tỉ lệ này.

## 3. Trả lời "có tốt hơn SignNMFlow không?" (đến thời điểm này)

| Loại chỉnh sửa | Vượt v2 trên DTW? | Vượt một cách thật sự (qua rào chắn M2/M3 và đối chứng co về trung bình)? |
|---|---|---|
| Lưới suy diễn p2/p3, NFE 1–2 | Có (đến −0.24 JPE body, −0.05 PA hand) | **Không**: phép co không dùng văn bản ở cùng tốc độ làm tốt bằng hoặc hơn; PA hand chỉ dư < 0.015 |
| Chọn checkpoint sớm | Không (xấu hơn có ý nghĩa) | Không |
| Chọn epoch 250 | Rất nhẹ (PA hand −0.005) | Không đáng kể |
| Thay path NG → geoshift khi suy diễn | Pha trộn (body tốt hơn, tay xấu hơn) | Không |

**Kết luận vòng 1.** Không có chỉnh sửa *chỉ ở khâu suy diễn* nào vượt SignNMFlow v2 một cách thật sự. Có thể làm số DTW đẹp hơn số v2 (ví dụ lưới p2 + ep250: PA hand 1.329 so với 1.369), nhưng đó là đổi lấy chuyển động kém sống động hơn, điều mà một phép co không dùng văn bản cũng làm được.

**Điều này định hướng lại kế hoạch.**

- (a) Mọi so sánh từ nay phải báo cáo theo *đường biên DTW–tốc độ*, và so với đường "co về trung bình" như đối chứng bắt buộc.
- (b) Các đòn bẩy có khả năng cải thiện thật nằm ở khâu huấn luyện: FK loss (E4), CFG (E5), prior (E6), self-conditioning (E7). Chúng phải *đẩy đường biên*, tức có DTW tốt hơn ở cùng tốc độ.
- (c) Chọn checkpoint sớm bị loại; độ dài sàng lọc ≥ 250 epoch.

## 4. Phép thử σ (E1-0)

Script: `probes/probe_sigma.py` → `probes/probe_sigma_phoenix_smplx.json`.

**Thiết lập.**
- Chỉ huấn luyện correction ε_θ theo Alg. 1, dùng lại prototype v2 của Phoenix (K = 500).
- 4000 bước, batch 8, cùng seed và cùng cosine LR cho cả ba σ. Huấn luyện trên train, đo trên 2000 frame val.
- Giới hạn: bản đầy đủ của paper là khoảng 89k bước. Với 4000 bước, σ = 0.5 đạt s = 2.27, chưa tới 2.55 như checkpoint đầy đủ. Vì vậy chỉ so sánh *tương đối* giữa các σ ở cùng ngân sách.

| σ | κ(t) trung vị tại t = 0.1/0.3/0.5/0.7/0.9 | Lệch vuông góc trung vị (rad) | s fit (RMSE) | Paper Table 4 (CSL): DTW-JPE All / hand PA |
|---|---|---|---|---|
| 0.25 | 0.007 / 0.075 / 0.225 / 0.463 / 0.796 | 0.014–0.044 | 2.92 (0.036, khớp kém) | 19.41 / 1.57 (tệ hơn GE 18.31 / 1.41) |
| 0.5 | 0.057 / 0.167 / 0.303 / 0.502 / 0.801 | 0.007–0.032 | 2.27 (0.007) | 16.86 / 1.30 |
| 0.75 | 0.057 / 0.167 / 0.304 / 0.503 / 0.802 | 0.007–0.033 | 2.26 (0.006) | 17.58 / 1.32 |

**Diễn giải.**

1. **σ = 0.5 và σ = 0.75 cho path giống hệt nhau** (κ và phần lệch vuông góc trùng tới 3 chữ số).
   - Với target-prediction, generator chỉ "thấy" path qua phân phối x_t. Hai generator huấn luyện trên hai path này vì vậy phải có chất lượng như nhau, trừ nhiễu giữa các lần chạy.
   - Paper báo chênh lệch 0.72 DTW-JPE All và 0.02 hand PA giữa hai dòng này. Nhiều khả năng đó là **nhiễu một-run**, cùng cỡ với độ lệch 0.01–0.08 khi GARD chạy lại SOKE.
   - Đây là bằng chứng gián tiếp rằng các hiệu ứng cỡ này trong Tables 3–4 (paper không có seed) cần thận trọng.
   - Caveat: chưa hội tụ đủ 89k bước, và chưa huấn luyện generator để xác nhận trực tiếp.
2. **σ = 0.25 cho một lịch nhiễu rất khác.**
   - Path gần như *đứng yên* ở vùng nhiễu trong 30% thời gian đầu (κ(0.1) = 0.007), rồi tăng tốc. Dạng này không khớp tốt với SD3 shift.
   - Paper báo σ = 0.25 tệ hơn cả GE. Điều đó tương thích với H1: lịch quá dồn về vùng nhiễu thuần làm phần lớn huấn luyện rơi vào trạng thái gần như không chứa tín hiệu.
3. **Kết luận E1-0.** H1 ("hiệu ứng của NG đi qua lịch nhiễu") **không bị bác bỏ**.
   - σ đổi metric của paper đúng ở chỗ σ đổi lịch (σ = 0.25).
   - Ở chỗ σ không đổi lịch (0.5 so với 0.75), chênh lệch paper báo nhỏ và nhiều khả năng là nhiễu.
   - Kết hợp với E0.3 (mô hình NG nhạy với tính không đồng nhất của path khi suy diễn), phiên bản đúng nhất là **H1'**: NG ≈ một lịch nhiễu *không đồng nhất theo joint và mẫu*, được quyết định bởi p0 và cấu trúc metric. H1' cần kiểm định ở khâu huấn luyện bằng E1-gate và E1-H1.


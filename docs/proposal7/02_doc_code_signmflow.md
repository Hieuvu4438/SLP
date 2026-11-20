# 02 — Đọc kỹ `methods/signmflow`: từng hàm, điểm yếu, điểm cần sửa

Phạm vi: chỉ code trong `methods/signmflow/` (HEAD `c859c93`, cộng diff chưa commit ở `jobs.py`) và metric dùng chung `src/slp/evaluation/metrics.py`. Không tham khảo method nào khác trong repo.

Mỗi hàm được đọc theo ba câu hỏi:

1. Hàm làm gì và có khớp với paper không?
2. Điểm yếu là gì, kèm bằng chứng?
3. Nếu thử nghiệm thì sửa ở đâu? Thí nghiệm `E*` tương ứng nằm trong `04_ke_hoach_thuc_nghiem.md`.

Đánh giá chung: implementation trung thành với paper. README đã ghi rõ các lựa chọn mà paper không nói. Các điểm yếu dưới đây là điểm yếu **của phương pháp** hoặc **của cấu hình paper**, không phải lỗi tái lập. Ngoại lệ được đánh dấu riêng.

---

## 1. `representation.py` — `MotionCodec`

| Hàm | Làm gì | Điểm yếu / cơ hội | Sửa ở đâu |
|---|---|---|---|
| `encode` (l.64) | Chuẩn hoá ngược về axis-angle, cộng `hands_mean`, đổi sang quaternion với quy ước w ≥ 0 (Eq. 37), hoặc sang 6D. | Đúng paper. Đã đo dữ liệu (P4): không có quaternion tay nào có w < 0.3; body có 0.7%; tỉ lệ đổi dấu giữa hai frame liên tiếp là 1e−4 (body) và 0 (tay). Vì vậy **vấn đề double-cover của nhãn là không đáng kể**, và loss bất biến dấu hay "temporal sign continuity" (gợi ý từ tài liệu) có ưu tiên thấp. | Không cần sửa cho nhãn. |
| `decode` (l.72) | Quaternion sang axis-angle, trừ `hands_mean`. | Không có vấn đề. | Dùng lại cho FK loss: cần một phiên bản vi phân được (xem §6). |

## 2. `geometry.py`

| Hàm | Ghi chú |
|---|---|
| `expmap` / `logmap` / `project` (l.115–142) | Đúng Eq. 47–48 và ổn định số (dùng sinc và atan2). Có thể tái dùng cho mọi thí nghiệm trên S³. |
| `uniform_hemisphere` (l.150) | p0 của paper. **Điểm yếu lớn** (P5): góc quay trung bình tới x1 là 126.5°, trong khi dữ liệu tay chỉ quay 31–34° (trung vị) quanh tư thế gốc. Với path naturalness, x_t còn cách x1 khoảng 80° hoặc hơn cho tới t ≈ 0.6 (P5), trong khi góc quay p95 của dữ liệu tay là 74–84°. Hướng sửa: thêm `wrapped_normal(mu, sigma)` trên không gian tiếp tuyến tại tư thế trung bình, thí nghiệm **E6-prior**. |
| `sign_convention` (l.38) | Đúng App. C.1. Không cần sửa. |

## 3. `naturalness.py`

| Hàm | Làm gì | Điểm yếu | Hướng |
|---|---|---|---|
| `kmeans_joint` / `extract_prototypes` (l.29, 87) | K-means Euclid trên R⁴ cho từng joint; v2 đã sửa lỗi dừng sớm. | Tốn thời gian (41 joint × K = 500). Theo P1–P3, dưới mục tiêu của paper, sản phẩm cuối của stage này về năng lượng tương đương một phép tái tham số hoá thời gian (s ≈ 2.5 ± 0.3), cộng một phần lệch nhỏ (khoảng 0.07 rad). Tương đương về *chất lượng sinh* là giả thuyết H1, chưa kiểm chứng. | Loại bỏ nếu E1-gate và E1-H1 xác nhận H1. |
| `NaturalnessMeasure.quad_form` (l.126) | Eq. 42 trong log-space: Σ π_k⟨δ_k, v⟩² + ρ‖v‖²/m. | (i) m(x) gần như không đóng vai trò (P1: số hạng ρ/m < 1e−4 năng lượng). (ii) Gần nguồn nhiễu (t < 0.3), hướng về x1 là hướng đắt nhất (P2); tính chất này gần như chung cho mọi metric dựng từ moment bậc hai chuẩn hoá. Đây là **vật thể toán học thực sự được tối ưu**; mô tả "density-weighted" trong paper không khớp với nó. | E11 (ưu tiên rất thấp): metric kiểu MFM/LAND. Time-shift có lẽ xuất hiện với mọi metric đắt ở xa dữ liệu (`01` A.2). |
| `metric_inverse` (l.136) | G⁻¹ dày đặc, dùng cho test và probe. | Không có vấn đề. | Probe P2 dùng hàm này. |

## 4. `paths.py`

| Hàm | Làm gì | Điểm yếu | Sửa |
|---|---|---|---|
| `CorrectionNet` (l.19) | MLP 3 lớp theo frame trên [t, x0, x1], projection Eq. 48, lớp cuối khởi tạo bằng 0. | Không có ngữ cảnh chuỗi hay text. Đầu ra ≈ −c(t)·ξ (P3b: cos ≈ −0.95). Năng lượng đạt cận dưới của phép tái tham số hoá thời gian (P-DA). | E1-H1: geoshift (κ dạng đóng) và NG-parallel (giữ κ_θ, bỏ phần bẻ cong). |
| `path_point` (l.47) | Eq. 11: linear / geodesic / naturalness. | — | **Thêm** `kind="geoshift"`: `G.expmap(x0, kappa(t) * xi)` với `kappa(t) = 1 - s(1-t) / (1 + (s-1)(1-t))`, và `kind="ng_parallel"`. Mọi chỗ gọi (`flow_loss`, `sample`) dùng lại nguyên vẹn. |
| `path_and_velocity` (l.62) | Vận tốc bằng forward-mode jvp. | Chỉ dùng cho velocity-prediction và stage 2. | Với `geoshift` thì có dạng đóng: κ'(t)·DExp[ξ]. |

## 5. `flow.py`

### `flow_loss` (l.60) — chỗ quyết định gần như mọi thứ

1. `t = torch.rand(B)` (l.68): t phân bố đều.
   - Với path naturalness, κ(t) có mật độ dồn về vùng nhiễu. Vậy *mật độ huấn luyện theo mức nhiễu* đã bị path quyết định một cách ngầm định.
   - Sửa (E2-F): thêm `train_shift` cho mức nhiễu khi huấn luyện, tách khỏi `sampler.shift` khi suy diễn.
2. `x_hat = G.normalize(pose)`, `lp = _masked_mse(x_hat, x1)` (l.81–82): loss L2 trên quaternion, mọi joint có trọng số bằng nhau. Có ba điểm yếu:
   - **(a) Lệch không gian so với metric.** Metric đo vị trí khớp sau FK, gồm body căn theo pelvis và tay căn theo cổ tay hoặc Procrustes. Một sai số quay nhỏ ở vai làm lệch cả cánh tay và bàn tay. Loss hiện tại không biết điều đó (QuaterNet, Pavllo 2018/2019; Liu và cộng sự TPAMI 2022).
     - Sửa (**E4-c**, rồi **E4-a**): trọng số cánh tay đòn, rồi `fk_loss` trên x̂1. Vị trí tính qua *toàn chuỗi*, rồi trừ cổ tay cho tay và pelvis cho body. Hand joint của metric hồi quy từ vertex đã posed, nên cần LBS thưa có posedirs (xem `04` E4-a).
   - **(b) Không có trọng số theo t.** JiT (Li & He 2025) báo x-pred với v-loss (trọng số 1/(1−t)², có clip) tốt hơn x-pred với x-loss (FID 8.62 so với 10.14).
     - Thuộc trục `train_shift` của E2-F (trọng số thời gian tương đương với mật độ t), không chạy riêng.
   - **(c) Đầu ra quaternion chuẩn hoá theo bán kính là ánh xạ không liên tục** (Zhou 2019; Geist 2024). Với dữ liệu tay quay nhỏ thì rủi ro thấp (P4).
     - Sửa (E8, ưu tiên thấp): head 6D, Gram-Schmidt rồi sang quaternion.
3. Không có `text dropout`, nên không làm được CFG.
   - Sửa (**E5-CFG**): với xác suất p_uncond = 0.1, thay `text` bằng một token null học được; mask length loss cho các mẫu bị drop.

### `sample` (l.92)

- Re-planned sampler (Eq. 53), lưới t đều, 5 bước.
- Với path naturalness, các mức nhiễu thực tế tại 5 bước là {1.0, 0.91, 0.78, 0.63, 0.42}. Bước cuối nhảy từ mức nhiễu 0.42 thẳng về x̂1.
- Sửa:
  - **E5-CFG**: CFG trên manifold, x̂1 = Exp_{x̂u}(w · Log_{x̂u}(x̂c)). Biểu thức này luôn nằm trên S³. Chi phí gấp đôi số lần gọi mạng.
  - **E7**: self-conditioning, đưa x̂1 của bước trước vào mạng làm đầu vào phụ.
  - **E0.3 / E2-F**: lưới suy diễn riêng, tách khỏi lưới huấn luyện (FrameFlow dùng lịch mũ chỉ khi suy diễn). E0.3 hoán đổi path khi suy diễn trên model NG đã huấn luyện, không cần huấn luyện lại.
  - **Độ dài cố định**: thêm `sampler.lengths_file` để mọi arm dùng chung độ dài.

## 6. `model.py` — `SignGenerator`

| Phần | Điểm yếu | Sửa |
|---|---|---|
| `inp = Linear(pose_dim + expr_dim)` (l.123) | Chỉ nhận x_t và e_t. | E7: concat x̂1_prev (self-conditioning), tức `Linear(2*pose_dim + 2*expr_dim)`. Khởi tạo phần mới bằng 0 để tương thích ngược. |
| Text cross-attention | Không có token null cho CFG. | E5-CFG: `null_text = nn.Parameter(1, 1, 1024)`. |
| `len_head` (l.132) trên đặc trưng text đã detach | Độ dài ảnh hưởng trực tiếp DTW tích luỹ (§8). Loss L1 trên log-length (tập *train*) ở epoch 300 là 0.124 (Phoenix) và 0.092 (CSL), tức khoảng ±10% độ dài. Sai số độ dài trên dev được đo trong probe P6. | Mọi arm dùng chung file độ dài (`04` §1.2) để loại confound độ dài. Chẩn đoán độ nhạy theo γ ∈ {0.9, 1.1} chỉ là phân tích, không dùng để tune. |
| Positional embedding học được, tối đa 400 frame | Không có vấn đề lớn. | — |
| Không có tách theo bộ phận cơ thể | Tay chiếm 30/41 joint nhưng chia chung một đầu ra tuyến tính. Tài liệu (ParCo, EMAGE, SOKE, Body2Hands) cho thấy tách và điều kiện chéo theo bộ phận có lợi. | E9 (cược lớn): head riêng cho tay, điều kiện trên cổ tay và cánh tay. |

## 7. `training.py`

| Hàm | Điểm yếu | Sửa |
|---|---|---|
| `train_correction` (l.48) | Khoảng 100 epoch cho một đối tượng mà kết quả ≈ một phép tái tham số hoá thời gian (P3, P-DA). | E1-0 dùng lại hàm này với σ ∈ {0.25, 0.75} (phép thử bác bỏ rẻ). Bỏ stage nếu H1 được xác nhận. |
| `train_generator` (l.127) | (i) **Không có EMA.** (ii) Text encoder dùng chung lr 2e−4; README ghi nhận đặc trưng text bị sụp trong vài epoch đầu (cosine giữa các câu 0.999). (iii) Không chọn checkpoint: val flow loss nhỏ nhất ở epoch 100 rồi tăng tới 300 (Phoenix 10.30 → 12.56; CSL 8.57 → 11.26), trong khi loss train vẫn giảm. Đây là quá khớp rõ ràng. | **E3**: EMA (decay 0.999 theo step), `text_lr: 2e-5`, warmup 1000 bước. Chọn checkpoint **bên ngoài method**: đánh giá snapshot `generator_ep*.pt` qua `scripts/run.py --method ... method.init.generator=...` trên tập con dev cố định, rồi báo cáo trên phần dev còn lại. Snapshot đã lưu mỗi 50 epoch, nên E0.1 chạy được ngay. |
| `validation_loss` (l.227) | Flow loss val không tương quan chắc chắn với DTW. | **Không** tính DTW trong method (CLAUDE.md). Chỉ tăng tần suất lưu snapshot (`save_every: 25`), rồi chọn bên ngoài. |

## 8. `method.py` — `predict` (l.165)

- Độ dài lấy từ length head, clamp 40..400. Gọi `sample` theo lô đã sắp theo độ dài.
- Seed sampler cố định (`ctx.seed`). Vì vậy kết quả hiện tại là **một mẫu duy nhất** cho mỗi câu, và phương sai do sampling chưa từng được đo.
- Sửa (E0.2): chạy suy diễn 3 seed sampler để ước lượng nhiễu đo.

## 9. Metric (`src/slp/evaluation/metrics.py`) — không sửa, chỉ ghi nhận

- `dtw_cost` trả về **chi phí tích luỹ** (tổng trên đường warping). Chuỗi ngắn hơn GT không bị phạt thêm; chuỗi dài hơn bị phạt. Có `dtw_jpe_norm` (chia theo độ dài path) để kiểm tra chéo.
- Body JPE căn theo pelvis. Hand JPE căn theo cổ tay. Hand PA dùng Procrustes trên 21 joint mỗi tay.
- Theo CLAUDE.md, metric chỉ được định nghĩa trong file này. Các metric chẩn đoán (tốc độ khớp, mean-pose gap) được tính trong probe, không đưa vào method.

---

## Bản đồ hàm → thí nghiệm

| Thí nghiệm (`04` v2) | File và hàm cần chạm | Khối lượng code |
|---|---|---|
| E0.0 bộ đo M1–M4, độ dài cố định, bootstrap | `src/slp/evaluation/metrics.py` (+ test), `scripts/` (evaluator truy hồi), `method.predict` (`sampler.lengths_file`) | ~300 dòng + test |
| E0.1–E0.3 (chỉ suy diễn: snapshot, sampler, hoán đổi path) | config `sampler`, `init.generator`; `flow.sample` nhận path suy diễn riêng | ~30 dòng |
| E1-0 phép thử σ | `training.train_correction` (dùng lại), `probes/probe_da_progress.py` để fit s(σ) | ~0 |
| E1-gate / E1-H1 (geodesic, geoshift, NG-parallel) | `paths.path_point`, `flow.FlowSpec.path`, `method.fit` (bỏ stage 1–2 khi không phải naturalness) | ~40 dòng |
| E2-F shift khi huấn luyện × shift khi suy diễn | `flow.flow_loss` (`train_shift`), `flow.sample` (`sampler.shift`) | ~20 dòng |
| E3 vệ sinh (EMA, text lr, warmup) | `training.train_generator`, `config.yaml` | ~40 dòng |
| E4-c trọng số cánh tay đòn | `flow._masked_mse` (trọng số theo joint), tiền tính từ template SMPL-X | ~40 dòng |
| E4-a/b FK loss (+ vận tốc) | `flow.flow_loss`, module mới `fk.py` (toàn chuỗi; LBS thưa có posedirs cho joint tay hồi quy từ vertex) | ~300 dòng + test |
| E5-CFG | `flow.flow_loss` (text dropout, mask length loss), `model.SignGenerator` (null text), `flow.sample` | ~40 dòng |
| E6-prior wrapped normal | `geometry`, `flow.sample_source`, tư thế trung bình từ cache train | ~40 dòng |
| E7 self-conditioning | `model.forward`, `flow.flow_loss`, `flow.sample` | ~40 dòng |
| E8 head 6D | `model.out`, `flow` | ~20 dòng |
| E9 head tay riêng | `model.py` | ~120 dòng |
| E10 prior âm vị học | `flow.flow_loss` + ranh giới ký hiệu | ~80 dòng + phân đoạn |
| E11 metric MFM (rất thấp) | `naturalness.quad_form` | ~30 dòng |

Mọi thay đổi đặt sau cờ cấu hình, với giá trị mặc định bằng hành vi hiện tại. Nhờ vậy `signmflow` vẫn tái lập đúng paper và các test hiện có vẫn đạt. Theo CLAUDE.md, cũng có thể tạo method mới bằng `scripts/new_method.py` thay vì sửa trực tiếp `signmflow`; xem khuyến nghị trong `04`.

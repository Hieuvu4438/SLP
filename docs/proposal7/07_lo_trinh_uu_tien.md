# 07 — Lộ trình thực nghiệm theo thứ tự ưu tiên (để người dùng chạy)

Tài liệu này là **bản bàn giao**. Nó trả lời bốn câu hỏi cho từng thí nghiệm:

1. Chạy cái gì?
2. Vì sao nên chạy, và vì sao có thể *không* nên?
3. Cần chuẩn bị gì?
4. Đọc kết quả thế nào để quyết định bước sau?

Chi tiết kỹ thuật nằm ở `04`, bằng chứng ở `01`, `03`, `06`.

**Trạng thái.** Vòng 1 (chỉ suy diễn và correction, CPU) đã có kết quả ở `06`. Ngoài vòng 1, chưa thí nghiệm nào được chạy. Mọi thí nghiệm dưới đây chờ người dùng chạy.

**Thời gian mỗi run 300 epoch** (đo từ log v2, GPU dùng chung, 1 tiến trình):

| Dataset | Thời gian/run |
|---|---|
| Phoenix | khoảng 18 GPU-h |
| CSL | khoảng 27 GPU-h |
| How2Sign | lâu hơn CSL; lấy theo job đang chạy |

Đánh giá trên dev chạy được bằng CPU (khoảng 1 s/chuỗi khi máy không quá tải).

---

## 0. Kiểm tra độ đầy đủ của kế hoạch

### 0.1 Mỗi điểm yếu đã có thí nghiệm kiểm định chưa?

| Điểm yếu / cơ hội (nguồn) | Thí nghiệm | Ưu tiên |
|---|---|---|
| Metric DTW thưởng trung bình hoá (`01` C.2, `06` §2) | E0.0 (M1–M4 + đường co về trung bình), áp dụng cho *mọi* arm | P0 |
| Thiếu baseline tầm thường, retrieval và phương sai (`01` B.1–B.2) | E0.4, B×3 | P0, P1 |
| NG có thật sự tốt hơn GE trong bản tái lập không (`01` A.3, `05` #11) | E1-gate | P1 |
| NG ≈ lịch nhiễu (H1/H1'): phần bẻ cong, tính không đồng nhất, lịch toàn cục (`01` A.3, `06` §2.4, §4) | E1-H1 (NG-parallel, geoshift) | P3 (có điều kiện) |
| Loss quaternion lệch với metric vị trí (`02` §5, `03` §2.1) | E4-c → E4-a → E4-b | P2 |
| Không EMA, text encoder sụp, quá khớp (`02` §7, `01` B.3) | E3 | P2 |
| Lịch nhiễu khi huấn luyện chưa tối ưu, tách khỏi khi suy diễn (`03` §1.3) | E2-F | P3 |
| Không có CFG; ngữ nghĩa yếu (`03` §1.5) | E5-CFG | P4 (cần M1) |
| p0 cách dữ liệu 126° (`01` A.4) | E6-prior | P4 |
| Không self-conditioning (`03` §1.5) | E7-SC | P4 |
| Confound bước MR → GE: x_t tuyến tính không renormalize (`05` #22) | **E1-lin** (bổ sung): linear + renorm so với GE | P5 (chỉ khoa học) |
| Đầu ra quaternion không liên tục (`03` §1.6) | E8 | P5 |
| Tay là nút thắt chính (`03` §2.3) | E9 | P5 (có điều kiện) |
| "Naturalness" không có cấu trúc ngôn ngữ (`03` §3.2) | E10 | P5 |
| Metric dữ liệu "đúng chiều" (MFM) (`01` A.2) | E11 | P6 (gần như bỏ) |
| Kết quả chỉ trên Phoenix (`05` #15) | Xác nhận trên CSL × 3, How2Sign × 3 | Cuối |
| So với số paper | Test một lần, khi được cho phép | Cuối |

**Thiếu sót đã bổ sung trong lần rà soát này:**

1. E1-lin (confound MR → GE), trước đó chỉ được nhắc trong phản biện.
2. Thủ tục dựng đường co về trung bình (§1.2).
3. Danh sách việc cần code (§1.1).
4. Điều kiện dừng theo từng mức ưu tiên (§2).
5. Sửa chi phí (18 GPU-h/run ở 300 epoch).

### 0.2 Những gì *cố ý không* có trong kế hoạch, và lý do

| Hướng | Lý do |
|---|---|
| Đổi text encoder (LLM) hoặc dùng retrieval từ từ điển | Thay đổi bài toán, nên không còn so sánh công bằng với SignNMFlow. |
| Chọn checkpoint sớm | Đã bị bác bỏ ở vòng 1. |
| Tinh chỉnh sampler, lưới suy diễn hay NFE như một "cải tiến" | Vòng 1 cho thấy đây là trung bình hoá. Lưới suy diễn chỉ còn là một trục trong E2-F, và phải qua đường co. |
| Loss bất biến dấu và liên tục dấu theo thời gian của quaternion | P4: dữ liệu gần như không có vấn đề dấu. |
| Phương pháp ít bước (MeanFlow, Shortcut) | Paper đã dùng 5 bước; không phải nút thắt chất lượng. |

---

## 1. Việc chuẩn bị (trước khi chạy GPU)

### 1.1 Việc cần code

Đặt trong method mới `methods/signmflow_plus/`, tạo bằng `scripts/new_method.py`. Mọi cờ có mặc định bằng hành vi paper.

| # | Việc | Nơi | Test chấp nhận | Phục vụ |
|---|---|---|---|---|
| C1 | `sampler.lengths_file`: đọc độ dài cố định thay cho length head | `method.predict` | Độ dài đầu ra đúng bằng file | Mọi arm |
| C2 | `flow.path ∈ {naturalness, geodesic, geoshift, ng_parallel, linear}` + `flow.shift` | `paths.path_point`; `method.fit` bỏ stage 1–2 khi không cần | `geoshift` với s = 1 trùng `geodesic`; `ng_parallel` trùng phép chiếu của NG lên ξ | E1, E2-F |
| C3 | `flow.train_shift` (mức nhiễu khi huấn luyện) và `sampler.shift` / `sampler.grid_power` (khi suy diễn) | `flow.flow_loss`, `flow.sample` | Mặc định trùng `flow.sample` hiện tại (đã kiểm chứng ở `probes/probe_inference.py`) | E2-F |
| C4 | `generator.ema` (decay), `generator.text_lr`, `warmup_steps` (đã có) | `training.train_generator`; lưu cả trọng số EMA | Resume giữ trạng thái EMA | E3 |
| C5 | `flow.joint_weights: lever_arm` | `flow._masked_mse` | Tổng trọng số chuẩn hoá bằng 41 | E4-c |
| C6 | `flow.fk_loss {lambda_body, lambda_hand, lambda_vel}`: FK vi phân được qua toàn chuỗi; tay = joint hồi quy từ vertex (LBS thưa có posedirs) hoặc proxy | Module mới `fk.py` (chuyển vào `src/slp/` nếu dùng chung, kèm test) | So với `slp.evaluation.smplx_fk.SMPLXJoints` trên 100 frame GT (sai lệch < 1 mm nếu không dùng proxy) | E4-a, E4-b |
| C7 | `flow.p_uncond` + token null + `sampler.cfg_w` (CFG trên manifold); mask length loss khi drop | `flow`, `model` | p_uncond = 0 trùng hành vi cũ | E5 |
| C8 | `flow.source: hemisphere | wrapped_normal(sigma0)` | `geometry`, `flow.sample_source` | Mẫu nằm trên S³; σ0 → 0 thì mẫu tiến về tư thế trung bình | E6 |
| C9 | `generator.self_cond` | `model.forward`, `flow` | Tắt thì trùng hành vi cũ | E7 |
| C10 | Bộ đo M1–M4, đường co về trung bình, bootstrap cặp, tốc độ lọc thông thấp | `src/slp/evaluation/metrics.py` (+ test), `scripts/analyze_signmflow_plus.py` | M1: R@1 của GT ≥ 3 lần của GT câu khác; mean pose thấp nhất | Mọi arm |

Sau mỗi thay đổi, chạy `python -m pytest -q`.

**Trạng thái (2026-11-20):** C1–C10 đã được code và có unit test, chưa chạy thí nghiệm nào.
- `methods/signmflow_plus/`: mặc định trùng bit với `methods/signmflow`.
- Metric nằm trong `src/slp/evaluation/metrics.py`.
- Baseline E0.4 nằm trong `methods/text_retrieval/`.
- Lệnh chạy từng arm: `python -m methods.signmflow_plus.experiments plan --prio P0,P1`.
- E9, E10, E11 chưa code (có điều kiện, xem README của method).

### 1.2 Thủ tục chấm một arm (áp dụng đồng nhất)

1. Sinh dự đoán trên **toàn bộ dev**, dùng độ dài cố định (C1) và 3 seed sampler.
2. Tính 4 cột DTW (JPE body/hand, PA body-ua/hand), M2 (tốc độ lọc thông thấp σ = 2 frame, body và ngón), M1, M4.
3. **Đường co về trung bình của baseline:** nhân dự đoán baseline với (1−a), a ∈ {0, 0.05, 0.1, 0.15, 0.2}. Được một đường "DTW theo tốc độ". Nội suy DTW của đường này tại đúng tốc độ của arm.
4. **Hiệu ứng thật** = DTW(arm) − DTW_đường(tại tốc độ của arm), kèm khoảng tin cậy bootstrap cặp theo chuỗi.
5. Arm thắng khi thoả cả 6 điều kiện của `04` §1.3.

---

## 2. Thứ tự ưu tiên

Ký hiệu: **[GPU-h]** là chi phí huấn luyện. **Điều kiện dừng** là kết quả khiến cả nhánh bị bỏ.

### P0 — Nền đo lường (bắt buộc trước mọi so sánh; gần như không cần GPU)

| ID | Việc | Vì sao nên chạy | Vì sao có thể không nên / rủi ro | Chi phí |
|---|---|---|---|---|
| **E0.0** | Dựng M1–M4, đường co, bootstrap (C10, C1); chạy trên v2 Phoenix và CSL | Vòng 1 đã *chứng minh* DTW đơn thuần bị "chơi" được. Thiếu nền này thì không kết quả nào sau đó diễn giải được. | Evaluator M1 có thể yếu, vì dữ liệu nhỏ và câu Phoenix lặp lại. Dự phòng: SiLVERScore/SignCLIP, hoặc back-translation likelihood (`03` §4). | 1–2 ngày code + khoảng 4 GPU-h |
| **E0.4** | Baseline tầm thường: mean pose, GT câu khác, **retrieval câu train gần nhất** (TF-IDF trên văn bản) | Đặt sàn và trần. Retrieval có thể là sàn mạnh trên Phoenix; nếu SignNMFlow không vượt retrieval thì đó là phát hiện quan trọng. | Không có. | CPU |
| E0.1–E0.3, E1-0 | **Đã chạy** (Phoenix) | — | Nên lặp lại E0.2 và đường co trên CSL (CPU, rẻ) để chắc kết luận không riêng cho Phoenix. | CPU |

**Điều kiện dừng P0:** nếu M1 không tách được GT với GT câu khác, *không* chạy P4 (CFG) cho tới khi có thước đo ngữ nghĩa khác.

### P1 — Ngưỡng nhiễu và đối chứng dương (trả lời câu hỏi khoa học chính)

| ID | Việc | Vì sao nên chạy | Vì sao có thể không nên / rủi ro | Chi phí |
|---|---|---|---|---|
| **B×3** | v2 (NG, config paper) × seed {0, 1, 2}, Phoenix, 300 epoch. Seed 0 đã có. | Mọi quyết định sau cần ngưỡng nhiễu giữa các *lần huấn luyện*. Hiện mới đo được nhiễu do sampler (rất nhỏ, ±0.007 PA hand). Paper không có seed nào. | Tốn 36 GPU-h mà không "cải thiện" gì. Nhưng không có nó thì mọi chênh 0.02–0.05 đều vô nghĩa. | 2 × 18 |
| **E1-gate** | GE (`path=geodesic`, bỏ prototype và ε) × 3 seed, Phoenix | **Đối chứng dương** cho đóng góp chính của paper (Table 3: NG tốt hơn GE). Nếu NG ≈ GE trong bản tái lập: (a) đó là kết quả khoa học chính; (b) mọi run sau bỏ được stage K-means và correction (tiết kiệm khoảng 1 ngày mỗi pipeline, đơn giản hoá). | Paper ablate trên CSL. Nếu Phoenix không kết luận được thì phải lặp trên CSL (3 × 27 GPU-h). | 3 × 18 (+ 3 × 27 nếu cần CSL) |

**Quyết định sau P1:**
- NG ≈ GE (khác biệt không có ý nghĩa): dùng GE làm nền đơn giản cho P2 trở đi, và bỏ E1-H1.
- NG > GE: giữ NG làm nền, và mở E1-H1 ở P3.

### P2 — Đòn bẩy có khả năng cải thiện *thật* cao nhất

| ID | Việc | Vì sao nên chạy | Vì sao có thể không nên / rủi ro | Chi phí |
|---|---|---|---|---|
| **E3** | Vệ sinh: EMA 0.999, `text_lr = 2e-5`, warmup 1000 | (1) README v1 ghi nhận đặc trưng text sụp (cosine giữa câu 0.999) khi fine-tune mBART ở lr 2e−4. (2) EMA là chuẩn của mọi mô hình khuếch tán và flow hiện đại (EDM2, JiT). (3) Rẻ, ít rủi ro. Nếu thắng thì run này *chính là* baseline mới (cùng tinh chỉnh cho mọi arm). | Vòng 1 cho thấy mô hình còn tốt lên tới epoch 300, nên lợi ích chống quá khớp có thể nhỏ. Kỳ vọng chủ yếu ở M1, không phải DTW. | 18 (+ 2 seed nếu giữ) |
| **E4-c** | Trọng số cánh tay đòn cho loss quaternion | Lệch pha lớn nhất giữa loss (quaternion, đồng đều) và metric (vị trí sau FK). Lỗi ở vai và khuỷu khuếch đại xuống cổ tay và ngón (QuaterNet, SGRE). Gần như không tốn code. | Có thể chỉ dịch cân bằng body ↔ tay mà không cải thiện tổng. | 18 |
| **E4-a** | FK loss qua toàn chuỗi (body căn pelvis, tay căn cổ tay; joint tay từ vertex) | Tối ưu trực tiếp đại lượng được đo. Có bằng chứng ở nhiều lĩnh vực lân cận (QuaterNet, Liu TPAMI 2022, MDM, EMAGE). Paper SignNMFlow và SOKE không dùng. | (1) MSE vị trí ở mức nhiễu cao có thể *tăng* trung bình hoá. Phải qua đường co (§1.2). (2) Tốn 3–4 ngày code (LBS có posedirs). Chỉ làm nếu E4-c cho thấy tín hiệu, hoặc đo proxy cho thấy cần. | 18 + 3–4 ngày |
| E4-b | E4-a + loss vận tốc, λ_v ∈ {0.5, 2}·λ_h | v2 chỉ đạt 65–76% tốc độ GT. Loss vận tốc nhắm thẳng vào độ sống động (EMAGE, MOJO, SignSparK). | Đánh đổi mượt/giật; λ phải quét. | 2 × 18 |

**Điều kiện dừng P2:** nếu E4-c và E4-a đều không nằm dưới đường co, kết luận "loss vị trí không đẩy được biên DTW–tốc độ". Chuyển trọng tâm sang P4 (ngữ nghĩa và độ sống động).

### P3 — Lịch nhiễu ở khâu huấn luyện và cơ chế (có điều kiện)

| ID | Việc | Vì sao nên chạy | Vì sao có thể không nên | Chi phí |
|---|---|---|---|---|
| **E2-F** | train_shift ∈ {1, 2.5, 4}, mỗi cái với trục suy diễn {1, 2.5, 4} (trục này miễn phí) | Tài liệu (SD3, Chen 2023, Kingma & Gao) cho thấy phân bổ mức nhiễu khi huấn luyện là núm mạnh. Vòng 1 mới thử *khi suy diễn*. | Vòng 1: dồn về vùng nhiễu chủ yếu sinh trung bình hoá. Khả năng có hiệu ứng thật trên đường co là trung bình–thấp. | 2 × 18 (một ô trùng nền) |
| **E1-H1** (chỉ khi NG > GE ở P1) | NG-parallel; geoshift s ∈ {2, 2.5, 3} | Tách lợi ích của NG thành phần bẻ cong, tính không đồng nhất của κ, và lịch toàn cục. Đây là đóng góp khoa học (diễn giải lại paper). | Không cải thiện metric so với NG. Thuần khoa học. | 4 × 18 |

### P4 — Ngữ nghĩa, nguồn nhiễu, chất lượng sinh (cần M1)

| ID | Việc | Vì sao nên chạy | Vì sao có thể không nên | Chi phí |
|---|---|---|---|---|
| **E5-CFG** | p_uncond 0.1; w ∈ {1, 1.5, 2, 3} khi suy diễn (một lần huấn luyện) | Chuẩn cho sinh có điều kiện (Ho & Salimans; SignSparK dùng trong SLP). Nhắm vào ngữ nghĩa, nơi B-T của paper không kiểm chứng được. | DTW dự kiến xấu đi (đẩy xa trung bình). Gấp đôi số lần gọi mạng. Chỉ đánh giá được khi có M1. | 18 |
| **E7-SC** | Self-conditioning | Gần như miễn phí với re-planned sampler. Dùng trong FrameDiff, FrameFlow, Proteina. | Bằng chứng trên chuyển động người còn ít. Chậm hơn khoảng 30% khi huấn luyện. | 24 |
| **E6-prior** | Wrapped normal tại tư thế trung bình, σ0 ∈ {0.25, 0.5} | Rút đường vận chuyển từ 126° xuống 51° (đã đo). PriorGrad và FrameFlow ủng hộ. | Có thể giảm đa dạng hoặc tăng trung bình hoá. Phải tinh chỉnh lại s. | 2 × 18 |

### P5–P6 — Cược lớn, thuần khoa học, hoặc ưu tiên thấp

| ID | Việc | Khi nào chạy |
|---|---|---|
| E1-lin | Linear + renorm so với GE (tách confound của Table 3) | Khi viết bài và cần ablation sạch |
| E9 | Head tay riêng, điều kiện trên cánh tay | Nếu sau P2 lỗi tay vẫn chiếm phần lớn |
| E10 | Prior âm vị học (bất biến hình tay, đối xứng) | Khi có ranh giới ký hiệu đáng tin |
| E8 | Head 6D | Chỉ để hoàn thiện ablation biểu diễn |
| E11 | Metric MFM | Gần như bỏ |

### Cuối — Ghép và xác nhận

1. Ghép các thành phần thắng theo thứ tự P2 → P3 → P4 (forward selection), mỗi bước phải nằm dưới đường co.
2. Leave-one-out trên Phoenix.
3. CSL × 3 seed (khoảng 81 GPU-h cho mỗi cấu hình), How2Sign × 3 seed.
4. **Test một lần**, chỉ khi người dùng cho phép. Báo cáo tách: (a) so với bản tái lập; (b) so với số paper, chỉ các cột DTW, kèm caveat.

---

## 3. Lịch gợi ý (1 GPU, 1 tiến trình)

| Tuần | Việc | GPU-h |
|---|---|---|
| 0 (song song với How2Sign) | Code C1–C5, C10; E0.0, E0.4; lặp vòng 1 trên CSL bằng CPU | ~4 |
| 1 | B×3 (2 run mới), E1-gate (3 run) | ~90 |
| 2 | E3, E4-c; code C6 (FK) | ~36 |
| 3 | E4-a, E4-b (×2) | ~54 |
| 4 | E2-F (2 run), E5-CFG | ~54 |
| 5 | E7, E6 (×2) hoặc E1-H1 tuỳ kết quả P1 | ~60–72 |
| 6–8 | Ghép, leave-one-out, CSL × 3, How2Sign × 3 | ~250+ |

**Tổng khoảng 550–650 GPU-h**, khoảng 7–8 tuần với 1 tiến trình.

**Nếu chỉ chạy được khoảng 150 GPU-h:** P0 + P1 (B×3 + E1-gate) + E3 + E4-c.

- Với ngân sách đó trả lời được: cơ chế NG có thật không, và có cải thiện thật nào ở khâu huấn luyện không.

## 4. Mẫu báo cáo cho mỗi arm (để ghi vào `docs/research_log.md`)

```
ID / config overrides / seed(s) / epochs / commit
DTW (4 cột) | Δ so với nền [CI 95%] | tốc độ lọc thông thấp body/ngón (tỉ lệ so với GT)
Hiệu ứng trên đường co (4 cột) [CI] | M1 R@1/R@3 | M4 | dtw_jpe_norm
Quyết định: giữ / bỏ / cần thêm seed — lý do theo 6 điều kiện của 04 §1.3
```

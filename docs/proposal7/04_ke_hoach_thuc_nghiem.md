# 04 — Kế hoạch thực nghiệm: có vượt được SignNMFlow không?

> Phiên bản 2. Phiên bản 1 đã được phản biện nội bộ; các thay đổi được liệt kê trong `05` §2.
> Ba thay đổi lớn nhất:
> (i) Giai đoạn 0 dựng **hệ đánh giá có khả năng phân định** trước khi huấn luyện bất cứ gì;
> (ii) E1 thành **phép thử có thể bác bỏ, có đối chứng dương** (GE so với NG);
> (iii) các thí nghiệm cùng xoay một "núm" phân bổ mức nhiễu được gộp thành một factorial.

> **Cập nhật sau vòng 1** (kết quả chi tiết ở [06](06_ket_qua_thuc_nghiem.md)):
> - E0.1: chọn checkpoint sớm làm DTW *xấu đi*, nên early selection bị loại và độ dài sàng lọc phải ≥ 250 epoch (khoảng 15–18 GPU-h/run).
> - E0.2/E0.3 (lưới, NFE, đổi path khi suy diễn): mọi cải thiện DTW đều đi kèm giảm tốc độ, và bị phép co không dùng văn bản ở cùng tốc độ làm bằng hoặc tốt hơn.
> - **Quy tắc mới (bắt buộc):** mỗi arm báo cáo điểm (DTW, tốc độ lọc thông thấp) và được so với *đường co về trung bình* của baseline. Một arm chỉ "thắng" nếu nằm *dưới* đường đó, tức DTW tốt hơn ở cùng tốc độ, và có ý nghĩa theo bootstrap cặp.

## 0. Câu hỏi và nguyên tắc

**Câu hỏi chính (kỹ thuật).** Giữ nguyên dữ liệu SMPL-X, mBART, kiến trúc ϕ và NFE = 5. Những thay đổi nào làm giảm DTW-JPE và DTW-PA-JPE (body/hand) so với **bản tái lập SignNMFlow v2** trên cùng giao thức, *mà không* tăng hồi quy về trung bình và không giảm khả năng truy hồi văn bản ↔ chuyển động?

**Câu hỏi phụ (khoa học).** Lợi ích của naturalness-guided path (nếu tồn tại trong bản tái lập) đến từ đâu? Có bốn khả năng:

1. lịch nhiễu toàn cục;
2. tính không đồng nhất của mức nhiễu giữa các mẫu và joint;
3. phần bẻ cong của path;
4. hồi quy về trung bình.

**Nói thẳng về mục tiêu.** Trên Phoenix test, bản tái lập đang kém số paper 0.5–1.0 điểm DTW (4.93 so với 4.37 body; 6.54 so với 5.57 hand).

- Không thay đổi nào dưới đây được kỳ vọng chắc chắn sẽ lấp hết khoảng cách đó.
- "Vượt SignNMFlow" trước hết có nghĩa là **(a) vượt bản tái lập**, với kiểm định thống kê.
- **(b) So với số paper** chỉ làm một lần trên test ở cuối, và chỉ trên các cột DTW, kèm caveat về giao thức (PA-body, "All", back-translator).

**Nguyên tắc** (CLAUDE.md và `docs/experiment_protocol.md`):

1. Tạo method mới: `python scripts/new_method.py signmflow_plus`. Chép mã từ `methods/signmflow`, thêm cờ cấu hình, mặc định bằng hành vi của paper. Không import từ method khác. `methods/signmflow` giữ nguyên làm baseline.
2. Mọi lần chạy qua `scripts/run.py`, split `val`. **Không chạy `--split test`** khi chưa được người dùng cho phép.
3. **Không tính metric báo cáo bên trong method.** Việc chọn checkpoint làm *bên ngoài* method: đánh giá các snapshot `generator_ep*.pt` qua `scripts/run.py` với `method.init.generator=...`, rồi chọn bằng một script phân tích.
4. Sau mỗi thay đổi ở `src/slp/` hoặc method, chạy `python -m pytest -q`. Mọi metric mới nằm trong `src/slp/evaluation/metrics.py` và có unit test.
5. Ít nhất 3 seed cho mọi con số báo cáo cuối (protocol §4). Ngân sách tinh chỉnh như nhau cho baseline và biến thể (protocol §5): EMA, chọn checkpoint và lưới NFE/sampler được áp dụng cho *cả* baseline v2.
6. Tài nguyên: 1 GPU dùng chung, tối đa 1 tiến trình. Job How2Sign v2 đang chạy phải xong trước. Máy CPU dùng chung (load khoảng 50), nên đánh giá DTW đặt `OMP_NUM_THREADS` thấp và có ngân sách CPU riêng (§3).

## 1. Đo lường: phải có khả năng phân định trước khi so sánh

### 1.1 Vì sao hệ đánh giá hiện tại chưa đủ

- **DTW tích luỹ thưởng cho hồi quy về trung bình** (SLRTP 2025; bản tái lập v1).
- **B-T đã bão hoà trên Phoenix** (v2 11.34, trần GT 11.49) **và hỏng trên CSL** (mô hình 6.00 > GT 4.74).
- **Tỉ lệ tốc độ một phía có thể bị "chơi"**: thêm giật làm tăng tốc độ.

### 1.2 Bộ đo mới (Giai đoạn 0, trước mọi lần huấn luyện)

| Thành phần | Mô tả | Nơi đặt |
|---|---|---|
| **M1 — evaluator truy hồi text ↔ motion** | Encoder tương phản nhỏ (chỉ train trên tập train) cho motion và text, theo kiểu đánh giá T2M. Báo cáo R-precision@1/3 (lô 32) và một khoảng cách phân phối kiểu FID trên đặc trưng của evaluator. **Hiệu chỉnh bắt buộc:** GT ≫ GT câu khác ≫ mean pose. | Mô hình đánh giá trong `scripts/` + `outputs/`; metric trong `src/slp/evaluation/metrics.py` (kèm test) |
| **M2 — dải thống kê động học hai phía** | Tỉ lệ (dự đoán/GT) của tốc độ trung bình *và* gia tốc/jerk, cho body và ngón tay (wrist-relative), là quãng đường kiểu SLRTP "total distance". Yêu cầu nằm trong [0.8, 1.2] cho cả tốc độ lẫn gia tốc. **Tính trên tín hiệu đã lọc thông thấp** (hoặc theo dải tần): P6 cho thấy tốc độ ngón tay của GT phần lớn là giật tần số cao do fitting (26 → 5.6 mm/frame sau làm mượt σ = 3). | `metrics.py` + test |
| **M3 — khoảng cách tới mean pose** | Δ(mean pose − mô hình) trên mọi cột DTW. Là chỉ số headline đi kèm. | Tính từ baseline có sẵn |
| **M4 — kiểm tra hoán đổi text** | DTW(dự đoán, GT đúng câu) so với DTW(dự đoán, GT câu khác cùng độ dài). | Script phân tích |
| **Độ dài cố định** | Độ dài của *mọi* arm lấy từ cùng một file (độ dài dự đoán của v2). Loại confound "text encoder thay đổi → độ dài thay đổi → DTW thay đổi". Báo cáo thêm `dtw_jpe_norm`. | Cờ `sampler.lengths_file` |
| **Kiểm định cặp** | Bootstrap theo từng chuỗi (cùng chuỗi dev, cùng seed sampler) cho mọi so sánh, cộng 3 seed huấn luyện cho các so sánh chính. | Script phân tích |

### 1.3 Quy tắc quyết định (đăng ký trước)

- **So sánh chính** (đăng ký trước, mỗi so sánh có một giả thuyết): E1-gate, E3, E4-c/E4-a, và E1-H1 (nếu được mở). Các so sánh còn lại là thăm dò.
- **Một arm thắng** khi đồng thời thoả:
  - (i) DTW-PA-JPE hand *hoặc* DTW-JPE hand cải thiện có ý nghĩa (bootstrap cặp, khoảng tin cậy 95% không chứa 0, nhất quán qua seed);
  - (ii) không xấu đi có ý nghĩa ở cột DTW chính nào khác;
  - (iii) M1 không xấu đi có ý nghĩa;
  - (iv) M2 nằm trong dải;
  - (v) M3 không thu hẹp. Nếu thu hẹp thì có thể là "thắng" nhờ trung bình hoá.
  - (vi) nằm dưới đường biên "co về trung bình" của baseline (`06` §2).
- **Chọn checkpoint:**
  - chọn trên một **tập con dev cố định** (200 chuỗi đầu), báo cáo trên **phần còn lại**;
  - hoặc dùng một epoch cố định giống nhau cho mọi arm, chọn theo kết quả E0.1.
- **Độ dài huấn luyện sàng lọc:** E0.1 cho thấy DTW tiếp tục tốt lên tới epoch 250–300, nên sàng lọc dùng **300 epoch** (hoặc ≥ 250), cùng độ dài cho mọi arm.

## 2. Danh mục thí nghiệm

Ký hiệu:

- **GPU-h**: ước lượng trên GPU dùng chung. Phoenix khoảng 220 s/epoch; CSL khoảng 330 s/epoch (log v2). Các ô "9 GPU-h" dưới đây là ước lượng cũ cho 150 epoch; **sau vòng 1 mỗi run Phoenix là khoảng 18 GPU-h** (300 epoch). Lộ trình đã sửa chi phí ở [07](07_lo_trinh_uu_tien.md).
- **CPU-h**: thời gian đánh giá DTW.
- **Kỳ vọng**: dự đoán có thể bị bác bỏ, viết *trước* khi chạy.

### Giai đoạn 0 — Không huấn luyện generator (chỉ CPU và suy diễn)

| ID | Việc | Chi phí | Kỳ vọng / quyết định |
|---|---|---|---|
| **E0.0** | Dựng M1–M4, file độ dài cố định, script bootstrap. Hiệu chỉnh M1 trên GT, GT xáo trộn và mean pose. | ~1 ngày code + ~4 GPU-h (encoder nhỏ) | Nếu M1 không tách được GT với GT xáo trộn (R@1 cách biệt dưới 3 lần), dừng và sửa evaluator trước. |
| **E0.1** | Đánh giá snapshot `generator_ep{50..300}` (Phoenix, CSL) trên tập con dev: DTW, M1, M2. | ~1 GPU-h + ~6 CPU-h | **Đã chạy (Phoenix, `06`):** DTW tốt dần tới epoch 250–300 dù val flow loss nhỏ nhất ở epoch 100. Kết luận: dùng epoch cuối, sàng lọc ≥ 250 epoch. |
| **E0.2** | Sampler trên model NG v2: NFE ∈ {1, 2, 5, 10}; 3 seed sampler; độ dài cố định. | ~1 GPU-h | **Đã chạy (Phoenix, `06`):** nhiễu sampler ±0.007 PA hand; NFE thấp thắng DTW nhưng giảm tốc độ, tức trung bình hoá. |
| **E0.3** | **Hoán đổi path khi suy diễn trên model NG đã huấn luyện** (target-prediction cho phép điều này): sampler re-planned trên (a) path NG gốc, (b) geoshift(s) với s ∈ {2, 2.5, 3}, đưa vào mạng t = κ_s⁻¹(κ), (c) geodesic. | ~1 GPU-h | **Đã chạy (Phoenix, `06`):** (b) khác (a): body tốt hơn, tay xấu hơn. Mô hình NG dựa vào tính không đồng nhất của path khi suy diễn. |
| **E0.4** | Baseline tầm thường trên toàn bộ dev Phoenix/CSL: mean pose, GT câu khác, retrieval câu train gần nhất (TF-IDF). | CPU | Đặt sàn. Retrieval là sàn mạnh trên Phoenix (câu thời tiết lặp lại nhiều). |
| **E1-0** | **Phép thử σ** (bác bỏ rẻ nhất cho H1): huấn luyện *chỉ* correction với σ ∈ {0.25, 0.75}, dùng lại prototype có sẵn, khoảng 20 epoch; fit s(σ). | ~4 GPU-h (hoặc CPU) | **Đã chạy (CPU, 4000 bước, `06` §4):** σ = 0.5 và 0.75 cho path giống hệt nhau; σ = 0.25 cho lịch nhiễu cực đoan. H1 không bị bác bỏ. |

### Giai đoạn 1 — Nhiễu đo, đối chứng dương, kiểm định cơ chế

| ID | Thay đổi | Chi phí | Kỳ vọng |
|---|---|---|---|
| **B×3** | Baseline NG (v2) × 3 seed ở độ dài sàng lọc, với cùng vệ sinh tối ưu với các arm (xem E3). | 3 × ~9 GPU-h | Ngưỡng nhiễu theo seed. Là đối chứng chung cho mọi so sánh. |
| **E1-gate** | GE (geodesic, s = 1, không prototype, không ε) × 2–3 seed so với B×3. Nếu Phoenix không kết luận được thì chạy trên CSL, dataset ablation của paper. | 3 × 9 GPU-h | **Đối chứng dương.** Paper: NG tốt hơn GE (CSL hand PA 1.41 → 1.30). Nếu NG − GE *không* có ý nghĩa trong bản tái lập, kết quả chính là "NG không có hiệu ứng đo được trong bản tái lập trung thành"; E1-H1 bỏ, và s chỉ còn là núm tinh chỉnh. |
| **E1-H1** (chỉ khi gate dương) | Ba arm: **NG-parallel** x_t = Exp_{x0}(κ_θ(t, x0, x1)·ξ) (giữ tính không đồng nhất, bỏ phần bẻ cong; dùng lại ε có sẵn); **geoshift** s ∈ {2, 2.5, 3} (lịch toàn cục). | 4 × 9 GPU-h | So NG với NG-parallel để tách phần *bẻ cong*; so NG-parallel với geoshift để tách phần *không đồng nhất*. Biên tương đương đặt trước: \|Δ\| < ½·(NG − GE). |

Ghi chú implementation E1:

```python
# paths.path_point — geoshift (SD3 shift on the geodesic)
n = s * (1 - t) / (1 + (s - 1) * (1 - t))
return G.expmap(x0, (1 - n) * G.logmap(x0, x1))
# NG-parallel: kappa_theta = <t xi + t(1-t) eps, xi> / |xi|^2  (per joint) -> Exp(kappa_theta * xi)
```

- `flow.sample` dùng chung `path_point`.
- `method.fit` bỏ qua stage prototype và correction khi `path ∈ {geodesic, geoshift}`.

### Giai đoạn 2 — Phân bổ mức nhiễu: một núm, thiết kế factorial

Các thay đổi sau đều chỉ phân bổ lại công sức huấn luyện hoặc trạng thái suy diễn theo mức nhiễu hiệu dụng. Chúng không được chạy như các arm tuần tự độc lập (forward selection sẽ tính trùng):

- geoshift s;
- t ~ logit-normal;
- trọng số w(t) = 1/(1−t)²;
- lưới NFE.

Thay vào đó dùng hai núm trực giao:

| ID | Thiết kế | Chi phí | Kỳ vọng |
|---|---|---|---|
| **E2-F** | Shift khi huấn luyện ∈ {1, 2.5, 4} × shift khi suy diễn ∈ {1, 2.5, 4}. 3 lần huấn luyện; trục suy diễn miễn phí. | 3 × 9 GPU-h (1 ô trùng E1) | **H2:** tồn tại cặp (s_train, s_inf) tốt hơn (2.5, 2.5) ≈ NG. **Cảnh báo:** shift lớn hơn đẩy x̂1 về trung bình có điều kiện. Chỉ chấp nhận nếu M2 và M3 không xấu đi. |

Trọng số kiểu v-loss (E3c cũ) được coi là một biến thể p_train, nằm trong trục s_train, không chạy riêng.

### Giai đoạn 3 — Vệ sinh tối ưu (nền mới, áp dụng cho cả baseline)

| ID | Thay đổi | Chi phí | Kỳ vọng |
|---|---|---|---|
| **E3** | EMA 0.999 + `text_lr = 2e-5` + warmup 1000 bước. Mask length loss khi có text dropout. | 9 GPU-h | Giảm sụp đặc trưng text và quá khớp. Kỳ vọng M1 tăng; DTW giữ nguyên hoặc tăng nhẹ. Là **vệ sinh**, không phải đóng góp phương pháp. Nếu giữ thì áp dụng cho cả baseline khi báo cáo. |

### Giai đoạn 4 — Loss trong không gian vị trí

| ID | Thay đổi | Chi phí | Kỳ vọng |
|---|---|---|---|
| **E4-c** (chạy trước) | Trọng số theo cánh tay đòn trong loss quaternion: trọng số mỗi joint tỉ lệ với ảnh hưởng của nó lên vị trí joint được đo (tổng độ dài xương con cháu, hoặc đạo hàm vị trí theo góc tính trên template SMPL-X). | 9 GPU-h | Xấp xỉ bậc nhất của FK loss. Nếu đã thu phần lớn lợi ích thì không cần E4-a. |
| **E4-a** | **FK loss trên x̂1.** Vị trí tính qua *toàn chuỗi* (pelvis → spine → vai → khuỷu → cổ tay → ngón), *sau đó* trừ vị trí cổ tay cho tay và trừ pelvis cho body, khớp đúng cách metric căn chỉnh. JPE hand giữ hướng toàn cục của cổ tay, nên lỗi ở vai và khuỷu cũng tính vào. Loss = 10·L_quat + λ_b·L_body + λ_h·L_hand. | 9 GPU-h + **3–4 ngày** code/test | **H3:** giảm JPE hand và PA hand. **Rủi ro:** MSE trong không gian vị trí ở mức nhiễu cao kéo x̂1 về trung bình có điều kiện; M2 và M3 là bắt buộc. |
| **E4-b** | E4-a + loss vận tốc/gia tốc trên vị trí, λ_v ∈ {0.5, 2}·λ_h. | 2 × 9 GPU-h | MOJO và EMAGE: λ lớn thì quá mượt, λ nhỏ thì giật. Chọn theo M2. |

**Tính khả thi của E4-a** (phản biện nội bộ đã sửa ước lượng ban đầu):

- Trong `src/slp/evaluation/smplx_fk.py`, 21 joint của mỗi tay được hồi quy từ **đỉnh mesh đã posed**, không phải từ khớp khung xương.
- Muốn khớp metric cần LBS thưa **có pose blendshapes (posedirs)** trên hợp các vertex mà các hàng regressor dùng tới.
- Bước 1: đo tương quan của một proxy rẻ (khớp ngón khung xương, chưa có đầu ngón) với metric trên các nhiễu loạn của GT. Chỉ viết LBS thưa nếu proxy tương quan kém.
- Unit test so với `SMPLXJoints` với dung sai đặt theo metric thật, không đặt 1e−5 m nếu dùng proxy.

### Giai đoạn 5 — Điều kiện văn bản, nguồn nhiễu, cấu trúc (sau khi có M1)

| ID | Thay đổi | Chi phí | Kỳ vọng |
|---|---|---|---|
| **E5-CFG** | Text dropout 10% (token null) trên nền E3. Suy diễn x̂1 = Exp_{x̂u}(w·Log_{x̂u} x̂c), w ∈ {1, 1.5, 2, 3}. | 9 GPU-h | CFG đẩy đầu ra xa trung bình, nên **DTW được dự kiến xấu đi**. Chỉ M1 có thể ghi nhận lợi ích. **Gấp đôi số lần gọi mạng**; báo cáo theo cùng compute (NFE 5 có CFG so với NFE 10 không CFG). |
| **E6-prior** | p0 = wrapped normal tại tư thế trung bình (đã đo: 126.5° → 51° với σ0 = 0.25). σ0 ∈ {0.25, 0.5}. Tinh chỉnh lại s sau khi đổi p0. | 2 × 9 GPU-h | Đường vận chuyển ngắn hơn (PriorGrad, FrameFlow). Rủi ro: giảm đa dạng; theo dõi M2 và M3. |
| **E7-SC** | Self-conditioning trên x̂1 trước. | 12 GPU-h | FrameDiff, FrameFlow, Proteina. |
| **E8** (thấp) | Head 6D → quaternion. | 9 GPU-h | P4: dữ liệu tay quay nhỏ, nên lợi ích dự kiến nhỏ. |
| **E9** (cược lớn) | Head tay riêng, cross-attention từ tay sang cánh tay. | 12 GPU-h + 2 ngày | Chỉ khi E4 cho thấy lỗi tay vẫn là nút thắt. |
| **E10** (dài hạn) | Prior âm vị học: bất biến hình tay trong một ký hiệu, đối xứng hai tay (SGNify), dùng ranh giới từ bộ phân đoạn. | 9 GPU-h + phân đoạn | "Naturalness" có cơ sở ngôn ngữ học. |
| **E11** (tuỳ chọn) | Metric dữ liệu kiểu MFM/LAND hoặc state cost −log m (GSBM). | 9 GPU-h + correction | Ưu tiên rất thấp: phản biện chỉ ra time-shift có lẽ xuất hiện với mọi metric đắt ở xa dữ liệu (`01` A.2). |

### Giai đoạn 6 — Ghép và xác nhận

- Nền E3 + (s_train, s_inf)* + E4* + (E5/E6/E7 nếu thắng theo §1.3). Ghép theo forward selection, bỏ thành phần không còn thắng.
- Leave-one-out trên Phoenix cho cấu hình cuối.
- CSL × 3 seed, How2Sign × 3 seed.
- Test **một lần** khi người dùng cho phép.

## 3. Ngân sách và lịch

| Khối | GPU-h | CPU-h (DTW) |
|---|---|---|
| G0 (E0.0–E0.4, E1-0) | ~12 | ~15 |
| B×3 + E1-gate | ~54 | ~10 |
| E1-H1 (nếu gate dương) | ~36 | ~6 |
| E2-F | ~18 (thêm) | ~8 |
| E3, E4-c, E4-a, E4-b | ~45 | ~10 |
| E5–E7 | ~40 | ~8 |
| Ghép, LOO, CSL×3, How2Sign×3 | ~200 | ~40 |
| **Tổng** | **~400 GPU-h** | **~100 CPU-h** |

> Sau vòng 1: vì sàng lọc phải ≥ 250 epoch (khoảng 18 GPU-h/run thay vì 9), tổng ước lượng tăng lên **khoảng 600 GPU-h**. Đánh giá suy diễn chạy được trên CPU (khoảng 1 s/chuỗi cho cả sinh và chấm khi máy không quá tải), nên mọi trục suy diễn của E2-F gần như miễn phí.

- Với 1 tiến trình, cộng thời gian viết code và thất bại, thời gian thực tế là **khoảng 4–5 tuần**, tính từ khi job How2Sign kết thúc.
- Đánh giá snapshot chỉ chạy trên tập con. Toàn bộ dev chỉ dành cho ứng viên cuối.

**Thứ tự khi ngân sách bị cắt** (khoảng 70 GPU-h đầu trả lời cả câu hỏi khoa học lẫn câu hỏi "có thắng không"):

1. G0;
2. B×3 + E1-gate;
3. E3;
4. E4-c, rồi E4-a;
5. E2-F;
6. còn lại.

## 4. Tích hợp

- Thêm job `plus_*` vào `methods/signmflow/jobs.py` (đồ thị phụ thuộc, tối đa 1 tiến trình, resume), tên thí nghiệm `signmflow_plus_<ID>_<data>_s<seed>`.
- M1–M4 và bootstrap là script phân tích đọc `predictions.npz`; metric dùng chung nằm trong `src/slp/evaluation/metrics.py`.
- Kết quả ghi vào `docs/research_log.md`. Không sửa tay `results/`.

## 5. Rủi ro và lộ trình

Rủi ro: xem `05_phan_bien_noi_bo.md`. **Thứ tự chạy, lý do nên/không nên chạy từng thí nghiệm, việc cần code và mẫu báo cáo: xem [07_lo_trinh_uu_tien.md](07_lo_trinh_uu_tien.md).**

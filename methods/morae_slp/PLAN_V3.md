# SignRAE v3: chẩn đoán khoảng cách tới SignNMFlow và kế hoạch method cải tiến

Ngày 2026-11-26. Đọc tiếp theo [PLAN.md](PLAN.md) (thiết kế gốc) và [PLAN_V2.md](PLAN_V2.md) (chẩn đoán v1,
kết quả v2). Tài liệu này **chỉ là kế hoạch**: mọi con số "[Đo]" đã có trong repo; mọi con số "[Dự phóng]"
là ước lượng có ghi giả định. Không có thí nghiệm nào trong tài liệu này được chạy trên split test.

**Nhãn bằng chứng**

| Nhãn | Ý nghĩa |
|---|---|
| [Đo] | Số đo trên dev (hoặc test đã có của replica), đường dẫn ghi kèm |
| [Paper] | Số hoặc mô tả lấy từ paper SignNMFlow (arXiv 2609.33339, bản PDF trong `methods/signmflow/docs/`) |
| [Code] | Suy ra trực tiếp từ code của repo |
| [Toán] | Suy ra toán học, có ghi giả thiết |
| [TL] | Tài liệu khác, đã được tổng hợp trong `docs/proposal7/sources/` và `docs/proposal9_signhiflow/02_literature_review.md` |
| [Dự phóng] | Ước lượng, chưa kiểm chứng |

---

## 0. Tóm tắt

1. **Mốc paper và khoảng cách hiện tại.** Paper báo trên test Phoenix 4.37 / 5.57 · 3.85 / 1.07 (JPE body / hand ·
   PA body / hand). Bản tái lập trung thực của repo (replica v2) chỉ đạt 4.93 / 6.54 · 4.35 / 1.30 trên cùng test [Đo].
   SignRAE tốt nhất hiện tại (một mẫu, dev Phoenix) đã **ngang replica**: 5.20 / 6.94 · 4.58 / 1.37 so với 5.17 / 6.98 ·
   4.59 / 1.37 [Đo]. Quy đổi số paper sang "tương đương dev" (§1.2), Phoenix cần thêm khoảng **−0.6 body và −0.9 hand**.
2. **Khoảng cách chủ yếu nằm ở bàn tay.** Body đã gần mức mà ngay cả một tư thế đứng yên cũng đạt được
   (metric thưởng co về trung bình, §2.1). Hand thì còn xa: paper 5.57 so với 6.54 (replica, test).
3. **Bốn kết luận đã đo về SignRAE** (§2.2):
   - Pretrain flow trên union **không** giúp; công thức fine-tune mBART + token dropout + chọn theo val có giúp.
   - Nhiệt độ nhiễu τ < 1 và guidance thấp (w ≈ 1.5) là điểm vận hành tốt nhất cho một mẫu.
   - Trung bình K mẫu vượt đường biên co-về-trung-bình của replica (Phoenix), nên mô hình có nội dung thật.
   - CSL kém replica ở mọi cấu hình đã thử; nguyên nhân chưa xác định (đang chẩn đoán).
4. **Method đề xuất: SignRAE v3**, sáu thành phần, chia hai lớp so sánh:
   - **Không gloss, không truy hồi** (so trực tiếp với SignNMFlow):
     - C1: dự đoán đích x̂₁ thay cho vận tốc;
     - C2: loss trong không gian vị trí của metric (FK, căn theo đúng khung của metric, nặng ở tay);
     - C5: giải mã tối thiểu rủi ro DTW (MBR-medoid, một mẫu thật);
     - C6: head độ dài và dung lượng latent theo chẩn đoán.
   - **Có giám sát nội dung** (báo cáo tách riêng, ghi rõ):
     - C3: loss nhất quán gloss qua một bộ nhận dạng đóng băng (Phoenix, CSL);
     - C4: bộ nhớ truy hồi chỉ từ train.
5. **Lộ trình có cổng** (§4): P0 không huấn luyện (giải mã, độ nhạy protocol) → P1 Phoenix ablation không gloss →
   P2 nội dung (gloss, truy hồi) → P3 chuyển sang CSL, How2Sign → P4 ba seed trên dev, test một lần khi người dùng yêu cầu.
   Tổng khoảng 40–50 GPU-giờ, một tiến trình tại một thời điểm.
6. **Kỳ vọng trung thực** (§5):
   - Số body đạt mục tiêu paper rất dễ nếu chỉ nhờ co về trung bình (trung bình K mẫu đã gần tới), nên body chỉ được
     tính khi vượt đường biên ở cùng tốc độ.
   - Ở hand, ngang paper *chỉ* khả thi nếu C3 hoặc C4 cho hiệu quả cỡ mà truy hồi đã cho trong tài liệu (−10 % trở lên).
   - Không thành phần nào được bảo đảm. Bản tái lập còn chưa đạt số paper, nên một phần khoảng cách có thể đến từ
     protocol không công bố (§2.4).
7. **Phản biện độc lập** (agent riêng, chỉ đọc, đối chiếu số với file gốc) đã được dùng để sửa bản này: 3 lỗi
   (quy lợi ích tay của paper cho quaternion; so sánh D3 không cùng sampler; quy tắc chấp nhận cho phép thắng nhờ co),
   cùng các rủi ro về dự phóng dev→test, tập clip, sampler của C1, bộ nhận dạng C3, truy hồi C4, thống kê.

---

## 1. Mốc so sánh

### 1.1 Số đã có

| | Phoenix (body / hand · PA body / PA hand) | CSL-Daily | How2Sign |
|---|---|---|---|
| Paper SignNMFlow (test) [Paper] | 4.37 / 5.57 · 3.85 / 1.07 | 5.73 / 7.43 · 5.10 / 1.30 | 6.76 / 9.72 · 6.01 / 2.07 |
| Replica v2 (test) [Đo] | 4.93 / 6.54 · 4.35 / 1.30 | 5.93 / 7.69 · 5.26 / 1.38 | 7.45 / 10.67 · 6.63 / 2.35 |
| Replica v2 (dev) [Đo] | 5.17 / 6.98 · 4.59 / 1.37 | 5.72 / 7.38 · 5.06 / 1.33 | 7.50 / 10.85 · 6.73 / 2.35 |
| SignRAE tốt nhất, một mẫu (dev) [Đo] | **5.20 / 6.94 · 4.58 / 1.37** (519 clip, w = 1.5, τ = 0.7) | 6.00 / 7.85 · 5.36 / 1.43 (256 clip, w = 1.5, τ = 0.7) | 7.90 / 10.97 · 7.04 / 2.46 (bản union; bản chỉ-How2Sign đang chạy) |
| SignRAE trung bình K = 8 (dev) [Đo] | 4.74 / 6.58 · 4.20 / 1.29 (519 clip, tốc độ 0.47 / 0.49) | 5.46 / 7.53 · 4.91 / 1.40 (256 clip) | 6.49 / 10.03 · 5.83 / 2.21 (256 clip, tốc độ 0.36 / 0.41) |
| Truy hồi có gloss: SignSparK-SR [TL] | 3.51 / 4.82 · 3.68 / 1.18 | 4.89 / 6.56 · 4.87 / 1.37 | – |

PA body luôn là biến thể "ua" (Procrustes trên chính 12 khớp thân trên). Biến thể này cho PA < JPE như số của
SignNMFlow và SOKE (`methods/signmflow/README.md`, mục "Unspecified"). Không phải mọi bảng đều như vậy: SignSparK có
PA body > JPE body (4.40 / 5.24 trên Phoenix), nên **không so cột PA body** với SignSparK(-SR).

Các số SignRAE ở CSL và How2Sign trong bảng trên là 256 clip, còn replica là đủ dev. Đây là so sánh *không* cùng tập
clip, chỉ để định hướng. P0.0 (§4) chấm lại dự đoán replica trên đúng các tập clip đó.

Hai dòng "K = 8" ở Phoenix (519 clip) và How2Sign lấy từ checkpoint pretrain union. Checkpoint chỉ-Phoenix tốt nhất cho
K = 8, w = 1.5, τ = 1 là 4.76 / 6.48 · 4.21 / 1.28 trên 256 clip [Đo].

### 1.2 Quy đổi số paper sang "tương đương dev"

Không được so số dev với số test. Bảng dưới **chỉ mang tính minh hoạ**: nó dùng độ lệch dev→test của chính replica
(cùng code, cùng dữ liệu) để đặt một mục tiêu dev thô [Dự phóng]. Các lý do khiến nó có thể lạc quan:

- Replica lấy checkpoint cuối, không chọn gì trên dev. SignRAE chọn checkpoint (`val_limit: 256`, `flow.select=flow`)
  và w, τ trên cùng 256 clip dev, nên số dev của SignRAE lạc quan hơn và độ lệch của nó sẽ khác.
- Độ lệch đổi dấu giữa các dataset (Phoenix âm, CSL dương), chỉ từ một seed. Test CSL của replica chạy `--allow-dirty`.
- DTW cộng dồn theo frame, nên độ lệch dạng *tỉ lệ* hợp lý hơn dạng trừ. Theo tỉ lệ, mục tiêu Phoenix là
  4.58 / 5.94 thay vì 4.61 / 6.01.

| | Độ lệch test − dev của replica [Đo] | Paper trừ độ lệch = mục tiêu tương đương dev |
|---|---|---|
| Phoenix | −0.24 / −0.44 · −0.24 / −0.07 | **4.61 / 6.01 · 4.09 / 1.14** |
| CSL | +0.21 / +0.31 · +0.20 / +0.05 | **5.52 / 7.12 · 4.90 / 1.25** |
| How2Sign | −0.05 / −0.18 · −0.10 / 0.00 | **6.81 / 9.90 · 6.11 / 2.07** |

Khoảng cách cần lấp (dev, một mẫu): Phoenix −0.59 / −0.93 · −0.49 / −0.23; CSL khoảng −0.5 / −0.7. Phần lớn là **tay**.

---

## 2. Chẩn đoán

### 2.1 Metric thưởng cho cái gì [Code] [Đo]

1. **DTW tích luỹ, không chuẩn hoá theo độ dài.**
   - `soke_protocol_scores` lấy `dtw_cost(...)[0]`: tổng sai số theo đường căn chỉnh, đơn vị mét
     (`src/slp/evaluation/metrics.py`, mô phỏng `SOKE mGPT/metrics/t2m.py`).
   - Đường căn chỉnh dài ít nhất max(T_sinh, T_GT). Sinh *dài hơn* GT làm tăng số cặp phải cộng; sinh *ngắn hơn* GT
     không làm giảm dưới T_GT. Metric vì vậy bất đối xứng theo độ dài [Toán].
   - Đo: dùng độ dài của replica (trung bình 101.8 frame, ngắn hơn GT 104.2) cho hand 6.99, còn độ dài GT cho 7.03;
     body thì ngược lại, 5.30 so với 5.25 (SignRAE v1, w = 2.5, 256 clip) [Đo]. Hiệu ứng nhỏ và không cùng chiều,
     nên độ dài chỉ là nguồn sai lệch phụ ở Phoenix.
2. **Co về trung bình được thưởng.**
   - Tư thế trung bình train Phoenix, đứng yên, độ dài GT: body 4.76, tốt hơn replica (5.11) trên cùng 256 clip [Đo].
     Các tài liệu khác có số khác (5.53 trên 150 clip trong `docs/proposal7/README.md`; 5.80 trên 519 clip với tư thế
     "không" của bộ chuẩn hoá SOKE) vì khác tập clip và khác định nghĩa trung bình.
   - Co dự đoán replica 25 % về tư thế trung bình làm tốt cả 4 cột (`docs/proposal7/README.md`) [Đo].
   - Replica di chuyển với tốc độ chỉ 0.65 / 0.77 so với GT [Đo], nên số của SignNMFlow cũng nằm trong vùng co.
3. **Mức người–người** (CSL dev, 280 câu có ≥ 2 người ký; `docs/proposal7/08`) [Đo]: 5.58 / 6.29 · 4.76 / 1.12.
   Body của replica đã ngang người; **tay còn kém người khoảng 15 %**. Đó là khoảng trống có ý nghĩa ngôn ngữ.

Hệ quả cho kế hoạch: cải thiện body thêm phần lớn là hành vi ước lượng. Cải thiện tay mới là tiến bộ thật, và mọi
cải thiện đều phải đọc kèm tỉ số tốc độ và so với đường biên co-về-trung-bình ở cùng tốc độ.

### 2.2 Điều đã học được từ SignRAE v1 → v2 [Đo]

| # | Phát hiện | Bằng chứng (dev) | Hệ quả cho v3 |
|---|---|---|---|
| D1 | Flow v1 học thuộc câu train | Train JPE 2.24 / 3.26 so với dev 5.40 / 7.08; val loss có text > không text (1.31 / 0.94) | Nhánh text là nút thắt; cần giám sát nội dung (C3, C4) |
| D2 | Pretrain flow trên union không giúp DTW | Phoenix 5.53 / 7.19 (union) so với 5.32 / 6.97 (chỉ Phoenix), w = 2; CSL 6.28 / 7.99 so với 6.20 / 8.02 | Không đầu tư thêm vào dữ liệu đa ngôn ngữ cho flow |
| D3 | Fine-tune mBART + token dropout + chọn theo val chỉ giúp **ít** khi so cùng sampler; phần lớn là nhờ dừng sớm | Cùng w = 1.5, τ = 1, 256 clip: v1 5.32 / 6.99 → v2-chỉ-Phoenix 5.26 / 6.88 (−0.06 / −0.11). Checkpoint 20k của chính v1 đã cho 5.14 / 6.98, nhưng ở tốc độ 0.68 / 0.78 so với 0.77 / 0.82 | Giữ làm baseline của v3; không quy lợi ích lớn cho mBART fine-tune |
| D4 | τ < 1 và w thấp tốt cho một mẫu | Phoenix w = 1.5, τ = 0.7: 5.196 / 6.941 · 4.576 / 1.367, tốc độ 0.66 / 0.68 | Đi theo hướng "ước lượng có điều kiện": C1, C5 |
| D5 | Trung bình K mẫu vượt đường biên của replica | K = 8, w = 1.5 (checkpoint pretrain union, 519 clip): hiệu số với đường biên −0.18 / −0.29 · −0.17 / −0.047 (CI loại 0); ở w = 3.5 thì không vượt | Mô hình có tín hiệu nội dung; giải mã tối ưu rủi ro có giá trị; P0.1 lặp lại trên checkpoint chỉ-Phoenix |
| D6 | Một mẫu ngang replica nhưng tay hơi kém ở cùng tốc độ | Hiệu số với đường biên: hand +0.12 [+0.01, +0.22] | Phải cải thiện riêng tay: C2, C3, C4 |
| D7 | CSL kém replica ở mọi cấu hình | Một mẫu 6.00–6.20 / 7.85–8.02 so với 5.72 / 7.38 | Chờ chẩn đoán trần tái tạo và độ dài (P0.4) |
| D8 | Trần tái tạo của latent Phoenix không phải nút thắt | 1.45 / 1.77 (G1) | d = 64 đủ cho Phoenix; CSL, How2Sign chưa đo |

### 2.3 Vì sao SignNMFlow có số tốt ở tay [Paper] [Toán]

1. **Dự đoán đích x̂₁** (paper Table 5): trên CSL, CFM vận tốc 7.43 / 9.49 → CFM đích 6.70 / 9.25; của họ vận tốc
   6.44 / 8.13 → đích 5.73 / 7.43.
   - Lý do toán: với đường thẳng x_t = (1−t)x₀ + t x₁, ta có x₁ − x̂₁ = (1−t)(v − v̂). Loss đích bằng loss vận tốc nhân
     (1−t)², tức là bớt trọng số ở t gần 1 và dồn vào cấu trúc thô [Toán].
   - Với MSE, x̂₁(x_t, t, c) học E[x₁ | x_t, c]. Ở t nhỏ nó tiến tới E[x₁ | c], một ước lượng *co về trung bình*
     [Toán]. Đây không chính xác là cực tiểu của DTW kỳ vọng: trung bình theo khung cực tiểu L2 bình phương ở một căn
     chỉnh cố định, còn DTW lấy L2 không bình phương sau khi căn chỉnh thời gian. Nhưng hai thứ cùng chiều: replica
     (sampler re-planned 5 bước) có tốc độ chỉ 0.65 / 0.77, và NFE 1 là cấu hình DTW tốt nhất của replica [Đo].
2. **Lợi ích ở tay đến từ đường đi trên manifold, không từ riêng biểu diễn quaternion** [Paper, Table 3, CSL test]:
   - chỉ đổi sang quaternion (MR) làm *xấu đi*: 6.70 / 9.25 → 6.84 / 9.73;
   - thêm geodesic (GE): hand 9.73 → 8.05;
   - thêm naturalness (NG): 8.05 → 7.43.

   Các phân tích nội bộ (`docs/proposal7`) cho thấy path NG thực chất chỉ đổi tốc độ đi trên geodesic, tức là một
   lịch nhiễu. Giả thuyết của kế hoạch này [Suy luận, chưa kiểm chứng]: một phần lợi thế ở tay đến từ việc loss đo
   trực tiếp trên từng khớp (pose 10 / biểu cảm 0.1, chuẩn hoá từng khớp lên S³), không qua một latent nén.
3. **mBART được fine-tune cùng generator** [Paper]. Paper không ghi lr riêng cho mBART; replica dùng lr chung 2e-4
   (lựa chọn của replica, `methods/signmflow/README.md`, mục "Unspecified" #6). SignRAE dùng 0.1× (2e-5).

SignRAE hiện học vận tốc trong latent d = 64. Khi nén, latent mất phần ngữ nghĩa mà teacher JEPA có (gloss probe 0.274
→ 0.196), nhưng không kém pose thô (0.199) [Đo]. Không có loss nào đo trực tiếp sai số ngón tay ở vị trí khớp.

### 2.4 Phần khoảng cách có thể không lấp được

- Replica cài đúng paper vẫn kém số paper trên test 0.56 / 0.97 (Phoenix), 0.20 / 0.26 (CSL), 0.69 / 0.95 (How2Sign) [Đo].
- Paper không nói cách chọn độ dài sinh ra [Paper]. Với DTW tích luỹ, chính sách độ dài thay đổi số đo (§2.1).
- How2Sign dùng cùng nguồn fitting NSA như paper (bản phát hành của SOKE, `data/README.md`), nên khoảng cách How2Sign
  không do khác nguồn dữ liệu.
- Kết luận: mục tiêu chính của v3 là **vượt replica có ý nghĩa thống kê ở cùng protocol**. Mốc paper là mục tiêu thứ hai,
  so sánh trên test một lần, kèm ghi chú protocol.

---

## 3. Method: SignRAE v3

Kiến trúc giữ nguyên: Sign-JEPA (union, đóng băng) → latent d = 64 (union AE) → DiT-M + cross-attention mBART
(fine-tune) + head độ dài. Baseline của v3 là cấu hình tốt nhất hiện tại: flow chỉ trên dataset đích,
`text.finetune=true`, `text.lr_scale=0.1`, `flow.token_drop=0.1`, `flow.select=flow`, 30k bước (Phoenix, CSL), 40k (How2Sign).

### C1. Dự đoán đích x̂₁ trong latent (không gloss)

**Thay đổi.** DiT xuất x̂₁ (latent đã chuẩn hoá) thay vì vận tốc.

- Loss: L = E‖z − x̂₁‖², tuỳ chọn trọng số w(t) = 1 (đúng paper) hoặc w(t) = 1/((1−t)² + ε) (tương đương vận tốc,
  dùng làm ablation).
- Sampler, cài cả hai biến thể, NFE 5, lưới đều:
  - (S-a) "vận tốc ngụ ý": x_{t+Δ} = x_t + Δ·(x̂₁ − x_t)/(1 − t). Đây là kiểu DDIM; với đường thẳng nó trùng cách
    paper cập nhật **kênh biểu cảm** (Algorithm 2, dòng 13) [Paper] [Toán].
  - (S-b) "neo lại tại x₀" như **kênh pose** của paper (Eq. 53): x_{t+Δ} = (1 − t − Δ)·x₀ + (t + Δ)·x̂₁, với x₀ là
    nhiễu ban đầu được giữ nguyên.
- CFG trên đích: x̂₁ = x̂₁ᵘ + w(x̂₁ᶜ − x̂₁ᵘ), tương đương CFG trên vận tốc ở mỗi t.
- Nhiệt độ τ giữ nguyên ý nghĩa (nhân nhiễu khởi tạo).

**Cơ sở.**
- Paper Table 5 (CSL test): với CFM thường, đích tốt hơn vận tốc −0.73 body / −0.24 hand. Đây là phép so tương tự nhất
  với trường hợp của ta (đường thẳng trong không gian Euclid); hàng của mô hình họ (−0.71 / −0.70) gắn với path manifold [Paper].
- Chồng lấn với D4: C1 phần lớn "mua" độ co về trung bình, giống τ < 1 và NFE thấp. Vì vậy P1 tách hai yếu tố:
  arm "vận tốc + Euler 5 bước" (đổi sampler, giữ mục tiêu) đặt cạnh arm "đích + S-a / S-b".
- [Toán] ở §2.3: đổi trọng số loss sang cấu trúc thô, đúng loại thông tin mà DTW tích luỹ đo.
- Hướng x-prediction cũng được ủng hộ cho dữ liệu chiều cao trong "Back to Basics: Let Denoising Generative Models
  Denoise" (Li & He, 2025) [TL].

**Cài đặt.** `flow.prediction: velocity | target` trong `config.yaml`; `flow.py::training_pair` trả thêm z;
`method.py::_flow_losses` và `_sample` (đổi field thành vận tốc ngụ ý, kẹp 1 − t ≥ 1e-3). DiT không đổi kiến trúc.
Test: (a) với dữ liệu Gauss, nghiệm tối ưu của loss đích cho cùng trường vận tốc như loss vận tốc;
(b) sampler Euler re-planned 1 bước từ x₀ cho đúng x̂₁.

**Rủi ro.** Đầu ra mượt hơn (tốc độ giảm). Đọc kèm đường biên. Nếu x̂₁ gần hằng ở t nhỏ, tăng w hoặc τ.

### C2. Loss trong không gian của metric (không gloss)

**Thay đổi.** Thêm loss vị trí khớp trên chuyển động giải mã từ x̂₁:

- x̂₁ lấy trực tiếp (C1) hoặc từ vận tốc: x̂₁ = x_t + (1 − t)·v̂. C2 vì vậy không phụ thuộc C1, và P1 thử dạng 2 × 2 (C1 × C2).
- Giải mã x̂₁ qua decoder đóng băng → 133-D → FK SMPL-X khả vi (`slp.skeleton_fk`, đã dùng trong `PoseFKLoss`).
- **Đích là pose GT** (FK của 133-D gốc), không phải decode(z). Nhờ vậy loss còn sửa được một phần sai số của chính
  autoencoder (trần tái tạo 1.45 / 1.77).
- Đúng khung của metric: 12 khớp thân trên trừ pelvis; 2 × 21 khớp tay trừ cổ tay; đơn vị cm; smooth-L1.
- L = L_flow + λ_body·L_body + λ_hand·L_hand, chỉ tính khi t ≥ t₀ (mặc định 0.3) để x̂₁ đã có nghĩa, trên 16
  frame lấy đều mỗi chuỗi (như `ae.fk_frames`).
- Mặc định λ_hand > λ_body (ví dụ 1.0 và 0.25), vì tay là khoảng trống thật (§2.1).

**Cơ sở.**
- Loss trên latent coi mọi chiều như nhau. Sai số ngón tay ở vị trí khớp (thứ metric đo) chỉ được phạt gián tiếp.
- Đây là lỗ hổng G3 đã ghi trong `docs/proposal9_signhiflow/02` §8 và E4 của `docs/proposal7`. SignNMFlow có ưu
  thế PA-hand đúng nhờ loss từng khớp [Paper].
- Decoder đóng băng giữ nguyên ý nghĩa latent; gradient chỉ đi vào DiT.

**Cài đặt.** `flow.fk_weight_body`, `flow.fk_weight_hand`, `flow.fk_t_min`, `flow.fk_frames`; tái sử dụng
`PoseFKLoss` (`autoencoder.py`) với decoder của `self.space`. Cache latent hiện chỉ lưu z, nên cần lưu thêm pose GT
(hoặc FK của nó trên các frame lấy mẫu) cho mỗi clip train.
Chi phí ước lượng: +20–35 % thời gian một bước (decoder 4×512 và FK trên 16 frame).

**Rủi ro.** Loss vị trí đẩy về trung bình theo khớp (co thêm). Kiểm tra bằng tỉ số tốc độ tay và đường biên.

### C3. Loss nhất quán gloss (có giám sát nội dung; Phoenix, CSL)

**Thay đổi.**
- Huấn luyện một bộ nhận dạng gloss R (2 tầng Transformer cộng CTC; từ vựng 1,085 gloss ở Phoenix, 2,000 ở CSL [Đo])
  trên *đúng loại đầu vào nó sẽ gặp*:
  - latent GT của train;
  - x̂₁ mà flow baseline dự đoán trên train ở các t ∈ [0.5, 1) (mờ như lúc dùng làm loss);
  - latent GT bị làm mượt hoặc co ngẫu nhiên.

  Chọn R trên một phần train giữ lại (không dùng dev). Sau đó đóng băng R.
- Khi huấn luyện flow, thêm λ_g·CTC(R(x̂₁), gloss) cho t ≥ 0.5.
- Lúc suy luận chỉ dùng text, như SignNMFlow.

**Cơ sở.**
- D1: nhánh text học thuộc câu thay vì học ký hiệu. Câu dev gần như mới hoàn toàn (Phoenix 494/509 câu dev
  không có trong train, CSL 796/797) nhưng gloss thì quen (CSL: 6,578 câu train dùng 2,000 gloss) [Đo].
- Một loss buộc x̂₁ chứa *đúng chuỗi ký hiệu* là tín hiệu trực tiếp cho thành phần nội dung (nguồn phương sai (a)
  trong `docs/proposal7/08`), thứ mà DTW lẫn mắt người đều cần.
- CTC không cần căn chỉnh khung và cho phép thứ tự gloss khác thứ tự chữ.

**Cài đặt.** Script `gloss_recognizer.py` (huấn luyện R trên latent cache; tiêu chí chọn: WER trên dev);
`flow.gloss_weight`, `flow.gloss_t_min`; gloss lấy từ `inputs[i]["gloss"]` (đã có ở Phoenix, CSL; How2Sign không có).

**Công bằng.** Dùng gloss khi huấn luyện, nên khác lớp với SignNMFlow (không gloss). Báo cáo ở dòng riêng.

**Rủi ro.** R có thể bị "đánh lừa" (x̂₁ thoả R mà không giống chuyển động thật). Biện pháp:

- λ_g nhỏ, bật sau 5k bước;
- **cổng hữu ích (P0.3)**: R phải nhận ra gloss từ *dự đoán của replica* tốt hơn hẳn từ dự đoán của một câu khác
  (hoán đổi văn bản), nếu không thì R không mang tín hiệu dùng được cho chuyển động sinh ra;
- đánh giá ngữ nghĩa không dựa vào back-translation: B-T đã bão hoà ở Phoenix (11.34 so với trần GT 11.49), còn trần
  B-T của CSL chỉ 4.74. Thay vào đó dùng khoảng cách hoán đổi văn bản (M4 của `signmflow_plus/analysis/score.py`) và
  một khảo sát người xem nhỏ nếu có thể.

### C4. Bộ nhớ truy hồi chỉ từ train (có giám sát nội dung)

**Thay đổi.**
- Với mỗi câu, lấy top-k = 3 câu train gần nhất theo embedding mBART gốc (đóng băng) cộng một số hạng độ dài.
- Mã hoá latent của 3 câu này bằng một Transformer nhỏ (2 tầng, 256 chiều) thành token bộ nhớ.
- DiT cross-attend tới [token text ; token bộ nhớ]. Bỏ toàn bộ bộ nhớ với xác suất 25 % khi huấn luyện.
- **Loại trùng:** khi huấn luyện, không truy hồi câu có văn bản trùng sau chuẩn hoá (CSL lặp mỗi câu khoảng 2.8 lần).
  Nhờ vậy điều kiện huấn luyện khớp với test, nơi câu gần như luôn mới.
- **Khoá độ dài** dùng độ dài dự đoán từ text (head độ dài) hoặc độ dài câu, không bao giờ dùng độ dài GT.
- **Câu dev có bản trùng trong train** (Phoenix 15/509 câu theo đếm ở đây; `docs/proposal7/08` §2.2 báo khoảng 4.8 %
  theo clip) sẽ nhận truy hồi gần như GT lúc suy luận. Luôn báo cáo kèm kết quả trên tập dev bỏ các câu đó.

**Cơ sở.** Truy hồi là đòn bẩy lớn nhất đã ghi nhận trong SLP:
- SOKE (từ điển ngoài) giảm DTW trung bình khoảng 19.6–23.9 % tuỳ dataset [TL];
- SignSparK-SR (text → gloss → từ điển) đạt 3.51 / 4.82 trên Phoenix, tốt hơn số SignNMFlow [TL].

Ta chỉ dùng train (không từ điển ngoài), nên hiệu quả kỳ vọng nhỏ hơn và chưa biết [Dự phóng].

**Cài đặt.** Chỉ mục truy hồi tính một lần lúc `fit`, lưu cùng checkpoint. Nếu method khác cũng cần, chuyển vào
`src/slp/retrieval.py` kèm test (quy tắc repo).

**Công bằng.** Không gloss, nhưng dùng *chuyển động train* lúc suy luận, như SOKE, ReMoDiffuse. Báo cáo dòng riêng.

**Rủi ro.** Mô hình sao chép chuyển động truy hồi; DTW có thể đẹp nhờ "giống train". Đối chứng: δ-0, tức chỉ trả về
trung bình top-k không qua mô hình.

### C5. Giải mã tối thiểu rủi ro DTW (không gloss)

Ba bộ giải mã từ cùng K mẫu (cùng checkpoint), báo cáo cả ba:

| Bộ giải mã | Đầu ra | Ý nghĩa |
|---|---|---|
| **MBR-medoid** | Mẫu y_i ∈ S có tổng DTW nhỏ nhất tới các mẫu khác (DTW trên vị trí FK, khung của metric, cài trong method) | Một chuyển động thật, tốc độ như một mẫu; xấp xỉ quy tắc Bayes trong tập mẫu |
| Trung bình theo khung (đã có, `sampler.ensemble`) | (1/K)·Σ y_i | Ước lượng điểm; mượt mạnh (tốc độ khoảng 0.47) |
| DBA (DTW barycenter averaging) | Trung bình sau khi căn chỉnh thời gian | Bớt nhoè do lệch nhịp, sắc hơn trung bình theo khung |

**Cơ sở.**
- Giải mã MBR là chuẩn mực khi chất lượng đo bằng một hàm tiện ích (MT: Kumar & Byrne 2004; Freitag và cộng sự 2022) [TL].
- DBA: Petitjean và cộng sự 2011 [TL].
- D5: trung bình K mẫu đã vượt đường biên ở Phoenix.
- MBR-medoid giữ được một chuyển động thật, nên là phương án duy nhất trong ba vẫn "một chuỗi, không nhoè"
  để đặt cạnh SignNMFlow; vẫn phải ghi chú K lần chi phí.
- **Cảnh báo:** medoid theo DTW có xu hướng chọn mẫu biên độ thấp (mẫu gần trung bình thì gần mọi mẫu khác), nên
  "tốc độ như một mẫu" chưa được chứng minh và phải đo. Cả ba bộ giải mã là **giải mã có nhận biết metric**
  (dùng tiện ích DTW), phải ghi rõ như vậy khi báo cáo.

**Cài đặt.** `sampler.decoder: single | mean | medoid | dba`, `sampler.ensemble: K`. DTW cài riêng trong method
(không gọi code metric, theo quy tắc "không tính metric trong method").

### C6. Độ dài và dung lượng latent (theo chẩn đoán)

- **Độ dài.** Dùng độ dài GT chỉ giúp −0.15 / −0.05 ở Phoenix [Đo]. Ở CSL có một manh mối cụ thể: độ dài sinh ra
  trung bình 111.0 frame so với GT 119.3 (thiếu khoảng 7 %), trong khi Phoenix gần đúng (103.3 so với 104.2) [Đo].
  Chờ `sweep_gtlen_csl` và `sweep_gtlen_how2sign`. Nếu oracle giúp > 0.2, sửa head độ dài:
  - tìm nguyên nhân thiên lệch (ví dụ L1 trên log T cho trung vị, còn CSL có phân phối độ dài lệch);
  - thêm đặc trưng số token và số dấu câu; hiệu chỉnh hệ số trên train.

  Đổi L1 sang Laplace một mình không thay đổi gì, vì cả hai cho cùng điểm ước lượng (trung vị).
  Chính sách "co độ dài" (nhân α < 1) chỉ dùng làm **phân tích độ nhạy**, không dùng cho kết quả chính, vì nó khai thác
  tính bất đối xứng của DTW tích luỹ (§2.1).
- **Latent.** Nếu trần tái tạo CSL hoặc How2Sign (`diag_recon_*`) cao hơn rõ rệt Phoenix (1.45 / 1.77), huấn luyện lại AE
  d = 128 (teacher đóng băng, khoảng 1.5 giờ) rồi chạy lại flow cho dataset đó.

---

## 4. Lộ trình thực nghiệm và cổng

Mỗi bước chạy **một tiến trình** (stage trong `job.py`), chỉ trên dev. Thời gian dựa trên tốc độ đã đo: 30k bước Phoenix
khoảng 1.5 giờ, CSL khoảng 1.6 giờ, How2Sign 40k khoảng 2.4 giờ. Sweep 256 clip khoảng 0.1–0.3 giờ. Đường biên (CPU)
khoảng 0.3–1 giờ.

| Pha | Thí nghiệm | Chi phí | Cổng / quyết định |
|---|---|---|---|
| **P0.0** | Chấm dự đoán replica đã lưu trên đúng các tập clip SignRAE dùng (CSL, How2Sign 256 clip; tập xác nhận); đường biên CSL, How2Sign | khoảng 1 giờ CPU | Thay bảng §1.1 bằng so sánh cùng tập clip |
| **P0.1** | MBR-medoid, trung bình, DBA với K ∈ {8, 16} trên checkpoint chỉ-Phoenix tốt nhất; đường biên | khoảng 1.5 giờ | Giữ bộ giải mã nếu thoả quy tắc 2–3 bên dưới ở JPE-hand; báo tốc độ |
| P0.2 | Độ nhạy độ dài α ∈ {0.85, 0.9, 1.0, 1.1} (chỉ phân tích) | 0.3 giờ | Không dùng cho kết quả chính |
| P0.3 | Cổng C3: huấn luyện R (chọn trên train giữ lại), kiểm tra trên dự đoán replica so với hoán đổi văn bản | 0.5 giờ | R phải phân biệt rõ câu đúng với câu hoán đổi; nếu không, C3 dừng |
| P0.4 | `diag_recon_csl/how2sign`, `sweep_gtlen_*` (đã xếp hàng) | 0.5 giờ | Chọn nhánh C6 |
| P0.5 | Baseline Phoenix với seed 1 | 1.5 giờ | Ước lượng nhiễu giữa seed |
| **P1.A** | Tách sampler khỏi mục tiêu: (i) vận tốc + Euler 5 bước; (ii) đích + S-a; (iii) đích + S-b; cùng ngân sách | 3 × 1.5 giờ | So baseline (5.196 / 6.941 · 4.576 / 1.367) cùng sweep (w, τ) |
| P1.B | C2 (FK; λ_hand ∈ {0.5, 1.0}) trên mục tiêu tốt nhất của P1.A, kèm C2 trên vận tốc (2 × 2) | 2–3 × 1.8 giờ | Quy tắc 2–3 |
| P1.C | mBART `lr_scale` ∈ {0.3, 1.0} trên cấu hình tốt nhất | 2 × 1.5 giờ | Quy tắc 2–3 |
| **P2.A** | + C3 (Phoenix, CSL) | 1.8 + 2.0 giờ | Báo cáo ở lớp "có gloss" |
| P2.B | + C4 (cài đặt khoảng 1–2 ngày công) | 2 + 2.2 giờ | Báo cáo ở lớp "truy hồi" |
| **P3** | Cấu hình không gloss tốt nhất (và có gloss) trên CSL, How2Sign; d = 128 nếu P0.4 yêu cầu | 4–8 giờ | |
| **P4** | 3 seed trên dev cho cấu hình cuối mỗi dataset; test một lần khi người dùng cho phép, git sạch | 12–15 giờ | Báo cáo cuối |

**Quy tắc chấp nhận một thành phần** (đăng ký trước; chỉ số chính là JPE-hand):

1. **Tách tập chỉnh và tập xác nhận.** Sampler (w, τ, K) và mọi siêu tham số chọn trên 256 clip "tune" (đang dùng).
   Quyết định chấp nhận tính trên phần dev còn lại: Phoenix 263 clip; CSL, How2Sign phần ngoài 256 clip đó.
2. **Thắng thật, không nhờ co.** Hiệu số với đường biên co-về-trung-bình của replica ở *cùng tốc độ* có cận trên CI 95 %
   < 0 ở JPE-hand, **và** hiệu số ghép cặp với cấu hình đang tốt nhất có cận trên < 0 ở JPE-hand.
3. **Không làm hỏng cột khác.** Không cột nào xấu đi có ý nghĩa thống kê (CI loại 0 về phía xấu).
4. **Thống kê.**
   - Bootstrap ghép cặp theo câu. CSL dùng bootstrap theo cụm câu (797 câu, 1,077 clip).
   - Hiệu chỉnh Holm cho số arm trong cùng pha.
   - Ước lượng nhiễu giữa các seed một lần: chạy lại baseline Phoenix với seed 1 (khoảng 1.5 giờ). Mọi hiệu số nhỏ hơn
     chênh lệch giữa hai seed coi như không phân biệt được.
5. **So cùng tập clip.** Mọi so sánh với replica dùng dự đoán đã lưu của replica, chấm trên đúng tập clip đó
   (trên Phoenix, replica ở 256 clip là 5.11 / 6.84 nhưng ở 519 clip là 5.17 / 6.98).

**Điều kiện dừng sớm.** Sau P1, nếu không thành phần nào thoả quy tắc ở Phoenix, dừng hướng latent-flow và chuyển ngân
sách sang C4 (truy hồi) hoặc sang các kế hoạch rộng hơn trong `docs/proposal7/08` (SignTAF) và `docs/proposal9`
(SignHiFlow).

---

## 5. Kỳ vọng định lượng

| Thành phần | Bằng chứng định lượng | Kỳ vọng trên Phoenix dev (hand) [Dự phóng] |
|---|---|---|
| C1 đích | Paper Table 5: −0.24 đến −0.70 hand (CSL test, hai mô hình) | −0.1 đến −0.4 |
| C2 FK | Chưa có số trực tiếp cho SLP | 0 đến −0.3 |
| C5 medoid | Chưa đo; trung bình K = 8 cho −0.36 (ở tốc độ 0.47) | 0 đến −0.2 ở tốc độ một mẫu |
| C3 gloss | Chưa có số trực tiếp | 0 đến −0.4 |
| C4 truy hồi | Tài liệu có từ điển ngoài: −20 % trở lên; chỉ train: chưa biết | 0 đến −0.7 |

Các hiệu quả không cộng thẳng được, và một phần của mỗi hiệu quả là co về trung bình. Số thô dễ "đạt" mục tiêu chỉ
nhờ co: trung bình K = 8 đã cho body 4.74 (Phoenix, mục tiêu 4.61) và 6.49 (How2Sign, mục tiêu 6.81) [Đo]. Vì vậy
các kịch bản dưới đây được nêu ở **cùng tốc độ với một mẫu hiện tại** (body khoảng 0.66, tay khoảng 0.68 ở Phoenix),
và chỉ được coi là đạt khi hiệu số với đường biên của replica có CI < 0:

- **Bi quan** (chỉ C1 và C2 có tác dụng nhỏ): Phoenix dev khoảng 5.1 / 6.75. Vượt replica ở cùng tốc độ với biên nhỏ,
  chưa tới mục tiêu paper.
- **Trung bình** (C1, C2, C5 đạt giữa khoảng): khoảng 5.0 / 6.5. Body gần mục tiêu, hand còn thiếu khoảng 0.5.
- **Lạc quan** (C3 hoặc C4 hiệu quả như tài liệu): khoảng 4.7 / 6.0 hoặc thấp hơn, tức ngang mục tiêu paper, nhưng chỉ ở
  lớp "có gloss" hoặc "truy hồi".

Mọi mức "tương đương test" chỉ mang tính minh hoạ (§1.2). Kết luận cuối chỉ đến từ test chạy một lần.

---

## 6. Protocol báo cáo và tính công bằng

| Dòng báo cáo | Dữ liệu | Suy luận | So được với |
|---|---|---|---|
| v3 chuẩn | SSL trên union chuyển động; flow chỉ dataset đích; không gloss; không truy hồi | Một mẫu (w, τ chọn trên dev), độ dài từ head | SignNMFlow, SOKE, MaDiS (ghi chú SSL union và một seed) |
| v3 + MBR | Như trên | Medoid của K mẫu (K lần chi phí, một chuỗi thật) | Như trên, ghi chú K |
| v3 ước lượng điểm | Như trên | Trung bình hoặc DBA K mẫu | Chỉ so tại cùng tốc độ (đường biên) |
| v3 + gloss | Thêm gloss train | Một mẫu | SignSparK-SR, SRVQ (lớp dùng gloss) |
| v3 + truy hồi | Thêm chuyển động train lúc suy luận | Một mẫu | SOKE (truy hồi) |

Quy tắc chung:
- Mọi dòng có tỉ số tốc độ và gia tốc (body, tay) và số clip.
- PA body là biến thể "ua".
- Không chọn gì trên test. Test chạy một lần qua `scripts/run.py --split test` từ git tree sạch, ghi vào `results/test_access_log.jsonl`.
- Nếu muốn dòng "chỉ dữ liệu dataset đích" hoàn toàn sạch, cần pretrain lại Sign-JEPA và AE trên riêng dataset đó
  (khoảng 6 giờ GPU cho Phoenix). Đây là đối chứng nên chạy trước khi viết bài.

---

## 7. Phản biện và rủi ro

1. **"C1 chỉ là co về trung bình."** Đúng một phần: x̂₁ ở t nhỏ là trung bình có điều kiện. Vì vậy C1 chỉ được chấp nhận
   nếu vượt đường biên co-về-trung-bình của replica ở cùng tốc độ (quy tắc 3), không chỉ ở số thô.
2. **"C3 và C4 làm so sánh với SignNMFlow không công bằng."** Đúng, nên chúng được báo cáo ở lớp riêng (§6).
   Kết quả chính của v3 là C1, C2, C5, C6.
3. **"Độ lệch dev→test của replica không áp dụng cho mô hình khác."** Có thể. Đó là lý do §1.2 là [Dự phóng], và
   đánh giá cuối là test một lần.
4. **"Metric không đo chất lượng."** Đã có bằng chứng (§2.1). Vì vậy mọi kết quả kèm động học, đường biên, và nếu có
   thời gian, back-translation độc lập cùng một khảo sát người xem nhỏ. Một DTW tốt hơn đi kèm tốc độ < 0.5 không được
   gọi là tiến bộ.
5. **"Một seed."** P4 chạy 3 seed trên dev trước khi đọc test.
6. **Rò rỉ.** Truy hồi loại câu trùng; bộ nhận dạng gloss chỉ học trên train; không đọc test để chọn bất cứ gì.
7. **Ngân sách.** P1 bị giới hạn một tiến trình. Nếu P1.A không qua cổng, bỏ P1.B–C và đi thẳng P2.

---

## 8. Bản đồ cài đặt

| Thành phần | File | Khoá config | Test cần thêm |
|---|---|---|---|
| C1 | `flow.py`, `method.py` | `flow.prediction`, `flow.target_weighting`, `sampler.replan: implied \| anchored` | Tương đương vận tốc ↔ đích trên dữ liệu Gauss; S-a và S-b một bước từ t = 0 đều cho x̂₁ |
| C2 | `method.py` (tái dùng `PoseFKLoss`), cache thêm pose GT | `flow.fk_weight_body`, `flow.fk_weight_hand`, `flow.fk_t_min`, `flow.fk_frames` | Gradient chỉ vào DiT (decoder đóng băng); với x̂₁ = z, loss bằng sai số tái tạo FK của AE |
| C3 | `gloss_recognizer.py` mới, `method.py` | `flow.gloss_weight`, `flow.gloss_t_min`, `gloss.recognizer` | CTC giảm trên batch đồ chơi; không chạy khi thiếu gloss |
| C4 | `retrieval.py` mới (hoặc `src/slp/retrieval.py`), `dit.py` | `retrieval.k`, `retrieval.dropout`, `retrieval.exclude_duplicates` | Không truy hồi chính nó hay câu trùng; đổi thứ tự batch không đổi kết quả |
| C5 | `decode.py` mới, `method.py::_sample` | `sampler.decoder`, `sampler.ensemble` | Medoid thuộc tập mẫu; DBA của các chuỗi giống nhau = chính chuỗi đó |
| C6 | `dit.py::DurationHead` | `duration.dist: l1 \| laplace` | Log-likelihood đúng với phân phối đồ chơi |
| Job | `job.py` (plan v3) | – | – |

Mọi thay đổi giữ mặc định của v2 để tái lập được kết quả cũ. `pytest tests/morae_slp` và toàn repo phải pass trước mỗi run.

---

## 9. Tài liệu tham khảo

- SignNMFlow: He và cộng sự, *Naturalness-Guided Manifold Flow Matching for Sign Language Production*, arXiv 2609.33339
  (Table 1, 2, 5, 8; App. C.5, D.1).
- MoRAE: arXiv 2607.29180. SOKE: arXiv 2411.17799. SignSparK: arXiv 2603.10446. MaDiS: arXiv 2601.19577.
- Kumar & Byrne, *Minimum Bayes-Risk Decoding for Statistical Machine Translation*, HLT-NAACL 2004.
- Freitag và cộng sự, *High Quality Rather than High Model Probability: Minimum Bayes Risk Decoding with Neural
  Metrics*, TACL 2022.
- Petitjean, Ketterlin, Gançarski, *A global averaging method for dynamic time warping, with applications to
  clustering*, Pattern Recognition 2011.
- Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017 (phương án thay thế cho C2).
- Li & He, *Back to Basics: Let Denoising Generative Models Denoise*, 2025 (x-prediction).
- Zhang và cộng sự, *ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model*, ICCV 2023.
- Phân tích nội bộ: `docs/proposal7/` (DTW thưởng co, mức người–người, SignTAF), `docs/proposal9_signhiflow/`
  (khoảng trống G1–G7, Estimate-then-Transport), [PLAN_V2.md](PLAN_V2.md) §8.

---

## 10. Trạng thái cài đặt (2026-11-26, chưa chạy thực nghiệm)

Mọi khoá mới mặc định về đúng hành vi v2 (đã kiểm tra: config và checkpoint cũ nạp như v2, cùng chuỗi số ngẫu nhiên).

| Thành phần | Code | Test (`tests/morae_slp/test_morae_slp.py`) |
|---|---|---|
| C1 | `flow.py` (`to_target`, `to_velocity`, `target_weight`, `replan`), `method.py` | Đồng nhất thức đích ↔ vận tốc; trọng số `velocity` = loss vận tốc; S-a / S-b theo đúng Eq. 53 và Alg. 2; fit → predict → reload; checkpoint ghi `flow_prediction`; đích + rk4/heun bị chặn |
| C2 | `method.py::_fk_losses` (dùng `PoseFKLoss`, pose GT trong cache) | Gradient tới DiT, không tới decoder đóng băng |
| C3 | `gloss.py` (bộ nhận dạng, CLI `train` / `check`), `method.py::_gloss_loss` | CTC học được; tăng cường giữ padding; gradient về đầu vào; từ chối bộ nhận dạng khác chuẩn hoá |
| C4 | `retrieval.py`, `dit.py::MemoryEncoder`, `method.py` | Không tự truy hồi, loại câu trùng; lưu / nạp lại; token bộ nhớ theo dropout của text và CFG; reload cho cùng dự đoán |
| C5 | `decode.py`, `method.py::_combine` | DTW trùng với DTW của evaluator; medoid là một mẫu thật; DBA của các chuỗi giống nhau là chính nó |
| C6 | `dit.py::DurationHead`, `method.py::_calibrate_duration`, `sampler.length_scale` | Đặc trưng số token dùng mặt nạ chưa bỏ; bias lưu trong `model.pt` |
| Công cụ | `sweep.py` (`--decoder --confirm --drop-train-duplicates --length-scale`), `frontier.py`, `smoke.py`, `job.py --plan v3` (27 stage), `slp.evaluation.metrics.cluster_bootstrap_ci` | Holm, tập tune / confirm, bootstrap theo cụm |
| Tổ hợp | Mọi tuỳ chọn cùng lúc | `test_all_v3_options_together` |

Kết quả: 65 test của method và 484 test toàn repo pass.

Một review độc lập (chỉ đọc) đã tìm ra 1 lỗi và một số rủi ro, tất cả đã được sửa kèm test hồi quy:
- **Lỗi:** khi `exclude_duplicates=false`, query train tự truy hồi chính nó. Đã sửa: luôn loại chính nó.
- Đặc trưng số token của head độ dài từng thấy mặt nạ đã bị bỏ token.
- Đích + rk4/heun từng đánh giá tại t = 1.
- `--confirm` phải khớp `val_limit` của run.
- `frame_costs` từng tốn bộ nhớ, nay chia khối.
- WER của gloss chỉ tính trên clip có gloss.
- Đường dẫn stats của FK dùng `resolve()`.
- Mặc định trong code nay khớp `config.yaml`.

Ghi chú: snapshot `model_step*.pt` không có bias độ dài (chỉ `model.pt` có). Sweep ghi `duration_bias` vào mỗi dòng.

Chưa chạy: smoke test trên dữ liệu thật (stage đầu của plan v3) và mọi thí nghiệm. Plan v3 chỉ khởi động khi người dùng yêu cầu, một tiến trình mỗi lúc.

**Cập nhật hàng đợi (2026-11-26 tối):** C3 và C4 vẫn được cài đặt và có test, nhưng bị loại khỏi hàng đợi mặc định của
`job --plan v3` (`EXCLUDED_V3`) vì thuộc lớp so sánh khác (có gloss, có truy hồi; §6) và để tiết kiệm khoảng 5 giờ GPU.
Dòng chính để so với SignNMFlow chỉ gồm C1, C2, C5, C6.

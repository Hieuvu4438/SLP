# SignRAE: chuyển MoRAE sang Sign Language Production

Kế hoạch thiết kế, cài đặt và thực nghiệm cho `methods/morae_slp`.

- Ngày: 2026-11-24.
- Nguồn chính: MoRAE, arXiv 2607.29180v2 (bản PDF ở `third_party/MoRAE/docs/`), cùng mã inference
  công khai ở `third_party/MoRAE/src/morae`.
- Bối cảnh SLP lấy từ: `docs/proposal9_signhiflow/02_literature_review.md` (đã kiểm chứng ngày
  2026-11-23), `methods/signmflow/README.md` (kết quả replica) và `docs/proposal7/06_ket_qua_thuc_nghiem.md`.

Quy ước: **[paper]** là điều MoRAE nói rõ. **[code]** là điều thấy trong mã công khai. **[ours]**
là lựa chọn của chúng tôi khi paper không nói, hoặc khi phải đổi cho SLP; mỗi lựa chọn kèm lý do.

---

## 0. Tóm tắt

MoRAE có ba ý chính:

1. Một encoder **Motion-JEPA** che toàn bộ quỹ đạo của một số khớp và dự đoán biểu diễn
   *(không phải giá trị thô)* của chuyển động sạch.
2. Feature 768-D của encoder đóng băng có số điều kiện rất xấu (rank-deficient). Một lớp attention
   **nén** chúng xuống latent 32-D, và một decoder nhẹ giải mã thẳng ra chuyển động.
3. Một **DiT flow-matching** tiêu chuẩn sinh latent đó từ text.

Lý do latent này tốt cho flow: các hướng phương sai lớn mang phần *phối hợp giữa các khớp* C(x),
chính là những hướng flow khôi phục chính xác nhất. Phần dư riêng từng khớp P(x) bị đẩy xuống đuôi
phổ, và decoder tất định lo phần đó.

**SignRAE** giữ nguyên ba ý này và thay những chỗ không hợp với SLP:

| Thành phần | MoRAE (HumanML3D) | SignRAE (Phoenix / CSL / How2Sign) | Lý do |
|---|---|---|---|
| Đơn vị che | 22 khớp, mỗi khớp gồm nhiều kênh (pos/rot/vel/contact) | 42 nhóm: 41 khớp xoay SMPL-X + 1 nhóm expression. Trộn 3 độ thô: *khớp / chuỗi động học (ngón tay, cánh tay) / cả bàn tay* | Biểu diễn 133-D chỉ có rotation cục bộ, nên các khớp liền nhau trên cùng một ngón là "đường tắt cục bộ" kiểu MoRAE §4.3 cảnh báo (§3.2) |
| Dữ liệu pretrain SSL | train của từng dataset | **hợp các tập train** của cả 3 dataset (không text, không val/test) | SSL không cần nhãn; SMPL-X 133-D giống nhau giữa các ngôn ngữ |
| Điều kiện text | CLIP sentence embedding, chỉ qua AdaLN | mBART (SOKE, frozen) token features: pooled → AdaLN, cộng **cross-attention theo token** | SLP là ánh xạ chuỗi → chuỗi; MoRAE tự nhận điều kiện câu là giới hạn |
| Độ dài | cho trước (benchmark T2M) | **duration head** dự đoán log T từ text | Protocol SOKE không cho độ dài GT |
| Latent | d = 32 | d = 64 mặc định, ablation 32/96 | 133 DoF thật (41×3 + 10), so với khoảng 70 DoF thật của HumanML3D |
| Đánh giá | FID, R-precision | DTW-JPE / DTW-PA-JPE (SOKE), cộng các guardrail động học | Benchmark SLP |

Mã cài đặt ở `methods/morae_slp/`. Mục 9 có bản đồ file.

---

## 1. Bài toán và benchmark

**Đầu vào / đầu ra.** Input là một câu văn bản (DGS-Đức, CSL-Trung, ASL-Anh). Output là chuỗi pose
SMPL-X 133-D đã chuẩn hoá theo stats SOKE:

- upper body: 10 khớp × 3 (axis-angle), kênh 0:30;
- tay trái: 15 × 3, kênh 30:75;
- tay phải: 15 × 3, kênh 75:120;
- jaw: kênh 120:123;
- expression: 10 kênh, 123:133.

Loader: `src/slp/data/sign_smplx.py`.

**Dữ liệu trên server.**

| Dataset | Config | train / val / test | Độ dài tb (train) |
|---|---|---|---|
| Phoenix-2014T | `phoenix_smplx` | 7,092 / 519 / 642 | ≈128 frame (25 fps) |
| CSL-Daily | `csl_smplx` | 18,399 / 1,077 / 1,176 | ≈112 |
| How2Sign | `how2sign_smplx_soke` (protocol SOKE) | 30,938 / 1,722 / 2,316 | dài hơn, 24 fps |

Mọi clip đã qua crop MaDiS, nên độ dài nằm trong [40, 400] frame.

**Metric.** Theo protocol SOKE, cài trong `src/slp/evaluation/metrics.py`:

- `soke_dtw_jpe_{body,hand}` và `soke_dtw_pa_jpe_{body,body_ua,hand}`;
- guardrail M2: `soke_{speed,accel}_ratio_{body,hand}`.

DTW-JPE là chi phí **tích luỹ**, nên nó thưởng cho việc hồi quy về pose trung bình (`docs/proposal7`).
Vì vậy mọi so sánh phải kèm tỉ số động học (mục 7.4).

**Mốc cần vượt (Phoenix).**

| Phoenix | JPE body | JPE hand | PA body (ua) | PA hand |
|---|---|---|---|---|
| SignNMFlow paper (test) | 4.37 | 5.57 | 3.85 | 1.07 |
| SignNMFlow replica v2 (dev) | 5.17 | 6.98 | 4.59 | 1.37 |
| SOKE paper (test) | 6.04 | 7.72 | 4.77 | 1.38 |
| MaDiS (test) | 4.97 | 6.59 | – | – |
| mean pose tĩnh (dev, sàn) | 5.80 | 8.13 | 5.09 | 1.61 |

Mục tiêu thực tế là vượt replica v2 *trên dev, cùng pipeline đánh giá*. Không so trực tiếp số dev
của mình với số test của paper.

---

## 2. Đọc kỹ MoRAE

### 2.1 Phương pháp, theo paper §3 và Table S14

**Motion-JEPA** [paper]:

- Input: x ∈ R^{T×D}, với HumanML3D D = 263, chia thành J = 22 khối khớp.
- Che ⌊0.3·J⌉ = 7 khớp. Mọi kênh của khớp bị che, ở mọi frame, được thay bằng một
  *giá trị mask học được*.
- Online encoder f_θ: 8 block Transformer, width 768, 12 head, MLP×4, **rotary**, một token cho mỗi frame.
- Predictor p_φ: 6 block, width 384, 6 head, chiếu ra 768.
- Teacher f_ξ là EMA của f_θ, momentum tăng từ 0.996 lên 1.
- Loss: L = (1/T) Σ_t smoothL1( p_φ(f_θ(x̃))_t , sg[LN(f_ξ(x))_t] ), tính trên mọi frame hợp lệ.
- Optimizer: AdamW β = (0.9, 0.95), lr 2e-4 → 1e-5 cosine, warmup 20 epoch, wd 0.05, batch 256,
  600 epoch, clip 24–200 frame.
- Feature cho các bước sau: H = LN(Σ_{l=5..8} h^(l)) ∈ R^{T×768}, lấy từ **teacher**.

**Compact latent** [paper]:

- Compressor E_ψ: **một** lớp multi-head attention (24 head, width 3072) có residual và LayerNorm,
  sau đó một linear head ra (μ, log σ²) của z ∈ R^{T×d}.
  - Kiểm tra lại số tham số: q, k, v là 768→3072 và out là 3072→768, tổng ≈ 9.44M. Khớp với
    9.50M ở Table S16.
- Decoder D_ω: linear vào → 4 block (width 512, 8 head, rotary) → LN → linear ra D.
  12.8M tham số, giải mã **một giai đoạn** thẳng ra chuyển động.
- Loss: ‖D(z) − x‖² + 1e-4·KL(q(z|H) ‖ N(0, I)).
- Optimizer: AdamW (0.9, 0.95), lr 2e-4 → 1e-5, warmup 10 epoch, wd 0.01, batch 384, 400 epoch.
  Chọn checkpoint có **reconstruction val thấp nhất**.
- Khi sinh: dùng posterior **mean** làm latent, chuẩn hoá từng kênh bằng stats của tập train.

**Latent flow (DiT-M)** [paper]:

- 12 block, width 512, 8 head, FFN 2048, dropout 0.1, adaLN-zero trên (timestep + CLIP ViT-B/32
  sentence embedding). 58M tham số, một token cho mỗi frame, tối đa 196 frame.
- Đường nội suy z_t = (1−t)ε + t z, regress vận tốc z − ε bằng MSE trong toạ độ đã chuẩn hoá.
- Bỏ text với xác suất 0.1 (để dùng CFG).
- AdamW (0.9, 0.99), lr 2e-4 cosine xuống 1%, warmup 2,000 iteration, wd 1e-5, batch 64, 500 epoch,
  EMA 0.9999 (dùng EMA khi sample).
- Sampling: ODE RK4 trên lưới đều, 5 bước (20 lượt đánh giá có guidance), w = 4.5 trên HumanML3D;
  12 bước, w = 9.5 trên KIT.

**Mã công khai** [code] **khác paper**:

- Dùng DiT-XL (16 lớp, width 1024).
- Decoder **hai giai đoạn**: FeatureDecoder 32 → 768, rồi MotionDecoder → 67-D "essential".
- Sampler Euler 24 điểm, w = 8.5.
- Uncond = embedding 0.
- Paper (Sec. C) lại báo decoder hai giai đoạn **kém hơn** (FID 0.0309 so với 0.0253). Chúng tôi
  theo paper: decoder một giai đoạn.
- Mã công khai không có code train. Mọi thứ về training ở đây là [ours], cài theo Table S14.

### 2.2 Phát hiện chính của paper, kèm ý nghĩa cho SLP

1. **Dự đoán tốt hơn tái tạo.** Ở cùng kiến trúc và d = 32, FID giảm 27.5% so với AE và 33.9% so
   với VAE (Table 3). MAE che cùng kiểu khớp nhưng tái tạo kênh thô vẫn kém hơn (0.0402).
   - Hàm ý cho SLP: lợi ích đến từ **mục tiêu trong không gian biểu diễn**, không chỉ từ mask.
2. **Đơn vị che quyết định kết quả.** Che theo khoảng thời gian cho 0.058, nhóm kênh xáo trộn cho
   0.038, khớp cộng thời gian cho 0.027, khớp nguyên quỹ đạo cho 0.025 (Table 4).
   - Hàm ý: phải bỏ các "đường tắt cục bộ". Với SLP, đường tắt cục bộ chính là **các khớp liền
     nhau trên cùng một ngón** (mục 3.2).
3. **Nén là bắt buộc.** Feature 768-D và chuyển động thô 263-D đều rank-deficient (κ ≈ 1e15), và
   flow 5 bước thất bại trên chúng. Latent nén có κ ∈ [2.2, 252] thì sinh tốt.
   - Nếu kéo dãn phổ tới κ ≈ 700 thì FID tụt về mức của chuyển động thô (Table S13).
4. **Thứ tự phổ quan trọng hơn κ.** Whitening giữ nguyên thông tin mà vẫn làm FID tăng 62%.
   Đổi thứ tự (reverse-aligned) ở cùng phổ làm FID tăng hơn gấp đôi (Table 7).
   - Hàm ý: **không** whiten, **không** dùng loss có trọng số kiểu whitened. Giữ MSE vận tốc
     trong toạ độ chuẩn hoá từng kênh.
5. **Độ rộng latent**: 32 cho 0.0253, 64 cho 0.0262, 16 cho 0.0501. Bão hoà ở 32–64.
6. **Bottleneck 32-D gắn thẳng vào encoder thì kém** (0.0385). Phải dùng encoder rộng cộng một
   compressor tách riêng.

### 2.3 Những gì paper không nói, và lựa chọn của chúng tôi

| Câu hỏi | Lựa chọn [ours] | Lý do |
|---|---|---|
| Vị trí của "learned mask value" | Một vector m ∈ R^{D_in}, thay vào kênh của nhóm bị che trước khi chiếu vào encoder | Cách tự nhiên nhất. Mã công khai không có encoder |
| Predictor có biết khớp nào bị che không | Không có thêm token. Giá trị mask học được đã đủ để phân biệt | Paper không nhắc thêm token |
| Target LN | `F.layer_norm` không affine trên output khối cuối của teacher | Theo I-JEPA |
| Lịch momentum | Tuyến tính 0.996 → 1 theo update | I-JEPA |
| Smooth-L1 β | 1.0 (mặc định PyTorch) | Target đã qua LN, nên có thang cỡ 1 |
| Compressor có RoPE không | Có | Nhất quán với encoder và decoder; feature đã chứa vị trí, nên ảnh hưởng nhỏ |
| KL | Trung bình theo frame của tổng theo chiều, β = 1e-4 | Eq. 2 |
| Uncond của CFG | pooled = 0 (như [code]), token text thay bằng **một null token học được** | Cross-attention cần ít nhất một key |
| Phân phối t khi train flow | Đều U(0, 1) | Paper chỉ nói "linear path" |
| Trọng số MSE theo phần | w_p·(số kênh của phần), nên `[1,1,1,1]` đúng bằng MSE thường trên 133 kênh | Giữ đúng Eq. 2 khi không chỉnh trọng số |
| Độ chính xác số | Duration head, tổng 4 lớp teacher và trích latent để cache đều chạy FP32; phần còn lại dùng bf16 | Ở bf16, log T có độ phân giải ≈3%, làm độ dài chuỗi bị lượng tử hoá |
| Checkpoint flow | EMA ở update cuối (như "epoch-500 EMA"). Có thêm snapshot để chọn trên dev | Paper chọn checkpoint trên val bằng FID; ta chọn bằng DTW trên dev bằng script ngoài |

---

## 3. Phân tích chuyển giao: chỗ nào giữ, chỗ nào phải đổi

### 3.1 Những gì giữ nguyên

- Kiến trúc và recipe ba giai đoạn, cùng số tham số encoder, predictor, compressor, decoder và DiT.
- Loss của từng giai đoạn.
- Giữ "toạ độ gốc + MSE gốc" khi train flow (finding 4). Không whiten, không reweight.
- RK4 trên lưới đều cộng CFG. Lưới (bước, w) được quét trên dev.

### 3.2 Đơn vị che cho thân trên và hai bàn tay

Ở HumanML3D, mỗi khớp xuất hiện qua bốn kênh dư thừa (vị trí, xoay, vận tốc, tiếp xúc). Che *một
phần kênh* thì vẫn chép được từ kênh khác của cùng khớp. Vì thế MoRAE che *toàn bộ* khớp.

Biểu diễn 133-D của SLP khác:

- mỗi khớp chỉ có 3 kênh axis-angle cục bộ, nên che một khớp là xoá hết thông tin của khớp đó;
- nhưng **ba khớp của cùng một ngón bị ràng buộc sinh cơ học** (góc DIP ≈ 2/3 góc PIP; ngón
  nhẫn và ngón út gập cùng nhau). Che khớp giữa thì khớp đầu và khớp cuối gần như "chép" ra nó.
  Đây chính là đường tắt cục bộ ở mức chuỗi động học, tương tự che thời gian ở MoRAE.

Thông tin ngữ âm học của ký hiệu (handshape, orientation, location, movement) là *phối hợp* ở mức
cao hơn:

- handshape là cấu hình đồng thời của 5 ngón;
- location là chuỗi vai → khuỷu → cổ tay.

Vì vậy chúng tôi đề xuất **mặt nạ đa độ thô** [ours]. Với mỗi mẫu, chọn một độ thô theo xác suất
`jepa.mask.unit_probs`:

- `joint` (mặc định 0.4): đơn vị là 1 trong 42 nhóm. Giống paper.
- `chain` (0.4): đơn vị là một chuỗi động học. Có 10 ngón (3 khớp mỗi ngón), cánh tay trái
  (collar, shoulder, elbow, wrist), cánh tay phải, đầu (neck, head, jaw) và expression.
- `hand` (0.2): đơn vị là cả bàn tay (15 khớp). Các nhóm còn lại là đơn vị đơn.

Tỉ lệ 0.4/0.4/0.2 là **lựa chọn của chúng tôi, chưa có công trình nào hỗ trợ trực tiếp**. Bằng
chứng gián tiếp ở mục 11.1 (đều là nhận dạng ký hiệu, không phải sinh): BEST cho thấy chỉ che tay
tốt hơn chỉ che thân (54.6 so với 47.7, chỉ-thân còn kém hơn scratch 52.0), nhưng kết hợp cả hai
tốt nhất (58.8); SignMAE cho thấy che vùng tay tốt hơn che tube. Chưa có ablation che ngón so với
che cả tay, và chưa có ablation nào cho bài toán sinh.

Sau đó xáo thứ tự các đơn vị và che dần cho tới khi số nhóm bị che ≥ round(0.3·42) = 13.

- Mức `hand` buộc mô hình suy tay từ tay kia, cánh tay và mặt. Điều này có ý nghĩa cho ký hiệu hai
  tay đối xứng; phần còn lại là P (không đoán được) và JEPA sẽ không nhớ nó, đúng như mong muốn.
- Ablation M1 (mục 7) so sánh `joint`-only (paper) với mặc định trộn và với trộn nặng `hand`.

Cờ `jepa.input_velocity` (mặc định bật) nối thêm Δx của từng nhóm vào input [ours]. Nó tạo kênh dư
thừa giống HumanML3D; Δx bị che cùng nhóm, nên không rò rỉ.

### 3.3 Điều kiện text phải theo từng token

Câu SLP dài 10–30 token, và chuỗi ký hiệu bám theo thứ tự nội dung. Một embedding câu qua AdaLN
không truyền được *trật tự*. Mọi phương pháp SLP mạnh đều có attention tới từng token:
SOKE và SignNMFlow dùng cross-attention vào mBART, MotionHiFlow-SLP dùng joint attention.

SignRAE giữ AdaLN(t + pooled text + embedding ngôn ngữ) của MoRAE và **thêm một lớp
cross-attention** vào token mBART trong mỗi block DiT. Output projection của cross-attention được
khởi tạo 0, nên lúc khởi tạo block tương đương block MoRAE.

`text.mode: adaln` giữ đúng paper (chỉ pooled) để ablate (T1).

Encoder text là `third_party/SOKE/deps/mbart-h2s-csl-phoenix`, đóng băng, giống
SignNMFlow / MotionHiFlow-SLP để so sánh công bằng.

### 3.4 Độ dài chuỗi

- Một MLP trên (pooled mBART, embedding ngôn ngữ) dự đoán log T, loss L1 trên log T, train cùng DiT.
  Bias khởi tạo = trung bình log T của tập train.
- Khi sinh: T̂ = clamp(round(exp(·)), 40, 400).
- `sampler.lengths_from_npz` khoá độ dài theo predictions của một run khác, để làm ablation có
  kiểm soát (giống MotionHiFlow-SLP).

### 3.5 Metric DTW và rủi ro hồi quy về trung bình

MoRAE tối ưu **độ trung thực phân phối** (FID). DTW-JPE so *từng mẫu* với *một* GT, nên kỳ vọng
có điều kiện E[x | text] thắng một mẫu thật từ p(x | text).

Một latent "flow-friendly" làm mẫu sắc nét và tự nhiên hơn, và **có thể làm DTW xấu đi** nếu không
kiểm soát. Ta xử lý bằng ba cách:

1. Báo cáo kèm tỉ số tốc độ và gia tốc (M2). Mọi kết luận về DTW phải ở **tốc độ tương đương**
   (mục 7.4).
2. Hai núm chỉnh ở inference, đều tất định và không dùng GT:
   - nhiệt độ nhiễu τ (ε ← τ·ε);
   - CFG w.
   Hai núm này cho một frontier DTW ↔ động học, và ta chọn điểm làm việc trên dev.
3. Lợi thế của MoRAE nên được đo ở chỗ nó *nên* thắng: DTW-PA-hand (hình dạng tay), back-translation
   và R-precision của evaluator M1, và độ tự nhiên.

### 3.6 SSL trên hợp các dataset

JEPA và AE không dùng text. SMPL-X 133-D cùng stats SOKE dùng chung cho cả ba dataset. Do đó
`pretrain.py` train JEPA (và tuỳ chọn AE) trên **hợp các tập train**: 7.1k + 18.4k + 30.9k ≈ 56k
clip.

- Không đọc val/test, trừ val dùng để chọn checkpoint AE (được phép, val là để chọn mô hình).
- DiT vẫn train trên **từng dataset** như protocol SOKE.

Đây là lợi thế thật của hướng SSL. MaDiS và SOKE đều cho thấy pretrain đa dataset giúp (xem
`docs/proposal9_signhiflow/02` §1.2 mục 3).

Ablation D1: so JEPA chỉ dùng train của một dataset với JEPA dùng hợp.

### 3.7 Độ rộng latent

133 DoF thật, trong đó hai bàn tay chiếm 90. Tư thế bàn tay có số chiều nội tại thấp: PCA của MANO
cho thấy khoảng 6–12 thành phần mỗi tay đã giữ phần lớn phương sai. Phần thân thì ít chiều.

Chúng tôi chọn d = 64 [ours] làm mặc định. Paper cho thấy 64 tương đương 32 trên HumanML3D
(κ = 252, vẫn dưới ngưỡng ≈ 700).

Gate G1 (mục 7) chọn d nhỏ nhất thoả cả ba điều kiện:

- trần reconstruction (DTW của decode(encode(x_GT)) trên dev) đủ thấp;
- κ < 300;
- không có hướng nào < 1e-3·λ₁.

### 3.8 Decoder và chi tiết tay

Decoder tất định cung cấp phần P (chi tiết riêng từng khớp). Với SLP, chi tiết ngón tay quyết định
DTW-PA-hand.

- Mặc định: MSE trên 133 kênh đã chuẩn hoá (paper), có `ae.part_weights` (body, lhand, rhand, face).
- Tuỳ chọn `ae.fk_weight` thêm loss vị trí khớp theo protocol SOKE: 12 điểm thân trên so với pelvis,
  và 2×21 điểm tay so với cổ tay. Loss dùng FK SMPL-X chính xác, khả vi, ở
  `src/slp/skeleton_fk.py`, đơn vị cm, smooth-L1.
  - Loss này đặt trọng số theo đòn bẩy (vai sai lệch ảnh hưởng tới vị trí cổ tay), đúng với cách
    DTW-JPE đo. Ablation A2.

---

## 4. Thiết kế SignRAE

### 4.1 Ký hiệu

- x ∈ R^{T×133}: pose đã chuẩn hoá.
- G = 42 nhóm: g_k với k < 41 gồm 3 kênh của khớp k; g_41 gồm 10 kênh expression.
- u = [x, Δx] ∈ R^{T×266} khi bật `input_velocity`; nhóm g_k bao gồm cả kênh Δx tương ứng.
- Mask M ∈ {0, 1}^G, không đổi theo thời gian (che *cả quỹ đạo*):
  ũ_t = u_t ⊙ (1 − M̄) + m ⊙ M̄, trong đó M̄ là M trải ra theo kênh.

### 4.2 Giai đoạn 1: Sign-JEPA (`jepa.py`)

```
h = f_θ(ũ)                         # 8 block RoPE, width 768, 12 head; Linear(266→768) ở đầu
p = p_φ(h)                         # Linear(768→384), 6 block width 384 / 6 head, Linear(384→768)
y = LN(f_ξ(u)_last)                # teacher EMA, no grad
L_JEPA = mean_{b, t ∈ valid} smoothL1(p_{b,t}, y_{b,t})   # trung bình theo kênh
ξ ← m_k ξ + (1 − m_k) θ,  m_k = 0.996 + (1 − 0.996)·k/K
H = LN(Σ_{l=5..8} f_ξ(u)^{(l)})     # feature cho các giai đoạn sau
```

Chi tiết train:

- Crop ngẫu nhiên dài U[32, 400] (paper dùng 24–200; clip SLP dài tới 400, và crop dài tránh lệch
  RoPE lúc dùng).
- Attention có mask padding. Loss chỉ tính trên frame hợp lệ.

### 4.3 Giai đoạn 2: compact latent (`autoencoder.py`)

```
a  = H + MHA(LN(H))               # q,k,v: 768→3072 (24 head × 128), out 3072→768, RoPE
a  = LN(a)
μ, logσ² = Linear(768 → 2d)(a)
z  = μ + σ ⊙ ε  (train) ;  z = μ (khi sinh latent cho flow)
x̂  = D_ω(z)                        # Linear(d→512), 4 block RoPE (512, 8 head), LN, Linear(512→133)
L_AE = Σ_parts w_p·MSE_p(x̂, x) + β·KL + λ_vel·MSE(Δx̂, Δx) + λ_fk·FK(x̂, x)
```

- Encoder và teacher đóng băng.
- Chọn checkpoint theo MSE reconstruction trên val.
- Sau đó tính μ_c, σ_c từng kênh trên posterior mean của tập train (calibration) và lưu vào checkpoint.

**Các đối chứng cùng kiến trúc** (`latent.kind`):

- `jepa` (mặc định): như trên.
- `ae` / `vae`: encoder cùng kiến trúc, khởi tạo ngẫu nhiên, train **chung** với compressor và
  decoder bằng reconstruction, không mask. β = 0 cho `ae`, β = 1e-4 cho `vae`. Đây là đối chứng
  của Table 3 trong paper.
- `raw`: flow chạy thẳng trên 133-D đã chuẩn hoá, decoder là hàm đồng nhất. Đây là đối chứng
  "uncompressed raw" của paper.

### 4.4 Giai đoạn 3: latent flow (`dit.py`, `flow.py`)

```
q = (μ(x) − μ_c) / σ_c                       # toạ độ flow, cache sẵn cho toàn bộ tập train
q_t = (1 − t) ε + t q,  t ~ U(0,1),  ε ~ N(0, I)
c   = TimeEmb(t) + W_p·pool(mBART(text)) + LangEmb(lang)       # pool, text = 0 khi drop (p = 0.1)
v̂   = DiT(q_t, c, tokens = W_t·mBART(text) | null)
L   = mean_{valid}‖v̂ − (q − ε)‖² + λ_dur·|logT̂ − log T|
```

DiT block (MARDM adaLN-zero, theo [code], thêm cross-attention):

```
x = x + g1 ⊙ SA(mod(LN x; s1, b1))            # mask padding
x = x + CA(LN x, text tokens)                 # out-proj khởi tạo 0; text_mode = adaln thì bỏ
x = x + g2 ⊙ MLP(mod(LN x; s2, b2))
```

Khối cuối là adaLN rồi Linear(512 → d), khởi tạo 0. Positional encoding sin tuyệt đối, như [code].

### 4.5 Inference

1. T̂ lấy từ duration head.
2. ε ~ N(0, τ²I), với seed tất định theo (seed, tên clip).
3. Giải ODE dq/dt = v_w(q, t) từ t = 0 đến 1 trên lưới đều (`rk4` | `heun` | `euler`), trong đó
   v_w = v_u + w(v_c − v_u), batch cond và uncond cùng lúc.
4. μ = q·σ_c + μ_c, rồi x̂ = D_ω(μ).
5. Trả về x̂[:T̂] (133-D chuẩn hoá). Bộ đánh giá chung chạy FK và DTW.

---

## 5. Cấu hình mặc định và tài nguyên

`methods/morae_slp/config.yaml` là nguồn chuẩn. Tóm tắt:

| Giai đoạn | Kiến trúc | Tối ưu | Batch / micro | Update |
|---|---|---|---|---|
| JEPA | enc 8×768/12h, pred 6×384/6h | AdamW (0.9, 0.95), lr 2e-4 → 1e-5, wd 0.05, warmup 3k | 128 / 32 | 60k (≈140 epoch trên hợp 56k clip; tương đương số update của paper) |
| AE | compressor 1×MHA 3072/24h, decoder 4×512/8h, d = 64 | AdamW (0.9, 0.95), lr 2e-4 → 1e-5, wd 0.01, warmup 1k | 128 / 32 | 30k |
| Flow | DiT 12×512/8h, FFN 2048, cross-attn vào mBART 1024-D | AdamW (0.9, 0.99), lr 2e-4 → 2e-6, wd 1e-5, warmup 2k, EMA 0.9999 | 64 / 32 | 60k Phoenix, 100k CSL, 150k How2Sign |
| Sampler | RK4, 5 bước, w = 4.5, τ = 1.0 | quét w ∈ {1.5, 2.5, 3.5, 4.5, 6.5}, bước ∈ {5, 8, 12}, τ ∈ {1.0, 0.8, 0.6} trên dev | | |

Ước lượng thời gian trên RTX 5880 Ada (bf16), mang tính định hướng:

- JEPA: khoảng 0.25–0.35 s/update, nên 60k update mất khoảng 4–6 giờ.
- AE: khoảng 1–2 giờ.
- Flow Phoenix: khoảng 3–4 giờ, vì latent đã cache nên không cần chạy encoder.

Tổng số tham số khi sinh: DiT (≈70M, gồm cả cross-attn) cộng decoder 12.8M cộng mBART encoder
(đóng băng).

---

## 6. Quy trình chạy

Chi tiết lệnh ở README. Người dùng tự chạy, theo `feedback-plan-only-no-runs`.

```bash
# (a) SSL trên hợp các tập train, một lần cho cả 3 dataset
python -m methods.morae_slp.pretrain --stage jepa \
  --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/jepa_union
python -m methods.morae_slp.pretrain --stage ae \
  --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/ae_union_jepa_d64 \
  --init-jepa outputs/morae_slp/pretrain/jepa_union/jepa.pt
# (b) Chẩn đoán biểu diễn (gate G1)
python -m methods.morae_slp.diagnostics --data phoenix_smplx --raw-cp --out g0.json   # G0, CPU
python -m methods.morae_slp.diagnostics --data phoenix_smplx \
  --ae outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt --spectrum --recon 519 --probe --out g1.json
# (c) Flow theo từng dataset, chấm trên dev (mặc định split val)
python scripts/run.py --method morae_slp --data phoenix_smplx --name morae_phoenix --seeds 0 \
  method.init_ae=outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt
# (d) Chọn sampler trên dev
python -m methods.morae_slp.sweep --run outputs/runs/morae_phoenix/morae_slp/<run_id> \
  --cfg 1.5,2.5,3.5,4.5,6.5 --steps 5,8 --temperature 1.0,0.8,0.6
```

Toàn bộ các pha ở mục 7 đã được viết thành script: `configs/experiments/morae_slp/run.sh {g0,g1,g2,sweep,ablations,scale}`.

Nếu không có `init_ae` / `init_jepa`, `fit()` tự train cả ba giai đoạn trên tập train của dataset
đang chạy. Cách này tự chứa nhưng chậm hơn.

---

## 7. Kế hoạch thực nghiệm

Mọi thực nghiệm chỉ dùng **dev**, seed 0 trước. Test chỉ chạy khi người dùng yêu cầu.

### 7.1 Gate, theo thứ tự

**G0. Không train (CPU, vài phút).**

1. `pytest tests/morae_slp`.
2. Smoke run CPU: `debug_limit = 8`, mỗi giai đoạn 2 update.
3. Probe ridge trên dữ liệu thô (`diagnostics --raw-cp`): mỗi nhóm khớp được các nhóm khác ở cùng
   frame dự đoán được bao nhiêu (R², tức tỉ lệ C/P), tách theo thân / tay / ngón. Đồng thời so
   R² của một khớp khi *có* và *không có* hai khớp cùng ngón.
   - **Kỳ vọng:** cùng ngón ≫ khác ngón. Điều đó xác nhận đường tắt cục bộ ở mục 3.2.
   - **Nếu sai:** chuyển mặc định về `joint`-only.

**G1. Biểu diễn (JEPA + AE, khoảng 6–8 giờ GPU).** Train JEPA trên hợp, rồi AE với d ∈ {32, 64}.
Chạy `diagnostics`. Gate đạt khi:

- trần reconstruction trên dev (decode(encode(GT))) có JPE-hand ≤ 0.5 × JPE-hand của replica v2
  (≤ 3.5) và PA-hand ≤ 0.9;
- κ < 300 và không có hướng nào dưới 1e-3·λ₁;
- ρ(λ, L_C/L_P) > 0.7, tức C nằm ở các hướng dẫn đầu, như paper;
- probe gloss tuyến tính (mAP top-300 gloss, Phoenix) theo thứ tự JEPA-H > AE-control-H > raw.
  - Nếu JEPA ≤ AE thì giả thuyết lõi chưa đứng trên SLP. Dừng và xem lại mask hoặc input trước
    khi train flow.

**G2. Sinh trên Phoenix (khoảng 4 giờ GPU mỗi arm).** Ba arm có DiT, sampler và budget giống hệt
nhau; chỉ latent khác:

- **S-jepa**: latent `jepa` d = 64;
- **S-ae**: latent `ae` (đối chứng Table 3);
- **S-raw**: latent `raw`.

Chấm DTW, PA và M2 trên 519 dev. Quét w và τ cho từng arm với cùng lưới, rồi so trên frontier ở
tốc độ tương đương (7.4).

- **Thành công:** S-jepa thắng S-ae và S-raw có ý nghĩa (paired bootstrap CI theo chuỗi không chứa
  0) trên JPE-hand và PA-hand ở tốc độ tương đương.

**G3. So với replica SignNMFlow v2 trên dev.** Dùng cùng evaluator. Báo cả điểm làm việc
"tốc độ = GT" và điểm "DTW tốt nhất" (kèm M2).

**G4. Mở rộng.** CSL rồi How2Sign (thứ tự theo ràng buộc của người dùng), sau đó 3 seed.

### 7.2 Ablation, theo mức ưu tiên

| ID | Biến | Mức | Câu hỏi |
|---|---|---|---|
| M1 | đơn vị che | joint-only (paper) / trộn mặc định 0.4/0.4/0.2 / nặng `hand` 0.2/0.4/0.4 | Bỏ đường tắt ngón tay có giúp không; che cả bàn tay giúp tới đâu |
| W1 | d | 32 / 64 / 96 | Độ rộng tối thiểu đủ cho tay |
| D1 | dữ liệu JEPA | chỉ dataset đích / hợp 3 dataset | SSL đa ngôn ngữ có giúp không |
| T1 | điều kiện text | adaln (paper) / adaln + cross-attn | Bắt buộc theo phân tích 3.3; ablation để định lượng |
| A2 | loss decoder | MSE / MSE + FK (λ_fk = 0.05) | Trần reconstruction và PA-hand |
| S1 | sampler | RK4-5 / RK4-8 / Euler-12 / Heun-8; w; τ | Điểm làm việc |
| A1 | AE trên hợp hay từng dataset | | Có cần decoder riêng theo dataset không |

Ablation lý thuyết của paper (whitening, isospectral) **không** ưu tiên. Chúng là phân tích, không
giúp tăng điểm. Chỉ làm nếu viết paper (mục 7.5).

### 7.3 Ngân sách tuning công bằng

Mọi baseline nội bộ (S-ae, S-raw) dùng cùng lưới sampler {w, bước, τ} và cùng số update.
Ghi không gian tìm kiếm vào README.

### 7.4 Guardrail chống hồi quy về trung bình

Với mỗi arm, báo cáo:

1. DTW-JPE và DTW-PA (body, hand);
2. `soke_speed_ratio_*` và `soke_accel_ratio_*`.

Muốn tuyên bố "DTW tốt hơn" thì phải đồng thời:

- (a) tỉ số tốc độ của tay nằm trong [0.85, 1.15], **hoặc**
- (b) thắng frontier shrink-to-mean của replica ở tốc độ tương đương (`metrics.frontier_effect`).

### 7.5 Chẩn đoán phụ, cho bài viết nếu có

- Phổ latent, κ và L_C / L_P theo từng hướng PCA (đã cài).
- Lỗi khôi phục theo hướng của DiT, ẑ = z_t + (1−t)v, chuẩn hoá theo λ_i (chưa cài; dễ thêm vào
  diagnostics).
- Phần dư P chứa bao nhiêu jitter của pseudo-GT, đo bằng năng lượng tần số cao của P so với C.

---

## 8. Phản biện và rủi ro

1. **FID-tốt có thể không thành DTW-tốt.** Đây là rủi ro lớn nhất (mục 3.5). Có thể kết quả cuối
   là SignRAE tự nhiên hơn (M2 ≈ 1, PA-hand tốt) nhưng JPE không thắng replica. Khi đó đóng góp
   phải được kể theo trục "độ trung thực + PA-hand + R-precision", không theo JPE.
2. **Pseudo-GT SMPL-X nhiễu.** Có lúc lợi: P chứa jitter của fitting, JEPA bỏ P, nên latent "lọc"
   bớt nhiễu. Có lúc hại: reconstruction và DTW so với GT nhiễu, nên trần DTW cao hơn. Theo dõi
   trần ở G1.
3. **JEPA collapse.** Teacher EMA cộng predictor hẹp thường đủ. Theo dõi độ lệch chuẩn của target
   theo kênh; nó phải > 0.1 sau LN. Gate G1 kiểm bằng probe gloss.
4. **Rank-deficiency trong input 133-D.** Một số kênh gần hằng (jaw, expression). Compressor có
   KL nên không sinh hướng suy biến. G1 đo κ.
5. **Duration head yếu**, mà DTW tích luỹ nhạy với độ dài. Báo MAE độ dài (log) trên dev. Thử
   ablation khoá độ dài bằng predictions của replica.
6. **Lệch tần số khung hình.** Phoenix 25 fps, How2Sign 24, CSL 30 nguyên bản (đã chuẩn bởi loader).
   JEPA trên hợp chịu lệch nhỏ; RoPE tương đối thì không nhạy.
7. **Không có bằng chứng SLP trực tiếp cho latent SSL kiểu RAE.** Đây là đóng góp mới, đồng thời
   là rủi ro; agent khảo sát được giao kiểm tra (mục 10).

---

## 9. Bản đồ cài đặt

| File | Nội dung |
|---|---|
| `config.yaml` | Mọi siêu tham số (mục 5) |
| `data.py` | Nhóm khớp và đơn vị che (`GROUPS`, `UNITS`), dataset/batch, crop, `text_input` |
| `layers.py` | RoPE, attention có mask, block Transformer (pre-LN) |
| `jepa.py` | `SignJEPA`: online encoder, predictor, teacher EMA, mask học được, loss, `features()` |
| `autoencoder.py` | `Compressor`, `MotionDecoder`, `LatentSpace` (jepa / ae / vae / raw), loss AE, calibration |
| `text.py` | mBART frozen (hoặc `bytes` cho test CPU) |
| `dit.py` | `LatentDiT` (adaLN-zero + cross-attn), `DurationHead` |
| `flow.py` | cặp huấn luyện FM, sampler Euler / Heun / RK4 có CFG |
| `training.py` | vòng lặp update chung: warmup + cosine, microbatch, bf16, EMA, chọn checkpoint |
| `stages.py` | train JEPA / AE, calibration, cache latent, checkpoint `jepa.pt` / `ae.pt` (dùng chung cho `method.py` và `pretrain.py`) |
| `sweep.py` | CLI quét sampler (w, số bước, solver, τ) của một run đã train, **chỉ trên dev** |
| `method.py` | `Method.fit/predict/save/load` qua ba giai đoạn, cache latent, export `predictions.npz` |
| `pretrain.py` | CLI train JEPA / AE trên hợp các tập train |
| `diagnostics.py` | CLI: phổ / κ / L_C, L_P, trần reconstruction, probe gloss, probe ridge trên dữ liệu thô |
| `tests/morae_slp/` | mask, RoPE, JEPA / EMA, AE, flow (oracle velocity), fit-predict-reload |

Phần dùng chung đặt trong `src/slp/evaluation/metrics.py`, mỗi hàm có test:

- `latent_spectrum` (κ, số hướng dưới ngưỡng);
- `directional_loadings` (Eq. S1);
- `multilabel_average_precision` (probe gloss).

---

## 10. Tài liệu tham khảo chính

- MoRAE: Zhu et al., arXiv 2607.29180v2.
- I-JEPA: Assran et al., CVPR 2023.
- RAE: Zheng et al., arXiv 2510.11690.
- FAE: Gao et al., arXiv 2512.07829.
- S-JEPA: Abdelfattah & Alahi, ECCV 2024.
- MARDM: Meng et al., CVPR 2025.
- SOKE: arXiv 2411.17799.
- MaDiS: arXiv 2601.19577.
- SignSparK: arXiv 2603.10446.
- SignNMFlow: arXiv 2609.33339.
- MotionHiFlow: arXiv 2604.23264.

Bảng số đầy đủ ở `docs/proposal9_signhiflow/02_literature_review.md`.

---

## 11. Khảo sát bổ sung: SSL trên pose ký hiệu và latent SSL cho sinh

Tra cứu ngày 2026-11-24. Các ID đều đã kiểm tra trên arXiv. **[abs]** nghĩa là chỉ đọc abstract.

### 11.1 Đơn vị che trong SSL cho pose ký hiệu (bài toán nhận dạng)

| Công trình | Che gì | Dự đoán gì | Kết quả liên quan |
|---|---|---|---|
| SignBERT+ (2305.04868, TPAMI'23) | token hai tay của từng frame; che khớp, frame hoặc clip 2–8 frame | khớp 2D qua decoder MANO | MSASL100: scratch 78.2 → che khớp 82.0, che frame 82.7, che clip 83.9, kết hợp cả ba 84.9 |
| BEST (2302.05075, AAAI'23) | các phần thân, tay trái, tay phải của từng frame | mã d-VAE rời rạc | MSASL1000: chỉ tay 54.6, **chỉ thân 47.7 (kém hơn scratch 52.0)**, cả hai 58.8 |
| MASA (2405.20666) | 90% frame, ưu tiên frame nhiều chuyển động | phần dư chuyển động | Mục tiêu dự đoán đóng góp nhiều hơn chiến lược che |
| SignMAE (2605.02094) | nửa bàn tay đang chuyển động | chỉ vùng tay | Top-1 keypoint WLASL2000: che tube 26.5 → che tay 47.1 |
| SHuBERT (2411.16765, ACL'25) | đoạn 3 frame, theo từng stream | ID cụm k-means | Kiểu che gần như không đổi kết quả trên How2Sign |

**Hệ quả cho SignRAE:**

- Che ở mức bàn tay và chuỗi động học có cơ sở thực nghiệm (BEST, SignMAE). Chưa có ablation trực
  tiếp giữa che ngón và che cả tay, nên M1 vẫn cần chạy.
- Dự đoán trong không gian biểu diễn hoặc chuyển động thường tốt hơn tái tạo toạ độ thô (MASA,
  SignMAE). Điều này cùng chiều với MoRAE Table S2 (MAE kém JEPA).
- Không tìm thấy mô hình kiểu JEPA nào cho pose ký hiệu. Gần nhất là S-JEPA, nhưng cho nhận dạng
  hành động.

### 11.2 Latent SSL cho sinh

- **FAE (2512.07829):** encoder là **một** lớp self-attention trên token DINOv2 đóng băng.
  - Ablation FID: một lớp attention 2.27, linear 2.38, Transformer 6 lớp 2.47, feature thô 12.99.
  - Lý do: tái tạo feature là mục tiêu yếu hơn pretrain, nên encoder sâu overfit. Attention bỏ được
    thông tin toàn cục dư thừa, điều mà linear không làm được.
  - Đây là cơ sở cho compressor một lớp ở §4.3.
- **RAE (2510.11690) [abs], VA-VAE (2501.01423) [abs], REPA-E (2504.10483) [abs]:** sinh trong hoặc
  căn chỉnh với feature của mô hình nền (ảnh).
- **SignTok (2611.23944):** SLP gần nhất có latent liên tục học tự giám sát.
  - MAE che 50% frame, có loss phụ ngữ nghĩa mBART.
  - Generator là flow matching của SignFlow.
  - Back-translation tăng ở 5/6 thiết lập, nhưng **DTW-JPE không cải thiện đều**.
  - Tác giả ghi nhận latent mã hoá cả danh tính người ký.
  - Pose lấy từ bản phát hành của SignSparK, không phải SMPL-X của SOKE, nên không so số trực tiếp.

**Khoảng trống.** Chưa có công trình nào dùng latent nén từ teacher SSL (JEPA) cho SLP theo protocol
SOKE. Đây là điểm mới của SignRAE.

**Rủi ro bổ sung, từ SignTok:**

- Danh tính người ký có thể nằm trong latent. Thân trên SMPL-X dùng shape cố định, nhưng phong cách
  ký vẫn còn.
- Lợi ích ngữ nghĩa có thể không chuyển thành DTW. Điều này khớp với §3.5 và §8.1.

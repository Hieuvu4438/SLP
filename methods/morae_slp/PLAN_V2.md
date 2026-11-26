# SignRAE v2: chẩn đoán run v1 và kế hoạch cấu hình lại

Ngày 2026-11-25. Đọc cùng [PLAN.md](PLAN.md); bản v1 không bị sửa.
Mọi số đo đều trên **dev (`val`)**. Split test chỉ chạy khi người dùng yêu cầu.

## 0. Tóm tắt

- Run v1 (`morae_phoenix_main`, giữ nguyên làm **đối chứng**) đạt Phoenix dev 5.55 / 7.35 · 4.89 / 1.50.
  Replica SignNMFlow v2 đạt 5.17 / 6.98 · 4.59 / 1.37, cùng split và cùng evaluator.
- Nguyên nhân chính là **flow học thuộc 7k câu Phoenix**. Trên câu train, model đạt JPE 2.24 / 3.26;
  trên dev, flow loss *có* text còn **tệ hơn** khi bỏ text (1.31 so với 0.94).
- Nguyên nhân thứ hai là **DTW-JPE thưởng cho việc co về trung bình**:
  - Một tư thế trung bình đứng yên đạt JPE-body 4.76 trên 256 clip dev, tốt hơn mọi model.
  - Replica chuyển động với tốc độ chỉ bằng 0.66 GT. Run v1 được báo cáo ở 1.01, tức điểm vận hành bất lợi nhất.
  - So ở cùng mức co (τ = 0.3, tốc độ 0.72), v1 gần như hoà replica: 5.19 / 6.87 · 4.57 / 1.36 so với 5.11 / 6.84 · 4.53 / 1.35.
- v2 sửa phần tổng quát hoá text → động tác bằng các thay đổi sau:
  - pretrain flow trên **union 56k câu** của ba ngôn ngữ;
  - **fine-tune mBART** và **token dropout**;
  - **chọn checkpoint theo val**;
  - fine-tune riêng cho từng dataset.
- v2 luôn báo cáo hai điểm vận hành:
  - **Tự nhiên:** một mẫu, tốc độ ≈ 1.
  - **Ước lượng điểm:** trung bình K mẫu, xấp xỉ E[pose | text], cùng điểm vận hành với replica và paper.
  - Mỗi điểm luôn kèm tỉ số tốc độ / gia tốc.

## 1. Đối chứng v1 (không chỉnh sửa)

| Thành phần | Đường dẫn |
|---|---|
| Sign-JEPA (union, 60k) | `outputs/morae_slp/pretrain/jepa_union/jepa.pt` |
| Latent d = 64 (union, 30k) | `outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt` |
| Chẩn đoán G1 | `outputs/morae_slp/diagnostics/g1_jepa_d64_phoenix.json` |
| Flow Phoenix (60k, `select: last`) | `outputs/runs/morae_phoenix_main/morae_slp/seed0_20261125-062634-648119` |
| Sweep + chẩn đoán sampler | `<run>/sweep.jsonl` (22 dòng, gồm các dòng chẩn đoán dưới đây) |
| Text gain theo checkpoint | `outputs/morae_slp/diagnostics/text_gain.jsonl` |
| So sánh replica (256 clip) | `outputs/morae_slp/diagnostics/compare_replica_dev256.json` |

Job v1 ở trạng thái `done` (`python -m methods.morae_slp.job --status`). Các thư mục trên không bị
job v2 ghi đè: v2 dùng `outputs/morae_slp/job_v2` và `pretrain/flow_union_v2`.

## 2. Chẩn đoán

Trừ khi ghi khác, các bảng đo trên 256 clip dev cách đều (cùng tập với sweep), bằng evaluator dùng chung và
cùng nhiễu cho từng câu. Đơn vị là cm. "speed" là tỉ số tốc độ so với GT (lọc thông thấp σ = 2).

### 2.1 Replica, mean pose và v1 trên cùng 256 clip

| Dự đoán | JPE body | JPE hand | PA ua | PA hand | speed body / hand | biên độ / GT |
|---|---|---|---|---|---|---|
| Mean pose tĩnh Phoenix train (độ dài GT) | **4.76** | 7.54 | **4.35** | 1.39 | 0 / 0 | 0 |
| Replica SignNMFlow v2 | 5.11 | **6.84** | 4.53 | **1.35** | 0.66 / 0.78 | 0.79 |
| v1, w = 4.5 (đã báo cáo) | 5.50 | 7.24 | 4.84 | 1.49 | 1.01 / 1.10 | 1.03 |
| v1, w = 1 | 5.30 | 6.95 | 4.66 | 1.41 | 0.86 / 0.96 | – |
| v1, w = 1, τ = 0.3 | 5.19 | 6.87 | 4.57 | 1.36 | 0.72 / 0.82 | – |
| v1 20k, Euler 1 bước (E[z \| text]) | 5.05 | 6.93 | 4.52 | 1.32 | 0.39 / 0.43 | – |
| v1, w = 0 (không text) | 6.36 | 8.44 | 5.69 | 1.65 | 0.90 / 0.93 | – |

Từ bảng này:
- JPE-body chủ yếu đo độ lệch tư thế trung bình. Mọi phương pháp co về trung bình đều được lợi.
- JPE-hand và PA-hand là nơi nội dung câu thực sự được đo.
- Độ dài dự đoán của hai model có sai số tương đương: |Δ log T| là 0.171 (v1) và 0.177 (replica);
  tương quan với GT là 0.87 và 0.85.

### 2.2 Text có được dùng không? Flow loss trong latent, cùng nhiễu, t ∈ {0.05, …, 0.95}

| Checkpoint | Dev: có text / null (gain) | Train: có text / null (gain) | Dev t = 0.05: có text / null |
|---|---|---|---|
| 20k | 1.065 / 1.076 (+0.011) | 0.979 / 1.063 (+0.084) | 1.08 / 1.12 |
| 40k | 1.135 / 0.946 (−0.19) | 0.725 / 0.924 (+0.20) | 1.31 / 1.08 |
| 60k (v1) | 1.314 / 0.943 (**−0.37**) | 0.663 / 0.917 (+0.25) | 1.42 / 1.08 |

Đo trên chính câu train (w = 2.5, 256 clip train):
- 60k: 2.24 / 3.26 · 1.97 / 0.66;
- 20k: 4.08 / 5.76.

Kết luận: **cross-attention học thuộc từng câu.** Đường cong val flow loss có đáy ở 26k
(1.006) rồi tăng lên 1.34 ở 60k, còn train giảm xuống 0.72. 60k cập nhật × 64 mẫu tương đương 541 epoch
của 7,092 câu, trong khi replica chạy 300 epoch.

### 2.3 Các yếu tố khác

| Yếu tố | Đo được | Đóng góp |
|---|---|---|
| Độ dài: head → GT (w = 2.5) | 5.40 / 7.08 → 5.25 / 7.03 | nhỏ (−0.15 body, −0.05 hand) |
| Độ dài: head → độ dài của replica | 5.40 / 7.08 → 5.30 / 6.99 | nhỏ |
| Dừng sớm: 60k → 20k (w = 1.5–2.5) | 5.32–5.40 / 6.99–7.08 → 5.14–5.16 / 6.97–6.98 | vừa, chủ yếu ở body |
| Guidance: w 4.5 → 1–1.5 | 5.50 / 7.24 → 5.30 / 6.95 | vừa; giảm w là đổi DTW lấy tốc độ |
| Nhiệt độ: τ 1 → 0.3–0.6 (w = 1) | 5.30 / 6.95 → 5.19 / 6.87 | vừa (co về mode) |
| Trần reconstruction (G1) | 1.45 / 1.77 · speed 0.90 / 0.96 | không phải nút thắt |
| Gloss probe (dev mAP) | teacher 0.274; latent d = 64 0.196; raw 0.199 | latent mất phần ngữ nghĩa mà JEPA học được; nguyên nhân phụ |
| Sign-JEPA | không collapse (target std 0.97); κ = 29; ρ(λ, C) = 0.95 | tốt |

### 2.4 Khác biệt cấu hình so với replica

| | Replica SignNMFlow v2 | SignRAE v1 |
|---|---|---|
| Không gian | quaternion vật lý S³, dự đoán x̂₁ | latent JEPA d = 64, dự đoán vận tốc |
| Text | mBART **fine-tune** ở lr 2e-4 | mBART **đóng băng** + cross-attention |
| Lượng huấn luyện | 300 epoch × 7k (≈ 8.3k bước × batch 256) | 541 epoch (60k bước × 64) |
| Chọn model | epoch cuối (val có đáy ở epoch 100) | bước cuối (val có đáy ở 26k) |
| Sampler | 5 bước re-plan theo x̂₁, tự co về trung bình | RK4-5, CFG 4.5: mẫu sắc nét, biên độ đầy đủ |

## 3. Nguyên nhân xếp hạng

1. **R1, overfit text → động tác.** Bằng chứng §2.2. Đây là nguyên nhân gốc, quyết định giới hạn trên.
2. **R2, điểm vận hành.** Báo cáo ở w = 4.5, τ = 1 cho mẫu sắc nét mà DTW phạt nặng. Ở cùng mức
   co, v1 gần như ngang replica (§2.1).
3. **R3, không dừng sớm.** Dùng `select: last` với EMA 0.9999 làm mất 0.1–0.2 body.
4. **R4, mBART đóng băng.** SOKE và replica đều fine-tune. Chưa đo riêng; arm này nằm trong v2.
5. **R5, độ dài.** Chỉ ảnh hưởng ≤ 0.15.
6. **R6, latent d = 64.** Gloss probe mất phần ngữ nghĩa. Ảnh hưởng chưa được đo trực tiếp lên DTW (xem §6).

## 4. Thay đổi code (v2)

Mặc định giữ nguyên hành vi v1, nên run v1 tái lập được.

| Khoá | Ý nghĩa |
|---|---|
| `pretrain.py --stage flow --init-ae` | Huấn luyện generator (DiT + duration + mBART) trên union các train split, trong chuẩn hoá latent của union. Không bao giờ đọc split test. |
| `method.init_flow` | Warm-start generator. Kiểm tra cùng latent space, cùng mean/std, cùng kiến trúc. Bắt buộc `latent.recalibrate=false`. |
| `text.finetune`, `text.lr_scale` | Fine-tune mBART với lr = `lr_scale × flow.lr`, trong param group riêng. Khi bật, checkpoint chứa encoder. |
| `flow.token_drop` | Bỏ ngẫu nhiên từng token text khi huấn luyện, luôn giữ ít nhất một token. |
| `sampler.ensemble` | Trung bình K mẫu trong không gian pose. Mẫu số 0 trùng với dự đoán một mẫu. |
| `sweep.py --ensemble --data --lengths gt\|npz --subset train --device` | Công cụ chẩn đoán. Chỉ dùng dev hoặc train. |

Có 4 test mới:
- ensemble bằng trung bình các mẫu thành viên;
- token dropout giữ một tập con khác rỗng;
- `init_flow` khôi phục đúng generator và từ chối chuẩn hoá khác;
- optimizer có nhóm lr phụ.

Smoke test CPU trên dữ liệu thật đã pass: union → fine-tune → reload → sweep zero-shot.

## 5. Kế hoạch v2: `python -m methods.morae_slp.job --plan v2`

Chạy tuần tự, mỗi lúc một tiến trình. Trạng thái nằm ở `outputs/morae_slp/job_v2`.

| # | Stage | Cấu hình | Mục đích |
|---|---|---|---|
| 1 | `flow_union` | union 3 train split (56,429 câu), latent `ae_union_jepa_d64`; DiT-M + xattn; mBART fine-tune (lr 2e-5); `token_drop` 0.1; `cond_drop` 0.1; 100k bước × 64; EMA 0.9999; **chọn theo val union** (768 clip); snapshot 40k / 70k | Text → động tác học từ dữ liệu nhiều gấp 8 lần |
| 2 | `zeroshot_phoenix` | model union trên dev Phoenix, w ∈ {1, 2.5} × K ∈ {1, 8} | Đo lợi ích của union khi chưa fine-tune |
| 3 | `ft_phoenix` | `init_flow`; 20k bước; lr 1e-4, warmup 500; EMA 0.999; `select: flow` (eval mỗi 1k); w = 2 trong `metrics.json` | Thích nghi với Phoenix |
| 4 | `sweep_phoenix` | w ∈ {1, 1.5, 2.5, 3.5} × K ∈ {1, 8} | Chọn hai điểm vận hành |
| 5–8 | `ft_csl` / `sweep_csl`, `ft_how2sign` / `sweep_how2sign` | như trên; 30k và 40k bước | Đủ ba bộ benchmark của paper |
| 9–10 | `ctrl_phoenix_only` / `sweep_ctrl` | công thức v2 nhưng **không** pretrain union (30k, `select: flow`) | Tách đóng góp của union khỏi fine-tune mBART và dừng sớm |

Ước lượng thời gian (GPU dùng chung, tính từ tốc độ đo được):

| Stage | Thời gian ước lượng |
|---|---|
| Stage 1 | ≈ 7–9 giờ |
| Mỗi lượt fine-tune | 1.2 / 1.8 / 2.5 giờ |
| Mỗi sweep | ≈ 0.5 giờ |
| Đối chứng | ≈ 2 giờ |
| **Tổng** | **≈ 17–20 giờ** |

Relikflow tự khởi động lại khi v2 xong (`relikflow-after-morae-v2.service`).

### Quy tắc quyết định

- **G-v2a, union có giúp không.** Ở stage 2, gain text trên dev phải dương rõ rệt, và JPE-hand ở w = 1, K = 1
  phải ≤ 6.87 (v1 tốt nhất). Nếu union không giúp, R1 không đến từ lượng dữ liệu. Khi đó chuyển sang §6 (B).
- **G-v2b, so với replica ở cùng tốc độ.** Mốc là replica 6.84 / 1.35 ở tốc độ 0.66 / 0.78.
  Nếu ở điểm ước lượng điểm (K = 8) có tốc độ ≥ 0.66 mà hand < 6.84, ta thắng mà không chỉ nhờ co.
  Dùng `frontier_effect` trong `slp.evaluation.metrics` để so tại cùng tốc độ.
- **Mục tiêu paper (test).**

  | Dataset | Paper SignNMFlow (test) |
  |---|---|
  | Phoenix | 4.37 / 5.57 · 3.85 / 1.07 |
  | CSL | 5.73 / 7.43 · 5.10 / 1.30 |
  | How2Sign | 6.76 / 9.72 · 6.01 / 2.07 |

  Replica của ta trên Phoenix test đạt 4.93 / 6.54 (dev 5.17 / 6.98), nên dev→test chênh khoảng −0.24 / −0.44.
  Để chạm số paper, dev Phoenix cần đạt khoảng ≤ 4.6 / 6.0. Đây là mục tiêu tham vọng; ngay replica cũng chưa đạt.

### Báo cáo

- Mỗi dataset báo cáo hai dòng: **tự nhiên** (K = 1, chọn w trên dev) và **ước lượng điểm** (K = 8).
- Luôn kèm tỉ số tốc độ / gia tốc và số clip.
- Không chọn sampler trên test. Chỉ chạy test khi người dùng yêu cầu, và log vào `results/test_access_log.jsonl`.

## 6. Bước tiếp theo nếu v2 chưa đủ

- **A. Latent giàu ngữ nghĩa hơn.** Thử d = 128 hoặc KL nhỏ hơn. Arm `latent.kind=raw` (DiT trực tiếp trên
  pose 133-D) là đối chứng quan trọng cho giả thuyết MoRAE trong SLP.
- **B. Regularise thêm nhánh text.** Thử DiT nhỏ hơn (8×512, giống replica), `cond_drop` 0.2,
  `token_drop` 0.2, weight decay 0.01.
- **C. Gloss phụ trợ** (Phoenix, CSL có gloss train). Dùng một head CTC từ token mBART hoặc từ latent để
  ép nhánh text mang nội dung ký hiệu. Inference vẫn chỉ dùng text.
- **D. Ước lượng điểm tốt hơn.** Thử K = 16 hoặc 32, trung bình trong latent thay vì pose, hoặc
  dùng Euler 1 bước sau khi dừng sớm (v1 20k cho 5.05 / 6.93 ở tốc độ 0.39).

---

## 8. Kết quả thực nghiệm v2 (dev, cập nhật 2026-11-26) và bước tiếp theo

Số dev, seed 0, bộ chấm SOKE dùng chung. "Một mẫu" là một chuỗi cho mỗi câu (protocol của các paper).

### 8.1 Điều đã học được

| # | Phát hiện | Bằng chứng |
|---|---|---|
| 1 | Pretrain flow trên union **không** cải thiện DTW một mẫu, dù val flow loss tốt hơn | Phoenix: union→fine-tune 5.53 / 7.19, flow chỉ-Phoenix 5.32 / 6.97 (w = 2). Val flow 0.921 so với 0.973 |
| 2 | Công thức v2 (mBART fine-tune lr 0.1×, bỏ 10% token, chọn checkpoint theo val) giúp Phoenix | 5.55 / 7.35 (v1) → 5.32 / 6.97 (w = 2). Chưa ablate từng thành phần, mới chỉ so cả gói với v1 |
| 3 | Hạ nhiệt độ nhiễu τ đưa một mẫu về ngang replica | Phoenix, w = 1.5, τ = 0.7: 5.196 / 6.941 · 4.576 / 1.367 (519 clip), tốc độ 0.66 / 0.68. Tại cùng tốc độ: body +0.02 (n.s.), hand +0.12, PA-body −0.02 (n.s.) |
| 4 | Guidance thấp tốt hơn cao ở mọi bộ | w = 1–1.5 tốt hơn w = 3.5 cho cả một mẫu và K = 8 |
| 5 | Trung bình K = 8 mẫu thắng đường biên của replica ở Phoenix với w = 1.5, không thắng với w = 3.5 | body −0.18, hand −0.29 (w = 1.5) so với +0.13 / −0.08 (w = 3.5) |
| 6 | **CSL kém replica ở cả hai công thức** | union→fine-tune 6.28 / 7.99; chỉ-CSL 6.20 / 8.02 (replica 5.72 / 7.38). Sweep τ = 0.7: 6.00 / 7.85 |
| 7 | How2Sign kém replica ở bản union | 7.90 / 10.97 (replica 7.50 / 10.85); bản chỉ-How2Sign chưa xong |

Nhãn "chỉ Phoenix / CSL / How2Sign" chỉ nói về flow và text. Sign-JEPA và latent d = 64 đều huấn luyện trên chuyển động của
cả ba dataset (không có text).

### 8.2 Cây quyết định cho CSL và How2Sign (chờ `diag_recon_*` và `sweep_gtlen_*`)

| Kết quả chẩn đoán | Hướng tiếp theo |
|---|---|
| Trần tái tạo của latent ở CSL hoặc How2Sign cao hơn rõ rệt so với Phoenix (1.45 / 1.77) | Nút thắt là dung lượng latent: huấn luyện lại AE với d = 128 trên union (khoảng 1.5 giờ, teacher đóng băng), rồi chạy lại flow |
| Dùng độ dài thật cải thiện nhiều (> 0.2 JPE) | Cải thiện head độ dài: tăng `duration_weight`, thêm đặc trưng độ dài câu |
| Cả hai đều nhỏ | Nút thắt nằm ở nhánh text: thử `text.lr_scale` 0.3 hoặc 1.0 (replica fine-tune mBART ở lr đầy đủ), hoặc đổi cách che token |

Mỗi thí nghiệm chạy một mình (một tiến trình), chấm trên dev bằng cùng bộ sweep (w, τ), và chỉ khi tốt hơn mới chuyển sang so cùng tốc độ.

### 8.3 Đánh giá trên test (chưa chạy)

Chỉ chạy khi người dùng yêu cầu rõ, một lần, với cấu hình đã chọn trên dev (Phoenix: công thức chỉ-Phoenix, w = 1.5, τ = 0.7).
`run.py --split test` huấn luyện lại rồi chấm test: Phoenix khoảng 1.6 giờ, CSL khoảng 1.7 giờ, How2Sign khoảng 2.5 giờ. Cần git tree sạch,
hoặc `--allow-dirty` (số không dùng làm số báo cáo). Báo cáo phải ghi: SSL trên union chuyển động, một seed, cấu hình chọn trên dev,
và tốc độ chuyển động.

# 08 — Bản thiết kế cài đặt (v2, theo quy ước của repo)

Tài liệu này mô tả *cấu trúc code cần viết*. Code chưa được viết.

**Quy ước bắt buộc** (CLAUDE.md, `methods/README.md`):

- Tạo method bằng `python scripts/new_method.py signhiflow`. Code riêng của method nằm trong `methods/signhiflow/`.
- **Không import từ method khác.** Phần dùng chung đưa vào `src/slp/` và phải có test.
- Method chỉ cài `fit(train, val)` và `predict(inputs)`. Method **không** tự tính metric báo cáo và
  **không** chạm vào test split.
- Metric mới chỉ thêm ở `src/slp/evaluation/metrics.py`, metric nào cũng có unit test.
- Sau mỗi thay đổi ở `src/slp/` hoặc ở method: chạy `python -m pytest -q`.

---

## 1. Thay đổi ở `src/slp/` (dùng chung, có test)

| File | Nội dung | Test bắt buộc |
|---|---|---|
| `src/slp/geometry/rotations.py` (mới) | `aa_to_6d`, `sixd_to_matrix` (Gram–Schmidt), `matrix_to_aa`; xử lý `hands_mean` (`flat_hand_mean=False`, cộng trong axis-angle) | round-trip GT 133-D → 6D → 133-D: sai lệch FK < 1e-4 m trên 100 clip dev |
| `src/slp/geometry/sparse_fk.py` (mới) | FK khả vi trên torch, fp32, **tái tạo chính xác** `smplx_fk`:<br>• shape SOKE cố định; root và chân đặt 0;<br>• khớp thân từ FK chain;<br>• landmark 55–59 và khớp tay lấy từ **mesh đã pose** = LBS + **pose blendshapes**, chỉ trên tập vertex có trọng số khác 0 trong các hàng hand-regressor cộng vertex đầu ngón và landmark (khoảng 1.5k vertex) | khớp `default_fk()` tới < 1e-5 m trên pose ngẫu nhiên và pose GT; kiểm tra gradient (gradcheck trên vài frame) |
| `src/slp/geometry/spectral.py` (mới) | DCT-II trực chuẩn có cache theo độ dài; $\tilde D$, $\tilde U$, $\Pi$ cho batch có mask độ dài | P1–P5 (05-T1); hiệp phương sai nhiễu đo bằng Monte Carlo |
| `src/slp/evaluation/metrics.py` (bổ sung) | `soke_ndtw_*` (chia cho độ dài path); `duration_rel_error`; DTW theo band (tham số cutoff); đăng ký vào registry | trường hợp cố định (chuỗi trùng nhau = 0, dịch thời gian); so với `dtw_jpe_norm` |
| `src/slp/evaluation/semantic.py` (mới) | wrapper SiCLIP (MaDiS) gồm score, R@k và embedding cho FD; DTWpp (tuỳ chọn) | input giả; GT so với GT bị xáo |
| `scripts/train_backtranslator.py` (sửa) | back-translator mới: khớp FK → encoder → decoder mBART, CE + CTC gloss phụ trợ; báo cáo trần GT | smoke |
| `src/slp/evaluation/metrics.py` (bổ sung, đợt 2) | speed/accel ratio **riêng tay trái, tay phải, mặt** và **trong khung cổ tay**; median theo chuỗi; tỉ lệ chuỗi có ratio < 0.5 | chuỗi GT cho ratio = 1; chuỗi tĩnh cho 0 |
| `scripts/matched_kinematics.py` (mới) | đọc predictions của các phương pháp ($\hat\mu$, $G$ hoặc mẫu $S$); slerp theo từng khớp; grid search $(\lambda_B,\lambda_H)\in[0,1]^2$ trên **val**; quy tắc không có nghiệm; tuỳ chọn DTW-warp; xuất predictions tại điểm vận hành | test trên dữ liệu giả (có trường hợp speed không đơn điệu) |
| `scripts/gelbrich_headroom.py` (mới, E0.9) | $\Sigma_\mu$, $\Sigma_n$ (Ledoit–Wolf) theo band DCT × articulator; $\eta$ (công thức đóng cho $W_2$ giữa hai Gaussian) | so với trường hợp Gaussian 1-D đã biết $(\sqrt{a+b}-\sqrt a)^2/b$ |
| `scripts/split_seen_unseen.py` (mới) | gắn nhãn câu dev/test là text đã thấy hay chưa thấy trong train (khớp chính xác và gần khớp) | — |

---

## 2. `methods/signhiflow/`

```
methods/signhiflow/
├── README.md        # status; giả thuyết; search space; kết quả
├── config.yaml      # §3
├── method.py        # Method(BaseMethod): fit() chạy Phase A (μ toàn phần + 2 cross-fit), cache μ out-of-fold, Phase B (T-net);
│                    #   predict() = 04 §8. Λ được nạp từ config (do scripts/matched_kinematics.py chọn trên val)
├── data.py          # 133-D → 6D chuẩn hoá theo kênh (thống kê train); part B/L/R/F; bucket độ dài; mask; chia fold (gộp câu trùng text)
├── text.py          # mBART (lịch freeze, LLRD), ConvNeXt-1D, language emb, duration head (Laplace)
├── mu_net.py        # MMDiT với query token tất định (time patch 2), PA-RoPE, output 6D; [Track R] memory
├── t_net.py         # DiT không nhận text; adaLN theo (stage, r_k, u, σ_k); RoPE thời gian/articulator; output được chiếu band
├── bands.py         # interpolant theo stage (04 §5); x→v; clamp band; rollout stage 1 (no-grad) làm anchor stage 2; nhiễu Πη ở stage 3; sampler Euler; đo ρ
├── losses.py        # L_x theo part; PES (gọi slp.geometry.sparse_fk; lấy mẫu 48 frame); Laplace; soft-DTW (tuỳ chọn)
├── retrieval.py     # Track R (tuỳ chọn): chỉ mục embedding câu train, top-k, memory token; test chống rò rỉ
└── training.py      # vòng lặp Phase A/B, EMA, lưu snapshot (không tự chấm metric), cache μ out-of-fold
```

**Code có thể sao chép rồi sửa** (không import) từ bản port MotionHiFlow của chính người dùng
(`methods/motionhiflow_slp`): `dit.py` (DualBlock, SharedBlock, RotaryAttention) và `text.py` (wrapper mBART).

**Chọn snapshot và $\Lambda$.** Việc này làm ở `configs/experiments/signhiflow/*.sh`: script gọi
`scripts/run.py` hoặc `scripts/score_predictions.py` trên **val**, cộng `scripts/matched_kinematics.py`.
Nhờ vậy method không tự tính metric báo cáo.

**Cạm bẫy phải có test:**

1. DCT theo độ dài: padding không được rò vào DCT. Dùng mask cộng ma trận riêng cho từng độ dài.
2. Ràng buộc band: output của T-net ở stage ≥ 2 phải được chiếu lên $(I-\Pi)$. Có test xác nhận phần
   dải thấp giữ nguyên tới 1e-6.
3. $\hat v=(\hat h-h_u)/(1-u)$: không đánh giá tại $u=1$. Kẹp $1-u\ge10^{-3}$.
4. `hands_mean`: cộng trước khi đổi sang 6D, trừ sau khi đổi ngược. Test round-trip.
5. FK chạy ở fp32. PES chỉ tính trên frame đã lấy mẫu.
6. Cross-fitting: các câu trùng text phải nằm cùng một fold. Có test xác nhận không có câu nào xuất hiện
   ở cả hai fold.
7. T-net **không** nhận text hay language. Có test: đổi text thì output của T-net không đổi, với cùng
   μ và cùng nhiễu.
8. Rò rỉ dữ liệu: chỉ mục, thống kê và μ cache chỉ được dùng tên clip của train.

---

## 3. `config.yaml` (khung, giá trị mặc định theo 04)

```yaml
data: {name: phoenix_smplx}
eval:
  metrics: [soke_dtw_jpe_body, soke_dtw_jpe_hand, soke_dtw_pa_jpe_body, soke_dtw_pa_jpe_body_ua,
            soke_dtw_pa_jpe_hand, soke_ndtw_jpe_body, soke_ndtw_jpe_hand, soke_ndtw_pa_jpe_hand,
            soke_speed_ratio_hand, soke_speed_ratio_body, soke_accel_ratio_hand]
  primary_metric: soke_ndtw_jpe_hand
  higher_is_better: false
method:
  device: auto
  precision: bf16
  representation: 6d                    # 6d | aa | quat (A3)
  part_weights: [1.0, 1.5, 1.5, 0.5]
  text:
    model: third_party/SOKE/deps/mbart-h2s-csl-phoenix
    finetune: true
    lr: 2.0e-5
    layer_decay: 0.9
    freeze_updates: 5000
    refine_convnext: 2
    pa_rope: true
  duration: {kind: laplace, weight: 0.1, decode: median}
  mu_net:
    width: 512
    heads: 8
    double_blocks: 4
    single_blocks: 10
    time_patch: 2
    dropout: 0.1
    losses: {pes: 0.05, sdtw: 0.0}
    updates: 40000
    crossfit: {folds: 2, budget_frac: 0.6}
  t_net:
    width: 512
    heads: 8
    blocks: 12
    stages: 3                           # 1 = PMRF phẳng (A5)
    ratios: [0.25, 0.5, 1.0]
    per_part_ratios: null               # vd {hands: [0.5, 0.5, 1.0]} nếu E0.3 ủng hộ
    text_conditioning: false            # true = A6
    anchor_source: crossfit             # crossfit | infold | pseudo (A19)
    sigma_train: {s1: [0.05, 0.4], s23: [0.2, 1.0]}
    stage2_anchor: rollout              # rollout (output thật của stage 1, no-grad 4 NFE) | gt_aug
    stage3_anchor: gt_aug               # gt_aug (Π y + 0.03 Π η) | cached_outputs (A14)
    anchor_noise: 0.03
    test_mu_input: random_fold          # random_fold | full | fold_mean (A24)
    pilot: {stage1_only_updates: 20000, rho_gate: 0.8}      # E2-pilot / G1.5
    early_stop: {metric: dev_rho_and_xloss, patience_snapshots: 3}
    stage_probs: [0.3, 0.35, 0.35]
    t_sampling: {kind: logit_normal, mean: 0.0, std: 1.0}
    pes: {weight: 0.05, min_u: 0.7}
    updates: 150000
    finetune_updates: 30000
    pool_datasets: true
  pes: {body: 1.0, hand: 1.0, shape: 1.0, loc: 0.5, vel: 0.5, frames: 48, exact_fk: true}   # exact_fk false = A21
  train:
    batch_size: 64
    microbatch: 16
    lr: 2.0e-4
    warmup: 5000
    min_lr_ratio: 0.1
    weight_decay: 0.01
    grad_clip: 1.0
    ema: 0.9999
    snapshot_every: 5000
  sampler:
    sigma: [0.2, 0.5, 0.5]
    steps: [4, 3, 3]
    lambda: {body: 0.5, hands: 0.5}     # λ_F = λ_B, λ_L = λ_R; ghi đè bằng giá trị matched-kinematics chọn trên val
    mix: slerp                          # slerp | linear6d (A23)
    dtw_warp_G: false                   # A23
  retrieval: {enabled: false, k: 4, frames: 64, drop: 0.3}   # Track R
  mbr: {enabled: false, n: 8}                                # chỉ dùng khi mọi baseline có cùng N
```

---

## 4. Các mốc cài đặt (khớp với 06)

| Mốc | Nội dung | Điều kiện hoàn thành | Phục vụ |
|---|---|---|---|
| M0 | Metric nDTW và DTW theo band, wrapper SiCLIP/FD, `matched_kinematics.py`, các script chẩn đoán E0.1/E0.2/E0.6/E0.7/E0.8 | test pass; có bảng E0 | E0 |
| M1 | `rotations.py`, `sparse_fk.py` (chính xác), `spectral.py` | test khớp FK < 1e-5 m; P1–P5 | tất cả |
| M2 | μ-net: data, text, mu_net, losses (L_x, PES, Laplace) | smoke test 16 clip; conformance test | E1 |
| M2b | `gelbrich_headroom.py`; cross-fitting và cache μ; T-net **chỉ stage 1**, có đo ρ | E0.9 xong; pilot chạy được | G1.5 |
| M3 | T-net đầy đủ (stage 2 dùng rollout, stage 3), bands; chỉ làm khi G1.5 đạt | test band-clamp; test T-net không nhận text; test chuyển tiếp dùng oracle velocity | E2 |
| M4 | Baseline quy thuộc: CorrDiff-style (μ + flow có text trên phần dư); MotionHiFlow-SLP chuyển x-pred; shrink-to-mean cho SignNMFlow-reimpl | chạy được qua `scripts/run.py` | E2 |
| M5 | Pretrain đa ngôn ngữ; T-net gộp dữ liệu; Track R | log cân bằng dataset; test rò rỉ | E5, E6 |

**Smoke test** (không phản ánh chất lượng):

`python scripts/run.py --method signhiflow --name signhiflow_smoke --seeds 0 method.debug_limit=16 method.mu_net.updates=2 method.t_net.updates=2 method.mu_net.crossfit.folds=0`

Đặt `text.freeze_updates` lớn hơn số update trong smoke test. Repo đã ghi nhận đặc trưng text bị sụp
khi fine-tune mBART trong run ngắn.

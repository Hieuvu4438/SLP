---
status: exploring
---

# SignMFlow với giám sát động học

Package thực thi các ablation trong [proposal](../../docs/proposal/03_thiet_ke_thuc_nghiem.md). Các nhánh dùng định dạng 133D và evaluator chung của SLP. Kiến trúc mặc định và biến thể S³ naturalness được chuyển từ bản replica SignMFlow tại `c859c93`. Package dùng checkpoint tương thích và không import từ method khác.

## Thứ tự và đối chứng

| Mục | Preset | Thiết kế |
|---|---|---|
| E0 | `b0` | Naturalness gốc, endpoint loss quaternion trên lift, 5 NFE. CLI chẩn đoán xuất GT, pose tĩnh, sai câu và đảo thứ tự chuyển động. |
| E1 | `e1_uniform`, `e1_late`, `e1_anchor_expr`, `e1_one`, `e1_ten` | Cùng checkpoint. Thử lưới t, expression neo ở nguồn, số bước. Có thể chạy text-swap, noise-swap, oracle length và đo latency trên dev. |
| E2 | `e2_continue`, `e2_position`, `e2_palm`, `e2_velocity`, `e2_jaw_expression` | Warm-start cùng trọng số, optimizer mới. Chạy vị trí, hướng lòng bàn tay, vận tốc, jaw/expr theo các nhánh riêng. `e2_position_scratch` và `e2_velocity_scratch` huấn luyện từ đầu với lịch của B0. |
| E3 | `e3_s3`, `e3_so3`, `e3_chordal`, `e3_invariant_input`, `e3_projected_6d`, `e3_euclidean_6d` | So S³ và SO(3) với cùng policy quaternion canonical, rồi đổi loss, đặc trưng đọc hoặc đường 6D. So `e3_projected_6d` với `e3_chordal` để giữ nguồn là quaternion. `e3_euclidean_6d` tái hiện nguồn Gaussian 6D. |
| E4 | `e3_chordal`, `e4_full`, `e4_diagonal`, `e4_joint_block`, `e4_density` | Giữ backbone, input, loss và prior chung. So geodesic với metric đầy đủ, đường chéo theo trục, đường chéo theo block joint và density-only. Có `e3_chordal_with_loss`, `e4_diagonal_with_loss`, `e4_full_with_loss` cho bảng chéo với E2. |
| E5 | `e5_ar`, `e5_length`, `e5_coarse`, `e5_matched_ot` | AR(1) giữ marginal từng frame. Length head dự đoán quantile có đặc trưng token count. Coarse plan sinh từ text, tái dùng ở mọi bước denoise. OT chỉ hoán vị source trong nhóm có cùng ngôn ngữ, text và thời lượng. |

E4 dùng tangent theo angular velocity của joint trong hệ tọa độ cục bộ. Cả ba metric FK có phần cơ sở `fk_base * ||v||²` và cùng trọng số cho từng nhóm FK. Metric đầy đủ dùng JVP. Đối chứng đường chéo dùng chính Jacobian tại trạng thái đó, giữ chi phí trên từng trục. Biến thể joint-block giữ tương tác giữa ba trục của mỗi joint. Hai đối chứng bỏ tương tác giữa các joint. Density-only có bandwidth tính bằng radian và clamp quanh mức median mass theo các cặp nguồn–đích chỉ từ train. [Kinematic energy](energy.py)

Giám sát E2 dùng các khớp thân, cổ tay theo thân, đầu ngón theo cổ tay và hai trục lòng bàn tay. Adapter dùng pose correctives và linear blend skinning cho đúng 10 vertex đầu ngón của evaluator với shape cố định và expression trung tính. Mục body dùng khớp skeleton. Loss vận tốc lấy chênh lệch có hướng và chia cho khoảng thời gian thật giữa các frame theo loader. Các frame trùng do resampling và các cặp băng qua padding được loại khỏi loss vận tốc. Hệ số preset là giá trị pilot. Bật `method.generator.probe_gradients=true` để log norm của các gradient đã nhân hệ số vào motion head ở microbatch đầu mỗi epoch. [FK](kinematics.py) · [Loss](flow.py) · [Cache và timestamps](data.py)

B0 mặc định lấy nhiễu theo ID clip để so sánh cùng nguồn dù đổi batch order. Để kiểm replay của checkpoint cũ với luồng RNG cũ, chọn `method.sampler.noise=legacy` và giữ batch size, length và seed như run gốc. S³ xử lý riêng antipode và raw quaternion bằng 0. Trainer mới giữ batch cuối cùng và tích lũy chính xác phần dư cuối epoch, nên baseline huấn luyện lại cần chạy bằng chính package này khi so với ứng viên train từ đầu.

Các nhánh E5 mở rộng phụ thuộc pilot. OT báo `train_ot_fraction` và `correction_ot_fraction`. Khi nhóm trùng text và chiều dài trong minibatch hiếm, phép ghép OT tác động lên ít clip. Coarse plan là mạng dự đoán low-rate features từ text, có một lượt tính thêm mỗi batch, ghi riêng khi benchmark. Semantic critic, prototype theo cửa sổ thời gian và source prior học có điều kiện cần đặc tả riêng sau khi có chẩn đoán, nên nằm ngoài gói hiện tại.

## Lệnh chạy

Chạy từ root repo. Cần PyTorch, transformers, SentencePiece, SMPL-X và assets theo config. SciPy chỉ cần cho OT. Preset mặc định **in lệnh**. Thêm `--run` để chạy qua `scripts/run.py`, luôn trên dev. Một process có thể dùng gradient accumulation với `effective_batch`. Giảm `correction.frame_chunk` cho metric Jacobian hoặc `generator.micro_batch` khi thiếu bộ nhớ. Các run ghép cặp cần cùng dataset, seed, budget và quy ước sampling.

```bash
baseline=outputs/runs/signmflow_v2_main_csl_smplx/signmflow/seed0_20261116-185337-969580/checkpoints

# E1, dùng đúng checkpoint cho mọi lịch t
python -m methods.signmflow_kinematic.experiments e1_uniform \
  --checkpoint-dir "$baseline" --legacy-signmflow
python -m methods.signmflow_kinematic.experiments e1_late \
  --checkpoint-dir "$baseline" --legacy-signmflow

# E2, chạy các nhánh với cùng checkpoint và lịch fine-tune
for preset in e2_continue e2_position e2_palm e2_velocity; do
  python -m methods.signmflow_kinematic.experiments "$preset" \
    --checkpoint-dir "$baseline" --legacy-signmflow \
    --set method.generator.probe_gradients=true
done

# E3/E4, ví dụ chuẩn bị cặp pilot cùng số epoch
for preset in e3_chordal e4_diagonal e4_full; do
  python -m methods.signmflow_kinematic.experiments "$preset" \
    --set method.generator.epochs=40
done
```

Chọn `--data phoenix_smplx` hoặc `--data how2sign_smplx_soke` để xác nhận ngoài CSL. Dùng `--seeds 0,1,2` cho cấu hình scratch. Với E1/E2, chạy từng seed cùng checkpoint tương ứng. Có thể đổi `method.sampler.seed` độc lập nếu cần lặp nhiễu. Các mốc paper nằm trong [audit benchmark](../../docs/proposal/01_phan_bien_va_code.md).

`method.init.generator` dùng trọng số đã train và bỏ các bước train. `method.init.warm_start` khởi tạo generator và optimizer mới. `method.generator.resume_from` khôi phục model, optimizer, scheduler và RNG tại cuối epoch. Khi resume, giữ nguyên lịch epoch, dataset, seed và correction checkpoint đã dùng. `checkpoints/experiment.json` ghi SHA-256 các checkpoint. Generator mới ký cả SHA-256 của correction để phát hiện ghép nhầm đường vận chuyển. Checkpoint SignMFlow v2 không có chữ ký format mới, cần bật `--legacy-signmflow` và chọn đúng run/seed. Callback lấy prototype và chuẩn hóa chỉ áp dụng với dữ liệu train. `method.debug_limit` giới hạn train và val loss nội bộ, còn bước chấm điểm của runner vẫn dùng toàn bộ dev.

## Chẩn đoán độc lập trên dev

```bash
python -m methods.signmflow_kinematic.diagnostics controls "$run" "$diagnostic" --inference
python -m methods.signmflow_kinematic.diagnostics score-controls \
  "$run" "$diagnostic" "$diagnostic/pose_scores.json"
python -m methods.signmflow_kinematic.diagnostics latency \
  "$run" "$diagnostic/latency.json" --limit 100 --warmup 5
python -m methods.signmflow_kinematic.diagnostics compare \
  --pair "$baseline_run_seed0" "$candidate_run_seed0" \
  --pair "$baseline_run_seed1" "$candidate_run_seed1" \
  --pair "$baseline_run_seed2" "$candidate_run_seed2" \
  --output "$diagnostic/paired.json"
```

`controls --inference` dùng predicted length ngay cả khi đổi text hoặc noise để cô lập điều kiện. Oracle length vào archive riêng, dành cho dev. `compare` tính bootstrap cặp trên clip hoặc cụm theo `--clusters IDs.json`, báo khoảng tin cậy theo clip với các training seed cố định và độ lệch giữa seed riêng. Latency bao gồm text encoder, length, lấy mẫu và decode 133D, loại thời gian tải checkpoint và FK evaluator.

Back-translation dùng checkpoint mBART nội bộ trên đúng tập ID và preprocessing của run dev. Adapter ghi SHA-256, SacreBLEU signature và từng dự đoán text. So sánh bootstrap tính lại corpus BLEU từ số n-gram của các clip đã lấy mẫu. Chạy bằng môi trường có `sacrebleu`, ví dụ venv scorer đang có trong workspace.

```bash
bt_python=third_party/MaDiS/.venv/bin/python
"$bt_python" -m methods.signmflow_kinematic.backtranslation score "$baseline_run_seed0" \
  --checkpoint outputs/bt/csl_mbart/best.pt --output "$diagnostic/bt_seed0.json" \
  --prediction wrong "$diagnostic/wrong_motion.npz" \
  --prediction reordered "$diagnostic/reordered_motion.npz" \
  --prediction static "$diagnostic/static.npz" \
  --prediction baseline "$baseline_run_seed0/predictions.npz" \
  --prediction candidate "$candidate_run_seed0/predictions.npz"

"$bt_python" -m methods.signmflow_kinematic.backtranslation compare \
  "$diagnostic/bt_seed0.json" "$diagnostic/bt_seed1.json" "$diagnostic/bt_seed2.json" \
  --output "$diagnostic/bt_paired.json"
```

GT trên dev được chấm tự động. Kiểm độ nhạy với wrong/reordered/static motion trước khi kết luận về ngữ nghĩa. Đây là scorer nội bộ. So sánh số BLEU trong paper cần evaluator và protocol gốc.

## Kiểm chứng

Unit tests kiểm quotient/chordal, endpoint, tangent, mixed gradients, diagonal control, skinning parity với mesh SMPL-X thật, mask và normalization khi chia microbatch, AR(1), OT, coarse plan và quantile. Smoke test với clip CSL-Daily thật chạy qua entry point SLP, train correction và generator, chấm điểm dev bằng evaluator chung, rồi nạp lại checkpoint. Checkpoint SignMFlow v2 thật đã được kiểm ở cả E1 và E2 warm-start. Resume giữa các epoch được đối chiếu với train liền mạch có phần dư gradient accumulation. Các smoke test dùng CPU và ít clip để kiểm luồng thực thi. Chưa có số liệu quality của các run huấn luyện đầy đủ.

# Khôi phục pilot temporal-plan ngày 07/11/2026

## Material Passport

- Nguồn: chỉ đạo người dùng bỏ toàn bộ giới hạn thời gian, hoàn tất vòng thực nghiệm đã setup.
- Phạm vi: PHOENIX-2014T, seed 0, hai variants MAP và sampled-q của temporal-plan; không đổi kiến trúc/loss đang dùng.
- Trạng thái: đã kiểm tra phần mềm; kết quả chất lượng cuối chưa có.
- Runtime: `outputs/phoenix_pilot72_20261106/plan.json`; tên cũ có 72 chỉ còn là đường dẫn lịch sử.

## Hủy các giới hạn thời gian

`hours=null`, mọi `jobs[].timeout_hours=null`, `gpu_policy.probe_timeout_hours=null`, không truyền `stage_max_seconds`, systemd `RuntimeMaxSec=infinity`. Worker không nhận deadline env khi job không có hạn. Không cắt job sau 3/11/17 giờ. Việc chờ VRAM, probe, train, sinh mẫu và đánh giá được phép hoàn tất theo công việc. Các timeout ngắn của lệnh điều khiển/đọc telemetry và grace period khi người dùng chủ động dừng service không phải runtime budget của training.

Bỏ cap output 18 GiB; vẫn kiểm tra free disk và bộ nhớ thực tế để không làm đầy server. Giữ cơ chế đo VRAM từng job, không có threshold 40 GiB/utilization. Pilot vẫn tối đa một dispatcher + một worker.

## Phần giữ lại và công việc còn lại

- Giữ nguyên prepare, preflight, common codec/segment và profiling đã hoàn tất.
- Giữ planner MAP đã train 1.284 updates và generator MAP raw-last checkpoint tại 2.050 updates.
- Mỗi nhánh hoàn tất planner 1.284, generator 2.645, exposure 161 updates. Số update chốt từ trước được giữ nguyên; không suy lại từ tốc độ GPU đang bận.
- MAP chỉ còn 595 generator updates và 161 exposure updates.
- Q planner cũ bị dừng ở 680 updates và không có optimizer/RNG snapshot; lưu nguyên thư mục/log trong archive, không giả vờ resume chính xác. Q bắt đầu lại từ checkpoint common để giữ cùng initialization và quy tắc train planner với MAP.
- Vì checkpoint generator MAP 2.050 chỉ có weights, cả hai variants cùng có **một reset optimizer/RNG/data order tại generator update 2.050**. MAP warm-start từ weights đó, Q train tới 2.050 rồi reset theo cùng quy tắc. Hai bên cùng chọn best validation ở pha sau reset, sau đó train exposure. Không coi warm-start legacy là exact optimizer resume.
- Evaluator cũ lỗi nonfinite gradient chưa có checkpoint. Lượt mới dùng target 35.460 successful updates (10 lượt train với batch 2 trên 7.092 mẫu), lr 3e-5, validation real dev và chọn best CE. Không dùng thời gian để dừng.
- Sau training: sinh đủ 519 dev mỗi nhánh → evaluate → report. Một seed không xác nhận SOTA/generalization.

## Sửa evaluator và bảo toàn tiến độ

Gradient norm được cộng ở float64 để tránh overflow square-sum float32; gradient thực sự NaN/Inf được từ chối trước optimizer step, ghi rõ sample IDs và số lần skip. Không tắt kiểm tra nonfinite hoặc đưa NaN vào weights. Nếu liên tiếp mười batch lỗi số học, giữ checkpoint và báo lỗi thực sự cần xử lý; không tiếp tục với weights hỏng.

Method lưu model, AdamW state, RNG torch/CUDA/Python/NumPy, epoch/data cursor và checkpoint validation tốt nhất mỗi 50 updates. Evaluator lưu state/optimizer/RNG/cursor mỗi 250 updates. Recovery dùng hai slots và manifest checksum commit nguyên tử; file chưa commit không phá checkpoint trước. CPU tests xác nhận resume có cùng weights với chạy liên tục và legacy weights restart khớp quy tắc reset chung.

CUDA OOM trong job mới sẽ nhả process/context, chờ thêm VRAM, rồi nạp checkpoint optimizer/RNG/cursor đã commit để chạy tiếp. Không khởi tạo lại từ đầu một lượt train đã có recovery. Thời gian chờ không làm mất ngân sách update. Biến động tài nguyên của tài khoản khác vẫn có thể khiến phải chờ; không dừng tiến trình của họ.

## Kiểm tra và provenance

- Full suite: **129 passed, 2 skipped**, 17,89 giây, CUDA bị ẩn; Ruff trên các file sửa pass.
- Tests mới: optimizer/noise/data-cursor recovery, matched reset equivalence, checkpoint atomicity/checksum, evaluator state/RNG, float64 gradient clipping, no job deadlines, OOM-wait/recovery.
- Giữ source/plan/state/log/checkpoint cũ và recovery protocol trong `control_changes/20261107T123401+0700/`.
- Snapshot mới có manifest hash. Code expression loss hiện tại được giữ nguyên cho cả hai variants; không đổi objective giữa lúc recovery.

Kết quả runtime cập nhật trong `queue_state.json`, `latest_control_change.json`, `evaluator_training.jsonl`, và từng thư mục method `resume/current.json`. Để dừng chủ động: `systemctl --user stop slp-phoenix-pilot72-20261106.service`.

Runtime GPU đã kiểm tra restart service: evaluator nạp lại 2.500 successful updates, optimizer 113 parameter states, RNG và data cursor, rồi tiếp tục tăng update; job deadline=null, worker không có env deadline, chỉ hai tiến trình pilot. Xem resumed_checkpoint trong evaluator_training.jsonl.

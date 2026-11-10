# Kết quả pilot PHOENIX-2014T — 07/11/2026

## Material Passport

- Artifact được kiểm tra: queue_state, final checkpoints/ready metadata, 1.038 NPY predictions, metrics và evaluator calibration.
- Hoàn tất: **21:12:35 ngày 07/11/2026 Asia/Ho_Chi_Minh**; 11/11 jobs completed, exit code 0; dispatcher đã thoát.
- Dữ liệu: 7.092 train có poses, 519 dev, không dùng test. Official split giữ nguyên; 4 train thiếu poses, 315 recording groups chung train/dev đã ghi trong coverage.
- Phạm vi: 1 seed, MAP và sampled-q, cùng codecs/checkpoint khởi tạo, cùng số update; optimizer generator reset chung tại update 2.050 theo recovery protocol.
- Kết luận khoa học: **INCONCLUSIVE — chưa đủ bằng chứng để scale hay claim SOTA**.

## Mức hoàn thành

Codec 3.535 và segment codec 12.285 updates đã được giữ nguyên. Mỗi nhánh downstream hoàn tất planner 1.284, generator 2.645 và exposure 161 updates. Evaluator mới hoàn tất 35.460 successful updates, không skip batch nonfinite. Hai nhánh mỗi bên xuất đủ 519 samples text-only với DDIM 50; NPY raw133 đúng shape, finite, length bằng hai lần tổng latent durations. Cả 1.038 mẫu đều EOS, không có rotation fallback; không có mẫu bị gắn collapse theo ngưỡng mean native-joint displacement hiện tại. Điều này không chứng minh motion hiểu đúng ngôn ngữ ký hiệu hoặc articulation tự nhiên.

## Kết quả chính — diagnostics nội bộ

| Chỉ số | MAP | Sampled-q | Diễn giải |
|---|---:|---:|---|
| Pilot BT whitespace BLEU-4, cao hơn tốt | 1.5857 | 1.9637 | q +0.3780 điểm; metric chưa qua calibration |
| Native41 DTW, mm, thấp hơn tốt | 129.833 | 131.717 | q kém 1.885 mm / 1.45% |
| Collapse theo ngưỡng tốc độ | 0/519 | 0/519 | Chỉ là diagnostic kỹ thuật |
| Cap-stop | 0/519 | 0/519 | Tất cả dừng EOS, không bảo đảm duration chính xác |
| Mean generated/reference length ratio | 0.9433 | 0.9321 | So sau inference, không cung cấp reference length cho generation |
| Length MAE, frame, thấp hơn tốt | 30.82 | 33.83 | q có sai số duration lớn hơn |
| Sinh dưới nửa reference length | 40/519 | 65/519 | Trung bình length ratio che một số output quá ngắn |

Q có DTW tốt hơn MAP ở 248/519 clips, không là kiểm định ý nghĩa thống kê. BT không phải published SOKE BLEU và native41 DTW không phải official mesh-hand evaluator. Không so các con số này trực tiếp với leaderboard/paper khác.

## Nút thắt calibration

BT real dev **9.4981**, shuffled **1.0842**, codec reconstruction **1.6705**. Codec reconstruction chỉ giữ **17.6%** điểm real, dưới mốc sàng lọc tạm 75%. Đây là lý do report tự động INCONCLUSIVE. Có thể codec mất semantic/articulation details hoặc evaluator nhạy với distribution shift; chưa tách hai nguyên nhân và không được gọi đây là bằng chứng chắc chắn codec làm mất ngữ nghĩa.

Ngay cả nếu tạm bỏ gate, gain BT +0,378 cũng chưa đạt margin +0,5 đặt trước pilot; tuy nhiên metric chưa hợp lệ nên không biến việc thiếu margin này thành bác bỏ toàn bộ giả thuyết. DTW/duration chưa có lợi ích ở q. Hiện chưa có lý do thực nghiệm đủ mạnh để mở rộng compute/dataset cho cấu hình hiện tại.

## Quyết định và bước tiếp theo

1. Giữ toàn bộ checkpoints/predictions; không mở rộng training chỉ để tìm điểm đẹp hơn.
2. Kiểm tra paired real motion ↔ codec reconstruction: geometry/velocity theo body/hands và xem các trường hợp articulation, orientation, mouth/NMM. Tách lỗi codec và độ nhạy evaluator bằng same-motion controls/perturbations hoặc bộ đo độc lập.
3. Khi tầng đo/representation đạt, chạy oracle-plan vs predicted-plan để tách planning/timing và realization; các tail duration hiện tại là diagnostic cần xem.
4. Chỉ khi có lợi ích rõ mới xác nhận nhiều seed và thêm jitter/uniform/K-matched controls. Một pilot MAP/q chưa xác nhận novelty, generalization hoặc SOTA.

## Lưu ý lịch sử vận hành

`queue_state.error=KeyboardInterrupt` và thông báo QUEUE_RESULT cũ là dấu vết của kiểm tra restart có kiểm soát. Trạng thái cuối hiện tại là completed, mọi job exit code 0. Không sửa metrics, predictions hay raw queue history để che các sự kiện cũ. `QUEUE_RESULT.md` được cập nhật để phân biệt hoàn tất vận hành và kết luận khoa học chưa rõ; bản cũ được giữ trong final_review_history.

Nguồn chính: `metrics.json`, `evaluator_calibration.json`, `final_audit.json`, `map_ready.json`, `sample_ready.json`, và `docs/proposal2/phoenix_recovery_20261107.md`.

# CPR-Flow: job training/evaluation với tối đa 2 tiến trình

Bộ job nằm trong package `cprflow.jobs`, public CLI `python -m cprflow.cli.jobs`. Scheduler riêng cho method này; không dùng shell queue cũ `scripts/jobq.py` vì queue đó không có dependency, recovery và gates của nghiên cứu.

## Cấu hình mặc định được triển khai

- Campaign `joint3_standard`: **một seed17**, joint training ba train splits; AE/duration/parent CFM, B1-cont/C0, teacher, full semantic continuation + equal-budget no-semantic control, C1/C2 inference controls, evaluation dev/test từng dataset, oracle dev diagnostics và BT integration jobs.
- Tối đa **2 payload jobs** đồng thời, mỗi job một Python process. Không dùng torchrun; DataLoader workers0. Job render/profiling chiếm cả2 slots để tránh ffmpeg helper/profiling bị chạy chồng.
- GPU0, OMP/MKL/OpenBLAS2threads; giữ nguyên main384/8/6, canonical296/expression50, global batch và số updates của spec. Các job GPU có VRAM reservation; không launch thêm khi GPU đang thiếu chỗ.
- Cache/text/stats/geometry/AE dùng chung giữa models/seeds. Cache smoke chỉ chứa48 train clips ngắn hoàn chỉnh và dev panel; text/geometry liên kết tới cache chung, không copy corpus.
- Checkpoint5k/update vẫn giữ nguyên cadence phục hồi; chỉ giữ periodic snapshot mới nhất cùng các milestone50k/100k/200k/300k, `last.pt`, `best_ema.pt` và duration `best.pt`. **Không xóa checkpoint parent200k cần warm-start**.
- Geometry cache float32 chỉ lưu52 joints duy nhất cần cho các loss hiện tại, rồi dựng body/hand/local groups bằng names. Không lưu trùng groups, không hạ precision motion/geometry. Cache cũ vẫn đọc được.
- Không tự render toàn bộ test, lưu mesh/vertices, copy raw LMDB hoặc nhân cache theo seed. Raw nguồn được giữ readonly.
- Mặc định budget dữ liệu+training outputs **100GiB**, reserve filesystem15GiB. Các write numeric/checkpoint kiểm tra reserve; scheduler dừng job có checkpoint khi gần ngưỡng. Đây là **giới hạn admission/guard**, không khẳng định corpus thật chắc chắn nhỏ hơn100GiB. Nếu vượt, job chờ người vận hành giải quyết, không tự xóa dữ liệu để ép vừa.

## Dependency và các điều kiện cần có thật

```text
Environment/source audit
  -> access record -> download pinned9LMDB -> import
  -> verified pose timing -> canonicalize -> native geometry renders -> G0 review
  -> train-only stats + shared text/geometry caches
  -> real smoke AE1k -> smoke latent cache -> CFM500 + exact resume tới1k -> sample/check
  -> full AE + duration -> frozen latent stats/cache -> real GPU profile
  -> CFM parent200k -> B1-cont/C0 continuations100k
  -> teacher + genuine calibration -> full semantic continuation/control
  -> full dev reports -> frozen release from clean source -> final test + macro reports
```

Không tự tạo `authorized=true`, verified FPS, G0 approval hoặc teacher calibration. Các file cần cung cấp:

1. `access_record.json`: quyền sử dụng ba nguồn dữ liệu, có references thật.
2. `source_timing.json`: đúng HF revision và có bằng chứng effective feature FPS.
3. `runs/joint3/geometry_audit/review.json`: G0 review với hashes và64 reviewed UIDs mỗi corpus.
4. `teacher_calibration.json`: approved calibration dev, teacher/asset/preprocess hashes, sensitivity/counts/trust budgets đúng spec.
5. `integrations/backtranslation.json`: adapter đã audit và thử real fit/AE; code/checkpoint recognizer cần được pin. Các BT jobs chờ file này, không xuất điểm0 thay thế.

Nguồn code phải được commit sạch trước final test theo protocol của SLP. Scheduler không tự commit hoặc mở test sớm. Test ledger append-only được loại khỏi phép kiểm tra source-dirty để các job test sau không tự chặn nhau. Final release được tạo tự động từ checkpoints đã chọn và full-dev reports, theo yêu cầu chạy toàn bộ training/evaluation của người dùng; không đặt thêm bước xin phép test sau đó.

## Khởi tạo và chạy

Từ thư mục method:

```bash
cd /home/haipd/SLP/methods/cpr_flow
python -m cprflow.cli.jobs create --id joint3_standard --preset standard --seeds 17 --device cuda --budget-gib 100
python -m cprflow.cli.jobs start --plan runs/jobs/joint3_standard/plan.json
python -m cprflow.cli.jobs status --plan runs/jobs/joint3_standard/plan.json
```

`start` chạy detached supervisor, không phụ thuộc phiên terminal. State/receipts/events dùng SQLite WAL và atomic JSON; stdout/stderr của từng attempt nằm trong `runs/jobs/<campaign>/attempts/<job>/<attempt>/output.log`. `status --json` xuất toàn bộ jobs, PID, reasons và attempts. Scheduler tự kiểm tra lại gates khi file bằng chứng được cung cấp; các job thiếu điều kiện ở trạng thái waiting, không failed giả và không skipped âm thầm.

```bash
python -m cprflow.cli.jobs stop --plan runs/jobs/joint3_standard/plan.json
python -m cprflow.cli.jobs start --plan runs/jobs/joint3_standard/plan.json
python -m cprflow.cli.jobs retry --plan runs/jobs/joint3_standard/plan.json --job <failed_job_id>
```

Stop gửi yêu cầu graceful, trainer hoàn tất logical update hiện tại rồi lưu `last.pt`; sau180s có thể bị dừng cứng nếu không phản hồi. Restart xác nhận PID+Linux start-time token: **adopt job còn sống**, không launch lại vì chỉ mất quan sát. Nếu process đã kết thúc, scheduler đọc durable receipt; không suy success chỉ từ file tồn tại. Job train retry tự chọn checkpoint tiến xa nhất và dùng `--resume`; fresh continuation mới dùng `--warm-start` parent EMA. Scheduler không reset optimizer hoặc overwrite run âm thầm.

Global flock và admission claims dùng chung cả campaigns. Khi supervisor bị kill, claim của payload còn sống vẫn chiếm slot/VRAM; campaign khác không vượt cap2. Payload chỉ bắt đầu sau khi admission đã được ghi bền vững. Failed prerequisites chặn dependents; các nhánh độc lập vẫn có thể chạy. Automatic retries có giới hạn; interrupted jobs giữ checkpoint để tiếp tục.

Manifest/configs/source digest bị đóng băng lúc create. Nếu sửa implementation hay config trong khi campaign chạy, scheduler dừng admission job mới. Commit không đổi nội dung source thì không đổi digest. Khi thay representation/stats/protocol, tạo campaign mới; không sửa manifest cũ để giả exact resume.

## Confirmatory matrix

Không tự mở rộng standard thành hàng loạt ablations/seeds tốn dung lượng. Khi cần toàn ma trận đã định trong proposal:

```bash
python -m cprflow.cli.jobs create --id joint3_confirmatory --preset confirmatory --seeds 17,42,123 --device cuda --budget-gib 200
```

Preset này thêm B0 diffusion, B2 auxiliary-only, B3 capacity control, mono×3 và các seed yêu cầu; vẫn cap2 và dùng shared AE/stats/text. Budget lớn hơn là trần guard, không phải dung lượng đã cấp sẵn. Global scheduler chỉ có một owner tại một thời điểm. Không launch campaign thứ hai chồng lên campaign đang giữ khóa; có thể chạy nó sau hoặc dùng recovery của campaign đã submit.

## Interface của BT adapter

File integration là metadata thật của adapter đã cài, không phải approval mẫu:

- `adapter_file`: Python adapter path; `adapter_sha256`: SHA256 thực.
- `validated_on_real_fit_and_ae`: chỉ true sau khi thực sự kiểm tra.
- `argv`: danh sách arguments; placeholders `{checkpoint}`, `{dataset}`, `{split}`, `{manifest}`, `{output}`.

Adapter phải viết report JSON có `metric_status=available`, metrics thực, dataset/split/checkpoint SHA đúng. Không dùng LLM caption hoặc đổi recognizer giữa methods. Thiếu adapter/checkpoint thì BT jobs waiting; geometry/time evaluation vẫn chạy được sau training và release tương ứng. Human comprehension/minimal-pair study vẫn cần dữ liệu/đánh giá người ký hiệu, không được tự suy từ job metrics.

## Operator-requested unattended run

Sau đó người vận hành đã xác nhận quyền nghiên cứu cả ba corpus và yêu cầu không chờ duyệt thủ công nữa. Dùng `--operator-waived-checks` khi tạo campaign mới để ghi chính sách này vào mọi resolved config:

- Không tạo G0 approval giả; gate trả `operator_waived`, `approved=false` và lưu reference yêu cầu của người vận hành.
- Tự đối chiếu timing với video/frame folders có sẵn trên server. Nếu chưa đủ bằng chứng về effective feature FPS, dùng nominal FPS như một **assumption** cho exploratory run, ghi `verified=false` trong evidence/manifest/report. Không mô tả các đại lượng thời gian đó là số đo vật lý đã xác minh.
- Source adapter nhận đúng các layout thực tế của pinned release: hands90 hoặc hands96 mà6 channels cuối trùng hoàn toàn wrist rotation trong body. Không thêm face giả hoặc đổi canonical296/native-left.
- Final evaluation được phép dùng source snapshot thay vì chờ commit; release/test ledger lưu rõ source digest và trạng thái worktree, không ghi giả `clean_committed`. Kết quả mang protocol `exploratory_operator_requested_v1`.
- Các kiểm tra thiết yếu về schema/shape/NaN, quyền đã xác nhận, resume, tài nguyên và chống chạy trùng vẫn giữ. Không tự tạo semantic calibration/human annotations; các nhánh đó không chặn C0 và các kết quả tự động hiện có.

## Optimized campaign (2026-11-12)

Campaign tiếp nối là `joint3_fast_measured`; `runs/jobs/current/plan.json` trỏ tới campaign đang hoạt động. Tạo mới bằng `--optimized --operator-waived-checks`. Effective batch, balanced sampler, toàn bộ clip hợp lệ, số updates, kiến trúc và trọng số loss giữ nguyên.

- AE: microbatch8, accumulation16, global batch128. Batched convolutions tái lập replicate boundary ở true length tại từng convolution/residual; decoder giữ internal length `2*ceil(T/2)` trước khi crop. FK ghép các frame hợp lệ qua clips, chunk512; mọi reduction vẫn tính riêng từng clip.
- Flow: microbatch4, accumulation16, global batch64; reservation16GiB theo phép đo clip dài (AE8GiB, teacher12GiB). Eligible auxiliary clips được batch decode/FK. Pearson lag windows/pairs/offsets được vectorize; coverage và quy ước offset không đổi.
- Cache text/latent/geometry được đọc một lần cho mỗi UID trong logical batch, không giữ cache vô hạn trong RAM. SHA của motion chỉ tính lại nếu size/mtime thay đổi. Logging và kiểm tra finite trên một GPU giảm đồng bộ CPU/GPU.
- Migration AE tại successful step346 chỉ cho phép đổi execution/microbatch/accumulation/FK chunk. Receipt và checkpoint gốc nằm trong `runs/joint3/ae_joint3_s17/runtime_migrations/batched_s346/`. Model, optimizer, scheduler, scaler, EMA, sampler và RNG được so sánh sau khi ghi lại. Không reset optimizer; thay microbatch trong BF16 không được mô tả là bitwise-equivalent trajectory.
- 22 job đã thành công được kế thừa kèm receipt nguồn trong `runs/jobs/joint3_fast_measured/carryover.json`; không train lại duration, smoke hay tạo lại dữ liệu/cache đã hoàn tất. Chính sách tiết kiệm dung lượng và giới hạn tối đa2 payload tiếp tục áp dụng.

Kiểm tra loss/latent/gradient, masking, packed FK, packed flow auxiliary, lag và migration: `verification/speedup_tests.log`. Đo AE trên dữ liệu thật: `verification/speedup_real_measurements.json`; phép đo flow tạm dùng AE chưa hoàn tất phải được đọc theo các giới hạn ghi trong `verification/speedup_flow_proxy.json`. ETA cần tính trọng số theo xác suất corpus/bucket thực tế; không nhân speedup của một clip ngắn cho toàn bộ pipeline.

> Cập nhật 07/11: đã bỏ **mọi hạn thời gian job/stage/probe**. Các số giờ trong nội dung cũ dưới đây là lịch sử, không còn áp dụng. Xem [kế hoạch khôi phục](phoenix_recovery_20261107.md) và runtime plan.json.

# PHOENIX-2014T: pilot chạy đến khi hoàn tất hàng đợi

## Material Passport

- Loại: cấu hình thực nghiệm và hồ sơ setup; chưa phải kết quả nghiên cứu.
- Ràng buộc được duyệt: 1 dataset, 1 seed, tối đa 2 tiến trình, bắt đầu sau 00:00 ngày 06/11/2026 Asia/Ho_Chi_Minh. Ngày 06/11 người dùng bỏ hạn tổng 72 giờ; chạy đến khi các job đã setup kết thúc. Tên file/unit có `72` được giữ để bảo toàn đường dẫn lịch sử.
- Thiết kế: so sánh point-MAP với sampled-q của proposal temporal-plan hiện tại. Không dùng kết quả thực nghiệm cũ để khởi tạo hoặc chọn cấu hình.
- Nguồn dữ liệu: official PHOENIX train/dev, poses SOKE đã đóng gói; pretrained mT5 và SMPL-X có provenance riêng. Không sử dụng test.
- Các artifact runtime nằm ở `outputs/phoenix_pilot72_20261106/` và không được commit.

## Giới hạn và cách đếm thời gian

Timer chỉ khởi động hàng đợi sau mốc cho phép. Dispatcher và từng worker còn có chốt giờ độc lập, cố định sớm nhất `2026-11-06T00:00:00+07:00`; sửa timer/plan về sớm hơn cũng không vượt qua chốt này. Lịch mặc định 00:01; nếu setup kết thúc muộn hơn thì hẹn sau thời điểm setup ít nhất 3 phút. Xem giờ thực tế trong `ARMED.json`.

Không còn hạn chót tổng: `hours: null`, `queue_state.deadline_epoch: null`, systemd `RuntimeMaxSec=infinity`. Thời gian chờ GPU không trừ vào ngân sách thực nghiệm. Tổng 66,1 giờ trong bảng là tổng timeout riêng của các job khi đã bắt đầu; không phải giới hạn thời gian toàn hàng đợi. Giữ số update, giới hạn train từng stage và watchdog từng job từ thiết kế cũ để phát hiện job lỗi/treo, không mở train thành số epoch vô hạn. Sau khi các job hoàn tất, bị lỗi hoặc bị skip theo dependency, hàng đợi lập báo cáo rồi thoát. Không tự retry job crash.

Việc đổi cấu hình được làm khi chỉ `prepare` đã hoàn tất. Kết quả prepare, dữ liệu và các neural config được giữ nguyên. Phiên bản dispatcher, plan, state và unit cũ được lưu trong `control_changes/20261106T080000+0700/`; manifest của snapshot mới được cập nhật.
Một dispatcher Python quản lý một worker Python. Worker không tạo subprocess, DataLoader dùng `num_workers=0`, tokenizer không mở nhóm xử lý song song, CPU threads đặt 2. Lệnh kiểm tra GPU chỉ được gọi khi chưa có worker. Chỉ có một job hoạt động tại một thời điểm. Chờ bằng handle `Popen.poll()/wait()` thật; không khởi động job sau chỉ vì tool trả timeout. Khóa `flock` ngăn hai dispatcher chạy trùng. Timer không giữ tiến trình pilot nào trước giờ bắt đầu.

User systemd + `Linger=yes` giúp lịch tồn tại sau khi thoát phiên đăng nhập. Service `Restart=no`: không tự retry job bị crash. Nếu service được khởi động lại thủ công, các job hoàn tất được giữ, job bị ngắt chuyển failed; chỉ job pending đủ dependency mới được xét tiếp. Không hỗ trợ resume chính xác optimizer/RNG giữa stage. Không giết tiến trình của người khác. Dispatcher dùng `gpu_policy.mode=measured_memory`; không có ngưỡng GiB cố định, điều kiện GPU phải rỗng hay điều kiện utilization thấp. Mỗi GPU job có một lượt chẩn đoán VRAM riêng trong worker trước khi chạy thật. Worker được giới hạn allocator ở 90% VRAM đang trống để chừa phần dự phòng theo tài nguyên thực tế. Probe đo max TRAIN motion canvas, text cap 128, plan dài tối đa; hai bước optimizer thật để gồm cả AdamW state, sau đó hủy weights. Probe thành công được cache theo job và đối chiếu lại VRAM trống trước khi khởi chạy job thật. Probe CUDA OOM thoát với code 75, giải phóng context, rồi hàng đợi chờ budget tăng 20% so với lần thiếu (giới hạn theo dung lượng GPU) trước khi thử lại. Utilization chỉ hiển thị, không chặn chạy. `gpu_wait` ghi free/budget/required bytes và thời điểm; `waiting_job` ghi job đang đợi; `memory_probe` và `memory_profiles` ghi diagnostic và peak đã đo. Đây không phải cơ chế khóa GPU cho toàn server; người khác vẫn có thể tự khởi động công việc của họ. Giới hạn hai tiến trình chỉ áp dụng cho pipeline của pilot, không áp dụng cho tài khoản khác trên server.

Mỗi job có log, PID, thời điểm bắt đầu/kết thúc, exit code và dependency trong `queue_state.json`. Hard timeout gửi TERM tới process group riêng, chờ tối đa 30 giây rồi KILL nếu cần để thu hồi tiến trình của pilot. Khi trống dưới 20 GiB hoặc output pilot vượt 18 GiB, dừng worker và ghi nguyên nhân; không xóa dữ liệu của người dùng. Không dừng hàng đợi chỉ vì đã qua 72 giờ. Nếu job lỗi hoặc timeout riêng, báo cáo chỉ rõ phần incomplete; không coi đó là kết luận method thất bại.

## Thứ tự job

| Job | Hạn mức | Nội dung / điều kiện hoàn tất |
|---|---:|---|
| prepare | 2 h | Đối chiếu official annotations và poses; tạo raw133 train/dev, hash từng NPY, tải đúng commit mT5. |
| preflight | 1 h | Kiểm tra FK bằng assets thật và gradient so với full SMPL-X; thử backward trên các clip train dài nhất với microbatch 2; ghi VRAM. |
| common | 14 h | Codec tối đa 8 h; segment codec tối đa 3 h; phần còn lại cho fit statistics toàn train và checkpoint. |
| evaluator | 6 h | Train evaluator pose→German độc lập tối đa 4,5 h trên real train; đóng băng, kiểm tra real/codec/shuffle trên dev. |
| profile | 1 h | Từ checkpoint chung, đo cả MAP và q; mỗi stage 2 lượt cập nhật, mỗi lượt 64 mẫu có clip dài nhất. Bỏ toàn bộ weights profiling. |
| map | 17 h | Planner ≤3 h, diffusion ≤11 h, exposure ≤2 h; phần còn lại dành checkpoint/validation. |
| sample | 17 h | Cùng initialization, cùng seed và cùng target số update từng stage như MAP. |
| predict_map | 3 h | Sinh toàn bộ 519 dev bằng text-only, DDIM 50 bước, không truyền độ dài/reference motion. |
| predict_sample | 3 h | Sinh cùng tập dev, cùng quy tắc seed và sampler. |
| evaluate | 2 h | BT đã freeze, native41 DTW, cap-stop, collapse, độ dài chẩn đoán; kiểm tra đủ 519 predictions mỗi nhánh. |
| report | 0,1 h | Tổng hợp GO / NO_GO_FOR_SCALING / INCONCLUSIVE và lý do. Vẫn chạy khi job trước thất bại nếu tài nguyên còn cho phép. |

Các job trên luôn tuần tự. Dependencies cho phép nhánh còn lại vẫn hoàn tất nếu một nhánh lỗi, nhưng không xuất kết luận so sánh khi thiếu một bên. Evaluator lỗi không được thay bằng một metric giả; báo cáo thiếu đánh giá ngữ nghĩa.

## Dữ liệu và tính công bằng

- Official annotations: `/home/dongvk/datasets/phoenix14T`.
- Poses: `/home/haipd/SLP/data/processed/madis/Phoenix_2014T/poses_video_level`.
- Có 7.092 / 7.096 train có poses trong gói hiện tại; dev đủ 519. Bốn train thiếu được liệt kê trong `coverage.json`; hai nhánh dùng cùng tập 7.092 này. Không tạo lại split.
- Theo source recording (bỏ clip suffix), official train/dev có 315 recording groups chung. Chỉ exception `source_id` được bật rõ ràng; name/file overlap vẫn bị cấm. Đây là official dev, không phải đánh giá độc lập recording. Coverage runtime sẽ xác minh lại.
- Raw133 = 30 upper-body + 45 left hand + 45 right hand + 3 jaw + 10 expression. Bỏ root/lower body/shape theo hợp đồng representation hiện tại; fixed SMPL-X shape được ghi trong provenance. Không crop, không chuẩn hóa trước từ dev.
- Motion/latent/content statistics chỉ fit trên train ở checkpoint chung, sau đó cố định cho cả hai nhánh. Cùng toàn bộ codec, segment codec, planner, projection và denoiser initialization từ checkpoint chung. Optimizer mới tại ranh giới stage được ghi rõ.
- Cấu trúc đầy đủ trong `methods/temporal_plan/config.yaml`: hidden 512, planner 4 lớp, denoiser 8 lớp, 2 teacher plans, DDIM 50. Chỉ microbatch giảm từ 8 xuống 2, accumulation tăng lên 32 để giữ effective batch 64. Không âm thầm giảm model.
- Số update bằng `floor(0.65 * stage_seconds / slowest_measured_seconds_per_update)` cho từng stage, tối thiểu 1. Dùng chậm nhất trong hai nhánh, reserve 35% cho biến thiên/validation. File `matched_budget.json` được ghi trước khi train MAP/q. Early stopping tắt để hai nhánh đạt đúng số update. Nếu một nhánh hết thời gian trước target thì stage không được đánh dấu complete và không có comparison hợp lệ.
- Native-only FK là phần native-joint chính xác của official LBS, gồm shape + expression + pose mean; không dùng mesh để tính loss 41 joints. Có test position/gradient và preflight assets thật. Không coi nó là metric mesh-hand của SOKE.
- Snapshot mã nguồn và config được tạo trước khi arm timer; train chạy từ snapshot, có `source_manifest.json`. Không tự lấy sửa đổi repo phát sinh sau khi arm.

## Đánh giá và điều kiện dừng

Không có evaluator BT PHOENIX public đã được xác minh sẵn trong checkout này. Pilot vì thế train một evaluator riêng, không gán tên hoặc tính tương đương với BLEU của paper khác. Evaluator nhận learning256 của poses thật, hai conv stride 2 và Transformer 2 lớp, nối decoder mT5 đã pretrained; chỉ pose encoder, decoder cross-attention và layer norms được train. mT5 encoder của generator vẫn frozen. Evaluator không dùng latent codec của generator, không train bằng generated samples và không đưa loss ngược vào generator. Statistics representation dùng cùng train-only statistics. Evaluator được freeze trước khi sinh/đánh giá hai nhánh.

Metric ngữ nghĩa là **case-sensitive whitespace corpus BLEU-4, không smoothing**, tên `pilot_BT_whitespace_BLEU4`. Không so trực tiếp với published SOKE BLEU. Evaluator chỉ được dùng để ra quyết định khi real dev ≥5, real trừ shuffled-caption ≥2 và codec reconstruction ≥75% real score. Các ngưỡng là quy tắc engineering của pilot, không phải chứng minh evaluator phản ánh đầy đủ người dùng/ngôn ngữ ký hiệu. Nếu không đạt: INCONCLUSIVE về ngữ nghĩa.

Geometric metric là DTW trên 41 native joints, fixed shape, mm; báo theo từng clip và trung bình. Đây là chẩn đoán riêng, không gắn nhãn official benchmark JPE. Primary text-only generation không biết reference duration; độ dài reference chỉ dùng sau khi sinh để chẩn đoán và DTW. Collapse = mean native-joint displacement dưới 0,1 mm/frame; cap-stop được phân biệt EOS.

Quy tắc đã chốt trước chạy:

1. NaN/Inf, OOM trong training thật, thiếu assets, dữ liệu sai, timeout hoặc dependency lỗi: dừng job, không tự retry training; kết luận thiếu bằng chứng khi không đủ comparison. Ngoại lệ được người dùng duyệt: CUDA OOM trong probe VRAM dùng một lần là tín hiệu chờ thêm tài nguyên, được thử lại sau khi đủ budget. Probe không lưu checkpoint thực nghiệm.
2. Ít hơn planner 100 / generator 200 / exposure 50 updates ở mỗi nhánh, hoặc evaluator không qua calibration: INCONCLUSIVE, không bác bỏ method.
3. Khi comparison hợp lệ: q collapse/cap-stop trên 5%, hoặc BT kém MAP ≥1 điểm, hoặc DTW kém trên 10%: NO_GO_FOR_SCALING.
4. q tăng BT ít nhất 0,5 điểm và DTW không kém quá 5%, đồng thời qua kiểm tra collapse/cap: GO_TO_MULTI_SEED_CONFIRMATION.
5. Các trường hợp hợp lệ còn lại: NO_GO_FOR_SCALING vì chưa đạt mức cải thiện hữu ích đã đặt ra trong pilot này. Không diễn giải một seed là bằng chứng phủ định method nói chung.

Không có kiểm định nhiều seed hay claim SOTA trong pilot này. Chỉ số loss teacher-conditioned dùng để chọn checkpoint từng stage không được trình bày như text-only generation quality.

## Theo dõi / dừng

Tên unit: `slp-phoenix-pilot72-20261106`.

```bash
# Xem service đang tiếp tục hàng đợi (timer 00:01 đã kích hoạt và được dừng sau lần đổi cấu hình)
systemctl --user status slp-phoenix-pilot72-20261106.service

# Xem trạng thái hàng đợi
python -m methods.temporal_plan.pilot.queue status --plan outputs/phoenix_pilot72_20261106/plan.json

# Dừng cả lịch còn chờ và service đang chạy
systemctl --user stop slp-phoenix-pilot72-20261106.timer slp-phoenix-pilot72-20261106.service
```

Khi kết thúc, đọc `REPORT.md` / `report.json`. Nếu queue gặp lỗi trước job report, đọc `QUEUE_RESULT.md`, `queue_state.json` và các log của job. Không tự chạy thêm một bản train bằng tay trong lúc service hoạt động.

## Kiểm tra setup

Các kiểm tra setup chỉ dùng CPU với CUDA bị ẩn, HF offline; không tải model hoặc đọc train/test để chạy thực nghiệm. Bao gồm scheduler real-child wait/timeout/exit code, fixed date gate, dependencies/no retry, raw133 mapping, source exception, shared stage-boundary forks, equal updates, incomplete-stage guard, official synthetic SMPL-X native-FK position/gradient parity, actual tiny mT5 evaluator forward/backward/compact checkpoint/generation và BLEU/DTW sanity. Kết quả cụ thể được ghi tại `docs/phoenix_pilot72_setup_verification.json`. Preflight dữ liệu thật và GPU chỉ chạy sau giờ hẹn.

## Vì sao chưa chạy tiếp sau prepare lúc 00:02?

Prepare bắt đầu 00:01:00, kết thúc thành công 00:02:26 ngày 06/11. Cấu hình cũ kiểm tra `nvidia-smi --query-compute-apps=pid` và chỉ cho chạy GPU khi danh sách hoàn toàn rỗng. Vì vậy một tiến trình nền có CUDA context cũng đủ chặn preflight. Đây là điều kiện quá chặt của dispatcher đã setup, không phải prepare tự tạo thêm training job.

Khi kiểm tra lúc khoảng 07:55–07:57, GPU có PID 1997660 (`root`, process khởi tạo 04/11 20:52, dùng khoảng 1,6 GiB VRAM) và PID 3503929 (`vtvan`, khởi tạo 06/11 01:08, dùng khoảng 33 GiB). Hai PID này không thuộc các worker pilot; worker prepare là PID 3458581, đã kết thúc. Dispatcher cũ là PID 3458580. Job của `vtvan` bắt đầu sau khi prepare hoàn tất, không phải do pilot sinh ra. Không có lịch sử PID GPU tại đúng 00:02 nên không khẳng định PID nào giữ CUDA context ở thời điểm đó. Hàng đợi cũ chỉ ghi thông báo chung GPU occupied; phiên bản mới ghi cả số đo tài nguyên và thời điểm.

GPU đang dùng khoảng 35 GiB và utilization khoảng 93% khi kiểm tra; việc bỏ hạn tổng không có nghĩa là ép train đồng thời khi thiếu tài nguyên. Không dừng hoặc sửa tiến trình của tài khoản khác.

Bản cập nhật đã qua 86 tests CPU (15,11 giây), gồm completion không có global deadline và GPU admission theo tài nguyên/ổn định. Xem `docs/phoenix_pilot72_setup_verification.json` và `latest_control_change.json` trong thư mục output.

## Cập nhật chờ VRAM theo số đo — 06/11

Bỏ hoàn toàn điều kiện `free >= 40 GiB`, `utilization <= 10%` và hai lượt chờ GPU nhàn. Policy hiện tại là `measured_memory`, với phần dự phòng 10% của VRAM **đang trống**, không phải yêu cầu trống 90% tổng GPU. Ví dụ nếu còn 12 GiB, probe có allocator budget khoảng 10,8 GiB và có thể chạy ngay kể cả utilization 100%. Nếu probe chỉ cần 5 GiB, job thật được chạy sau khi xác nhận budget hiện tại vẫn đủ. Không giữ chỗ toàn bộ budget bằng một tensor lớn.

Chỉ có một probe hoặc một worker train tại một thời điểm, đi cùng dispatcher: tối đa hai tiến trình của pilot. Trước mỗi lần query GPU, worker trước đã `wait()` xong. Probe được chạy trong process riêng để context/allocator/weights được giải phóng hoàn toàn trước khi chờ. Chờ không giữ CUDA context. Thời gian chờ/probe không trừ vào timeout train của job; probe có watchdog riêng một giờ. Không có hạn tổng.

Các probe train đo backward và optimizer hai lượt cho mỗi stage; các probe inference đo canvas tối đa theo train cap. Probe có targets plan dài tối đa mang tính chẩn đoán shape; chúng không đi vào training thật, báo cáo chất lượng hay checkpoint khoa học. Hai nhánh thực nghiệm vẫn bắt đầu từ checkpoint chung cũ theo cùng quy tắc seed/update. `profile` cũng đã bổ sung peak allocated/reserved cho từng stage/mode và giải phóng các tham chiếu gradient cũ giữa hai mode.

Nếu một probe lỗi không liên quan bộ nhớ hoặc quá watchdog thì ghi failed, không tự retry vô hạn. OOM xảy ra sau khi training thật đã bắt đầu cũng ghi failed; không hứa resume optimizer/RNG và không tự chạy lại từ đầu. Lượt đo không thể bảo đảm chống mọi biến động: tiến trình khác có thể lấy thêm VRAM sau khi kiểm tra. Worker thật cũng được đặt allocator budget từ VRAM trống lúc nhận job; probe/wait không dừng tiến trình của tài khoản khác.

Log và kết quả probe ở `memory_probes/<job>/`. Các thay đổi trước/sau, cùng hash snapshot và kết quả 90 kiểm tra CPU (15,18 giây), được ghi trong `latest_control_change.json` và `docs/phoenix_pilot72_setup_verification.json`. Bản threshold 40 GiB trong phần lịch sử ở trên đã được thay thế.

Xác minh runtime lúc 08:47 ngày 06/11: preflight đã hoàn tất trên assets thật; probe common (codec + segment, microbatch 2, canvas 474 frame) peak allocated 1,527 GiB, peak reserved 1,580 GiB. Job common bắt đầu 08:47:07, không còn bị chặn bởi ngưỡng 40 GiB. Đây là số đo của common, không suy ra nhu cầu VRAM của planner/diffusion/evaluator.

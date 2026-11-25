# RelIKFlow

status: exploring

RelIKFlow dùng quan hệ tay–cơ thể và động học ngược để tạo nguồn cho flow matching, theo proposal 10 về text-to-sign SMPL-X liên tục. Phương án C dự đoán thời lượng và descriptor 39 chiều từ văn bản, giải IK ở các mốc pha, lấy mẫu quanh tâm bằng nhiễu tiếp tuyến AR(1), rồi dùng generator dự đoán đích với 5 lần gọi mặc định. Phương án B dùng cùng điều kiện với nguồn bán cầu đều. Phương án D dùng head quay 6D tạo tâm trực tiếp từ hidden state của planner. [Đặc tả](../../docs/proposal10/SLP_Flow_Matching_Implementation_Spec.md)

| Mô-đun | Phạm vi |
|---|---|
| `data.py`, `rotations.py`, `fk.py`, `relations.py` | Manifest, SOKE-133 đã giải chuẩn hóa, adapter NSA ánh xạ theo tên, mask và resampling, FK và descriptor |
| `models.py`, `source.py`, `bridge.py`, `sampling.py` | Planner mBART đóng băng, head D, IK 78 biến, bank prototype, correction JVP, generator và replan |
| `pipeline.py`, `train.py` | OOF theo video, seed theo clip, cache có checksum, gradient accumulation, giảm tổng gradient nhiều process, checkpoint nguyên tử và resume |
| `commands.py`, `evaluation.py` | Prediction, scorer adapters, bảng per-clip và paired bootstrap |

## Kiểm thử phần mềm

```bash
pip install -e '.[relikflow,dev]'
python methods/relikflow/smoke_slp_math.py
pytest -q tests/relikflow tests/test_methods.py
python -m methods.relikflow verify --output /tmp/slp-fm-smoke
python -m methods.relikflow verify --output /tmp/slp-fm-smoke --resume
```

Profile smoke tạo 8 clip train tổng hợp, 3 clip dev và 2 clip test. Dùng encoder nhỏ và FK giả lập cho luồng CLI. Unit test FK dùng thêm một model SMPL-X tổng hợp để đối chiếu packing, hand mean và landmark FK với lớp SMPL-X đầy đủ. Kết quả smoke chỉ biểu thị trạng thái phần mềm.

## Dữ liệu và cấu hình chạy thật

Dùng một YAML ghi đè lên `config.yaml`. Điền `assets.smplx` tới file neutral hoặc thư mục chứa `smplx/SMPLX_NEUTRAL.npz`. Điền `text.revision` bằng commit mBART đã khóa. Có thể thêm `text.local_snapshot` tới snapshot tương ứng. `data.soke_stats` trỏ tới thư mục mean/std khi manifest chứa pose normalized. Đặt `data.fps.<corpus>` nếu dữ liệu có nhiều timebase. Khi toàn bộ train corpus có một FPS, pipeline dùng giá trị đó cho track. Đặt `data.corpus` thành `csl_daily`, `phoenix_2014t`, `how2sign` hoặc `multi`.

Mỗi dòng JSONL khai báo `clip_id`, `video_id` nếu có, `corpus`, `split` (`train/dev/test`), `source_language`, `target_sign_language`, `text`, `pose_path`, `pose_layout`, `fps`, `num_frames`, `timebase_source`, `pose_fit_id`. Pose NPZ dùng `pose133` shape `[T,133]`, có thể thêm `timestamps`, `q_valid`, `expr_valid`, `r_confidence` và checksum trên manifest. Test thiếu toàn bộ reference khai báo `missing_reason` và để trống `pose_path`. Với `nsa_named`, khai báo `nsa_mapping` theo từng parameter `body_pose`, `left_hand_pose`, `right_hand_pose`, `jaw_pose`, `expression` với `key` và `indices` của release. Ví dụ nếu `body_pose` lưu 63 channels, ánh xạ 30 kênh trên bằng indices 33–62.

## Lệnh theo giai đoạn

Ví dụ gọi một run cho seed 17. Mỗi seed và mỗi profile có thư mục output riêng. Dùng `--resume` với cùng cấu hình và topology process. `runtime.microbatch` dùng theo từng process và `batch_effective` dùng toàn cục. Profile full giữ effective batch 256 cho generator theo bảng gốc, profile pilot dùng 64.

```bash
fm() {
  python -m methods.relikflow "$1" --config csl.yaml \
    --profile full --output outputs/fm/seed17 --seed 17 "${@:2}"
}

fm prepare --manifest data/csl.jsonl
fm fit-planner --oof 3 --fit-full
fm fit-direct-head --oof 3 --fit-full
fm build-sources --arms A-matched,B,C,D,E-rest,E-shuffle
fm audit-ik
fm fit-bank --split train
fm train-correction --arms A-matched,B,C,D
fm train-generator --arms A-matched,B,C,D
fm sample --split dev --arms A-matched,B,C,D,E-rest,E-shuffle,F-oracle-dev --nfe 5 --debug
fm evaluate --split dev --arms A-matched,B,C,D,E-rest,E-shuffle,F-oracle-dev --nfe 5
fm compare --split dev --baseline B --challenger C --nfe 5
fm benchmark --arm C --batch-size 16 --nfe 5
```

`train-correction` chia sẻ một checkpoint cho các nguồn đều. Đặt `bridge.mode: geodesic` để dùng bridge không correction. A-paper là cấu hình dựng lại từ proposal, dùng nguồn đều và thời lượng từ planner. A-zero giữ projection quan hệ với đầu vào 0. E mặc định dùng weights C khi sample, dùng `--retrained` sau khi học từng biến thể E. F chỉ có trên dev. `source.ablate_relation_groups` bỏ nhóm tương ứng khỏi điều kiện và trọng số IK. Đặt `generator.aux_relation_weight` và `generator.aux_velocity_weight` cho biến thể loss vật lý.

Sinh một câu từ checkpoint bằng `fm generate --text '你好' --source-language zh_CN --target-sign-language CSL --sample-id example --arm C --out example.npz`. NPZ chứa `pose133` vật lý, FPS và độ dài dự đoán. `--debug` lưu quan hệ, nguồn, tâm và trạng thái NFE. `generate` dựng input từ request, train stats, weights và donor bank nếu dùng E-shuffle.

## Protocol và artifact

`prepare` lưu `run.json`, `splits.json`, `assets.lock.json`, `dependencies.lock.json`, `protocol.json`, `stats.npz` và `coverage.json`. Các stage ghi `best.pt`, `last.pt`, `dev_history.jsonl` và `budget.json`. Khi chấm test, khóa cùng protocol, NFE, cohort và các seed rồi chạy `sample/evaluate/compare --split test`. `compare --run-roots` nhận các run seed khác nhau. Cột published nằm trên các hàng riêng trong CSV.

Chấm hình học dùng SMPL-X neutral, joint selection, hand regressor và DTW tích lũy. Mặc định body PA fit trên toàn bộ joints theo implementation SOKE. `body_pa_alignment: upper_body` chọn biến thể fit trên vùng thân. B-T và SiCLIP cần khai báo plugin `module:function` kèm `id`, `revision`, `preprocessing`, `checkpoint` và `sha256` trong protocol. Có thể đưa protocol ngay vào config để chọn checkpoint cũng dùng scorer đó.

- B-T nhận `(pose133_physical, fps, text_reference, corpus, spec)`, trả `hypothesis_tokens` và `reference_tokens`. Nếu checkpoint nhận feature theo khớp, đặt `input: fk_features` và khóa `feature_dim`, `feature_groups` (region, indices, root và axes theo release) cùng mean/std nếu cần. Khi bật, plugin nhận tensor feature đã ánh xạ ở đối số đầu tiên. Core tính corpus BLEU-4 không smoothing trên token đã khóa.
- SiCLIP nhận `(outputs, corpus, spec)` và trả R@1. Cần thêm `candidate_pool` và `candidate_sha256`.
- Khi thiếu scorer tương thích, hình học vẫn được chấm và checkpoint chọn theo hand PA rồi hand JPE. Cột B-T ghi `unavailable`. Các so sánh xác nhận, parity với fitting thật, và đánh giá người ký hiện ở trạng thái `pending`.

SOKE/NSA thực, checkpoint mBART và các protocol ngữ nghĩa cần tài sản của release tương ứng. [Danh mục tài sản](assets.lock.json)

## Job pilot PHOENIX trên một GPU

Runner dùng profile pilot, arm C và seed 17. Nó chuyển pose đã đóng gói trong repo thành manifest PHOENIX, giữ nguyên split train/dev và chia OOF theo broadcast trong train. Các split chính thức PHOENIX có thể chia sẻ broadcast. Job chấm hình học trên dev để chọn và kiểm tra checkpoint pilot.

Runner kiểm tra VRAM trống mỗi 60 giây và chỉ khởi chạy một giai đoạn khi GPU đã chọn còn ít nhất 24.576 MiB. Khóa dùng chung của các pilot RelIKFlow được giữ cả lúc chờ và lúc chạy. Mỗi lúc chỉ có tối đa một tiến trình xử lý giai đoạn. Job theo dõi dung lượng thực tế do NVIDIA báo và kiểm tra lại trước mỗi giai đoạn.

```bash
systemctl --user status relikflow-phoenix-pilot.service
journalctl --user -fu relikflow-phoenix-pilot.service
cat outputs/relikflow/phoenix_pilot_seed17/job_state.json
```

Chạy runner trực tiếp hoặc qua `systemd-run --user` để tách khỏi phiên terminal.

```bash
python scripts/relikflow_pilot_job.py --gpu-uuid GPU-...
```

Cấu hình nằm ở `configs/experiments/relikflow/phoenix_pilot.yaml`. Log chi tiết của từng giai đoạn nằm trong `logs/` của run. Khi chạy lại cùng output, runner bỏ qua giai đoạn đã hoàn tất và stage huấn luyện tiếp tục từ checkpoint mới nhất. Pilot ghi checkpoint mỗi 100 optimizer step và cuối mỗi epoch. Dự đoán OOF lưu theo nhóm 32 clip, nguồn IK lưu theo nhóm 16 clip, prototype bank lưu sau từng khớp và bước sinh pose lưu theo nhóm 8 clip.

Khi phát hiện CUDA hết bộ nhớ, runner thử lại cùng giai đoạn tối đa hai lần và kiểm tra lại ngưỡng 24 GiB trước mỗi lần. Các lỗi khác được giữ ở trạng thái `failed` cùng đường dẫn log để chẩn đoán.

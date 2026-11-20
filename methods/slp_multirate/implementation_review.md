# Rà soát triển khai Proposal 8

## Kết luận kỹ thuật

Pipeline đã được hiện thực trong `methods/slp_multirate/` với điểm vào
`python -m slp.cli`. Phạm vi đầu ra gồm manual signing và rotation đầu/cổ
trên rig 44 joints cố định. Kết luận về độ hiểu ASL cần đánh giá với người ký
hiệu theo protocol riêng.

| Phần đặc tả | Bằng chứng trong code và kiểm tra |
| --- | --- |
| Chuẩn bị dữ liệu, §1–2 | Adapter named NPZ và SignSparK LMDB. Hash payload, raw record, rig, manifest và cohort. Kiểm video-group overlap, FPS, split, joint mapping và observations. Fixture có short clip, two-hand contact và wrist quay 90°. |
| Geometry và coarse grid, §1–2 | Row-major 6D, fallback Gram–Schmidt, quaternion averaging trong bin, SLERP theo clock, FK mét và mask tổ tiên. Unit test kiểm SO(3) gần 0/π, stride 1, stride 4, T=1/5, partial bins và padding. |
| Anchors, §3 | Budget chung cho uniform, event và onset/mid/offset. Planner dự báo thời điểm liên tục, payload theo tay và hỗ trợ confidence hiệu chỉnh trên dev_cal. Event windows lấy từ reference. Paired comparison bỏ hợp các vùng ±1 frame quanh mốc oracle. |
| Cascade, §4–5 | Hai Transformer endpoint độc lập. Flow B có self-attention local dạng dense mask, coarse cross-attention và anchor cross-attention. `build_fine_target` tính correction từ body thực cấp vào mạng. Unit test kiểm khôi phục GT trong cap, all-masked loss, padding NaN và Euler NFE 1/2/4/8. |
| Training và inference, §5–7 | Length/body/planner/fine có stage độc lập, AdamW, cosine schedule, clipping, EMA, cache và checkpoint atomic. Kiểm CPU cho bốn stage cho thấy chạy liên tục và resume khớp trọng số, scheduler và RNG. Inference chỉ tiếp nhận text và clip ID. |
| Evaluation và ma trận, §7–8 | Metric mm, độ, duration, timing theo giây thật, event matching một-một, contact hysteresis, bootstrap theo video/seed và P1/P2 gates. CLI xuất P3 plan, tổng hợp matrix, profile và đối chiếu compute. Final test dùng bundle, calibration và lock gắn với commit. |
| P0 và bàn giao, §9 | Smoke dùng mock encoder và dữ liệu synthetic. Mỗi stage 20 optimizer updates, checkpoint/resume ở giữa, inference 2+2 NFE và chấm oracle/text-only. Tiny-overfit dùng tám clip, 300 updates, paths cố định. |

## Kết quả xác minh CPU

- `pytest` toàn repo đạt **379 passed, 3 skipped**. Có 24 kiểm tra thuộc Proposal 8.
- P0 chạy đủ 20 updates mỗi stage và xác minh hash của các checkpoint.
  Oracle và text-only chạy hết ba clip fixture mỗi chế độ, với 0 lỗi.
- Tiny-overfit đi từ loss trung bình 0,4744 ở 20 updates đầu xuống
  0,1372 ở 20 updates cuối, giảm **71,1%**. Mã nguồn được kiểm tra tại `4286e22`.
- Đã chạy thử CLI text-only, hiệu chỉnh confidence trên dev_cal synthetic,
  hiệu chỉnh detector trên dev_tune synthetic và dựng overlay rig fixture.
  Stride 1 có sai số joint tối đa 0 mm trên clip kiểm tra.
- Artifact kiểm tra ở `outputs/slp_multirate_final/`. Chỉ số báo cáo P1–P3 và
  độ đúng của geometry với nguồn ASL thật đang chờ dataset và rig đã xác minh.

## Các quyết định trong triển khai

- Model và pipeline nằm trong thư mục method. CLI chung là lớp gọi mỏng và các metric nằm trong `src/slp/evaluation/multirate.py` theo quy ước repo.
- Mean/std payload mốc được fit từ tất cả frame train có quan sát để B và D dùng cùng chuẩn hóa. Statistics correction dùng trọng số kỳ vọng của các nhánh oracle và predicted trong training mixture. Cả hai loại statistics được giữ cố định trong checkpoint.
- Mốc P0 dùng mạng width 32, một block và mock encoder. Block fine duy nhất có anchor residual để smoke kiểm được nhánh đó. Cấu hình sáu block gắn anchor residual ở block 2, 4 và 6.
- So sánh hình học khi dự đoán lệch độ dài dùng normalized time. Event matching, duration và contact dùng thời gian vật lý. Contact proxy được bật khi có audit xác nhận tối thiểu 30 quan sát và có nhãn để hiệu chỉnh trên dev_tune.
- Adapter SignSparK yêu cầu khai báo rotation convention, FPS hoặc timestamps, named schema và local axes của rig. Layout SMPL-X `body_pose` 21 joints cần thêm pelvis từ cùng bản fit. Nguồn đã xác minh pelvis canonical cố định có thể khai báo `pelvis_source: verified_identity` trong source manifest. Metadata lưu lựa chọn này.

## Runbook

Cài editable từ thư mục gốc. Bản ghi phiên bản môi trường đã kiểm tra nằm
trong `requirements.lock.txt`. Trước khi tạo cache thật, đặt revision bất biến
40 ký tự bằng `--set text.revision=...` hoặc một YAML đã resolve.

```bash
pip install -e ".[dev,slp-multirate]"
python -m slp.cli smoke --device cpu --tiny-overfit --out outputs/slp_multirate_p0
python -m slp.cli verify-run --stage-dir outputs/slp_multirate_p0/stages/fine
python -m pytest -q
```

### Dữ liệu và kiểm rig

`source_manifest.example.jsonl` mô tả một dòng nguồn. `path` tương đối với
manifest, `key` là khóa LMDB, `provenance` ghi dataset/pose-fit version và
quyền sử dụng. Lấy cấu trúc `rig.json` từ lệnh `synthetic` rồi điền offsets,
palm convention và local axes đã được xác minh từ pose fit.

Tùy chọn `--adapter named_npz` nhận `rotations[T,J,3 hoặc 6]`, `joint_names`,
`observation[T,J]` tùy chọn và source schema `named_local_v1`. Nhãn contact
dùng `contact_pairs[P,2]` theo index của mười tips chuẩn, `contact[T,P]` và
`contact_valid[T,P]`, cùng nguồn nhãn trong manifest. Event annotation gồm
`event_source`, `event_types` được annotate và `events` có `kind` cùng
`time_s` tính từ đầu clip. Giữ source manifest có signer ID trong kho riêng.

```bash
python -m slp.cli prepare --source "$MULTIRATE_SOURCE" --rig "$MULTIRATE_RIG" \
  --adapter signspark_lmdb --out outputs/multirate/canonical
python -m slp.cli audit --manifest outputs/multirate/canonical/manifest.jsonl \
  --split all --out outputs/multirate/audit
python -m slp.cli review-geometry --manifest outputs/multirate/canonical/manifest.jsonl \
  --clip-id "$MULTIRATE_REVIEW_CLIP" --out outputs/multirate/geometry --render
python -m slp.cli make-cohort --pilot p1 --seed 17 \
  --manifest outputs/multirate/canonical/manifest.jsonl --out outputs/multirate/p1_cohort.json
```

`review-geometry --render` dùng Matplotlib. So ảnh với dữ liệu nguồn được phép
truy cập để xác nhận trục cổ tay và trái/phải. `p0-readiness` đếm clip
dev_gate có event và hand coverage chung giữa uniform/event. Có thể truyền
`--geometry-review` kèm `rig_sha256`, `axes_verified`, `handedness_verified`
và `source_overlay_clip_id` sau khi đối chiếu clip nguồn. Nếu có signer IDs, tạo slice
dev độc lập bằng `signer-diagnostic --private-source ... --out ...` và đánh
giá với `--eval-cohort`. Giữ `--cohort` gốc để khớp checkpoint contract.

### P1/P2/P3

1. P1 dùng `train --pilot p1 --variant B|D --stage fine` với cùng
   `--manifest`, `--cohort`, text revision và seed 17. Lấy đường dẫn fine
   checkpoint được CLI in ra. Chạy `calibrate-eval` trên dev_tune để tạo
   detector chung, sau đó dùng `evaluate --protocol oracle --split dev_gate
   --fine-ckpt ... --calibration ...` cho mỗi variant. So bằng `compare
   --gate p1 --baseline <eval B> --candidate <eval D>`.
2. P2 dùng chung length và body stride 4 giữa B/D. Train planner mỗi policy.
   Dùng `cache-conditions` và `train --stage fine` với `--body-ckpt` và
   `--anchors-ckpt` tương ứng. Dùng `calibrate-confidence` nếu chạy learned q.
   Tạo checkpoint bundle qua `bundle` và chấm oracle, predicted body + oracle
   anchors, predicted GT-length, oracle body + predicted anchors và text-only.
   `compare --gate p2` nhận cặp predicted GT-length và hai đường dẫn
   `--baseline-text` / `--candidate-text`.
3. P3 dùng `pilot-plan --pilot p3` để xuất arm/seed/protocol. `compare-matrix`
   nhận JSON `{"arms":{"A":{"17":"path", "23":"path", "41":"path"}, ...}}`.
   Paths tương đối với file JSON. A–D là bắt buộc, E/F tùy chọn. Dùng
   `profile` trên cùng prompts và hardware, rồi dùng `compare-compute` để
   xác nhận độ lệch latency hoặc FLOPs nằm trong 10% cho bảng matched-compute.
   FLOPs là ước lượng trên các toán tử PyTorch profiler có hỗ trợ. Dùng
   `lock-test` cho cấu hình cuối trước khi mở test.
4. `infer --bundle ... --rig ... --text ... --out ...` sinh file NPZ chứa 44
   local rotations, joints/tips mét và timestamps. Với JSONL prompts, mỗi
   record có đúng `clip_id` và `text`. Dùng YAML resolved từ stage fine để
   giữ nguyên tham số model và pipeline.

## Trạng thái gate

**P0 synthetic đạt. P0 dữ liệu thật ở trạng thái HOLD.** Đầu vào còn cần gồm
manifest có quyền sử dụng, timestamp/FPS kiểm được, mapping và rig cùng pose
fit, và dev_gate đủ clip có event/tay quan sát. P1–P3 đang chờ dữ liệu và lịch
thực nghiệm. Giá trị loss trên fixture xác minh đường đi huấn luyện.

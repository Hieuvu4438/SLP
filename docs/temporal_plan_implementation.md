# Temporal-plan implementation — trạng thái bàn giao

**Đã hoàn tất phạm vi code và kiểm chứng CPU của goal.** Method theo [proposal](proposal2/07_main_method.md), hướng dẫn tại [README](../methods/temporal_plan/README.md), đối chiếu từng module tại [verification report](temporal_plan_verification.md).

## Các hạng mục hoàn tất

- [x] Raw133 ↔ effective rotations/expressions 256D; SOKE active-joint/shape/hand-mean convention và differentiable FK adapter.
- [x] Streaming train-only statistics, checkpointed fitted state, padding/odd-length invariants.
- [x] Four-part stride-two VAE, joint decoder, geodesic/expression/FK/velocity/KL objectives.
- [x] Segment score, eight-phase reconstruction distortion, exact partition normalizer/backward sampling/MAP/marginals.
- [x] Segment codec, content statistics, teacher plan batches và source-index alignment.
- [x] Pinned/frozen mT5 với adapters/language embeddings; complete-load validation và actual tiny-model tests.
- [x] Causal Gaussian-mixture/duration/stop planner, threshold/caps và EOS/cap reason.
- [x] Cell-integrated conditions, adjacent overlap và explicit training-only exposure rescaling cho K>L.
- [x] Joint temporal/cross-part/text denoiser, cosine DDPM và DDIM inference.
- [x] Full five-stage objectives, parameter freezing và gradients qua frozen decoder được kiểm tra riêng bằng auxiliary-only backward.
- [x] Data manifests/explicit Split normalization, source/split/FPS guards và whole-frame loading.
- [x] Five-stage optimizer loop, correctly weighted accumulation, validation selection/best weights/early stop và support diagnostics.
- [x] Atomic checkpoint/config/tokenizer persistence, corruption/partial-state checks và fresh-process inference load.
- [x] Bounded CPU/disk teacher-score cache có feature/config/code/version keys, source IDs và checksums; vectorized score calculation có independent reference.
- [x] CLI train/sample/inspect, BaseMethod bridge, config.yaml và optional dependencies.
- [x] CPU unit/integration/serialization/conformance verification và module-to-spec report.

## Kết quả thực chạy cuối

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests/temporal_plan tests/test_methods.py
```

**53 passed, 2 third-party deprecation warnings, 11.24s.** Bao gồm 47 tests của temporal_plan và 6 repository conformance tests. Warnings từ SWIG/sentencepiece import, không phải failures hoặc CUDA usage.

```bash
CUDA_VISIBLE_DEVICES='' python -m ruff check methods/temporal_plan tests/temporal_plan
python -m compileall -q methods/temporal_plan
```

Cả hai pass. CLI import/help không khởi tạo torch/model; sample subprocess kiểm tra CUDA chưa khởi tạo. Các model/optimizer tests chỉ dùng tensors nhỏ, actual tiny MT5 và official SMPL-X engine trên synthetic 100-vertex rig. Không job trên corpus thật, không GPU, không tải weights lớn.

Machine record và source hashes: [cpu_validation.json](temporal_plan_cpu_validation.json). Hashes ghi đúng code được kiểm tra, không tự thay semantic audit.

## Completion audit theo yêu cầu người dùng

| Yêu cầu | Evidence đã xem | Kết luận trong scope |
|---|---|---|
| Code method đã đề xuất | system.py nối đầy đủ codec→teacher→planner→joint diffusion, năm stages và text-only generation; config theo spec | Đã triển khai đầy đủ main method, không chỉ toy neural blocks |
| Mỗi module đúng đắn/hợp lệ | Bảng module-to-spec, independent math/library references, finite gradients, padding/causality/normalization/termination checks, full runner và file reload | Đã kiểm chứng phần mềm trong phạm vi tests; không tuyên bố correctness tuyệt đối cho mọi data/runtime |
| Có thể tham khảo ngoài/clone/download khi cần | Dùng installed official torch/transformers/smplx, public SOKE convention; pinned mT5 reference và explicit allow-download | Không cần clone hoặc tải weights lớn để hoàn thành coding/verification; không dùng silent fake backbone/FK |
| Giữ server không bị chiếm GB VRAM | Mọi execution tests khóa CUDA_VISIBLE_DEVICES rỗng, CPU tensors và hai host threads; sample subprocess assert CUDA uninitialized | Không khởi chạy GPU hoặc training/inference lớn |
| Kết quả dùng được trong repo | Dedicated CLI, strict manifest/asset contracts, checkpoint, typed BaseMethod bridge, documentation/dependencies | Train/sample workflow đã được kiểm tra bằng CPU fixtures; chưa tạo experiment results |
| Giữ thay đổi có sẵn | Implementation/tests nằm trong thư mục mới, shared dependency thêm ở pyproject.toml; spec có hand-mean clarification | Không sửa user changes trong shared data/metrics/tests hoặc methods khác |

## Giới hạn và việc thực nghiệm tương lai

- Muốn train primary configuration cần actual raw133 data, pretrained mT5 cache hoặc allow-download, và licensed SMPL-X assets. Goal này không thực hiện các jobs đó.
- Actual human assets/full mT5-Base chưa được tải/chạy. Official engine/API đã được kiểm tra bằng synthetic assets/small real architectures; không là validation anatomical quality hoặc comprehension.
- Checkpoint lưu stage/inference state; exact optimizer/RNG resume không phải API được cung cấp. Trainer từ chối giả resume.
- Generic classifier/joint/token metrics không nhận trực tiếp pose parameters. Bridge fail-fast với các metrics không tương thích; final benchmark cần explicit shared pose-to-joints/BT evaluator đúng contract.
- Efficacy, convergence, novelty và SOTA còn phải kiểm tra theo research plan; không được suy từ unit tests.

## No-progress/process audit

Hai lượt implementation đều là **progress**: core + CPU verification, sau đó full data/training/checkpoint/CLI integration và final audit. Không có blocker/verified wait còn phải xử lý. Các test/import sessions 46387, 30808, 41086, 76367, 46702, 41622, 80032, 21475, 90949 đã terminal; không có process do goal này tạo còn live. Các sessions của lượt trước cũng đã terminal.

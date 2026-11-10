# Temporal-plan implementation: module-to-spec audit

Scope: triển khai method trong [07_main_method](proposal2/07_main_method.md) và kiểm chứng tính đúng đắn phần mềm trên CPU. Không có training corpus thật, GPU execution, human study hoặc claim efficacy/SOTA. Code ở [methods/temporal_plan](../methods/temporal_plan/README.md).

## Đối chiếu nội dung

| Spec/module | Implementation | Evidence kiểm tra thực tế |
|---|---|---|
| Raw133/learning256 và active joints | geometry.py, representation.py | Known 90° rotations, identity/pi, round trips, gradient finite differences, zero lower body/root |
| Hand mean convention | representation.py, kinematics.py | Official SMPL-X engine vs analytic FK; effective rotation round-trip sau axis-angle wrap; nonzero hand means |
| Positive FK/shape/context | kinematics.py, losses.py | FK missing backend fails; actual official engine trên synthetic rig; five-stage runner với FK=1; asset-independent inference reload |
| Padding/odd lengths | tensors.py, codec.py | Invalid padding NaNs/large values không ảnh hưởng valid outputs/gradients; single-example vs padded-batch equality; stride-two ceil lengths |
| Train statistics | normalization.py, data.py, training.py | Streaming moments vs direct computation; persisted fitted counts; train-only means/caps; explicit inverse normalization |
| VAE codec và KL | codec.py, losses.py | Per-part/joint reconstruction graph, KL zero reference, finite gradients và frozen-decoder input gradients trong full pipeline |
| Scores/phases | segments.py | Endpoint/constant/linear sampling cases; vectorized interval scores vs interval-by-interval reference |
| Distribution trên partitions | PartitionDistribution | Exhaustive normalizer/probability sums/MAP/marginals; logZ gradient vs exact marginals; sampler frequencies và reproducibility |
| Cached teacher scores | cache.py | Same values, invalidation theo features/config/code/version; bounded CPU LRU, consumer mutation protection, checksum corruption rejection |
| Eight-phase content | SegmentCodec/teacher.py | Shape contract, gradients, source-index expansion, duration covers; segment representation/statistics dùng fixed q across downstream modes |
| Frozen mT5 | text.py | Actual tiny MT5: eval/frozen backbone, adapter/language gradients; offline actual loading transport; reject incomplete/unpinned frozen loads |
| Planner | planner.py | Mixture NLL vs torch.distributions; current/future target intervention; text/plan padding invariance; finite backward; duration/cap/EOS contracts |
| Temporal conditions | conditioning.py | Fractional overlap analytic reference, sum-to-one/nonnegative weights, K>L exposure, constant preservation, content gradients |
| Joint denoiser | denoiser.py | Per-part temporal/cross-part/text paths, valid mask invariance, plan intervention and finite gradients |
| DDPM/DDIM | diffusion.py | Cosine schedule monotonicity/identity; reverse step vs closed-form path; final x0/no clipping/single-step cases; masked training backward |
| All stages/exposure | system.py, training.py | One small CPU optimization step/window each stage; correct parameter freezing; 100% forced exposure fixture exercises K>L; trained-stage barriers |
| Batching/validation | training.py | Last short microbatch derivative equals full-example mean; out-of-support validation remains whole-length and unavailable rather than cropped; fixed budget fallback logged |
| File checkpoints | checkpoint.py | Save/load in fresh process, exact inference equality, immutable tokenizer hash, missing state/config/corruption rejection; pretrained weights excluded but pin reloaded |
| Primary inference | application.py, cli.py | Reference/gloss/motion fields changed to poison values without output changes; reference_latents monkeypatched to fail; no reference argument on public generate |
| CLI/exports | cli.py/application.py | Lazy import without torch, explicit train routing, CPU overriding accidental config CUDA, fresh-process sample, transactional output/duplicate rejection |
| Repo interface | method.py | BaseMethod conformance, no cross-method imports; normalized Split round-trip; checkpoint output convention honored; incompatible metrics fail before fit |

## Các quyết định làm rõ để code đúng

1. SOKE-compatible parameters có hand-pose mean trong official SMPL-X. Encode effective rotations, decode trừ mean; không đổi flat_hand_mean để che khác biệt. Offset buffer được checkpoint hóa.
2. 6D chart dùng hai rows đầu. Degenerate predictions được project bằng fallback orthogonal basis; fallback counts được export.
3. Geometry backend dùng retained native joints cho training loss. Đây không là mesh-regressed hand evaluator của SOKE; benchmark phải dùng pinned shared evaluator riêng.
4. Stage transitions freeze/eval đúng modules, clear stale gradients và tạo optimizer mới. Frozen decoder vẫn truyền gradient từ decoded losses tới generator.
5. Content codec/statistics được fit bằng sampled q độc lập teacher-mode ablation. Primary expected loss không bị đổi thành exact marginal likelihood.
6. Temporal membership tích phân theo cells để fractional exposure durations vẫn hợp lệ khi số units vượt canvas. Inference không có reference-based rescaling.
7. KL/reconstruction/latent objectives được normalize theo example/plan để microbatch accumulation không vô tình ưu tiên clip dài.
8. Training-derived max latent length và stop threshold nằm trong config/checkpoint. Cap-stop được phân biệt EOS và vẫn export mọi sample.
9. Validation objective có thể dùng reference vì là diagnostic cho training. Nó được đặt tên đúng; nếu không hỗ trợ toàn validation, không tính một subset thuận lợi rồi gọi full score.
10. Checkpoint lưu đầy đủ trainable state/statistics/pose means/readiness và tokenizer; frozen HF encoder có immutable reference. Exact optimizer/RNG resume chưa là API được cung cấp và bị từ chối thay vì giả resume.

## Ranh giới chứng minh

Các tests xác nhận số học, interfaces, gradients và nhiều invariants quan trọng, không chứng minh correctness cho mọi input có thể xảy ra. Actual licensed human assets và full pretrained mT5-Base weights chưa được tải/chạy; engine và loader API được kiểm tra bằng small real architectures/synthetic assets có reference độc lập. Người chạy experiments cần cung cấp các assets hợp lệ và kiểm tra dữ liệu/fitting theo benchmark contract.

Không run `scripts/run.py` để tạo paper numbers; method không tự score classification/joint/token metrics trên raw parameters. Không sửa các shared files đang có user modifications. Thay đổi shared duy nhất cho dependency setup là optional extra trong pyproject.toml; research spec có clarification về hand mean.

## Kết quả kiểm tra

Lệnh chuẩn và output mới nhất được ghi trong [implementation state](temporal_plan_implementation.md). CPU visibility bị khóa rỗng, threads giới hạn hai; các subprocess sample kiểm tra CUDA chưa khởi tạo. Không có job lớn hoặc GPU session do implementation goal tạo.

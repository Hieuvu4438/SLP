---
status: exploring
created: 2026-11-10
---

# CPR-Flow

Triển khai PyTorch theo [detailed implementation spec](../../docs/proposal4/CPR_Flow_Detailed_Implementation_Spec_vi.md) và [proposal](../../docs/proposal4/SLP_Flow_Diffusion_Research_Proposal_vi.md). Package độc lập nằm trong `methods/cpr_flow`, public CLI là `python -m cprflow.cli.<module>`. Không sửa các methods khác.

**Đã có code và kiểm tra kỹ thuật; chưa train/evaluate trên corpus thật.** Trạng thái từng milestone, logs, missing paths và giới hạn ở [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). Các runs fixture được gắn `synthetic=true`; không phải benchmark CSL-Daily/How2Sign/PHOENIX-2014T.

## Thiết kế được giữ nguyên

- Canonical296: body60, native-left90, native-right90, jaw6, expression50. Rot6D lưu hai hàng đầu; không flip tay trái, không pad face10 thành50.
- AE deterministic bốn part, stride2, latent128/part; replicate boundary theo true length tại mọi convolution; decoder fusion residual MLP512→256→512. Không KL/VQ.
- Frozen `google/mt5-base` encoder768, literal `[CSL]/[ASL]/[DGS]`; model/tokenizer revisions từ lock. Duration log-normal frames, input800, inference predicted length.
- Typed DiT384, 8 blocks, 6 heads, FFN1536; temporal attention, gathered local part windows ±4, global context mỗi block chẵn, text cross-attention và adaLN-Zero.
- Linear CFM `t=0` noise, `t=1` data; loss trung bình mỗi clip/bốn part. P5 thêm co-generated relation33, weight0.25. Geometry/cycle/lag auxiliaries theo thời gian, ramp và frequency compensation của spec.
- B0 riêng: epsilon DDPM cosine, indices0..1000, DDIM eta0. Euler/Heun/CFG ghi actual field calls và forward-equivalents.
- Teacher multi-positive InfoNCE cùng language và masked reconstruction. Semantic loss/guard chỉ dùng calibration dev hợp lệ; default tắt. C1 geometry-only và C2 feature-only có nhãn control riêng.
- AdamW, AMP, gradient clipping sau unscale, EMA, atomic checkpoints, strict resume, CPU/GPU DDP; worker prefetch không làm tiến sampler cursor của checkpoint.

`Method(BaseMethod)` là bridge cho SLP registry. `fit` yêu cầu các canonical manifest splits có `manifest_path`; `predict` chỉ nhận generation requests. Các CLIs/preprocessing/evaluation chuyên biệt nằm trong package này theo layout của detailed spec; `Method.predict` không đọc targets hay tính final metrics.

## Cài đặt và kiểm tra đã chạy

Từ root SLP, môi trường hiện có sẵn Torch/SMPL-X/Transformers:

```bash
python -m pip install lmdb imageio-ffmpeg
python -m pip install -e 'methods/cpr_flow[dev,eval]' --no-deps
OMP_NUM_THREADS=2 python -m pytest methods/cpr_flow/tests -q
python -m pytest -q
python -m ruff check methods/cpr_flow
```

Môi trường **đã kiểm tra**: Python3.13.5, Torch2.11.0+cu128, SMPL-X0.1.28; versions đầy đủ trong `requirements.lock.txt` và `sources.lock.json`. Python3.10/3.11 chưa được chạy trên máy này. Trên máy mới, chọn Torch build phù hợp trước rồi `pip install -e '.[dev,eval]'` từ thư mục này. Lock là snapshot môi trường đang dùng, gồm cả các packages của lab, không phải danh sách dependency tối thiểu.

Tất cả lệnh bên dưới chạy từ:

```bash
cd /home/haipd/SLP/methods/cpr_flow
```

Đã chạy source resolution thật:

```bash
python -m cprflow.cli.audit_sources --config configs/base.yaml --resolve-lock
```

Lock chứa resolved Git commits, SHA256 source files, HF dataset/model/tokenizer revisions, dependency-lock hash và neutral SMPL-X hash. Reference snapshots nằm trong `/source_snapshots/` bị gitignore; không vendoring code/assets upstream vào package.

## Assets và preprocessing dữ liệu thật

Hiện có local licensed neutral asset được trỏ qua `assets/smplx/SMPLX_NEUTRAL.npz`. Máy khác phải cung cấp asset của mình từ [SMPL-X](https://smpl-x.is.tue.mpg.de/); không commit/download tự động asset này. Pose release đúng là [LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data), không phải generator weights.

Cần hai metadata artifacts do lab cung cấp từ bằng chứng thật:

- `access_record.json`: mỗi canonical dataset ID có `authorized: true` và `reference` tới authorization hiện có. Không tự gửi yêu cầu quyền truy cập.
- `source_timing.json`: `revision` đúng lock; `datasets.<id>` có `verified`, `feature_fps`, `verification_method`, `evidence` (URL hoặc file+SHA256). Raw-video FPS không thay verified pose feature FPS.

```bash
bash scripts/download_authorized_data.sh --config configs/base.yaml
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase import
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase verify-timing
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase canonicalize
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase audit-geometry --samples-per-dataset 64
python -m cprflow.cli.fit_stats --config configs/data/joint3.yaml --kind motion-relation --split train
```

Đây là runbook cho dữ liệu thật, **chưa chạy thành công** do chưa có các LMDB/metadata nêu trên. Exact expected paths có trong `verification/asset_audit.json`.

LMDB read-only, trusted/hash-checked metadata pickle, process-local handles; processed/cache NPZ chỉ numeric và `allow_pickle=False`. Parser không truncate, mirror hoặc fill modality thiếu. Quarantine và long-train exclusions có JSONL/counts. Train statistics streaming, equal clip/equal dataset, second moments và floor0.01; không học statistics từ dev/test. Cache kiểm tra AE/stats/source/text revision và không cho trộn synthetic/real.

Resampling dùng SLERP và endpoint `round(duration*30)+1`, lưu timestamps thật. Nếu sourceFPS>30, policy **`quaternion_fir_v1`**: sign-continuous quaternion component FIR (Hamming, cutoff0.45×targetFPS, taps `2*ceil(4*sourceFPS/targetFPS)+1`, nearest boundaries), normalize quaternion, giữ endpoints, rồi SLERP; coefficients dùng cùng FIR rồi linear interpolation. Policy nằm trong preprocess hash và có high-frequency attenuation test. Đây là lựa chọn triển khai được ghi rõ cho phần anti-alias còn mở trong spec; linguistic retention vẫn cần audit thật.

Sau render, G0 review ở `runs/geometry_audit/review.json` phải chứa reviewer, hash asset/train manifest/source lock, 64 reviewed UIDs/corpus, và xác nhận native-fitting render, left/right mapping, facial basis. Code không tự tạo approval. Skeleton preview là diagnostic; native-video/fitting side-by-side review vẫn phải thực hiện khi dữ liệu có sẵn.

## Train, cache, resume và controls

```bash
python -m cprflow.cli.cache --config configs/runs/ae_joint3.yaml --kind geometry --splits train dev test
python -m cprflow.cli.train --config configs/runs/smoke.yaml --task ae
python -m cprflow.cli.train --config configs/runs/ae_joint3.yaml --task ae
python -m cprflow.cli.cache --config configs/runs/ae_joint3.yaml --kind text --splits train dev test
python -m cprflow.cli.train --config configs/runs/duration_joint3.yaml --task duration
python -m cprflow.cli.fit_stats --config configs/runs/ae_joint3.yaml --kind latent --split train
python -m cprflow.cli.cache --config configs/runs/ae_joint3.yaml --kind latent-relation --splits train dev test
python -m cprflow.cli.train --config configs/runs/baseline_joint3.yaml --task flow
python -m cprflow.cli.train --config configs/runs/baseline_joint3.yaml --task flow --resume runs/cfm_joint3_parent_s17/checkpoints/last.pt
python -m cprflow.cli.train --config configs/runs/baseline_cont_joint3.yaml --task flow --warm-start runs/cfm_joint3_parent_s17/checkpoints/step_200000_ema.pt
python -m cprflow.cli.train --config configs/runs/relation_joint3.yaml --task flow --warm-start runs/cfm_joint3_parent_s17/checkpoints/step_200000_ema.pt
```

Source lock/profile/stats/checkpoint paths phải khớp. `--resume` giữ raw weights/optimizer/scheduler/scaler/RNG/sampler/EMA; `--warm-start` lấy EMA và reset optimizer/scheduler, ghi parent hash/global step. Continuation configs từ chối chạy nếu thiếu một trong hai flag. P4→P5 yêu cầu relation-key ramp2k; P5→P5 semantic continuation giữ relation keys hoạt động.

```bash
torchrun --standalone --nproc_per_node=4 -m cprflow.cli.train \
  --config configs/runs/baseline_joint3.yaml --task flow \
  --set run.device=cuda --set train.microbatch_per_rank=2 --set train.accumulate=8
python scripts/experiment_matrix.py --output EXPERIMENT_MATRIX.json
```

Matrix chỉ chứa **planned commands**, không chứa kết quả: B0 parent+continuation, B1-pre/B1-cont/B2/B3/C0, C1/C2/C3, equal-added-update no-semantic control, Mono×3, OracleT/OracleR; seeds17/42/123 và shared s17 AE/text/stats. B3 dùng P5 stream33 target random ổn định theo UID, không chứa relation labels; cùng tensor counts/parameter count với C0.

Không auto launch confirmatory training. Full config cần real teacher checkpoint và approved calibration; khi thiếu, thất bại rõ ràng. Không đổi một run full thành semantic0/guard-off rồi vẫn gọi full.

## Sampling, evaluation và render

Prompts JSONL: `{request_id,text,sign_language,seed}`; optional `duration_seconds` được gắn `duration_source=user`. Main API không nhận motion/GT length/gloss/keyframes. Mỗi request có RNG riêng; noise được tạo ở true shape.

```bash
python -m cprflow.cli.sample --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt --text-file prompts/asl_demo.jsonl --output runs/demo --nfe 16 --cfg 1.0
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt --dataset csl_daily --split dev --duration-source predicted
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt --dataset how2sign --split dev --duration-source predicted
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt --dataset phoenix2014t --split dev --duration-source predicted
python -m cprflow.cli.profile --config configs/runs/baseline_joint3.yaml --warmup 100 --updates 200
```

Output NPZ physical296/timestamps + JSON provenance/costs/failures. Geometry metrics recompute từ cùng neutral FK; exact min-sum DTW, path-average cost, deterministic ties, global/relative hand riêng. Relation metrics dùng pool-after-relation và train normalization; không chấm relation head như physical GT. Duration, angular velocities, high-frequency fraction, lag KL/coverage, bootstrap theo prompt/source video có labels/units rõ.

Official BT/SiBLEU, physical-contact F1/penetration, independent sign-FID và human comprehension được báo **unavailable** khi chưa có audited evaluator/annotations, không ghi0. Retrieval/FID utilities không tự biến train teacher thành independent evaluator. Final test CLI yêu cầu explicit `--final-test`, clean committed tree và access ledger của SLP; phiên triển khai này không mở test thật.

Controls cùng checkpoint có `--guard-mode off|unrestricted|feature_only|semantic`; C1 không dùng teacher và không ghi semantic score giả. C2/C3 cần `--teacher-checkpoint` và `--teacher-calibration` thực, hashes khớp. Full semantic guard dùng bounded backtracking2×3, original-reference budgets và exact rollback.

Oracle diagnostics chỉ ở evaluator: `--duration-source oracle --diagnostic-only` hoặc `--relation-source oracle --diagnostic-only`. OracleR protocol được đặt tên rõ: GT normalized relation được linear-resample theo normalized time tới **predicted** latent length, rồi clamp relation state theo linear noise→GT endpoint path trong ODE. Không tự cấp GT length; report `oracle=true` và loại khỏi main aggregation.

```bash
python -m cprflow.cli.render --motion runs/demo/<sample>.npz --smplx-model assets/smplx/SMPLX_NEUTRAL.npz --output runs/demo/preview.mp4
python -m cprflow.cli.aggregate runs/<run>/metrics/csl_daily_dev_predicted.json runs/<run>/metrics/how2sign_dev_predicted.json runs/<run>/metrics/phoenix2014t_dev_predicted.json --output runs/<run>/metrics/macro.json
```

Aggregation từ chối oracle/main mix, khác checkpoint/profile/fitting/stats/evaluator/seeds/split/correction mode và synthetic/real; không pooled BLEU xuyên ngôn ngữ.

## Engineering reproduction đã chạy

```bash
OMP_NUM_THREADS=2 python scripts/synthetic_integration.py --output runs/verified_pipeline --smplx-model assets/smplx/SMPLX_NEUTRAL.npz
OMP_NUM_THREADS=2 python scripts/verify_fixture_evaluation.py --checkpoint runs/verified_pipeline/runs/cfm_fixture/checkpoints/last.pt --output runs/verified_pipeline/evaluation --summary verification/evaluation_fixture_summary.json
OMP_NUM_THREADS=2 python scripts/verify_runtime.py --output verification/runtime_cpu.json
```

Lệnh integration phải dùng output directory mới. Fixture AE2 + duration2 + teacher2 updates, CFM4 + relational1 updates, exact resumed vs uninterrupted CFM, actual geometry/cycle backward và deterministic physical sample. Fixture text training features là random768D, **không** giả là mT5. Bước evaluation riêng chạy encoder mT5 thật trên synthetic prompts và kiểm tra routing ba corpus.

Hardware profile đã đo: **CPU,2threads, synthetic B1 T16**, main DiT384/8/6; detailed timings/parameter counts ở `verification/runtime_cpu.json`. Đây không phải throughput real clips hay GPU profile100+200. Máy có RTX5880 Ada48GB, nhưng CUDA training/NCCL2GPU chưa được kiểm tra. Không suy training-hours/SOTA/linguistic quality từ các checks này.

Assets/weights/raw data/runs không commit. Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) về upstream references và điều kiện assets.

## Job scheduler và chế độ tiết kiệm dung lượng

Xem [JOBS.md](JOBS.md): DAG đầy đủ, cap2 payload processes, shared caches, checkpoint retention, automatic resume và gates dữ liệu/calibration/final-test. CLI quản lý: `python -m cprflow.cli.jobs`. Budget mặc định100GiB cho data+training outputs, reserve15GiB filesystem; không tự xóa raw data hoặc chạy full-test khi source chưa sạch.

status: exploring

# Structured temporal-plan supervision

Đã triển khai phương án B trong [proposal](../../docs/proposal2/07_main_method.md): motion codec → distribution trên các phân đoạn → text-to-plan → joint diffusion. Có đầy đủ năm stage, data loader, checkpoint, CLI và BaseMethod bridge. Trạng thái nghiên cứu vẫn exploring: chưa train trên corpus thật hoặc xác nhận chất lượng/SOTA.

## Sử dụng

Chạy từ thư mục gốc SLP. Dependencies nằm trong optional extra:

```bash
pip install -e '.[dev,temporal-plan]'
python -m methods.temporal_plan.cli --help
```

Các lệnh train/sample dưới đây là hướng dẫn; không có job lớn nào đã chạy trong quá trình triển khai.

```bash
python -m methods.temporal_plan.cli train \
  --train /path/train.jsonl --val /path/val.jsonl \
  --smplx-model-dir /path/models --allow-download \
  --output /path/new_run --device cpu

python -m methods.temporal_plan.cli inspect --checkpoint /path/new_run/final.pt

python -m methods.temporal_plan.cli sample \
  --checkpoint /path/new_run/final.pt --inputs /path/texts.jsonl \
  --output /path/new_predictions --device cpu
```

Mặc định luôn CPU. GPU chỉ được chọn nếu người chạy truyền tường minh device; trong goal này không dùng GPU. `--allow-download` cho phép tải mT5 tại revision đã pin; khi không có flag, chỉ dùng local cache. Default config tại [config.yaml](config.yaml), có thể chọn file khác bằng `--config` khi train.

Primary FK loss mặc định 1: cần official licensed SMPL-X assets, ví dụ `/path/models/smplx/SMPLX_NEUTRAL.npz`. Trường hợp FK-disabled ablation phải đặt loss weight 0 và cung cấp `--rotation-offsets` là file `[41,3]` đã biết đúng quy ước; không tự thay FK bằng zeros. Inference xuất parameters không cần load lại SMPL-X assets, vì pose means/stats đã trong checkpoint.

## Data contract

Một dòng JSONL train/val:

```json
{"name":"clip_001","text":"Tomorrow it rains","lang":"en_ASL","source_id":"original_recording_001","fps":25,"motion":"motions/clip_001.npy","split":"train","format":"raw133"}
```

`motion` là đường dẫn tương đối so với manifest hoặc absolute path, trỏ `.npy` float `[T>0,133]`. Các tags hỗ trợ: en_ASL, zh_CSL, de_DGS. Khai đúng original recording source ID để kiểm tra train/val overlap. FPS phải được chuẩn hóa/khai báo nhất quán trong mỗi ngôn ngữ trước training; loader giữ nguyên FPS và toàn bộ frames, không tự áp MaDiS crop hoặc uniform length resampling.

133D phải là **raw SOKE-compatible SMPL-X parameters trước statistical normalization**: body 0:30, LH 30:75, RH 75:120, jaw 120:123, expression 123:133. Có thể tạo raw arrays từ pose dictionaries có sẵn bằng shared `slp.data.sign_smplx.feats133`, trước khi normalize/crop. Quy ước hand-pose mean phải tương ứng assets của fitting; representation mã hóa effective rotations rồi trừ mean khi export về raw parameters.

Text-only inference JSONL chỉ cần:

```json
{"name":"new_sentence_001","text":"Tomorrow it rains","lang":"en_ASL"}
```

Prediction không đọc motion, gloss, reference length hoặc keyframes nếu các trường đó có trong input. Từng mẫu có RNG seed ổn định theo run seed/language/name, không phụ thuộc input ordering. Output gồm raw133 `.npy`, predicted durations, frames, FPS từ training metadata, EOS/cap reason và fallback diagnostics. Tên file hash theo sample ID; output transaction không overwrite hoặc để lại partial run.

## Stages và checkpoint

1. Fit motion statistics chỉ trên train; train motion VAE.
2. Chọn codec checkpoint, fit latent statistics train-only; train eight-phase segment codec.
3. Fit content statistics train-only; train causal content/duration/EOS planner.
4. Freeze codec/planner/text; train joint diffusion với teacher plans.
5. Exposure training với 25% reference examples dùng plans tự dự đoán, normalize temporal condition chỉ trong training.

Gradient accumulation cân đúng theo số examples, gồm microbatch cuối ngắn. Validation objective là reference/teacher-conditioned diagnostic, không phải text-only quality. Chọn best objective khi toàn validation có support; nếu có mẫu dài hơn training-derived cap, báo rõ unavailable cho toàn objective và dùng fixed-budget last epoch, không crop/drop riêng mẫu đó.

Checkpoints lưu config, pose means, normalization/counts, train-derived cap, stop threshold, stage readiness, data/code/FK provenance và tokenizer có hash. Frozen pretrained backbone được tham chiếu bằng model ID/revision, không lưu lặp weights. Khi di chuyển checkpoint, giữ tokenizer directory đi kèm và local cache đúng revision, hoặc bật `--allow-download` để tải lại backbone. Checkpoint là stage/inference snapshot; chưa hỗ trợ exact resume optimizer/RNG. Trainer từ chối coi checkpoint loaded là fresh fit.

## Repository bridge và evaluation

[method.py](method.py) có `fit/predict/save/load` theo BaseMethod. Với existing Split, phải khai tường minh `target_format`, `target_preprocessing`, source key và FPS; normalized133 cần 133D mean/std `.npy` để hoàn nguyên. Bridge không khôi phục những frames đã bị upstream crop. Standalone manifest CLI là đường dùng cho full sequences.

Method trả pose parameters, nên default accuracy/macro-F1, joint-coordinate DTW hoặc token SiBLEU trong generic runner không nhận trực tiếp output này. Bridge báo lỗi trước expensive training nếu cấu hình nhầm các metrics đó. Dùng shared evaluation với pose-to-joints/BT adapter đúng protocol; method không tính final benchmark metrics hoặc đọc test targets.

## Modules và verification

| Thành phần | Files |
|---|---|
| Geometry/convention | geometry.py, representation.py, kinematics.py |
| Mask/statistics | tensors.py, normalization.py |
| Motion/segment codecs | codec.py, segments.py |
| Teacher | teacher.py, cache.py |
| Planner/text | plans.py, planner.py, text.py, text_model.json |
| Realization/losses | conditioning.py, denoiser.py, diffusion.py, losses.py |
| Full execution | system.py, data.py, training.py, checkpoint.py, application.py, cli.py, method.py |

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests/temporal_plan tests/test_methods.py
python -m ruff check methods/temporal_plan tests/temporal_plan
```

[Verification report](../../docs/temporal_plan_verification.md) đối chiếu từng module với specification và independent checks. Tests dùng CPU/tensors nhỏ, actual tiny MT5 và official SMPL-X engine trên rig tổng hợp 100 vertices; chưa dùng human assets hoặc pretrained mT5-Base weights. Các kiểm tra xác nhận thuật toán/plumbing/invariants trong phạm vi tests, không chứng minh hội tụ hoặc sign comprehension.

Nguồn: [SOKE convention audit](../../docs/proposal2/evidence/S01_soke.md), official [transformers mT5](https://huggingface.co/docs/transformers/model_doc/mt5), [SMPL-X](https://github.com/vchoutas/smplx), PyTorch APIs. Không copy implementation của method nội bộ khác. Môi trường đã kiểm tra: Python 3.13.5, torch 2.11.0+cu128 chạy CPU, transformers 5.5.4, smplx 0.1.28.

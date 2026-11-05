# Audit mã nguồn phục vụ proposal

Ngày kiểm tra 04/11/2026. Chỉ đọc code và đối chiếu; không chạy training/inference baseline, không sửa source của các phương pháp. Đọc checkout sẵn có không đồng nghĩa xác nhận mọi dependency/asset đang đủ để tái lập.

## 1. Snapshots

| Repository | Commit đã đọc | File/hàm trọng tâm |
|---|---|---|
| [SOKE](https://github.com/2000ZRL/SOKE) | `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` | `mGPT/metrics/dtw.py`: `dtw`, `l2_dist_align`; `mGPT/metrics/t2m.py`: `update` |
| [MaDiS](https://github.com/2000ZRL/MaDiS) | `071828ea1b94530dc7ff0a6c009c0a7c6d1f0784` | `mGPT/archs/masked_diffusion.py`: UTC/generate; `mGPT/archs/madis.py`: tokenizer/EOS; `mGPT/metrics/generation.py`; `mGPT/metrics/dtw.py` |
| [SignSparK](https://github.com/JianHe0628/SignSparK) | `916402d0ecd6322fb4a9e5f513229a94a57bf4b6` | `signspark/rflow.py`: interpolation/training/sampler; `signspark/keyframes.py`; README |
| [MoMask](https://github.com/EricGuo5513/momask-codes) | `94a6636c9c463b7a9414c3401a6f1b67e6c51824` | `models/vq/residual_vq.py`: residual/dropout; `models/mask_transformer/transformer.py`: `generate` |

SOKE/MaDiS/SignSparK có sẵn trong `third_party`; MoMask được clone shallow trong đợt này. Các untracked assets của MaDiS không được coi là source modifications hay bằng chứng checkpoint provenance. Khi tái lập phải hash asset riêng. Không dùng kết quả của pipeline nội bộ SLP.

## 2. SOKE: tên biến có PA nhưng hàm đang dùng translation alignment

Trong [`t2m.py` đoạn evaluation](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/t2m.py#L147), comment giải thích `align_idx=0` là DTW-JPE, `None` là DTW-PA-JPE, nhưng tên state vẫn có `PA`. Code đang gọi `align_idx=0`. [`dtw.py`](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/dtw.py#L84) dịch root của generated về reference trước tính mean joint Euclidean distance; nếu `None` thì gọi `rigid_align`.

Hàm `dtw` trả accumulated minimum path cost; không normalize bởi path length ở call site đã đọc. Hands lấy từ mesh hand regressors và align wrist độc lập. **Hệ quả:** cần nhìn function arguments và aggregation, không chọn metric bằng suffix tên log. Tái lập PA cần cấu hình và kiểm tra transform thực tế, không chỉ rename key.

## 3. MaDiS: mismatch SiBLEU-4

Paper §3.4/§4 gọi SiBLEU-4. Nhưng [`generation.py` tại commit audit](https://github.com/2000ZRL/MaDiS/blob/071828ea1b94530dc7ff0a6c009c0a7c6d1f0784/mGPT/metrics/generation.py#L16):

```python
predicted = Counter(prediction)
expected = Counter(reference)
overlap = sum((predicted & expected).values())
```

`corpus_sibleu` lấy clipped unigram precision nhân brevity penalty và 100, không tạo n-grams 2/3/4 hay geometric mean bốn precision. Đối chiếu raw GitHub của **đúng commit** trả HTTP 200, bytes giống file local; SHA256 file: `ecb391f2edaeaa4f91729bb6ef105ea72399e54bc0d95011ffdbf28300f34c90`.

Kiểm tra diễn giải bằng dãy reference `[1,2,3,4]`, prediction `[4,3,2,1]`: unigram overlap bằng 1.0 trong khi 4-gram overlap bằng 0. Đây là khác biệt định nghĩa có thể chứng minh từ code, **không phải** kết quả tái tính trên benchmark hay cáo buộc về số liệu paper.

Action khi implementation: pin released-code metric để tái lập release; đồng thời tính genuine token BLEU-4 với tokenization/aggregation công bố. Đối với continuous output, re-encode bằng **cùng frozen SOKE tokenizer** để token-space score có nghĩa; không lấy tokens của coarse prior để chấm final refined motion. Nếu sử dụng tokenizer mới, không so trực tiếp với SiBLEU cũ.

## 4. SignSparK: path derivative và release drift

[`rflow.py` training](https://github.com/JianHe0628/SignSparK/blob/916402d0ecd6322fb4a9e5f513229a94a57bf4b6/signspark/rflow.py#L220) xây `keyframed_x0`, interpolation giữ observed keyframes, rồi dùng:

```python
vf_gt = x_1 - keyframed_x0
x_1_est = x_t_masked + (1 - t) * vf_pred
```

Với path tuyến tính, đạo hàm ở unobserved frames là `x_1 - x_0`; ở observed frames bằng zero. Công thức paper Eq. 3 viết target dưới dạng `x_1 - C_t`; nhìn chung khác đạo hàm trên. Proposal dùng công thức path-consistent ở [05](05_method_pact_sign.md), không sao chép mismatch này.

Sampler ở file đó tái áp observed mask; release theo stream. README thông báo processed data, checkpoints và BT khác thiết lập paper; released model còn có training-corpus policy riêng. Cần ghi rõ paper-version hay release-version. Không có cơ sở để quy kết mismatch notation làm vô hiệu kết quả; cần tái lập mới kết luận được.

## 5. MoMask: hiểu generation thực tế

[`residual_vq.py`](https://github.com/EricGuo5513/momask-codes/blob/94a6636c9c463b7a9414c3401a6f1b67e6c51824/models/vq/residual_vq.py#L99) quantize residual qua từng layer và có quantization dropout. [`MaskTransformer.generate`](https://github.com/EricGuo5513/momask-codes/blob/94a6636c9c463b7a9414c3401a6f1b67e6c51824/models/mask_transformer/transformer.py#L328) nhận `m_lens`, cosine mask schedule, categorical sampling và confidence-based giữ/mask tokens.

**Hệ quả:** không thể chỉ port transformer rồi dùng GT motion length khi test text-only SLG. RVQ cần residual generator và decoder tương thích; không thêm vài token heads vào tokenizer SOKE rồi gọi đó là RVQ đúng paper.

## 6. Ranh giới bằng chứng

Audit xác minh các cơ chế/file đã nêu và phát hiện protocol risks. Chưa kiểm tra training convergence, assets, weight conversion, license compatibility cho một derivative implementation hay reproduced scores. Không yêu cầu cài environment/khởi chạy GPU chỉ để hoàn thành literature proposal. Các kiểm tra runtime này thuộc Stage 0/1 của kế hoạch thực nghiệm.

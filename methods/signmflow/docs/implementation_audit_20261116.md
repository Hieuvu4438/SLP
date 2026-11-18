# Đối chiếu implementation SignNMFlow với paper

## Material Passport

- Ngày kiểm tra: 2026-11-16.
- Paper: He et al., *Naturalness-guided Manifold Flow Matching for Sign Language Production*, arXiv:2609.33339v1, 25 trang. [Bản online](https://arxiv.org/pdf/2609.33339).
- Bản PDF đối chiếu: [2609.33339v1.pdf](2609.33339v1.pdf); SHA-256 `4de7963d4d11cc1e848ec18f3dca7b2ae4d66e1024d242fb301afbb43e37e153`.
- Phần dùng để kiểm tra: Sec. 3–4, Algorithms 1–2, Appendices C–D và Tables 1–8; không thực hiện citation audit cho danh mục tài liệu tham khảo.
- `ref_slug`: `signnmflow_2609_33339v1`; PDF preflight: **PASS**, số trang khai báo/parser/reader đều là 25.
- Phạm vi: toàn bộ Python/config/README của method, scripts thí nghiệm, tests của method, data loader/packing, metric/FK dùng chung và config/log/metrics của các run hiện có. Đối chiếu thêm source SOKE có sẵn trong `third_party/SOKE`.
- Trạng thái: **ANALYZED** về độ trung thành với paper; lỗi K-means và số tham số được kiểm chứng riêng bằng CPU. Chưa tái lập kết quả huấn luyện/test của paper.
- Không sửa implementation/config/checkpoint, không chạy lại training, không chấm test, không can thiệp các tiến trình đang chạy. Báo cáo này là artifact mới duy nhất trong repository.

## Kết luận

**Chưa đủ cơ sở khẳng định đạt hoặc tái lập được kết quả paper.** Vấn đề rõ nhất là lỗi K-means; tiếp theo là các khác biệt về dữ liệu, loss weighting, độ dài sinh và protocol đánh giá. Các công thức manifold/naturalness/sampler chính và phần lớn hyperparameter Table 6 khớp mô tả công khai.

Báo cáo chia thành **I: các điểm ảnh hưởng hoặc có khả năng ảnh hưởng kết quả**, **II: các khác biệt vận hành không trực tiếp làm thay đổi metric chất lượng trong điều kiện nêu rõ**, và phụ lục các phần đã khớp/kiểm chứng. Phần I tách riêng bằng chứng chắc chắn với chi tiết tác giả chưa công bố. “Có thể ảnh hưởng” không có nghĩa đã chứng minh nguyên nhân hay mức giảm metric. Test pass không chứng minh tái lập được kết quả paper.

## I. Các điểm liên quan đến việc đạt hoặc đối chiếu kết quả paper

### I.1. Tổng hợp theo loại ảnh hưởng

| Điểm cần xử lý/xác nhận | Đường ảnh hưởng | Mức chắc chắn | Bảng liên quan |
|---|---|---|---|
| K-means dừng sai | Thay prototypes → measure → correction → generator | Bug đã kiểm chứng; chưa đo mức ảnh hưởng DTW/B-T | 1–5, 8 |
| How2Sign preprocessing khác SOKE local | Khác dữ liệu/frame train và reference evaluation | Khác biệt với SOKE đã xác nhận; protocol chính xác của tác giả chưa rõ | How2Sign ở 2 và mọi kết quả multilingual |
| Loss reduction pose/expression | Có thể thay đổi tương quan gradient giữa hai mục tiêu | Cách code tính đã rõ; paper thiếu reduction chi tiết | 1–5, 8 |
| Loss weighting khi đổi microbatch | Khác trọng số valid frames giữa các nhóm sequence | Cơ chế toán học đã rõ; chưa xác nhận cách normalize của code tác giả | 1–5, 8 |
| Độ dài motion lúc inference | Thay length, context của generator và DTW tích lũy | Length head ngoài mô tả paper; protocol tác giả chưa rõ | 1–5, 8 và latency ở 7 |
| “All” và PA-body | Có thể thay chính giá trị metric với cùng predictions | Chưa xác nhận định nghĩa đầy đủ của paper | 1–5, 8 tùy cột |
| Split và B-T evaluator khác | Số đo chưa cùng tập/reference/model với paper | Khác biệt đã xác nhận | 1–2; các bảng khác khi đang chấm dev |
| Chi tiết architecture/text encoder chưa xác nhận | Có thể thay capacity, conditioning và tối ưu hóa | Số params lệch đã xác nhận; chưa biết nguyên nhân/cách đếm tác giả | Chất lượng 1–5, 8: chưa rõ; #Params ở 7: lệch trực tiếp |
| Model selection, seed/resume và artifact version | Khác checkpoint/trajectory hoặc dùng lại kết quả của code cũ | Giới hạn code đã rõ; paper thiếu một số quy tắc | Các bảng được tái lập |
| Hardware/timing boundary benchmark | Thay latency báo cáo, không chứng minh thay accuracy | Hardware/measurement setup chưa đồng nhất | Chỉ latency Table 7 |

**Không coi số GPU là một nguyên nhân độc lập làm metrics hiện tại kém hơn paper.** Chạy một GPU với accumulation có thể thay cho nhiều GPU nếu cùng training semantics. Vấn đề loss weighting do cách chia batch được phân tích riêng bên dưới. Lỗi DDP nằm ở phần II vì các run đang xét dùng một process; nếu thực sự chạy `torchrun` thì lỗi này phải được sửa trước khi coi các process là một training job đồng bộ.

### I.2. K-means dừng sớm ngay vòng đầu

Vị trí: [naturalness.py:56](/home/haipd/SLP/methods/signmflow/naturalness.py:56).

Code đặt `prev = math.inf`, sau đó kiểm tra:

```python
abs(prev - inertia) <= tol * max(prev, 1e-12)
```

Nếu vòng đầu không có empty cluster, hai vế đều là `inf`, và Python trả `True` cho `inf <= inf`. Do đó vòng lặp dừng sau một cập nhật centroid. `kmeans_iters: 50` không đảm bảo chạy tối ưu K-means như dự định. Nếu có empty cluster ở vòng đầu, điều kiện có thể bị chặn và lỗi không nhất thiết xuất hiện theo đúng dạng một vòng.

Kiểm chứng CPU với 2.000 quaternion, K=8, seed=7:

```text
stop_initial True
iters1==iters50 True
counts_equal True
centroid_residual_max 0.17907646298408508
inertia_returned 0.30103734135627747
inertia_after_one_more 0.28268569707870483
```

Một cập nhật Lloyd bổ sung tiếp tục giảm inertia: đây là bằng chứng dừng sai, không chỉ vấn đề tên biến hay số vòng cấu hình.

**Ảnh hưởng:** prototypes/frequency weights, measure, correction và generator học theo correction đều có thể bị ảnh hưởng; đặc biệt Tables 1–5/8 và ablation K. Chưa có bằng chứng định lượng lỗi này giải thích bao nhiêu chênh lệch metric.

**Cách sửa đề xuất:** chỉ kiểm tra convergence khi `prev` hữu hạn hoặc bắt đầu từ vòng thứ hai. Thêm kiểm tra dữ liệu chưa hội tụ mà `iters=1` và `iters>1` phải có thể cho kết quả khác. Sau khi sửa, cần tạo lại prototypes → correction → generator cho các kết quả muốn công bố; không coi checkpoint cũ là kết quả của thuật toán đã sửa.

### I.3. Kiến trúc/text encoder: chưa xác nhận tương đương; #Params khác trực tiếp Table 7

Paper Table 7 báo **233,20M**. Đếm trực tiếp kiến trúc hiện tại bằng meta tensors, không load weight lên GPU:

| Thành phần | Số tham số |
|---|---:|
| mBART encoder, vocab 19.011 của checkpoint đang dùng | 171.676.672 |
| SignGenerator phần ngoài text encoder, có length head | 100.684.463 |
| Generator tổng | **272.361.135** |
| Correction | 515.748 |
| Tổng hai network | **272.876.883** |
| Length head trong generator | 525.313 |

Log run Phoenix cũng ghi `generator parameters: 272.36M`. Bỏ length head vẫn không giải quyết được chênh lệch khoảng 39M.

Vị trí: [model.py:57](/home/haipd/SLP/methods/signmflow/model.py:57), `model.py:112`, `benchmark.py:51`. Riêng các decoder layers chiếm 94.527.488 tham số; cross-attention dùng text dimension 1024 và mỗi StylizationBlock có projection từ time dimension 2048.

**Kết luận chắc chắn:** số tham số của bản này khác số paper công bố. **Chưa thể kết luận:** tác giả dùng đúng kiểu projection/block nào hoặc phạm vi đếm tham số nào. Paper không cung cấp đủ chi tiết để tự suy ra một kiến trúc duy nhất đạt 233,20M. Cần làm rõ counting convention và text/block implementation trước khi gọi Table 7 đã tái lập.

Với **DTW/B-T**, số tham số lệch chỉ là dấu hiệu cần kiểm tra kiến trúc; không phải bằng chứng model lớn hơn làm metric kém hơn. Chỉ khi xác nhận khác block/weight/text processing mới xác định được khác biệt mô hình cụ thể cần xử lý. Với **#Params Table 7**, chênh lệch này trực tiếp làm kết quả không khớp, không phụ thuộc giả thuyết về accuracy.

### I.4. Protocol đánh giá chưa đủ để đối chiếu số paper

#### Split và kết quả hiện có

Các run main kiểm tra đều là **val, seed 0**. Queue mặc định chạy val; shell scripts cho phép `SPLIT=test`. Đây là lựa chọn đúng cho giai đoạn phát triển theo protocol repository, nhưng chưa phải kết quả test như paper.

Run Phoenix đã hoàn tất: [metrics.json](/home/haipd/SLP/outputs/runs/signmflow_main_phoenix_smplx/signmflow/seed0_20261116-031211-761164/metrics.json).

| Metric Phoenix | Bản hiện tại, dev | Paper Table 1, test |
|---|---:|---:|
| DTW-JPE body | 5,4107 | 4,37 |
| DTW-JPE hand | 7,4620 | 5,57 |
| DTW-PA-JPE body | 6,1756 | 3,85 |
| DTW-PA-JPE hand | 1,4161 | 1,07 |

**Các số không trùng nhưng cũng không phải phép so sánh cùng split/protocol.** Không dùng bảng này để kết luận khoảng cách test hoặc quy nguyên nhân cho một lỗi cụ thể. Các run CSL/How2Sign được đọc log trong audit chưa có final metrics; không diễn giải training loss thành kết quả paper.

#### “All” và PA body

Vị trí: [metrics.py:190](/home/haipd/SLP/src/slp/evaluation/metrics.py:190).

- `all` lấy mean trên toàn bộ 144 joint output SMPL-X. Paper không định nghĩa đủ joint set/aggregation cho “All” của Tables 3/4/5/8; không được khẳng định metric này tương đương cột paper chỉ vì tên giống nhau.
- PA body chính fit Procrustes trên toàn bộ joint, rồi đo subset 12 upper-body joints. Cách này theo utility/cách truyền subset của SOKE local, nhưng tập joint tối ưu alignment khác tập joint tính error.
- `soke_dtw_pa_jpe_body_ua` fit trên chính 12 upper-body joints; đây là một protocol khác. Số thấp hơn/PA < JPE chưa chứng minh tác giả dùng protocol đó.
- SOKE local mặc định đang dùng `align_idx=0` để chấm JPE; utility hỗ trợ `align_idx=None` cho PA. Không có xác nhận độc lập ở audit rằng tác giả SignNMFlow bật PA theo đúng cách nào.

Không nên đổi primary metric hoặc chọn variant gần số paper nhất rồi coi là xác nhận. Cần cố định joint set, alignment set, DTW aggregation/unit và reference implementation. Phần FK/hand regressors hiện có cơ sở từ SOKE; kiểm tra metric riêng không thay thế được xác nhận protocol tác giả.

#### B-T chưa tương đương paper và chỉ chấm dev

Paper Sec. 4.1 yêu cầu cùng back-translator với SOKE. Script hiện tại dùng back-translator mBART tự huấn luyện của repository; README đã công bố khác biệt này.

Còn một giới hạn pipeline: [train_backtranslator.py:324](/home/haipd/SLP/scripts/train_backtranslator.py:324) cố định reference ở dev/val; [score:285](/home/haipd/SLP/scripts/train_backtranslator.py:285) chỉ giữ predictions có name thuộc reference này. Vì vậy chuyển `SPLIT=test` ở script sinh pose **không** làm B-T scorer chấm test; có thể mất mẫu hoặc không còn mẫu hợp lệ. Chưa có kiểm tra coverage đầy đủ tại đoạn lọc này.

**Đề xuất:** scorer cần split/reference rõ ràng, kiểm tra số lượng/key coverage và model/version của back-translator. Nếu vẫn dùng model nội bộ thì ghi rõ là B-T nội bộ, không so trực tiếp BLEU với paper.

### I.5. Preprocessing How2Sign khác SOKE

Vị trí: [sign_smplx.py:65](/home/haipd/SLP/src/slp/data/sign_smplx.py:65), [sign_smplx.py:94](/home/haipd/SLP/src/slp/data/sign_smplx.py:94); đối chiếu `third_party/SOKE/mGPT/data/humanml/dataset_t2m.py` và `load_data.py`.

- Crop/resample về 40..400 frames, bội số 4, nhìn chung khớp SOKE config và dataset code local. Đây là bước tiền xử lý kế thừa; paper không liệt kê toàn bộ chi tiết này.
- SOKE loại clip có duration ≥30 giây và các `bad_how2sign_ids`. Loader hiện tại giữ mọi dòng có pose file, không thực hiện hai bộ lọc này.
- Kiểm tra **metadata train/val**, không chấm test: train hiện có 30.965 mẫu, gồm **254** mẫu duration ≥30 giây và **27** ID nằm trong bad list SOKE; val 1.724 mẫu, gồm **5** mẫu dài và **2** bad ID. Hai nhóm có thể giao nhau; không cộng số để suy ra số mẫu cần bỏ.
- Downsampling fps >24 hiện dùng `linspace(0, n−1, count)`. SOKE dùng `floor(i*n/count)`. Hai cách không đồng nhất, ví dụ n=120/count=96 khác một index.

**Kết luận:** đã xác nhận khác preprocessing SOKE local; chưa thể khẳng định paper sử dụng mọi bộ lọc y hệt source SOKE này vì thiếu code chính thức. Muốn tuyên bố theo SOKE phải giải quyết và ghi rõ khác biệt này, nhất là với SignNMFlow-M.

### I.6. Loss, inference length và những chi tiết có thể ảnh hưởng nhưng paper chưa nói đủ

| Lựa chọn hiện tại | Cơ chế ảnh hưởng và điều cần xác nhận |
|---|---|
| Length head: log-length L1 từ mean-pooled detached text; clamp 40..400 | Ngoài paper. Detach ngăn length loss cập nhật text/generator qua nhánh này, nhưng length được dự đoán dùng trực tiếp ở inference, thay context, số frame và accumulated DTW; cũng thay latency. Cần xác nhận tác giả cấp length thế nào. |
| mBART reduced-vocabulary checkpoint SOKE, text max length 128 | Có thể khác weight khởi tạo, vocabulary/unknown tokens và text bị truncate, làm thay conditioning. Paper chỉ xác nhận họ mBART-large-cc25 và joint fine-tuning, chưa đủ để xác nhận checkpoint/truncation hiện tại. |
| Loss reduction mean theo valid frames và theo channel của từng term | Paper ghi squared norm; implementation không chỉ là một hằng số chung so với cách đọc sum-channels. Với pose 164 và expression 10 channels, hệ số tương đối cho từng tọa độ là `(10/164)/(0.1/10) ≈ 6,10`, thay vì 100 nếu giữ nguyên tổng bình phương. Có thể đổi cân bằng pose/expression; cần xác nhận reduction thực tế, chưa kết luận đây là bug so với code tác giả. |
| Frame-wise correction trên concat `[t,x0,x1]`, SiLU, zero-init layer cuối | Input embedding/activation/init có thể thay năng lực và quá trình hội tụ của correction trong 100 epochs; paper không chốt các chi tiết này. Không có bằng chứng cấu hình hiện tại sai hoặc làm kết quả kém hơn. |
| Learned positional embeddings; scaling t×1000; residual/LN của Stylization | Có thể thay temporal/time conditioning và cách học generator. Paper chưa đủ chi tiết để khẳng định giống từng block. |
| Cộng hand mean trước chuyển quaternion | Round-trip bảo toàn FK nhưng manifold path/prototypes được xây trên physical rotations thay vì residual pose. Nếu tác giả dùng residual thì training path sẽ khác; paper chưa làm rõ. Có cơ sở cho lựa chọn hiện tại từ SMPL-X SOKE, chưa xác nhận khác tác giả. |
| Chọn final epoch, không chọn best validation checkpoint | Nếu tác giả chọn checkpoint theo validation thì model được báo cáo có thể khác. Paper chưa cho rule đủ rõ; không được tự chọn checkpoint gần số test paper nhất. |
| Gaussian source cho baseline 6D; velocity manifold sampler dùng Exp Euler | Source/solver làm thay training distribution và sampling của các ablation Tables 3/5. Hợp lý nhưng paper chưa mô tả đủ để xác nhận giống từng variant. |

Các lựa chọn này cần gắn nhãn **implementation assumption**, không gom chung thành “đã đúng paper”. Riêng mBART fine-tuned jointly **được paper App. D.1 xác nhận**, không phải một assumption.

### I.7. Các chi tiết accumulation/resume/artifact có thể làm kết quả khác

Config generator là microbatch **32**, effective batch **256**. Queue `jobs.py:47` mặc định override thành **16**; Phoenix run cuối dùng 32, CSL/How2Sign run đang đọc dùng 16. Paper ghi **64/GPU × 4 GPU**.

**Khác số GPU/microbatch tự nó không phải lỗi accuracy.** Phần cần kiểm tra là training semantics:

- `training.py:142` dùng floor division để tính accumulation, không validate effective batch có chia hết. Config hiện tại 16/32/64 chia hết 256; override khác có thể tạo global batch khác yêu cầu.
- `_masked_mse` normalize theo số valid frames của từng microbatch, rồi gradients được chia đều cho `accum`. Khi lengths khác nhau, mean của các microbatch mean khác loss normalize trên tổng valid frames của batch gộp. Với DDP, mean loss của từng GPU cũng có thể được average; đổi nhóm 16/32 thành 64 sequence vẫn có thể đổi weighting. Chưa biết code tác giả normalize thế nào, nên không khẳng định loss gộp global mới là loss paper. Cần xác nhận cùng weighting; khác biệt tiềm tàng không chỉ là RNG/dropout.
- Gradient của nhóm microbatch cuối không đủ `accum` không được optimizer step; đầu epoch kế tiếp bị zero. Đây có thể tương đương drop cuối ở global batch cố định. **Không coi là sai lệch độc lập nếu số mẫu/updates bị drop khớp protocol cần tái lập**; chỉ ảnh hưởng so sánh nếu chính sách hoặc số updates khác.
- Checkpoint lưu model/optimizer/scheduler/epoch, không lưu đầy đủ Python/NumPy/Torch/CUDA RNG state. Resume tiếp tục tối ưu nhưng không tái tạo đúng trajectory của một run không bị gián đoạn.

Seeds: shell mặc định 0,1,2; queue mặc định seed 0. Queue `--extra-seeds` chưa mở rộng SignNMFlow-M vì phần M hardcode seed 0. Điều này giới hạn đánh giá biến thiên giữa các run; chưa biết seed/aggregation của tác giả nên không quy single-seed thành lỗi làm giảm chất lượng. Quy tắc ≥3, main 5 seeds là của repository, không phải yêu cầu đã xác nhận của paper.

Vấn đề provenance: `jobs.done_dir` coi job hoàn tất theo name/seed và `metrics.json`, không kiểm tra code/config/metric version; `resume_overrides` cũng tự lấy stage của các lần chạy cũ. `data.build_cache` hash danh sách tên, chưa hash preprocessing/fps/stats/content. Vì vậy sửa K-means hoặc preprocessing rồi chạy lại cùng tên có thể vẫn skip job/reuse stage/cache cũ. Cần namespace/version mới hoặc fingerprint tương thích khi tái lập bản sửa; không xóa đè kết quả cũ.

### I.8. Những khác biệt chỉ ảnh hưởng kết quả efficiency Table 7

Paper đo latency trên **một H100**. `benchmark.py:58` đo trên GPU đang có, gồm text encoding và length prediction, batch size 1, predicted length, bỏ 5 mẫu warm-up. Boundary/length setup chính xác của paper chưa rõ. Timing này chưa đủ để đối chiếu trực tiếp với **0,0103 s/motion**.

Đây là ảnh hưởng trực tiếp tới **latency báo cáo**, không phải bằng chứng về DTW/B-T. Việc benchmark trên GPU khác chỉ có thể xếp là khác biệt vận hành khi không so sánh latency với paper. #Params khác Table 7 đã được tách riêng ở I.3.

### I.9. Thứ tự xử lý để việc đối chiếu kết quả có ý nghĩa

1. **Sửa convergence K-means**, kiểm chứng trên dữ liệu chưa hội tụ; version lại artifact và tạo lại prototypes → correction → generator cho bản sửa.
2. **Chốt protocol đánh giá:** joint/alignment set của “All” và PA-body, inference length, split, back-translator và coverage. Định nghĩa này quyết định số đo có so được với paper hay không.
3. **Chốt dữ liệu và training semantics:** How2Sign filtering/frame sampling, loss reduction, weighting khi accumulation và số updates. Có thể tiếp tục dùng một GPU nếu các semantics cần tái lập được giữ đúng.
4. **Làm rõ architecture/text checkpoint/model selection** còn thiếu từ paper. Số params lệch là dấu hiệu, chưa đủ để tự suy ra cách sửa architecture giúp tăng accuracy.
5. **Đóng băng code/config/artifact version rồi thực hiện các run báo cáo** theo seed protocol đã định. Nếu báo cáo efficiency Table 7 thì cần measurement setup tương ứng. Chỉ chấm test ở giai đoạn báo cáo được người dùng yêu cầu; audit này không chấm test.

Chưa có cơ sở cam kết sửa các điểm trên sẽ đạt đúng số paper, hoặc quy bao nhiêu chênh lệch cho từng điểm khi chưa có thí nghiệm kiểm soát.

## II. Khác biệt vận hành không trực tiếp làm giảm metric chất lượng

Phần này **không gộp vào danh sách nguyên nhân làm DTW/B-T kém hơn paper**. Các điều kiện giữ tương đương được nêu rõ; không khẳng định mọi thay đổi runtime đều bảo toàn từng chữ số của một stochastic run.

| Khác biệt/lựa chọn | Vì sao không phải nguyên nhân accuracy độc lập | Điều kiện/giới hạn |
|---|---|---|
| Một GPU + accumulation thay cho 4 GPU | Số GPU quyết định tài nguyên/tốc độ, không tự thay objective hoặc năng lực mô hình | Cùng effective batch, loss weighting, optimizer, LR schedule, số updates; có thể khác do RNG/sai số số học. Weighting cụ thể hiện tại được tách ở I.7 |
| Microbatch 16/32 thay cho 64 | Giá trị microbatch tự nó không buộc metric kém đi | Accumulation phải giữ semantics tương ứng. Không mặc nhiên kết luận bản hiện tại đã tương đương hoàn toàn |
| Lỗi DDP chưa khởi tạo process group, trong khi run thực tế chỉ có một training process | Nhánh DDP không được dùng nên lỗi này không trực tiếp làm hỏng các run một GPU đang xét | Nếu chuyển sang `torchrun` thì không còn trung tính: các process chưa đồng bộ và không đại diện một run DDP. Chi tiết bên dưới |
| `workers`, `cache_workers`, `pin_memory`, persistent workers | Với dataset preprocessing xác định như hiện tại, đây là lựa chọn throughput/I/O | Giữ cùng dữ liệu, sample order/batching; không thêm augmentation ngẫu nhiên |
| Cache feature, đường dẫn cache/checkpoint, export NPZ, logging | Lưu trữ/output không phải phép biến đổi mô hình hay loss | Cache phải đúng version/data/stats. Stale cache hoặc stage cũ thuộc I.7 |
| `frame_chunk` của correction | Chia energy thành các tổng có cùng normalization trước optimizer step, giữ gradient về mặt toán học | Có thể khác sai số dấu phẩy động; chưa bật ở các config chính được đọc |
| Joint-wise energy tính qua closed form Eq. 42 thay vì dựng ma trận nghịch đảo | Là biểu thức tương đương toán học của đúng objective paper | Có tests đối chiếu dense quadratic form. Không xếp là khác method |
| Forward-mode JVP để tính velocity | Là autodiff đạo hàm path như paper yêu cầu | Có tests đối chiếu chain rule/finite difference; không đổi target về mặt toán học |
| Reuse đúng frozen prototypes/correction khi đổi NFE hoặc objective generator | Tránh tính lại thành phần đã cố định của cùng thí nghiệm | Phải cùng dữ liệu/config/version tương thích. Reuse stage sai version thuộc I.7 |

### II.1. Chi tiết lỗi DDP — giới hạn hỗ trợ, không quy nguyên nhân cho run một GPU

Vị trí: [training.py:31](/home/haipd/SLP/methods/signmflow/training.py:31), [scripts/run.py:48](/home/haipd/SLP/scripts/run.py:48), `method.py:49`, README hướng dẫn `torchrun`.

`dist_info()` chỉ trả world size >1 khi process group đã khởi tạo; entry point/method/hạ tầng hiện không gọi `init_process_group`. Theo đường chạy hướng dẫn, `torchrun` tạo nhiều process nhưng mỗi process vẫn nhận `(rank, world) = (0, 1)`, không bọc generator DDP và train trên toàn dữ liệu riêng.

Run directory cũng tạo riêng theo timestamp; prototype/correction không có rank coordination/barrier và correction không có DistributedSampler/DDP. Nếu cần đường chạy nhiều GPU, phải hoàn thiện các phần này và kiểm tra việc đồng bộ encoder đang gọi ngoài `model.forward`. Chỉ thêm initialization chưa đủ. Audit chưa chạy thử 4 GPU.

**Có thể tái lập accuracy bằng đường chạy khác mà không cần sửa DDP**, nếu giữ đúng dữ liệu, mô hình, loss và training/sampling semantics cần tái lập. Không đưa “thiếu 4 GPU” thành yêu cầu tăng metrics.

## Phụ lục A. Những phần đã khớp và không nằm trong danh sách cần sửa

| Hạng mục | Paper | Implementation | Đánh giá |
|---|---|---|---|
| Representation | Sec. 3.1, Eq. 5; Table 6: 41 quaternion + expression 10-D | `representation.py`, `MotionCodec.encode` | Khớp số chiều/manifold |
| Quaternion sign | App. C.1, Eq. 37 | `geometry.sign_convention` | Khớp |
| Exp/Log, tangent projection | App. C.4, Eq. 47–48 | `geometry.expmap/logmap/project` | Khớp; có tests |
| Prototype theo từng joint | App. C.2, Eq. 38–39 | `extract_prototypes` | Cấu trúc đúng; stopping rule sai ở I.2 |
| Naturalness measure/energy | Eq. 7–8, 10/12/42 | `NaturalnessMeasure.quad_form` | Đúng joint-wise và `vᵀG⁻¹v` |
| Learned path/velocity | Eq. 11, Alg. 1 | `paths.path_point/path_and_velocity` | Khớp; có tests derivative |
| Hai giai đoạn học | Học correction rồi freeze khi học generator | `method.py`, `training.py` | Khớp |
| Target/expression | Eq. 13, App. C.5 | `flow_loss` | Target, radial normalization, Gaussian expression source/linear path đúng; reduction chưa xác nhận ở I.6 |
| Source quaternion | Uniform sign hemisphere | `uniform_hemisphere` | Khớp |
| Correction settings | MLP 3 layers, hidden 512, batch 8, 100 epochs | Config/`CorrectionNet` | Khớp giá trị công bố |
| Generator settings | 8 layers, latent 512, FF 2048, 8 heads, time 2048, dropout .1 | Config/`SignGenerator` | Khớp kích thước mô tả; block/counting chưa xác nhận |
| Text/time conditioning | mBART joint fine-tuning; Stylization sau self/cross/FF | Model/training | Khớp cấu trúc mô tả; chi tiết ở I.6 |
| Optimizer | AdamW, cosine, LR 2e−4, wd 0, fp32 | Training/config | Khớp mặc định |
| Measure settings | K=500, sigma=.5, rho=1e−6 | Config | Khớp |
| Sampler | Alg. 2, Eq. 53, NFE=5 | `flow.sample` | Khớp main sampler |
| Multilingual | Ba dataset, language token theo nguồn | `_train_caches`, `LANG_CODE` | Đúng union/language token; preprocessing ở I.5 |
| Tables 3/5 | Representation/path/prediction ablations | Scripts/FlowSpec | Mapping rows đúng; chi tiết variant còn thiếu ở I.6 |
| Table 4 | K=250/500/750; sigma=.25/.5/.75 | Scripts | Đúng giá trị; bug K-means ảnh hưởng ở I.2 |
| Table 8 | NFE=5/10/20, cùng checkpoint | Scripts | Đúng thiết kế |

**Main sampler không cần đổi:** code giữ `x0`, dự đoán lại target rồi gọi path ở `t+dt`, đúng Eq. 53 và Alg. 2 dòng 12, kể cả thời gian correction. Table 6 gọi explicit Euler nhưng chi tiết thuật toán dùng re-planned path point. Không thay bằng Euler velocity thông thường chỉ để khớp tên trong bảng.

## Phụ lục B. Kiểm chứng đã thực hiện và giới hạn

Lệnh test trên CPU:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/signmflow tests/test_metrics.py
```

Kết quả: **38 passed**, 18 deprecation warnings; 11,67 giây. Tests kiểm tra geometry, paths/derivatives, quadratic form, codec, shapes/sampler và job graph. K-means test dùng các cluster tách tốt nên một bước sau k-means++ có thể đã đạt tolerance của test; chưa kiểm tra convergence/stopping rule.

Các kiểm tra bổ sung: K-means `iters=1` vs `50`, một Lloyd update tiếp theo, đếm params bằng meta tensor, đối chiếu source path multi-GPU, đọc resolved config/log/metrics, thống kê metadata How2Sign train/val. Không chạy benchmark latency, không chạy thử DDP, không xác nhận số liệu của README bằng chấm lại tất cả predictions.

Không thực hiện suy luận thống kê, p-value, CI hoặc significance từ các single-run metrics; các kiểm tra statistical-fallacy không áp dụng cho audit implementation này. Các chênh lệch metric chỉ được ghi nhận với split/protocol cụ thể.


# Storage và pilot go/no-go cho RACE-Sign

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-11-07
- Verification Status: UNVERIFIED for the proposed experiment decisions; filesystem/tensor-size calculations inspected separately
- Version Label: race_budget_plan_v1
- Execution Status: proposed only; no pilot or full training launched

## 1. Dung lượng một seed CSL-Daily

Scope giữ nguyên method chính: frozen author base, flow decoder và selector; FP32; kế hoạch 14.719 decoder-train, 1.077 decoder-val, 3.680 selector-development samples. Chưa tạo split/manifest thực. GiB = 2^30 bytes.

Đã đếm token caption thật bằng author Qwen tokenizer: train trung bình 10,04 tokens, dev 9,99. Tensor storage tính từ frame lengths thật, cộng overhead đo bằng `torch.save` với đúng cấu trúc record. Một teacher record có payload `2600*T + 4096*L` bytes; candidate record thêm `532*T` bytes. Native generated lengths chưa có cho cả corpus, nên selector storage dùng nhiều scenario.

| Thành phần mới | Ước lượng |
|---|---:|
| Export raw133 NPY cho train + dev | 1,13 GiB |
| Decoder teacher/text/code cache | 5,16 GiB |
| Selector generated cache + candidate cache, nếu giữ cả hai | 2,58 GiB khi native lengths gần target; ~3,71 GiB khi dài hơn 1,5× |
| Checkpoints, gồm peak ghi atomic | ~0,32 GiB; dành ~0,5 GiB với logs/metadata |
| Tổng thông thường | **~9–11 GiB** |
| Nếu mọi selector output đều chạm 400 frames | ~14,8 GiB, gồm checkpoint |
| Thêm cache geometry để tối ưu decoder, thiết kế chưa triển khai | ~1,25 GiB |
| Thêm một robustness cache đầy đủ | ~4,80 GiB |

**Dành 20–30 GiB cho một seed/cấu hình** là planning reserve hợp lý với các phần tùy chọn trên. Không cần sao chép lại original dataset và ~9,3 GiB primary pretrained assets đã tồn tại. Ổ hiện còn khoảng 202 GiB. Không có vấn đề dung lượng cho kế hoạch này; GPU contention và compute lặp hiện là bottleneck.

Trainer ghi đè `last.pt`/`best.pt`, không tự lưu 100 checkpoint lớn theo epoch. Lưu toàn bộ predictions/video ở mọi epoch hoặc giữ nhiều ablation/candidate caches sẽ làm dung lượng tăng thêm. Nếu giữ fixed splits/preparation seed/base, teacher cache và frozen-base generated cache có thể dùng chung các training seed; RACE candidate cache phụ thuộc decoder checkpoint nên phải tạo lại.

Pilot đầu 1.000 train + 200 dev, thêm khoảng 200 generated evaluation pairs, thường chỉ cần dưới 1–2 GiB mới cùng checkpoint. Không build toàn bộ full caches trước khi qua pilot.

Evidence: `outputs/race_storage_20261107/{estimate_storage.py,storage_estimate.json}`. Chi phí filesystem đã đo, còn generated length scenarios và chưa có geometry cache là các giả định planning. Các con số không bao gồm video renders hoặc toàn bộ ablation grid.

## 2. Early stopping và đóng một hướng là hai quyết định khác nhau

- Dừng một run: plateau/overfitting/numerical failure, giữ best checkpoint hoặc debug.
- Dừng một module/hypothesis: không còn headroom hoặc lợi ích không bù chi phí sau các đối chứng thích hợp.
- Không thể chứng minh method sẽ không tốt ở mọi budget chỉ từ vài epoch. Một kết quả chưa significant trên pilot nhỏ không chứng minh không có hiệu quả.

Tham khảo nguyên tắc tăng dần budget và loại sớm cấu hình yếu của [Hyperband](https://www.jmlr.org/papers/v18/16-558.html). Kế hoạch dưới đây là một funnel đề xuất cho một GPU, không phải tuyên bố đã cài/chạy Hyperband hoặc ASHA.

## 3. Gate đề xuất — khóa trước khi chạy

Các budget/thông số dưới đây là đề xuất, không phải chuẩn cộng đồng hoặc ngưỡng đã được chứng minh tối ưu. Không chạy/lọc theo test set. Dùng original train để học, fixed dev cho selection; giữ một confirmation-dev chưa dùng để chọn hyperparameters. Lấy nhóm video gốc làm đơn vị tách và bootstrap khi nhiều câu cùng video.

| Gate | Thực hiện | Điều kiện dừng/đổi hướng |
|---|---|---|
| G0: kiểm tra kỹ thuật | 32–64 train clips, 200–500 optimizer steps; thử overfit có kiểm soát, gradient/normalization/FK/padding checks | Loss không học, NaN, sai coordinate hoặc freeze contract: dừng run và sửa implementation/data. Chưa kết luận hypothesis sai. |
| G1: decoder pilot | ~1.000 train, ~200 fixed dev; 5–10 epochs; đánh giá D0 và flow trên **cùng native generated tokens**, với 2–3 sampling seeds cố định | Không lấy reconstruction/training loss làm kết luận. Nếu tay mượt hơn nhưng nội dung sai hơn hoặc không thấy lợi ích vượt noise, chưa cấp full budget. Phân tích lỗi omission/EOS/word order trước khi quyết định. |
| G2: kiểm tra ổn định | Một hoặc hai cấu hình bounded (ví dụ LR hiện tại và một LR thấp hơn), hai training seeds ngắn; nếu đáng tiếp tục mở lên ~4.000 train và 10–20 epochs | Nếu sau budget đã khóa vẫn không có gain thực dụng hoặc chỉ tăng compute, dừng/pivot. Nếu CI còn rộng thì ghi inconclusive, không gọi là chứng minh vô hiệu. |
| G3: selector headroom | Chỉ sau khi decoder đáng dùng: so always-base, always-flow, và oracle-selection dùng GT trên diagnostic-dev | Nếu ngay oracle đơn giản cũng gần như không còn headroom trên additive surrogate, không đầu tư selector lớn. Đây không phải upper bound cho mọi DTW/semantic metric hoặc blended outputs. Decoder tốt vẫn có thể giữ lại. |
| G4: method đầy đủ | Train selector trên held-out groups; so predicted selection với always-flow và sentence-level selection; xét hiệu quả/latency | Selector không hơn control hoặc calibration không ổn: bỏ module này thay vì tự động bỏ cả decoder. Chỉ full seed khi lợi ích được xác nhận trên confirmation-dev. |

Pilot G1 khoảng 1–3 giờ/cấu hình theo profiling shared-GPU hiện tại và CPU preprocessing, tùy độ dài và số lượt đánh giá. Pilot G2 cỡ 4.000 train/20 epochs + validation có thể 12–24 giờ mỗi cấu hình. Đề xuất cap tổng khảo sát ban đầu **24–48 giờ**, không tự động dùng hết. Đây là dự trù mới từ kích thước pilot và đo throughput; chưa phải thời gian pilot đã chạy. Compute của fluent-signer evaluation chưa được tính vào wall-time GPU này.

## 4. Quy tắc quyết định có thể kiểm tra

Trước pilot chọn primary geometry outcome và minimum worthwhile effect, chẳng hạn **giảm 3% hand JPE tương đối**, cùng relational error không xấu đi. 3% là ví dụ để thảo luận ngân sách, không phải target đã được user chốt hoặc threshold paper.

Với cùng dev sentences và cùng sampling policy, tính paired improvement và uncertainty theo source video. Nếu ở mốc budget cố định, **upper uncertainty bound vẫn thấp hơn minimum worthwhile effect**, đó là tín hiệu futility thực dụng cho model ở budget đó. Nếu lower bound đã dương và đạt mức đáng kể, đó là tín hiệu để mở rộng budget, chưa là SOTA. Nếu khoảng uncertainty còn bao trùm cả tác hại và mức gain mục tiêu, kết luận là chưa đủ bằng chứng. Không dùng repeated uncorrected significance peeking để tạo claim thống kê; confirmatory evaluation phải tách khỏi tuning.

Bắt buộc thêm semantic guard bằng evaluator độc lập có GT ceiling/protocol đã kiểm tra, và kiểm tra một số clip với người thông thạo đúng sign language khi cần. Độ mượt và JPE giảm không chứng minh đúng ký hiệu. Semantic non-inferiority margin cần chọn theo evaluator/baseline variance và ý nghĩa sử dụng; không tự đặt một BLEU threshold khi chưa có baseline. Nếu scorer chưa sẵn sàng, ghi semantic evidence missing, không tự promote method bằng geometry alone.

Nhánh decoder phải bị xem xét lại nếu lỗi chính là missing signs, sai thứ tự hoặc dừng sớm: decoder giữ native token length không thể mặc nhiên sửa được. Một module làm giảm geometry cost nhưng làm giảm intelligibility là lý do dừng/pivot, không phải cách đạt mục tiêu SLP.

Trước khi kết luận hướng không đáng đầu tư, kiểm tra comparator chạy đúng, optimizer học được trên G0, pilot không chỉ gồm câu dễ/động tác phổ biến, và lỗi lặp lại ở ít nhất hai short seeds hoặc được giải thích bằng một giới hạn cấu trúc rõ ràng. Đây là quyết định phân bổ compute có uncertainty, không phải editorial verdict về paper.

## 5. Hiện trạng software

Trainer hiện lưu best checkpoint và hỗ trợ exact epoch resume cho decoder; **chưa có `patience`/automatic research-pruning controller**. Có thể dùng CLI bounded budgets rồi resume trên cùng data/config, ví dụ dừng ở 5 epochs để review rồi resume `last.pt` tới tổng 10/20 epochs. Không cần train lại decoder từ đầu chỉ để tăng budget. Nếu thay dữ liệu/điều kiện phải là experiment mới hoặc initialization có provenance, không giả làm exact resume.

Một policy automatic early-stop nếu bổ sung nên có warmup, minimum delta, patience theo validation checks và wall-time/step cap, đồng thời phân biệt STOP_RUN, DROP_MODULE, INCONCLUSIVE và PROMOTE. Không nên tự tuyên bố CLOSE_METHOD từ training loss. Các gate và hành động ở tài liệu này đang là kế hoạch đề xuất; chưa có run nào được tự dừng/đóng hướng.

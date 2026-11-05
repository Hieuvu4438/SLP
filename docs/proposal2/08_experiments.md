# Kế hoạch thực nghiệm: từ kiểm tra giả thuyết đến mục tiêu SOTA

Chưa chạy training hoặc benchmark model trong goal này. Đây là kế hoạch để kiểm chứng [method chính](07_main_method.md), không phải bảng kết quả dự kiến. Mọi experiment theo [benchmark contract](05_benchmark_contract.md), gồm cả How2Sign, CSL-Daily và PHOENIX-2014T.

## 1. Những claim cần kiểm tra riêng

| ID | Giả thuyết | Bằng chứng cần có | Kết quả không đủ để xác nhận |
|---|---|---|---|
| H1 | Structured temporal targets tốt hơn point target khi teacher không chắc chắn | Predicted-plan semantics/articulation tăng với matched backbone/data/compute | Chỉ teacher-plan hoặc boundary F1 tăng |
| H2 | Lợi ích vượt generic augmentation hoặc duration statistics | Hơn matched jitter, shuffled duration và entropy-matched controls | Hơn một point baseline chưa tuning |
| H3 | Planner tạo thông tin useful cho realizer | Intervention trên plan làm output đổi đúng hướng, independent comprehension tăng | Attention/coverage trông có vẻ hợp lý |
| H4 | Method có giá trị trên SLG thực tế | Kết quả nhiều trục trên cả ba datasets, text-only inputs hợp lệ | Chỉ thắng geometry trên một dataset |
| H5 | Có thể cạnh tranh các hệ hiện thời | Matched SOKE/MaDiS/SignSparK/BLANC và các track liên quan | So trực tiếp các số cross-paper khác evaluator |

H1–H3 kiểm tra cơ chế. H4–H5 kiểm tra mục tiêu toàn bài toán. Xác nhận H1 không tự đạt H5, và thất bại H5 không được che bằng việc chỉ báo noise stress test.

## 2. E0 — Kiểm tra representation và evaluator trước training lớn

Trong mỗi dataset, audit một subset được chọn trước theo length, signer, một/hai tay, động tác gần mặt và fitting confidence. Dùng cùng subset cho tất cả models. So video, fitted reference, codec reconstruction và decoded oracle-plan controls.

- Kiểm tra joint maps, canonical shape, rotation conversions, frame rate, masks và units; baseline và candidate cùng adapter.
- BT reference ceiling và codec ceiling; evaluator có đáp ứng khi đảo order, thay số lượng/polarity đã xác nhận, phá hand orientation hoặc mặt không?
- Kiểm tra cả wrist-relative và body-relative errors. PA không được là metric duy nhất cho orientation/location.
- Audit long sequences, missing frames và cropping. Không loại khó samples riêng cho candidate.

Nếu reference/codec phá những distinctions muốn nghiên cứu, sửa measurement/representation trước. Không gọi model ceiling thấp là generator đã đạt semantics đúng.

Diagnostic MaDiS SiBLEU đã thực chạy trong literature audit chỉ kiểm tra scoring function, không thay E0 trên dữ liệu thật. [Kết quả synthetic](evidence/madis_metric_diagnostic.json).

## 3. E1 — So ba hướng ở ngân sách có thể kiểm soát

Chạy scaffold không có cơ chế thêm, A, B và C với cùng target training pool trong từng dataset. Báo annotation, teacher, index và segment-codec costs riêng. Không ép tất cả có cùng parameter count bằng cách bỏ module cần thiết; thêm capacity-matched control và compute-matched control.

Nếu A/C thêm human supervision cho contrasts hoặc useful-retrieval labels, đánh dấu track có annotation bổ sung; cùng motion pool không đồng nghĩa cùng thông tin. Không dùng kết quả này để quy gain chỉ cho kiến trúc so với T0 không có annotation. A dùng verified contrast subset; có baseline nhận cùng annotations dưới global contrast loss. B dùng motion-derived plans; các B controls dùng cùng phase content codec và planner. C dùng cùng train-only index; có always-use và no-retrieval controls.

Nếu ngân sách chỉ đủ pilot, đánh dấu pilot và giữ nguyên kế hoạch đầy đủ cho cả ba datasets. Không đổi mục tiêu cuối thành thắng ở bộ nhỏ hoặc dễ nhất.

## 4. E2 — Ablation quyết định cho temporal distribution

Giữ encoder, content representation, planner, denoiser, optimizer updates và tổng số plan examples:

| Variant | Target temporal structure | Câu hỏi |
|---|---|---|
| No-plan | Chỉ text + learned length | Có cần plan không? |
| Uniform | Equal intervals, matched segment-count distribution | Boundary evidence có hữu ích hơn duration/count prior không? |
| Point-MAP | Best partition từ cùng score | Baseline trực tiếp của uncertainty treatment |
| Confidence-point | Point target với confidence weights | Có vượt cách giảm trọng số targets yếu đã có ở BLANC không? |
| Point+jitter | Perturb cuts trong constraints, matched noise/entropy | Có chỉ là augmentation không? |
| Shuffled durations | Cùng durations, đổi mapping content/time | Có chỉ học duration distribution không? |
| Structured samples | Hai samples từ q, expected loss | Cơ chế chính |
| Collapsed temperature | Cùng code path, temp gần 0 | Gains do code path hay do distribution? |

So thêm **K-matched**: cố định số segments bằng point partition nhưng sample boundaries có cấu trúc. Một thí nghiệm riêng mới cho phép K thay đổi. Nếu không, gain do số units/capacity có thể bị gán sai cho uncertainty.

So same-support scores, same random seeds và matched sampling budget. Nếu distribution dùng hai plans mỗi example, baseline cũng có hai examples/augmentation opportunities; không để nhiều updates giả làm cơ chế tốt hơn.

Không gọi expected NLL trên q samples là exact marginal log-likelihood. Hai objectives khác nhau do vị trí của log/kỳ vọng; nếu sau này đổi objective cần một experiment riêng.

## 5. E3 — Teacher uncertainty và segmentation noise

Có hai tình huống khác nhau:

1. **Natural ambiguity:** phân tầng test/validation diagnostics theo entropy của teacher distribution được tính sau generation chỉ để phân tích; teacher không cung cấp input cho model. Kiểm tra q entropy có liên hệ với annotation disagreement trên một subset được fluent signers đánh giá không. Không gọi entropy là uncertainty đã calibrated trước phép thử này.
2. **Controlled corruption:** thay teacher boundaries/scores của training bằng jitter có mức tăng dần; retrain các variants cùng corruption. Đánh giá trên nguyên test set. Thêm stress test perturb predicted plan ở inference, nhưng báo riêng, vì nó không thay cho training-noise experiment.

Giữ cả uniform baseline theo phản bằng chứng từ SegMo. Nếu point hoặc uniform thắng trên natural data, không lấy kết quả noise nhân tạo ở mức cực đoan để khẳng định method tốt hơn trong thực tế. [SegMo](evidence/M18_segmo.md).

Đo phase reconstruction distortion, duration calibration, cut/count distributions và downstream generation. Boundary agreement với CSLR chỉ là auxiliary metric, không là ground-truth sign segmentation. How2Sign không có coverage gloss tương đương nên dùng annotated subset hoặc articulation agreement, không giả tạo full-corpus gloss labels.

## 6. E4 — Tách planning, realization và exposure gap

Báo cùng lúc:

- Ground-truth length + teacher plan: ceiling diagnostic.
- Predicted content + GT-normalized duration: content/relative-time diagnostic.
- Teacher content + predicted duration: duration diagnostic, vẫn oracle track.
- Fully predicted plan và length: primary inference.
- No plan nhưng cùng text encoder/denoiser budget: baseline.

Chỉ fully predicted row được dùng cho main text-only comparison. Không trộn các oracle controls thành proposed full model.

So teacher-only training với exposure training, cùng schedule cho point/distribution controls. Dùng plan swap giữa semantically khác utterances để kiểm tra model có dùng plan; đây là stress diagnostic, không phải một linguistic alternative đúng. Với valid same-meaning timing variants đã xác nhận, kiểm tra model giữ nội dung khi đổi duration trong range hợp lệ.

Nếu teacher-plan tốt mà free-running kém, đo sai số content prior, duration, EOS và motion realization riêng. Không giải quyết bằng GT length tại inference. Nếu realizer bỏ qua plan mà vẫn đạt score, không giữ claim về planning contribution.

## 7. E5 — Baselines hiện thời và lộ trình ba datasets

Trong T0 target-only track: SOKE no-dictionary, MaDiS adapted cùng pretraining budget, SignSparK no-keyframe với predicted duration hợp lệ, BLANC và generic joint diffusion. Nếu official releases chưa hỗ trợ How2Sign, adaptation phải được ghi rõ và cho đủ validation tuning; không điền original CSL score vào cột How2Sign.

Trong multilingual/extended-data track: so SOKE full và mọi baseline nhận cùng union training data, dictionaries hoặc external corpora khi có thể. Không quy multilingual/data gain thành architecture gain.

Trong gloss/retrieval tracks: SIGNER và SignRR, cùng predicted-gloss pipeline và resource budget. Oracle gloss chỉ là control. Các 2D systems như SignFLIP được thảo luận và có thể có a matched 2D projection track; không so trực tiếp DTW scales với 3D SMPL-X.

Nguồn hiện thời có thể thay đổi trước ngày chạy experiment. Khi bắt đầu thực nghiệm, cập nhật primary-source search, lưu cutoff và frozen baseline list; không âm thầm đổi evaluator sau khi biết kết quả.

## 8. Human study và semantic subsets

Thiết kế một nghiên cứu riêng cho mỗi target language, với fluent signers và ưu tiên Deaf signers. Không giả định người thông thạo BSL đánh giá ASL/CSL/DGS đầy đủ. Không gửi lời mời hoặc tuyển người trong goal này.

Hai task chính:

1. Blind meaning recovery: xem generated signing, mô tả nội dung theo hình thức người tham gia thuận tiện. Người chấm độc lập theo proposition/lexical information được xác định trước, cho phép paraphrases.
2. Contrast discrimination: chọn giữa các nội dung gần nhau đã được xác nhận hợp lệ; balance position/distractors và có “không rõ/không khớp”. Báo chance level, confusion và omissions/substitutions.

Naturalness/hand clarity/NMM appropriateness ratings là bổ sung, không thay comprehension. Reference video, fitted reference và codec reconstruction là calibration controls. Samples lấy theo protocol phân tầng khóa trước, không cherry-pick video đẹp. Counterbalance model order, giữ renderer/view/shape giống nhau.

Pilot ước lượng variance và rater disagreement; từ đó tính sample size theo effect size tối thiểu có ý nghĩa với comprehension, thay vì tự đặt một số người rồi gọi đủ power. Báo qualification, language background, compensation, consent và grouping trong uncertainty analysis. Human study là phần kế hoạch tương lai, chưa có participant data hoặc kết quả.

## 9. Resource feasibility và trình tự quyết định

Không có runtime training thực tế để ước lượng chính xác. Ngân sách phải tính codec, teacher score/cache, segment codec, planner, denoiser, baselines, tuning và human annotations; không chỉ lấy giờ generator của BLANC làm total cost.

Với sequence L, teacher DP cần O(L·Dmax), ngoài interval-score cost; realizer factorized attention cần temporal attention trong từng part và cross-part theo timestep. Cache teacher features/scores giúp giảm repeated preprocessing nhưng tăng storage. Dùng mixed precision/checkpointing khi triển khai; báo peak memory và actual GPU-hours. Proposal không phụ thuộc giả định có sẵn H100 trong repo.

Trình tự: E0→small-scale E1/E2 trên cả ba datasets→E3/E4→full matched baselines E5→human study đủ lực. Dừng mở rộng một cơ chế bị bác bỏ; vẫn báo kết quả đủ để người đọc biết nó thất bại ở đâu.

## 10. Tiêu chí giữ, bác bỏ và phương án thay thế

- **Giữ B:** gains ở primary predicted-plan inference vượt point/confidence/jitter trên semantics độc lập, không đánh đổi articulation/motion quality ở mức quan trọng; kết quả ổn định và có evidence trên cả ba datasets.
- **Bác bỏ claim uncertainty:** same-budget jitter/uniform cho kết quả tương đương hoặc tốt hơn; gains chỉ do K/capacity, teacher-forcing hoặc reference length.
- **Bác bỏ claim semantic benefit:** training teacher/BT tăng nhưng independent comprehension không tăng, hoặc orientation/NMM contrasts tệ hơn.
- **Chưa đạt mục tiêu SOTA:** chưa vượt SOKE/current systems trong matched settings trên cả ba datasets hoặc có trade-off quan trọng. Không đổi nhãn mục tiêu để gọi đó là thành công toàn diện.

Nếu B thất bại vì pseudo boundaries không có ích, giữ generic whole-sequence generator và quay sang A sau khi audit teacher/contrast annotations. Nếu semantics tốt nhưng rare articulation thiếu và train index có coverage đáng tin, kiểm tra C trong retrieval track. Nếu reference/codec đã mất distinctions, ưu tiên sửa measurement/representation trước cả ba hướng. Đây là các điều kiện chuyển hướng, không phải ghép thêm mọi module vào B để cứu một claim đã sai.

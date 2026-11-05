# Benchmark contract: ba datasets, input thực tế và đánh giá nhiều trục

Đây là thiết kế kiểm chứng cho nghiên cứu tiếp theo, **không phải kết quả đã chạy**. Trong goal literature review này chưa tải toàn bộ datasets, kiểm tra actual manifests hoặc tái lập model. Những việc đó là bước thực nghiệm được đặc tả dưới đây; không được viết như đã xong.

## 1. Dữ liệu và đơn vị so sánh

| Dataset | Ngôn ngữ đích / miền | Nguồn xác minh | Quyết định khi làm thực nghiệm |
|---|---|---|---|
| How2Sign | ASL / instructional signing, English transcripts | [Dataset chính thức](https://how2sign.github.io/) | Pin bản original hoặc manually realigned timestamps; giữ official split, kiểm tra nguồn video liên quan |
| CSL-Daily | CSL / đời sống, Chinese translation và gloss | [Dataset chính thức](https://ustc-slr.github.io/datasets/2021_csl_daily/) | Pin official annotations/splits; phân biệt chữ Hán và word tokenization khi tính BT |
| PHOENIX-2014T | DGS / dự báo thời tiết, German translation và gloss | [Dataset chính thức](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | Dùng đúng 2014T, không thay bằng 2014 recognition corpus; kiểm tra frame rate và crop |

How2Sign có hơn 80 giờ video và chỉ một subset được ghi thêm trong Panoptic studio; không gọi toàn corpus là mocap. Trang chính thức cảnh báo sentence clips theo timestamps gốc có thể lệch ASL/transcript và cung cấp manually realigned annotations. Không giả định có gloss đầy đủ và sẵn dùng trên toàn training set chỉ vì header webpage nhắc gloss.

PHOENIX nguồn là video 25 FPS, crop 210×260; fitting ngón tay/mặt có giới hạn quan sát đáng kể. Các split thường dùng của PHOENIX và CSL lần lượt là 7.096/519/642 và 18.401/1.077/1.176 theo setup của [SignRR](evidence/S06_signrr.md). Trước training phải đối chiếu actual manifests và báo mọi mẫu bị loại; chưa coi các số đó là manifest audit đã hoàn thành.

## 2. Các tracks phải báo riêng

| Track | Train có thể dùng | Inference nhận | Cách dùng |
|---|---|---|---|
| T0, primary | Target training text–motion; pretrained text encoder công khai | Text + target-language ID + random seed | So architecture trong cùng data budget, tự dự đoán length |
| T1, multilingual | Union của ba train splits, không test/val motions | Text + language ID + seed | So với multilingual baselines được train cùng pool |
| T2, gloss-supervised | Thêm target gloss annotation hoặc T2G teacher | Text→predicted gloss→motion | Báo T2G cost/errors; không gọi gloss-free training |
| T3, retrieval | Dictionary từ train, hoặc tài nguyên ngoài có liệt kê | Text cùng retrieved exemplars qua pipeline đã xác định | So cùng dictionary và retrieval input; khai báo coverage |
| O, diagnostic oracle | Chỉ dùng để đo ceiling | GT length, gloss, keyframes hoặc plan | Bảng riêng, tuyệt đối không gộp vào main text-only result |

Trong T0, codec/tokenizer/semantic teacher cũng phải train lại chỉ trên target train split. Official multilingual/tokenizer weights không được đưa vào T0 rồi gọi target-only; chúng thuộc T1 hoặc extended-data track. Pretrained language encoder công khai được liệt kê riêng và giữ cùng điều kiện khi cô lập cơ chế.

T0/T1 vẫn cho phép motion-derived teacher ở training: supervision đó chỉ được tính từ train motion. Nếu thêm BOBSL, dictionary data hoặc 600K web pairs, tạo extended-data track và cho đối thủ cùng thông tin khi quy gain cho method. Target-language ID là input hợp lệ; signer ID, reference speed hoặc reference length không được ngầm thêm vào main track.

## 3. Số báo cáo trong literature: mốc tham chiếu, chưa là leaderboard hợp nhất

| Nguồn / setting | How2Sign | CSL-Daily | PHOENIX | Vì sao chưa so trực tiếp |
|---|---|---|---|---|
| SOKE full, Table 1: BT BLEU-4 | 14.48 | 11.30 | 11.87 | Multilingual + dictionary; evaluator/fitting riêng |
| SOKE full: DTW-JPE body/hand | 7.75 / 10.08 | 7.38 / 9.68 | 6.04 / 7.72 | Accumulation/alignment convention cần pin |
| MaDiS: DTW-JPE body/hand | 6.59 / 9.45 | 5.99 / 7.96 | 4.97 / 6.59 | Frozen SOKE tokens; SiBLEU không phải BT BLEU-4 |
| SignSparK no-KF: DTW-JPE body/hand | 6.30 / 11.43 | 6.27 / 10.63 | 4.40 / 7.10 | Fitting/representation khác; duration policy cần tách khỏi released defaults |
| SIGNER: BT BLEU-4 | — | 15.60 | 11.46 | Predicted gloss, 529D và evaluator mới |
| SignRR: BT BLEU-4 | — | 1.57 | 10.96 | Gloss input, lifted keypoints; CSL GT BT chỉ 5.06 |
| SignFLIP: BT BLEU-4 | 7.10 | 9.75 | — | 2D representation; extensive pretraining |
| BLANC: BT BLEU-4 | — | 2.13 | 9.85 | Per-dataset fitting/training, evaluator riêng, no gloss |

Tất cả ô trên là **author-reported**, chưa tái lập. Deep notes và table locators: [SOKE](evidence/S01_soke.md), [MaDiS](evidence/S02_madis.md), [SignSparK](evidence/S03_signspark.md), [SIGNER](evidence/S04_signer.md), [SignRR](evidence/S06_signrr.md), [SignFLIP](evidence/S07_signflip.md), [BLANC](evidence/S09_blanc.md). Dấu gạch chỉ không có main result trong phạm vi study đã đọc, không khẳng định tác giả không có mọi artifact khác.

Không dùng bảng này để tuyên bố SIGNER thắng SOKE bằng 4.30 BLEU hoặc BLANC kém mọi phương pháp trên CSL. Chỉ có within-study comparisons sau khi xác định cùng contract mới hỗ trợ xếp hạng tương ứng.

## 4. Hai mức kiểm tra representation

**Mức A: benchmark compatibility.** Dùng cùng fitting, decoded joint set, frame rate, motion channels và evaluator cho mọi method trong từng bảng. Với reproduction original, giữ nguyên convention của tác giả; với adaptation, ghi rõ retrained/adapted. SOKE 133D có active head/neck rotations nhưng không cung cấp mọi bậc tự do cần cho gaze và facial grammar. Không hứa toàn bộ NMM trong mức A.

**Mức B: expanded articulatory evaluation.** Nếu method cần head/face channels bổ sung, dùng một common expanded reference cho tất cả baselines, không đổi fitting riêng cho model mới. Báo cả phần output giao nhau với mức A và phần bổ sung. Ground-truth reference fitting, codec reconstruction và generation đều phải qua cùng renderer và diagnostics. Không trộn gains do fitting vào gains do temporal/semantic mechanism.

Trước training lưu data card: source IDs, split hashes, fitting version, coordinate system, canonical shape, active joint list, normalizer train-only, resampling/cropping, full-length coverage và invalid-frame masks. Đây là yêu cầu thực nghiệm, chưa có data card thực tế trong goal này.

## 5. Metric contract

### Geometry và dynamics

1. **DTW-JPE:** định nghĩa per-frame-pair mean joint error, pelvis/wrist centering, joint selection và units. Reproduce cumulative DTW nếu đó là original protocol. Báo thêm path-normalized error ở cột riêng; không đổi normalization rồi dùng tên cũ.
2. **DTW-PA-JPE:** chỉ rõ fit similarity transform theo frame-pair nào, fit trên joints nào và trước/sau part selection. PA có thể loại sai orientation/scale, vì vậy cần unaligned hoặc body-relative measures bổ sung.
3. **Hand articulation:** wrist-relative joint positions, geodesic rotation error trên các joints đáng tin, hand orientation trong torso frame và wrist location so với body/face. Không dùng một metric đã loại orientation để khẳng định orientation đúng.
4. **Coordination:** inter-wrist relative pose, relative hand orientation, timing/phase errors cho subset hai tay; contact chỉ đo trên annotation đáng tin, không áp một khoảng cách cố định lên mọi sign.
5. **Dynamics:** velocity/acceleration/jerk deviation so với reference distribution ở cùng FPS, hold/stroke duration nếu có labels; không tối ưu “jerk càng thấp càng tốt”. Báo bone/joint violations và truncation/invalid outputs.

### Semantics và comprehension

- BT dùng frozen checkpoint train trên real train motion; pin input representation, tokenization, SacreBLEU signature và scoring scripts. Báo reference ceiling cùng evaluator trên raw fitted motion và codec reconstruction.
- Dùng ít nhất hai evaluator có kiến trúc/training signal khác nhau; một metric trùng training teacher chỉ là diagnostic phụ. Không chọn evaluator sau khi biết model nào thắng.
- SiBLEU release của MaDiS đã audit là unigram overlap với brevity penalty. Diagnostic synthetic đảo [1,2,3,4] thành [4,3,2,1] vẫn đạt 100, dù không có 4-gram đúng. [JSON thực chạy](evidence/madis_metric_diagnostic.json). Điều này không chứng minh paper dùng chính script release hoặc mọi kết quả paper sai.
- Minimal-pair tests cần xác minh linguistic contrast theo từng ngôn ngữ: polarity, số lượng, roles, lexical handshape, orientation, order và NMM scope. Không tự coi mọi word swap hoặc mirror motion là câu khác hợp lệ. Human-verified subset và paired real examples được ưu tiên.
- Human study phải tách naturalness rating, forced-choice comprehension và blind free-form meaning recovery. Dùng người thông thạo đúng ngôn ngữ đích, ưu tiên Deaf signers; mô tả qualification, trả công, randomization, sample selection và confidence intervals. Không gán một nhóm BSL raters để xác nhận cả ASL/CSL/DGS.

## 6. Baselines, budgets và điều kiện kết luận

Bắt buộc giữ SOKE no-dictionary và full model ở đúng tracks, MaDiS, SignSparK no-keyframe với length prediction hợp lệ, BLANC cùng một generic whole-sequence baseline. SIGNER/SignRR bổ sung trong gloss/retrieval tracks; không loại khỏi thảo luận chỉ vì khác regime. How2Sign cần adapt những model chưa báo kết quả, và ghi rõ adaptation. Không dùng original score ở một bảng để giả lập kết quả matched reproduction ở bảng khác.

Mỗi comparison ghi total trainable/frozen parameters, pretraining data, tokenizer/teacher cost, optimizer updates, tuning budget, hardware và end-to-end latency từ text đến motion. Với length prediction, sampler chỉ nhận text và prediction do chính model tạo; reference-derived padding mask không được truyền vào condition. Released SignSparK defaults cần đặc biệt kiểm tra chỗ này. [Code audit](evidence/S03_signspark.md).

Primary experiment dùng ít nhất ba training seeds khi ngân sách thực tế cho phép; nếu chưa đủ, ghi exploratory và không đưa population-level claim. Chọn checkpoint/tuning trên validation, khóa test và dùng paired uncertainty estimates với grouping theo utterance/source recording. Không dùng best-of-K gần reference ở main result; nếu sample nhiều lần phải báo mean/variance và compute tương ứng.

**Mục tiêu vượt mốc:** trên cả ba datasets, so với SOKE và các đối thủ hiện thời trong matched tracks, báo đồng thời semantics, articulation, motion quality và inference validity. Không đạt mục tiêu chỉ bằng một PA/BT metric thuận lợi. Muốn tuyên bố SOTA cần kết quả trên protocol hợp lệ, không có trade-off quan trọng bị giấu, cùng bằng chứng comprehension đủ mạnh. Nếu chỉ thắng một dataset hoặc một trục, kết luận đúng phạm vi đó và giữ mục tiêu tổng thể là chưa đạt.

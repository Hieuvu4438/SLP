# Ba phương án khác nhau về cơ chế

Tài liệu này phát triển các phương án từ [motion synthesis](03_motion_literature_synthesis.md), [SLG synthesis](04_slg_literature_synthesis.md) và [benchmark contract](05_benchmark_contract.md). Các thành phần dưới đây là **thiết kế nghiên cứu đề xuất**, chưa được triển khai hoặc chứng minh hiệu quả. Không sử dụng kết quả thực nghiệm nội bộ để chọn cấu trúc.

Ba phương án cùng giải bài toán text→motion, tự quyết định chiều dài ở inference. Chúng khác nhau ở nơi can thiệp: A thay supervision để giữ những phân biệt ngữ nghĩa; B thay cách học cấu trúc thời gian khi boundaries không chắc chắn; C thay nguồn thông tin motion thông qua retrieval có quyền từ chối. Không định nghĩa ba biến thể chỉ bằng việc đổi diffusion sang flow hoặc đổi backbone.

## Điều kiện chung để so sánh

Input là tokenized spoken text `y`, target-language ID `l` và random seed. Output `X` là chuỗi motion, độ dài `T` do model dự đoán. Mọi normalization học trên train. Root/shape, FPS, active joints và face channels phải giữ giống nhau trong cùng bảng. Khi bổ sung head/face representation, tất cả baselines được adapt vào cùng expanded track; không chỉ nâng representation cho candidate.

Để kiểm tra cơ chế, dùng cùng multilingual text encoder, cùng motion reference và cùng generic joint denoiser cho các đối chứng phù hợp. Một scaffold khởi đầu là continuous variational motion codec với temporal stride 2, latent width 256, sau đó 8-layer Transformer denoiser hidden 512, 8 heads. Đây là cấu hình đề xuất để ước lượng tính khả thi, không phải hyperparameters đã tối ưu hoặc đã chứng minh đủ capacity. Part embeddings phân biệt body, left/right hand và face; temporal self-attention cùng cross-part attention cho phép trao đổi thông tin.

Denoiser dự đoán clean latent từ Gaussian corruption. Latent loss là MSE; decoded-motion auxiliaries gồm rotation reconstruction, FK joint error và velocity error trên reference đáng tin. Codec phải được so với VQ codec matched capacity: không giả định continuous thắng. Dùng per-part mean losses để tránh số chiều lớn tự chi phối. Không thêm unconditional jerk minimization vì có thể làm mất stroke.

Scaffold A/C dùng total-length head từ text; B thay head này bằng tổng durations và EOS của text-to-plan prior như đặc tả riêng. Với A/C, có categorical distribution trên chiều dài trong support do training/validation xác định. Inference dùng mode hoặc một sample theo policy đã khóa; báo truncation ngoài support. Training được biết target length để tạo motion canvas và masking loss là hợp lệ, nhưng reference length không được truyền vào inference. Oracle-length chỉ ở diagnostic track.

## A. Học sinh motion bằng supervision phân biệt các contrast đáng tin

### Vấn đề và giả thuyết

Một generator có thể tạo chuyển động plausible nhưng không phân biệt “có/không”, số lượng, orientation hoặc một lexical contrast nhỏ. Giả thuyết A: một semantic teacher được kiểm tra trên contrast cụ thể có thể cung cấp gradient tốt hơn global retrieval alignment đối với các lỗi này, khi data/capacity và generator giữ nguyên.

Cơ sở là evaluator stress tests của [STRIDE](evidence/M14_stride.md), hạn chế của geometry-only probes trong [gesture token semantics](evidence/M06_gesture_token_semantics.md), semantic-aware representation của [MoLingo](evidence/M01_molingo.md) và khoảng cách human ratings trong [SignFLIP](evidence/S07_signflip.md). Những nguồn này không chứng minh một teacher mới chắc chắn giải quyết được SLG; chúng chỉ làm câu hỏi đáng kiểm tra.

### Kỹ thuật gốc và cách chuyển giao

Contrastive learning đưa text/motion đúng gần nhau và tách các cặp không khớp. Frozen teacher có thể chấm predicted clean motion để hướng generator giữ semantic condition. Cơ chế này đã có trong HumanTOMATO, SignFLIP, BLANC và các alignment methods; semantic loss đơn thuần không phải đóng góp mới.

Điều chỉnh đề xuất là chỉ tạo **hard negatives đã xác nhận** và đánh giá teacher theo từng loại contrast, từng bộ phận, từng ngôn ngữ. Với cặp text chỉ khác phủ định, người thông thạo xác nhận cặp không đồng nghĩa và text âm không phù hợp reference motion. Với contrast handshape, cần matched examples hoặc annotation của fluent signer. Không tự mirror motion rồi gán nghĩa đối lập; không gọi mọi swapped noun là negative.

### Representation, modules và objective

Teacher nhận decoded joint rotations/positions và expression channels được phép, có các heads cho whole-sequence semantics và các contrast được gán scope đáng tin. Text encoder giữ token sequence. Một local head chỉ được train khi có annotation xác định relevant interval/part; không suy toàn bộ handshape từ transcript.

Với similarity `s(X,y)`, teacher loss có paired alignment và margin:

`L_teacher = L_pair + mean max(0, m - s(X,y+) + s(X,y-))`.

Chỉ những negative hợp lệ được tính; semantically equivalent pairs là multi-positive hoặc bị loại khỏi negative pool. Chia train/validation theo source recording để giảm nhận dạng signer/scene shortcuts. Kiểm tra teacher trên cùng số lượng random negatives để tách hard-negative effect khỏi sample count.

Frozen teacher chấm clean estimate `Xhat_t` của generator ở các noise levels đủ thấp:

`L_A = L_denoise + λ_geo L_geo + λ_vel L_vel + λ_sem(t) L_contrast(Xhat_t,y+,y-) + λ_T L_length`.

`λ_sem(t)` bằng 0 ở vùng noise cao và được ramp trong training; threshold là hyperparameter validation. Gradient qua teacher đi tới generator, teacher weights không đổi. Geometry loss không thay semantic label. Việc dùng low-noise estimates thay vì full sample tiết kiệm compute nhưng tạo proxy gap cần đo.

### Training, inference và dữ liệu

Train codec; train/audit teacher trên target train pairs và contrast subset; freeze chúng; train generator. Candidate có thêm chi phí annotation và teacher training, phải tính vào data/compute budget. Chạy một control dùng cùng annotations cho baseline semantic loss để không quy annotation gain thành architecture gain.

Inference chỉ dùng text, language ID, predicted length và Gaussian seed. Không dùng teacher, negative text, target motion hoặc reference duration; không best-of-K theo ground truth. Có thể đánh giá reranking như track phụ, tính thêm cost, nhưng không ngầm dùng trong main result.

### Prior art, failure modes và phép thử phân biệt

Prior art gần: HumanTOMATO semantic loss; MoLingo representation alignment; FGDM semantic CTC; BLANC alignment và counterfactual event margin; SignFLIP feature distillation. A không nhận novelty từ contrastive training hoặc frozen semantic supervision. Phần ứng viên là dữ liệu contrast đáng tin và cách tách scope/invariance, với giá trị chỉ được xác nhận nếu tạo comprehension gain ngoài teacher.

Rủi ro chính: false negatives, teacher shortcut, mất valid variants, quá khớp một evaluator hoặc thiếu đủ contrast trên How2Sign. A cũng không trực tiếp sửa planning/duration. Giới hạn claim theo những contrast thực sự annotated, không tuyên bố đã giải quyết toàn grammar.

Thử nghiệm quyết định: cùng backbone/data/budget, so no semantic loss, random negatives, global hard negatives và scoped verified negatives. Đánh giá bằng teacher độc lập, minimal pairs mới và human meaning recovery; giữ motion quality và diversity. Nếu chỉ training teacher score tăng, hoặc gain mất khi matched annotations được đưa cho baseline, bác bỏ lợi ích cơ chế.

## B. Học từ phân phối các temporal plans thay vì một partition được coi là nhãn đúng

### Vấn đề và giả thuyết

Motion-derived boundaries là pseudo targets. Một cut có thể ở giữa hold, transition hoặc một sign kéo dài; nhiều cách chia có thể mô tả cùng continuous motion. Giả thuyết B: học trên một phân phối có cấu trúc của các partition hợp lý giúp generator ít phụ thuộc vào lỗi của một teacher segmentation hơn point-target supervision.

Cơ sở chuyển giao gồm train-only alignment của [Glow-TTS](evidence/M17_cross_domain_transfer.md), temporal realization của [FlowMDM](evidence/M16_temporal_planning.md), tổ chức semantic units của [Event-T2M](evidence/M04_event_t2m.md), và temporal SLG evidence từ [SIGNER](evidence/S04_signer.md) cùng [BLANC](evidence/S09_blanc.md). Đây không phải kết luận “motion boundaries là lexical boundaries”.

### Điều phải thay so với kỹ thuật gốc

Glow-TTS alignment giả định phoneme order monotonic; spoken-word order không được ép tương tự trong SLG. B xây plan theo **motion time**, rồi học text→plan bằng full-text cross-attention, cho phép reordering. Plan units là motor intervals, chưa gọi là gloss/sign units nếu chưa có evidence.

BLANC đã có motion-derived DP partition, confidence weighting, duration distribution và EOS. B chỉ có lý do tồn tại nếu **giữ nhiều partition và kiểm chứng lợi ích của uncertainty treatment**, thay vì đổi tên clock hoặc thêm một duration predictor. Marginal inference trong semi-Markov models là kỹ thuật có từ trước; không nhận là phát minh mới.

### Representation và temporal teacher

Dùng frozen continuous codec từ scaffold. Trên mỗi reference training sequence, tính một segment score `a(i,j;X)` cho mọi interval có duration trong giới hạn `[d_min,d_max]`. Một cấu hình cụ thể dùng local latent reconstruction cost quanh segment, boundary change evidence và penalty cho số segments. Các trọng số được chọn trên validation, không trên BT test.

Tạo distribution:

`q(τ | X) = exp(Σ_s a(τ_(s-1),τ_s;X) / temp) / Z(X)`.

`τ` là partition phủ hết sequence, không gap/overlap; `temp>0` điều khiển độ tập trung. Forward dynamic programming tính `Z` bằng log-sum-exp; backward sampling lấy toàn partition. Không lấy riêng từng cut độc lập vì có thể tạo durations không hợp lệ. `temp→0` gần point partition là control quan trọng.

Mỗi sampled segment được chuẩn hóa theo phase thành 8 samples của codec features, rồi qua projection thành content vector `e_s` 128D. Duration giữ riêng dưới dạng `d_s`. Frozen segment encoder được train tái tạo phase samples, không dùng future test motions. B không thêm một codebook bắt buộc.

### Planner, realization và objective

Text-conditioned planner dự đoán chuỗi `(e_s,d_s,stop_s)` theo target motion order. Content head dùng mixture của bốn diagonal Gaussians, duration head categorical, stop head Bernoulli; history là các plan units trước, không cần observed motion tại inference. Mỗi bước được cross-attend toàn text, không monotonic word alignment.

Planner loss là expected negative log-likelihood dưới các sampled teacher plans:

`L_plan = E_(τ~q) Σ_s [-log p(e_s|history,y) - log p(d_s|history,e_s,y) + BCE(stop_s,final_s)]`.

Đây là pseudo-target distribution training; không tự gọi là exact ELBO của toàn motion model. Content vector mang trajectory information nên phải kiểm tra prior có dự đoán được từ text hay chỉ memorize style.

Realizer là cùng joint latent diffusion scaffold, nhận text và frame-aligned plan features. Mỗi segment tạo content embedding, relative phase và duration features; các cửa sổ conditioning có overlap mềm gần boundary. Attention vẫn cho phép body–hands–face phối hợp. Không sinh từng segment độc lập rồi nối cứng.

`L_B = L_denoise + λ_geo L_geo + λ_vel L_vel + λ_plan L_plan`. Đây là tổng các mục tiêu theo stages; main specification đóng băng planner khi train realizer, không tự nhận joint gradient qua discrete plans. Quy tắc căn predicted durations vào canvas training được đặc tả tại [method chính](07_main_method.md).

Training lấy hai partition samples cho mỗi reference trong một phần minibatches để ước lượng kỳ vọng. So với deterministic partition và confidence weighting trên cùng teacher scores, codec, planner capacity và số updates. Không dùng thêm semantics loss chỉ cho B rồi quy gain cho uncertainty.

### Training, inference và dữ liệu

Teacher chỉ đọc train motion. Inference nhận text và language ID, planner sinh content/duration tới EOS, tổng predicted durations quyết định length; realizer sinh toàn sequence với plan này. Không dùng target partition, GT length, gloss hoặc target keyframes. Fallback quá max length phải được ghi là truncation, không crop reference để làm đẹp result.

Train realizer ban đầu dùng sampled teacher plans. Một giai đoạn exposure training dùng text-predicted plans; để so với paired reference, các predicted relative durations được normalize lên training reference length **chỉ trong training**. Inference dùng tổng duration dự đoán thật. Cách này giảm một phần mismatch về content/relative phase, chưa loại mọi length/exposure bias; phải so oracle-plan, predicted-plan và no-plan để đo gap. Không che gap bằng kết quả teacher-forced.

Không cần gloss, nên về nguyên tắc áp dụng được cả ba datasets. Với non-manuals, common joint realizer giữ channels hiện có, nhưng motor segmentation chung chưa mô hình hóa grammatical scope riêng. Independent face intervals là một extension riêng cần labels/reference đủ tin cậy; không gộp vào claim chính để giả lập đã giải quyết NMM.

### Prior art, failure modes và phép thử phân biệt

Prior art trực tiếp nhất là BLANC; SIGNER cho temporal conditions; Event-T2M và FrankenMotion cho event/part timelines; semi-Markov segmentation có lịch sử dài. Đóng góp ứng viên giới hạn ở uncertainty-aware temporal supervision và tác động xuống free-running SLG, không phải planning/segmentation/duration nói chung.

Rủi ro: q chỉ mô tả thiếu quyết đoán của heuristic, không phản ánh valid timings; averaging tạo mờ; prior không học được content; diffusion bỏ qua plan; segmentation temperature chỉ là augmentation strength; full-sequence attention tốn bộ nhớ. Point teacher hoặc uniform segmentation có thể tốt hơn do ổn định.

Thử nghiệm quyết định: point teacher, confidence-weighted point, jitter quanh point, shuffled durations, uniform partition và sampled full distribution. Giữ matched teacher support, data, codec, generator và training samples. Đánh giá dưới controlled boundary noise và trên data chưa perturb. Nếu B không hơn jitter/point trong predicted-plan semantics hoặc chỉ hơn oracle-plan, bác bỏ giá trị chính. Nếu cut alignment tăng nhưng comprehension không tăng, không gọi là giải quyết linguistic timing.

## C. Retrieval có quyền từ chối và điều kiện hóa theo độ tin cậy

### Vấn đề và giả thuyết

Một generator có thể thiếu mẫu articulation hiếm, còn retrieval có thể mang sang sai nghĩa hoặc sai context. Giả thuyết C: cho model quyền bỏ qua exemplar kém phù hợp và sử dụng retrieval như soft condition có thể giữ lợi ích real motion mà giảm negative transfer.

Cơ sở là [SignRR](evidence/S06_signrr.md), [SignSparK](evidence/S03_signspark.md), composition của [FlowMDM](evidence/M16_temporal_planning.md) và cảnh báo về lost semantics khi stochastic decoder bổ sung detail trong [DisCoRD](evidence/M13_discord.md). Không mặc định retrieved articulation luôn đúng ngôn ngữ chỉ vì là real motion.

### Kỹ thuật gốc và điều chỉnh

Retrieval lấy train exemplars liên quan với input; refiner hoặc conditional generator đưa chúng vào context mới. SignRR đã refine whole sequence từ gloss segments, nên C không nhận novelty từ retrieve-and-refine.

Để không giả định How2Sign có gloss, candidate chính truy xuất **sentence-level text–motion pairs** bằng text embeddings từ train index. CSL/Phoenix có thể có gloss-indexed variant ở track riêng. Sentence retrieval dùng được trên cả ba datasets nhưng có nguy cơ thiếu lexical matches; phải đo coverage thay vì coi top-1 luôn là đúng.

Train retrieval luôn leave-source-recording-out: loại target utterance, các đoạn cùng video và duplicates liên quan. Validation/test chỉ truy xuất train entries. Index và thresholds được freeze trước test. Không dùng reference motion hoặc target geometry để chọn exemplar ở inference.

### Representation, modules và objective

Encode retrieved motions bằng cùng codec, giữ sequence lengths/FPS của nguồn. Một retrieval branch cross-attend các latent sequences; không dùng interpolation để ép mọi exemplar thành cùng sign timeline. Model nhận retrieval text, similarity, fitting confidence nếu có từ train sources và padding masks của **retrieved** sequences.

Gate `g(y,r)` dự đoán khả năng exemplar hữu ích. Denoiser hidden state nhận `h + g * CrossAttn(h, retrieval_latents)` bên cạnh text condition. Nếu không đủ match, `g=0` và model hoạt động như generic generator. Gate được train bằng matched/mismatched train retrieval examples; teacher labels phải đến từ annotation hoặc within-train validation, không nhìn target test geometry.

`L_C = L_denoise + λ_geo L_geo + λ_vel L_vel + λ_gate BCE(g,label_useful) + λ_T L_length`.

Không thêm hard articulation-preservation loss khi source và target khác nghĩa: giữ nguyên handshape sai sẽ hại semantics. Chỉ có thể thêm preservation trên subset đã xác nhận same lexical content; subset đó phải khai báo annotation cost và cùng cung cấp cho control baseline.

### Training, inference và dữ liệu

Train codec và base generator trên target train; xây index, tạo leave-source-out retrieval pairs, train conditional branch/gate cùng retrieval dropout. Có control train model không retrieval trong cùng số updates để tách extra training cost. Muốn dùng external dictionaries, tạo extended-data track.

Inference: text→retrieve top-k train pairs→gate→predict target length→joint denoise từ Gaussian noise. Đây là retrieval-conditioned generation, không dùng target keyframes hoặc observed target motion. Gate chỉ nhìn thông tin có sẵn từ input/index; teacher hoặc target reference không chạy ở inference. End-to-end cost gồm index lookup và retrieved-sequence encoding/cache.

### Prior art, failure modes và phép thử phân biệt

Prior art gần là SignRR, SignSparK và motion retrieval augmentation; retrieval attention hoặc confidence gate đơn thuần rất dễ trùng công trình khác. Trước khi chọn C phải đọc sâu các nguồn retrieval-augmented motion tìm được; hiện chưa có evidence để nhận novelty tổng quát.

Rủi ro: text similarity không suy ra cùng sign sequence; rare signs không có trong index; gate chỉ học “ngắn/dễ” thay useful; domain/signer style bị sao chép; mismatched retrieval tạo lexical substitution. Coverage thấp có thể khiến model luôn abstain, làm retrieval branch không có giá trị.

Thử nghiệm quyết định: cùng index và top-k, so no retrieval, random retrieval, always-use retrieval và gated retrieval; thêm semantically close wrong exemplars. Báo quality theo coverage và calibration, cùng performance khi index thiếu rare concepts. Nếu gate chỉ cải thiện nhờ bỏ các mẫu khó khỏi evaluation hoặc cần reference ở test, bác bỏ C. Mọi test sample vẫn phải sinh output và được tính vào main metrics.

## So sánh trước khi lựa chọn

| Tiêu chí | A: contrast supervision | B: temporal distribution | C: selective retrieval |
|---|---|---|---|
| Đối tượng can thiệp chính | Gradient cho những semantic distinctions đáng tin | Bất định trong train-only temporal targets | Nguồn exemplar và quyết định dùng/bỏ |
| Supervision thêm | Contrast subset cần người/nhãn đáng tin | Chỉ motion-derived pseudo plans; không gloss | Useful-retrieval labels hoặc calibration signal đáng tin |
| Khả năng trên How2Sign | Có thể, nhưng annotation cost đáng kể | Có thể học từ motion/text; prior quality chưa biết | Có sentence index, lexical coverage chưa biết |
| Rủi ro leakage | Teacher/evaluator trùng hoặc negatives từ test | Target plan/length lọt vào inference | Target/self retrieval và test geometry reranking |
| Prior art gần nhất | BLANC, HumanTOMATO, FGDM, SignFLIP | BLANC, SIGNER; segmental planning literature | SignRR, SignSparK; retrieval-augmented motion |
| Lợi thế kỹ thuật dự kiến | Trực tiếp nhắm conditional distinctions | Không yêu cầu gloss; kiểm chứng tác động của boundary noise | Tận dụng real exemplars, có fallback đầy đủ |
| Bất lợi chính | Teacher blind spots và annotation burden | Pseudo-plan khó dự đoán, uncertainty có thể gây nhiễu | Coverage và semantic matching khó; extra index cost |
| Đóng góp ứng viên | Scoped trustworthy contrast supervision, cần novelty audit sâu | Structured uncertainty thay point temporal supervision | Calibrated abstention khi exemplar không phù hợp, novelty chưa rõ |
| Bằng chứng tối thiểu để giữ hướng | Independent comprehension gain | Predicted-plan gain vượt jitter/point, không chỉ oracle | Gain với same index, robust wrong-retrieval controls |

**Lựa chọn sau so sánh:** ưu tiên B để kiểm tra một research question có thể falsify với train text–motion pairs trên cả ba datasets: structured temporal targets có hơn point/confidence/jitter không? A cần contrast supervision và teacher validation sâu theo từng ngôn ngữ; C cần coverage/calibration và có prior art retrieval rất gần. Những điều này không chứng minh A/C kém hơn về kết quả, chỉ giải thích ưu tiên nghiên cứu hiện tại.

B cũng có rủi ro novelty cao vì BLANC đã có motion-derived scheduling, còn SegMo hỗ trợ uniform segmentation trong body task. Vì vậy không chọn B bằng claim planning là mới; chỉ giữ giả thuyết tác dụng riêng của structured uncertainty với controls tương ứng. Chi tiết lựa chọn, phương trình, train/inference và failure criteria nằm ở [main method](07_main_method.md) và [kế hoạch thực nghiệm](08_experiments.md).

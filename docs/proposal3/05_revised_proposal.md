# RACE-Sign v2: token-conditioned flow decoding với lựa chọn cải thiện có hiệu chỉnh

**Đây là proposal chính hiện tại, thay cho kiến trúc post-refiner trong `03_method_proposal.md`.** Mọi số trong phần nghiên cứu là paper-reported, không phải kết quả chạy trong repo SLP. Không sử dụng pilot, training log hoặc experiment có sẵn của SLP làm bằng chứng. Nguồn quyết định thiết kế là primary papers được lấy từ bên ngoài và source code tác giả; xem [đọc sâu bổ sung](06_external_evidence_review.md).

## 1. Khuyến nghị nghiên cứu

Chọn **SOKE pose protocol, CSL-Daily làm dataset chính**, PHOENIX làm xác nhận, với hai bảng GF và dictionary riêng. Xây **decoder liên tục có điều kiện từ token** trên generator/tokenizer cố định; sau đó kiểm tra một bộ lựa chọn dự đoán **mức cải thiện**, cho phép giữ decoder cũ ở vùng decoder mới có nguy cơ làm hỏng nội dung. Không chọn benchmark bằng cách nhìn score nhỏ/lớn rồi suy ra dễ vượt. Khả năng đạt SOTA là giả thuyết có điều kiện; vượt SOKE cần so tiếp các model mới trong cùng điều kiện.

Không khởi đầu bằng đổi tokenizer, LM, duration và decoder đồng thời. Đầu tiên kiểm tra decoder có headroom về hand articulation và relations khi base semantics đã đúng. Nếu lỗi chủ đạo là thiếu/sai dấu hoặc dừng sớm, ưu tiên cải thiện generator và duration; decoder giữ length không giải được lỗi đó.

## 2. Vì sao đổi từ v1 sang v2

[DisCoRD §3.2 và §4.2](https://arxiv.org/html/2411.19527) đã thay VQ feed-forward decoder bằng rectified-flow decoder. Ablation của họ cho thấy flow post-refinement kém decoder trực tiếp; với token nghèo thông tin, naturalness tăng có thể đi kèm giảm faithfulness. Đây là **counter-evidence** chống giả định “thêm refiner thì tốt hơn”. Chưa có bằng chứng kết quả HumanML3D chuyển thẳng sang signing, nhưng đủ lý do ưu tiên direct decoder làm đối chứng trung tâm.

[HybridSign §4.2–4.4](https://aclanthology.org/2026.acl-long.31/) đã có self-forcing và confidence-aware attention; [RAM §3.3](https://arxiv.org/html/2601.14788) đã có reconstruction-based error guidance. Do đó novelty của RACE không phải flow, rollout training, uncertainty, part interaction hay correction riêng lẻ. Câu hỏi hẹp hơn và kiểm chứng được là: **lựa chọn giữa decoder rời rạc và decoder liên tục dựa trên mức cải thiện dự đoán, với quyết định phối hợp trên body–hai tay, có giảm semantic damage so với thay decoder cho mọi vùng không?**

## 3. Các giả thuyết

- H1: Với cùng token sequences, continuous decoder giữ chuyển động ngón tay và phối hợp wrist tốt hơn decoder cũ, nhưng lợi ích không đều giữa các câu/vùng.
- H2: Entropy của generator không đủ dự đoán lợi ích thay decoder; model có thể tự tin nhưng sai lexical sign, hoặc thiếu tự tin giữa nhiều biến thể hợp lệ.
- H3: Chọn theo expected improvement, đồng thời ràng buộc tay với thân, cải thiện trade-off giữa local articulation và semantic fidelity so với chọn riêng từng tay.
- H4: Text conditioning giúp decoder xử lý token không đủ thông tin; nó cũng có thể bỏ qua token và hallucinate. Cần dropout/conditioning ablation để kiểm tra cả hai khả năng.

H1–H4 là giả thuyết, không phải kết luận từ experiment cũ của workspace.

## 4. Kiến trúc và tensor contract

Giữ tokenizer `E,Q,D0` và generator `G` của SOKE cố định trong vòng đầu. Với input text `x`, target language `ℓ`, generator tạo token ba nhánh `z=G(x,ℓ)` cùng EOS. Dictionary input chỉ xuất hiện ở dictionary track. `b=D0(z)` là candidate gốc.

Decoder mới `Fθ` nhận Gaussian motion state, timestep, frame-expanded codebook embeddings và token-level text features; output là velocity trong biểu diễn continuous rotation 6D. Dùng learned token-to-frame projection tương tự nguyên lý DisCoRD, không chỉ repeat cùng một vector cho bốn frame. Factor ×4 phải lấy từ tokenizer contract thực, không suy ra từ lời mô tả chung.

**Motion representation:** giữ đúng joint schema của SOKE, chuyển mỗi 3-vector axis-angle sang rotation matrix rồi 6D; expression giữ kênh riêng với normalization train-only. Sau sampling chiếu 6D về SO(3), chuyển về schema evaluator. Không reuse phần parse HumanML3D 263D hoặc skeleton pooling root/legs của SALAD cho fingers. Projection cần xử lý hai vector gần song song; normalization và numerical epsilon là implementation requirements.

**Coupled decoder:** temporal block cho mỗi nhánh, cộng inter-part block nhận body và cả hai tay cùng timestamp, có relative wrist-to-shoulder/head và wrist-to-wrist features tính từ FK. Không ép đối xứng hay loại bỏ contact. Các layer cross-part là cơ chế kế thừa SIGNER/SALAD; đóng góp nếu có nằm ở objective/lựa chọn đầu ra và bằng chứng cải thiện trong signing, không ở tên attention.

MVP giữ expression từ D0 và công khai giới hạn. Không tuyên bố linguistic completeness; M3T chỉ ra richer face fittings là bottleneck riêng. Không tự nâng dataset facial reconstruction rồi so với SOKE trên pose cũ như thể chỉ thay method.

## 5. Học continuous decoder

### Giai đoạn A: reconstruction-conditioned training

Với motion train `y` và tokens `z*=Q(E(y))`, pose coordinates đã chuẩn hóa, lấy `ε~N(0,I)`, `t~U(0,1)`:

`y_t=(1−t)ε+t y`,

`L_FM = E ||Fθ(y_t,t,z*,x,ℓ) − (y−ε)||²_valid`.

Endpoint estimate `ŷ_1=y_t+(1−t)Fθ(...)`. Dùng FK/geodesic rotation losses trên endpoint đã project:

`L_A=L_FM+λj L_joint+λr L_relation+λv L_velocity`.

`L_relation` match relative wrist/body và bilateral geometry với target; không phạt mọi hand-hand contact. `L_velocity` match vận tốc target, không ép vận tốc bằng zero. Loss theo độ dài hợp lệ và finite differences cùng mask; cân scale các part để số chiều body/expression không lấn hands.

Không cần random warp của GT ở giai đoạn A: teacher tokens và target cùng timeline. Train local windows nhưng dùng context halo ở hai biên; chỉ tính loss vùng trung tâm. Khảo sát 32/64/128-frame windows sau kiểm tra FPS/duration thực; các con số là grid khởi đầu đề xuất, không phải cấu hình tối ưu. Whole-sentence variant là đối chứng vì nhiều ký hiệu cần context dài.

### Giai đoạn B: robustness với token sinh

Decoder trained on GT tokens có train/inference gap. Không mặc định Gaussian noise trên code embeddings mô phỏng đúng lexical mistakes. Có ba đối chứng:

1. A-only: tokens GT train, generated tokens test — baseline trực tiếp theo DisCoRD.
2. Synthetic corruption: thay token bằng nearest-code neighbors, part dropout và ngắn hạn code repeat, giữ length; corruption rate fit từ **train-held-out** generator errors. Đây là synthetic approximation, không claim là distribution thực.
3. Generated-token tuning: base train trên phần train riêng, sinh tokens cho held-out train. Chỉ dùng cặp có correspondence hợp lý và kiểm tra độ nhạy alignment. Native length không thay bằng GT ở inference. Các câu thiếu dấu/sai thứ tự nghiêm trọng không được “sửa target” bằng DTW để coi là supervised repair đúng.

Generated-token tuning là thí nghiệm cần đo, không điều kiện mặc định để RACE thành công. Nếu alignment bất ổn hoặc làm tăng hallucination, giữ A-only/synthetic variant và báo hạn chế. Text encoder/generator không được fine-tune trong ablation decoder ban đầu để tránh confound.

## 6. Chọn theo lợi ích thay vì confidence đơn thuần

Sau khi train `Fθ`, freeze cả D0 và Fθ. Với cùng z và x sinh hai candidate `b=D0(z)` và `c=Fθ(ε,z,x)`. Trên **selector-training split** tách khỏi train decoder, dùng GT chỉ để tính các paired losses. Lưu policy noise seed cố định hoặc trung bình trên số seed đã định trước; cùng policy ở dev/test.

Ở các temporal block B cố định, target improvement:

`Δ_B = L_B(b,y) − L_B(c,y) − κ C_boundary(B)`.

`L_B` gồm body/hand joint error, hand orientation và relational error; correspondence chỉ là training diagnostic khi length khác, không oracle inference input. `C_boundary` đo discontinuity khi chuyển decoder; κ chọn trên dev. Vì signed semantics không được mô tả đầy đủ bằng geometry, thêm sentence-level semantic guard như chỉ số độc lập để reject cấu hình; không dùng chính test evaluator làm reward.

Predictor `sφ` nhận text, z, native length, b, c và token scores nếu có; dự đoán `Δhat_B` bằng robust regression. Train binary “c tốt hơn b” như một ablation; classifier xác suất thắng không cho biết magnitude thiệt hại và không tương đương expected improvement. Calibrate trên dev theo loss delta bins, Brier cho sign-of-improvement, MAE và risk–coverage curve.

**Selection unit:** block body+LH+RH chung để tránh chọn một tay từ c, tay còn lại từ b trong cùng ký hiệu. Thử part-independent selection làm ablation; có thể kém vì phase mismatch. Joint selection không mặc định tối ưu, cần chứng minh bằng kết quả.

Hai candidate có cùng length từ z; blend vùng overlap bằng rotation SLERP và expression interpolation nếu cần, với blend width khóa theo dev. Không average axis-angle; không coi window boundary là gloss boundary. Report lỗi ở các biên switch riêng. Sau blend phải tính lại FK, không trộn coordinates tay và body từ hai candidate không nhất quán.

**Chi phí:** cả hai candidate đã được tính nên selector này không tiết kiệm decoder compute. Đo tổng latency/NFE/VRAM; pre-screen trước flow chỉ là extension nếu dự đoán lợi ích đủ tin cậy. Không tuyên bố nhanh hơn SOKE gốc.

## 7. Lập luận toán học và giới hạn của nó

Với một block, đặt `Δ(f)=E[L(b,Y)−L(c,Y)|f]`, f là mọi inference-visible feature. Bỏ qua interaction giữa block và boundary penalty trong mô hình lý tưởng, quyết định tối ưu là dùng c khi `Δ(f)>0`. Expected loss của rule này bằng:

`E[L(b,Y)] − E[max(Δ(f),0)]`.

Nếu dùng `Δhat`, regret so với oracle conditional rule không vượt:

`E[|Δhat(f)−Δ(f)|]`.

Lý do: chỉ có regret khi hai dấu khác nhau; trên tập đó `|Δ|≤|Δhat−Δ|`. Đây là kết quả cho **expected additive surrogate loss**, không phải định lý cho DTW, BLEU, linguistic correctness hay sampling artifacts. Lỗi ước lượng calibration hoặc distribution shift có thể làm rule thực kém hơn cả b. Với temporal switching, cần đưa boundary interaction vào empirical loss; công thức đơn giản không còn bảo đảm cho tổng sequence. Đây là lý do giữ whole-sentence selection baseline và đo switch-boundary errors.

Đây cũng cho thấy dùng token entropy thay cho Δ là thiếu căn cứ: entropy không trực tiếp ước lượng hiệu số hai alternative losses.

## 8. Đối chứng bắt buộc để loại giải thích thay thế

| ID | Biến thể | Mục đích |
|---|---|---|
| C0 | Original generator + D0 | Baseline cùng resource |
| C1 | Same z + direct token-conditioned flow, không text | DisCoRD-style adaptation |
| C2 | C1 + text + coupled relations, dùng c mọi nơi | Lợi ích adaptation signing |
| C3 | Post-refiner sau D0, cùng params/NFE | Phản chứng DisCoRD có lặp lại ở signing? |
| C4 | C2 + entropy threshold selection | Confidence proxy có đủ không? |
| C5 | C2 + predicted improvement, whole sentence | Lựa chọn không switch có hiệu quả không? |
| C6 | C2 + predicted improvement, coupled blocks | Method đầy đủ được đề xuất |
| C7 | C6 nhưng chọn từng part độc lập | Có cần coupling khi chọn không? |
| C8 | Always b / always c / random selection cùng coverage | Loại lợi ích chỉ do mixture |
| O | Oracle GT selection, tách diagnostic | Headroom selector, không tính method score |

Tuning-on-generated-tokens, window size, decoder text dropout, relation loss và flow steps là ablations riêng sau C0–C6. Không chạy toàn bộ grid Cartesian; dùng dev và khóa cuối trước test.

RAM-REG và HybridSign không thể ghép tùy tiện làm drop-in baseline: REG cần reconstruction branch co-trained; HybridSign dùng 2D pose và 60-frame protocol. Có thể xây equivalent-control trong framework nhưng phải ghi “adapted”, không gán số paper cho bản implementation tự viết.

## 9. Đánh giá, kế hoạch triển khai và stop rules

Primary track: GF native-length, cùng tokenizer/base/data; dictionary track dùng đúng index của SOKE như một bảng riêng. Reproduce cả baseline và proposal bằng cùng metric code đã audit. Bảng paper chỉ là mục tiêu tham chiếu. SignSparK no-KF không tự tương đương native-length nếu dùng length của loader.

Để claim cải thiện SOKE protocol, báo JPE và PA thật, body và hand, semantic metric độc lập kèm GT ceiling, native length ratio, hand orientation, wrist/body relation, velocity/jitter và latency toàn pipeline. Không gộp full-face M3T với body/hand legacy như một leaderboard duy nhất; metadata data-fitting khác phải hiện rõ.

Trước train lớn: synthetic fixture cho axis-angle↔6D/FK, masks, linear flow endpoint; identity/distance fixtures cho evaluator; profile 100–500 steps để ước lượng compute. Các bước này là **kế hoạch tương lai**; các experiment cũ trong SLP không được tái sử dụng làm xác nhận.

Thứ tự làm: (1) reproduce C0 bằng external official release đã ghim; (2) train A và C1; (3) C2/C3 để chọn decoder family; (4) kiểm tra oracle selection có headroom; (5) train/calibrate selector và C4–C8; (6) khóa model và evaluate test; (7) xác nhận PHOENIX, rồi How2Sign nếu compute đủ. Human study đánh giá comprehension với fluent signers, blinded order; sample size dựa pilot power, tách naturalness khỏi đúng nội dung.

Dừng selector nếu không vượt whole-sentence rule hoặc oracle có headroom nhỏ. Dừng hướng decoder nếu C1/C2 chỉ tăng naturalness nhưng giảm semantic adequacy. Chuyển hướng generator nếu omission/length chi phối. Không thay primary endpoint sau khi nhìn test để giữ câu chuyện SOTA.

**Kết luận nghiên cứu hiện tại:** direct flow decoding có primary evidence mạnh hơn generic post-refinement; expected-improvement selection là giả thuyết đáng kiểm tra để hạn chế semantic damage khi chuyển từ discrete sang continuous signing. Chưa có kết quả nào chứng minh C6 sẽ vượt C2, MaDiS, M3T hay SignSparK. Novelty giới hạn trong search được ghi ở tài liệu evidence; không claim “đầu tiên” khi chưa exhaustive prior-art review.

# Kế hoạch thực nghiệm và đặc tả triển khai

## Mục tiêu và thứ tự

- **RQ1, chất lượng.** Bổ sung giám sát vị trí cổ tay, đầu ngón, orientation và vận tốc có giảm DTW-JPE tay và hand DTW-PA-JPE, đồng thời giữ khả năng hiểu câu?
- **RQ2, cơ chế.** Khi giữ endpoint loss, p₀ và generator cố định, metric có tương tác giữa joint cải thiện so với metric đường chéo đã match, shortest geodesic và NG gốc?
- **RQ3, tốc độ.** Tác động có giữ được ở 5 NFE, với độ dài dự đoán và phạm vi chuyển động đúng? Đánh giá CSL trước, Phoenix xác nhận độc lập, How2Sign đo tổng quát hóa đến ASL.

Giai đoạn đầu ưu tiên kiểm evaluator và khai thác checkpoint, sau đó thử endpoint loss, rồi huấn luyện các đường vận chuyển. Chốt một ứng viên cuối trên dev. Dùng trực tiếp holdout test chỉ sau khi khóa protocol và có quyền chạy theo quy tắc repo. [Benchmark và audit](01_phan_bien_va_code.md) · [Cơ sở chọn phương pháp](02_literature_review.md)

## E0 · Baseline và thiếu hụt hiện tại

1. Chốt baseline `B0` như config v2, endpoint MSE, 5 NFE, predicted length. Lưu code/diff, prototype và correction ckpt, generator ckpt, thống kê, SMPL-X asset, encoder, cấu hình, version evaluator và manifest train/dev. Chốt seed, số update, schedule, optimizer, vị trí resume, FPS, lọc clip, độ dài, ID/thứ tự và cách dùng RNG. Tách baseline `B0-short` độc lập nếu giới hạn budget. Mọi biến thể pilot phải có số update và schedule cùng `B0-short`.
2. Chạy ba chuẩn evaluator trên dev trước. Self-score GT phải về 0 cho pose, clip nhiễu/chuyển sai câu phải xấu hơn, và static mean pose phải bị phát hiện bởi metric chuyển động. Negative controls có `GT đúng text`, `motion swap giữa câu cùng bin length`, `thứ tự đoạn đảo`, và `pose tĩnh`. Báo cả GT score cho B-T và tỉ lệ split bao phủ. GT BLEU là mốc thực nghiệm của evaluator, có thể thấp hơn một output cụ thể. Fail nếu scorer không phân biệt được đúng text với perturbed text/motion ở mức tập dữ liệu.
3. Cố định baseline paper theo Table 1–2. Đối chiếu checkpoint B-T gốc, PA body/All và cách chọn length với tài nguyên công khai hiện có. Chốt đánh giá nội bộ cùng một evaluator local khi các thành phần đó thiếu. Thử SignSparK-BT như evaluator thứ hai nếu adapter qua dev controls. Convert rotation từ 133D về physical rotations dùng MotionCodec, tạo 6D theo đúng quy ước checkpoint 240D, kiểm mean-hand, scale và joint order với một clip GT được cung cấp. So dev trên IDs giao nhau, ghi coverage và ID bị loại. Với PA body vẫn ghi cả full-joint-fit và upper-joint-fit. Mọi số đối chiếu phải ghi evaluator và alignment version. [SignSparK-BT](https://github.com/JianHe0628/SignSparK_BT)
4. Tạo báo cáo dev theo slice được khóa bằng train. Các bin gồm độ dài, tốc độ tay, hoạt động một/hai tay, lỗi length, signer khi metadata cho phép, non-manual annotated, và high-confidence contact nếu có. Slice theo tốc độ/thời lượng dựa trên reference chỉ dùng để phân tích, không đưa nhãn vào inference. Chỉ xác nhận các slice đã đặt trước và đủ mẫu, còn lát cắt sau quan sát sẽ ghi exploratory.
5. So early checkpoint với final để chốt selection chỉ trên dev. Repo cho thấy CSL `val_loss` 8.567 ở epoch 100 và 11.26 ở 300, song metric gen trên dev mới được dùng để chọn checkpoint. Nếu thay text encoder LR/freeze, early stop hoặc gradient accumulation, áp dụng cùng mới cho cả baseline và các candidate cùng nhóm.

## E1 · Suy luận từ checkpoint, không đổi huấn luyện

Thêm optional `time_grid` và `expression_update` vào [flow.sample](../../methods/signmflow/flow.py#L92), chuyển cấu hình qua [Method.predict](../../methods/signmflow/method.py#L165). Kiểm grid bắt đầu 0, kết thúc 1, tăng nghiêm ngặt. So các grid 5 NFE có node tập trung giữa hoặc cuối khoảng với uniform. Ví dụ `[0,.20,.40,.60,.80,1]` và `[0,.25,.50,.70,.90,1]`. Dùng NFE 1 và 10 làm đường chi phí phụ. Tái lập log trước đó nếu có thể. Thử cập nhật expression theo `e_s=(1-s)e0+s*e1_hat` như đối chứng riêng. Ghi endpoint drift giữa step theo SO(3), p95 angular velocity, length, BT score và chi phí end-to-end.

Giữ `x0,e0` cho cùng clip, seed và số frame ở mọi biến thể. Nếu làm oracle length, dùng như upper-bound diagnostic trên dev và lặp cả B0, thay đổi seed protocol sao cho prefix noise tái sử dụng được. Oracle length tuyệt đối không vào bảng text-only. Với text-swap, giữ đúng length và noise của clip để cô lập điều kiện ngôn ngữ. So khoảng cách output khi đổi text cùng noise và khi đổi noise cùng text. So thêm output → reference đúng với output → reference hoán vị cùng bin, và dùng hai rendition GT cùng câu làm mốc nếu có đủ mẫu. Báo motion separation cùng semantic scorer. Khác biệt lớn giữa output khi swap chưa tự chứng minh nội dung đúng. Bước thay sign ở sampler checkpoint cũ là stress-test train–sample mismatch. Chốt tác động SO(3) bằng run train/sampling nhất quán trong E3. [Hong & Košecká, App. F–G](https://arxiv.org/html/2606.01643v2)

## E2 · Endpoint supervision qua forward kinematics

**Ứng viên có đường triển khai ngắn nhất.** Giữ đường NG gốc, encoder, generator và loss pose/expr. Tại [flow_loss](../../methods/signmflow/flow.py#L60), từ `x_hat,x1` tính các đại lượng FK:

- **Body và vị trí tay.** Vị trí body tương đối pelvis và cổ tay trong hệ trục torso từ FK, scale bằng chiều rộng vai cố định theo shape hoặc thống kê train đã ghim. Đo thêm unaligned wrist displacement theo đúng evaluator.
- **Handshape và orientation.** 5 đầu ngón mỗi tay trong hệ wrist-local, chuẩn hóa bằng chiều dài palm cố định. Giữ thêm 2 trục palm trong hệ torso để tránh mất thông tin hướng. Định nghĩa joint indices, rest frames và handedness ngay trong adapter FK, sau khi đối chiếu SMPL-X skeleton của evaluator.
- **Temporal.** Dùng sai khác có hướng của tọa độ FK qua hai frame liên tiếp, chia cho \(\Delta \tau\) từ FPS trong từng dataset. Bắt đầu với wrist torso-local và fingertips wrist-local. Tính chuẩn robust theo từng nhóm. Loss vận tốc so `ΔFhat − ΔFgt` mới kiểm động học. Phạt đơn thuần `|ΔFhat|` có thể làm mất chuyển động nhanh.

Với \(F_g\) là đại lượng nhóm \(g\), đặt \(L_P=\mathrm{mean}_g\,\mathrm{mean}_{i,k}\,\rho(F_g(\hat x_{i,k})-F_g(x_{i,k}))\). Dùng Huber trên residual đã chuẩn hóa theo scale cố định trích từ train. Định nghĩa \(L_V\) tương tự trên \(\Delta F_g / \Delta\tau\), dùng giao mask của hai frame trong cùng clip. Giữ cùng kiểu average cho mọi microbatch bằng count group-valid frame hoặc pair toàn logical batch. Với subgroup t ≥ 0.5, multiply từng tử số theo gate và dùng một normalization toàn logical batch đã khóa, để batch không có clip đạt gate cho loss 0. Lần đầu thử toàn t và gate t ≥ 0.5 nếu toàn t cho collapse. Theo dõi relative gradient norm của Lpos, Lvel và pose loss trên vài batch train. Chọn hệ số từ lưới ngắn cho mục tiêu gradient ban đầu khoảng 10% hoặc 30% tổng gradient motion head, báo hệ số thực tế, khóa cùng tiêu chí selection.

| Run | Thay đổi | Dùng lại |
|---|---|---|
| E2a | Lpos phần body/wrist/fingertips, thêm palm orientation dưới cờ riêng nếu E2a cải thiện PA nhưng placement/orientation kém | Prototype và correction B0. Fine-tune cùng init với một B0-continue để sàng lọc. |
| E2b | Lpos + Lvel có mask theo FPS | Cùng init và lịch như E2a. Muốn quy cải thiện cho velocity, thay đúng hệ số Lvel. |
| E2c | Tách jaw/expr logging, aux face landmark hoặc tăng expr gradient có kiểm chứng | Chỉ đi tiếp khi labels đủ ổn định và scorer/rater xác nhận nhạy với non-manual. |

Vòng xác nhận huấn luyện từ khởi tạo matched. Viết unit check FK khi nhận rotation physical trực tiếp so với `MotionCodec.decode` + evaluator forward pass. Adapter phải dùng offsets/topology chính xác, tránh cộng hand mean lần hai và phân biệt khớp skeleton với joint cuối hồi quy từ mesh. Nếu fast FK khác vị trí các điểm evaluator dùng thì bổ sung kiểm bias và giữ fast FK làm proxy đã ghi rõ. Giao tiếp chuẩn gồm positions/orientations, validity mask, và gradient tới pose. Làm một lần JVP/grad smoke với batch nhỏ để phát hiện NaN. Ghi chi phí train và inference.

Khi sàng lọc bằng fine-tune, khởi tạo trọng số generator từ checkpoint chung và tạo optimizer/schedule mới cho cả B0-continue và các nhánh E2. `method.init.generator` hiện tải để bỏ qua huấn luyện generator, còn `resume_from` phục vụ tiếp tục optimizer/epoch của run cũ. Đường fine-tune cần cờ warm-start riêng trong `Method.fit` hoặc `train_generator`.

## E3 · Đường vận chuyển trên SO(3)

Ký hiệu \(q\) biểu diễn rotation \(R\), quaternion có thể mang một trong hai lift. Với mỗi pair, chọn lift của target sao cho \(q_0\cdot q_1\ge 0\). Trong quotient distance dùng \(2\operatorname{atan2}(\|q_{\mathrm{rel},v}\|,|q_{\mathrm{rel},w}|)\) với \(q_{\mathrm{rel}}=q_0^{-1}\otimes q_1\) đơn vị. Chọn quy ước xác định trên relative rotation ở trường hợp 180° đúng cut locus và log lại tần suất vùng gần biên. Dùng \(q_0\otimes\operatorname{Exp}_{\mathbb H}(\xi/2)\) cho retraction mũ của velocity góc \(\xi\in\mathbb R^3\). So rotation matrices trong test endpoint. Trọng số mật độ chuyển sang SO(3) sẽ đổi half-angle scale, do đó phải xuất log bandwidth theo đơn vị cụ thể.

Tại [geometry.py](../../methods/signmflow/geometry.py#L125) thêm primitive quotient align và body angular Exp/Log. Tại [paths.path_point](../../methods/signmflow/paths.py#L47) thêm biến thể shortest geodesic. Tại [flow.flow_loss và sample](../../methods/signmflow/flow.py#L60) giữ endpoint prediction quaternion, dùng cùng path specs train/infer, mask và cùng initial rotations. Khi eval, check kể cả `pose` head gần 0 và precision hỗn hợp. Không ép lại dấu theo từng frame giữa quỹ đạo. Để tất cả dự báo cũng bất biến theo lift, thử đầu vào hai cột matrix trong [SignGenerator.forward](../../methods/signmflow/model.py#L138), với sign-invariant chordal loss \(4(1-\langle\hat q,q\rangle^2)\). Từ đẳng thức \(\|R-\hat R\|_F^2=8(1-\langle\hat q,q\rangle^2)\), nên loss đề xuất bằng một nửa squared Frobenius. Báo gradient scale ở cùng khởi tạo.

| Run | Đối chứng | Để rút ra kết luận |
|---|---|---|
| E3a | Quat S³ geodesic + canonical MSE vs shortest SO(3) + canonical MSE | Giữ 4D net, đơn nhất hóa tọa độ nhập về policy đã khóa. Với baseline S³ input nguyên bản, thêm một control cũng theo policy nếu policy đổi. Hiệu ứng trên cung dài được đo trực tiếp. |
| E3b | SO(3) path + canonical MSE vs quotient chordal loss | Tách mục tiêu endpoint. |
| E3c | E3b 4D đầu vào vs dùng 6D/matrix đầu vào cho motion net | Thêm ngẫu nhiên đổi dấu lift với cùng training distribution khi muốn kiểm sign invariance. Báo tham số tăng ở input projection, match capacity nếu chênh lệch đáng kể. |
| E3d | Giữ prior physical và endpoints, nội suy 6D rồi decode từng t so với cùng input/output quotient geodesic | Ablation decoder hợp lệ. Kiểm tần suất cột gần rank-deficient và giữ công bố row 6D Euclidean nguyên bản. |

Một MLP correction cho quotient dùng rotation-invariant features của hai endpoint và output 3D body angular vector/joint. Gắn correction vào chart gốc của \(R_0\) với \(t(1-t)\), giữ các endpoint chính xác theo rotation và same-size hidden/training schedule. Khi giải đường mới, retrain correction, retrain generator với cùng source và output semantics. Với các đối chứng trên SO(3), dùng chung correction architecture này.

## E4 · Metric kéo ngược qua FK cho correction energy

**Đóng góp nghiên cứu chính.** Giả thuyết là nhãn chuyển động có các tổ hợp vai–cổ tay–ngón mà cost mỗi joint không phân biệt. Một metric đo motion perceptual features có thể giúp generator học từ trạng thái trung gian với tín hiệu phối hợp rõ hơn.

Với \(R\in SO(3)^J\), gọi \(F(R)\) là vector observables FK ở E2, gồm vị trí body, cổ tay, hình dạng bàn tay trong wrist-local và trục palm trong torso-local. Ở phiên bản đầu dùng mỗi frame, pose-only và shape chuẩn. Định nghĩa năng lượng
\[
E(R,v)=\lambda_0\,v^T D_0 v + \lambda_F\,\|W^{1/2}dF_R[v]\|^2.
\]
Biểu diễn \(v\) bằng body angular tangent 3D/joint, \(D_0\) là diagonal dương cố định. \(W\) là trọng số nhóm dương theo số tọa độ và scale đã chốt trên train. \(\lambda_0>0\) cho SPD kể cả khi FK có null space. Tính \(dF_R[v]\) bằng JVP theo t của \(F(R_t)\) hoặc Jacobian FK giải tích, tránh dựng dense metric trong nhánh đầy đủ. Giữ float32/float64 cho các primitive và kiểm đạo hàm ngược qua JVP. Biểu thức objective giữ Euclidean base nên ổn định hơn chỉ dùng rank-deficient \(dF^TWdF\). Ghi energy excursion và path length để phát hiện detour lớn.

**Đối chứng nhận diện tương tác.** Tính \(A=dF_R\) trong **cùng basis body angular**. Đối chứng giữ \(\lambda_0 D_0+\lambda_F\operatorname{diag}(A^T W A)\), bỏ mọi cross-term và giữ đúng độ lớn trên mỗi trục tại trạng thái \(R\). Khóa cùng \(W,\lambda_F,D_0\) giữa hai run. Có thể tính diagonal theo squared norms của columns bằng Jacobian giải tích trên skeleton và các vector tương đối. Nếu quá đắt, dùng joint-block diagonal giữ các block 3×3 để cắt tương tác **giữa** các joint, với cùng test từng block. Trong nhánh đối chứng không thay \(A(R)\) bằng train mean nếu muốn xác nhận riêng cross-joint effect. Trên model toy kiểm đối chứng có cùng năng lượng với metric đầy đủ cho basis single joint (single axis với diagonal), đường chéo dương, và hiệu ứng phối hợp cho hai joint. Benchmark chi phí trên batch thật.

**Control density, ngăn việc quy sai cơ chế.** Thử một variant riêng \(E_D(R,v)=\sum_j a_j(R_j)\|v_j\|^2\) với \(a_j\) là hàm log-mass nghịch biến có chặn và normalize theo mức trung vị trên tập **train pairs**. Chọn mức chặn có thể tái lập, ví dụ [0.5, 2] sau khi lấy tương đối log-mass train, giữ prototype count và kernel cùng ngữ nghĩa SO(3). Nếu dùng shortest quotient prototypes, fit hoặc tinh chỉnh tâm theo \(1-(q\cdot c)^2\), có kiểm dữ liệu trùng. Variance theo từng group và rare-pose slices quyết định có giữ density term trong tương lai. Bốn đối chứng chính là NG nguyên bản, shortest geodesic, FK-diagonal và FK-full. Density-only là đối chứng mở rộng khi đã có kết quả cùng baseline.

**Nối code.** Thêm ability truyền năng lượng dạng callable vào [train_correction](../../methods/signmflow/training.py#L48) và [geodesic_energy_gap](../../methods/signmflow/training.py#L109). Raw energy của các metric khác nhau không so trực tiếp do scale. Báo các đường dưới cùng bộ metrics, gồm travel SO(3), FK energy, density mass quantiles và cuối cùng pose + semantic. Đưa kernel/vật lý adapter và spec vào checkpoint. Khởi tạo correction như geodesic, không dùng target khi suy luận. Thuật toán re-plan vẫn chạy ở 5 NFE, thêm FK energy chỉ làm tăng chi phí train.

**Tách hiệu ứng pha.** Dùng bảng chéo geodesic/FK-diag/FK-full × endpoint loss gốc/bản FK khi full và endpoint loss đều qua pilot. Tối thiểu train diagonal và full với endpoint loss gốc rồi so E2 ở B0. Giữ prior và temporal correction theo baseline cho bảng chéo. Thử giới hạn biên độ correction nếu đường học đi vòng lớn, và thử correction theo thời gian nếu phân tích chuyển tiếp cho thấy cần thêm ngữ cảnh. Một ablation loại bỏ \(\lambda_F\), zero correction và control MLP giữ cùng capacity giúp định vị nguồn hiệu ứng.

## E5 · Nhánh bổ sung theo kết quả E0–E4

| Hướng | Điều kiện mở | Thay đổi và kiểm trade-off |
|---|---|---|
| Prior thời gian tương quan | Hand velocity tail vẫn hụt hoặc jitter tăng mạnh theo NFE | Ở `sample_source`, dùng Gaussian AR(1) \(z_k=a z_{k-1}+\sqrt{1-a^2}\eta_k\), \(z_0,\eta\) chuẩn, các joint/coordinate độc lập, lấy unit quaternion và cùng sign convention. Marginal một frame giữ nguyên, coupling theo thời gian đổi. Thử a=0 và 0.5 trước. Dùng cùng prior ở train/infer, correction frame-wise giữ để cô lập nếu representation/metric giữ nguyên. Log p95 speed, hold duration và fast signs, tránh biến loss thành smoothing. |
| Độ dài và text | Oracle dev giảm sai số đáng kể hoặc scorer không nhạy when swapping text | Train riêng length head với clip-level log-length residual/quantiles và token count + pooling có mask. Giữ gold length chỉ ở training. So lower text LR/freeze dưới cùng B0 nếu collapse. Nếu cần semantic supervision, train critic riêng chỉ trên train, xác nhận contrastive pairs, dùng assessor khác để đánh giá. Không optimize B-T evaluator dùng trong bảng. |
| Coarse-to-fine hoặc keyframes | Lỗi semantic order hoặc transition vẫn lặp rõ trên câu dài | Predict kế hoạch theo text ở độ phân giải 4×, prior định lượng length và mask. Train conditioned resynthesis high-rate với coarse dự đoán hoặc corruption matched. So oracle anchors chỉ trên dev và zero-keyframe/text-only baseline, báo NFE tổng. |
| Mật độ/coupling giàu hơn | FK-full có lợi nhưng hiếm pose/nhịp chịu thiệt | Dùng window prototypes theo nhóm, validate held-out coverage. Với OT, thực hiện pairing trong bucket đã khóa theo c nếu có sufficient repeats hoặc học conditional source có thể sample tại inference. Chỉ claim khi train/infer source \(p_0(\cdot\mid c,L)\) phù hợp, mask giữ, match budget và kiểm diversity. |

## Thứ tự chạy, ngân sách và điều kiện dừng

| Giai đoạn | Thiết kế và ngân sách | Quyết định |
|---|---|---|
| Kiểm toán và CPU | Probe đã chạy. Thêm smoke FK, anti-pairs, snapshot RNG và parity evaluator. 23 unit tests hiện hành đã qua. | Fail ở alignment, split, mask hoặc non-finite thì sửa trước khi đánh giá chất lượng. |
| Checkpoint-only | CSL dev, seed model 0, dùng 2 sampling seeds khóa sẵn, full dev và số checkpoint hạn chế. Thiết lập timing trong môi trường cùng tải. Giữ baseline GPU đang chạy. | Mở E2 sau khi có negative controls và baseline reproduces. Không dùng best-of-k lấy từ reference. |
| Pilot huấn luyện | CSL exact-coupled dev comparison. Mỗi ứng viên đắt thử 1 seed pilot, một matched control, 30–50 epoch hoặc cùng số update từ checkpoint nếu là E2. Capture profile một epoch. Log CPU/GPU-time, peak memory, disk. Với E3/E4 không dùng chung generator nếu path semantics đổi. | Giữ nhiều nhất 1 variant mỗi họ nếu điểm hand PA giảm ≥3% so với matched control và JPE tay, motion, text guards không xấu đi ngoài margin. Pilot sàng lọc, không chứng minh thắng paper. |
| Xác nhận nội bộ | Giữ B0 và candidate cùng schedule full 300 epoch / matched updates, seed 0,1,2 tối thiểu. Top candidate đánh giá CSL & Phoenix. How2Sign chỉ scale sau khi có baseline và khóa cùng model family, không dùng kết quả để dò siêu tham số. | Xác nhận cải thiện theo quy tắc dưới. Báo failure hoặc Pareto trade-off nếu không đạt. |
| Báo cáo holdout | Chốt manifest và evaluator, chạy cấu hình khóa trên test khi được phép. So tất cả 4 motion metrics và B-T mỗi dataset. | Chỉ ghi “vượt SignNMFlow” kèm metric, dataset, protocol và độ bất định. Claim rộng trên mọi metrics cần mọi kết luận tương ứng hoặc ghi rõ trade-off. |

Số đo log cho một epoch generator đang ở hàng trăm giây trên GPU chung, ví dụ Phoenix khoảng 170–246 s ở các checkpoint đã xem, CSL 336–440 s, How2Sign khoảng 623–890 s ở epoch 74–79. Do đó một run 300 epoch ở cùng tốc độ là cỡ 14–21, 28–37 và 52–74 GPU-giờ tương ứng, cộng correction, prototype và đánh giá. Đây là ngoại suy từ log, throughput phụ thuộc tài nguyên thực tế. Vòng đầu ưu tiên reuse B0 và chi phí checkpoint-only. Single RTX 5880 Ada quan sát lúc audit đang phục vụ job How2Sign baseline. Giới hạn tối đa mỗi đợt pilot theo GPU-giờ đo được. Giữ riêng output namespace và không thay đổi job baseline.

Hai lệnh dev dưới đây dùng entry point hiện có. Lệnh đầu dùng checkpoint v2 đã hoàn tất, lệnh sau tạo control S³ geodesic với schedule paper. Candidate cần thêm config flags qua `FlowSpec` trước khi chạy. Đo thời gian, dung lượng và số batch trước khi chọn số epoch pilot.

```bash
sgnm_ckpt=outputs/runs/signmflow_v2_main_csl_smplx/signmflow/seed0_20261116-185337-969580/checkpoints
python scripts/run.py --method signmflow --data csl_smplx \
  --name signmflow_proposal_e1_csl_nfe1 --seeds 0 \
  method.init.prototypes="$sgnm_ckpt/prototypes.pt" \
  method.init.correction="$sgnm_ckpt/correction_final.pt" \
  method.init.generator="$sgnm_ckpt/generator.pt" method.sampler.steps=1

python scripts/run.py --method signmflow --data csl_smplx \
  --name signmflow_proposal_e3_csl_s3_geodesic --seeds 0 \
  method.flow.path=geodesic
```

## Quy tắc định lượng

- **Metric chính** là hand DTW-PA-JPE theo cùng joint-set/version. Các metric phụ gồm hand/body DTW-JPE, body DTW-PA-JPE, BLEU-4 và chrF/BT likelihood khi evaluator đã được xác thực. Dùng per-sample wrist displacement, palm angle, velocity matching và length MAE để giải thích cơ chế. Báo NFE, batch-1 latency p50/p95, clip length bins, tổng train time và peak memory. Dùng corpus BLEU với tokenizer/locale ghi rõ, tính lại trên bootstrap samples.
- **Định trước mức khác biệt.** Đặt 3% relative improvement hand PA như ngưỡng thực tiễn sàng lọc. Giới hạn xuống cấp ban đầu 2% tương đối với unaligned JPE tay/thân và giảm tối đa 0.5 điểm BLEU tuyệt đối trên scorer valid. Các margin là quyết định protocol của proposal. Hiệu chỉnh trên variance của xác lập baseline trước khi xem kết quả candidate. Hoãn claim semantic khi không có scorer nhạy hoặc human evidence. Yêu cầu temporal p95/hold errors hoặc signer fluency hỗ trợ khi claim nhắm vào chuyển động. Vật lý và motion profile cần so với reference, không tối đa hóa velocity.
- **Cặp và uncertainty.** Cùng training seed pair, cùng example ID, cùng sampling seed, cùng length policy và nếu khả thi cùng noise draw. Báo mỗi seed, trung bình và độ lệch chuẩn giữa seed. Dùng 10,000 paired bootstrap replicate cho delta theo clip hoặc theo cụm signer/video nếu có phụ thuộc, giữ cặp across models và dùng cùng resample cho corpus metrics. Chênh lệch qua training seeds cũng phải được báo riêng vì ba seed có CI hạn chế. Với CSL và Phoenix, xác nhận chênh lệch primary cùng chiều ở ít nhất 2/3 training seed và CI bootstrap cặp của trung bình delta theo clip/cluster nằm dưới 0. Không dùng sampling seeds như training replicates. Giới hạn claim confirmatory ở primary đã chốt. Nếu kiểm định nhiều endpoint, hiệu chỉnh Holm trong family đã đăng ký hoặc ghi exploratory.
- **Mốc paper.** Với protocol tương đương, yêu cầu điểm xác nhận nhỏ hơn 1.07/1.30/2.07 hand PA cho Phoenix/CSL/How2Sign theo dòng đơn bộ. Báo CI candidate quanh paper point và thiếu CI gốc, không tuyên bố ý nghĩa thống kê trực tiếp đối với paper point. Gap seed 0 hiện có là khoảng 17.7% Phoenix và 5.8% CSL tính từ replica test tới mốc paper. Công bố hai kết luận riêng cho baseline và mốc ngoài. Kiểm các mốc JPE, PA body và BLEU theo bảng audit. Nếu evaluator paper không khả dụng, so BLEU chỉ trên chính scorer đã ghim.
- **Signer study** cho claim dễ hiểu. Mẫu clip khóa trước (ví dụ 48–60 câu/bộ trên length/rare/non-manual bins), ước lượng variance qua pilot vừa đủ trước khi chọn N. Nhờ signer thông thạo từng ngôn ngữ qua quy trình consent phù hợp, trả công, ẩn danh mô hình, randomize thứ tự/cùng avatar. Thu câu hiểu được ở lượt đầu (không hiện text), rồi rating adequacy theo text, handshape, fluency và biểu cảm ở lượt sau. Có GT fitting/video khi quyền dùng dữ liệu cho phép. Phân tích paired với random intercept theo người và clip hoặc bootstrap theo hai cụm, báo độ bất định và các lỗi đặc trưng. [Jiang et al.](https://aclanthology.org/2025.wmt-1.4.pdf) · [BackTranslation2.0](https://arxiv.org/abs/2606.28673)

## Phản biện nội bộ và quyết định

| Góc nhìn | Phản biện mạnh nhất | Điều kiện chọn hướng |
|---|---|---|
| Hình học | Cung dài trên một phần source pairs có thể cung cấp trạng thái trung gian giàu thông tin. Shortest path có thể làm mất tín hiệu hoặc gặp bất ổn gần cut locus. | E3 so cùng khởi tạo/nguồn, report theo mức detour. Giữ S³ nếu quotient không thắng. Metric FK phụ thuộc mốc hình học được chọn, không mặc định hợp với mọi sign. |
| Động học | Một metric có biến thiên riêng có thể đánh giá cao tương tác bất lợi và sinh đường đi vòng để tránh chuyển động trong observable space. | SPD base, matched diagonal, so travel length và sample endpoint quality. Nếu full chỉ hạ training energy hoặc thua diagonal, không giữ claim phối hợp. |
| Ngôn ngữ | Cải thiện PA tay có thể đến từ pose trung bình, mất palm orientation hoặc thông tin non-manual, hoặc nhãn fitting yếu. | Unaligned JPE + wrist/palm/velocity, shuffle tests và signer comprehension. Chưa đủ ngôn ngữ thì giới hạn claim ở motion fidelity. |
| Phương pháp thực nghiệm | Một lợi thế có thể đến từ checkpoint selection, vocabulary, length protocol, weighting hoặc định nghĩa PA. | Ghim contract theo E0. Kiểm seed, ablation và artifact manifest. Không lấy thắng replica làm bằng chứng đã thắng mốc paper. |
| Biên tập khoa học | Đã có FK/SLP, CFM trên manifold, data-metric và keyframe trong prior art. Một thay đổi loss đơn thuần có novelty thấp. | Đề cương publishable chỉ giữ luận điểm hẹp về metric kéo ngược và đối chứng off-diagonal, kèm pilot phủ từ hai ngôn ngữ nếu có tín hiệu. [Literature review](02_literature_review.md) |

**Quyết định hiện tại.** Cho phép prototype và pilot E1–E4 trên dev, ưu tiên E2 và E3. Chưa có chứng cứ rằng model đề xuất vượt replica hoặc paper. Nếu E4 thua diagonal, dừng nhánh metric phức tạp và giữ kết quả âm. Nếu E2 cải thiện pose với semantic guard nhưng E4 chưa rõ, chọn E2 làm ứng viên chất lượng và mô tả đóng góp đúng phạm vi.

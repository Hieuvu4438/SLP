# Đặc tả triển khai multi-rate flow matching cho Sign Language Production

**Phiên bản 1.0 · 20/11/2026 · Tài liệu bàn giao cho AI coding agent**

## 0. Phạm vi và đầu ra cần bàn giao

Xây pipeline **text-to-3D signing offline**, bắt đầu với một ngôn ngữ đích là ASL. Hệ có hai conditional flows Euclidean. Tầng A sinh thân trên ở stride 1 hoặc 4. Tầng B sinh cả hai cổ tay và ngón ở từng frame, đồng thời điều chỉnh cục bộ vai/khuỷu. Tầng B nhận text, thân đã sinh và mốc cấu âm. Predictor mốc học từ pose nhưng chỉ nhận text và độ dài ở inference. Tách **motion time** là chỉ số frame từ **flow time** là thời gian noise → sample.

**Câu hỏi chính.** Với cùng số mốc, lượng thông tin trên mỗi mốc và kiến trúc, mốc theo sự kiện có cải thiện fidelity quanh sự kiện và độ dễ hiểu khi **tự dự đoán từ text** hay không? Câu hỏi phụ là body stride 4 có giảm chi phí mà vẫn giữ các chuyển động quyết định nghĩa hay không?

**MVP** gồm bộ chuyển dữ liệu, synthetic fixture, kiểm tra rig và mask, train/predict cho length, body và anchors, fine flow, sampler, CLI, checkpoint/resume, evaluator, ma trận pilot và báo cáo có kiểm tra leakage. Chạy được CPU smoke test độc lập dữ liệu thật. Thực nghiệm người dùng và dữ liệu thật chạy khi có dữ liệu, phần cứng và điều kiện phù hợp. File này là đặc tả, chưa phải triển khai hay kết quả train.

**Phạm vi ngôn ngữ của MVP.** Body và tay, kèm rotation đầu/cổ. Pelvis translation cố định, rig cố định. Biểu cảm mặt, mouthing, retarget, retrieval, classifier-free guidance, MeanFlow và flow nội tại là phần mở rộng. Ghi rõ phạm vi manual signing trong báo cáo. Nếu mục tiêu nghiên cứu gồm non-manual grammar, bổ sung nhánh full-rate có supervision phù hợp và chỉ số riêng trước khi đưa ra kết luận cho phần đó.

### 0.1 Thứ tự ưu tiên khi code

1. P0 kiểm tra dữ liệu và tái tạo qua rig.
2. P1 đối chứng mốc oracle theo cùng budget, để biết hypothesis có tín hiệu trước khi làm predictor phức tạp.
3. P2 text-only với predicted duration và predicted anchors. Thêm per-anchor confidence sau khi có phương án trọng số đều.
4. P3 ma trận 2×2, seed lặp lại, đối chiếu chi phí và đánh giá ngôn ngữ.
5. Chỉ sau các bước trên mới thử geodesic, giảm NFE hoặc biến thể face.

Các giá trị số trong tài liệu là cấu hình mặc định và ngưỡng quyết định đề xuất. Lưu mọi điều chỉnh vào cấu hình, nêu lý do trong báo cáo. Dùng quy ước của repo xuất phát nếu có, với lớp adapter và test chứng minh tương đương contract dưới đây.

## 1. Cấu trúc dự án và contract

~~~text
configs/
  base.yaml
  pilots/{p0,p1,p2,p3}.yaml
  variants/{A,B,C,D,onset_mid_offset,no_anchor}.yaml
src/slp/
  cli.py
  config.py
  data/{adapters,manifest,prepare,dataset,collate,statistics}.py
  geometry/{rotations,rig,fk,temporal}.py
  anchors/{mining,selectors,planner,calibration}.py
  models/{text,length,body,fine,conditioning}.py
  flow/{paths,losses,sampling}.py
  train/{engine,checkpoint,condition_cache}.py
  eval/{metrics,event_matching,protocols,report}.py
tests/
  fixtures/
  test_geometry.py
  test_data_and_masks.py
  test_anchors.py
  test_flow_and_sampler.py
  test_pipeline.py
~~~

Sau khi cài package dạng editable, thống nhất điểm vào là **python -m slp.cli**. Tách nhóm chức năng bằng subcommands. Cấu hình merge theo thứ tự base → pilot → variant → CLI override. Lưu cấu hình đã resolve, ngăn khóa không hợp lệ và kiểm tra các cặp giá trị ràng buộc trước khi train.

| Hàm công khai | Input chính | Output và bất biến |
|---|---|---|
| prepare_data | manifest nguồn, rig, adapter config | Clip canonical, manifest split, audit và SHA-256. Dừng nếu không xác định được rotation convention, đơn vị hoặc timestamps nguồn |
| collate_clips | list clip canonical | Batch padding, mask, per-sample length. Padding hữu hạn và không tham gia loss/attention |
| build_body_grid | body local rotations, frame/observation masks, stride | Body rotations coarse và timestamps/tensor masks. Stride 1 khôi phục đúng các frame hợp lệ |
| upsample_body | body coarse, timestamps, padding mask, target length | Body local rotations full-rate, từng clip riêng, không đọc padding |
| build_oracle_anchors | canonical clip, policy, K | Tối đa K vị trí thật khác nhau, cùng shape/payload ở mọi policy, tăng dần |
| plan_anchors | text embeddings, length, policy checkpoint | Thời điểm chuẩn hóa và payload dự đoán. Không phụ thuộc video, pose, gloss hoặc mốc ở inference |
| build_fine_target | ground truth body/hand và **body conditioning thực tế** | Hand 6D full-rate và correction 3D theo từng joint |
| predict_clean | noisy state, flow time, condition | Endpoint normalized cùng shape noisy state |
| sample_cfm | model, noise, condition, NFE | Clean state normalized, chính xác NFE lượt model forward mỗi tầng khi CFG tắt |
| decode_pose | body full-rate, fine endpoint, rig | 44 rotations cục bộ và 3D joints/tips, với đúng một rotation nguồn cho mỗi joint |
| evaluate | predictions, references, protocol config | Per-clip scores, coverage, aggregate, bootstrap intervals, NFE và latency |

### 1.1 Kiểu dữ liệu

| Tên | Shape lúc batch | Kiểu và nội dung |
|---|---|---|
| text_ids / text_mask | [B,L] | int64 / bool. True cho token hợp lệ |
| C | [B,L,D_text] | Float. Hidden states từ frozen text encoder, chưa qua projection riêng của từng model. Đọc D_text từ encoder config |
| body_rot6d / body_obs | [B,T,12,6] / [B,T,12] | Rotation cục bộ và mask quan sát |
| hands_rot6d / hands_obs | [B,T,2,16,6] / [B,T,2,16] | Left rồi right. Wrist ở index 0 |
| frame_mask / timestamps_s | [B,T] | Bool / float32. Index không hợp lệ có mask False |
| body_coarse / coarse_mask | [B,Tc,12,6] / [B,Tc] | Tc theo clip, pad tới max của batch. Lưu timestamps coarse và observation mask riêng |
| anchor_time / anchor_mask | [B,K] | Float32 [0,1] / bool. a = time × (T−1) là vị trí theo frame |
| anchor_payload | [B,K,2,24] | Mỗi tay gồm 3 wrist XYZ, 6 palm rotation và 15 wrist-local tip XYZ, theo đúng thứ tự này |
| anchor_feature_mask | [B,K,2,24] | Bool cho payload quan sát/độ tin cậy. Oracle và predicted mask có nguồn khác nhau |
| anchor_confidence | [B,K,2] | Float32 [0,1] cho từng tay. Một slot có thể hợp lệ với một tay |
| fine_state / fine_obs | [B,T,204] | 192 hand 6D + 12 corrections. Mask được broadcast từ joint masks |

**FK-confidence.** Với FK label, mask một joint/tip nếu observation đó hoặc một ancestor cần thiết bị thiếu. Nếu corpus có label trực tiếp đáng tin cho tình huống đó, lưu trong trường riêng và ghi nguồn. Padding zero trong tensor normalized. Các giá trị 6D chưa chuẩn hóa bị thiếu phải được thay bằng identity hoặc nearest valid nội clip để tensor vẫn hữu hạn, đồng thời mask loss giữ False. Không truyền observation mask của ground truth làm conditioning khi inference chỉ có text.

### 1.2 Quy ước geometry và joint map

- Dùng vectors cột, rotation cục bộ $R_j$ từ hệ joint sang hệ parent. Với rest offset $d_j$ trong hệ parent, $R_j^g=R_{p(j)}^gR_j,\ p_j^g=p_{p(j)}^g+R_{p(j)}^g d_j$. Tip có offset cố định từ distal joint và không thêm rotation state.
- Rotation 6D nội bộ là **hai hàng đầu tiên** của ma trận rotation ghép theo row-major, tương thích convention của PyTorch3D. Nếu source lưu hai cột, chuyển bằng ma trận qua adapter. Dùng một fixture rotation 90° có điểm lệch trục để xác minh hướng, vì identity không phân biệt row/column. Giải mã qua Gram–Schmidt được chuẩn hóa. Khi hai vector gần bằng 0 hoặc gần đồng tuyến, chọn trục trực giao dự phòng xác định và ghi số lần fallback. Tính geometry bằng FP32. [PyTorch3D rotation conventions](https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html)
- Body slots chuẩn theo tên **pelvis, spine1, spine2, spine3, neck, head, left_collar, right_collar, left_shoulder, right_shoulder, left_elbow, right_elbow**. Joint map cần cả parent index và topological order để chạy FK. Left/right wrists được dự đoán duy nhất bởi fine state. Với mỗi tay, 15 phalanges theo **thumb, index, middle, ring, pinky**, proximal → intermediate → distal. Nếu rig hoặc corpus dùng trật tự khác, map theo tên từ tài liệu/mã nguồn của adapter.
- Đơn vị geometry là mét. Với anchor/loss position, dùng **body scale h** là khoảng cách rest pose giữa hai shoulder joints của rig canonical, kiểm tra h > 0. Wrist XYZ trong hệ pelvis là $(R_\mathrm{pelvis}^g)^\top(p_\mathrm{wrist}^g-p_\mathrm{pelvis}^g)/h$. Palm orientation là rotation toàn cục của wrist chuyển về hệ pelvis với wrist/palm offset cố định từ rig nếu có. Tip trong hệ wrist tính cùng cách với wrist làm frame gốc. Các output metric vật lý chuyển về mm.
- Việc dùng pelvis làm hệ đo giữ rotation pelvis trong chuỗi sinh. Fix pelvis translation tại gốc cho v1. Nếu adapter loại bỏ một rotation toàn clip khi canonicalize, lưu phép biến đổi để dựng lại và giữ quy ước giống nhau ở mọi phép so.
- Rig manifest chứa version, joint names, parent names, offsets, tip offsets, palm convention, unit, handedness và hash. Test với rig đồ chơi và một clip thật. Không trộn ngầm SMPL-X body với MANO đang dùng local axes khác.

## 2. Dữ liệu, split và chuẩn bị supervision

### 2.1 Nguồn và manifest

Chọn một pipeline pose. Nguồn thử khả thi là How2Sign + bản pose fit đi kèm một baseline. SignSparK có mô tả bộ LMDB dựng sẵn gồm pose 6D, văn bản và metadata. Kiểm tra dữ liệu thực tế theo commit/version. Các chiều, rotation convention, frame IDs, đơn vị và quyền dùng phụ thuộc nguồn. Nếu thiếu wrist hoặc timestamps đáng tin, dừng tại audit dữ liệu hoặc chọn adapter phù hợp. Tránh tự bổ sung nhãn thực. [How2Sign](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html), [SignSparK data documentation](https://github.com/JianHe0628/SignSparK/blob/main/DATA.md)

Mỗi record raw cần clip_id, origin_video_id, split, target_language, text, path/key, source_joint_schema, FPS hoặc timestamps, provenance và quality metadata nếu có. signer_id và annotated segments/contact là optional, ghi missing coverage. Preprocessor sinh clip normalized schema, masks, rig ID, raw hash, command/config version và quality flags. Giữ riêng raw và canonical caches. Không lưu thông tin nhận diện signer trong output công khai.

- Dùng split chuẩn của dataset, gộp kiểm tra theo origin_video_id để chặn clip chồng lấn. Nếu có signer_id và đủ dữ liệu, tạo phân tích signer-held-out riêng. Reference set chẩn đoán chia bằng clip/video theo group.
- Train normalization, anchor-score scales và predictor trên train. Chia dev thành **dev_tune** để chọn checkpoint/threshold và **dev_gate** để chấm pilot. Có thể giữ thêm **dev_cal** cho confidence hoặc dành một fold train riêng cho calibration. Chốt policy trước khi mở test. Không điều chỉnh theo test.
- Ghi số clip và phân bố T, tay hoạt động, occlusion, clip hai tay, số segments và contact coverage theo split. Mỗi loại clip bị loại có reason code.
- 25 FPS là cấu hình prototype, xác minh nguồn raw. Resample rotations bằng nội suy theo timestamp trên SO(3) khi đổi FPS. Dùng nearest hoặc event-time conversion cho flags và boundaries, không nội suy tuyến tính nhãn category. Audit clip ngắn, L=0, nonfinite hoặc timestamps trùng.
- Mặc định nhận 8–320 frame ở pilot. Dùng cùng cohort và cùng quy tắc lọc giữa các biến thể. Clip quá dài báo riêng, giữ khả năng đánh giá bằng chunk theo ranh giới đã có trong một biến thể sau. Không ghép clip từ video khác vào cùng utterance.
- Dùng chuẩn hóa mean/std trên train với masked features. Clamp std tối thiểu 1e−3 trong đơn vị representation trước normalize. Lưu statistics theo feature, stage, rig và stride. Khi thử correction từ body dự đoán, tính stats correction từ một cache pilot đại diện training mixture. Cố định sau khi bắt đầu phép so. Chọn stats per stride để không dùng nhầm state body, ghi rõ khác biệt.

### 2.2 Lưới coarse

Với stride $s$, chia các frame thật thành bin liên tiếp kích thước $s$, bin cuối có thể ngắn. Mỗi bin dùng timestamp trung bình của các frame thật. Với mỗi joint, chuyển matrices → unit quaternions, chọn dấu tương thích quaternion hợp lệ đầu tiên trong bin, lấy trung bình có mask rồi chuẩn hóa. Nếu norm quá nhỏ, dùng rotation hợp lệ gần timestamp bin nhất và đếm trường hợp này. Bin thiếu hoàn toàn observation ở joint được fill hữu hạn bằng nearest valid trong clip hoặc identity, mask False. Stride 1 là phép đồng nhất trên joints hợp lệ. Upsample từng joint bằng SLERP theo timestamps giữa các bin, lấy nearest endpoint ở hai biên. Kiểm tra clip dài 1–5 frame trong unit test kể cả pilot data filter dài hơn. Các phép dựng coarse/upsample là cố định và giống nhau giữa train và inference.

## 3. Chọn mốc và anchor planner

### 3.1 Chọn $K$ thời điểm trên clip đã biết pose

Mặc định $K=8$, dùng $K_\mathrm{eff}=\min(K,T)$. Bỏ padding. Tất cả policy dùng cùng cách lấy payload, mask và normalization. Nếu một policy đặt được mốc vào nhiều vùng có quan sát hơn, báo **số quan sát hợp lệ** ngoài số mốc. Trên subset dữ liệu tay tốt, kiểm tra lại để kiểm soát chênh lệch này.

| Policy | Quy tắc xác định |
|---|---|
| uniform | Dùng index gần $(k+0.5)T/K_\mathrm{eff}-0.5$ với k bắt đầu từ 0. Round nhất quán, clip về [0,T−1], loại trùng |
| onset_mid_offset | Tạo ứng viên onset, floor midpoint, offset inclusive từ segments có sẵn hoặc một segmenter frozen. Dedupe. Nếu quá K, lấy K ứng viên gần nhất với các vị trí uniform theo greedy không lặp, tie là frame nhỏ hơn |
| event | Tính score ở §3.2. Lấy top K theo nonmaximum suppression. Tie là frame nhỏ hơn; ưu tiên mốc cách ít nhất hai frame nếu còn ứng viên |

Nếu vẫn thiếu, chọn các frame chưa dùng xa nhất tập đã chọn, tie theo index tăng. Với mỗi clip, trả indices đã sort, mask và payload. Khi thiếu segment annotation, ghi missing và chạy fallback giống mọi seed. Không diễn giải frame phân đoạn tự động như gloss label thật.

### 3.2 Event-score có thể code ngay

1. Từ pose hợp lệ, lấy wrist-local tips, wrist position theo pelvis và nhãn contact nếu có. Chạy median filter 3 frame trên dữ liệu dùng để **chọn mốc**, ở từng đoạn liên tục có quan sát. Giữ supervision pose gốc.
2. Với $1\le n<T-1$, tính $g_\mathrm{shape}(n)$ là max qua hai tay của mean đầu ngón $\|u_{n+1}-u_{n-1}\|_2/2$. Tính turning point $g_\mathrm{turn}(n)=\arccos(\mathrm{clip}(\cos(v^-,v^+),-1,1))/\pi$ với $v^-=w_n-w_{n-1},v^+=w_{n+1}-w_n$. Bỏ turning signal nếu một displacement nhỏ hơn 0.005 shoulder widths/frame. Lấy max qua hai tay hợp lệ. Signal khi thiếu observation bằng 0 sau normalization, lưu coverage.
3. Tính $g_\mathrm{contact}(n)=1$ nếu có onset/offset contact được xác minh tại n. Label ưu tiên annotation. Pseudo-contact có thể dùng nếu có nguồn geometry đủ tin cậy, hysteresis và tối thiểu hai frame liên tiếp. Tắt nhánh contact toàn cohort trong P1 nếu độ chính xác label không qua audit. Không suy ra rằng mọi cặp đầu ngón gần nhau đều có tiếp xúc.
4. Chia shape/turn cho $P90$ dương tính trên **train** của từng loại, clamp [0,1] với epsilon 1e−6. Score v1 là $1.0g_\mathrm{shape,norm}+0.5g_\mathrm{turn,norm}+1.0g_\mathrm{contact}$. Các frame biên thiếu tín hiệu có score 0. NMS chỉ dùng score dương và fallback theo §3.1. Lưu từng thành phần để audit.

Giữ định nghĩa event windows **độc lập với mốc mô hình được cấp** trong evaluation. Tốt nhất dùng event do người annotate trên subset locked. Nếu dùng proxy từ pose reference, chốt code rồi dùng cùng event evaluator trên mọi cấu hình. Báo subset contact riêng và đối chiếu kết quả sau khi bỏ ±1 frame quanh hợp của mốc oracle các policy để xem gain lan khỏi điểm được cấp trực tiếp.

### 3.3 Predictor và confidence

Encoder text mặc định mBART-large-cc25 đóng băng theo snapshot cố định. Dùng token-level hidden states, attention mask, max 128 tokens. Audit truncation theo split. Cache hidden nếu dùng cùng checkpoint, tokenizer, revision và preprocessing. Toàn bộ pilot dùng cùng text encoder. Mỗi body/fine/planner có projection riêng về width của mình. Freeze encoder cả train và inference.

- **Length head.** Masked mean pool → Linear(D_text,256) → GELU → Linear(256,1). Đích $\log T$, loss SmoothL1 beta 0.10. Inference exponentiate, round và clamp min/max theo config, log count bị clamp.
- **Anchor head.** K learnable queries width 256, Transformer decoder 2 blocks, 8 attention heads, cross-attention vào text. Thêm length embedding từ $\log T$. Mỗi policy dùng cùng architecture và budget. Sinh $K+1$ logits gap từ pooled query state. Đổi logits thành $g=\mathrm{softmax}(\ell)$ FP32, clamp từng gap tối thiểu $10^{-6}$ rồi chuẩn hóa lại để mọi gap dương. Dùng $s_k=\sum_{j=0}^{k}g_j$ cho $k=0,\ldots,K-1$. Vị trí mốc liên tục $a_k=s_k(T-1)$. Không round mốc trong attention. Thêm collision loss trên khoảng cách dưới một frame để giảm mốc chồng nhau. Mask slot mốc nếu $k\ge K_\mathrm{eff}$.
- **Payload head.** Mỗi query xuất 48 chiều normalized, mask và confidence sau hiệu chỉnh. Loss ghép theo thứ tự thời gian, cùng số slot cho mọi policy. Tầng train oracle và predicted shape đều đi qua cùng module conditioning. Mặc định anchor dự đoán luôn có payload hữu hạn, valid mask do length/slot quyết định, confidence xử lý độ chính xác. Ground truth missing vẫn được mask trong supervision.
- **Confidence.** P1 đặt q=1. P2 baseline cũng đặt q=1. Với ablation, sau khi đóng băng predictor, fit logistic head riêng trên dev_cal hoặc train fold giữ lại. Feature là query hidden đã detach. Nhãn q tốt cho mỗi tay nếu sai số time ≤3 frame và mean payload position error ≤0.05 shoulder widths trên những kênh quan sát. Với oracle có thiếu label, bỏ mẫu đó khi fit. Dùng sigmoid làm soft gate, báo Brier score và reliability bins. Không dùng oracle confidence hoặc label từ test lúc inference. Đánh giá q=1 và learned q trên cùng predictor frozen.

Tầng B nhận metadata thứ tự thời gian, payload, mask và q. Dùng thêm time bias $-\min((n-a_k)^2/(2\sigma^2),30)$, mặc định $\sigma=3$ frame ở 25 FPS. Dùng q nhân content của từng tay trước key/value projection và giữ null token hợp lệ để bỏ mốc; mask slot invalid. Trường hợp no-anchor, layer anchor cross-attention phải trả residual zero chính xác. Thử không bias, không confidence như ablation sau main pilot.

## 4. Kiến trúc và forward contracts

### 4.1 Flow A · coarse body

State normalized có shape $[B,T_c,72]$, 12 rotations × 6D. Linear input → width 256 → 4 Transformer blocks → Linear output 72. Mỗi block có pre-LayerNorm, 8 heads, FFN ratio 4, GELU, dropout 0.1 và residual. Body self-attention bidirectional trên coarse tokens. Cross-attention đến text trong mỗi block. Dùng sinusoidal encoding từ timestamps theo giây, nhất quán giữa stride 1 và 4. Embedding $\tau$ là sinusoidal 128 chiều → MLP width → cộng ở input của mỗi block. Dùng padding mask theo clip. Text hidden size gốc lấy từ encoder config rồi project về width model.

Tầng A nhận text, T, stride và body noise. Tầng này độc lập với anchor policy để tái sử dụng cùng checkpoint giữa uniform/event. Khi sample, reshape output, denormalize, decode rotations rồi upsample theo timestamps. Cache cả coarse state và full-rate matrices nếu bộ nhớ cho phép.

### 4.2 Flow B · full-rate hand và correction

State normalized $[B,T,204]$. Flatten hand theo left: 16×6 rồi right: 16×6, sau đó bốn joint corrections theo thứ tự **left_shoulder, right_shoulder, left_elbow, right_elbow**, mỗi joint 3 chiều.

- Một frame token cho cả hai tay. Linear input 204→256. Linear riêng body 72→256 từ pose coarse đã upsample tại từng $n$. Cộng motion positional embedding, flow-time embedding và body projection.
- 6 pre-LayerNorm Transformer blocks, width 256, 8 heads, FFN ratio 4, dropout 0.1. Self-attention thời gian ±8 frame, gồm chính frame. Áp dụng cả local mask lẫn padding mask. Invalid query có output zero. Nếu báo tốc độ, phân biệt local mask trên dense kernel với local kernel thật.
- Mỗi block đọc text và coarse body tokens qua cross-attention. Thêm anchor cross-attention ở các block 2, 4, 6. Giữ thứ tự residual giữa variants. Anchor token encoding gồm time embedding, payload/mask của mỗi tay và confidence gate. Null anchor có value zero; với không mốc, bypass anchor residual.
- Output projection 256→204. Model nhận $\tau$ dạng [B]. Padded outputs luôn zero. Bool masks nghĩa True hợp lệ tại biên module, chuyển sang quy ước của attention API tại đúng một lớp adapter.

### 4.3 Đích correction và decode

Với shoulder/elbow $j$, lấy $\Delta_j^\star=\operatorname{Log}(R_{\uparrow,j}^{\top}R_j^\star)$ từ **body condition đang cấp cho model**. Nếu condition là mẫu sinh thay vì oracle, tính lại đích từ mẫu đó. Clamp norm từng vector vào 0.35 rad theo mặc định. Theo dõi tỷ lệ đích phải clamp và lỗi trần do cap. Log/Exp cần ổn định quanh 0 và kiểm tra trường hợp gần $\pi$. Không truyền gradient qua body generator đã đóng băng.

Decode với $R_{\mathrm{out},j}=R_{\uparrow,j}\operatorname{Exp}(\mathrm{clipnorm}(\widehat\Delta_j,0.35))$. Ghép body, wrist và phalanges vào một rig rồi FK. Dùng cùng phép clip, đơn vị, thứ tự multiplication khi train và inference. Bản tắt correction giữ 204 chiều và mask 12 chiều correction để công bằng về kích thước mạng. Với stride 1 + oracle body, correction target gần zero là expected. Với stride 1 + predicted body, vẫn tính correction từ body thực tế.

## 5. Objective, mask và sampler

### 5.1 Flow matching với endpoint prediction

Mỗi clip có một flow time $\tau\sim U[0,1)$, broadcast trên frame và channel. Hai tầng có noise độc lập $\epsilon\sim\mathcal N(0,I)$. Tạo $x_\tau=(1-\tau)\epsilon+\tau x_1$. Mạng dự đoán $\hat x_1=f_\theta(x_\tau,\tau,c)$. Normalize state trước tạo path. Đặt padded normalized features = 0 sau noise/sampling update. Observation mask áp dụng vào objective, frame mask dùng cho attention. Train flow theo geometry Euclidean trên state đã normalize. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), [MotionHiFlow](https://arxiv.org/html/2604.23264v1)

Định nghĩa mean_masked là sum lỗi hợp lệ chia số scalar hợp lệ, tích lũy FP32. Tính per-clip trước rồi mean trên clips có nhóm quan sát để clip dài không tự động nặng hơn. Mask joint broadcast qua channels. Nếu một nhóm không có mẫu hợp lệ, trả zero còn graph và ghi coverage. Nếu dùng DDP, aggregate đúng numerator cùng số clip hợp lệ cho reporting và weighting. Temporal difference cần cả hai frame/joint hợp lệ. Dùng $\rho_\beta$ SmoothL1, beta mặc định 1 ở các feature đã normalize.

**Tầng A.**

$$
L_A = L_{\mathrm{body},x} + 0.10L_{\mathrm{body},R}.
$$

- $L_{\mathrm{body},x}$ là masked MSE trên normalized endpoint.
- $L_{\mathrm{body},R}=\mathrm{mean}_{m}\| \hat R-R^\star \|_F^2 / 8$ trên coarse rotations sau decode. Hệ số 8 chuẩn hóa range của squared chordal distance trên SO(3). Geometry FP32. Log riêng tỷ lệ fallback Gram–Schmidt.

**Tầng B.**

$$
L_B =
  1.00L_H
  +0.25L_\Delta
  +0.25L_\mathrm{tip}
  +0.15L_\mathrm{wrist}
  +0.10L_\mathrm{palm}
  +0.10L_\mathrm{vel}
  +\lambda_cL_\mathrm{contact}.
$$

| Term | Công thức và đơn vị | Mask |
|---|---|---|
| $L_H$ | Trung bình hai nhóm wrist và finger. Mỗi nhóm là MSE normalized 6D, weight 0.5 dù số joint khác nhau | Joint × frame quan sát |
| $L_\Delta$ | MSE normalized 12 chiều với target correction đã clamp | Condition body và GT joint hợp lệ; zero khi tắt correction |
| $L_\mathrm{tip}$ | SmoothL1 beta 1 của sai số wrist-local fingertip XYZ chia $0.02h$, với h là rest shoulder width | Tip và ancestor cần thiết đã quan sát |
| $L_\mathrm{wrist}$ | SmoothL1 beta 1 của sai số wrist XYZ trong hệ pelvis chia $0.05h$ | Wrist và ancestor quan sát |
| $L_\mathrm{palm}$ | Squared chordal distance / 8 của palm rotation trong hệ pelvis | Palm orientation và reference frame hợp lệ |
| $L_\mathrm{vel}$ | SmoothL1 beta 1 giữa các local-tip velocities trong hai chuỗi, chuẩn hóa theo $0.25h/\mathrm{s}$, tương đương chia $0.01h$ cho frame difference ở 25 FPS | Cả hai frame và joint hợp lệ. Dùng timestamp thật |
| $L_\mathrm{contact}$ | Mặc định $\lambda_c=0$. Khi có pair labels đáng tin, với pair và frame positive dùng SmoothL1 beta 1 của $(d^\mathrm{pred}_{ab}-d^\mathrm{gt}_{ab})/(0.02h)$ | Pair cùng vị trí bề mặt/landmark được định danh, frame và label valid |

Nếu FK features đã ở đơn vị shoulder widths thì chia 0.02, 0.05 hoặc 0.25/s tương ứng, không chia h lần nữa. Mục tiêu velocity bám chuyển động reference. Nếu bật contact loss, giữ đúng mục tiêu khoảng cách theo nhãn và báo false positives ở close non-contact. Giữ loss giống nhau trong phép so mốc/stride. Ablation P3 để diễn giải loss gồm endpoint-only, endpoint + FK và endpoint + FK + velocity.

**Planner.** Length loss như §3.3, train riêng. Anchor predictor dùng

$$
L_{\mathrm{anchor}} = 1.00\,\rho_{0.10}(\hat a-a)
+1.00L_{\mathrm{payload}}
+0.05\,\mathrm{mean}\big[\mathrm{ReLU}(1-(\hat a_{k+1}-\hat a_k))\big],
$$

với $\hat a,a$ dùng đơn vị frame. $L_\mathrm{payload}$ là mean của ba group normalized SmoothL1 beta 1 cho wrist XYZ, palm 6D và fingertips, weight đều giữa các group hợp lệ. Time/spacing loss mask theo slots thật. Hai slots cùng hợp lệ mới có spacing loss. q fit riêng bằng binary cross entropy với logits, đo calibration trên held-out dev. Không backprop q vào pose planner.

**Loss fairness.** Với linear path, squared velocity error là squared endpoint error chia $(1-\tau)^2$. Objective chính dùng endpoint MSE với trọng số time đều. Nếu so velocity head và endpoint head, thêm ablation weight theo $\max(1-\tau,0.05)^{-2}$, dùng cùng time sampling và cùng auxiliary losses. Log gradient norm và unclipped loss của từng group.

### 5.2 Euler sampler

Dùng seed theo clip, stage và sample index. Với N NFE, lưới $\tau_i=i/N$ cho $i=0,\ldots,N-1$. Với $i<N-1$, cập nhật

$$
x \leftarrow x + \frac{1}{N}\frac{\hat x_1-x}{1-\tau_i}.
$$

Lần gọi cuối tại $1-1/N$ trả $\hat x_1$ trực tiếp, tương đương Euler đến endpoint. Với N=1, gọi một lần tại 0. Denormalize và decode **sau** solver. Không project 6D giữa các bước Euler của Euclidean state. Reset padding về zero sau từng bước. Tầng B dùng noise riêng và nhận endpoint tầng A đã hoàn tất, vì vậy không cần upsample noisy state. Mặc định Euler 4+8 NFE. Heun, CFG hoặc re-sampling phải đếm số lượt network forward thực tế.

~~~python
def sample_cfm(model, x, condition, frame_mask, nfe):
    if nfe < 1:
        raise ValueError("nfe must be positive")
    m = frame_mask[..., None]
    x = x.float().masked_fill(~m, 0)
    for i in range(nfe):
        tau = x.new_full((x.shape[0],), i / nfe)
        clean = model(x, tau, condition).float().masked_fill(~m, 0)
        if i == nfe - 1:
            return clean
        x = x + (clean - x) / (nfe - i)
        x = x.masked_fill(~m, 0)
~~~

### 5.3 Conditioning trong train và inference

| Stage | Train | Evaluation oracle | Evaluation text-only |
|---|---|---|---|
| Length | Text → $\log T_\mathrm{gt}$ | Override $T_\mathrm{gt}$ | $\hat T$ tự dự đoán |
| Body A | Text + $T_\mathrm{gt}$, supervision coarse GT | Body GT coarse-then-upsample để cô lập Flow B | Body sample từ Flow A với $\hat T$ |
| Anchors | Text + $T_\mathrm{gt}$, targets theo policy | Mốc trích từ GT, q=1 | Payload, time và q từ predictor với $\hat T$ |
| Fine B | Text + mixture body/anchor, target ở độ dài GT | Cùng $T_\mathrm{gt}$, oracle body và oracle mốc | Text + predicted length/body/mốc |

P1 dùng body GT coarse-then-upsample và anchors oracle, q=1. P2/P3 dùng **60% predicted body + predicted anchors, 30% oracle body + oracle anchors bị jitter, 10% predicted body + no anchors** theo sample. Mọi branch train dùng $T_\mathrm{gt}$ lúc gọi A/planner để target vẫn có cùng frame count. Jitter time đều ±3 frame, clip về miền hợp lệ, giữ payload để mô phỏng lệch thời điểm. Payload noise Gaussian std 0.05 sau normalize, chỉ ở feature valid. Cache lưu pred outputs có $T_\mathrm{gt}$ và config/seed tương ứng. Đo riêng **oracle length** và **predicted length** ở dev. Nếu thêm duration perturbation, thiết kế biến thể riêng với phép resize target theo timestamp được định nghĩa và phép đo timing.

## 6. Training config mặc định

Dùng PyTorch, config validator, data loader và schema có version. Frozen text encoder có thể dùng Transformers. Khóa package versions cùng CUDA/runtime trong lockfile của repo khi triển khai. CPU synthetic trong CI. GPU training mặc định BF16 khi backend hỗ trợ, FP32 fallback. Geometry, normalization accumulators, gap/time math, loss reductions và solver state chạy FP32. [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)

~~~yaml
experiment:
  seed: 17
  run_dir: "runs/slp_multirate"
  protocol_version: "1.0"
data:
  manifest: "data/canonical/manifest.jsonl"
  rig: "data/canonical/rig.json"
  pilot_language: "ASL"
  fps: 25
  min_frames: 8
  max_frames: 320
  bucket_boundaries: [64, 96, 128, 192, 256, 320]
  normalize_std_floor: 0.001
  num_workers: 4
text:
  encoder: "facebook/mbart-large-cc25"
  revision: null  # khóa snapshot hash trước real-data cache
  freeze: true
  max_tokens: 128
  cache_hidden: true
body:
  stride: 4
  width: 256
  blocks: 4
  heads: 8
  dropout: 0.10
anchors:
  count: 8
  policy: "event"
  width: 256
  blocks: 2
  heads: 8
  sigma_frames: 3.0
  nms_min_gap_frames: 2
  turn_min_speed_shoulder_widths: 0.005
  score_weights: [1.0, 0.5, 1.0] # shape, turn, contact
  contact_mining: false
  confidence: "uniform"
fine:
  width: 256
  blocks: 6
  heads: 8
  dropout: 0.10
  local_radius: 8
  state_dim: 204
  correction_enabled: true
  correction_max_rad: 0.35
loss:
  body_endpoint: 1.0
  body_chordal: 0.10
  hand_endpoint: 1.0
  correction: 0.25
  fingertip_fk: 0.25
  wrist_fk: 0.15
  palm_chordal: 0.10
  velocity_fk: 0.10
  contact: 0.0
  endpoint_time_weight: "uniform"
train:
  optimizer: "adamw"
  betas: [0.9, 0.95]
  weight_decay: 0.01
  max_grad_norm: 1.0
  precision: "bf16"
  gradient_accumulation: 4
  microbatch: 4
  schedule: "warmup_cosine"
  warmup_ratio: 0.05
  min_lr_ratio: 0.10
  ema_decay: 0.999
  val_every_updates: 500
  checkpoint_every_updates: 500
  keep_last: 2
  stages:
    length:  {lr: 0.0003, total_updates: 6000}
    anchors: {lr: 0.0003, total_updates: 10000}
    body:    {lr: 0.0002, total_updates: 20000}
    fine:    {lr: 0.00015, total_updates: 35000}
  condition_prob: {predicted: 0.60, oracle_jitter: 0.30, no_anchor: 0.10}
  anchor_jitter_frames: 3
  anchor_payload_noise_std: 0.05
sample:
  solver: "euler"
  body_nfe: 4
  fine_nfe: 8
  cfg: false
  samples_per_text: 1
eval:
  bootstrap_resamples: 2000
  event_window_radius_frames: 3
  timing_match_tolerance_frames: 3
  min_usable_gate_clips: 100
  gate_epsilon: 0.000001
~~~

Mốc total_updates đếm **optimizer updates**. Gradient accumulation chia loss theo số microbatches tương đương. Batch hiệu dụng trên một GPU là 16 nếu microbatches đủ mẫu. Các setting là khởi điểm, chưa đo VRAM hoặc GPU-hours. Khi OOM thử microbatch 2 và accumulation 8 để giữ batch hiệu dụng. Ưu tiên một GPU end-to-end trước. Bucket theo độ dài và pad với masks.

- Warmup round $\max(1,0.05U)$ updates, cosine về 0.1× peak LR. AdamW weight decay trên weights, trừ bias, LayerNorm và one-dimensional scales. Clip gradient sau accumulation và unscale nếu backend yêu cầu. Chọn EMA weights cho validation/sampling nhất quán.
- Val ghi scalars và chạy tập clip fixed. Body chọn checkpoint theo coarse masked endpoint/chordal validation loss và ghi thêm full-rate wrist position proxy từ rest offset. Fine chọn theo primary dev_tune metric và kiểm tra guardrails. Length chọn median relative duration error. Planner chọn weighted time/payload dev losses. Stage ngắn validate ít nhất tại 50%, 75% và mốc cuối.
- Checkpoint ở **optimizer boundary** gồm model, EMA, optimizer, scheduler, scaler nếu dùng, CPU/CUDA RNG, data sampler epoch/offset, update, split hash, rig hash, config hash và dependency versions. Ghi atomic bằng temporary path rồi rename. Resume xác minh hash và seed, luyện tiếp update kế. Lưu best và latest riêng. Dùng run mới khi đổi dataset/config có ảnh hưởng contract.
- Cache condition key gồm clip_id, target length, policy, stage checkpoint hashes, stride, rig/stats hashes, seed và solver NFE. Mismatch buộc regenerate. Cache train tách dev/test. Dùng sampling key độc lập cho body/fine và per-run seed. Dùng cùng noise seeds ở paired comparisons.

## 7. Runbook implementation và pilot

### 7.1 Lệnh và artifact

Các lệnh dưới đây là **contract để agent triển khai**. Đường dẫn và checkpoint đặt trong run directory. Chạy stage sau bằng checkpoint hoặc run manifest chỉ rõ file, không tự dò checkpoint mới nhất từ experiment khác.

~~~bash
# P0 trên CPU. Không cần tải model text; dùng frozen mock encoder cố định.
python -m slp.cli synthetic --out data/synthetic --seed 17
python -m slp.cli prepare --source data/synthetic/raw.jsonl --adapter synthetic --out data/synthetic/canonical
python -m slp.cli audit --manifest data/synthetic/canonical/manifest.jsonl --split all
python -m slp.cli smoke --config configs/pilots/p0.yaml --device cpu
python -m pytest -q

# Dữ liệu thật. FILES và RIG là các input do project cung cấp.
python -m slp.cli prepare --source FILES --adapter signspark_lmdb --rig RIG --out data/canonical
python -m slp.cli audit --manifest data/canonical/manifest.jsonl --split train,dev,test --out runs/audit
python -m slp.cli make-cohort --manifest data/canonical/manifest.jsonl --pilot p1 --seed 17 --out cohorts/p1.json

# Ví dụ P1. B = stride 4 + uniform, D = stride 4 + event.
python -m slp.cli train --pilot p1 --variant B --stage fine --cohort cohorts/p1.json --seed 17
python -m slp.cli train --pilot p1 --variant D --stage fine --cohort cohorts/p1.json --seed 17
python -m slp.cli evaluate --pilot p1 --variant B --protocol oracle --split dev_gate --seed 17
python -m slp.cli evaluate --pilot p1 --variant D --protocol oracle --split dev_gate --seed 17
python -m slp.cli compare --baseline runs/p1/B/17/eval/oracle/dev_gate --candidate runs/p1/D/17/eval/oracle/dev_gate --gate p1
~~~

P2/P3 làm theo dependency graph:

~~~text
prepare → audit → frozen text cache
                 ├─ train length (một lần cho mỗi seed)
                 ├─ train body  (một lần cho mỗi stride/seed)
                 └─ train anchors (một lần cho mỗi policy/seed)
length/body/anchors + cohort → cache conditions → train fine theo variant
fine + frozen checkpoint bundle → sample → evaluate → compare → gate
~~~

Thêm subcommands cache-text, cache-conditions, infer, evaluate, compare và verify-run. CLI infer bắt buộc có text hoặc JSONL texts, checkpoint bundle, rig, seed và output. Nếu caller bật controls ngoài text, label output protocol riêng. Default infer không nhận T, body/mốc từ GT. Mỗi stage cần test resume và kiểm tra input stage requirements. Stage P1 chạy fine trực tiếp với oracle provider nên không phụ thuộc checkpoint body/length/anchors. Dùng một stage registry để hiển thị đúng dependency theo pilot.

Mỗi run ghi config.resolved.yaml, versions.json, dataset/audit/cohort manifests, metrics.jsonl, checkpoints, seed/index policy và stage_condition_hashes.json. Mỗi eval xuất per_clip.parquet hoặc JSONL, summary.json, latency.json và report.md. Ghi n hợp lệ cho từng metric, các slice, nguồn event labels, SI units, số model evaluations và số mặc định/time rounding. Ghi cả lỗi, timeout, OOM, missing checkpoints. Không chỉ lưu variant thắng.

### 7.2 P0 · dữ liệu, geometry và sanity

**Synthetic fixture.** Tạo rig đầy đủ slots với offsets giả lập hợp lệ, một chuỗi tay đổi hình ngón ở giữa, một chuỗi hai tay có contact đã biết, clip ngắn và dài lẻ. Fake text encoder là hàm cố định hoặc tiny frozen embedding. Lưu fixture và golden tensors có seed. Demo render vài clip thật bằng cùng rig/camera để xem wrist axis, vai/khuỷu, trái/phải, contact, pelvis frame. Artifact thực của người ký hiệu chỉ dùng nơi phù hợp với quyền dữ liệu.

**Smoke mặc định.** Batch 2, T trong [8,24], width 32, 1 block, 4 heads, FP32, 20 optimizer updates; 2+2 NFE với random init. Chạy prepare → train từng stage → infer → evaluate → resume. Thêm tiny-overfit trên 8 clip cố định, FP32, 300 updates, xác nhận loss trên cùng clip/noise giảm rõ từ đầu đến cuối, khởi điểm yêu cầu giảm ít nhất 50% so với trung bình 20 update đầu. Đây là test pipeline, không dùng để dự đoán benchmark.

**Real-data audit.** Kiểm 32 clip phân tầng theo độ dài/tay làm ký hiệu/occlusion, ghi tỷ lệ dữ liệu sạch và các lỗi. Kiểm 100% manifest về timestamps và split overlap xác định được. Oracle encode/decode giữ cùng rig và cùng representation ở frame hợp lệ; với fill hoặc correction cap, so đúng contract và báo khác biệt do thay đổi intentional. Nếu contact labels có sẵn, kiểm tối thiểu 30 positive/near-miss observations trước khi bật contact metrics.

**P0 gate.** Tất cả unit/smoke tests pass, không NaN/Inf, rig left/right và axes xác minh, split hash/provenance rõ, và dev_gate còn tối thiểu 100 clip có event windows và tay hợp lệ cho hai policy. Nếu ít hơn, báo thiếu dữ liệu trước khi diễn giải pilot. Khi dữ liệu tay fit sai hình, giảm phạm vi cohort theo tiêu chuẩn chung rồi audit lại hoặc đổi adapter.

### 7.3 P1 · oracle proof of concept

- Cohort đề xuất **tối đa 512 train, 128 dev_tune, 160 dev_gate**, stratified theo độ dài, số tay, mật độ sự kiện. Profile bước đầu; giảm cỡ cohort chỉ nếu mọi variant cùng giảm và vẫn đủ n cho gate.
- Train **B** (stride 4, uniform) và **D** (stride 4, event) với body oracle coarse-then-upsample, cùng T và 8 anchors oracle, cùng payload, q=1. P1 pipeline không dùng checkpoint body/planner. Thêm một fine no-anchor cùng 204D và 8 slot masked nếu chi phí cho phép để đo mốc có ích tuyệt đối hay không.
- Dùng 2.000 optimizer updates mỗi variant, microbatch 4 × accumulation 4, LR fine 1.5e−4, warmup 100, một seed 17. Validation tại 1.000, 1.500 và 2.000. Dùng 8 NFE fine cho evaluation, chung noise seed theo clip. Nếu cả hai còn chưa hội tụ, quyết định tăng đồng đều trước khi đọc dev_gate.
- **Primary** là mean wrist-local fingertip error (mm) trong event windows ±3 frame cố định theo reference, sau khi áp joint/visibility masks. Tính per-clip rồi macro mean. Baseline dùng nguyên cohort và event masks. Báo cả điểm **ngoài cửa sổ sự kiện** và điểm **bỏ những frame kề mốc oracle** để đo khả năng nội suy quanh mốc.
- **GO.** D giảm primary ít nhất 5% so với B, cận dưới 95% paired bootstrap CI của improvement > 0. Guardrails là all-frame wrist error tăng tối đa 5% và outside-event fingertip error tăng tối đa 3%. Khi có >=30 clip có nhãn contact đáng tin, contact F1 không giảm quá 2 điểm phần trăm. Đây là ngưỡng kế hoạch; có thể hiệu chỉnh sau audit P0 và trước khi xem dev_gate.
- **HOLD.** improvement dương nhưng CI còn qua 0, hoặc event cohort quá ít. Cho phép một lần tăng số clip/seed hoặc sửa event mining có ghi phiên bản, rồi dùng một dev gate mới hoặc báo chẩn đoán rõ là exploratory. **STOP hướng mốc.** D ngang/thua B ổn định hoặc làm xấu nhịp/hand fidelity ngoài các mốc. Kiểm tra P0 và anchor-window leakage trước khi quyết định có chạy một hướng model khác.

### 7.4 P2 · predicted anchors thật sự giúp gì?

- Cohort đề xuất **tối đa 1.500 train, 200 dev_tune, 200 dev_gate**. Train length 1.500 updates, body stride 4 6.000, anchors 3.000 mỗi policy, fine 6.000 mỗi variant. Freeze các stage để dùng chung body/length giữa B và D. Dùng anchor conditioning mixture §5.3. Train lại fine cho chế độ no-anchor có compute tương đương nếu baseline no-anchor P1 chưa từng thấy body dự đoán.
- Báo theo thứ tự **oracle body + oracle anchors**, **predicted body + oracle anchors + GT length**, **predicted body + predicted anchors + GT length**, **text-only hoàn chỉnh với predicted length**. Thêm condition swap hữu ích là oracle body + predicted anchors với GT length. So q=1 và learned q trên cùng planner frozen nếu bắt đầu triển khai confidence.
- Cùng $T_\mathrm{gt}$, đo sai số frame-to-frame theo timestamp. Khi $\hat T$ khác, báo spatial comparison ở normalized time như một chỉ số hình học riêng, còn duration, event timing và contact report theo giây thật. Không dùng DTW làm thước đo timing chính. Cùng paired seed khi T khác thì dùng generator seed theo clip/stage và log actual T.
- **GO.** Ở **predicted anchors + GT length**, D cải thiện primary P1 ít nhất 3%, cận dưới paired CI > 0 và qua guardrails P1. Ở **text-only hoàn chỉnh**, D có spatial primary theo normalized time không xấu hơn B, duration error median không tệ hơn 2 điểm phần trăm, và event timing hoặc recall trong tolerance không xấu hơn theo cùng protocol. So thêm với no-anchor. Nếu có nhóm đủ nhãn, dùng pilot người xem ẩn text để kiểm tra lỗi nghĩa rõ ràng.
- **Chẩn đoán nếu trượt gate.** oracle tốt nhưng predicted kém → kiểm anchor time/payload, length và confidence. Predicted anchor tốt khi body oracle nhưng mất gain ở body predicted → kiểm wrist target/correction clip cap và thử body stride 2. Spatial tốt mà thời điểm lệch → audit duration và anchor clocks. Nếu predicted anchors chẳng hơn no-anchor trên dev, bỏ nhánh đó hoặc giới hạn kết luận ở mốc được cung cấp.
- Ngưỡng trên phục vụ quyết định đầu tư, chưa đủ để kết luận về độ hiểu. Khi proxy tăng nhưng signer nhận sai nhiều hơn, ưu tiên kết quả ngôn ngữ.

### 7.5 P3 · ma trận xác nhận

| Variant | Body stride | Hand rate | Mốc | Stage dùng chung |
|---|---:|---:|---|---|
| A | 1 | full | uniform | Body với C; uniform planner với B |
| B | 4 | full | uniform | Body với D; uniform planner với A |
| C | 1 | full | event | Body với A; event planner với D |
| D | 4 | full | event | Body với B; event planner với C |
| E | 4 | full | onset_mid_offset | Body với B/D; planner riêng |
| F | 4 | full | none | Body với B/D; anchor slots masked |

Full runs dùng ba seed **17, 23, 41**, split/hash cố định và cùng train sampling contract. Dùng bảng mặc định §6 hoặc lịch giảm đã chốt trước benchmark. Thử B/D trước, rồi đủ A/C và E/F khi P2 pass. Ablations ưu tiên ở D gồm q=1/learned q, correction on/off, endpoint-only vs +FK/+velocity và NFE 2+4/4+8/8+16. Giữ chỉ một thay đổi mỗi ablation. Add stride 2 nếu stride 4 hỏng contact/chuyển động đầu. Geodesic là phần mở rộng sau khi train và eval Euclidean ổn.

- So chính A↔B, C↔D cho stride; A↔C, B↔D cho mốc; B↔E đối chiếu boundary; D↔F kiểm lợi ích anchor. Báo interaction effect nếu đủ seed/cỡ mẫu. Báo P3 test một lần với cấu hình khóa.
- **Hai bảng compute.** Bảng đầu giữ width/blocks/NFE để đo tác động thực của stride. Bảng hai chọn cấu hình trên dev có FLOPs hoặc latency trong ±10%, rồi so chất lượng. Cùng GPU, batch, độ dài, precision, NFE and cached/uncached policy. Đo riêng text/planner, body, fine, FK và end-to-end. Warmup 10, đo ít nhất 50 lượt có đồng bộ GPU. Báo P50/P95, peak memory, FLOPs theo phương pháp estimator hoặc profiler đã ghi. Frame-token count chỉ là chẩn đoán. Với T=120 và 4+8 NFE, stride 1 khoảng 1.440 lượt frame-token và stride 4 khoảng 1.080 trước layer/attention overhead.
- Nếu đánh giá multiple samples, chốt sample indices ngay lúc run. Báo mean và phân bố lỗi, bao gồm tỷ lệ mẫu không thể nhận đúng. Oracle best-of-K chỉ có thể làm phép chẩn đoán riêng.

## 8. Evaluation chi tiết

### 8.1 Căn chỉnh và metric

- **Event-near fingertip mm.** Mỗi event window ±3 frame ở 25 FPS tương ứng khoảng ±120 ms. Dùng event times của reference chung, hợp/tách windows bằng quy tắc cố định. So wrist-local tip points ở frame có quan sát và cùng rig. Convert sang mm bằng 1000, bỏ frame/tip invalid. Tính macro per-clip. Dùng cùng tập clip intersection ở paired comparisons, báo excluded n và lý do.
- **Wrist, palm và body.** Wrist XYZ error 3D trong hệ pelvis theo mm. Palm geodesic angle trong hệ pelvis $ \arccos(\mathrm{clip}((\mathrm{tr}(\hat R^\top R)-1)/2,-1,1)) $ đổi sang độ. Báo body MPJPE đơn vị mm trên cùng rig. Nếu báo DTW-JPE/DTW-PA-JPE để đối chiếu paper thì cố định joints, alignment, scaler và evaluator; ghi protocol được dùng. Projection hoặc alignment của metric không thay đổi motion đầu ra.
- **Timing và contact.** Đánh giá length error $|\hat T-T|/T$. Dựng một detector evaluation cố định cho cả variants, tách khỏi mốc mà model nhận. Cho sự kiện shape/turn, tính các tín hiệu động học theo §3.2 trên chuỗi sinh, dùng threshold đã khóa trên dev_tune và NMS cùng khoảng cách. Detector phải cho số event thay đổi, không ép top K. Cho contact, dùng cặp landmarks/điểm bề mặt đã định danh, bật khi khoảng cách dưới ngưỡng enter θ trong ít nhất 2 frame và tắt khi vượt 1.5θ trong ít nhất 2 frame. Chọn θ trên dev_tune có nhãn trong lưới 0.01h, 0.02h, 0.03h, dùng giá trị giảm false positives khi F1 hòa. Thời điểm onset/offset theo frame đầu của đoạn xác nhận. Nếu geometry proxy không qua audit thủ công, giới hạn contact eval ở subset được annotate trên outputs hoặc báo thiếu nhãn. Match events theo loại/cặp trong tolerance ±120 ms ở 25 FPS, assignment một-một tối đa match rồi giảm tổng sai số timestamp. Báo precision/recall/F1, false positives, missed events và MAE timing của cặp match. Contact frame F1 dùng pair labels và mask chung khi đủ supervision. Với $\hat T\neq T$, giữ timestamps thật và đếm event ngoài thời lượng chuỗi kia là miss/false positive. Không warp trước khi chấm timing.
- **Trong/ngoài event, ngôn ngữ và độ dài.** Cắt slice theo clip hai tay, type event, clip dài, chất lượng fit. Gộp sample macro trước, sau đó aggregate; ghi n và CI. Nếu dùng recognizer hoặc back-translation, khóa checkpoint ngoài pipeline, báo nguồn và cùng protocol; đối chiếu độ nhạy với perturbations. Các proxy này bổ sung cho bài kiểm tra độ hiểu.
- **Signer evaluation.** Khi mở rộng P3, chọn cặp tối thiểu đã được kiểm định với người dùng ASL và tập hiểu câu. Cùng rig, góc, scale và frame rate. Ngẫu nhiên hóa/ẩn model, giữ text nguồn ẩn cho bài hiểu. Dùng rubric hoặc forced choice rõ ràng, báo phương án từ chối/không xác định, rating độ tự nhiên riêng, annotator agreement hoặc uncertainty. Có thể bắt đầu với 80–120 prompt độc lập, tối thiểu 3 người đánh giá có năng lực phù hợp nếu điều kiện cho phép. Chốt kích thước nghiên cứu theo variance pilot, sự đồng thuận của người tham gia và quy trình project. [Phonological Perception of Sign Language Models](https://arxiv.org/abs/2606.28667), [BackTranslation2.0](https://arxiv.org/abs/2606.28673)

### 8.2 Paired statistics và anomaly review

Định nghĩa improvement $(\bar e_\mathrm{base}-\bar e_\mathrm{cand})/\max(\bar e_\mathrm{base},10^{-6})$. Bootstrap 2.000 resamples theo clip hoặc origin-video cluster khi có nhiều clip cùng video. Giữ predictions base/candidate gắn cặp. Với 3 seed, báo mỗi seed và mean; bootstrap hierarchical seed/video khi đủ nhóm và ghi lựa chọn. Đối với rate, báo absolute percentage points. Khai báo primary metric, guardrails và threshold trước khi chấm dev_gate. Aggregate báo cả count excluded và n tín hiệu có nhãn. Kiểm 10–20 case improvement lớn/degradation lớn để phát hiện scale, handedness, hand identity hoặc bad event alignment. Case review không được thay đổi metric test sau khi đã xem kết quả.

## 9. Kiểm thử và tiêu chuẩn bàn giao

| Test | Điều kiện pass |
|---|---|
| Rotation 6D round trip | Identity, phép quay 90° không đối xứng, random rotation và input suy biến hữu hạn. $R^\top R≈I$, $\det R≈1$ với tolerance 1e−4 FP32, kiểm row/column adapter |
| SO(3) Log/Exp và FK | Khoảng gần 0, gần $\pi$, correction clamp và hai tay. XYZ fixture đúng trong 1e−5 m. Thay rotation wrist chỉ thay wrist orientation/descendants theo geometry |
| Lưới thời gian | Stride 1 identity, stride 4 có bin cuối ngắn, T=1, T=5, timestamps không đều nếu adapter cho phép. Upsample thời điểm chính xác, không đọc padding |
| Shape, normalization và mask | Batch mixing T và T_c, pad contamination test, all-masked group loss finite zero, time diff bỏ cặp thiếu. Normalize→denormalize ổn định theo std floor |
| Anchor policies | Same K, no duplicates, stable tie break, missing segments/contact fallback, time sorted, all-invalid/no-anchor đúng residual zero. GT label leakage test chặn predictor đọc pose và T ở regime text-only |
| Fine correction target | Cùng GT rotations nhưng hai body conditions khác cho $\Delta^\star$ khác. Vòng build/decode phục hồi GT khi trong cap |
| Endpoint objective | Với fake perfect predictor, endpoint losses xấp xỉ 0. Check gradient hữu hạn khi mask rỗng và khi padded NaN đã được sanitize ở input |
| Euler sampler | NFE chính xác cho N=1/2/4/8; perfect constant endpoint trả đúng; solver padding zero; stage noises độc lập, kết quả deterministic theo seed trong tolerance |
| Resume và run contract | Stage bị ngắt ở optimizer boundary rồi resume có step, LR và seed tiếp theo đúng. Wrong rig hash/split hash báo lỗi thay vì chạy ngầm |
| End-to-end | CPU synthetic train từng module → sample → eval. Một real clip có thể decode, render nếu renderer sẵn và kiểm overlay side-by-side. Không yêu cầu GPU để chạy unit tests |

**Bàn giao tối thiểu.** Repo có config base/variants/pilots, CLI làm được P0, các stages train độc lập, inference từ text, scripts compare/gates và manifest mô tả inputs có quyền sử dụng. Kết quả bàn giao gồm lệnh tái lập, mã phiên bản, checkpoint nếu có, JSON measurements và một báo cáo ghi lựa chọn GO/HOLD/STOP. Khi chưa có dữ liệu hoặc GPU, vẫn bàn giao CPU smoke và ghi cụ thể bước real-data còn chờ đầu vào nào. Không mô tả pilot là đã pass nếu chưa chạy.

## 10. Nguồn và giới hạn kiến trúc

Thiết kế hiện thực hóa cascade Euclidean để tách tác động của body stride và sự kiện. MotionHiFlow có cơ chế chuyển noise giữa các scale khác với cascade độc lập ở đây. SignSparK là đối chứng gần cho sparse keyframes, BLANC cho multi-rate và SignNMFlow cho geometry. Con số config/threshold là đề xuất thực nghiệm. [MotionHiFlow](https://arxiv.org/html/2604.23264v1), [SignSparK](https://arxiv.org/html/2603.10446v4), [BLANC](https://arxiv.org/abs/2610.24109), [SignNMFlow](https://arxiv.org/abs/2609.33339)

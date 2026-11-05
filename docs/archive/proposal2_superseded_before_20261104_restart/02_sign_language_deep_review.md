# Đọc sâu sign language generation

## 1. SOKE — Signs as Tokens

Nguồn: [arXiv v3, §3–4, Tables 1–3](https://arxiv.org/html/2411.17799v3). Accepted ICCV 2025 được xác nhận trên [metadata chính thức](https://arxiv.org/abs/2411.17799v3).

**[E] Bài toán và cơ chế.** SOKE đưa text→3D sign vào multilingual LM. DETO gồm tokenizer riêng cho body, left hand và right hand. Input 133 chiều gồm upper-body rotations, hai tay và expression coefficients. Codebook body có 96 entries, mỗi tay 192; latent dimension 512. Generator dùng mBART-large-cc25; ba output heads sinh đồng thời một bộ part tokens tại mỗi bước. Input decoder là trung bình có trọng số của ba embeddings. Đây là **parallel theo bộ phận nhưng autoregressive theo thời gian**, không phải sinh cả câu song song.

**[E] Retrieval.** Dictionary từ isolated-sign resources/online sources được chuyển thành motion tokens; từ trong câu được đối chiếu với dictionary và thêm các exemplar vào encoder prompt. Một từ có nhiều exemplar thì giữ bản có reconstruction error thấp nhất. Vì có external dictionary, dòng SOKE đầy đủ khác dòng no-dictionary về nguồn dữ liệu và conditioning.

**[I] Giá trị cho proposal.** Decoupling giảm việc body motion lấn át fingers; LM cung cấp prior ngôn ngữ. Nhưng average embedding và conditionally independent heads không trực tiếp ràng buộc vị trí chạm giữa hai tay. Tokenizer tốt trên reconstruction không đảm bảo token được dự đoán đúng. Expression hiện có cũng không chứng minh bao phủ đầy đủ non-manual markers.

**[P] Đối chứng đúng.** Giữ SOKE đầy đủ để trả lời mục tiêu gốc; thêm SOKE no-dictionary và MaDiS để tách retrieval với model paradigm. Không thể dùng một baseline SOKE bị bỏ dictionary rồi tuyên bố vượt dòng SOKE Table 1.

## 2. MaDiS

Nguồn: [v2, §3–4, Tables 1–6](https://arxiv.org/html/2601.19577v2). Ghi nhận là preprint trong khảo sát này; metadata đã đọc chưa xác nhận venue.

**[E] Những gì đã được giải.** MaDiS dùng chính tokenizer SOKE, chuyển Qwen3-0.6B-Base sang bidirectional attention. Pretraining trên ba training sets kết hợp có ba objective: masked token prediction; dự đoán codebook embeddings qua MLP; decode dự đoán để supervise pose. Fine-tuning theo từng dataset giữ text không bị mask. MoP dùng codebook embeddings và gate học được để trộn các bộ phận. Vì vậy “tri-level physical supervision” và “adaptive part fusion” không thể tái giới thiệu làm contribution mới.

**[E] UTC và độ dài.** UTC ràng buộc trạng thái ở noise levels 0.75/0.5 để các token đã mở phủ đều thời gian, rồi chọn theo confidence trong từng giai đoạn. Default 100 vị trí motion token, 4 token positions/step, 25 bước; decoder upsample 4 lần, tương ứng capacity 400 frames. EOS cắt phần đuôi. Lập luận giảm số thứ tự unmask là tính tổ hợp, không phải bảo đảm chất lượng ngữ nghĩa.

**[I] Khoảng trống còn lại.** Uniform temporal checkpoints khác với các điểm thay đổi handshape/contact. Confidence của categorical prediction không phải xác suất hình học hoặc dịch đúng. MaDiS vẫn sinh trên representation lượng tử; physical training không biến output cuối thành continuous correction ngoài codebook. Tuy nhiên, độ chính xác của tokenizer tốt có thể làm lợi ích correction rất nhỏ: đây là giả thuyết cạnh tranh cần đo trước.

**[E] Evaluation.** Paper dùng DTW-JPE không PA; thêm SiBLEU token overlap và SiCLIP retrieval. User study CSL quy mô nhỏ có lợi cho MaDiS, nhưng không chứng minh calibration của metric ở mọi ngôn ngữ. Phiên bản code đã kiểm tra có sai khác với tên SiBLEU-4: xem [07](07_code_audit.md). Do đó số liệu paper là reported results, chưa là baseline reproduced ở workspace.

## 3. SignSparK

Nguồn: [§3–4, Tables 2–8](https://arxiv.org/html/2603.10446), [repo chính thức](https://github.com/JianHe0628/SignSparK). Repo ghi ECCV 2026; phân biệt paper với release mới.

**[E] Cơ chế.** FAST gán BIO labels từ hai luồng hand features. Keyframe policy lấy onset, midpoint, offset của segment. Generator học tái tạo dense motion từ sparse keyframes và text bằng flow/reconstruction/temporal-velocity objectives. Body dùng 6D rotations, hands dùng MANO, face dùng FLAME. Code release huấn luyện theo stream body, hand, face.

**[E] Ba regime phải phân biệt.** KF2P lấy keyframes từ target sequence: đây là completion có spatial evidence. GF-T2P không có keyframe. SR-T2P dùng text→gloss prediction để retrieve dictionary keyframes, đặt tại khoảng cách cố định rồi sinh motion. Không được lấy KF2P error thấp để quảng bá khả năng text-only. SR-T2P có thêm text-to-gloss model, khác word-based retrieval prompt của SOKE. How2Sign không được báo SR-T2P trong Table 3 vì không có gloss annotations như hai dataset còn lại.

**[I] Bài học.** Sparse control hiệu quả là bằng chứng cho việc dùng articulatory evidence; chưa chứng minh keyframe từ model đủ đúng để hard clamp. Chính paper đã có zero-keyframe inference, nên không nên nói mô hình chỉ chạy được với GT keyframes. Đề xuất mới phải so với cả zero-KF và predicted/retrieved-KF settings.

**[E] Hai vấn đề tái lập.** README release thông báo fitting được cải thiện và checkpoint được train trên lượng dữ liệu lớn hơn bản paper; BT cũng được retrain. Ngoài ra công thức velocity target trong paper và code snapshot không đồng nhất; code dùng đạo hàm của linear interpolation đúng kiểu standard CFM. [07](07_code_audit.md) chỉ ra dòng cụ thể. Không suy từ lỗi ký hiệu này rằng toàn bộ kết quả SignSparK sai.

## 4. Các đối thủ làm thu hẹp novelty

| Nguồn | Bằng chứng chính đã đọc | Hệ quả thiết kế [I/P] |
|---|---|---|
| [Neural Sign Actors, §3–5](https://arxiv.org/html/2312.02702) | Diffusion trên SMPL-X với anatomical graph và pipeline fitting cho How2Sign | Direct continuous sign generation và anatomical prior đã có tiền lệ |
| [SIGNER, §4–5](https://arxiv.org/html/2506.07460v2) | Text→gloss; dự đoán duration; local temporal fusion; inter-part attention; part-wise VQ | Không claim temporal grounding hoặc inter-part attention là mới. Gloss supervision cần ghi riêng |
| [M3T, §3–4](https://arxiv.org/html/2603.23617) | Representation SMPL-FX/FLAME giàu hơn, modality-specific FSQ và auxiliary translation | Thêm face tokens/FSQ/translation loss đều đã có. Fitting giàu hơn là thay đổi dữ liệu, không thuần architecture |
| [SignRR, §3–4](https://arxiv.org/html/2608.28568) | Gloss→pose, train-derived dictionary, part-wise RVQ refinement và joint decoder, latent length adjustment | “Retrieve and refine” không mới. Không xếp dòng GT-gloss cùng text-only nếu chưa có T2G |
| [SignFlow, abstract chính thức](https://icmi.acm.org/2025/proceedings/) | Conditional flow matching cho one-to-many SLG | Flow matching cho SLG tự nó không mới; chưa dùng số liệu vì chưa đọc đầy đủ protocol |

### SIGNER: đối chứng đặc biệt quan trọng

**[E]** Full text v2 của `2506.07460` là SIGNER, không giữ tên GLOS của v1. Nó đã có cả duration prediction lẫn tương tác body–hands–face, và đưa gloss semantics vào các vị trí thời gian qua local fusion. **[P]** Nếu phát triển event/duration planner, phải so trực tiếp với SIGNER hoặc một tái lập cùng supervision; không đặt word embedding theo thời gian và gọi đó là ý tưởng chưa có.

### M3T: giới hạn của benchmark cũ

**[E]** M3T phân biệt hạn chế của fitting với khả năng biểu diễn nguyên thủy của SMPL-X; richer facial recovery cần estimator và representation phù hợp. **[I]** Nếu giữ pseudo-GT expression chất lượng thấp của SOKE, mô hình không có đủ supervision để chứng minh sinh non-manual grammar tốt. Benchmark body/hand SOTA là một kết quả hẹp hơn linguistic completeness. **[P]** Rich-face experiment là một nhánh riêng với cùng fitting cho mọi baseline, không lặng lẽ thay target rồi so với bảng cũ.

## 5. Phản bằng chứng: metric có thể tăng mà conditioning vẫn yếu

**[E]** Conditional Collapse phân biệt ảnh hưởng của input, output diversity và độ gần đúng target bằng latent-distance ratios, so với random pairing và GT-vs-GT reference. Bản mới nhấn mạnh coverage; không đưa một threshold tuyệt đối dùng cho mọi dataset. [Conditional Collapse, §3–6](https://arxiv.org/html/2606.01643).

**[I]** Không được ngoại suy rằng công trình này chứng minh MaDiS hay mọi SLG model đều collapse: phạm vi checkpoint, evaluator và representation của nó phải được tôn trọng. Tuy nhiên đây là lý do đủ mạnh để thêm text-shuffle/random-pair diagnostics và kiểm tra intelligibility bởi signers vào kế hoạch.

## 6. Các mâu thuẫn được giải quyết thế nào

| Quan sát tưởng mâu thuẫn | Cách diễn giải có căn cứ |
|---|---|
| VQ mạnh trên HumanML3D nhưng MoMask++ chưa mạnh nhất trong bảng MaDiS | Domain, representation và conditioning khác; số liệu motion không chứng minh transfer SLG. Cần sign-specific controls |
| SignSparK đạt error thấp hơn nhưng MaDiS tự báo SOTA | Khác ngày công bố, regime, fitting và metric suite; không có một scalar SOTA dùng chung |
| M3T cải thiện mặt trong khi SOKE cũng có expression coefficients | Có trường expression không đồng nghĩa fitting chứa thông tin non-manual đủ giàu |
| Refiner cần sửa lỗi nhưng cũng phải giữ semantic anchors | Reliability chỉ là giả thuyết giải trade-off; cần compare hard clamp, no anchors, constant weight và calibrated weight |

**Kết luận thiết kế [P]:** sử dụng MaDiS làm semantic prior khả thi; nghiên cứu correction dưới predicted conditioning và kiểm tra relational articulation. Không đặt trọng tâm vào một kiến trúc masked/flow mới nói chung. Điều kiện để giữ hướng là cải thiện **cả** protocol tái lập và semantic checks, không chỉ tăng một token-overlap score.

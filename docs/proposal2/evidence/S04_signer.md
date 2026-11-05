# S04 — SIGNER: target-order gloss timeline và local fusion

**D có phạm vi, chưa code.** Đã đọc main v2/Tables 1–5 và [official four-page supplement](https://taeryunglee.github.io/projects/signer/asset/SIGNER_ECCV2026_suppl.pdf) về evaluator/human study. Nguồn: [arXiv 2506.07460v2](https://arxiv.org/html/2506.07460v2), [project](https://taeryunglee.github.io/projects/signer/). Authors ghi ECCV 2026, chưa đối chiếu proceedings entry. V1 tên GLOS, không là independent study. Supplement trích bằng PyMuPDF, không validated page anchors; không tự gán code-audit C.

## Task và temporal conditions

Text→off-the-shelf text-to-gloss→predicted durations→motion. Có gloss supervision và T2G dependency, không cùng training regime với gloss-free model. Main inference dùng predicted gloss/duration, Table 3 tách oracle controls. CSL-Daily/Phoenix, không How2Sign; max motion 180 frames.

Duration MLP nhận mBART gloss embeddings, dự đoán frames/gloss. Repeat embeddings theo duration rồi concatenate theo gloss order để tạo time-resolved condition. Main/supplement chưa rõ đủ duration-label extraction, rounding/loss và alignment targets; không tự thêm MAS/CTC rồi gán cho authors.

## Representation và generator

529D/frame: 43 joints×(3 positions+6 rotations+3 velocities), cộng 10 expressions/3 jaw. Four-part VQ cho body/LH/RH/face, khác SOKE 133D. Discrete diffusion có retention/replacement/masking, 14 layers/hidden 512/16 heads, 50 steps. Negative log-probability clean-token objective sau codec training. AdamW 1e-4, 30K iterations/stage, batch 56, RTX 4090.

Inter-Part Attention dùng body hub trao đổi với hands/face. Local Temporal Fusion tạo per-timestep AdaLN scale/shift từ timed gloss condition rồi temporal Conv1D kernel 3. Không phải global cross-attention tới mọi gloss ở mỗi position.

Temporal conditioning là inductive bias, không hard guarantee không bỏ/đổi sign. Model có thể ignore condition, còn self-attention/conv mang broad context. Part codec/cross-part denoiser/local timeline đã là prior art.

## Ablations và phản bằng chứng

Table 1 CSL: global cross-attention BT BLEU-4 5.81, AdaLN+FC 7.85, LTF 15.60. DTW body/hand .251/1.258→.242/1.189; semantic proxy gain lớn hơn geometry gain, cần independent evaluator/comprehension.

Table 2 no-IPA BT 13.27→15.60; full không best mọi joint-error component. Table 3 alternative T2G cho 14.78–14.87 so off-the-shelf 15.60 và GT gloss 15.69. Không kết luận T2G quality không quan trọng từ một pool gần nhau.

GT duration giảm hand DTW 1.189→.448 nhưng BT 15.60→14.44. Các perturbations duration cho 15.18/14.47/14.06, chưa là immunity với timing errors. Mean khoảng 16 frames/gloss không có nghĩa mọi sign bằng duration.

## Evaluator và human contract

Table 4 CSL DTW body/hand .242/1.189, BT 15.60; Phoenix .229/1.096, BT 11.46. SOKE retrain cùng representation/evaluator cho 4.09/5.52, khác SOKE original 11.30/11.87; không xếp hạng cross-paper BT từ tên metric.

Supplement evaluator thích nghi TwoStream recognition+translation, dùng positions/rotations/face và mBART initialization. Reference BT CSL 22.44/Phoenix 20.73. Với previous skeleton evaluator, SIGNER 11.78/8.76, SOKE 4.17/6.81. Direction gain trong study còn, magnitude đổi theo evaluator; chưa xác nhận đủ NMM semantics.

Peak velocity/jerk thấp chưa loại over-smoothing nếu không so đúng reference dynamics. Lexical strokes/holds không theo quy tắc càng ít motion càng tốt.

Human study 23 participants, sáu samples/method, ratings 0–5: semantic 4.69/naturalness 4.77. Qualification, allocation theo language và uncertainty chưa đủ để suy native Deaf comprehension. Rating chưa là blind meaning recovery.

## Chuyển giao

Target-order structure và timed-condition fusion là hai vấn đề riêng; có gloss embeddings chưa đủ nếu injection trộn semantics theo time. Tuy nhiên paper chưa chứng minh gloss-free models bắt buộc thêm gloss hoặc hard locality.

Candidate planning cần giải thích nguồn supervision và predicted inference trên How2Sign, so SIGNER/temporal-gloss baselines ở matched tracks, và đo cả order, articulation, NMM/comprehension. Ghép duration + part attention không tự là novelty.

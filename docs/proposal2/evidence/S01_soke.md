# S01 — SOKE / Signs as Tokens

**D+C có phạm vi.** Đã đọc main method, setup, kết quả và ablations, human study, các appendix fitting/tokenizer; đã kiểm tra static code ở những đường nêu bên dưới. Chưa xem hết video, chạy model hoặc audit toàn bộ generation/back-translation pipeline.

Nguồn: [ICCV 2025 proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html), [arXiv 2411.17799v3](https://arxiv.org/html/2411.17799v3), [official code](https://github.com/2000ZRL/SOKE). Snapshot mới tại `sources/soke/`; code commit `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`. Không dùng implementation hoặc logs thực nghiệm nội bộ SLP.

## Bài toán, dữ liệu và fitting

SOKE sinh 3D sign motion từ text cho How2Sign/ASL, CSL-Daily/CSL và PHOENIX-2014T/DGS. Full model dùng multilingual training và dictionary retrieval; tokenizer còn được train với word-level dictionary motions. No-retrieval ablation và full system phải ở đúng tracks, không coi full score là equal-data text-only baseline.

How2Sign dùng fitting từ NSA. CSL/Phoenix dùng OSX body, WiLoR hands và Mediapipe reprojection refinement. Đây là pseudo-3D reference, không phải mocap ground truth trên toàn ba datasets. Appendix A kiểm tra fitting trên SGNify mocap: body vertex error 46.73 mm so với NSA 46.42; left/right hand 10.55/8.94 so với 16.17/15.23. Kết quả hỗ trợ cải thiện lỗi tay trên subset đó, chưa xác nhận mọi handshape hoặc facial marker được fit đúng.

## Representation và DETO

Code xác định 133 parameters/frame: 30 body rotation dimensions, 45 mỗi hand, 3 jaw và 10 expression. Như vậy có **10 active body joints cộng jaw**, không phải 11 body joints ngoài jaw. Root, 11 early body joints và shape bị bỏ khỏi learning vector; decode thêm zero poses và fixed shape.

DETO có ba VQ-VAE streams B/LH/RH. Body stream 43D gồm 30 body, jaw và expressions; hai hands mỗi stream 45D. Encoder là temporal CNN, nearest-code quantization và part decoder. Codebooks body 96, mỗi hand 192, code width 512. Objective gồm reconstruction, embedding/commitment terms.

Tách capacity giúp tay có codebook riêng nhưng không tự đảm bảo relational consistency. Jaw và expressions được học chung B ở default branch; có face-VQ branch khác nhưng không phải default configuration đã audit. Head/neck vẫn thuộc active upper-body joints theo SMPL-X map; thiếu explicit gaze không đồng nghĩa thiếu toàn bộ head motion. [Official joint names](https://raw.githubusercontent.com/vchoutas/smplx/main/smplx/joint_names.py).

## Generator và retrieval

Generator dùng mBART-large-cc25, 12 encoder/decoder layers, hidden 1024, thêm motion vocabulary và target-language IDs. Multi-head decoder nhận trung bình ba embeddings ở timestep trước rồi dự đoán ba part tokens ở timestep mới. Các streams chia sẻ history nhưng conditionally factorize tại cùng timestep; không phải ba models độc lập, cũng không là joint distribution đầy đủ của pose tại một instant.

Stage 2 dùng token CE. Inference greedy, dừng khi **bất kỳ head nào ra EOS**. Formulation không cần reference length, nhưng actual evaluation path/max-length policy còn cần audit đầy đủ. DETO train 500 epochs, batch 320/GPU, AdamW 2e-4; LM 150 epochs, batch 32/GPU, sáu RTX 3090. So efficiency phải tính pretraining/tokenizer/data budget.

Retrieval lookup lemmatized written words trong dictionary. Nếu một từ có nhiều instances, chọn instance có tokenizer reconstruction error thấp nhất; append retrieved motion tokens vào encoder input. Model vẫn sinh sequence mới, không bắt buộc copy/chắp exemplar. Word lookup chưa là target-sign-order planner; reconstruction tốt cũng chưa xác nhận variant phù hợp ngữ cảnh nhất.

## Kết quả và ablations

Table 1 full multilingual/retrieval model, **author-reported, chưa tái lập**:

| Dataset | DTW-PA-JPE body/hand theo tên paper | DTW-JPE body/hand | BT BLEU-4 |
|---|---|---|---|
| How2Sign | 6.82 / 2.35 | 7.75 / 10.08 | 14.48 |
| CSL-Daily | 6.24 / 1.71 | 7.38 / 9.68 | 11.30 |
| PHOENIX-2014T | 4.77 / 1.38 | 6.04 / 7.72 | 11.87 |

Tên PA/units trong bảng không đủ xác định implementation protocol; xem audit bên dưới.

- Table 2 trên How2Sign: sequential no-retrieval average DTW 4.64/latency 3.26 s; parallel 5.06/1.39 s; multi-head 4.17/1.46 s; multi-head+retrieval 3.34/1.55 s. Cả decoder design lẫn thêm retrieved evidence đều ảnh hưởng; không quy hai gains thành một.
- Table 3 thêm ngôn ngữ cải thiện reconstruction/generation nhưng cũng thêm training data. Cần matched-size/pose controls để xác định cross-lingual transfer.
- Table 4 ba streams tốt hơn các grouping đã thử; capacity/bitrate và generator factorization chưa được cô lập hoàn toàn. Không kết luận mọi part separation đều tốt hơn coupled representation.
- Table S6 hand codebook 192 tốt hơn 128/256 trong setting của paper; không phải quantization ceiling chung của SLG.

## Linguistic claims và human study

Paper lập luận linguistic discreteness “necessitates autoregressive modeling”. Review không chấp nhận suy luận bắt buộc này: masked hoặc continuous conditional models cũng có thể học cấu trúc ngôn ngữ. SOKE thay representation, LM pretraining, data, retrieval và decoding cùng lúc; chưa có causal evidence rằng diffusion yếu vì không có temporal AR.

Human study có 5 professional ASL và 4 CSL signers, 15 samples/model, text–motion matching ratings 1–10. SOKE means 5.32/6.45 so với S-MotionGPT 1.36/2.70. Đây là rating trên pool nhỏ, không phải comprehension accuracy 53.2%/64.5%. Không thấy DGS human study trong phạm vi đã đọc; xem text prompt cũng khác blind meaning recovery.

## Static code audit và ảnh hưởng benchmark

1. [load_data.py, L46 trở đi](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/data/humanml/load_data.py#L46) bỏ 36 dimensions đầu và shape từ vector 179D. [H2S.py, L106](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/data/H2S.py#L106) thêm 36 zeros, fixed shape rồi gọi SMPL-X; mapping cuối đúng 30+45+45+3+10.
2. [models/mgpt.py, L421](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/models/mgpt.py#L421): LH `30:75`, RH `75:120`, B nối `:30` và `120:`. Không hứa full root/lower-body/gaze freedom khi representation không cung cấp chúng.
3. [deto.yaml](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/configs/deto.yaml): UNIT_LEN 4, max motion 400, declared FPS 20, mean/std paths trỏ CSL-Daily. How2Sign loader lại cap trên 24 FPS về 24. Actual preprocessing/rate cần pin; một config value không chứng minh mọi dataset chạy cùng FPS.
4. Default loss config dùng Smooth L1, commitment weight .02 và velocity weight 0; paper mô tả reconstruction L2. Không gọi current defaults là exact paper reproduction.
5. [t2m.py, L147–170](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/t2m.py#L147) tự comment `align_idx=0` là DTW-JPE, `None` là DTW-PA-JPE, nhưng variable/log names vẫn giữ PA. Default 0 dịch root/body và wrist/hand. [dtw.py, L84](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/dtw.py#L84) chỉ gọi rigid alignment khi `None`. Không suy Procrustes từ log key.
6. Hand joints được regress từ mesh vertices qua hand regressor riêng. Frame-pair cost là mean joint L2; DTW trả accumulated terminal cost, không chia warping-path length. `compute()` chỉ chia theo sequence count. FPS/length/path length ảnh hưởng score.

Các khác biệt paper/code không chứng minh paper numbers sai. Chúng yêu cầu tường minh nhánh JPE/PA, normalization và source version trước so SOTA. Cờ C không bao hàm dictionary split audit hoặc full BT checkpoint audit đã hoàn thành.

## Ý nghĩa cho proposal

SOKE là baseline mạnh cho part tokenization, LM và retrieval trên ba datasets. Nó chưa xác lập bottleneck chính nằm ở codec, fitting, planning hay length. Proposal mới phải so các cơ chế và nguồn thông tin công bằng; không mặc định giữ VQ/AR hoặc mặc định bỏ chúng.

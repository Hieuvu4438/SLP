# Benchmark contract và các mốc phải vượt

Các bảng dưới đây là **số liệu do tác giả báo cáo**, đã đối chiếu full text; chưa tái lập model inference trong nghiên cứu này. Giữ nguyên thang điểm báo cáo, không tự đổi DTW sang mm/cm hay gọi accumulated DTW là mean error mỗi frame.

## 1. Mốc gốc: SOKE v3, Table 1

Nguồn: [SOKE Table 1](https://arxiv.org/html/2411.17799v3#S4). Dòng đầy đủ có external dictionary và multilingual training.

| Dataset | DTW-PA-JPE body ↓ | hand ↓ | DTW-JPE body ↓ | hand ↓ | BT BLEU-4 ↑ |
|---|---:|---:|---:|---:|---:|
| How2Sign | 6.82 | 2.35 | 7.75 | 10.08 | 14.48 |
| CSL-Daily | 6.24 | 1.71 | 7.38 | 9.68 | 11.30 |
| PHOENIX-2014T | 4.77 | 1.38 | 6.04 | 7.72 | 11.87 |

SOKE Table 2 no-dictionary: PA body/hand lần lượt **7.91/3.10**, **7.58/2.17**, **6.16/1.85** cho ba dataset theo cùng thứ tự. Đây là các dòng khác điều kiện; không dùng thay thế dòng đầy đủ.

## 2. Mốc MaDiS v2, Table 1

Nguồn: [MaDiS Table 1](https://arxiv.org/html/2601.19577v2#S4). Pretrain ba training sets kết hợp rồi fine-tune riêng từng dataset; MaDiS không đánh dấu external dictionary, còn các dòng SOKE có dictionary.

| Dataset | DTW-JPE body ↓ | hands ↓ | SiBLEU body ↑ | hands ↑ | SiCLIP R@1 ↑ |
|---|---:|---:|---:|---:|---:|
| CSL-Daily | 5.99 | 7.96 | 4.87 | 4.08 | 38.61 |
| PHOENIX-2014T | 4.97 | 6.59 | 4.54 | 3.33 | 57.94 |
| How2Sign | 6.59 | 9.45 | 4.41 | 3.46 | 34.27 |

R@1 ghi theo phần trăm như bảng paper. Candidate pool, text deduplication và aggregation phải trùng evaluator; R@1 trên batch 32 và trên toàn test set không tương đương. Paper không dùng PA trong bảng này.

**[U] SiBLEU-4 chưa được reconcile giữa paper và code snapshot.** File `generation.py` ở commit đã audit đếm clipped unigram và brevity penalty. Không gọi output đó là BLEU-4 chuẩn. Khi triển khai phải lưu hai cột riêng: `released_code_token_overlap` và `token_BLEU4` đã khai báo cách tính. Không đổi metric âm thầm rồi đối chiếu với Table 1. [07_code_audit.md](07_code_audit.md).

## 3. SignSparK: các regime khác nhau

Nguồn: [SignSparK Table 3](https://arxiv.org/html/2603.10446#S4.T3). Dùng thang số của paper, không xem như đã chạy lại trên fitting SOKE.

| Regime / dataset | PA body ↓ | PA hand ↓ | JPE body ↓ | JPE hand ↓ |
|---|---:|---:|---:|---:|
| GF-T2P no KF / How2Sign | 7.26 | 2.72 | 6.30 | 11.43 |
| GF-T2P no KF / CSL-Daily | 7.26 | 2.00 | 6.27 | 10.63 |
| GF-T2P no KF / PHOENIX | 5.24 | 1.52 | 4.40 | 7.10 |
| SR-T2P / CSL-Daily | 4.87 | 1.37 | 4.89 | 6.56 |
| SR-T2P / PHOENIX | 3.68 | 1.18 | 3.51 | 4.82 |

SR dùng predicted gloss và dictionary keyframes. KF2P Table 4 lấy target keyframes và không phải mốc text-only. Không chép BLEU từ Table 4 sang Table 3. Repo release mới thay đổi fitting/checkpoints/BT; số có thể tốt hơn paper nhưng không còn là tái lập paper nguyên trạng. [README release](https://github.com/JianHe0628/SignSparK#data--checkpoints).

## 4. Bốn track để tránh kết luận sai

| Track [P] | Input inference | Dữ liệu/representation | Claim được phép |
|---|---|---|---|
| A: text-only | Text + target-language ID; mọi pose/length do model sinh | Fitting/splits SOKE; dữ liệu pretrained phải kê khai | Controlled improvement so với MaDiS/SOKE cùng điều kiện |
| B: retrieval | Text + language ID + dictionary đã khóa; T2G nếu dùng phải công bố | Cùng fitting và dictionary cho baseline/ours | Hiệu quả của retrieval-conditioned generation |
| C: keyframe completion | Keyframes từ target hoặc người điều khiển; length nếu được cho | Khai báo đúng oracle/control input | Upper bound/control capability; không claim text-only SOTA |
| D: richer data/face | Text; model train thêm fitting/nguồn dữ liệu giàu hơn | Tất cả baseline dùng cùng data mới khi so architecture | Data/representation gain hoặc extended-setting SOTA |

Tokenizer SOKE có sử dụng dictionary lúc train theo paper; do đó Track A với tokenizer này là **text-only inference**, không tự động là “không dùng external data trong training”. Nếu muốn train-only-data claim, phải train lại tokenizer và các pretrained components trong track đó.

## 5. DTW: phải khóa cả implementation, không chỉ tên

**[E]** SOKE code snapshot gọi `dtw(...)[0]`, tức accumulated cost; per-frame cost tính mean Euclidean joint distance. `align_idx=0` dịch root trước khi tính distance; `None` gọi rigid alignment. Tên output có hậu tố `PA` vẫn có thể dùng `align_idx=0`. Hands được regressed từ mesh và align trong frame của từng bàn tay. [SOKE evaluator tại commit audit](https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/t2m.py#L147).

**[I]** Vì vậy “không PA” không nhất thiết là global-coordinate error, còn wrist-aligned hand metric không đo đầy đủ hand placement so với torso. Không chia path length, đổi joint subset hoặc đổi mesh regressor chỉ vì tên metric giữ nguyên. Nếu cần metric normalized bổ sung, báo song song, có tên khác.

Manifest tối thiểu phải đóng băng trước run:

- Dataset version, split IDs, FPS, frame crop/subsample, missing samples và pseudo-GT hashes.
- Rotation convention, joint IDs, root normalization, mean/std fit trên train, SMPL-X version/shape/camera/hand regressor.
- DTW local cost, accumulated hay normalized, path policy, PA transform và phần nào được align; per-sequence/dataset aggregation.
- LM/tokenizer/evaluator checkpoint hashes; toàn bộ nguồn pretraining và dictionary overlap policy.
- Length/EOS policy; mọi input inference; samples per prompt; seed; decode/CFG/NFE.
- BT preprocessing, language tokenization, corpus BLEU signature; SiCLIP candidate pool và duplicate-text handling.

## 6. Quy tắc đưa ra claim SOTA

**[P]** Cần hai lớp bằng chứng:

1. **Mục tiêu gốc:** vượt SOKE đầy đủ trên test splits gốc với protocol đã reconcile, báo cả PA, non-PA và BT khi có đúng BT evaluator.
2. **Đối thủ hiện thời:** so với MaDiS và các nguồn mới có protocol tương thích; báo các track khác nhau cho SignSparK/SIGNER/M3T/SignRR khi fitting hoặc supervision khác. Không loại đối thủ vì kết quả bất lợi.

Không gộp PA body, wrist-aligned hands, BT và retrieval thành một số “SOTA score” tự đặt. Đặt primary metric và guardrails trên dev trước khi mở test; báo trade-off nếu hand error tăng dù body giảm. Với evaluator không tái lập được, dùng nhãn **reported-only / protocol unresolved**, không điền số mới dưới tên cũ.

Mọi speed claim phải cùng GPU, precision, batch size, output frame count, warmup, synchronized timing; báo prior + refiner + tokenizer decoding và retrieval, cùng p50/p95. Không dùng chênh lệch 0.01s và 1.55s từ hai thiết lập khác để đảm bảo 100× cho method đề xuất.

# S09 — BLANC: prior art mới về multi-rate tokens và motion-derived scheduling

**D có phạm vi, chưa kiểm tra code.** Đã đọc §§3–5, Tables 1–6, Appendix B/C/D.1–D.5; chưa đọc toàn bộ phần cuối appendix hoặc xem video. [arXiv 2610.24109v1](https://arxiv.org/html/2610.24109v1), nộp tháng 10/2026; [project](https://winston1214.github.io/BLANC/) ghi under review. Không coi là paper đã peer review. Snapshot mới tại `sources/blanc/`; không dùng implementation nội bộ trong repo SLP.

## Cơ chế và điều kiện sử dụng

MRVQ có ba VQ-VAE riêng: body stride 4, mỗi hand stride 2. Mỗi body-rate unit chứa một body token và bốn hand tokens. Codebook body có 96 entries, mỗi hand 192; encoder/code vectors được chuẩn hóa trước nearest-neighbor assignment. Tokenizer dùng part reconstruction, FK joint loss và commitment loss (App. B/D.5).

Motion-derived teacher đo thay đổi cosine giữa hai phía của một cut ở nhiều thang thời gian, tổng hợp qua ba articulators và mức agreement. Sau đó bounded exact-cover DP chọn **một partition**, với giới hạn duration và penalty cho số cuts. Mỗi block có confidence; duration loss được confidence-weighted. Đây không phải gloss supervision và confidence không phải một posterior trên mọi segmentation (App. C.1).

Latent Articulatory Clock dùng text coverage, event position và GRU tóm tắt motion history. Controller dự đoán duration qua bounded hazard distribution và EOS sau khi hoàn tất block. Train history là ground truth; inference history là generated. Coverage được paper xác định là tiến độ học được, không phải alignment từ từng từ sang sign (App. C.2).

Generator causal giữa các blocks và bidirectional trong block. Inference phân bổ số mask tokens theo predicted duration, sinh draft bằng một parallel pass, rồi remask hand tokens trong khi body giữ cố định. Không dùng target length hoặc teacher partition ở inference theo đặc tả paper. Có max 32 blocks và giới hạn capacity; cần báo truncation khi tái lập (App. C.4).

## Objective và prior art cần giữ

Loss gồm masked motion prediction, duration, total length, EOS, frozen text–motion alignment và **counterfactual event-conditioning margin**. Counterfactual lấy event state từ text không khớp trong minibatch, giữ direct text input và motion history cố định; đòi motion NLL dưới state đúng thấp hơn state sai một margin 0.25. Vì vậy cả ý tưởng “motion-derived duration không gloss” lẫn “counterfactual plan để buộc generator dùng semantic condition” đã có prior art trực tiếp (App. C.3).

Frozen aligner nhận mixed latent: predicted ở masked positions, target code embeddings ở chỗ còn lại. Objective này không trực tiếp kiểm tra full free-running sequence; cần tách teacher-forced semantic loss với khả năng comprehension lúc tự sinh.

## Bằng chứng và giới hạn

Table 1, tác giả báo PHOENIX BT BLEU-4 9.85, body/hand DTW-PA-JPE 7.58/1.48; CSL BT 2.13 và PA 9.75/1.71. SOKE no-dictionary ở cùng bảng có BT 5.13/2.66. Như vậy BLANC cao hơn trên PHOENIX nhưng **thấp hơn SOKE trên CSL BLEU-4**. Không lặp claim thắng mọi trục của project.

Các số này thuộc fitting và evaluator mới. App. D.3 nói retrain baselines từ official implementations; không đồng nhất SOKE no-dictionary của bảng này với SOKE original multilingual+dictionary. MaDiS, SIGNER và How2Sign không được đánh giá trong main study này.

Table 3 adaptive blocks vượt các fixed-block variants đã thử. Table 4 multi-scale teacher cho BLEU-4 cao hơn uniform, shuffled và single-scale; tuy nhiên CSL body PA 9.75 kém single-scale 9.63. Shuffled durations vẫn cho CSL BLEU 1.91 so với full 2.13: cần kiểm tra lợi ích đến từ content-aware boundaries hay chỉ duration distribution.

Table 5 bỏ motion feedback làm CSL BLEU giảm 2.13→0.23, nhưng PHOENIX chỉ giảm 9.85→9.64. Hiệu ứng khác nhau giữa datasets; chưa đủ để kết luận feedback là bottleneck chung. Hand refinement tăng BLEU nhưng làm một số geometry scores xấu nhẹ. PHOENIX dùng eta=1.0, tức remask **toàn bộ** hand tokens; ở chế độ đó ranking bằng entropy không thay tập tokens được chọn (App. C.4).

App. D.2 đặc tả DTW là accumulated cost không chia path length. PA fit trên full SMPL-X joint set trước khi chọn body evaluation joints; mỗi hand được fit riêng. Protocol này cần giữ khi đối chiếu original numbers; bổ sung normalized DTW ở bảng riêng.

Generator train khoảng 48 GPU-hours/dataset trên bốn A6000, chưa tính tokenizer 500 epochs, fitting, aligner, baselines hoặc tuning. Reported generation dùng fixed checkpoint và seed 42; không phải confidence interval qua nhiều training seeds (App. D.4–D.5).

Không có đánh giá của fluent signers. Boundary evaluation dùng CSLR-derived pseudo boundaries với tolerance ±4 frames, không phải ground-truth linguistic segmentation. Fitting giữ WiLoR hands cố định trong optimization; không tự chứng minh hand orientation/rare contrasts luôn đúng (App. A; limitations).

## Hệ quả cho việc chọn candidate

Một candidate chỉ gồm multi-rate tokenization, motion-derived partitions, duration/EOS controller và body-conditioned hand refinement sẽ trùng cơ chế cốt lõi với BLANC. Muốn tiếp tục hướng planning cần đặt câu hỏi khác: ví dụ giữ bất định về segmentation thay vì một target partition; đánh giá tính ổn định khi teacher nhiễu; hoặc mô hình hóa interval scopes không đồng bộ. Những thay đổi này vẫn là đóng góp **ứng viên**, chưa được chứng minh mới hoặc hiệu quả chỉ nhờ khác tên kiến trúc.

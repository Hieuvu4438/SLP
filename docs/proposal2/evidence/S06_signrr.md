# S06 — SignRR: retrieval và refinement đã là prior art

**Phạm vi D, chưa kiểm tra code.** Đọc main method, setup, kết quả, ablations, limitations và supplementary A–G của [arXiv 2608.28568](https://arxiv.org/abs/2608.28568). Snapshot tại `sources/signrr/`. Giữ trạng thái preprint; chưa xác minh venue. Không chạy lại benchmark.

## Bài toán và cơ chế

Input chính là **chuỗi gloss**, không phải spoken text tự do. Dictionary được tạo từ continuous training motions, dùng CorrNet+ và thứ tự gloss ground truth để ước lượng boundaries. Đây là alignment có ràng buộc gloss, không phải recognition hoàn toàn không giám sát (§3.1; App. B).

Representation là 61 keypoints 3D, tổng 183D. Body stream gồm 19 điểm, trong đó có một số facial landmarks; không có biểu diễn chi tiết mouthings hoặc eye movements. Hai tay có 21 điểm mỗi bên. Không được gọi representation này hoàn toàn không có điểm mặt, cũng không được coi nó đủ để đo non-manual grammar (App. A; §5).

Mỗi part có encoder riêng, downsample 4 và ba tầng RVQ. Latent body rộng 192, mỗi hand 256. Sau quantization, các latent được nối, nội suy theo chiều thời gian rồi đưa qua **joint decoder**. Gloss encoder chỉ phục vụ dự đoán chênh lệch tổng chiều dài, không trực tiếp điều kiện hóa decoder chính (§3.2).

Training ngẫu nhiên chọn một dictionary segment cho mỗi gloss rồi học ánh xạ assembled motion sang continuous reference. Loss gồm part-weighted Smooth L1, velocity, commitment và length L1. Trong training dùng target length để nội suy latent; inference dùng predicted length, clamp theo giới hạn dataset. Khoảng lệch giữa hai chế độ là rủi ro cần kiểm tra. Inference chọn segments bằng dynamic programming với chi phí discontinuity và chất lượng mẫu, sau đó refine toàn chuỗi (§3.3).

## Bằng chứng và giới hạn suy luận

Trên PHOENIX14T test, tác giả báo BLEU-4 10.96 và MPJPE 53.70; CSL-Daily BLEU-4 1.57 và MPJPE 115.40. GT BT của CSL chỉ 5.06 trong evaluator này. Không so trực tiếp với SIGNER 15.60 hoặc SOKE 11.30: representation, evaluator và gloss regime khác nhau (Tables 2–3).

Table 5: với DP, refinement nâng BLEU-1 22.06→28.78. DP đơn thuần còn kém random selection trên BLEU-1; khi có refiner, DP chỉ thêm 0.55 BLEU-1 và 0.20 BLEU-4. Vì vậy không gán phần lớn gain cho retrieval selector. Table 6: real dictionary input tốt hơn synthetic periodic input; synthetic input có gloss cross-attention vẫn đạt BLEU-1 25.06, so với 28.78 của hệ chính. Đây là bằng chứng trong tập baseline đã kiểm tra, không chứng minh mọi generator từ noise sẽ kém hơn retrieval.

GT length giảm MPJPE 53.70→47.61 nhưng giảm BLEU-4 10.96→10.28. Geometry và back-translation không đồng biến hoàn toàn. RVQ depth 2 có BLEU-4 11.23, cao hơn depth 3; không được gọi depth 3 tốt nhất mọi metric (Table 4).

Supplement cho biết unseen gloss được thay bằng mục gần nhất theo edit distance. Cách này đảm bảo có output nhưng không bảo toàn nghĩa; không phải giải pháp open vocabulary. Dictionary train-only không tự loại khả năng self-retrieval trong training: tài liệu đã đọc chưa làm rõ leave-one-utterance-out. Khi xây baseline mới phải loại toàn utterance và nguồn video liên quan.

App. F rebuild dictionary và retrain sau jitter; đó không phải độ bền với nhiễu mới chỉ xuất hiện ở test. App. G báo 87.8 ms/sequence trên A100, gồm CPU DP 54.6 ms và refinement 33.1 ms; không gồm text-to-gloss hoặc rendering.

## Hệ quả cho proposal

“Retrieve rồi refine toàn chuỗi bằng part-aware RVQ” đã có prior art trực tiếp. Một candidate retrieval phải giải quyết thêm vấn đề có thể kiểm chứng, như lựa chọn độ tin cậy, coverage hoặc bảo toàn articulation khi sửa timing. Thêm refiner hay đổi loại generative head không tự tạo novelty. Khả năng áp dụng How2Sign cần giải quyết dictionary indexing mà không ngầm giả định có gloss toàn corpus.

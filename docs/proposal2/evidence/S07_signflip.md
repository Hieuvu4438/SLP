# S07 — SignFLIP: shared representation, dữ liệu lớn và giới hạn semantic evidence

**Phạm vi D, chưa kiểm tra code.** Đã đọc §§3–4, limitations, Appendices A/B/D/E của [arXiv 2609.35225v1](https://arxiv.org/html/2609.35225v1). Chưa xem đầy đủ video, mọi case study hoặc questionnaire. Venue chưa được xác minh; xử lý là preprint. Không chạy model.

## Phương pháp

SignFLIP dùng 69 keypoints 2D: 42 điểm tay, 9 body và 18 face. Pose encoder/decoder là spatial GCN; mT5-Base dùng chung giữa translation và generation. Ba giai đoạn gồm masked motion reconstruction cộng contrastive alignment; sign-to-text training; sau cùng text-to-sign training. SLG dự đoán tổng chiều dài từ text, dùng learned motion queries cross-attend text rồi decode poses (§3).

SLG có Smooth L1 reconstruction, feature distillation từ pose encoder đã pretrain và wrist-relative hand loss. Đây là prior art rõ cho semantic representation transfer, latent feature supervision và hand-relative reconstruction. Mask nhiều frames buộc dùng temporal context, nhưng bản thân việc giải bài inpainting không chứng minh representation đã học đủ semantics.

Pretraining dùng CSL-News hoặc YouTube-ASL, khoảng 600–700K pairs, huấn luyện trên bốn H100. Vì thế không quy toàn bộ gain cho kiến trúc khi baseline chỉ dùng target training set (§4.1; App. A/B).

## Kết quả và phản bằng chứng

Table 3 báo BT BLEU-4 7.10 trên How2Sign và 9.75 trên CSL-Daily. Đây là 2D pose generation với evaluator riêng; không xếp cùng bảng số với SOKE/MaDiS 3D. Table 8 cho CSL DTW 0.034 khi bỏ pretraining và SLT, 0.030 khi bỏ SLT, 0.025 đầy đủ. Ablation cho thấy các stages có ích trong protocol của paper; cần equal-data/equal-compute để phân biệt data gain.

App. E: bỏ cả hand loss và distillation cho DTW 0.044; chỉ bỏ hand loss 0.029; chỉ bỏ distillation 0.037; đầy đủ 0.025. Không có matched semantic comprehension ablation để khẳng định mọi gain geometric là lexical gain.

Human study chỉ có hai professional signers và 15 mẫu. Generated ratings 1.69/5 trên How2Sign và 1.94/5 trên CSL; reference poses cũng chỉ 3.48/3.44. Kết quả này là phản bằng chứng quan trọng đối với việc gọi paired geometry/BT tốt là đủ dùng cho người ký hiệu. Paper thừa nhận finger motion còn collapse về mean (§4.2–4.3; limitations).

## Hệ quả

Một proposal không thể nhận novelty từ shared sign–text embedding, pretraining theo stages hoặc pose-feature distillation đơn thuần. Ngược lại, cải thiện biểu diễn/conditioning vẫn là hướng hợp lý nếu có test xác định thông tin nào được giữ và người dùng có nhận ra nghĩa hay không. Fine-grained articulation còn yếu là quan sát trong paper này; không tự suy ra bottleneck chung của mọi SLG model.

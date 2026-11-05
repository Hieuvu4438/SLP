# M18 — SegMo: phản bằng chứng đối với việc thêm segmentation phức tạp

**D có phạm vi:** đọc §§3.1–3.5, setup/implementation, §4.2–4.3, Table 3 và limitations. Chưa đọc supplementary riêng, kiểm tra code hoặc toàn bộ application demos. Nguồn [WACV 2026 official PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Dang_SegMo_Segment-aligned_Text_to_3D_Human_Motion_Generation_WACV_2026_paper.pdf); snapshot và hash tại `sources/segmo/acquisition.json`. Chỉ dùng section/table locators, không có validated PDF page anchors.

SegMo giữ MoMask RVQ và masked/residual Transformers. Qwen3:8B phân tích caption thành action phrases theo thứ tự thực hiện, không nhất thiết thứ tự xuất hiện trong câu. Motion được chia cùng số đoạn với text; default là uniform segments. Mean/max pooling của predicted motion tokens tạo segment embeddings. Contrastive loss phân biệt các segments **trong cùng sample**, kết hợp masked-token CE (§3).

Dataset HumanML3D/KIT-ML dùng body-motion representation; max số segments 5, CLIP condition. Không có detailed hand/face hoặc sign-language translation. Caption decomposition dựa action semantics không tự chuyển thành gloss order.

Table 3 trên HumanML3D: uniform SegMo R@1 .553, FID .042; CPD .548/.046; clustering .548/.051. Bỏ alignment cho .538/.065; global alignment .539/.072. Đây là bằng chứng trong cùng backbone rằng local segment alignment có ích, nhưng segmentation phức tạp chưa thắng uniform. CI là qua 20 evaluation repetitions, không phải 20 independent training runs.

§4.3 cho rằng variability của segmentation errors quan trọng hơn mean error và liên hệ với training instability. Đây là diễn giải từ các segmentation strategies của study; chưa phải một controlled causal result chứng minh mọi stochastic partition gây hại. ISC phụ thuộc learned embedding và cách chia ở evaluation nên không là ground-truth segmentation correctness.

**Hệ quả bắt buộc cho candidate B:** ambiguity không tự chứng minh phải marginalize nhiều plans. Uniform và low-variance point supervision là baselines nghiêm túc. Cần tách structured expectation khỏi random boundary jitter/extra augmentation, giữ teacher scores và compute tương đương; nếu chỉ làm gradient nhiễu hoặc chỉ cải thiện oracle-plan, bỏ hướng đó. SegMo là phản bằng chứng được giữ trong quyết định, không bị loại chỉ vì không hỗ trợ candidate.

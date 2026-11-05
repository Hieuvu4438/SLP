# S08 — FGDM: semantic CTC trong denoiser đã có trước

**Phạm vi M có định vị.** Đã đọc overview, forward diffusion, phần embedding/focal overview, toàn §3.3 và phần mô tả kết quả §4.2. Chưa audit đầy đủ ASGCN, mọi ablation, supplementary hoặc code. Không nâng lên D.

Nguồn đã xác minh: [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yu_Focal-General_Diffusion_Model_with_Semantic_Consistent_Guidance_for_Sign_Language_CVPR_2026_paper.pdf), [project tác giả](https://yuyiheng-eu.github.io/fgdm/). PDF mới được lưu và hash trong `sources/fgdm/acquisition.json`; trích xuất bằng PyMuPDF, chỉ dùng section/table locators. Nút arXiv trên project còn là placeholder, nên không viện dẫn thành paper arXiv hợp lệ.

Input là gloss; output là 3D keypoints. Focal stage mô hình hóa joint dependencies, general stage dùng Transformer cho toàn sequence. Semantic Consistent Guidance nhận **features bên trong denoiser**, chuyển qua adapter, TCN, BiLSTM và gloss classifier; CTC loss ghép với gloss target và giảm trọng số khi noise lớn (§3.3, Eq. 15–18).

Hệ quả xác minh được: thêm gloss-recognition/CTC auxiliary loss vào diffusion đã có prior art. CTC ở đây dùng thứ tự target gloss; không cho phép áp monotonicity lên spoken-word order để bỏ bài toán translation.

Paper đánh giá PHOENIX14T và USTC-CSL, không đồng nhất USTC-CSL với CSL-Daily. §4.2 còn báo WER tốt hơn reference trong evaluator và đưa ra một số cách giải thích. Đây không tự chứng minh motion có nghĩa đúng hơn con người; cần kiểm tra evaluator bias hoặc dataset noise bằng evidence riêng. Không dùng các giải thích của tác giả như kết luận nhân quả đã xác lập.

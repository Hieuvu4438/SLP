# S05 — Neural Sign Actors: diffusion trực tiếp và fitting trên How2Sign

**Phạm vi D, chưa kiểm tra code.** Đã đọc phương pháp fitting, generator, thiết lập đánh giá, Tables 1–3 và human study trong [bản CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.pdf), đối chiếu [arXiv v2](https://arxiv.org/html/2312.02702v2). Không tuyên bố đã đọc toàn bộ supplementary hoặc chạy model.

## Cơ chế và điều kiện

NSA sinh tham số pose và expression SMPL-X từ transcript. Forward process thêm Gaussian noise; mạng học dự đoán noise với trọng số phần tay gấp đôi. Pose encoder dùng message passing trên cây động học và joint identity embeddings; expression đi qua MLP. CLIP ViT-L/14 cung cấp sentence embedding. Decoder gồm bốn lớp LSTM rồi MLP. Đây là diffusion có decoder theo thời gian, cho thấy diffusion và autoregressive temporal modeling có thể cùng tồn tại (§4).

Fitting khởi tạo từ OSX, tối ưu reprojection 2D và bổ sung PCA pose prior cùng temporal regularization (§3). How2Sign không vì vậy trở thành bộ mocap ground truth toàn bộ. Table 1 kiểm tra fitting trên SGNify: body/left/right MPVPE 46.42/16.17/15.23 mm, so với OSX 47.32/18.34/18.12. Kết quả hỗ trợ cải thiện lỗi hình học trung bình, chưa chứng minh mọi lexical contrast đã được giữ.

Generator được train 2.000 epochs, Adam, learning rate giảm từ 1e-3 xuống 1e-6. Các baseline được retrain trên How2Sign. Table 2 báo BLEU-4 13.12 và body MPJPE 35.87; đây là evaluator của NSA, không cùng thước đo với SOKE hoặc SIGNER chỉ vì cùng tên BLEU/MPJPE.

## Ablation và phản bằng chứng

Table 3 cho thấy bỏ joint identity hoặc đổi CLIP làm chất lượng giảm mạnh. Bỏ GNN làm body MPVPE tăng từ 31.47 lên 37.51. LSTM và Transformer cho kết quả gần nhau: MPVPE 31.47 so với 32.73, nhưng Transformer có MPJPE 35.41 so với LSTM 35.87. Không có một decoder thắng mọi chỉ số. So sánh sentence CLIP với word encoder khác pretraining/capacity không chứng minh word-token conditioning về bản chất kém hơn.

Paper mô tả một số MLP là permutation equivariant; nhận xét này phụ thuộc cách chia sẻ weights, không áp dụng cho mọi MLP trên vector đã flatten. Cũng không nên lặp lại nhận định sign language chỉ liên quan tay và mặt: đây là giản lược trong thiết kế, không phải kết luận ngôn ngữ học.

## Human evaluation và giới hạn

15 người thông thạo ASL chấm 15 mẫu: model 5.8/10, fitted reference 8.1 và video 8.7. Một bài chọn câu trong năm đáp án báo top-1 40%, top-2 80%. Bài này gần kiểm tra nội dung hơn naturalness rating, nhưng vẫn có distractor và tập mẫu nhỏ; không phải free-form comprehension trên toàn test set (§5).

NSA là prior art trực tiếp cho anatomy-aware continuous diffusion và hand weighting. Một proposal dùng các thành phần đó cần đóng góp cụ thể khác. Fitting prior có thể làm giảm các cấu hình hiếm là giả thuyết cần kiểm tra, không phải bottleneck được NSA xác lập.

# Đề xuất cải tiến SignNMFlow

**Câu hỏi nghiên cứu.** Với cùng dữ liệu SMPL-X, text encoder và ngân sách 5 lần gọi generator, việc học đường vận chuyển có tính đến chuỗi động học của tay và thân có giảm lỗi khớp mà vẫn giữ nội dung ký hiệu tốt hơn SignNMFlow không?

**Quyết định của lab.** Chạy hai hướng độc lập trước. Hướng có đường triển khai ngắn nhất là bổ sung lỗi forward kinematics cho đích dự đoán, rồi thêm giám sát vận tốc nếu kiểm chứng được thiếu chuyển động. Hướng có giá trị nghiên cứu cao nhất là thay đường hiệu chỉnh theo từng joint bằng metric kéo ngược qua forward kinematics, có đối chứng đường chéo. Chạy thử hình học SO(3) trước để tránh quy hiệu ứng chọn dấu quaternion cho metric mới. [Kế hoạch và đặc tả](03_thiet_ke_thuc_nghiem.md)

| Bằng chứng tại thời điểm rà soát | Quyết định |
|---|---|
| Khoảng 10–12% cặp quaternion nguồn–đích mô phỏng ở khớp tay trên dev dùng cung dài hơn cung SO(3) ngắn nhất. Đổi dấu giữa các frame của **nhãn** tay gần như không xuất hiện trong mẫu đã đo. | Ưu tiên kiểm tra đường vận chuyển. Giữ thay đổi biểu diễn đầu vào thành ablation riêng. |
| Trong 256 clip dev mỗi bộ, p95 tốc độ góc tay target → dự đoán v2 là 37.38 → 32.31°/frame trên Phoenix, 24.35 → 17.03°/frame trên CSL. Đây là mô tả mẫu, còn nhiễu fitting và sai độ dài có thể góp vào chênh lệch. | Kiểm tra vận tốc theo ngữ cảnh và giám sát sai số vận tốc so với nhãn. Giữ các đối chứng để phát hiện oversmoothing và học nhiễu. |
| Seed 0 trên CSL test đạt hand DTW-PA-JPE 1.3794, mốc paper là 1.30. Phoenix đạt 1.3000 so với 1.07. Evaluator BLEU-4 nội bộ và evaluator paper khác nhau. | Tách kết luận thắng replica và vượt mốc paper. Chỉ đối chiếu trực tiếp từng metric khi xác minh được protocol tương ứng. |

## Nội dung

- [Phản biện paper và audit code](01_phan_bien_va_code.md) có bảng đủ các metric paper, bằng chứng local và mức độ chắc chắn của từng nhận định.
- [Literature review và sổ nguồn](02_literature_review.md) nối bằng chứng về rotation, skeleton, biểu diễn ký hiệu, đánh giá và flow matching với từng quyết định thiết kế.
- [Thiết kế thực nghiệm và phản biện nội bộ](03_thiet_ke_thuc_nghiem.md) chỉ rõ hàm cần sửa, đối chứng, phụ thuộc checkpoint, ngân sách, tiêu chí dừng và cách kết luận.
- [Probe CPU](probe_signmflow.py) và [kết quả JSON](probe_results.json) có thể tái chạy từ root repo bằng `PYTHONPATH=.:src python docs/proposal/probe_signmflow.py --local-cache`. Bỏ `--local-cache` để chạy các ví dụ tổng hợp.

## Phạm vi bằng chứng

Paper đích là v1 ngày 27/09/2026. Audit dùng code của `methods/signmflow` tại workspace có HEAD `c859c93cf28480fa7ca49f7f06846adf9dae086d`, cùng test, cache, log và kết quả của riêng replica. Probe đọc thống kê SMPL-X qua loader dữ liệu dùng chung. Tổng quan dựa trên tài liệu gốc công khai. Không sử dụng code, thiết kế hoặc kết quả của method khác trong SLP. [SignNMFlow v1](https://arxiv.org/pdf/2609.33339v1)

Kết quả mới gồm probe CPU, đo lại thống kê trên dự đoán dev đã lưu và 23 test của SignMFlow chạy đạt. Mốc test trong báo cáo lấy từ artifact có sẵn. Các biến thể được đề xuất chưa được huấn luyện. Job baseline How2Sign đang chạy khi rà soát. Proposal dùng AI hỗ trợ tìm tài liệu, kiểm tra code, thiết kế thí nghiệm và phản biện nội bộ.

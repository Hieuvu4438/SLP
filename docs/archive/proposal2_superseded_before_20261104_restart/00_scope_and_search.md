# Phạm vi và phương pháp khảo sát

## Câu hỏi nghiên cứu

RQ chính: **Có thể cải thiện đồng thời độ đúng ngữ nghĩa và độ chính xác articulatory trên benchmark SOKE bằng cách chuyển các tiến bộ motion generation sang SLG, với toàn bộ điều kiện suy luận lấy từ text thay vì target motion hay không?**

Ba câu hỏi phụ:

- Lỗi hiện tại đến từ language-to-motion mapping, quantization, phối hợp bộ phận, hay dữ liệu/evaluator? Cần thí nghiệm nào phân biệt được các nguyên nhân này?
- Các kỹ thuật từ masked generation, continuous diffusion/flow, retrieval, gesture và speech có giả định nào không phù hợp với ký hiệu?
- Phần khác biệt nào còn có thể bảo vệ sau khi xét MaDiS, SignSparK, SIGNER, M3T, SignRR và DC-Motion?

Đầu ra là bộ Markdown dùng để quyết định và triển khai nghiên cứu. Không yêu cầu thực hiện model training trong lượt nghiên cứu này. Không coi “method nghe hợp lý” là bằng chứng sẽ đạt SOTA.

## Chiến lược tìm kiếm

Ngày thực hiện: 04/11/2026. Dùng web search để khám phá; dùng arXiv full text, proceedings CVF/PMLR/ACL, trang tác giả và repo chính thức để xác minh. Không sử dụng bản tóm tắt tự động từ website tổng hợp làm bằng chứng kỹ thuật.

Các nhóm truy vấn thực tế:

| Nhóm | Truy vấn đại diện | Mục đích |
|---|---|---|
| Seed SLG | `SignSparK sign language generation paper`; `MaDiS sign language diffusion generation paper` | Xác định tên, phiên bản, code |
| Motion hiện đại | `text to motion generation 2025 2026 flow matching residual quantization motion DART MoMask` | Mở rộng ngoài các seed |
| Hội nghị | `site:openaccess.thecvf.com CVPR2025 text motion generation flow matching masked residual`; tương tự ICCV2025 | Ưu tiên nguồn peer reviewed |
| Whole body | `HumanTOMATO arxiv`; `EMAGE arxiv`; motion generation 2026 part flow | Chi tiết tay, mặt, phối hợp đa vùng |
| Cross-domain | `Flow matching speech F5-TTS E2 TTS`; MAR, FSQ, Diffusion Policy, RDT-1B | Alignment, continuous token, bimanual generation |
| Novelty/negative | `sign language generation uncertainty keyframes`; `sign language generation event flow matching`; `sign language production contact diffusion` | Tìm đối chứng gần nhất và phản bằng chứng |
| Cập nhật SLG | `M3T Motion Tokens`; `SignRR sign`; `SignFML`; `sign language generation 2026 duration alignment` | Tránh đề xuất đã có |

Sau khám phá, truy ngược related work của SOKE/MaDiS/SignSparK và đối chiếu full text. Hai sửa đổi quan trọng do kiểm tra phiên bản: arXiv `2506.07460v2` đổi từ GLOS thành **SIGNER**; `2606.01643` hiện mang phụ đề **A Diagnostic and a Coverage Argument**, khác một số kết quả tìm kiếm còn giữ tên cũ. Dùng tên trên full text đã đọc. [SIGNER](https://arxiv.org/html/2506.07460v2), [Conditional Collapse](https://arxiv.org/html/2606.01643).

## Chọn nguồn và độ sâu

Nhận paper có ít nhất một trong bốn vai trò: baseline trực tiếp; cơ chế representation/generation có thể chuyển; bằng chứng về evaluation; đối chứng làm giảm novelty. Ưu tiên CVPR, ICCV, ECCV, ICML, ICLR, NeurIPS, ACL khi có proceedings chính thức. Preprint 2026 được giữ vì liên quan sát, nhưng không tự gán venue.

Không xếp chung SOTA cho các bài text→video, gloss→pose, pose completion và text→3D pose. Paper video/robotics/speech cung cấp nguyên lý chuyển giao; metric của chúng không chứng minh hiệu quả SLG. Những nguồn chỉ tìm được abstract hoặc thông báo chưa công bố không được dùng để mô tả chi tiết implementation.

Ledger ở [08](08_sources_and_reading_ledger.md) phân biệt:

- **D:** đọc sâu phương pháp, thiết lập thực nghiệm và kết quả/ablation liên quan.
- **M:** đọc chọn lọc các mục phương pháp cần cho thiết kế, không xác nhận đã đọc toàn bộ supplement.
- **S:** screening qua abstract/metadata hoặc đoạn phương pháp; chỉ dùng cho định hướng và quan hệ prior art.

39 full text được tải thành snapshot HTML và text, kèm hash; không phải cả 39 đều ở mức D. Locator dùng **section/table/code function**, không suy đoán số trang từ PDF. Không chạy PDF preflight vì nguồn thực đọc trong khảo sát là HTML; không đưa page anchor chưa kiểm chứng.

## Giới hạn

Đây là **targeted narrative review**, không phải systematic review đăng ký trước hay PRISMA hoàn chỉnh. Không có denominator tin cậy cho toàn bộ search index và không dựng số lượng “records screened” giả. Việc không tìm thấy paper cùng ý tưởng không chứng minh tính mới tuyệt đối.

Tìm được SignFML qua thông báo tác giả nhưng chưa xác minh full paper trong corpus; không dùng các chi tiết kỹ thuật hay điểm số của công trình đó để ra quyết định. SignFlow được xác minh qua chương trình/abstract chính thức ICMI, chưa dùng để tái lập bảng số liệu. Các giới hạn này không ngăn thiết kế proposal, nhưng cần cập nhật trước khi nộp bài. [ICMI proceedings](https://icmi.acm.org/2025/proceedings/).

Checkpoint phản biện phạm vi: RQ có thể trả lời bằng controlled experiments trên ba dataset; không yêu cầu huấn luyện motion foundation model mới từ đầu. Các thí nghiệm với fitting, gloss, face hoặc dữ liệu bổ sung phải khai báo thành điều kiện riêng. Đây là quyết định thiết kế [P], nhằm kiểm tra nguyên nhân cải thiện.

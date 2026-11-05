# Phản biện và audit phạm vi hoàn thành

Đây là **self-review trong cùng phiên nghiên cứu**, không phải blind peer review hay ý kiến độc lập của một lab. Ngày 04/11/2026.

## 1. Phản biện trước khi chọn hướng

**Phản biện mạnh nhất:** proposal có thể chỉ là MaDiS + một postprocessor flow với nhiều auxiliary losses. DC-Motion đã có discrete–continuous factorization; SUREFlow/UA-Flow đã dùng uncertainty; SIGNER đã có inter-part attention và temporal grounding; SCAMP đã có sparse-anchor correction. Không có đủ bằng chứng để gọi tổ hợp này là đóng góp thuật toán mới về nguyên lý.

**Cách xử lý:** [04](04_transfer_and_novelty.md) ghi rõ prior art; [06](06_experiment_plan.md) bắt buộc plain-flow, deterministic refiner, constant/random confidence và projection baselines. Chỉ giữ contribution nào có ablation và diagnostic ủng hộ. Nếu ordinary refiner đã đạt cùng kết quả thì bỏ claim reliability-specific contribution.

## 2. Phản biện sau synthesis

| Vấn đề | Tác động | Điều đã sửa / cách kiểm nghiệm |
|---|---|---|
| Ground-truth anchors làm task dễ hơn text-only | Có thể tạo SOTA claim sai regime | Tách Track A/B/C/D; oracle chỉ dùng diagnostics |
| Confidence cao có thể sai nghĩa | Preservation giữ lexical error | Không hard clamp/cắt correction theo confidence; q chỉ là compatibility score |
| Generated conditioning train quá sạch | Flow thắng oracle setting nhưng thất bại ngoài train | Free-running caches, corrupted examples và cross-fit pilot; công bố nếu chưa loại được mismatch |
| Timing reference được đưa vào inference | Rò target length, benchmark không còn text-only | Length/EOS do prior dự đoán; target resampling chỉ ở train |
| Part-local error bỏ placement/contact | Tối ưu blind spots của evaluator | Thêm torso-relative/relational metrics và semantic guardrails |
| Thay fitting/face rồi so bảng cũ | Quy data gain thành architecture gain | Rich-face track riêng; cùng target fitting cho mọi model trong controlled comparison |
| SiBLEU tên không khớp code | Điểm mới không comparable với paper | Đã đối chiếu upstream commit; hai định nghĩa được báo riêng |
| Chọn paper thuận hướng flow | Confirmation bias | Giữ VQ/RVQ/FSQ và coverage/alignment alternatives; có tiêu chí bác bỏ hướng chính |

## 3. Phản biện bản proposal cuối

Hướng hiện tại khả thi để **thử nghiệm**, nhưng thành công chưa được chứng minh. Ba rủi ro còn lớn:

1. Prior có thể sai lexical order/content nhiều hơn lỗi local articulation; refiner không có đủ thông tin để sửa. Pilot phải xác định bottleneck trước khi train lớn.
2. Reliability supervision dùng một rendition và pseudo-GT; đây không phải semantic uncertainty ground truth. Cần kiểm tra bằng renditions khác/người ký hiệu để tránh phạt diversity hợp lệ.
3. Captioned human-motion priors có thể không mang kiến thức về ngữ pháp ký hiệu; scale/refinement không thay thế paired language coverage. Không có bảo đảm SOTA từ một phép ghép kiến trúc.

Giới hạn không ngăn bàn giao proposal; chúng là giả thuyết và điều kiện quyết định đã đưa vào thí nghiệm. Không điền kết quả giả để đóng các khoảng trống thực nghiệm.

## 4. Integrity và attribution

Các nguồn cốt lõi có full text/locator; source registry phân biệt screening với method reading và deep reading. Không gán hội nghị cho preprint khi chưa xác minh. Các ý tưởng từ paper được ghi nguồn ở 01/02/04; kiến trúc/loss đề xuất mang nhãn [P]. Không có human-read declaration, không có giả lập user study, không có fabricated citations hay số liệu PACT-Sign. AI assistance đã disclosure ở README.

Không thu thập human data hay gửi nội dung workspace tới model bên ngoài trong công việc này. Kế hoạch human study yêu cầu cộng tác người ký hiệu và báo consent/compensation; không tự nhận ethics approval. Quyền dùng dataset/model assets cần được kiểm tra khi triển khai; việc public code không tự cấp quyền mọi asset phái sinh.

## 5. Completion audit theo yêu cầu người dùng

| Yêu cầu gốc | Bằng chứng hiện có | Kết luận cho deliverable nghiên cứu |
|---|---|---|
| Độc lập các hướng thực nghiệm đang chạy trong SLP | Không dùng results/configs nội bộ làm bằng chứng phương pháp; không sửa pipeline thực nghiệm | Đạt |
| Literature review rộng human motion/text-to-motion | 01: discrete, continuous, hybrid, retrieval, streaming, scaling; 08: nguồn/mức đọc | Đạt trong phạm vi narrative review đã khai báo |
| Đọc kỹ các kỹ thuật hiện đại từ paper uy tín | Core SLG methods/experiments, selected methods từ nhiều paper CVPR/ICCV/ICML/ECCV/ACL/NeurIPS; có nguồn primary và locator | Đạt; không claim mọi supplement/proof đều được đọc |
| Mở rộng ngoài human motion khi hữu ích | 01/04: image MAR/MaskGIT/FSQ/FM, F5-TTS, Diffusion Policy/RDT/SUREFlow | Đạt |
| Deep research SLG, gồm SOKE/MaDiS/SignSparK | 02, bảng số đối chiếu ở 03, source audit ở 07 | Đạt |
| Đề xuất method phù hợp với mục tiêu vượt benchmark SOKE | 04 chọn hướng/alternatives; 05 tensor/loss/train/inference; 06 controls và SOTA criteria | Đạt ở mức proposal; performance chưa được xác minh |
| Source code nếu cần nằm trong third_party | SOKE/MaDiS/SignSparK sẵn có; MoMask clone thêm; paper snapshots tại third_party/proposal2_sources | Đạt |
| Proposal dưới dạng .md trong docs/proposal2 | README và 00–09 | Đạt |

**Ranh giới hoàn thành:** deliverable được yêu cầu là nghiên cứu và proposal method. Việc triển khai, chạy full experiments, xác lập novelty đủ cho submission và thực sự đạt SOTA vẫn là công việc thực nghiệm tương lai, chưa được nhận là hoàn thành. Tài liệu chứa kế hoạch kiểm chứng các việc đó; không lấy kế hoạch làm bằng chứng kết quả.

## 6. Kiểm tra artifacts đã thực hiện

- 11 Markdown files hiện diện, không có local link trỏ tới file thiếu; code fences cân bằng.
- 39 HTML snapshots có `article`, text trích xuất và SHA256 khớp manifest. Ledger: 3 nguồn mức D, 33 mức M, 3 mức S; screening bổ sung không tính vào 39 snapshots.
- Trích lại trực tiếp các dòng SOKE Table 1, MaDiS Table 1 và SignSparK Table 3 từ HTML: các chuỗi số khớp bảng ở 03.
- Mismatch SiBLEU đã kiểm tra bằng source tại đúng upstream commit và ví dụ đảo thứ tự token; không có baseline benchmark run được suy diễn từ kiểm tra này.
- Không chạy model tests/training cho một thay đổi chỉ gồm tài liệu. Các kết quả kiểm tra trên xác nhận deliverable và provenance, không xác nhận PACT-Sign đạt SOTA.

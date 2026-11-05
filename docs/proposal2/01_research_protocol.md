# Giao thức nghiên cứu

## Câu hỏi và ranh giới

**RQ chính:** Từ các tiến bộ motion generation đã có bằng chứng, cơ chế nào đáng ưu tiên để sinh chuyển động ngôn ngữ ký hiệu từ văn bản, với ngữ nghĩa đúng và chuyển động tốt, trong điều kiện dữ liệu và benchmark SLG thực tế?

1. Các họ biểu diễn, sinh và tổ chức thời gian giải quyết những vấn đề gì; bằng chứng nào tách hiệu quả cơ chế khỏi dữ liệu, capacity và evaluator?
2. Giả định nào chuyển được sang body–hands–face của SLG; giả định nào thất bại do tính chất ngôn ngữ, khác trật tự hoặc supervision?
3. Ít nhất ba cơ chế khác nhau nào có thể được đặc tả và so sánh công bằng; phương án nào có lập luận ưu tiên tốt nhất và có thể bị bác bỏ bằng thí nghiệm?

Đây là **critical scoping review có tìm kiếm ghi vết**, không tự nhận là systematic review bao phủ toàn bộ hoặc meta-analysis. Không gộp các con số khác dữ liệu/giao thức thành bảng xếp hạng nhân quả. Không dùng khung phân cấp RCT của y khoa để chấm paper ML.

## Nguồn và phạm vi

- Bắt đầu tìm kiếm 04/11/2026, cập nhật đến 05/11/2026; kiểm tra ngày/version từng nguồn, không suy ra accepted từ tên preprint.
- Nguồn gốc: CVF Open Access, OpenReview với quyết định chính thức, proceedings NeurIPS/ICML, ACL Anthology, arXiv; trang tác giả/repo chính thức để đối chiếu.
- Thời gian: ưu tiên 2024–2026; giữ các nền tảng 2022–2023 và nguồn cũ hơn nếu cơ chế cần hiểu. Bổ sung công trình gần đây qua forward/backward search.
- Phạm vi: T2M; whole-body/hands/face; representations; AR/masking/diffusion/flow; planning/duration/alignment/composition; control/contact; semantics/evaluation; speech/gesture/video/robotics khi xác định được cơ chế chuyển giao.
- Loại: bài nghiên cứu có full text và thiết lập thí nghiệm; survey chỉ dùng định vị/tìm nguồn gốc. Metadata/abstract-only được giữ để screening, không đủ chứng minh cơ chế.
- Loại trừ: reconstruction/recognition/video-only nếu không có cơ chế chuyển giao xác định; trùng version; nguồn thứ cấp dùng thay bằng chứng gốc; kết quả sau mốc tìm kiếm.

## Ma trận độ phủ trước khi chọn

| Trục | Bắt buộc khảo sát | Dấu hiệu thiếu phủ cần tìm thêm |
|---|---|---|
| Representation | discrete, residual/multiscale, continuous latent, direct pose; whole-body | Chỉ đọc VQ hoặc chỉ một tokenizer |
| Generator | autoregressive, masked, diffusion, flow | Một họ chi phối shortlist vì quen thuộc |
| Temporal structure | length, ordering, composition, alignment, streaming | Chỉ xem frame quality |
| Articulation | hands/face coupling, contact, relative geometry | Dùng body-only để suy ra handshape |
| Semantics | token/text alignment, semantic supervision, OOD/compositionality | Đồng nhất FID/R-precision với tính đúng ngôn ngữ |
| Transfer | gesture/speech và ít nhất một cơ chế domain khác có lý do | Chỉ liệt kê tên domain |
| Counterevidence | reconstruction vs generation; evaluator bias; gain do data | Chỉ lấy claim/leaderboard có lợi |

Không có chỉ tiêu số paper thay cho chất lượng. Paper quyết định phải có ghi chú đủ mục dưới đây. Cần phủ các cơ chế cạnh tranh trước khi chọn, không chỉ tìm thêm nguồn cho một phương án.

## Mức đọc và sổ bằng chứng

- **V:** xác minh metadata/abstract từ nguồn gốc. Không hàm ý đã đọc method.
- **M:** đọc chọn lọc, ghi rõ section và khoảng chưa đọc.
- **D:** đọc sâu method, setup, kết quả/ablation liên quan và limitations; ghi được assumptions, training/inference, confounds và những gì chưa chứng minh.
- **C:** kiểm tra code chính thức, kèm commit/file/locators; là cờ bổ sung, không tự nâng M lên D.
- **Acquired:** file đã tải; độc lập với V/M/D/C. Không tự đặt cờ human-read.

Mỗi deep note phải có: nguồn/version/venue; bài toán; biểu diễn và kích thước khi paper cho; cơ chế/loss; train/inference và supervision; datasets/protocol; ablation với locator; limitations/counterevidence; ý nghĩa chuyển giao (suy luận, không phải kết quả SLG); câu hỏi mở.

Mỗi claim quyết định cần `claim → primary source + section/table/code → điều kiện → sức mạnh bằng chứng → suy luận/hypothesis`. Tách **paper chứng minh**, **paper báo cáo**, **suy luận của review**, **giả thuyết đề xuất**.

## So sánh phương án và chống chọn sớm

Không định tên hoặc backbone trước khi tổng hợp và so sánh. Lựa chọn hiện tại được giải thích trong 07_main_method.md sau motion/SLG synthesis và bảng ba phương án 06. Sau khi synthesis motion và SLG xong, lập ít nhất ba đặc tả khác nhau về cơ chế. Chấm bằng lập luận có nguồn trên cùng các trục: linguistic semantics, temporal ordering/duration, articulation/NMM, data efficiency, exposure to inference conditions, compute, prior-art overlap, falsifiability. Không dùng tổng điểm tuỳ ý thay cho lập luận; nêu trường hợp đổi thứ hạng khi giả định thay đổi.

## Các cổng chất lượng và kết quả audit

1. **Scope checkpoint:** ranh giới độc lập thực nghiệm nội bộ, RQ và cách phân loại evidence được giữ. User đã yêu cầu tự chủ trọn goal; không cần thêm approval cho nghiên cứu này.
2. **Evidence/synthesis checkpoint:** có motion/SLG synthesis, multi-family coverage và counterevidence, gồm reconstruction/generation trade-offs và BLANC/SegMo tension. Không chọn family theo một leaderboard.
3. **Proposal checkpoint:** có three-candidate comparison, main specification, bounded novelty claim, explicit inputs, matched baselines và falsification. Claim–source audit/biên tập hoàn tất; efficacy và SOTA còn là research questions, không positive findings.

Các checkpoints là self-review trong cùng phiên/model, không independent peer review. [Completion audit](10_completion_audit.md) và [claim–source audit](11_claim_source_audit.md) nêu evidence và giới hạn từng requirement; không dùng số papers/files thay content quality.

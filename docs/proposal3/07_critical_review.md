# Review phản biện và kiểm tra claim của proposal

Cùng một AI thực hiện review, không giả lập reviewer độc lập. Ngày 06/11/2026. Không dùng experiment có sẵn trong SLP; không có training results của RACE.

## Checkpoint 1 — câu hỏi và benchmark

**Vấn đề phát hiện:** yêu cầu “dễ đạt SOTA hơn” dễ dẫn tới chọn hàng có GT keyframe/gloss hoặc evaluator thuận lợi. SOKE, SignSparK, MaDiS, M3T và SIGNER còn khác fitting/BT/metric implementation. Severity: major, critical nếu dùng sai track để claim SOTA.

**Sửa:** cố định SOKE pose protocol, chọn CSL vì language/domain và protocol control, tách GF/dictionary/KF2P; benchmark paper numbers và future matched reproductions là hai bảng. Không kết luận dễ hơn dựa vào score scale. Metric provenance chưa tái lập là gate cho future experiments, không được bỏ qua khi công bố kết quả.

## Checkpoint 2 — cơ chế và counter-evidence

**Vấn đề phát hiện:** post-refiner là hypothesis yếu nếu không so direct decoder; DisCoRD có ablation chống nó. Confidence, self-forcing, part coupling, adaptive duration và semantic tokenization đều có prior art. Severity: major về novelty.

**Sửa:** v2 dùng direct token-conditioned decoder làm trung tâm, post-refiner thành C3. Expected-improvement selection được phân biệt với entropy và so whole-sentence/independent-part/random/always-output controls. IRG-MotionLLM là prior art về deciding whether to refine, nên không claim first. Novelty còn có điều kiện; nếu không vượt C2 hoặc sentence-level rule, rút component hoặc hạ claim thành replication/adaptation.

**Counterargument mạnh nhất còn lại:** selection có thể chỉ là ensemble reranking nhỏ, không đủ thành một method mạnh; thêm cost và giữ native token length khiến omission/EOS errors chưa giải. Không có lập luận toán học nào chứng minh cải thiện DTW hoặc SOTA. Thí nghiệm C0–C8 và semantic adequacy sẽ quyết định, không lời mô tả kiến trúc.

## Checkpoint 3 — kiểm tra tính trung thực và tính khả thi

- Không dùng số đo/pilot/checkpoint tự train của SLP. Code official được clone lại độc lập; paper tải mới và hash rõ. Synthetic metric probe chỉ xác nhận isolated distance function từ source tác giả, không là benchmark experiment hay evidence SOTA.
- Không ghi “đã đọc mọi paper/code”. Toàn inventory được screen, methods liên quan được review sâu có chọn lọc; code chưa phát hành hoặc chưa tìm thấy được đánh dấu. HTTP success/bot page/acquisition không bị nâng thành reading.
- Không nhận rằng SMPL-X intrinsically thiếu rich expressions; M3T v2 phân biệt fitting limitation. Recognition probe không thay human evaluation của generation.
- Không gán paper metrics vào implementation adapted. Code availability được kiểm tra thực; RAM code public, HumanTOMATO release chưa đầy đủ generator, M3T project ghi coming soon.
- Expected-improvement bound chỉ áp conditional additive loss trong mô hình lý tưởng. Boundary switching, distribution shift và non-additive metrics phá guarantee; all-frame/whole-sentence controls phải được giữ.
- Compute budget cần profiling; hai candidates + selector không làm pipeline nhanh hơn base theo định nghĩa. Không hứa training hours từ spec hay copy throughput khác hardware.
- AI assistance được công khai. Source material là public papers/code; chưa gửi manuscript riêng hay corpus lên external model. Không liên hệ tác giả/người khác.

## Ma trận deliverable theo yêu cầu người dùng

| Yêu cầu | Artifact/evidence | Kết luận |
|---|---|---|
| Đọc hai danh mục để nghiên cứu đủ rộng | Hai bản gốc giữ nguyên; inventory 190 entries/139 URLs và screening records | Breadth được audit; không claim tất cả đã full-text |
| Đọc kỹ phương pháp SLP hiện đại | Primary methods/tables và review SOKE, SignSparK, MaDiS, SIGNER, BLANC, M3T, HybridSign, T2S-GPT, SignPR, FGDM, G2P-DDM, DARSLP, SignRR, SeRV | Selected deep review có nguồn, phạm vi rõ |
| Đọc source code tác giả | 8 fresh official repos; 22 file hashes và function-level notes | Static code audit, không runtime validation |
| Kế thừa modern T2M có điều chỉnh domain | DisCoRD/SALAD/RAM/MATE và synthesis rộng; proposal v2 §4–8 | Có donor attribution, mismatch analysis, alternatives |
| Không ghép module thiếu phản biện/chứng minh | Counter-evidence đổi v1→v2; expected-improvement argument, controls và rejection rules | Có rationale và falsifiable tests; không giả kết quả |
| Chọn SOKE hoặc SignSparK benchmark | `01_benchmark_audit.md` và v2 §1/9 | SOKE protocol được chọn; native-length/resource conditions explicit |
| Đề xuất method hướng tới SOTA | `05_revised_proposal.md` | Concrete research hypothesis; chưa có empirical SOTA claim |
| Lưu proposal trong proposal3 | README cùng v2, evidence, review và research manifests | Tài liệu reviewable đã có |
| Độc lập với experiment có sẵn của repo | Fresh external papers/checkouts; provenance note ở mọi tài liệu chính | Không dùng SLP experiment làm căn cứ |

## Giới hạn và phần việc sau proposal

Đây là focused deep research và method proposal, không phải systematic review đầy đủ mọi database hoặc chứng nhận novelty tuyệt đối. Một số source bị chặn, duplicate URL được giữ để truy vết, và không phải mọi paper trong danh mục đều có code công khai. Full dataset release và evaluator paper cũ có thể cần làm việc riêng khi triển khai. Các thí nghiệm, human study, profiling và SOTA verification là bước thực nghiệm tương lai; không được mô tả như kết quả đã có.

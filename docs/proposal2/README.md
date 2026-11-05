# Nghiên cứu Motion Generation → Sign Language Generation

**Đã hoàn tất bộ literature review và proposal**, bắt đầu lại 04/11/2026, audit cuối 05/11/2026 theo [goal gốc](00_goal_original.txt). Chưa chạy training/model benchmark hoặc human study; hiệu quả, novelty và mục tiêu SOTA vẫn cần kiểm chứng.

Nên đọc theo thứ tự:

1. [Motion literature: cơ chế và phản bằng chứng](03_motion_literature_synthesis.md).
2. [SLG synthesis: ngôn ngữ, articulation và prior art](04_slg_literature_synthesis.md).
3. [Benchmark contract cho ba datasets](05_benchmark_contract.md).
4. [Ba phương án khác cơ chế và lựa chọn](06_candidate_methods.md).
5. [Method chính: trực giác, ví dụ, công thức, train/inference](07_main_method.md).
6. [Thực nghiệm, ablations, tiêu chí bác bỏ và fallback](08_experiments.md).
7. [Critical review về novelty, giả định và tính khả thi](09_critical_review.md).

Phương án được ưu tiên kiểm tra việc học từ phân phối các temporal partitions thay vì một pseudo partition cố định. BLANC đã có gloss-free scheduling; SegMo cung cấp phản bằng chứng có lợi cho uniform segmentation. Đóng góp chỉ là ứng viên về cách xử lý temporal-target uncertainty, cần vượt matched point/confidence/jitter controls; không nhận novelty từ duration predictor, part attention hoặc việc ghép modules quen thuộc.

Tra cứu evidence và phạm vi:

- [Research protocol](01_research_protocol.md), [search log](evidence/search_log.md), [reading ledger](02_reading_ledger.md).
- [Các cặp findings cần đối chiếu](evidence/cross_paper_tensions.md).
- [Claim–source và method audit](11_claim_source_audit.md).
- [Completion audit tám yêu cầu](10_completion_audit.md), [research state](RESEARCH_STATE.md).

Không dùng proposal1, archived proposal2, implementation/config hoặc kết quả thực nghiệm nội bộ để hình thành đề xuất. Sources/code công khai mới lưu tại `third_party/proposal2_review_20261104/`. Acquisition không được coi là deep reading; mỗi note có scope riêng. Paper numbers là author-reported trừ hai diagnostics được ghi rõ.

Có hỗ trợ AI và self-review trong cùng phiên; không có independent peer review hoặc human-read attestation. Review có phạm vi tìm kiếm được ghi vết, không nhận exhaustive coverage hay guarantee SOTA.

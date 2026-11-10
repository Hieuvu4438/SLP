# Proposal3 — text-to-motion cho sign language production

Cập nhật 06/11/2026. **Nghiên cứu dựa trên paper gốc và code tác giả bên ngoài; không dùng các experiment/pilot sẵn có trong repo SLP làm bằng chứng.** Chưa có kết quả training của method mới hoặc bằng chứng đạt SOTA.

Đọc theo thứ tự:

1. [Proposal v2 — khuyến nghị hiện tại](05_revised_proposal.md): token-conditioned direct flow decoder, lựa chọn theo expected improvement, quan hệ tay–thân, training/inference, lập luận toán học, ablations và điều kiện bác bỏ.
2. [External evidence review](06_external_evidence_review.md): nguồn ngoài đã kiểm tra, phản chứng DisCoRD/HybridSign, giới hạn data/eval và code availability.
3. [Benchmark audit](01_benchmark_audit.md): SOKE và SignSparK, tách GF/dictionary/KF2P, số paper và gate tái lập.
4. [Source code audit](02_code_audit.md): SOKE, SignSparK, MaDiS, MoMask; đối chiếu lại fresh author checkouts.
5. [Literature synthesis](04_literature_synthesis.md): donor mechanisms, những chuyển giao không phù hợp và các cặp paper cần so.
6. [Critical review và kiểm tra claim](07_critical_review.md): counterarguments, scope và điều kiện công bố.
7. [Reading status](research/reading_status.md): phạm vi đọc và phần chưa hoàn tất; [inventory](research/input_inventory.csv), [screening decisions](research/input_screen_decisions.csv), [external code manifest](research/external_code_manifest.json), [primary bibliography](research/primary_bibliography.csv).

[Proposal v1](03_method_proposal.md) giữ để review alternative post-refiner; đã hạ ưu tiên sau khi đọc counter-evidence từ paper ngoài. Không triển khai cả v1 và v2 như một model ghép module.

Khuyến nghị benchmark hiện tại: CSL-Daily trên pose/evaluator SOKE cố định, PHOENIX để xác nhận; GF và dictionary là hai track riêng. Lựa chọn này dựa trên khả năng kiểm soát protocol, không phải bằng chứng nó dễ đạt SOTA hơn. SignSparK có các khác biệt release về data/checkpoint/BT cần matched reproduction trong công việc thực nghiệm tương lai.

Hai file danh mục đầu vào gốc được giữ nguyên. Tất cả thay đổi triển khai hoặc experiment đã có trong `methods/`, `configs/`, `docs/proposal2/` nằm ngoài căn cứ của nghiên cứu này.

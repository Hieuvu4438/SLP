# Proposal 7 — Phản biện SignNMFlow và kế hoạch vượt nó

**Câu hỏi nghiên cứu.** Giữ nguyên dữ liệu SMPL-X, text encoder, kiến trúc và ngân sách 5 bước sampling. Thay đổi nào có cơ sở giúp vượt SignNMFlow (arXiv 2609.33339) trên DTW-JPE, DTW-PA-JPE và ngữ nghĩa, mà không đổi lấy chuyển động bị "trung bình hoá"? Và "naturalness-guided path" của paper thực sự đóng góp gì?

## Kết luận của lab sau vòng này

1. **Dưới chính mục tiêu của paper, path "naturalness" không bẻ cong geodesic; nó chỉ đổi tốc độ đi trên geodesic.** [Đo, Phoenix và CSL]
   - Năng lượng của path học được (23.76 / 23.75) bằng cận dưới của mọi phép tái tham số hoá thời gian của geodesic (23.66–23.92 / 23.89–24.16).
   - Path chỉ cách đường cong geodesic 0.04–0.08 rad.
   - Tiến độ của nó khớp với time-shift kiểu SD3, s ≈ 2.5 ± 0.3.
   - Mật độ m(x) không tham gia vào mục tiêu khi ρ = 1e−6. Paper tự viết điều này ở Eq. 44, nhưng diễn giải ngược lại trong App. C.3.
   - Interpolant và cách huấn luyện là của Metric Flow Matching (NeurIPS 2024), bài mà paper không trích dẫn.
   - → **Giả thuyết H1:** lợi ích (nếu có) của NG là lợi ích của lịch nhiễu. H1 *chưa* được kiểm chứng trên chất lượng sinh, và cần đối chứng dương: liệu NG > GE có tồn tại trong bản tái lập không.
2. **Metric DTW thưởng cho trung bình hoá.** [Đo] Co dự đoán v2 25% về tư thế trung bình, một phép biến đổi không dùng văn bản, làm *tốt lên* cả 4 cột DTW. Làm mượt mạnh, xoá khoảng 75–80% tốc độ, vẫn tốt lên 3/4 cột (bảng dưới). B-T của repo đã bão hoà trên Phoenix và vượt trần GT trên CSL. Vì vậy **không thí nghiệm nào có thể kết luận trước khi có thước đo phân định được** (M1–M4 trong `04`).
3. **Các đòn bẩy đáng thử** (paper bỏ qua, tài liệu ủng hộ):
   - loss trong không gian vị trí qua toàn chuỗi khớp (metric đo vị trí, loss đo quaternion);
   - phân bổ mức nhiễu khi huấn luyện và khi suy diễn, như hai núm tách biệt;
   - EMA và learning rate thấp cho text encoder. Chọn checkpoint sớm theo val loss đã được thử ở vòng 1 và **làm xấu đi**, nên bị loại;
   - CFG trên manifold;
   - prior wrapped-normal (126.5° → 51° tới dữ liệu);
   - self-conditioning.
4. **Thứ tự chạy**: xem bản bàn giao [07](07_lo_trinh_uu_tien.md) (khoảng 550–650 GPU-h). Tóm tắt cũ:
   1. **G0**: bộ đo M1–M4, độ dài cố định, phép thử σ, hoán đổi path khi suy diễn, snapshot (gần như không cần GPU);
   2. baseline × 3 seed + **E1-gate** GE so với NG;
   3. E1-H1 (NG-parallel, geoshift);
   4. E3 vệ sinh;
   5. E4 FK loss;
   6. E2-F factorial lịch nhiễu;
   7. E5–E7.
5. **"Vượt SignNMFlow"** có hai nghĩa, báo cáo tách biệt:
   - (a) vượt bản tái lập v2 với kiểm định cặp và ≥ 3 seed;
   - (b) đối chiếu số paper trên test một lần, chỉ các cột DTW, kèm caveat giao thức.
   - Bản tái lập hiện kém số paper 0.5–1.0 điểm DTW trên Phoenix test. Không thay đổi nào được kỳ vọng chắc chắn lấp hết khoảng cách đó.

## Kết quả thực nghiệm vòng 1 (đã chạy; chỉ suy diễn và correction, CPU, Phoenix dev 519 chuỗi) — [06](06_ket_qua_thuc_nghiem.md)

- **Có thể làm số DTW đẹp hơn v2 chỉ bằng sửa sampler.** Ví dụ: lưới t_k = (k/5)² cộng checkpoint epoch 250 cho PA hand 1.369 → 1.329 và JPE hand 6.98 → 6.87; NFE 1 cho JPE body 5.17 → 4.93. Khoảng tin cậy bootstrap cặp loại trừ 0.
- **Nhưng không biến thể nào vượt v2 *thật sự*.** Mọi cải thiện DTW đều đi kèm giảm tốc độ chuyển động. Ở *cùng* tốc độ, chỉ co dự đoán về mean pose (không dùng văn bản) đã cho JPE body, JPE hand và PA body tốt hơn; PA hand chỉ dư dưới 0.015.
- **NFE nhiều hơn** thì chuyển động sống động hơn (tốc độ ngón 0.76 → 0.88 so với GT) nhưng DTW xấu đi. Metric và độ sống động đi ngược chiều nhau.
- **Chọn checkpoint sớm** (theo val flow loss) làm DTW xấu đi có ý nghĩa, nên quy tắc này bị loại.
- **Đổi path NG → geoshift khi suy diễn**: body tốt hơn, tay xấu hơn. Mô hình NG phụ thuộc tính không đồng nhất của path.
- **Phép thử σ (E1-0)**: σ = 0.5 và 0.75 sinh path giống hệt nhau, nên chênh lệch ở Table 4 nhiều khả năng là nhiễu; σ = 0.25 sinh một lịch nhiễu cực đoan. H1 ("NG tác động qua lịch nhiễu") không bị bác bỏ.
- **→ Trả lời tạm thời cho "có tốt hơn SignNMFlow không":** *chưa*, với các chỉnh sửa chỉ ở khâu suy diễn. Cải thiện thật phải đến từ thay đổi ở khâu huấn luyện (E4 FK loss, E5 CFG, E6 prior, E7 self-conditioning), và phải nằm dưới đường "co về trung bình". Các thí nghiệm này cần GPU và chưa chạy.

## Nội dung

| File | Nội dung |
|---|---|
| [01_phan_bien_paper.md](01_phan_bien_paper.md) | Phản biện lý thuyết, thực nghiệm và đánh giá; mỗi nhận định gắn nhãn [Đo], [Toán], [Văn bản] hoặc [Suy luận]. |
| [02_doc_code_signmflow.md](02_doc_code_signmflow.md) | Đọc từng hàm của `methods/signmflow`: điểm yếu, dòng code cần sửa, ánh xạ sang thí nghiệm. |
| [03_literature_review.md](03_literature_review.md) | Flow matching (RFM, MFM, lịch nhiễu, tham số hoá dự đoán), biểu diễn quay và chuyển động, SLP SOTA 2024–2026, cấu trúc ngôn ngữ ký hiệu, phê phán đánh giá. |
| [04_ke_hoach_thuc_nghiem.md](04_ke_hoach_thuc_nghiem.md) | Kế hoạch v2: bộ đo, giả thuyết có thể bác bỏ, đối chứng dương, factorial, ngân sách, quy tắc quyết định. |
| [05_phan_bien_noi_bo.md](05_phan_bien_noi_bo.md) | Devil's advocate: 24 phát hiện, phản hồi, thay đổi v1 → v2, rủi ro, điều kiện bác bỏ. |
| [07_lo_trinh_uu_tien.md](07_lo_trinh_uu_tien.md) | **Bản bàn giao để chạy:** kiểm tra độ đầy đủ (điểm yếu → thí nghiệm), việc cần code C1–C10, thứ tự ưu tiên P0–P6 với lý do nên và không nên chạy, điều kiện dừng, lịch, mẫu báo cáo. |
| [06_ket_qua_thuc_nghiem.md](06_ket_qua_thuc_nghiem.md) | Kết quả vòng 1: 20 biến thể suy diễn (bootstrap cặp, rào chắn tốc độ, đối chứng co về trung bình) và phép thử σ. |
| [08_thiet_ke_lai_signtaf.md](08_thiet_ke_lai_signtaf.md) | **Thiết kế lại có novelty (v2, 22/11/2026):** bốn nguồn phương sai và cách DTW đối xử với từng nguồn; hướng R (kết quả: truy hồi δ, vệ sinh, τ-loss) và hướng S (phân tích pha–biên độ; EPA-Flow); cổng không cần huấn luyện P0.S; **đã đo trên CSL dev (chỉ GT + dự đoán v2 đã lưu): body của v2 ngang người, tay kém người 14–18%; căn thời gian phi tuyến xoá 1/3 khác biệt giữa người ký ở tay (null ≈ 0), nên khoảng trống thật nằm ở tay và có thành phần nhịp điệu**; SOTA giao thức SOKE; kế hoạch G0–G4. |
| [09_phan_bien_08.md](09_phan_bien_08.md) | Phản biện độc lập của 08 v1 (25 điểm, không điểm nào bị bác) và phản hồi; phát hiện mới: sampler re-planned của paper thiên lệch trên manifold. |
| `probes/` | Script CPU và JSON có thể chạy lại. |
| `sources/` | Sổ nguồn: 37 (flow matching) + 33 (chuyển động) + 47 (SLP và đánh giá) tài liệu đã xác minh, cùng bản phản biện tiếng Anh đầy đủ. |

## Probe (chạy lại từ thư mục gốc repo; chỉ CPU, split val)

```bash
export CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src OMP_NUM_THREADS=4
R=outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824
python docs/proposal7/probes/probe_theory.py --dataset phoenix_smplx --run $R     # P1–P5
python docs/proposal7/probes/probe_parallel.py                                     # P3b
python docs/proposal7/probes/probe_da_progress.py phoenix_smplx val 600            # κ, khoảng cách tới đường cong, s theo bộ phận
python docs/proposal7/probes/probe_da_energy.py phoenix_smplx                      # cận dưới năng lượng tái tham số hoá
python docs/proposal7/probes/probe_metrics.py --dataset phoenix_smplx --run $R --n 150   # P6
```

`probe_da_*.py` (từ phản biện nội bộ) dùng đường dẫn checkpoint v2 cố định trong file và ghi `probe_da_check_*.json` và `probe_da_energy_*.json` cạnh script.

| Probe | Đại lượng | Phoenix | CSL |
|---|---|---|---|
| P1 | Tỉ lệ năng lượng của số hạng mật độ ρ‖v‖²/m (p95) | 1e−4 | 1e−4 |
| P2 | Vị trí hướng x_t → x1 trong [λ_min, λ_max] của G⁻¹, trung vị (body/tay), t < 0.3 so với t > 0.7 | 0.99/0.97 so với 0.79/0.60 | tương tự |
| P-DA | Năng lượng: geodesic / learned / cận dưới tái tham số hoá | 30.53 / 23.76 / 23.66–23.92 | 30.84 / 23.75 / 23.89–24.16 |
| P-DA | Khoảng cách tới đường cong geodesic, trung vị t = 0.3/0.5/0.7 (rad) | 0.042/0.067/0.078 | 0.041/0.068/0.083 |
| P-DA | Δlog m tại cùng tiến độ (trung vị) | −0.004 … −0.017 | −0.003 … −0.016 |
| P-DA | s fit trên κ: toàn bộ / body / tay / hàm | 2.56 / 2.63 / 2.50–2.55 / 2.83 | 2.54 / 2.61 / 2.47–2.54 / 2.82 |
| P3b | cos(ε, Log_{x0}x1), t ∈ [0.1, 0.9] | −0.94 … −0.97 | −0.92 … −0.97 |
| P4 | Tỉ lệ w < 0.3, tay / body; tỉ lệ đổi dấu theo thời gian | 0 / 0.7%; ~1e−4 | — |
| P5 | Góc quay trung bình còn lại tới x1 tại t = 0.5: geodesic / NG / 6D-Gauss tuyến tính | 65° / 90° / 85° | tương tự |
| E6-prior (đo) | Góc quay p0 → x1: hemisphere / wrapped normal σ0 = 0.25 / tư thế trung bình | 126.5° / 51° / 21° | — |

## Kết quả probe metric

Phoenix dev, 150 chuỗi đầu, checkpoint v2 seed 0, giao thức SOKE: body = 12 khớp thân trên, căn pelvis; tay = 21 joint, căn cổ tay; PA = Procrustes; DTW tích luỹ. Nguồn: `probes/probe_metrics_phoenix_smplx.json`. Mọi biến thể ngoài `GT câu khác` đều **không dùng văn bản**.

| Biến thể | JPE body | JPE hand | PA body (ua) | PA hand | Tốc độ body / ngón (mm/frame) |
|---|---|---|---|---|---|
| Dự đoán v2 | 5.04 | 6.87 | 4.47 | 1.334 | 4.45 / 23.5 |
| Mean pose tĩnh | 5.53 | 7.95 | 4.87 | 1.542 | 0 / 0 |
| Co 25% về mean pose | **4.77** | **6.70** | **4.24** | **1.290** | — |
| Co 50% về mean pose | **4.77** | 6.92 | **4.24** | **1.323** | — |
| Làm mượt Gaussian σ = 1 frame | **5.01** | **6.86** | **4.45** | **1.317** | — |
| Làm mượt Gaussian σ = 3 frame | **4.96** | 6.91 | **4.41** | **1.315** | **1.22 / 5.6** |
| GT của câu khác | 6.82 | 8.81 | 6.11 | 1.697 | — |
| (GT) | 0 | 0 | 0 | 0 | 4.86 / 26.1 |

Số **đậm** là tốt hơn dự đoán gốc.

**Đọc bảng.**

1. Metric *có* phân biệt đúng câu với sai câu: GT câu khác kém xa. Mô hình v2 cũng thắng mean pose tĩnh ở cả 4 cột.
2. Nhưng các phép biến đổi không chứa nội dung, kéo dự đoán về trung bình hoặc làm mất chuyển động, lại *cải thiện* metric:
   - co 25% về mean pose thắng dự đoán gốc ở cả 4 cột;
   - làm mượt σ = 3 xoá khoảng 75–80% tốc độ mà vẫn thắng 3/4 cột.

   Đây là cùng hiện tượng mà báo cáo SLRTP 2025 mô tả. Mọi "cải thiện" DTW cỡ 0.05–0.3 vì vậy cần rào chắn động học và ngữ nghĩa trước khi được tin.
3. Tốc độ ngón tay của GT (26 mm/frame) phần lớn là dao động tần số cao. Làm mượt σ = 3 giảm nó xuống 5.6, gợi ý rằng pseudo-GT có nhiều giật do fitting.
   - Tỉ lệ tốc độ thô (0.90–0.92 cho v2) vì vậy là thước đo lẫn lộn.
   - Rào chắn M2 trong `04` phải so sánh trên tín hiệu đã lọc thông thấp, hoặc theo dải tần.
4. Sai số độ dài trung bình là 16.2 frame trên chuỗi dài khoảng 100 frame (dự đoán 99.4, GT 100.6).
   - Độ dài là một nguồn phương sai DTW đáng kể, nên các arm cần dùng chung độ dài (`04` §1.2).


## Phạm vi bằng chứng và giới hạn

- **Thời điểm và mã nguồn.** Phân tích ngày 2026-11-19 trên HEAD `c859c93`. Probe dùng checkpoint v2 seed 0 (Phoenix, CSL), **chỉ split val** (train Phoenix làm đối chứng ở P-DA).
- **Đã chạy:** E0.1–E0.3 (suy diễn trên CPU với checkpoint v2) và E1-0 (huấn luyện chỉ correction, 4000 bước, CPU). **Chưa chạy:** mọi thí nghiệm cần huấn luyện generator (GPU đang chạy job How2Sign; ràng buộc 1 tiến trình). Các kỳ vọng còn lại trong `04` là dự đoán có thể bác bỏ.
- **Ràng buộc nguồn.** Theo yêu cầu, đề xuất không tham khảo code, thiết kế hay kết quả của method khác trong SLP, kể cả `docs/proposal` chưa commit và `docs/proposal1–6`. Chỉ dùng `methods/signmflow`, metric và loader dùng chung, và tài liệu công khai.
- **Hình thức tổng quan.** Literature review là tổng quan tường thuật/tích hợp, không phải systematic review. Nguồn được xác minh qua trang arXiv hoặc proceedings. Các chi tiết then chốt sau mốc kiến thức của mô hình được kiểm tra lại trên HTML gốc: MFM Eq. 4/6/LAND; báo cáo SLRTP 2025; sự tồn tại của RMG, Conditional Collapse.
- **Công cụ AI.** Đề xuất được soạn với hỗ trợ AI (Claude Code): tìm và tóm tắt tài liệu, viết và chạy probe, phản biện nội bộ bằng một agent độc lập. Mọi con số [Đo] tái tạo được bằng các script ở trên.

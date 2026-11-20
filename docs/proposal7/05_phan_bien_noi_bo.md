# 05 — Phản biện nội bộ (devil's advocate) và cách đề xuất đã thay đổi

**Cách làm.** Một reviewer độc lập (agent riêng, không thấy lập luận của người viết ngoài văn bản nháp) đọc:

- paper;
- code `methods/signmflow`;
- metric dùng chung;
- bản nháp `01`–`04`.

Reviewer tự chạy kiểm tra CPU trên checkpoint v2 (val, cộng train Phoenix làm đối chứng). Bản đầy đủ bằng tiếng Anh nằm ở `sources/devils_advocate_review_en.md`. Script và JSON của reviewer: `probes/probe_da_progress.py` (ghi ra `probe_da_check_*.json`) và `probes/probe_da_energy.py` (ghi ra `probe_da_energy_*.json`).

Mức độ: **CRITICAL** là vô hiệu một khẳng định chính hoặc khả năng đi tới kết luận của kế hoạch. **MAJOR** là sai hoặc thiếu căn cứ như đang viết. **MINOR** là cần sửa nhưng không chí mạng.

---

## 1. Các phát hiện và phản hồi

| # | Mức | Phát hiện của reviewer | Kiểm tra lại | Phản hồi |
|---|---|---|---|---|
| 1 | CRITICAL (văn bản) | "Lệch không gian 0.28 rad" thực chất là **độ trễ thời gian**, vì so hai đường tại cùng t. Khoảng cách thật tới *đường cong* geodesic là 0.04–0.08 rad. | Đã đọc mã và JSON: Phoenix 0.042/0.067/0.078, CSL 0.041/0.068/0.083 rad tại t = 0.3/0.5/0.7. | **Chấp nhận.** Đã sửa `01` A.3, README; quy tắc quyết định E1 trong `04` không còn dựa trên 0.28 rad. |
| 2 | CRITICAL (văn bản) | "Δlog m ≈ −0.6, path đi qua vùng mật độ thấp hơn" là hệ quả của độ trễ. Tại cùng tiến độ, Δlog m ≈ −0.01. | Đã kiểm tra: −0.004 … −0.017. | **Chấp nhận.** Phát biểu đúng: path *đi qua cùng những điểm* như geodesic (không bẻ về vùng dày như Fig. 1), chỉ đi chậm hơn ở gần nguồn. |
| 3 | MAJOR | s = 2.75 phụ thuộc cách fit. Fit trực tiếp trên κ cho 2.55; theo bộ phận 2.5–2.8; theo mẫu p10–p90 1.85–3.15. Về năng lượng, s ∈ [2, 2.75] là tương đương. | Đã kiểm tra JSON (Phoenix 2.56, CSL 2.54). | **Chấp nhận.** Báo cáo s ≈ 2.5 ± 0.3; E1-H1 dùng lưới s ∈ {2, 2.5, 3}. |
| 4 | (củng cố) | Bằng chứng mạnh nhất cho H1 là *năng lượng*: learned 23.76 ≈ cận dưới tái tham số hoá thời gian 23.66–23.92 (Phoenix); CSL 23.75 so với 23.89–24.16. Bẻ cong không đem lại gì dưới mục tiêu của paper. | Đã đọc `probe_da_energy_*.json`. | **Chấp nhận.** Đưa lên đầu `01`. |
| 5 | MAJOR | Phê phán "mật độ bị triệt tiêu" cần ghi công Eq. 44 của paper, vốn tự viết G⁻¹ = Σ + (ρ/m)I. Lỗi nằm ở *diễn giải* trong App. C.3, không ở toán. | Đúng: Eq. 43–44 trong App. C.3. | **Chấp nhận.** Đã viết lại `01` A.2. |
| 6 | MAJOR | "Hướng về x1 đắt nhất" gần như là hệ quả tất yếu khi ở xa dữ liệu, và giảm còn 0.6–0.8 khi t > 0.7. Time-shift có lẽ cũng xuất hiện với metric kiểu MFM, nên E6 cũ ("metric đúng chiều") mất động cơ. | Đã kiểm tra: t > 0.7 cho 0.60–0.79. | **Chấp nhận.** Giới hạn khẳng định cho t < 0.3. E6 cũ đổi thành E11 với ưu tiên rất thấp. |
| 7 | MAJOR | Bỏ sót lập luận sạch nhất: với target-prediction, vận tốc path không vào loss; path chỉ tác động qua phân phối x_t và trạng thái của sampler. Có một cơ chế thứ ba: **tính không đồng nhất của κ** giữa mẫu và joint (giống tăng cường mức nhiễu theo token). | — | **Chấp nhận.** Thêm arm **NG-parallel** để tách phần bẻ cong khỏi tính không đồng nhất (E1-H1). |
| 8 | MAJOR | Thiếu các phép thử không cần huấn luyện: hoán đổi path và lưới khi suy diễn trên model NG đã có. | — | **Chấp nhận.** Thêm E0.3. |
| 9 | MAJOR | Table 4 (σ) của paper là phép thử bác bỏ rẻ cho H1: nếu s(0.25) ≈ s(0.5) thì time-shift không giải thích được việc σ = 0.25 tệ hơn GE. | — | **Chấp nhận.** Thêm E1-0 (chỉ stage correction). |
| 10 | MAJOR | Giải thích khác: NG có thể "thắng" DTW bằng cách **hồi quy về trung bình nhiều hơn**. Lần gọi mạng cuối của NG thấy trạng thái ở mức nhiễu khoảng 0.41, so với khoảng 0.2 của GE. | **Đã đo bổ sung (P6):** co dự đoán v2 25% về tư thế trung bình (không dùng văn bản) làm *tốt lên* cả 4 cột DTW (JPE body 5.04 → 4.77; PA hand 1.334 → 1.290). | **Chấp nhận và củng cố.** Mọi thay đổi tăng trọng số vùng nhiễu, kể cả shift lớn hơn, đều phải qua rào chắn M2 và M3. |
| 11 | CRITICAL (kế hoạch) | **Không có đối chứng dương.** Nếu NG ≈ GE trong bản tái lập thì "geoshift ≈ NG" là tầm thường. | — | **Chấp nhận.** E1-gate (GE so với NG, 2–3 seed) là so sánh đầu tiên. Nếu gate âm, kết luận chính là "NG không có hiệu ứng đo được trong bản tái lập". |
| 12 | CRITICAL (kế hoạch) | **Metric không có khả năng phân định.** DTW thưởng trung bình hoá; B-T bão hoà trên Phoenix (11.34 so với trần 11.49) và hỏng trên CSL (6.00 > 4.74); tỉ lệ tốc độ một phía có thể bị chơi bằng cách thêm giật. | P6 xác nhận phần DTW (mục 10). | **Chấp nhận.** Giai đoạn 0 dựng M1 (evaluator truy hồi đã hiệu chỉnh), M2 (dải tốc độ và gia tốc hai phía), M3 (khoảng cách tới mean pose), M4 (hoán đổi text) và độ dài cố định. CFG (E5) chỉ chạy khi đã có M1. |
| 13 | MAJOR | Ngưỡng nhiễu chỉ từ 1 seed thêm; chọn checkpoint trên 200 chuỗi dev rồi báo cáo trên cả 519 chuỗi gây rò rỉ; chọn max-of-6 snapshot gây winner's curse. | — | **Chấp nhận.** Baseline × 3 seed ở độ dài sàng lọc; bootstrap cặp; chọn trên tập con và báo cáo trên phần còn lại. |
| 14 | MAJOR (quy tắc repo) | Đề xuất tính DTW bên trong `training.validation_loss` vi phạm CLAUDE.md. | Đúng. | **Chấp nhận.** Chọn snapshot ở bên ngoài, qua `scripts/run.py` với `init.generator`. |
| 15 | MAJOR (protocol) | CSL 2 seed và How2Sign 1 seed vi phạm yêu cầu ≥ 3 seed; ngân sách tinh chỉnh không cân bằng giữa baseline và biến thể. | Đúng (`docs/experiment_protocol.md`). | **Chấp nhận.** ≥ 3 seed; vệ sinh tối ưu áp dụng cho cả baseline. |
| 16 | MAJOR | Sàng lọc 150 epoch có thể đảo thứ hạng so với 300 epoch, vì nhiều can thiệp mang tính regularizer. | — | **Chấp nhận.** Độ dài sàng lọc chọn sau E0.1; hai ứng viên tốt nhất được xác nhận ở độ dài đầy đủ. |
| 17 | MAJOR | Shift, phân phối t, trọng số w(t), prior và lưới suy diễn cùng xoay *một* núm. | — | **Chấp nhận.** Thay bằng factorial E2-F (s_train × s_inf); bỏ E3c riêng; tinh chỉnh lại s sau khi đổi prior. |
| 18 | MAJOR | Độ dài dự đoán là confound không được kiểm soát, vì DTW tích luỹ và length head phụ thuộc text encoder. | — | **Chấp nhận.** File độ dài dùng chung; báo cáo thêm `dtw_jpe_norm`. |
| 19 | MAJOR | FK loss bị ước lượng thấp: joint tay của metric hồi quy từ **vertex đã posed** (cần LBS có posedirs). JPE hand giữ hướng toàn cục của cổ tay, nên phải tính qua toàn chuỗi khớp. | Đã đọc `smplx_fk.py`: `_hand_regressor` dùng `o.vertices`. | **Chấp nhận.** Sửa E4-a: tính toàn chuỗi, đo proxy trước, chi phí 3–4 ngày code. |
| 20 | MAJOR | Ngân sách bỏ quên chi phí CPU của DTW; lịch quá lạc quan. | P6 thật sự chạy rất lâu trên 150 chuỗi khi máy tải cao. | **Chấp nhận.** Thêm khoảng 100 CPU-h; lịch 4–5 tuần. |
| 21 | MAJOR | "Gần như trùng MFM" là nói quá. Phát biểu đúng: dùng interpolant và cách huấn luyện của MFM mà không trích dẫn; metric và bối cảnh khác. | — | **Chấp nhận.** Đã sửa `01` và README. |
| 22 | MAJOR | "GE bị confound bởi lịch nhiễu" sai với bước MR → GE (lịch góc gần trùng). Confound thật là chuẩn ngoài manifold của x_t tuyến tính không được renormalize. | Đã kiểm tra P5: 65.3° so với 65.3° tại t = 0.5. | **Chấp nhận.** Đã sửa `01` mục 6 và B.1. |
| 23 | MAJOR (nguồn gốc) | Các mục [Đo] về probe metric trích dẫn một JSON chưa tồn tại. | Đúng tại thời điểm review. | **Đã xử lý.** P6 đã chạy; số liệu ở README và `01` C.2. |
| 24 | MINOR | Table 8 (+0.7%, một run) không phải bằng chứng; dựa vào NFE sweep của v1. Diễn đạt chính xác hơn về độ dài trong DTW. Lấy mẫu `arr[:2000]` trong `probe_parallel.py` là frame liên tiếp. Eigvalsh float32 làm λ_min không tin cậy. | Đúng. Reviewer đã kiểm tra rằng mẫu ngẫu nhiên cho cùng kết quả tới 3 chữ số. | **Chấp nhận.** Sửa văn bản. Số "log10 condition number" bị loại khỏi diễn giải; chỉ thống kê vị trí, vốn ổn định, được giữ. |

**Điểm phản biện không chấp nhận hoàn toàn.**

- Reviewer đề xuất *trì hoãn* E4 (FK loss) cho tới sau factorial lịch nhiễu. Chúng tôi giữ E4-c (trọng số cánh tay đòn, rẻ) trước E2-F, vì nó trực giao với lịch nhiễu và nhắm thẳng vào lệch pha giữa loss và metric.
- E4-a vẫn chỉ chạy sau khi có M1–M3.

---

## 2. Thay đổi từ kế hoạch v1 sang v2

| v1 | v2 |
|---|---|
| E0 chỉ đánh giá checkpoint và sampler | E0.0 dựng bộ đo M1–M4 + độ dài cố định + bootstrap; E0.3 hoán đổi path khi suy diễn; E1-0 phép thử σ |
| E1-b geoshift s = 2.75 so với NG | E1-gate GE so với NG (đối chứng dương) → E1-H1 NG-parallel + geoshift s ∈ {2, 2.5, 3}, biên tương đương đặt trước |
| E1-d, E3c, phân phối t là các arm riêng | Factorial E2-F: s_train × s_inf |
| Seed: 1 baseline thêm; CSL 2; How2Sign 1 | 3 seed baseline ở độ dài sàng lọc; ≥ 3 seed cho số báo cáo |
| Chọn checkpoint bằng DTW bên trong training | Chọn bên ngoài method, trên tập con, báo cáo trên phần còn lại |
| FK loss "~150 dòng, chỉ khung xương" | Toàn chuỗi + LBS có posedirs cho joint tay hồi quy từ vertex; đo proxy trước; 3–4 ngày |
| CFG ở giai đoạn 3 | CFG sau khi có M1, so ở cùng compute |
| Khoảng 280 GPU-h, khoảng 12 ngày | Khoảng 400 GPU-h + 100 CPU-h, khoảng 4–5 tuần |

---

## 3. Rủi ro còn lại và cách nhận biết sớm

1. **Mọi arm nằm trong nhiễu.**
   - Nhận biết: E1-gate âm, và B×3 cho khoảng tin cậy rộng hơn mọi hiệu ứng ở E0.
   - Phản ứng: báo cáo kết quả âm (NG không có hiệu ứng; DTW không phân định được), rồi chuyển trọng tâm sang M1 (ngữ nghĩa).
2. **"Thắng" nhờ trung bình hoá.**
   - Nhận biết: M3 thu hẹp, hoặc M2 tụt dưới 0.8.
   - P6 cho thấy co 25% về trung bình đã cải thiện mọi cột DTW, nên rủi ro này là *thật*.
   - Phản ứng: loại arm đó dù DTW tốt.
3. **Evaluator M1 không đủ mạnh** (dữ liệu nhỏ, câu Phoenix lặp lại nhiều).
   - Nhận biết: hiệu chỉnh ở E0.0 thất bại.
   - Phản ứng: dùng SiLVERScore/SignCLIP đã kiểm định trên Phoenix và CSL (`03` §4.2), hoặc back-translation likelihood.
4. **Pseudo-GT tay nhiễu** (Phoenix 210×260 px).
   - Hiệu ứng tay dưới khoảng 0.05 hand PA khó phân biệt với nhiễu của fitter.
   - Báo cáo cả JPE hand và PA hand; không tuyên bố thắng chỉ dựa trên PA hand.
5. **Lịch.** Job How2Sign và ràng buộc 1 tiến trình là nút cổ chai. Giai đoạn 0 (gần như chỉ CPU và suy diễn) có thể làm ngay khi GPU rảnh vài giờ.

## 4. Điều gì sẽ bác bỏ từng giả thuyết

| Giả thuyết | Bị bác bỏ nếu |
|---|---|
| H0-gate: NG > GE trong bản tái lập | Bootstrap cặp trên 3 seed không cho hiệu ứng có ý nghĩa trên CSL lẫn Phoenix |
| H1: lợi ích NG ≈ lịch nhiễu | E1-0: s(0.25) ≈ s(0.5); hoặc E1-H1: NG − geoshift vượt biên tương đương ở mọi s trong lưới |
| H1': tính không đồng nhất κ có ích | NG-parallel ≈ geoshift |
| H2: lịch của paper chưa tối ưu | Không ô nào của E2-F thắng (2.5, 2.5) mà vẫn giữ M2 và M3 |
| H3: FK loss giảm lỗi tay | E4-c và E4-a không cải thiện JPE hand hoặc PA hand, hoặc chỉ cải thiện bằng cách thu hẹp M3 |

## 5. Sau vòng thực nghiệm 1 (`06`)

- Rủi ro 2 ("thắng" nhờ trung bình hoá) **đã xảy ra đúng như dự báo**: mọi chỉnh sửa sampler cải thiện DTW đều nằm trên hoặc ngay sát đường "co về trung bình". Quy tắc (vi) trong `04` §1.3 được thêm để chặn điều này.
- Dự đoán của chúng tôi ở E0.1 (early selection có lợi) **sai**. Reviewer đã cảnh báo rằng val flow loss không phải proxy, và điều đó đúng.
- E1-0 không bác bỏ H1, nhưng cho thấy chênh lệch σ = 0.5 so với 0.75 trong Table 4 của paper nhiều khả năng là nhiễu một-run.


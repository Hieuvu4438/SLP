# 08 — Thiết kế lại SignNMFlow: hướng nghiên cứu có novelty và lộ trình vượt paper (v2)

> **Lịch sử.**
> - **v1** (22/11/2026, sáng) đặt giả thuyết "một cơ chế duy nhất": nhịp điệu gây mất biên độ, và DTW thưởng cho đúng sự nhoè đó.
> - Phản biện độc lập (`09`) chỉ ra hai lỗi:
>   - giả thuyết đó *mâu thuẫn* với chính toy của v1;
>   - một số mệnh đề toán sai hoặc thiếu giả thiết.
> - **v2** (bản này) sửa toàn bộ:
>   - nhịp điệu chỉ là *một trong bốn* nguồn phương sai, cần đo;
>   - tách hai hướng: *kết quả* (R) và *khoa học/novelty* (S);
>   - thay β bằng một mô hình phân rã pha–biên độ có mô hình nhịp điệu riêng;
>   - hạ các nhãn novelty;
>   - thêm đối chứng và thống kê.
>
> Bảng phản hồi từng điểm phản biện nằm ở `09`.

**Câu hỏi.** Giữ ý tưởng gốc của SignNMFlow (flow matching trên manifold quay, dự đoán đích, sampler re-planned, "tự nhiên") nhưng thiết kế lại sao cho:

- (a) có cơ chế mới với trực giác rõ;
- (b) vượt số paper báo cáo một cách *thật*, tức không phải nhờ hồi quy về trung bình;
- (c) mọi tuyên bố đều kiểm định được.

**Phạm vi.** Đây là tài liệu nghiên cứu và kế hoạch. **Không chạy thí nghiệm nào trên mô hình.** Có hai loại tính toán phụ:

- đếm thống kê trên annotation train/dev (mục 2.2);
- toy numpy trên dữ liệu tổng hợp (mục 2.5).

Không đọc split test.

**Nhãn bằng chứng.**

- [Đo]: số đo trên checkpoint hoặc dữ liệu đã có (`01`, `06`);
- [Toán]: suy ra trực tiếp;
- [Toy]: mô phỏng tổng hợp;
- [V]/[A]: tài liệu đã đọc toàn văn / chỉ đọc tóm tắt (mục 4, `sources/lit_08_signtaf.md`);
- [Suy luận]: chưa kiểm chứng.

---

## 0. Tóm tắt

1. **Lộ trình `07` là tối ưu kỹ thuật, không phải đóng góp khoa học.**
   - Mỗi arm là kỹ thuật chuẩn, kỳ vọng 1–3%.
   - Phần cải thiện *thật* đo được tới nay chỉ khoảng 1%: PA-hand dư < 0.015 ở cùng tốc độ (`06`).

2. **v2 nhoè vì lấy trung bình qua bốn nguồn phương sai.** DTW cộng dồn đối xử với chúng khác nhau:

   | Nguồn | DTW cộng dồn khi bị lấy trung bình |
   |---|---|
   | (a) nội dung: ký hiệu nào, thứ tự nào, trên câu mới | **thưởng** |
   | (b) không gian / phong cách người ký | **thưởng** |
   | (c) nhịp điệu / pha | **phạt** [Toy] |
   | (d) lỗi mô hình (chưa hội tụ; sampler 5 bước) | thưởng phần phương sai (kỳ vọng 0); không tác động lên phần thiên lệch |

   - Hai quan sát "co về trung bình làm DTW tốt hơn" và "NFE 1 tốt nhất" vì vậy là bằng chứng rằng **(a), (b) hoặc phần phương sai của (d)** chiếm ưu thế trong phần v2 lấy trung bình. Trung bình hoá cũng thưởng cho lỗi mô hình/sampler có kỳ vọng 0.
   - Phần của (c) *chưa biết*. Đó là câu hỏi thực nghiệm trung tâm, đồng thời là cơ hội novelty chính: phân tích pha–biên độ của SLP.

3. **Hai hướng song song:**

   | Hướng | Mục tiêu | Nội dung | Kỳ vọng |
   |---|---|---|---|
   | **R** (kết quả) | Vượt paper một cách trung thực | Chẩn đoán giao thức (độ dài oracle; khoảng cách How2Sign) → **bộ nhớ truy hồi δ** (đòn bẩy lớn nhất đã ghi nhận trong SLP, −20–32%) + vệ sinh (E3) + FK loss + loss tiếp tuyến không thiên lệch. Luôn có rào chắn đường biên. | CSL khả thi; How2Sign chỉ sau khi chẩn đoán khoảng cách |
   | **S** (novelty) | Đóng góp khoa học | Phân rã pha–biên độ của phương sai *giữa người ký* và *phương sai dự đoán của mô hình*. Nếu phần nhịp điệu đáng kể → **EPA-Flow**: template barycentric + mô hình pha + RFM biên độ trên (S³)^J trong thời gian đã đăng ký (registered). Nếu không → paper phân tích "phương sai của SLP nằm ở đâu và DTW thưởng cho cái gì", với δ làm minh hoạ. | Novelty trung bình–cao ở phần phân tích; trung bình ở phương pháp |

4. **Cổng chính không cần huấn luyện (P0.S).**
   - Lấy K = 8 mẫu v2 cho mỗi câu val.
   - So median hình học *sau căn chỉnh* với median *theo khung*, tại cùng tỉ lệ quãng đường.
   - Lợi ích phải vượt lợi ích của cùng phép so trên null chỉ-không-gian.
   - Phương án dự phòng là ensemble B×3 seed.
   - **Đối chứng con người (mới).** 280 câu dev của CSL có ≥ 2 người ký khác nhau [Đo]. DTW giữa hai người ký là mức mà một sampler *hoàn hảo* đạt được. Nếu mô hình "tốt hơn con người", đó là dấu hiệu metric thưởng cho trung bình.

5. **Đã đo: đối chứng người–người trên CSL dev (P0.S-H, 280 câu, chỉ GT + dự đoán v2 đã lưu).**

   | | JPE body | JPE hand | PA body | PA hand |
   |---|---|---|---|---|
   | Người A so với người B | 5.58 | 6.29 | 4.76 | 1.12 |
   | v2 | 5.54 | 7.18 | 4.89 | 1.32 |

   - **Body:** v2 đã *ngang người* (Δ −0.04, CI chứa 0). Giảm DTW body thêm là hành vi ước lượng.
   - **Tay:** v2 kém người 14–18% (CI loại 0). **Khoảng trống thật nằm ở tay**, và hướng R dồn vào đó.
   - **Đã đo thêm (P0.2-lite, cùng 280 cặp, chỉ GT).** Căn thời gian phi tuyến xoá được 33–35% khác biệt giữa hai người ký ở tay và 16% ở body. Null cùng phổ: khoảng 0. Thành phần nhịp điệu của *dữ liệu* có thật và lớn nhất ở tay. Việc *mô hình* có lấy trung bình qua nó hay không vẫn chờ P0.S.

6. **Mục tiêu "vượt paper", nói thẳng.**
   - **CSL** (thiếu 3–6%): khả thi, chủ yếu nhờ δ và các thành phần cho tay. Riêng PA-hand có khoảng trống khoảng 15% tới mức người.
   - **How2Sign** (thiếu 9–12%): *chưa đáng tin* với kế hoạch này. Cần chẩn đoán vì sao bản tái lập (7.45/10.67) kém MaDiS (6.59/9.45) trên cùng giao thức trước.
   - **Phoenix** (thiếu 11–18%): paper không nói cách chọn độ dài, trong khi metric cộng dồn nhạy với độ dài. P0.1 đo phần này. Trong lúc chờ, chỉ so với bản tái lập.

---

## 1. Vì sao các hướng trong `07` chưa đủ

| Nhóm arm `07` | Bản chất | Kỳ vọng | Novelty |
|---|---|---|---|
| E3 (EMA, text lr, warmup) | Vệ sinh | Nhỏ trên DTW | Không |
| E4 (FK loss, lever-arm, velocity) | Khớp loss với metric | Trung bình; vẫn là trung bình theo khung | Thấp |
| E2-F, E1-H1 (lịch nhiễu, path) | Khoa học về paper | Vòng 1: đổi lưới suy diễn chỉ là trung bình hoá | Thấp |
| E5–E7 (CFG, prior, self-cond) | Kỹ thuật FM chuẩn | Không chắc | Thấp |

`07` thiếu một mô hình về *nguồn gốc* lỗi. Không có nó thì mọi arm chỉ dò trên cùng một đường đánh đổi DTW–tốc độ.

---

## 2. Chẩn đoán: phương sai của SLP nằm ở đâu

### 2.1 Quan sát đã đo [Đo, `01` C.2, `06`]

| Quan sát | Số liệu |
|---|---|
| Thiếu biên độ | Tốc độ thông thấp v2 = 0.65 (body) / 0.76 (ngón) × GT |
| Co về trung bình làm DTW tốt hơn | Co 15%: −0.18 JPE body, −0.18 JPE hand, −0.04 PA hand |
| Ít bước sampler → gần kỳ vọng hậu nghiệm | NFE 1: DTW tốt nhất, tốc độ 0.42/0.48 |
| Nhiều bước → sống động | NFE 10: tốc độ 0.73/0.88, DTW xấu nhất |
| **Thiếu biên độ giảm dần theo huấn luyện** | Tỉ lệ tốc độ 0.46 (epoch 50) → 0.65 (epoch 300); DTW cũng tốt dần |
| Ngón ít suy giảm hơn body | 0.76 so với 0.65. Lưu ý: pseudo-GT ngón có giật cao tần, nên tốc độ GT bị thổi phồng |

### 2.2 Cấu trúc dữ liệu [Đo, đếm trên annotation train/dev, không dùng mô hình]

| | Phoenix-2014T | CSL-Daily | How2Sign |
|---|---|---|---|
| Train / dev | 7,092 / 519 | 18,399 / 1,077 | 30,965 / 1,724 |
| Câu train duy nhất | 6,849 | **6,578** (mỗi câu 2–3 lần ký; 6,533 câu có ≥ 2 người ký) | 29,919 |
| Câu dev có ≥ 2 người ký | — | **280 câu** (tổng 797 câu duy nhất) | — |
| Câu dev có y hệt trong train | 4.8% | 0.2% | 2.5% |
| Cosine TF-IDF tới câu train gần nhất, p25 / p50 / p75 / p90 | 0.31 / 0.41 / 0.58 / 0.81 | 0.22 / 0.27 / 0.35 / 0.45 | 0.24 / 0.30 / 0.38 / 0.53 |
| Độ phủ gloss của dev | 99.7% | 100% | — |
| Tần suất trung vị trong train của gloss dev | 385 | 209 | — |
| Alignment theo khung (`alignments['pami0']`, trạng thái HMM) | **train 7,092/7,092; dev 519/519** | Không | Không |

Ba hệ quả:

1. **Câu thì mới, ký hiệu thì quen.** Bài toán là *ghép ký hiệu*. Điều này ủng hộ bộ nhớ ở mức ký hiệu (δ).
2. **CSL có lặp nhiều người ký**, ở cả train lẫn dev. Từ đó đo được:
   - phân rã pha–biên độ giữa người ký (P0.2);
   - đường đối chứng con người trên dev (P0.S-H).
3. **Phoenix có ranh giới ký hiệu theo khung ở cả train và dev.** Từ đó đo được:
   - phân bố thời lượng ký hiệu của dự đoán (cần căn ranh giới qua DTW tới GT);
   - bộ nhớ đoạn ký hiệu (δ-gloss).
   - **Nguồn gốc `pami0`** (đã kiểm tra):
     - nằm trong file annotation `phoenix14t.{train,dev,test}` của bản phát hành SOKE (`data/README.md`);
     - mỗi chuỗi có số trạng thái đúng bằng số khung video gốc (25 fps);
     - trạng thái 0 là im lặng; khoảng 3,600 trạng thái, tức khoảng 1,200 gloss × 3;
     - nhiều khả năng là alignment HMM lai CNN-LSTM của Koller và cộng sự (TPAMI 2019). Bản phát hành không ghi HMM được huấn luyện trên dữ liệu nào.
   - **Quy tắc dùng:**
     - (i) cắt đoạn ký hiệu *train* cho δ-gloss;
     - (ii) metric phân tích trên val.
     - **Không bao giờ** là đầu vào mô hình ở val/test. Alignment dev được tạo từ chính video và gloss GT của dev.

### 2.3 Bốn nguồn phương sai và cách DTW đối xử với từng nguồn

| Nguồn | Ví dụ | Trung bình Euclid theo khung | DTW cộng dồn | Dấu hiệu quan sát | Phép đo phân biệt |
|---|---|---|---|---|---|
| (a) Nội dung | Câu mới: chọn ký hiệu, thứ tự | Nhoè giữa các ký hiệu khác nhau | **Thưởng** (DTW không căn được hai ký hiệu khác nhau, nên trung bình gần cả hai) | Co về trung bình có lợi; M1 thấp | Phương sai giữa mẫu *sau* căn chỉnh, so với giữa người ký trên câu train |
| (b) Không gian / phong cách | Biên độ, không gian ký, tay thuận | Co biên độ | **Thưởng** | Như trên | P0.2: phần dư sau căn chỉnh |
| (c) Nhịp điệu / pha | Nhanh chậm, giữ tư thế | Lọc thông thấp | **Phạt** (toy: trung bình Euclid tệ hơn cả một mẫu đơn) | Thiếu biên độ *và* DTW tệ | f_τ so với null (P0.S, P0.2) |
| (d) Lỗi mô hình | Chưa hội tụ; sampler ít bước | Nhoè | Thưởng phần phương sai (kỳ vọng 0); không tác động lên phần thiên lệch | Biên độ tăng theo epoch | Đường cong theo epoch; NFE |

**Hai đính chính so với v1:**

- **Hình chữ U không phân biệt được (b) với (c).** Phương sai không gian cũng gần 0 ở hai đầu câu (tư thế nghỉ). Toy phản biện: chỉ có biến thiên không gian dạng sin² cho tỉ lệ tốc độ theo ba phần ba câu là 0.88 / 0.57 / 0.88, cũng là chữ U [Toy]. **P0.3 của v1 bị loại.**
- **"DTW thưởng cho sự nhoè do nhịp điệu" là sai chiều.** DTW thưởng trung bình hoá kiểu (a)/(b) và phạt kiểu (c).

### 2.4 Lý thuyết (đã sửa)

**T1: nhịp điệu là bộ lọc thông thấp. Đây là *mô hình*, không phải định lý.**

- Với x(τ) = m(τ − δ(τ)), phân phối biên tại mỗi τ cho đúng: E[x](τ) = (m * p_{δ(τ)})(τ).
- Hệ số suy giảm exp(−ω²s²/2) cho tần số ω cần ba giả thiết:
  - δ có phân phối Gaussian;
  - s biến thiên chậm;
  - **δ độc lập với m**.
- Giả thiết cuối đáng ngờ với ngôn ngữ ký hiệu: ký nhanh thường kéo theo biên độ nhỏ hơn.
- T1 vì vậy chỉ được dùng để *thiết kế phép đo*, không phải bằng chứng.

**T2: ước lượng Bayes dưới DTW của evaluator.**

- a* = argmin_a E_x[DTW(a, x)] đúng theo định nghĩa. Nhưng DTW không phải metric, a* không nhất thiết duy nhất, và với chi phí *cộng dồn* thì **độ dài đầu ra là một phần của quyết định**.
- Chi phí cục bộ của evaluator là trung bình **khoảng cách L2 không bình phương** giữa các khớp (`pairwise_part_cost`). Với đường căn chỉnh cố định, lựa chọn tối ưu theo từng khung vì vậy là **median hình học theo từng khớp** của các khung đã căn, *không* phải trung bình số học. Hệ quả: DBA (trung bình số học) và soft-DTW với chi phí bình phương là barycenter cho *chi phí khác*.
- **Có sàn rời rạc hoá.** Hai phiên bản co giãn của cùng một đường cong có DTW khoảng 3.7–4.5, *không* phải ≈ 0 [Toy]. Vùng bị nén bị lấy mẫu thưa.
- Barycenter vì vậy thắng trung bình Euclid rõ rệt khi nhịp điệu đáng kể. Nhưng **không đảm bảo thắng một mẫu đơn** (toy: hàng 0.10/0 và 0.15/0).

**T3: trọng số không đổi nghiệm, và đính chính "Bregman".**

- Trong Rⁿ, với W(x_t, t, c) đối xứng xác định dương, nghiệm của argmin_f E‖f − x1‖²_W vẫn là E[x1 | x_t]. Đúng cả khi W tính tại sg(x̂1).
- **Trên (S³)^J thì khác.**
  - Loss L2 dây cung (chordal, như paper) cho nghiệm là trung bình dây cung đã chuẩn hoá. Loss geodesic cho nghiệm là trung bình Karcher.
  - Sampler re-planned đi theo Log_{x_t}(trung bình đó), *không* theo E[Log_{x_t} x1 | x_t], tức không theo vận tốc biên của RFM.
  - Vậy **sampler của chính paper (và v2) không đi theo trường biên.** Kể cả với mô hình hoàn hảo và NFE vô hạn, nó không đẩy chính xác p0 thành p1.
  - Đây là hiệu ứng *độ cong*, bậc hai theo độ rộng hậu nghiệm, và tiêu biến khi t → 1. Tác động khả dĩ là lên *trọng số các mode*, không rõ là lên độ nhoè.
  - Ở NFE 5, sai số rời rạc hoá và lỗi mô hình nhiều khả năng lớn hơn.
  - Điểm này *không mới*: đó là khoảng cách đã biết giữa loss x̂1 dây cung và loss tiếp tuyến của FrameFlow.
- **Sửa (τ-loss, lấy từ FrameFlow).** Huấn luyện đầu ra ở không gian tiếp tuyến, v̂ = Log_{x_t} x̂1, bằng λ(t)‖v̂ − Log_{x_t} x1‖².
  - FrameFlow dùng λ = 1/(1−t)², cắt tại t = 0.9.
  - Mọi λ(t) > 0 cho cùng nghiệm E[Log_{x_t} x1 | x_t, c] (Bregman trên không gian tuyến tính T_{x_t}). Nhưng λ(t) đổi cách phân bổ năng lực theo t, nên phải chọn và báo cáo.
  - Với path geodesic + time-shift toàn cục, vận tốc có điều kiện là κ′(t)/(1−κ(t)) · Log_{x_t} x1, với hệ số chỉ phụ thuộc t. Sampler vì vậy đi theo **κ′(t)/(1−κ(t)) · E[Log_{x_t} x1 | x_t]**, đúng vận tốc biên của RFM.
- **Điều kiện áp dụng τ-loss:**
  - **Không tương thích với path NG** (κ theo mẫu và theo khớp, cộng số hạng ε_θ vuông góc). Nếu E1-gate giữ NG thì phải hồi quy trực tiếp ẋ_t, hoặc bỏ NG.
  - **Quy ước dấu:** `geometry.logmap` của bản tái lập làm trên S³, với dữ liệu cố định trong một nửa cầu và không lật theo cung ngắn SO(3). Vì vậy path và Log nhất quán. Mọi bước lật dấu kiểu SO(3) sẽ phá sự nhất quán này.
  - **Điều kiện số:** "126.5°" trong `01` là góc *quay*, tức khoảng 63° trên S³. Hệ số khuếch đại lỗi của Log, θ/sin θ, vì vậy chỉ khoảng 1.2 ở nguồn đều, không đáng ngại.
  - **NFE 1** với τ-loss cho Exp(E[Log]). Đó không phải trung bình dây cung hay Karcher, nên đường biên của mô hình τ phải tính lại.
  - **Độ lớn hiệu ứng chưa biết.** Test N0 dùng hỗn hợp trên S² với đích đã biết.
- **Hệ quả cho γ (trọng số kéo lùi).** γ *không đổi nghiệm*. Lợi ích nếu có chỉ đến từ cách mạng hữu hạn phân bổ lỗi. Ngược lại, E4a (FK loss trên x̂1) *đổi nghiệm* thành trung bình trong không gian vị trí. Hai thứ này **không tương đương**.

**Sampler hoàn hảo và đường đối chứng con người.**

- Với y, x độc lập cùng phân phối p(· | c): E DTW(y, x) ≥ min_a E DTW(a, x).
- Vậy DTW trung bình giữa hai người ký *cùng một câu* là mức mà mẫu của một sampler hoàn hảo đạt được.
- CSL dev có 280 câu như vậy. Đây là đường tham chiếu trực tiếp để đặt cạnh v2 (5.72/7.38 trên dev) và số paper.
- **Cách đọc đúng.** Một ước lượng kiểu Bayes thắng một mẫu dưới *mọi* độ đo méo, kể cả MSE. Nếu mô hình "tốt hơn người", điều đó chỉ cho thấy đầu ra của v2 hành xử như một *ước lượng*, không phải như mẫu. Bệnh lý *riêng* của DTW là việc co về trung bình *không dùng văn bản* làm điểm tốt lên.

**Kết quả đã đo (P0.S-H)** [Đo].

- Probe `probes/probe_human_line_csl.py` → `probe_human_line_csl.json`.
- Chỉ dùng GT và dự đoán v2 đã lưu; không chạy mô hình; CPU; split val.
- Tập đo: 280 câu dev CSL, mỗi câu có hai người ký *khác nhau*. Chấm người A với người B theo cả hai chiều. v2 và tư thế trung bình được chấm trên đúng các chuỗi đó. Bootstrap theo câu.

| CSL dev (280 câu) | JPE body | JPE hand | PA body (ua) | PA hand |
|---|---|---|---|---|
| Người A so với người B | 5.58 | 6.29 | 4.76 | 1.12 |
| v2 (bản tái lập, seed 0) | 5.54 | 7.18 | 4.89 | 1.32 |
| Tư thế trung bình tĩnh (độ dài GT) | 6.75 | 9.40 | 5.94 | 1.75 |
| v2 − người, CI 95% | −0.04 [−0.21, 0.11] | **+0.89 [0.70, 1.10]** | +0.13 [−0.01, 0.28] | **+0.20 [0.16, 0.24]** |
| Tỉ lệ câu v2 tốt hơn người | 51% | 24% | 46% | 27% |

Tỉ lệ độ dài giữa hai người ký cùng câu: trung vị 1.12, p90 1.41. Nhịp chung (tempo) giữa người ký vì vậy khác nhau đáng kể.

**Hệ quả (quan trọng nhất của vòng nghiên cứu này).**

1. **Body đã chạm mức người–người.**
   - v2 ngang "một người khác ký cùng câu". Paper (CSL test 5.73) cũng ở quanh mức này; khác split nên chỉ so được tương đối.
   - Mọi cải thiện DTW body xuống dưới mức này đến từ hành vi *ước lượng điểm*, tức trung bình hoá, không phải từ chuyển động hiện thực hơn.
   - Vì vậy cải thiện body **bắt buộc** phải đọc trên đường biên ghép quãng đường.
2. **Tay là khoảng trống thật.**
   - v2 kém mức người–người 14% (JPE hand) và 18% (PA hand), với CI loại 0.
   - Một mô hình chỉ cần sinh *hình tay đúng như một người ký thật* đã giảm PA-hand khoảng 15%. Mức này vượt xa khoảng cách 5.8% PA-hand tới paper trên CSL, và không cần trung bình hoá.
   - Đây là mục tiêu "vượt paper một cách thật" rõ nhất: **hình tay** (PA-hand) và **vị trí tay** (JPE hand).
3. **Phân rã pha–biên độ giữa người ký** (P0.2-lite, đã chạy) [Đo].
   - Probe: `probes/probe_phase_amplitude_csl.py` → `.json`. Chỉ dùng GT, CSL dev, đúng 280 cặp ở trên.
   - Với mỗi bộ phận: f = 1 − (DTW mỗi bước)/(khoảng cách theo khung, sau co giãn *tuyến tính* về cùng độ dài, tức đã bỏ nhịp chung). Null: phần dư theo khung được xáo pha (cùng phổ, không có cấu trúc thời gian).

   | | Body | Tay trái | Tay phải |
   |---|---|---|---|
   | f (trung bình) | 0.158 | 0.334 | 0.349 |
   | f_null (trung bình; tỉ lệ DTW giữ đúng đường chéo) | 0.001 (91%) | 0.001 (96%) | 0.000 (100%) |
   | f − null, CI 95% | 0.157 [0.146, 0.168] | 0.333 [0.317, 0.350] | 0.349 [0.337, 0.360] |

   - **Nhịp điệu phi tuyến** (ngoài nhịp chung) chiếm khoảng **1/3 khác biệt giữa hai người ký ở tay** và khoảng 1/6 ở body. Null cho thấy DTW không "ăn" khác biệt thuần không gian có cùng phổ.
   - Hệ quả:
     - (i) Ở mức *dữ liệu*, thành phần nhịp điệu của T1 có thật và lớn nhất ở tay, đúng chỗ v2 kém người.
     - (ii) Câu chuyện phân tích S1 có kết quả đo đầu tiên, sạch và có null.
     - (iii) **Chưa** chứng minh mô hình đang lấy trung bình qua nhịp điệu: phương sai dự đoán của v2 có thể khác phương sai dữ liệu. Đó vẫn là việc của cổng P0.S.
   - Giới hạn:
     - đây là khác biệt *giữa người ký* (gồm cả phong cách nhịp riêng);
     - chỉ 2 lần ký mỗi câu;
     - null xáo pha không bắt được việc DTW khai thác các đoạn giữ tư thế giống nhau.
4. **Ưu tiên của hướng R vì vậy dồn vào tay** (§3.1–3.2). Các thành phần cho tay nên *nhận biết nhịp điệu*: bộ nhớ đoạn ký hiệu đã căn, và E9 nhận đặc trưng tốc độ của cánh tay.
   - δ với bộ nhớ *hình tay*;
   - trọng số theo cột cho tay;
   - E9 (head tay có điều kiện cánh tay), vốn bị hoãn trong `07` với điều kiện "chỉ khi tay vẫn là nút thắt". Nay điều kiện đó đã có bằng chứng.

### 2.5 Toy [Toy] (`scratchpad/toy_timing*.py`, `toy_review.py`; synthetic, numpy)

Thiết lập:

- 7 "nét ký" Gaussian;
- co giãn thời gian kiểu cầu Brown với hai đầu cố định;
- biến thiên không gian trơn.

"Tỉ lệ quãng đường" là độ dài quỹ đạo chia cho độ dài của mẫu giữ lại, bất biến với co giãn thời gian.

| Nhịp / không gian | Quãng đường: TB Euclid | Quãng đường: DBA | DTW: TB Euclid | DTW: DBA | DTW: một mẫu | DTW: một mẫu co 25% |
|---|---|---|---|---|---|---|
| 0.05 / 0 | 0.81 | 0.99 | 8.25 | **3.72** | 3.86 | — |
| 0.10 / 0 | 0.54 | 0.97 | 22.6 | 5.39 | **4.09** | — |
| 0.15 / 0 | 0.46 | 0.96 | 30.0 | 6.01 | **5.55** | — |
| 0.05 / 0.15 | 0.88 | 1.13 | 14.9 | **12.1** | 15.7 | 17.5 |
| 0.10 / 0.15 | 0.55 | 0.97 | 25.6 | **13.4** | 17.0 | 19.9 |
| 0.05 / 0.30 | 1.06 | 1.51 | **23.1** | 23.7 | 37.4 | 32.0 |
| 0.10 / 0.30 | 0.67 | 1.32 | 32.2 | **24.6** | 32.7 | 30.8 |

Kiểm tra bổ sung của phản biện:

- **Sàn rời rạc hoá:** DTW giữa hai warp của cùng một đường cong là 3.74 (nhịp 0.05) và 4.47 (nhịp 0.10).
- **Chỉ có 2 tham chiếu**, như CSL:
  - DBA − Euclid = −1.4 (sd 2.1), −5.6 (sd 4.8) và −1.4 (sd 3.6);
  - tỉ lệ quãng đường của DBA tới 1.20 khi nhiễu không gian lớn.

**Đọc toy (đã sửa).**

- (i) Vùng không gian chiếm ưu thế tái hiện dữ liệu thật: trung bình hoá và co về trung bình thắng một mẫu.
- (ii) Khi nhịp điệu đáng kể, trung bình Euclid tệ hơn *cả một mẫu đơn* trên DTW.
  - Barycenter thắng trung bình Euclid ở mọi hàng có nhịp điệu, **trừ** hàng 0.05/0.30, nơi không gian chiếm ưu thế (23.7 so với 23.1).
  - Barycenter thua một mẫu đơn ở các hàng chỉ có nhịp.
- (iii) DBA với chi phí bình phương thừa quãng đường khi nhiễu không gian lớn. Cần γ > 0, hoặc chi phí không bình phương / median.
- (iv) Với chỉ 2 tham chiếu, phương sai lớn. P0.2 cần khoảng 500 câu để sai số chuẩn ≈ 0.2.

---

## 3. Phương pháp

### 3.0 Cấu trúc chung

Cả hai hướng dùng chung khung của paper:

- (S³)^J, Exp/Log;
- dự đoán đích, sampler re-planned NFE 5;
- mBART + transformer, length head.

Khung này được bổ sung:

- **vệ sinh** (E3: EMA, text lr, warmup);
- **τ-loss** (mục 2.4), sửa thiên lệch của sampler. Rẻ, và áp dụng cho *cả* baseline khi so sánh.

Path: geodesic + time-shift nếu E1-gate (`07` P1) cho NG ≈ GE; giữ NG nếu NG > GE.

### 3.1 Hướng R. (δ) Bộ nhớ truy hồi: thành phần kết quả chính

**Bằng chứng** [V].

- SOKE: truy hồi từ điển từ giảm DTW-PA body/hand 20–24% trên cả ba dataset.
- SignSparK-SR (gloss dự đoán + keyframe từ điển): JPE giảm 20–32%.
- Cả hai dùng nguồn *ngoài train* (từ điển ký hiệu) hoặc gloss dự đoán. Truy hồi là kỹ thuật **đã biết**, không phải novelty.

**Ba bậc, từ rẻ đến đắt:**

| Bậc | Mô tả | Thiết lập |
|---|---|---|
| δ-0 (không huấn luyện) | kNN câu train: top-1; trung bình top-k; barycenter top-k. Khoá truy hồi = TF-IDF + số hạng độ dài (ReMoDiffuse: thiếu số hạng này thì truy hồi "unacceptable") | gloss-free; CPU |
| δ-sent | top-k câu (k = 3), encoder chuyển động nhỏ, cross-attention trong generator, retrieval dropout 20–30% | gloss-free |
| δ-gloss | text→gloss nhỏ, cộng từ điển *đoạn ký hiệu* cắt từ `pami0` (Phoenix). Đoạn được đưa vào như bộ nhớ hoặc như keyframe có mask (kiểu SignSparK) | gloss khi huấn luyện, chỉ văn bản khi test (như SRVQ); ghi rõ khác SignNMFlow |

**Kiểm soát rò rỉ** (bắt buộc).

- Loại khỏi kết quả truy hồi mọi mẫu có văn bản **gần trùng**: chuẩn hoá chữ thường, dấu câu, khoảng trắng, cộng ngưỡng cosine. CSL lặp mỗi câu 2–3 lần, trong khi 99.8% câu dev là mới.
- Chỉ truy hồi từ train.
- Mọi phép chia fold cho cross-fitting phải **nhóm theo văn bản**.

**Rủi ro đo lường.** DiMo [V]: R-precision thưởng cho sự giống chuyển động train, trong khi người chấm đảo thứ hạng. Vì vậy M1 của các arm truy hồi luôn được đọc kèm hoán đổi văn bản (M4) và đối chứng δ-0.

### 3.2 Hướng R. Vệ sinh, khớp loss và tập trung vào tay

- **E3** (từ `07`).
- **E4a** (FK loss trên x̂1, đổi nghiệm sang trung bình vị trí). Là thành phần thực dụng, không có tuyên bố lý thuyết.
- **τ-loss** (mục 2.4).
- Có thể thêm γ, chỉ như ablation phân bổ năng lực.

**Tập trung vào tay** (theo P0.S-H: body ở mức người–người, tay kém 14–18%):

- **E4c (trọng số lever-arm theo cột)**, đã code trong `signmflow_plus`;
- **E9: head tay có điều kiện cánh tay**, với decoder riêng cho 2 × 15 khớp ngón, nhận body đã dự đoán. Cơ sở: Body2Hands, ParCo/EMAGE/SOKE tách bộ phận;
- **bộ nhớ hình tay trong δ:** mỗi đoạn ký hiệu lưu riêng chuỗi hình tay (khớp ngón, căn cổ tay), và cross-attention riêng cho head tay.

Endpoint quyết định không đổi, nhưng mọi tuyên bố "cải thiện thật" phải thể hiện ở **cột tay**. Cải thiện chỉ ở body bị coi là hành vi ước lượng, trừ khi đường biên ghép quãng đường cho thấy điều ngược lại.

### 3.3 Hướng S. (α) Template barycentric trong không gian nhiệm vụ

**Loss của anchor/template** (chế độ riêng, không có x_t):

```
L_α = Σ_{p ∈ {body, lhand, rhand}} D_γ^{band,warp}( P_p(a), P_p(x1) ) + λ_rot · ℓ_geo(a, x1)
```

- P_p: FK vi phân được tới đúng điểm của evaluator (`fk.py::SparseSMPLX`, khớp tới 0.0004 mm).
- Tính DTW riêng cho body và từng tay, như evaluator.
- **Chi phí cục bộ khớp evaluator:** L2 *không bình phương* theo khớp. Nghiệm ứng với median hình học (T2), không phải trung bình.
- **Divergence:** D = sdtw(a, b) − ½ sdtw(a, a) − ½ sdtw(b, b).
  - Không âm chỉ được chứng minh cho chi phí đặc biệt (Blondel 2021). Bản TPAMI của DILATE thấy chuẩn hoá không giúp gì mà nặng hơn.
  - Vì vậy so cả sdtw thuần trong smoke test.
- **Band 0.15–0.25 T cộng phạt bước lệch đường chéo** (Parallel Tacotron 2: band 60, phạt 128, γ = 0.05).
  - Chống singularities, tức "đứng yên thay vì ký" (Keogh & Pazzani).
  - Band chỉ giới hạn, không ngăn được singularity.
  - Tuỳ chọn thêm số hạng méo thời gian của DILATE.
- **γ:** quét {0.1, 0.3, 1}× chi phí mỗi bước. γ nhỏ cho barycenter "spurious" (Cuturi & Blondel).
- **ℓ_geo** (λ_rot 0.05–0.1): giữ các bậc tự do vị trí không quan sát được.

**Đối chứng bắt buộc** (cùng kiến trúc, cùng ngân sách tinh chỉnh; mỗi arm có cùng số cấu hình quét):

| Arm | Khác A-SDTW ở điểm nào |
|---|---|
| A-L2 | L2 bình phương theo khung trong không gian nhiệm vụ |
| A-L1 | L2 *không bình phương* theo khung. Tách "median so với mean" khỏi "bất biến thời gian" |
| A-L2+kin | A-L2 + loss khớp vận tốc / quãng đường. Hỏi: có cần căn chỉnh không, hay chỉ cần phạt mất biên độ? |
| A-SDTW | Arm điều trị |

Thêm hai đối chứng không huấn luyện:

- barycenter của K mẫu v2 (P0.S);
- NFE 1 + unsharp masking theo thời gian (P0.7).

**Dự báo có thể bác bỏ** (tại cùng tỉ lệ quãng đường):

- A-SDTW < A-L1 ≤ A-L2 trên DTW.
- Lợi ích lớn hơn khi f_τ lớn hơn: CSL > Phoenix, nếu P0.2 cho f_τ(CSL) > f_τ(Phoenix).

### 3.4 Hướng S. EPA-Flow: flow phân rã pha–biên độ trên manifold (thay cho β của v1)

**Vì sao không giữ β của v1.**

- β v1 điều kiện hoá trên ã = a∘w*, với w* là đường DTW tính từ x1 khi huấn luyện, và dùng ã = a (warp đồng nhất) khi suy diễn.
- Phản biện chỉ ra hai vấn đề:
  - w* là thống kê dung lượng cao của x1. ã có răng cưa kiểu bậc thang, không bao giờ xuất hiện khi test.
  - Warp đồng nhất hầu như không nằm trong tập điển hình của p(w* | c).
- Đây là lệch giá đỡ (support mismatch), không chỉ là sai số ước lượng.
- TTS (FastSpeech 2, Glow-TTS, Grad-TTS) đều *dự đoán* thời lượng khi test, bằng một mô hình học từ alignment đã trích xuất.

**Thiết kế (huấn luyện hai giai đoạn).**

1. **Template.** Huấn luyện α xong rồi **đóng băng**. Không đồng huấn luyện bằng token chế độ, vì điều kiện sẽ trôi.
2. **Đăng ký (registration)** mỗi x1 vào thời gian của template:
   - đường DTW (band, phạt lệch) → **làm trơn thành warp đơn điệu**: spline đơn điệu, hoặc log-tốc độ trên K nút, để bỏ răng cưa;
   - x̄1 = x1∘w*⁻¹, nội suy SLERP trên S³. Đây là "biên độ" trong thời gian đã đăng ký.
   - **Lưới đăng ký phải lấy mẫu dày (≥ 2T).** Trên lưới T khung, các đoạn nhanh hơn template bị lấy mẫu thưa, và mất thông tin đúng ở phần có nhiều biên độ. Báo lỗi tái tạo x1 → (x̄1, w*) → x1 *theo dải tần*.
   - **Tuỳ chọn:** dùng hình học elastic (TSRVF trên manifold; Su và cộng sự 2014) thay DTW. Nó là metric thật, không có "pinching".
3. **Mô hình pha** p(w | c, template): flow 1-D, hoặc hồi quy + nhiễu, trên hồ sơ log-tốc độ (K nút, ràng buộc tổng). Học từ các w* đã trích xuất.
   - **Độ dài.** Mặc định, T là *ngoại sinh* như paper: GT khi huấn luyện, length head khi test; template thấy T. Mô hình pha chỉ phân bổ nhịp *trong* T.
   - Biến thể "pha sở hữu độ dài": giống FastSpeech, thay length head. Template được sinh ở độ dài chuẩn L_c, và mô hình pha quyết định T.
   - **"w trung tâm"** (cho chế độ điểm) = nghiệm ODE của flow pha từ trung bình của nguồn. "Mode" không được định nghĩa cho flow.
4. **Mô hình biên độ** p(x̄1 | c, template, w):
   - RFM trên (S³)^J, dự đoán đích với τ-loss;
   - nguồn wrapped normal quanh template, σ0 hiệu chuẩn theo phân bố d(template, x̄1);
   - template và đặc trưng tốc độ cục bộ của w làm điều kiện, để mô hình hoá sự phụ thuộc giữa pha và biên độ ("ký nhanh thì biên độ nhỏ").
5. **Lấy mẫu:** w ~ p(w | c); x̄ ~ flow; x = x̄∘w. Chế độ "điểm" dùng "w trung tâm".

**Tính hợp lệ** [Toán] (đã sửa theo `09` vòng 2).

- R: x1 ↦ (x̄1, w*) *không* phải phép đổi biến. Nó ánh xạ T×D giá trị sang T_reg×D + K giá trị, nên không có Jacobian và "p(x1 | c) = p(w | c) · p(x̄1 | c, w)" không phải đồng nhất thức mật độ.
- Lập luận đúng là **đẩy tiến (push-forward)**. Gọi C(x̄, w) = x̄∘w. Nếu C(R(x1)) = x1 *chính xác* và cả hai điều kiện p(w* | c), p(x̄1 | c, w*) được học chính xác, thì lấy mẫu tổ tiên (ancestral) rồi ghép sẽ tái tạo đúng p(x1 | c).
- Điều kiện cần là **tái tạo chính xác**: lưới đăng ký dày, và đo lỗi tái tạo theo dải tần (R6).
- Nguồn của flow chỉ phụ thuộc (c, template, w), tức là điều kiện, nên data-dependent coupling (Albergo 2024) hợp lệ.
- **Cross-fitting** (nếu cần) phải áp cho *cả* template lẫn w* dùng để huấn luyện mô hình pha.

**Vì sao có thể tốt hơn.**

- Trong thời gian đã đăng ký, phương sai nhịp điệu đã bị lấy ra. Vì vậy:
  - các bước đầu của sampler không còn khoá một cấu trúc nhoè;
  - nguồn quanh template ngắn thật (Grad-TTS: prior N(μ, I) thắng N(0, I) ở 10 bước; Issachar 2025: prior có điều kiện thắng ở 15 bước).
- **Giới hạn:** nếu không qua cổng S (P0.S), lợi ích này nhỏ, và EPA-Flow không đáng làm.

**σ0.**

- Hiệu chuẩn theo trung vị–p75 của d(template, x̄1) trên *nửa val dùng để chọn*.
- Prior quá hẹp hại nặng: Issachar, σ 0.2 cho FID 23.6 so với 7.6. ODE thuần từ nguồn quá tất định cho kết quả nhoè (DDBM).

**Lệch train/test của template.**

- Template trên chính mẫu train đã fit gần x1 hơn trên val. Đo tỉ lệ sai số template train/val.
- Nếu chênh > 20%: cross-fitting 2 fold, **nhóm theo văn bản**. Chỉ làm được khi template đã đóng băng.

**Ablation:**

| Arm | Mục đích |
|---|---|
| EPA (w lấy mẫu) | Mô hình đầy đủ |
| EPA (w trung tâm) | Chế độ điểm |
| EPA (w* oracle khi test) | Cận trên của phần pha; chỉ chẩn đoán |
| EPA (w = đồng nhất khi test) | Đo đúng vấn đề lệch giá đỡ của β v1; chỉ suy diễn |
| p(w) không điều kiện | Mô hình pha có học được gì từ văn bản không |
| Chỉ pha: x = template∘w | Biên độ có cần flow không |
| Không điều kiện tốc độ | Đặc trưng tốc độ cục bộ có đủ (và cần) để mô hình hoá phụ thuộc pha–biên độ không |
| K nút 16 / 32 / 64 | Độ phân giải của pha |
| No-register | Template làm nguồn/điều kiện, không đăng ký (≈ β-noalign của v1) |
| E6 **trong thời gian đã đăng ký** | Nguồn wrapped normal tại tư thế trung bình toàn cục, cùng σ0. Kiểm soát hiệu ứng *lịch nhiễu*, vì `01` A.3 cho thấy đó là thứ NG thực sự đổi. Phải chạy trong thời gian đã đăng ký, nếu không thì lẫn với việc đăng ký |
| Template A-L2 thay A-SDTW | Template barycentric có cần thiết không |
| EPA Euclid (6D, không manifold) | Manifold có cần không |

Mọi arm đều báo **độ đa dạng mẫu** (DTW giữa các mẫu). DTW giảm do bớt đa dạng không được tính là cải thiện.

### 3.5 (ε) Giải mã: MBR, so tại cùng quãng đường

- Medoid DTW chọn mẫu *trung tâm nhất*, thường là mẫu *ít biên độ nhất*. Đó là hồi quy về trung bình trá hình, nên phải so tại cùng tỉ lệ quãng đường.
- **Thổi phồng tiện ích** (Freitag 2022: BLEURT 26.8 → 50.0, BLEU giảm). Tiện ích dùng soft-DTW với γ lớn. Bắt buộc báo M1, back-translation (có trần GT) và đường biên.
- Áp dụng cho cả baseline. ε là thành phần **phụ**.

### 3.6 Giữ gì, bỏ gì của SignNMFlow

| Thành phần | Quyết định | Lý do |
|---|---|---|
| (S³)^J, Exp/Log, dự đoán đích, sampler re-planned | **Giữ**, thêm τ-loss | Sampler hiện tại thiên lệch (mục 2.4) |
| mBART + transformer, length head | Giữ, thêm E3 | — |
| K-means + correction ε_θ | Bỏ nếu E1-gate cho NG ≈ GE | `01` A.3 |
| Nguồn đều trên S³ | Thay bằng template/prior có điều kiện (S) | 126.5° so với 21° [Đo] |
| L2 quaternion đồng đều | τ-loss (+ E4a cho R; α cho S) | Mục 2.4 |
| Một đầu ra NFE 5 | Tách "điểm" và "mẫu"; báo cả hai | Mục 2.4 |

---

## 4. Định vị và novelty (đã hạ nhãn sau phản biện)

### 4.1 SOTA trên giao thức SOKE: phần so sánh được (test; DTW-JPE body/hand · DTW-PA body/hand)

| Phương pháp | Phoenix | CSL-Daily | How2Sign | Ghi chú |
|---|---|---|---|---|
| SOKE (ICCV'25) [V] | 6.04/7.72 · 4.77/1.38 | 7.38/9.68 · 6.24/1.71 | 7.75/10.08 · 6.82/2.35 | Truy hồi từ điển ngoài |
| MaDiS (2601.19577) [V] | 4.97/6.59 · – | 5.99/7.96 · – | **6.59/9.45** · – | Cùng nhóm SOKE |
| SignNMFlow (paper) [V] | **4.37/5.57 · 3.85/1.07** | **5.73/7.43 · 5.10/1.30** | 6.76/9.72 · 6.01/2.07 | Không truy hồi; không nói cách chọn độ dài |
| Bản tái lập v2 (test, seed 0) [Đo] | 4.93/6.54 · 4.35/1.30 | 5.93/7.69 · 5.26/1.38 | 7.45/10.67 · 6.63/2.35 | Length head |
| GARD [V] | 6.69/6.90 · 5.03/1.21 | 7.73/9.32 · 6.98/1.56 | – | Đầu vào gloss |
| SRVQ (WACV'27) [V] | 4.70/6.41 · 5.32/1.26 | 6.30/8.38 · 7.11/1.51 | – | Gloss khi huấn luyện |

- **Không so được trực tiếp** (pseudo-GT riêng): SignSparK (SR-T2P 3.51/4.82 Phoenix; 4.89/6.56 CSL), M3T, BLANC, MV-Sign.
- **Về Phoenix:** MaDiS gần bản tái lập trên Phoenix nhưng thắng SignNMFlow trên How2Sign. Điều này chỉ *tương thích* với giả thuyết khác giao thức, không chứng minh gì. Việc phương pháp khác trùng với bản tái lập không nói được giao thức của SignNMFlow là gì (phản biện, `09`).
- Không có bài nào sau 09/2026 so với SignNMFlow.

### 4.2 Từng thành phần

| Thành phần | Công trình gần nhất | Novelty (đã hạ) |
|---|---|---|
| Phân rã pha–biên độ, "trung bình cấu trúc so với trung bình cắt ngang" | **FDA:** Kneip & Gasser 1992; Wang & Gasser 1997 (căn đường cong bằng DTW); Marron và cộng sự 2015; Srivastava/TSRVF (Su và cộng sự 2014). **Hoà trộn chuyển động cần đăng ký:** Bruderlin & Williams 1995; Kovar & Gleicher 2003. **Time-Agnostic Prediction** (Jayaraman và cộng sự, ICLR 2019): bất định thời gian gây nhoè, sửa bằng loss min theo thời gian. Ren và cộng sự (ACL'22, TTS) [V]. | **Thấp–trung bình** về lý thuyết (sách giáo khoa). **Trung bình–cao** ở *ứng dụng và đo lường trong SLP*: đo f_τ cho phương sai giữa người ký và phương sai của mô hình; liên hệ với thiên lệch của DTW-MJE; đường đối chứng con người. Không tìm thấy nghiên cứu SLP nào làm việc này. |
| α (template soft-DTW) | soft-DTW, DILATE [V]; HybridSign (ACL'26, SLP 2-D, soft-DTW, không ablation) [V]; DeepNAG [V] | **Thấp.** Divergence theo bộ phận trong không gian nhiệm vụ là kỹ thuật. |
| EPA-Flow | **TTS:** Grad-TTS (MAS khi huấn luyện + dự đoán thời lượng, prior N(μ, I)) [V]; Glow-TTS (MAS); Matcha-TTS [V]. **SLP:** NAT-EA (Huang và cộng sự 2021; dự đoán thời lượng + aligner ngoài); FS-Net trong Signing at Scale (Saunders và cộng sự, CVPR'22). **Pha–biên độ sinh:** Elastic Shape VAE (Rahman, …, Srivastava 2026, *preprint*) [A]: VAE *không điều kiện*, TSRVF trên đa tạp hình dạng Kendall. Motion infilling từ keyframe lệch thời gian (Goel và cộng sự, EG'25) [A]: xuất hàm warp + phần dư không gian. | **Trung bình.** Không tìm thấy flow có điều kiện văn bản, phân rã pha–biên độ liên tục (không gloss, không thời lượng rời rạc) trên (S³)^J cho chuyển động hay SLP. Nhưng từng mảnh đều đã có. |
| τ-loss (sampler không thiên lệch trên manifold) | **Chính là loss SO(3) của FrameFlow** (Yim và cộng sự 2023, arXiv 2310.05297): loss tiếp tuyến tại r_t, trọng số 1/(1−t)², cắt tại 0.9. Xem thêm RFM (Chen & Lipman) [V]. | **Không mới.** Giá trị là tính đúng đắn. Nó cũng chỉ ra rằng SignNMFlow chọn loss x̂1 dây cung *khác* với FrameFlow, bài mà chính sampler của họ dựa vào. |
| δ | SOKE, SignSparK-SR, Spoken2Sign, SemRAG, SignRR [V]; ReMoDiffuse [V] | **Không mới.** Giá trị kết quả cao nhất. |
| ε | Freitag 2022 [V]; DiffuSeq [V] | Thấp; dễ thổi phồng. |
| Đánh giá | SLRTP 2025 (total distance) [V]; Ham2Pose (bệnh lý của DTW-MJE); Hong & Košecká 2026 [V] | Thấp–trung bình. Phần mới là ghép tại quãng đường và đường đối chứng con người. |

**Bị loại:** RL/GRPO với phần thưởng DTW. Phần thưởng tới một tham chiếu bị cực đại bởi một đầu ra trung vị, nên sụp đa dạng (DRaFT, DDPO, Pref-GRPO [V]).

### 4.3 Hai câu chuyện paper khả thi

- **S1 (phân tích; khả thi nhất về novelty).**
  - Câu hỏi: *"Where does SLP variance live, and what does DTW-MJE reward?"*
  - Nội dung:
    - phân rã pha–biên độ (có null) cho phương sai giữa người ký và phương sai dự đoán của các mô hình (v2, NFE, δ);
    - đánh giá tại biên độ khớp;
    - đường đối chứng con người;
    - một nghiên cứu người dùng nhỏ;
    - δ và α làm minh hoạ.
  - Phản biện đánh giá hướng này có thể hợp CVPR/ICCV nếu phân rã sạch.
- **S2 (phương pháp).** EPA-Flow + δ. Chỉ theo đuổi nếu qua cổng S (P0.S) và G2 dương.
- **R** (kết quả) đứng riêng. Một hệ thống tốt nhất (δ + vệ sinh + τ-loss) có tuyên bố trung thực trên CSL.

---

## 5. Giao thức đánh giá

1. **Hai chế độ đầu ra**: "điểm" (template / mode / MBR) và "mẫu". Báo tách biệt; áp cùng quy tắc giải mã cho baseline.
2. **Rào chắn biên độ bất biến thời gian (C11).**
   - `path_length_ratio_{body,hand}`: quãng đường sau thông thấp σ = 2, chia cho của GT. Đây là "total distance" của SLRTP.
   - Thêm vào `src/slp/evaluation/metrics.py` kèm test.
   - Tỉ lệ tốc độ (M2) có thể bị thổi phồng giả khi dồn thời gian. Tỉ lệ quãng đường thì không.
3. **Đường biên:** so tại *cùng tỉ lệ quãng đường* với phép co về trung bình của baseline.
4. **Độ dài:** báo cả độ dài dự đoán (chính) và độ dài GT (oracle; để so với paper).
5. **Ngữ nghĩa và hiện thực:**
   - M1 (R-precision, Fréchet) kèm M4 (hoán đổi văn bản);
   - M3;
   - gia tốc/jerk;
   - **độ đa dạng mẫu**;
   - **tính tự nhiên của nhịp điệu**: phân bố thời lượng ký hiệu (Phoenix, qua `pami0` + căn DTW) và tỉ lệ khung "giữ". Nhịp điệu mang nghĩa trong ngôn ngữ ký hiệu (ngữ điệu, thể), nên không chỉ là nhiễu.
6. **Đường đối chứng con người:** DTW giữa hai người ký cùng câu, trên 280 câu dev CSL (4 cột).
7. **Thống kê.**
   - **Bootstrap phân tầng** (seed × chuỗi).
   - **Mỗi hướng một endpoint chính, đăng ký trước:**
     - **R (quyết định, trên val):** hiệu ứng *ghép đường biên* (`frontier_effect`) trên trung bình tương đối 4 cột DTW so với bản tái lập, với độ dài dự đoán; khoảng tin cậy 95% < 0.
       - Rào chắn ngữ nghĩa là M4 (hoán đổi văn bản) và back-translation kèm trần GT, *không* dùng R-precision, vì δ có thể thổi phồng R-precision (DiMo).
       - Bỏ dung sai "quãng đường ≥ bản tái lập − 0.02": dung sai đó tự mua được khoảng 1% DTW chỉ bằng co về trung bình (`06`: co 5% giảm tốc độ 0.03–0.04 và DTW 1.2–1.4%).
     - **R (tuyên bố so với paper, thứ cấp, trên test ở cuối):** dưới số paper ở ≥ 3/4 cột, trung bình 3 seed, độ dài dự đoán, *và* endpoint quyết định đã qua.
     - **S:** Gain − Gain_null của P0.S, và hiệu A-SDTW − A-L1 tại cùng quãng đường.
   - **Chia val thành hai nửa A/B, nhóm theo văn bản, rồi hoán đổi.**
     - Chọn σ0, γ và cổng trên A, báo cáo trên B; rồi làm ngược lại; gộp hai nửa báo cáo.
     - Cách này không rò rỉ mà vẫn dùng đủ val. Một lần chia đơn thì khoảng tin cậy rộng hơn khoảng 1.4× (Phoenix chỉ còn khoảng 260 chuỗi).
   - **SD seed theo từng dataset.** B×3 của `07` chỉ cho Phoenix. CSL cần seed riêng: R-base × 3 ở G4, hoặc sớm hơn nếu hiệu ứng CSL sát ngưỡng.
   - Hiệu ứng < 2 × SD seed thì cần ≥ 3 seed.
   - Test một lần duy nhất ở cuối, khi người dùng cho phép.
8. Luôn báo kèm lỗi theo khung (không DTW) để minh bạch.

---

## 6. Kế hoạch có cổng

Thời gian mỗi run đo từ log v2 (GPU dùng chung, 1 tiến trình):

| Dataset | Correction | Generator 300 epoch | Arm không correction |
|---|---|---|---|
| Phoenix | 2.5 h | 16.6 h | ~17 h |
| CSL | 6.9 h | 31.2 h | ~32 h |
| How2Sign | 10.5 h | ~60 h (~0.2 h/epoch) | ~60 h |

soft-DTW ước lượng thêm 20–40% [Suy luận]; đo ở smoke test. Độ dài sàng lọc ≥ 250 epoch (`06`). VRAM huấn luyện ước lượng 10–17 GB (chưa đo).

### G0 — Không huấn luyện (người dùng chạy)

| ID | Câu hỏi | Cách đo | Chi phí | Quyết định |
|---|---|---|---|---|
| P0.1 | Độ dài oracle giải thích bao nhiêu? | v2 val, length head so với độ dài GT, **cả 3 dataset** | ~2 GPU-h | Lấp ≥ 50% khoảng cách Phoenix mà ít tác dụng trên CSL → không tuyên bố gì về Phoenix so với paper |
| P0.H | Vì sao bản tái lập How2Sign kém MaDiS? | Rà soát bàn giấy so với giao thức SOKE/MaDiS: fps 24, quy tắc cắt 40–400, danh sách lọc, số chuỗi, cách chọn độ dài, ngân sách huấn luyện. Dùng P0.1 cho How2Sign. | ~1 GPU-h + đọc mã | Có nguyên nhân giao thức → sửa trước khi đặt mục tiêu How2Sign |
| **P0.S** | Phương sai dự đoán của v2 có phần nhịp điệu *có ích* không? | Phoenix và CSL val, K = 8 mẫu/câu (seed sampler khác nhau). Ước lượng gồm: **median hình học theo khung** (không căn) và **median hình học sau căn** (soft-DTW γ > 0, chi phí không bình phương), cộng trung bình Euclid, NFE 1, một mẫu, medoid. Đo 4 cột DTW và quãng đường. **Null trên chính lợi ích:** cùng phép so sánh trên các mẫu nhiễu loạn *chỉ không gian*, phổ khớp. Thêm (iv) kiểm tra nhất quán T1: độ phân tán warp giữa các mẫu (đo trực tiếp) → suy giảm dự đoán của trung bình theo khung → so với suy giảm đo được. | ~2.5 + 5 GPU-h | **Cổng S (theo từng dataset, trên nửa val "chọn").** Gain = [DTW(median khung) − DTW(median căn)] / DTW(median khung). **Kiểm tra trội trước:** nếu median căn có DTW thấp hơn *và* quãng đường ≥, thì lấy Gain thô, vì không đổi biên độ lấy DTW. Ngược lại thì so tại cùng quãng đường, co ước lượng có biên độ cao hơn về tư thế trung bình (quãng đường co đúng theo tỉ lệ 1 − a). Qua cổng khi Gain − Gain_null ≥ 2%, khoảng tin cậy 95% loại 0, và (iv) nhất quán. **Hiệu chuẩn trên toy** (8 mẫu): chỉ nhịp → Gain − null = 0.49; chỉ không gian → −0.002 (không qua); hỗn hợp → 0.11. **Script đã viết sẵn, chưa chạy:** `probes/probe_gate_ps.py` (`--runs` một run cho K seed sampler, hoặc danh sách run cho ensemble seed). Ngưỡng cũ "f_τ − null ≥ 0.1" bị bỏ: trên toy nó luôn qua khi có *bất kỳ* nhịp điệu nào và không dự báo được lợi ích (`09` vòng 2). **Phương án dự phòng** khi mẫu v2 ít đa dạng: dùng **ensemble B×3 seed** (phương sai giữa seed trên câu *mới* đo trực tiếp phương sai nội dung/nhận thức), không dùng P0.2. |
| P0.S-H | Đường đối chứng con người | CSL dev, 280 câu ≥ 2 người ký: DTW 4 cột giữa hai lần ký | CPU | **Đã chạy** (§2.4): body v2 ≈ người; tay v2 kém người 14–18%. → hướng R dồn vào tay. Phoenix không có cặp người ký trên dev. Có thể làm bản tương tự trên How2Sign nếu tìm được câu lặp. |
| P0.2 | Phân rã giữa người ký (train) | **Bản rút gọn đã chạy trên 280 cặp dev (§2.4): f − null = 0.16 body, 0.33–0.35 tay.** Bản đầy đủ: CSL train, khoảng 500 câu × 3 lần ký. Phân tầng theo người ký; kiểm tra tay thuận (lật trái/phải nếu cần); **một độ dài chung** cho mọi ứng viên; báo thêm `dtw_jpe_norm`; null như P0.S; barycenter soft-DTW γ > 0. | CPU, 1–2 h | f_τ của dữ liệu. So với f_τ của mô hình (P0.S) để biết mô hình đang lấy trung bình *cái gì*. |
| P0.4 | MBR có ích không? | Từ K = 8 mẫu của P0.S: medoid so với một mẫu, **tại cùng quãng đường** | 0 (dùng lại) | Lợi ích < ngưỡng nhiễu, hoặc chỉ có khi quãng đường giảm → bỏ ε |
| P0.6 | Truy hồi không huấn luyện (δ-0) mạnh tới đâu? | kNN TF-IDF + độ dài: top-1, trung bình top-k, barycenter top-k; trên Phoenix, CSL, How2Sign val. Thêm chế độ vào method `text_retrieval` (chạy qua `scripts/run.py`). Chỉ mục truy hồi mà cả `text_retrieval` lẫn `signtaf` cần thì chuyển vào `src/slp/retrieval.py` kèm test, theo quy tắc của repo; không import chéo method. | CPU | Định cỡ tiềm năng của δ. Nếu top-k barycenter đã gần v2 → δ-sent rất hứa hẹn. |
| P0.7 | Khử nhoè không dùng văn bản | NFE 1 + unsharp masking theo thời gian, tại cùng quãng đường | CPU | Nếu khử nhoè đơn thuần đã lấy lại được phần lớn → α/EPA phải vượt mức này |

### G1 — Hướng R (luôn chạy) và hướng S (chỉ khi qua cổng S)

| Arm | Hướng | Dataset | GPU-h |
|---|---|---|---|
| R-base: v2 + E3 + τ-loss | R | Phoenix | ~17 |
| R-hand: R-base + E4c + E9 (head tay có điều kiện cánh tay) | R | Phoenix | ~20 |
| R-δsent: R-base + δ-sent (bộ nhớ hình tay riêng) | R | Phoenix, CSL | ~20 + ~37 |
| A-L2, A-L1, A-L2+kin, A-SDTW | S | Phoenix trước | 4 × ~17–22 |
| (cùng bộ trên CSL nếu Phoenix có tín hiệu) | S | CSL | 4 × ~32–40 |

### G2

| Arm | Hướng | GPU-h |
|---|---|---|
| R-δsent + E4a; R-δgloss (Phoenix) | R | ~20 + ~22 |
| EPA (w lấy mẫu / mode / oracle), no-register, E6 cùng σ0, template A-L2 | S | ~5 × 20 (Phoenix) |

### G3

- Cấu hình tốt nhất của R kết hợp thành phần S đã qua cổng.
- ε (suy diễn).
- Chạy CSL cho cấu hình thắng.

### G4 — Xác nhận

- 3 seed × (cấu hình tốt nhất + baseline v2 cùng vệ sinh và cùng quy tắc giải mã) trên Phoenix và CSL: khoảng 6 × 20 + 6 × 37 ≈ 340 GPU-h.
- How2Sign (6 × ~62 ≈ 370 GPU-h) **chỉ khi P0.H giải thích được khoảng cách**.
- Test một lần duy nhất, khi người dùng cho phép.

### Tổng chi phí

| Giai đoạn | GPU-h |
|---|---|
| G0 | ~10 (+ CPU) |
| `07` P0–P1 (giữ: B×3 cho ngưỡng nhiễu, E1-gate cho path) | ~90 |
| G1 (R) | ~95 |
| G1 (S, nếu qua cổng) | ~80 (Phoenix) + ~145 (CSL) |
| G2 | ~40 (R) + ~100 (S) + ~120 (ablation EPA bổ sung + cross-fitting) |
| G3 | ~60 |
| G4 | ~340 (+ ~370 How2Sign) |

| Kịch bản | GPU-h | Thời gian (1 GPU dùng chung) |
|---|---|---|
| S không qua cổng, không How2Sign | ~635 | ~4 tuần |
| S qua cổng, không How2Sign | ~1,080 | ~6–7 tuần |
| S qua cổng, có How2Sign | ~1,450 | ~9 tuần |

**Cổng S cắt khoảng 445 GPU-h** nếu không qua.

Các arm của `07` được hấp thụ hoặc hoãn:

- E3, E4a → R;
- E6 → đối chứng trong EPA;
- E2-F, E1-H1, E1lin, E8, E5, E7 → hoãn.

---

## 7. Rủi ro và tiêu chí bác bỏ (đăng ký trước)

| # | Rủi ro | Dấu hiệu | Phương án |
|---|---|---|---|
| R1 | Phần nhịp điệu không có ích | P0.S: Gain − Gain_null < 2% hoặc CI chứa 0 | S chuyển sang S1 (phân tích; vẫn là kết quả khoa học). Kết quả do R gánh. |
| R2 | soft-DTW bất ổn, singularities | Đường warp có đoạn dốc 0 hoặc ∞ dài; gia tốc nổ | Band + phạt lệch + γ lớn hơn + ℓ_geo; lịch siêu tham số (ISMIR'23) |
| R3 | Bị gọi là hồi quy về trung bình | — | Ghép tại quãng đường; báo độ đa dạng và chế độ "mẫu"; đường đối chứng con người |
| R4 | Số Phoenix/How2Sign của paper khác giao thức | P0.1, P0.H | Chỉ tuyên bố thắng paper trên dataset đã chẩn đoán; còn lại so với bản tái lập |
| R5 | δ thổi phồng M1 (DiMo) | M1 tăng mà M4 hoán đổi không đổi | Báo M4; đánh giá người nhỏ |
| R6 | Lỗi đăng ký/nghịch đảo warp của EPA | Lỗi tái tạo x1 → (x̄, w) → x1 > ngưỡng nhiễu sampler | Warp trơn hơn; TSRVF; nếu vẫn lớn thì bỏ EPA |
| R7 | Lệch train/test của template | Tỉ lệ sai số template train/val > 1.2 | Cross-fitting 2 fold nhóm theo văn bản |
| R8 | Chi phí | ~635–1,450 GPU-h tuỳ kịch bản | Cổng S; How2Sign có điều kiện; bản rút gọn 1 seed cho How2Sign |
| R9 | Rò rỉ | Truy hồi trùng văn bản; nguồn gốc `pami0` | Loại gần trùng; ghi nguồn gốc `pami0` |

---

## 8. Việc cần code (khi người dùng yêu cầu; chưa làm)

Tạo `python scripts/new_method.py signtaf` và chép module từ `methods/signmflow_plus`. Không import chéo giữa các method. Các probe G0 đặt trong `methods/signtaf/analysis/`; chỉ val, P0.2 dùng train.

| ID | Việc | Test |
|---|---|---|
| N0 | τ-loss: đầu ra tiếp tuyến, sampler theo vận tốc biên | Tắt cờ thì trùng bit với v2; toy: nghiệm = E[Log] |
| N1 | soft-DTW / divergence (torch, theo đường chéo, band, phạt lệch, chi phí không bình phương) | γ → 0 khớp `dtw_cost`; D(a, a) = 0; gradcheck |
| N2 | Chế độ template (α) + các loss đối chứng A-L2 / A-L1 / A-L2+kin | Khớp tập điểm evaluator |
| N3 | Đăng ký: DTW → warp đơn điệu trơn (spline/log-tốc độ), nghịch đảo, SLERP; (tuỳ chọn) TSRVF | Lỗi tái tạo; warp đồng nhất khi x1 = template |
| N4 | Mô hình pha (flow 1-D trên log-tốc độ) + flow biên độ có điều kiện (EPA) | Lấy mẫu w đơn điệu, tổng đúng T |
| N5 | δ-0 (mở rộng `text_retrieval`), δ-sent (cross-attention, loại gần trùng), δ-gloss (bộ cắt `pami0` 25 → 24 fps, text→gloss) | Không mẫu train nào truy hồi văn bản gần trùng của chính nó |
| N6 | MBR / medoid tại quãng đường khớp | Medoid là một trong K mẫu |
| N7 | Metric `path_length_ratio_{body,hand}`; ghép đường biên theo quãng đường; bootstrap phân tầng | Bất biến khi co giãn thời gian |
| N8 | Probe G0: P0.S (+ null), P0.S-H, P0.2, P0.6, P0.7 | — |
| N9 | Registry arm và lệnh | — |

---

## 9. Kết quả kỳ vọng (kịch bản; [Suy luận])

| Thành phần | Kỳ vọng trên DTW (so bản tái lập, tại quãng đường khớp) | Ghi chú |
|---|---|---|
| Thành phần cho tay (E4c, E9, bộ nhớ hình tay) | Khoảng trống *thật* đo được: PA-hand khoảng 15%, JPE hand khoảng 12% tới mức người–người (P0.S-H) | Mục tiêu chính của hướng R |
| δ (truy hồi) | Lớn nhất, nhất là PA-hand. Tài liệu: −20–32% *với từ điển ngoài*; với chỉ train thì nhỏ hơn, chưa biết | Khoảng cách 5.8% PA-hand của CSL là vấn đề hình tay, đúng chỗ δ nhắm tới |
| Vệ sinh + E4a | Nhỏ đến vừa (1–3%) | Theo tài liệu về FK loss và EMA |
| τ-loss | **Chưa biết.** Có thể ≈ 0 ở NFE 5 | Thiên lệch là hiệu ứng độ cong bậc hai; giá trị chính là tính đúng đắn |
| α | ≤ khoảng 2% vượt đường co, *trừ khi* P0.S cho Gain − Gain_null lớn | Ước lượng của phản biện |
| EPA-Flow | Không kỳ vọng thay đổi bảng DTW (chế độ "điểm" ≈ α). Lợi ích ở mẫu: hiện thực, đa dạng, nhịp điệu tự nhiên | Có giá trị cho S2 |
| ε | Dễ thổi phồng; phụ | — |

**Mục tiêu thực tế:**

- **CSL:** vượt paper trên ≥ 3/4 cột (3 seed, độ dài dự đoán) bằng R; khả thi nếu δ-sent đạt ≥ 4–6%.
- **How2Sign:** chỉ đặt mục tiêu sau P0.H.
- **Phoenix:** so với bản tái lập và MaDiS, trừ khi P0.1 giải thích được số paper.
- **Khoa học:** S1 là đóng góp chắc chắn nhất về mặt novelty. Nó không phụ thuộc vào việc α/EPA có thắng hay không.

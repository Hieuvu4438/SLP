# 09 — Phản biện độc lập cho `08` (v1) và phản hồi

**Hình thức.**

- Một agent phản biện độc lập, đóng vai area chair (flow matching, mô hình sinh Riemann, sinh chuyển động, SLP), đọc `08` v1, `01` và `06`, mã metric và script toy.
- Agent không được đọc các method khác trong repo, không chạy huấn luyện hay suy diễn mô hình, và không đụng test.
- Agent chạy lại toy và thêm kiểm tra riêng (`scratchpad/toy_review.py`, numpy).
- Mọi kiểm tra toy của phản biện đều đã được **chạy lại và xác nhận** (22/11/2026).

**Kết luận của phản biện về v1.** Không thể triển khai như đang viết. Bốn lý do:

1. Giả thuyết lõi ("một cơ chế": nhịp điệu) không được bằng chứng ủng hộ, và một phần bị chính toy của v1 phủ định.
2. Cổng chính (P0.2/P0.3) không phân biệt được các giả thuyết.
3. β có vấn đề lệch giá đỡ giữa train và test.
4. Novelty của T1/T2 và β bị thổi phồng.

---

## 1. Phản hồi từng điểm

Trạng thái:

- **Chấp nhận** = sửa trong `08` v2.
- **Một phần** = chấp nhận có điều chỉnh.
- **Bác** = giữ nguyên, có lý do.

| # | Điểm phản biện | Trạng thái | Thay đổi trong `08` v2 |
|---|---|---|---|
| 1 | **Mâu thuẫn trung tâm.** Theo chính toy, DTW *phạt* trung bình Euclid khi nhịp điệu chiếm ưu thế. Quan sát "co về trung bình và NFE 1 làm DTW tốt hơn" vì vậy chỉ ra phương sai *không gian/nội dung*, không phải nhịp điệu. "DTW thưởng cho sự nhoè do nhịp điệu" là sai chiều. | **Chấp nhận** | §0.2, §2.3: bảng bốn nguồn phương sai (nội dung, không gian, nhịp điệu, lỗi mô hình) và cách DTW đối xử với từng nguồn. Nhịp điệu thành *giả thuyết cần đo*. Bỏ "một cơ chế duy nhất". |
| 2 | Hình chữ U không đặc thù cho nhịp điệu. Toy: chỉ không gian dạng sin² cho 0.88/0.57/0.88. | **Chấp nhận** (đã chạy lại) | Loại P0.3 v1. §2.3 ghi đính chính. |
| 3 | T1 là mô hình (δ Gaussian, độc lập với m, s biến thiên chậm), không phải định lý. Tính độc lập đáng ngờ. | **Chấp nhận** | §2.4, T1. |
| 4 | T2: chi phí evaluator là L2 *không bình phương*, nên nghiệm theo khung là median hình học theo khớp, không phải trung bình. Có sàn rời rạc hoá (DTW giữa hai warp là 3.7–4.5). Độ dài là một phần quyết định. | **Chấp nhận** (đã kiểm tra `pairwise_part_cost` và chạy lại toy) | §2.4, T2. α dùng chi phí không bình phương. Thêm đối chứng A-L1 để tách "median so với mean" khỏi "bất biến thời gian". |
| 5 | Đọc toy sai: "barycenter thắng cả hai" sai ở hàng 0.10/0 và 0.15/0 (bị bỏ sót). | **Chấp nhận** | §2.5: thêm hàng 0.15/0; sửa cách đọc (ii). |
| 6 | **"Bregman ⇒ không thiên lệch" sai trên manifold.** Loss geodesic/dây cung cho nghiệm là trung bình Karcher/dây cung; Log_{x_t}(trung bình) ≠ E[Log_{x_t} x1]. | **Chấp nhận** | §2.4, T3. Đây cũng là một điểm phê bình mới đối với *paper*: sampler re-planned của SignNMFlow không đi theo vận tốc biên của RFM. Thêm **τ-loss** (đầu ra tiếp tuyến; N0), áp dụng cho cả baseline. |
| 7 | γ không đổi nghiệm; E4a đổi nghiệm. Hai thứ không "tương đương". | **Chấp nhận** | §2.4, §3.2. γ chỉ còn là ablation phân bổ năng lực. |
| 8 | β: anchor đồng huấn luyện làm điều kiện trôi; w* là thống kê dung lượng cao; ã có răng cưa; warp đồng nhất không điển hình; TTS *dự đoán* thời lượng. | **Chấp nhận** | §3.4: thay β bằng **EPA-Flow**. Template đóng băng (hai giai đoạn); warp trơn đơn điệu; **mô hình pha p(w \| c)**; flow biên độ trong thời gian đã đăng ký, có điều kiện trên tốc độ cục bộ (phụ thuộc pha–biên độ); kiểm tra lỗi tái tạo. |
| 9 | "Nguồn không đổi phân phối đầu ra" không áp dụng cho β; DTW giảm có thể chỉ do bớt đa dạng. Hệ quả hữu ích: DTW giữa hai người ký là mức của sampler hoàn hảo. | **Chấp nhận** | §2.4. **Đường đối chứng con người** trên 280 câu dev CSL (P0.S-H). Mọi arm báo độ đa dạng. |
| 10 | Thiếu probe quyết định: barycenter của K mẫu v2 (đo phần nhịp điệu trong phương sai *của mô hình*). P0.2 đo phương sai giữa người ký trên câu *đã thấy*, không phải phương sai của mô hình trên câu mới. | **Chấp nhận** | §6: **P0.S** thành cổng chính. P0.2 thành phụ, để so dữ liệu với mô hình. |
| 11 | Nhiễu trong P0.2: phong cách người ký, tay thuận, biến thể từ vựng, chỉ 2 tham chiếu, cần null, độ dài chung. | **Chấp nhận** | §6, P0.2: phân tầng, kiểm tra tay thuận, null có phổ khớp, độ dài chung, `dtw_jpe_norm`, khoảng 500 câu. |
| 12 | MBR-medoid chọn mẫu ít biên độ nhất, tức hồi quy trá hình. Dùng soft-DTW γ nhỏ làm tiện ích gần như dùng chính metric. | **Chấp nhận** | §3.5: so tại cùng quãng đường; tiện ích γ lớn; ε thành phụ. |
| 13 | Đối chứng A-L2 so với A-SDTW lẫn nhiều yếu tố; thiếu L1, kin, ngân sách tinh chỉnh bằng nhau; dùng val để vừa chọn vừa báo cáo. | **Chấp nhận** | §3.3 (A-L1, A-L2+kin, ngân sách bằng nhau); §5.7 (chia val thành nửa chọn và nửa báo cáo, nhóm theo văn bản). |
| 14 | Thiếu baseline: barycenter mẫu v2, kNN + barycenter, NFE 1 + unsharp masking, L2 + khớp vận tốc. | **Chấp nhận** | P0.S, P0.6 (δ-0), P0.7, A-L2+kin. |
| 15 | Thiếu đối chứng cho β: 2×2 nguồn × điều kiện; E6 cùng σ0; anchor L2 so với SDTW; w oracle/lấy mẫu; Euclid. | **Một phần** | §3.4: bảng ablation EPA (w lấy mẫu/mode/oracle, no-register, E6 cùng σ0, template A-L2, Euclid 6D). Bỏ 2×2 vì trong EPA nguồn và điều kiện cùng hệ thời gian. Arm "no-register" bao phần "nguồn/điều kiện trên a". |
| 16 | Thống kê: sàng lọc 1 seed không phân giải được hiệu ứng 1–3%; bootstrap theo chuỗi bỏ qua phương sai seed. | **Chấp nhận** | §5.7: bootstrap phân tầng; endpoint chính đăng ký trước; ≥ 3 seed khi hiệu ứng < 2 × SD seed. Vẫn giữ sàng lọc 1 seed, nhưng chỉ để *loại* arm, không để tuyên bố. |
| 17 | Chi phí khoảng 0.9–1.2k GPU-h. How2Sign (khoảng cách lớn nhất) bị để tới G4. | **Chấp nhận** | §6: tổng 0.84–1.3k, có cổng cắt; P0.H chẩn đoán How2Sign ngay ở G0. |
| 18 | Rò rỉ: fold nhóm theo văn bản; loại *gần* trùng; nguồn gốc `pami0`. | **Chấp nhận** | §3.1, §2.2, R9. |
| 19 | Khả năng vượt paper: CSL chủ yếu nhờ δ; How2Sign không đáng tin; NFE 1 đơn thuần đã "vượt" được CSL, đúng điều kế hoạch muốn tránh. | **Chấp nhận** | §0.5, §9: δ thành thành phần kết quả chính (hướng R); How2Sign có điều kiện; endpoint R ràng buộc quãng đường và R-precision. |
| 20 | MaDiS không phải "bằng chứng mạnh" cho giả thuyết khác giao thức của Phoenix. | **Chấp nhận** | §4.1: "chỉ tương thích, không chứng minh". |
| 21 | Novelty: T1/T2 là sách giáo khoa (FDA: Kneip & Gasser 1992; Wang & Gasser 1997; Marron 2015; Time-Agnostic Prediction ICLR'19; hoà trộn chuyển động cần đăng ký). β gần Grad-TTS/Glow-TTS/Matcha; trong SLP có NAT-EA và FS-Net. α là kỹ thuật. Thiếu phân đoạn ký hiệu và Ham2Pose. | **Chấp nhận** | §4.2: hạ nhãn; thêm các công trình này cùng Elastic Shape VAE (2026) và Motion Infilling (EG'25), tìm thêm sau phản biện. |
| 22 | §4.3 chưa publishable như đang viết. Bản khả thi: paper *phân tích* "SLP variance nằm ở đâu và DTW-MJE thưởng cho cái gì", với α/δ làm minh hoạ. | **Chấp nhận** | §4.3: S1 (phân tích) thành câu chuyện chính về novelty; S2 (EPA-Flow) có điều kiện. |
| 23 | Nhịp điệu mang nghĩa trong ngôn ngữ ký hiệu (ngữ điệu, thể), không chỉ là nhiễu. Nên có metric tính tự nhiên của nhịp điệu và đánh giá người. | **Chấp nhận** | §5.5: phân bố thời lượng ký hiệu (Phoenix dev có `pami0`, 519/519) và tỉ lệ khung "giữ". Đánh giá người nhỏ trong S1. |
| 24 | Pha và biên độ không độc lập (ký nhanh thì biên độ nhỏ). | **Chấp nhận** | EPA: flow biên độ có điều kiện trên đặc trưng tốc độ cục bộ của w. |
| 25 | Kết hợp E4a vào hướng R mà không có tuyên bố lý thuyết. | — (phản biện không nêu; tự bổ sung) | §3.2 ghi rõ E4a là thành phần thực dụng. |

**Không có điểm nào bị bác.**

**Điểm 15 được chấp nhận một phần.** Thiết kế 2×2 là cần cho β v1. Trong EPA, template và x̄1 cùng hệ thời gian, nên câu hỏi "nguồn hay điều kiện" được trả lời bằng arm no-register và E6.

## 2. Những gì phản biện xác nhận là đứng vững

- Toy tái lập đúng mọi con số của v1.
- T3 đúng trong Rⁿ (kể cả bản sg(x̂1)).
- Hướng "tách ước lượng điểm và sinh mẫu" và đường đối chứng con người có giá trị.
- δ là thành phần có khả năng đem lại kết quả nhất. Khoảng cách 5.8% PA-hand của CSL là vấn đề hình tay, đúng chỗ δ nhắm tới.

## 3. Phát hiện mới sinh ra từ vòng phản biện

1. **Sampler của SignNMFlow không đi theo trường biên trên manifold** (điểm 6).
   - Với target-prediction và loss dây cung, sampler re-planned không đi theo vận tốc biên của RFM.
   - Vòng 2 đính chính: đây là khoảng cách *đã biết* giữa loss x̂1 dây cung và loss tiếp tuyến của FrameFlow. Đó là hiệu ứng độ cong bậc hai, và ở NFE 5 nhiều khả năng nhỏ hơn sai số rời rạc hoá.
   - τ-loss (chính là loss của FrameFlow) là cách sửa rẻ. Độ lớn hiệu ứng chưa biết.
2. **Đường đối chứng con người trên CSL dev** (280 câu, ≥ 2 người ký). Nếu bản tái lập (5.72/7.38) hoặc số paper thấp hơn mức người–người, đó là minh chứng trực tiếp rằng metric thưởng cho trung bình. Đây là một hình quan trọng cho S1.
3. **Phoenix dev cũng có `pami0`.** Cho phép đo tính tự nhiên của nhịp điệu ở mức ký hiệu trên val.

---

## 4. Vòng 2: kiểm tra `08` v2

**Kết luận của phản biện.** Từ "từ chối" lên "sửa lớn". **G0 được phép chạy** (khoảng 10 GPU-h, chỉ val, không huấn luyện), sau ba sửa cổng. Hướng S chỉ được huấn luyện sau khi đặc tả EPA được sửa.

| Điểm vòng 2 | Trạng thái | Sửa trong `08` |
|---|---|---|
| §0.2: trung bình hoá cũng thưởng phần phương sai của lỗi mô hình (d) | Chấp nhận | "(a), (b) hoặc phần phương sai của (d)" |
| §2.5 (ii): barycenter thua trung bình Euclid ở 0.05/0.30 | Chấp nhận | Sửa câu đọc |
| Vận tốc τ thiếu hệ số κ′(t) | Chấp nhận | κ′(t)/(1−κ(t)) · E[Log] |
| τ-loss chính là loss SO(3) của FrameFlow | Chấp nhận | Trích nguồn; hạ novelty xuống "không mới" |
| τ-loss không tương thích NG; quy ước dấu; cut locus; trọng số λ(t); NFE 1 đổi điểm | Chấp nhận, với một đính chính | Mục "Điều kiện áp dụng τ-loss". Đính chính: 126.5° là góc *quay*, tức khoảng 63° trên S³, nên θ/sin θ ≈ 1.2 chứ không phải 2.7. `logmap` của bản tái lập làm trên S³, không lật SO(3), nên nhất quán (đã kiểm tra mã). |
| "Sampler của paper thiên lệch" là khoảng cách đã biết; bậc hai; không rõ hướng; "1–3%" không có cơ sở | Chấp nhận | Hạ giọng; §9: hiệu ứng τ "chưa biết" |
| EPA: không phải đổi biến, mà là đẩy tiến; cần tái tạo chính xác; lưới đăng ký ≥ 2T; lỗi theo dải tần | Chấp nhận | §3.4 "Tính hợp lệ" |
| EPA: độ dài thuộc về ai; cross-fitting cả w*; "mode" không xác định cho flow | Chấp nhận | T ngoại sinh (mặc định) hoặc "pha sở hữu độ dài"; "w trung tâm" |
| Thêm ablation: w đồng nhất khi test, p(w) không điều kiện, chỉ pha, không điều kiện tốc độ, K nút; E6 *trong thời gian đã đăng ký* | Chấp nhận | Bảng ablation EPA |
| Cổng S: "f_τ − null ≥ 0.1" luôn qua khi có bất kỳ nhịp điệu nào; ngưỡng 3% có thể qua khi không có nhịp; so median với mean là lẫn | Chấp nhận | Cổng mới: Gain = median theo khung − median sau căn, tại cùng quãng đường; null *trên chính lợi ích*; ≥ 2%, CI loại 0; kiểm tra nhất quán T1; theo từng dataset |
| Phương án dự phòng của P0.S dùng P0.2 là sai đối tượng | Chấp nhận | Dùng ensemble B×3 seed (phương sai trên câu mới) |
| Endpoint R: dung sai quãng đường −0.02 tự mua được khoảng 1%; R-precision bị δ thổi phồng; mâu thuẫn với tiêu chí §9 | Chấp nhận | Endpoint quyết định: `frontier_effect`, rào chắn M4 và back-translation. Tuyên bố so với paper là thứ cấp và đăng ký trước. |
| Đường đối chứng con người: "tốt hơn người ⇒ thiên lệch metric" là nói quá | Chấp nhận | Cách đọc đúng: v2 hành xử như ước lượng; bệnh lý riêng của DTW là co về trung bình *không dùng văn bản* làm điểm tốt lên |
| Thống kê: SD seed của Phoenix không áp được cho CSL; chia đôi val làm CI rộng khoảng 1.4× | Chấp nhận | Hai nửa A/B hoán đổi; SD seed theo từng dataset |
| `pami0`: nguồn gốc còn là TODO; P0.6 mở rộng một method thì phạm quy tắc chia sẻ mã | Chấp nhận | Nguồn gốc đã kiểm tra (bản phát hành SOKE; độ dài = số khung 25 fps; khoảng 1,200 gloss × 3 trạng thái); quy tắc dùng. Chỉ mục truy hồi dùng chung chuyển vào `src/slp/retrieval.py` kèm test |
| GPU-h: cận dưới 0.84k không đúng; cross-fitting và Euclid-6D chưa tính | Chấp nhận | Bảng kịch bản: khoảng 615 / 1,060 / 1,430 GPU-h; cổng S cắt khoảng 445 |

**Việc còn phải làm trước khi huấn luyện hướng S** (theo phản biện): đã ghi vào đặc tả. Mỗi mục được kiểm bằng test của N0/N3/N4 khi code:

- NG có tương thích với τ-loss không;
- quy ước dấu cho τ-loss;
- độ dài thuộc về ai trong EPA;
- lưới đăng ký dày;
- bảng ablation bổ sung.

---

## 5. Kiểm chứng sau phản biện: đường đối chứng người–người (P0.S-H, đã chạy)

- Probe: `probes/probe_human_line_csl.py` → `probe_human_line_csl.json`.
- Chỉ dùng GT và dự đoán v2 đã lưu; không chạy mô hình; CPU; val.
- Tập đo: 280 câu dev CSL, mỗi câu do hai người ký khác nhau thực hiện.

| | JPE body | JPE hand | PA body (ua) | PA hand |
|---|---|---|---|---|
| Người A so với người B | 5.58 | 6.29 | 4.76 | 1.12 |
| v2 | 5.54 | 7.18 | 4.89 | 1.32 |
| Tư thế trung bình tĩnh | 6.75 | 9.40 | 5.94 | 1.75 |
| v2 − người (CI 95%) | −0.04 [−0.21, 0.11] | +0.89 [0.70, 1.10] | +0.13 [−0.01, 0.28] | +0.20 [0.16, 0.24] |

**Kết luận.**

- Body của v2 đã ở mức người–người. Giảm DTW body thêm là hành vi ước lượng điểm.
- Tay kém người 14–18%. Đây là khoảng trống thật và là mục tiêu chính của hướng R.

Điều này trả lời một phần câu hỏi "cải thiện ở đâu thì không phải chơi metric", mà phản biện vòng 1 nêu ở điểm 19. Nó cũng đổi ưu tiên của `08`: thêm R-hand (E4c + E9) vào G1, và bộ nhớ hình tay vào δ.

## 6. Kiểm chứng thêm: phân rã pha–biên độ giữa người ký (P0.2-lite, đã chạy)

- Probe: `probes/probe_phase_amplitude_csl.py` → `.json`.
- Chỉ dùng GT; cùng 280 cặp CSL dev ở §5; CPU.
- f là phần khác biệt theo khung mà căn DTW xoá được, *sau khi* đã bỏ nhịp chung bằng co giãn tuyến tính.
- Null theo đề nghị của phản biện: phần dư theo khung được xáo pha (cùng phổ).

| | Body | Tay trái | Tay phải |
|---|---|---|---|
| f | 0.158 | 0.334 | 0.349 |
| f_null (tỉ lệ DTW giữ đúng đường chéo) | 0.001 (91%) | 0.001 (96%) | 0.000 (100%) |
| f − null, CI 95% | 0.157 [0.146, 0.168] | 0.333 [0.317, 0.350] | 0.349 [0.337, 0.360] |

**Kết luận.**

- Thành phần nhịp điệu phi tuyến của *dữ liệu* có thật: khoảng 1/3 khác biệt giữa người ký ở tay, khoảng 1/6 ở body. Nó lớn nhất đúng ở tay, nơi v2 kém người 14–18% (§5).
- Đây là bằng chứng *về phía dữ liệu* cho T1. Nó không thay cho cổng P0.S (phương sai của *mô hình*), đúng như phản biện điểm 10 yêu cầu.
- Giới hạn: gồm cả phong cách nhịp riêng của từng người ký; chỉ 2 lần ký mỗi câu; null xáo pha không mô phỏng được việc DTW khai thác các đoạn giữ tư thế giống nhau.

## 7. Cổng P0.S: script sẵn sàng, hiệu chuẩn trên toy

- `probes/probe_gate_ps.py` (chỉ suy diễn, val; người dùng chạy).
- **Sửa thống kê cổng.** Luôn ghép quãng đường thì phạt cả ước lượng *trội hơn hẳn*: trên toy chỉ có nhịp, lợi ích giảm từ 0.65 xuống 0.02. Cổng vì vậy kiểm tra trội trước (DTW thấp hơn *và* quãng đường ≥ thì lấy Gain thô), chỉ ghép quãng đường khi không trội.
- Hiệu chuẩn trên toy (8 mẫu, 5 lần lặp), Gain − null:

| Toy | Gain − null | Kết quả cổng |
|---|---|---|
| Chỉ nhịp | 0.49 | Qua |
| Chỉ không gian | −0.002 | Không qua (đúng) |
| Hỗn hợp | 0.11 | Qua |

- Thống kê tách đúng ba trường hợp. Điều này trả lời điểm hiệu chuẩn cổng của vòng 2.
- `dtw_path` khớp `metrics.dtw_cost` (đã kiểm tra trên ma trận ngẫu nhiên).

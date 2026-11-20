# 03 — Literature review: biểu diễn chuyển động, biểu diễn ngôn ngữ ký hiệu, flow matching và đánh giá

**Hình thức.** Đây là tổng quan tường thuật/tích hợp (narrative/integrative review) phục vụ thiết kế thí nghiệm, không phải systematic review. Nguồn được chọn theo mức liên quan tới từng quyết định thiết kế của SignNMFlow.

**Quy trình.** Ba nhánh tìm kiếm được chạy song song (flow matching; biểu diễn và loss chuyển động; SLP và đánh giá). Mỗi tài liệu được xác minh bằng cách tải trang arXiv hoặc trang proceedings. Tài liệu không xác minh được thì bị loại. Danh mục đầy đủ và tóm tắt từng bài nằm ở `sources/` (sổ nguồn gốc).

**Kiểm tra trực tiếp.** Các chi tiết then chốt của MFM (Eq. 4, Eq. 6, metric LAND và RBF) được kiểm tra lại trên bản HTML arXiv 2405.14780v2.

**Mức chứng cứ.**

- Peer-reviewed (NeurIPS/ICML/ICLR/CVPR/ICCV/ECCV/AAAI/TPAMI) được ưu tiên.
- Preprint được đánh dấu *(preprint)*.
- Bài blog được đánh dấu *(grey)* và chỉ dùng làm gợi ý.

---

## 1. Flow matching: đường đi, metric và tham số hoá

### 1.1 Riemannian flow matching là chuẩn, không phải đóng góp mới

- Chen & Lipman (ICLR 2024) định nghĩa conditional path trên manifold qua một premetric. Path Exp_{x0}(t Log_{x0}x1) của SignNMFlow chính là path RFM trên (S³)^J.
- Trên các tích nhóm quay, đây là công thức đã được dùng rộng rãi:
  - FrameDiff (Yim và cộng sự, ICML 2023);
  - FrameFlow (Yim và cộng sự 2023);
  - FoldFlow (Bose và cộng sự, ICLR 2024);
  - Flow Matching on Lie Groups (Sherry & Smets, GSI 2025).
- Với chuyển động người:
  - PoseD-Flow (Nadar và cộng sự, CVPR 2026) làm RFM trên tích các phép quay khớp;
  - RMG (Miao và cộng sự 2026, *preprint*) làm RFM cho sinh chuyển động từ văn bản và báo cáo SOTA FID trên HumanML3D.
- **Hệ quả.** "GE" trong Table 3 là một baseline cần có, không phải đóng góp mới. Hai bài trên là các so sánh mà paper bỏ sót.

### 1.2 Path học được dưới metric phụ thuộc dữ liệu: trùng lặp với Metric Flow Matching

- **MFM** (Kapuśniak và cộng sự, NeurIPS 2024).
  - Interpolant: x_{t,η} = (1−t)x0 + t·x1 + t(1−t)·φ_{t,η}(x0, x1) (Eq. 4).
  - φ được huấn luyện bằng E[ẋᵀG(x_t; D)ẋ] (Eq. 6), mà paper gọi là kinetic energy.
  - Metric LAND: G = (diag(h(x)) + εI)⁻¹, với h là phương sai cục bộ có trọng số kernel *không chuẩn hoá*.
  - Metric RBF: đặt tại các tâm K-means.
  - SignNMFlow lặp lại công thức này: thay đường thẳng bằng geodesic trên S³, và thay metric đường chéo bằng ma trận scatter đầy đủ. SignNMFlow không trích dẫn MFM.
- **Hai khác biệt quan trọng**, và cả hai đều bất lợi cho SignNMFlow (probe P1–P2 trong `01`):
  1. MFM nghịch đảo phương sai cục bộ, nên đi dọc hướng dữ liệu là rẻ. SignNMFlow dùng (S + ρI)/m, nên đi dọc hướng đó lại đắt.
  2. h của MFM không chuẩn hoá nên mang thông tin mật độ. Trong SignNMFlow, m(x) bị triệt tiêu.
- **Các hướng liên quan khác:**
  - GSBM (Liu và cộng sự, ICLR 2024): path tối ưu dưới kinetic energy cộng state cost.
  - Stochastic interpolants (Albergo và cộng sự, JMLR) và bản dùng data-dependent coupling (ICML 2024).
  - Kinetic-optimal paths (Shaul và cộng sự 2023). Bài này lưu ý rằng năng lượng của path *có điều kiện* khác với năng lượng của path *biên*.
  - EGGFM (2025, *preprint*): phê phán metric kiểu kernel/k-means ở chiều cao.
  - GeodesicNVS (CVPR 2026): path theo geodesic của mật độ.
- **Hệ quả cho thí nghiệm.** Trước khi quy lợi ích cho "naturalness" cần có đối chứng dương GE so với NG, và các arm geoshift và NG-parallel (E1 trong `04`).

### 1.3 Đổi path ≈ đổi lịch nhiễu và trọng số thời gian

- Variational Diffusion Models (Kingma và cộng sự, NeurIPS 2021): mục tiêu liên tục bất biến với lịch nhiễu, chỉ trừ hai đầu mút. Lịch nhiễu chỉ quyết định *trọng số* và *phương sai* của quá trình tối ưu.
- Kingma & Gao (NeurIPS 2023): các loss khuếch tán đều là ELBO với trọng số theo mức nhiễu.
- Chen (2023): dữ liệu càng dư thừa càng cần lịch nhiễu "nhiều nhiễu" hơn.
- SD3 (Esser và cộng sự, ICML 2024): logit-normal timestep và shift theo độ phân giải. Đổi phân phối t tương đương đổi trọng số loss.
- **Kết nối với probe.** Đường naturalness học được đạt năng lượng bằng cận dưới của mọi phép tái tham số hoá thời gian của geodesic, và tiến độ của nó khớp với time-shift s ≈ 2.5 ± 0.3. Điều này đặt SignNMFlow đúng vào vùng mà các tài liệu trên đã giải thích: lợi ích (nếu có) đến từ trọng số mức nhiễu.

### 1.4 Tham số hoá dự đoán

- **JiT** (Li & He 2025, *tech report*): dự đoán dữ liệu sạch dễ hơn dự đoán nhiễu hay vận tốc khi dữ liệu nằm trên manifold thấp chiều. Ủng hộ lựa chọn target-prediction của SignNMFlow.
  - Tuy nhiên, x-pred với v-loss (trọng số khoảng 1/(1−t)², có clip) tốt hơn x-pred với x-loss: FID 8.62 so với 10.14.
- **MDM** (Tevet và cộng sự, ICLR 2023): dự đoán x0 cho phép dùng loss hình học (vị trí, vận tốc).
- **Meng và cộng sự** (CVPR 2025): với diffusion chuyển động, dự đoán x0 và vận tốc tốt hơn ε rất nhiều.
- **FrameFlow**: dự đoán khung sạch, và vector field tính bằng log_{r_t}(r̂1)/(1−t). Đây đúng là cấu trúc re-planned sampler của SignNMFlow; FrameFlow còn dùng lịch mũ riêng khi suy diễn.

### 1.5 Kỹ thuật chuẩn của conditional FM mà SignNMFlow chưa dùng

| Kỹ thuật | Bằng chứng | Ánh xạ sang SignNMFlow |
|---|---|---|
| CFG | Ho & Salimans 2021; Guided Flows (Zheng và cộng sự 2023); Proteina, JiT | Text dropout 10%. Trên manifold: Exp_{x̂u}(w·Log_{x̂u} x̂c). |
| Self-conditioning | Analog Bits (Chen, Zhang, Hinton, ICLR 2023); FrameDiff, FrameFlow, Proteina | Gần như miễn phí với re-planned sampler. |
| EMA / post-hoc EMA | EDM2 (Karras và cộng sự 2023); JiT | Paper không nhắc. |
| Prior có thông tin | PriorGrad (Lee và cộng sự, ICLR 2022); stochastic interpolants với data-dependent coupling (ICML 2024); IGSO(3) | Wrapped normal tại tư thế trung bình (E6-prior). |
| Minibatch OT | Tong và cộng sự (TMLR 2024); FoldFlow-OT | Ghép cặp theo khoảng cách geodesic. |
| Ít bước | MeanFlow (Geng và cộng sự, NeurIPS 2025); Shortcut (Frans và cộng sự, ICLR 2025) | Chỉ khi muốn NFE < 5. |

### 1.6 Biểu diễn quay khi học

- Zhou và cộng sự (CVPR 2019): mọi biểu diễn ≤ 4 chiều đều không liên tục; 6D thì liên tục.
- Levinson và cộng sự (NeurIPS 2020): SVD 9D là phép chiếu tự nhiên.
- Brégier (3DV 2021): khuyến nghị loss min ‖q ± q*‖².
- Geist và cộng sự (ICML 2024): quaternion nửa không gian chỉ ổn cho góc nhỏ, và đổi loss không sửa được tính không liên tục.
- **Đối chiếu dữ liệu (P4).** Quaternion tay không bao giờ có w < 0.3. Body chỉ 0.7% có w < 0.3. Đổi dấu theo thời gian ~1e−4. Vậy các vấn đề này *có thật về lý thuyết nhưng nhỏ trên dữ liệu này*. Head 6D được để ở ưu tiên thấp (E8).

---

## 2. Biểu diễn chuyển động và loss

### 2.1 Loss trong không gian vị trí (FK)

- QuaterNet (Pavllo và cộng sự, BMVC 2018; IJCV 2019): quaternion nhưng loss tính qua forward kinematics, vì "rotation regression accumulates errors along the kinematic chain".
- Liu và cộng sự (TPAMI 2022): biểu diễn có tác động then chốt; thay L2 bằng loss geodesic và FK.
- SGRE (Wang & Zhang, ICCV 2023): lỗi tích luỹ lớn nhất ở khớp đầu mút (cổ tay, đầu ngón).
- MDM và EMAGE (CVPR 2024): loss vị trí, vận tốc, gia tốc trên x̂0.
- **Hệ quả.** Metric của paper đo vị trí sau FK (tay căn theo cổ tay hoặc Procrustes), nhưng loss của paper là L2 trên quaternion với trọng số đồng đều. Đây là lệch pha lớn nhất giữa mục tiêu huấn luyện và metric (E4).

### 2.2 Biểu diễn dư thừa và biểu diễn vị trí

- HumanML3D (Guo và cộng sự, CVPR 2022): đặc trưng dư thừa gồm vị trí cục bộ, vận tốc và 6D.
- Meng và cộng sự (CVPR 2025): đặc trưng dư thừa hại diffusion; dùng nhóm liên tục thiết yếu thì tốt hơn.
- ACMDM (Meng và cộng sự 2025, *preprint*): toạ độ tuyệt đối giúp sinh chuyển động dễ hơn mà không cần loss động học phụ.
- Sign-IDD (Tang và cộng sự, AAAI 2025, cùng nhóm tác giả SignNMFlow): biểu diễn xương gồm hướng và độ dài.
- **Hệ quả.** Cược lớn dài hạn là sinh trực tiếp vị trí hoặc hướng xương rồi IK. Với ngân sách hiện có, FK loss (E4) là phương án rẻ hơn mà hướng tới cùng mục tiêu.

### 2.3 Cấu trúc theo bộ phận và tay

- ParCo (ECCV 2024), TalkSHOW (CVPR 2023), EMAGE (CVPR 2024) và SOKE (ICCV 2025) đều tách body, tay và mặt rồi điều phối chéo.
- Body2Hands (CVPR 2021): bàn tay dự đoán được một phần từ chuyển động cánh tay.
- Neural Sign Actors (CVPR 2024): denoiser dạng đồ thị theo khung xương SMPL-X.
- **Hệ quả.** Một head tay có điều kiện cánh tay (E9) có cơ sở, nhưng là thay đổi lớn. Chỉ làm sau khi loss đã được sửa (E9).

### 2.4 Mượt và đánh đổi mượt/giật

- Mao và cộng sự (ICCV 2019, DCT): cắt bớt tần số cao giảm giật.
- MOJO (CVPR 2021): trọng số loss vận tốc lớn thì quá mượt, nhỏ thì giật.
- SmoothNet (ECCV 2022): tinh chỉnh hậu kỳ.
- **Hệ quả.** Mọi loss vận tốc (E4-b) phải được quét trọng số và có rào chắn tốc độ. Lý do: DTW-JPE không phạt việc quá mượt (xem `01` mục C.2 và README).

---

## 3. Sign language production và biểu diễn ngôn ngữ ký hiệu

### 3.1 Bức tranh SOTA trên benchmark SMPL-X của SOKE

| Họ phương pháp | Đại diện (đã xác minh) | Bài học cho SignNMFlow |
|---|---|---|
| Token + LM | T2S-GPT (ACL 2024); SOKE (ICCV 2025): tokenizer tách body/tay + mBART + retrieval từ từ điển; M3T (BMVC 2026): FSQ đa kênh, mặt 100-d | Tách bộ phận cơ thể; mặt cần nhiều chiều hơn 10-d. |
| Diffusion | NSA (CVPR 2024): GNN theo khung xương SMPL-X; Sign-IDD (AAAI 2025): biểu diễn xương; GARD (2026, *preprint*): tự hồi quy theo gloss, dẫn hướng geodesic tại ranh giới; PIDiffSign (2026, *preprint*): đầu FK + ràng buộc động lực học | Ràng buộc hình học thuần tuý không cải thiện ngữ nghĩa (PIDiffSign). Chuyển động sinh ra "ít động năng hơn" chuyển động thật. |
| Flow matching | SignFlow (ICMI 2025): CFM tuyến tính + OT; SignSparK (ECCV 2026): CFM trên 6D, infill từ keyframe, **có CFG và loss vận tốc** | Keyframe/retrieval là đòn bẩy mạnh nhất chống trung bình hoá. CFG và loss vận tốc đã được dùng trong SLP. |
| Retrieve-and-connect | Spoken2Sign (ECCV 2024) | Baseline tra cứu phải có mặt trong bảng. |

**Các bảng so sánh của SignNMFlow có chọn lọc** (bảng số liệu đầy đủ ở `sources/lit_slp.md`).

- SignSparK bản retrieval (SR) tốt hơn SignNMFlow ở 3/4 cột trên Phoenix: PA-body 3.68 so với 3.85, JPE-body 3.51 so với 4.37, JPE-hand 4.82 so với 5.57. SignNMFlow chỉ thắng ở PA-hand (1.07 so với 1.18).
  - Caveat: SignSparK dùng pseudo-GT riêng, nên so sánh không chặt theo cả hai chiều.
- GARD được trích dẫn nhưng không có trong bảng. M3T và SignMask (ACM TOMM 2026) vắng mặt.
- Lần chạy lại SOKE của GARD lệch 0.01–0.08 so với số SOKE tự báo. Đó là thang nhiễu tái lập, ngang với nhiều hiệu ứng trong Tables 3–4 của SignNMFlow.

### 3.2 Biểu diễn ngôn ngữ ký hiệu: cấu trúc có thể khai thác

- **Âm vị học ký hiệu.** Hình tay (handshape), vị trí, chuyển động, hướng lòng bàn tay, và dấu hiệu phi thủ công (non-manual).
  - ASL-LEX 2.0 (JDSDE 2021) mã hoá các thuộc tính này cho 2,723 ký hiệu ASL.
  - Hong & Košecká (2026, *preprint*): điều kiện hoá theo thuộc tính âm vị học tăng khả năng phân biệt gloss.
  - HandMDM (2025, *preprint*): sinh chuyển động tay từ mô tả thuộc tính.
- **Ràng buộc kiểu Battison** (SGNify, CVPR 2023). Hai loại:
  - đối xứng trái/phải cho ký hiệu hai tay đối xứng;
  - hình tay *không đổi* (hoặc SLERP giữa hai hình tay) trong một ký hiệu.
  - Cả hai đều là penalty geodesic tự nhiên trên S³. Chúng khớp với bộ máy hình học của SignNMFlow mà không cần K-means.
- **Ranh giới ký hiệu.** Phân đoạn ký hiệu (Moryossef và cộng sự, EMNLP Findings 2023), keyframe onset/mid/offset (SignSparK), dẫn hướng chuyển tiếp giữa gloss (GARD).
  - Đây là các tín hiệu có cấu trúc ngôn ngữ. Prototype per-joint của SignNMFlow hoàn toàn không có cấu trúc này.
- **Dấu hiệu phi thủ công.**
  - SignNMFlow giữ kênh biểu cảm 10-d Euclid với trọng số loss 0.1 và không bao giờ đánh giá nó.
  - M3T cho thấy biểu cảm 10-d của SOKE chủ yếu chỉ thay đổi độ mở hàm.
- **Hệ quả.**
  - Các prior âm vị học là hướng "naturalness" có cơ sở ngôn ngữ học, đáng thử *sau* các sửa đổi rẻ.
  - Ví dụ: penalty bất biến hình tay trong ký hiệu, cần ranh giới từ một bộ phân đoạn. Hướng này được ghi là E10 (dài hạn) trong `04`.

---

## 4. Đánh giá SLP

### 4.1 DTW joint error thưởng cho chuyển động "ít biên độ"

- **Báo cáo SLRTP 2025 SLP Challenge** (CVPR 2025 workshop, arXiv 2508.06951). Hai câu dưới đây đã được kiểm tra trực tiếp trên bản HTML:
  - "DTW-MJE tended to favor methods that produced less articulated or longer sequences."
  - Baseline Progressive Transformer chỉ đi được 25% quãng đường của tay so với GT nhưng vẫn có DTW-MJE cạnh tranh.
  - Đội hfut-lmc (Tang và He, tác giả SignNMFlow) xếp thứ 2, với dự đoán dài gấp 2.46 lần GT trên PHOENIX14T.
  - Thách thức này đề xuất chỉ số **total distance**: quãng đường tay sinh ra chia cho quãng đường tay GT.
- **Jiang và cộng sự** (WMT 2025): nDTW và nAPE chưa tinh chỉnh có tương quan *âm* với đánh giá của người. Back-translation likelihood là chỉ số ổn định nhất.
- **Kiểm toán bằng baseline không chứa nội dung** (Pelykh và cộng sự, NeurIPS 2026 workshop): các tín hiệu không chứa nội dung đạt điểm tốt ở cả metric khoảng cách lẫn metric back-translation.
- **Conditional collapse** (Hong & Košecká 2026, *preprint*): mức trung thực theo câu đi theo độ phủ của câu train trên câu test, không theo FID. Bài đề xuất kiểm tra hoán đổi text.
- **Đối chiếu với probe P6 của chúng tôi.** Xem README, mục "Kết quả probe metric".

### 4.2 Back-translation BLEU

- B-T phụ thuộc hoàn toàn vào back-translator. SOKE không công bố back-translator của mình.
- M3T đo trần B-T trên GT là 2.04 (How2Sign), 7.11 (CSL) và 11.84 (Phoenix), với back-translator riêng.
- Trong khi đó SignNMFlow báo B-T 16.88–17.49 trên How2Sign, cao hơn cả dịch từ video thật (khoảng 12–16 BLEU, theo YouTube-ASL và SSVP-SLT).
- Repo này cũng có trần GT riêng: Phoenix dev 11.49, CSL dev 4.74.
- **Hệ quả.** B-T chỉ dùng được để so sánh *nội bộ* và luôn kèm trần GT. Nên bổ sung:
  - back-translation likelihood;
  - một metric embedding như SiLVERScore (đã kiểm định trên Phoenix và CSL).

### 4.3 Pseudo-GT và nhiễu

- SMPL-X của Phoenix và CSL do bộ fitter của SOKE tạo ra. Sai số đỉnh (per-vertex) ở tay khoảng 9–11 mm trên dữ liệu mocap sạch (SOKE, Table S5).
- How2Sign dùng fitter NSA, có prior PCA và prior mượt theo thời gian.
- Ảnh Phoenix chỉ 210×260 px, nên pseudo-GT tay nhiễu nhất (M3T cũng nhận xét vậy).
- **Hệ quả.** Chênh lệch DTW-PA-JPE tay 0.1–0.3 phải được so với ngưỡng nhiễu đo (seed, sampler) trước khi kết luận. Kế hoạch `04` §1 làm việc này.

---

## 5. Tổng hợp: bằng chứng nào ủng hộ thí nghiệm nào

| Thí nghiệm (`04`) | Bằng chứng chính | Mức độ |
|---|---|---|
| E1 (gate GE so với NG; geoshift, NG-parallel), E2-F (shift huấn luyện × suy diễn) | Probe P3, P5, P-DA (đo); VDM, Kingma & Gao, SD3, Chen 2023 | Mạnh về cơ chế path; hiệu ứng lên chất lượng sinh chưa biết |
| E4 FK loss, vận tốc | QuaterNet, Liu TPAMI, SGRE, MDM, EMAGE, MOJO | Mạnh (nhiều lĩnh vực lân cận; chưa có trên SLP SMPL-X) |
| E3 EMA, text lr, chọn checkpoint | Log v2 (đo quá khớp); EDM2, JiT | Mạnh (đo) |
| E5-CFG | Ho & Salimans; Guided Flows; SignSparK (SLP) | Mạnh, nhưng cần evaluator ngữ nghĩa (M1) mới đo được |
| E6-prior có thông tin | PriorGrad; data-dependent coupling; FrameFlow | Trung bình |
| E7 self-conditioning | Analog Bits; FrameDiff; Proteina | Trung bình |
| E11 metric MFM | MFM; GSBM | Yếu: time-shift có lẽ xuất hiện với mọi metric đắt ở xa dữ liệu |
| E8 head 6D | Zhou; Geist; Levinson | Yếu trên dữ liệu này (P4) |
| E9 head tay riêng | ParCo; EMAGE; Body2Hands; SOKE | Trung bình, chi phí cao |
| E10 prior âm vị học (bất biến hình tay, đối xứng) | SGNify; ASL-LEX; Hong & Košecká 2026 | Trung bình; cần ranh giới ký hiệu |
| Rào chắn đánh giá (total distance, mean-pose floor, hoán đổi text) | SLRTP 2025; Pelykh 2026; Hong & Košecká 2026; Jiang WMT 2025 | Mạnh |

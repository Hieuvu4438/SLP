# Vượt SignNMFlow với một GPU: integrative review, proposal và phản biện

Ngày chốt nguồn: **16/11/2026**. Trạng thái: **đề xuất, chưa có thí nghiệm nào được chạy cho tài liệu này; test split chưa được dùng.**
Tên làm việc của phương pháp: **PMA-Flow (Posterior-Mean-Anchored manifold Flow)**. Đây là nhãn của proposal, không phải phương pháp đã kiểm chứng.

Công bố về AI: tài liệu do trợ lý AI (Claude) tìm, đọc và tổng hợp; mọi số liệu trích từ paper được lấy qua công cụ đọc trang có tóm tắt, **phải đối chiếu lại PDF trước khi đưa vào bài báo**.

---

## 0. Tóm tắt một trang

1. **DTW-JPE thưởng cho dự đoán trung bình.** Bằng chứng nội bộ (v1, Phoenix dev): pose trung bình tĩnh 4.70/7.51, mô hình đã train 5.41/7.46, NFE 1 < NFE 5 < NFE 20. Lý thuyết perception–distortion giải thích đúng hiện tượng này: bộ ước lượng MSE tối ưu là kỳ vọng có điều kiện, còn một mẫu hậu nghiệm thật có MSE gấp đôi (Blau & Michaeli, 2018; Ohayon et al., 2025). Nhiều nhóm SLP độc lập cũng ghi nhận (MaDiS; SRVQ; issue của tác giả SignSparK; Hong & Košecká, 2026).
2. **Sampler của SignNMFlow về bản chất là hồi quy lặp.** Bước cuối đặt trạng thái tại `s = 1`, nên đầu ra đúng bằng `x̂1 = ϕ(x_0.8, ·)`: một kỳ vọng có điều kiện, không phải một mẫu. Cùng dạng với InDI, nơi 1 bước cho PSNR tốt nhất (Delbracio & Milanfar, 2023).
3. **Đóng góp hình học của SignNMFlow gần trùng Metric Flow Matching** (Kapuśniak et al., 2024): metric RBF từ K-means + hiệu chỉnh `t(1−t)·net` tối thiểu hóa động năng. MFM báo gain lớn trên dữ liệu tổng hợp nhưng dưới 10% trên bài toán thực, và không có thí nghiệm noise→data có điều kiện.
4. **Nút thắt là điều kiện text, không phải đường đi.** Text chỉ giảm val pose loss từ 0.0084 (rỗng) xuống 0.0059 trong bản v1; Hong & Košecká báo "conditional collapse" trên How2Sign.
5. **Đề xuất.** Giữ khung x1-prediction trên (S³)ᴶ của SignNMFlow, thay ba thứ: (a) loss trong không gian khớp qua FK, đúng không gian mà metric đo; (b) nguồn của flow là *kỳ vọng có điều kiện + nhiễu nhỏ* thay cho phân bố đều (PMRF trên đa tạp), kèm một núm α đi dọc đường cong distortion–perception; (c) tăng phụ thuộc text bằng CFG và loss ngữ nghĩa với teacher khác evaluator.
6. **Kết luận phản biện.** RQ1 (chẩn đoán) và RQ2 (FK loss + vệ sinh huấn luyện) đáng chạy ngay, rẻ, xác suất có kết quả cao. RQ3 (nguồn neo kỳ vọng) chỉ chạy nếu cổng G1 cho thấy B-T thật sự phân biệt được mẫu sắc nét với mẫu trung bình. RQ4 (CFG/ngữ nghĩa) có khả năng tăng B-T nhưng dễ làm xấu DTW. RQ5 (metric tự nhiên với nguồn gần dữ liệu) là tùy chọn.
7. **Rủi ro lớn nhất:** số công bố của SignNMFlow có thể không tái lập được trong evaluator của ta (hand JPE 5.57 so với mức sàn ~7.0–7.5 mà mọi cấu hình của ta đang đạt). Vì thế tuyên bố chính phải là so sánh matched với bản replicate; so với số công bố là tuyên bố phụ.

---

## 1. Câu hỏi nghiên cứu của review và phương pháp

**Câu hỏi.** Với cùng dữ liệu SMPL-X và giao thức SOKE trên Phoenix-2014T, CSL-Daily, How2Sign, thay đổi nào về đường đi/objective/sampler, biểu diễn motion và text conditioning cho phép vượt SignNMFlow trên DTW-JPE, DTW-PA-JPE và B-T cùng lúc, với một GPU 48 GB và tối đa hai process?

**Hình thức.** Integrative review (tác giả chọn). Không phải systematic review; không có PRISMA, không có hai người sàng lọc độc lập.

### 1.1 Cách tìm

- Ba luồng tìm song song ngày 16/11/2026: (i) text-to-motion và biểu diễn motion; (ii) SLP và đánh giá SLP; (iii) flow matching/diffusion ngoài motion (ảnh, protein, robot, audio, phục hồi ảnh).
- Công cụ: tìm kiếm web tổng quát, arXiv API (kiểm tra tiêu đề/tác giả/ngày của 52 ID ở luồng i), mở trực tiếp trang arXiv abs/HTML, ACL Anthology, PMLR, chương trình BMVC 2026, README và issue trên GitHub, Hugging Face API. Không dùng Scopus/Web of Science/Semantic Scholar có cấu trúc.
- Truy vết trích dẫn từ SignNMFlow và từ ba bản tổng hợp cũ trong repo (`docs/proposal2–4`).
- Bằng chứng nội bộ: `methods/signmflow/README.md` (kết quả v1), log huấn luyện, `docs/proposal2/phoenix_pilot_results_20261107.md`.

### 1.2 Tiêu chí chọn / loại

| Chọn | Loại |
|---|---|
| Nguồn sơ cấp đã mở được trang (arXiv, CVF, ACL, PMLR, repo chính thức) | Blog, trang tổng hợp, bài chỉ thấy trong đoạn trích kết quả tìm kiếm |
| 2022 – 11/2026 (thêm các bài nền trước đó) | SLP chỉ 2D keypoint/video, trừ khi nói về metric hoặc độ dài |
| Có cơ chế chuyển được sang mô hình chuỗi pose ~170 chiều/frame trên một GPU, hoặc có bằng chứng về biểu diễn, objective, sampler, metric | Bài không xác minh được tồn tại: loại hoàn toàn, liệt kê ở mục 1.4 |

### 1.3 Độ sâu đọc

Khoảng 110 nguồn được xác minh tồn tại. Đọc phương pháp + thí nghiệm: ~45 (ví dụ MFM, JiT, PMRF, InDI, C²OT, PixelGen, SD3, Dispersive Loss, Contrastive FM, MARDM, ACMDM, SALAD, MotionStreamer, FlowMotion, SOKE, MaDiS, SignSparK, M3T, BLANC, SeRV, SRVQ, MV-Sign, Conditional Collapse). Phần còn lại chỉ đọc abstract; trong văn bản các nhận định dựa trên abstract được ghi "(abstract)".

### 1.4 Giới hạn độ bao phủ

- Một "người đọc" (AI), không có sàng lọc độc lập; có thể sót bài.
- Số liệu lấy qua công cụ đọc trang có tóm tắt; chưa đối chiếu bằng mắt với PDF.
- **Không xác minh được, nên không dùng làm bằng chứng:** nội dung paper SignFlow (chỉ đọc repo), SignFML (chỉ có tên trong chương trình BMVC 2026), Spectral OT, T2S-GPT/SignLLM/Sign-IDD (không mở), mô hình và dữ liệu huấn luyện back-translator của SOKE, một ablation có kiểm soát Euclid-vs-geodesic cho quaternion/SO(3), một bài text-to-motion đo trực tiếp "NFE thấp giúp metric khoảng cách".
- Không tìm thấy bài nào áp dụng PMRF hoặc khung perception–distortion cho SLP/text-to-motion. "Không tìm thấy" chỉ giới hạn trong các truy vấn đã chạy; không phải tuyên bố "đầu tiên".
- Một số trang dài bị cắt ở ~100k ký tự (MaDiS, M3T, BLANC, MV-Sign).

---

## 2. Tổng hợp theo chủ đề

### 2.1 SignNMFlow thực sự làm gì

- Trạng thái: 41 quaternion đơn vị/frame + 10 hệ số biểu cảm Euclid. Nguồn: đều trên bán cầu.
- Đường đi: `x_t = Exp_{x0}(t·Log_{x0}x1 + t(1−t)·ε_θ)`, `ε_θ` là MLP 3 lớp học trước bằng động năng dưới metric `G(x)` dựng từ 500 prototype K-means.
- Generator: transformer decoder 8 lớp, cross-attention tới mBART fine-tune, **dự đoán x1**, loss `10‖x1−x̂1‖² + 0.1‖e1−ê1‖²` trên quaternion.
- Sampler 5 bước: mỗi bước dự đoán lại `x̂1` rồi đặt trạng thái lên đường đi của cặp `(x0, x̂1)` tại `t+Δ`. Ở bước cuối `t+Δ = 1`, nên **đầu ra = `x̂1` dự đoán từ trạng thái tại t = 0.8** (đọc từ Alg. 2 và `methods/signmflow/flow.py:104-123`).
- Không có: EMA, CFG, chọn checkpoint, loss không gian khớp, mô tả cách chọn độ dài đầu ra.
- Bảng 8 của paper: NFE 5/10/20 gần như phẳng (16.86/16.91/16.98), sai số tăng nhẹ theo NFE; cùng chiều với quan sát của ta.

**So với Metric Flow Matching.** MFM: `x_t = (1−t)x0 + t·x1 + t(1−t)·φ(x0,x1,t)`, φ học trước bằng động năng dưới metric RBF từ tâm K-means, rồi đóng băng. SignNMFlow thay đoạn thẳng bằng geodesic trên (S³)ᴶ và dùng x1-prediction. MFM nói rõ họ cần không gian ambient Euclid, nên phần nâng lên mặt cầu là phần mới thật. Gain MFM báo cáo: −87% EMD trên vòm tổng hợp; −9% FID trên AFHQ latent; −7 đến −8% W1 trên single-cell 100 chiều. Tất cả là vận chuyển không điều kiện giữa hai phân bố có cấu trúc. Với nguồn đều trên bán cầu, phần lớn đường đi nằm xa mọi prototype, nơi kernel RBF gần như phẳng. Suy luận của tôi: MFM không cho lý do kỳ vọng gain 16% như Bảng 3 của SignNMFlow; đây là điều RQ1 phải kiểm tra, không phải kết luận.

### 2.2 Metric: cái gì đang được đo

| Quan sát | Nguồn | Độ mạnh |
|---|---|---|
| Pose trung bình tĩnh cạnh tranh trên DTW-JPE; DTW không tách được mô hình 2 epoch với mô hình 300 epoch; B-T tách được (10.17 vs 0.05) | Nội bộ, v1 Phoenix dev | 1 seed, 100–519 clip |
| Đầu ra "semantic-only" mượt thắng metric hình học, mô hình đầy đủ thắng retrieval R@1 (82.3 vs 77.8, CSL) | SRVQ (Liu et al., 2026) | 1 paper |
| DTW "may over-optimistically reflect model performance"; bỏ cột PA | MaDiS (Zuo et al., 2026) | nhận định tác giả |
| Sinh không keyframe "predicts near-average poses" | Tác giả SignSparK, issue #5 | issue |
| Trên How2Sign, mọi checkpoint gần như không phụ thuộc text; checkpoint FID tốt nhất là checkpoint collapse nhất | Hong & Košecká (2026) | 1 paper |
| DTW đã tinh chỉnh tương quan với người tốt nhất (ρ = 0.76), B-T BLEU-4 chỉ 0.21 | Jiang et al. (2025, WMT) | 2D keypoint, DSGS/LSF/LIS |
| B-T BLEU tương quan ≈ 0 với mức hiểu của người trên bộ corruption | Cory et al. (2026) | 1 paper, đọc một phần |

Hai nhóm bằng chứng mâu thuẫn bề ngoài: DTW "thưởng trung bình" nhưng lại tương quan tốt với người trong Jiang et al. Cách hòa giải hợp lý: DTW phân biệt tốt các lỗi thô giữa hệ thống khác nhau, nhưng trong vùng mà mọi mô hình đều gần trung bình thì nó không còn phân biệt. Hệ quả: **không metric đơn lẻ nào đủ**; phải báo đồng thời DTW, B-T (nhiều evaluator) và một chỉ số phụ thuộc text.

**Giao thức không ổn định giữa các paper** (bằng chứng từ code/issue): code SOKE ghi DTW-JPE căn theo root dưới tên "PA" (issue #6); DTW là chi phí tích lũy không chuẩn hóa độ dài (issue #7 SignSparK); checkpoint SignSparK công bố cho CSL body JPE ~8.0–8.5 thay vì 6.27; back-translator của SOKE không công khai nên câu "our Back-Translation model is the same as SOKE" trong SignNMFlow không kiểm chứng được; thang B-T lệch nhau nhiều lần giữa các evaluator (SOKE trên How2Sign: 14.48 với mô hình của họ, 2.70 với mô hình của SeRV).

**Vị trí thật của SignNMFlow.** Không tìm thấy bài nào thắng mọi cột trong cùng giao thức, nhưng bảng của họ chọn lọc: không trích MaDiS (tốt hơn trên How2Sign DTW-JPE: 6.59/9.45 so với 6.76/9.72), chỉ lấy hàng không-keyframe của SignSparK; SeRV tốt hơn nữa trên How2Sign (6.44/9.16). Mọi số đều tự báo cáo.

### 2.3 Perception–distortion và sampler ít bước

- Blau & Michaeli (2018): giảm distortion buộc phân bố đầu ra lệch khỏi phân bố thật. Freirich et al. (2021): với MSE và W2, bộ ước lượng tối ưu ở mọi điểm trên đường cong là nội suy giữa kỳ vọng hậu nghiệm và bộ ước lượng "phân bố hoàn hảo".
- PMRF (Ohayon et al., 2025): hồi quy kỳ vọng hậu nghiệm, rồi chạy rectified flow từ *kỳ vọng + nhiễu nhỏ* tới phân bố dữ liệu; đạt MSE nhỏ nhất dưới ràng buộc phân bố đúng.
- InDI (Delbracio & Milanfar, 2023): sampler "dự đoán lại đích" giống SignNMFlow; 1 bước cho PSNR tốt nhất, chất lượng cảm nhận chỉ tăng khi thêm bước.
- Text-to-motion: FlowMotion (x1-prediction, không gian thô) có FID 1.528 ở 5 bước so với 0.278 ở 100 bước; DisCoRD: giải mã ngẫu nhiên làm MPJPE tăng (29.5→33.3 mm) nhưng FID giảm (0.019→0.011); R-precision phẳng hoặc tốt nhất ở 1 bước trong MotionLCM/MotionPCM. Phản chứng: hồi quy thuần sụp đổ về phân bố trong MotionStreamer (FID 59.3) và mô hình tất định của TEMOS có sai số vị trí *xấu hơn* mô hình biến phân.

Suy ra (suy luận của tôi, các nguồn hỗ trợ chứ không phát biểu trực tiếp): tại `t = 0`, `x0` độc lập với `x1`, nên `x̂1` tối ưu MSE là `E[x1 | text]`. NFE 1 thắng trên DTW là hành vi kỳ vọng; pose trung bình tĩnh là cùng hiệu ứng ở mức không điều kiện. Số bước là vị trí trên đường cong perception–distortion, không phải núm chất lượng.

### 2.4 Biểu diễn và loss

- Tái tạo tốt hơn không kéo theo sinh tốt hơn (5 nguồn độc lập: MoMask, MotionStreamer, SnapMoGen, MoLingo, MARDM). ACMDM: không gian thô và latent causal-AE gần hòa (FID 0.121 vs 0.109).
- Bằng chứng nội bộ chống lại codec rời rạc: pilot temporal-plan, codec chỉ giữ 17.6% B-T của dữ liệu thật.
- Xoay hay vị trí: chưa ngã ngũ. Jin & Haworth (2025): dưới x0-prediction 6D ≈ vị trí, quaternion kém nhất (FID 59.2), nhưng trên 1.2k clip, 1 seed, và không có flow nhận biết đa tạp. Cùng bài: thêm loss hình học 42.9→27.9 FID.
- Loss không gian khớp có bằng chứng ở nhiều nơi: MDM (abstract), SALAD (loss vị trí khớp ở VAE: MPJPE 0.023→0.016), DART (loss phụ giảm jerk 3.67→2.45), MotionLCM (giám sát không gian khớp giảm sai số quỹ đạo nhưng tăng FID 0.319→0.419).
- Mô hình toàn thân/tay (OmniMotion-X, HandX, HandMDM, Kimodo) đều dùng x0-prediction; không bài nào ablate.
- Mục tiêu dự đoán tương tác với cách điều kiện: ACMDM, x0/ε/v = 0.969/0.419/0.208 với concat nhưng 0.133/0.125/0.121 với AdaLN. JiT (Li & He, 2025): x-prediction thắng áp đảo ở 768 chiều/token, nhưng ở 48 chiều/token mọi tổ hợp đều chạy được. Với ~174 chiều/frame, lợi thế của x-prediction hợp lý nhưng chưa được xác lập.

### 2.5 Điều kiện text và ngữ nghĩa

- T5 + cross-attention theo token thắng CLIP gộp (SnapMoGen, MotionStreamer, MoLingo, StableMoFusion). SignNMFlow đã có cross-attention tới mBART.
- CFG đổi đa dạng lấy khớp text, FID hình chữ U (SALAD, Hu et al. 2024, RAM, ACMDM). Thang chọn 2.5–7.5 trên HumanML3D và 15 cho motion tay từ dữ liệu ký hiệu (HandMDM). **Chưa nguồn nào đo CFG so với metric khoảng cách có cặp.**
- Trong SLP: loss phụ sign-to-text (M3T) tăng BLEU-4 tương đối 8.9% / 22.4% / 4.1%; tokenizer căn với gloss (SRVQ) tăng R@1 75.6→82.3; pretraining (MaDiS, CSL) 6.93/9.27→5.99/7.96; retrieval từ điển (SOKE) giảm body PA ~20–24%. Huấn luyện đa ngôn ngữ gần như không giúp SignNMFlow (How2Sign 6.76→6.74).
- Loss trên mẫu sạch dự đoán: PixelGen thêm LPIPS + DINO trên `x̂` , FID 23.67→7.46, recall giảm và được khôi phục một phần khi tắt loss ở 30% mức nhiễu cao nhất. Chỉ có trên ảnh.
- REPA và họ hàng: chỉ kiểm chứng ở quy mô ImageNet; LUMA (abstract) cho thấy teacher contrastive tự huấn luyện trong miền là đủ cho motion.

### 2.6 Những thứ rẻ nhưng có bằng chứng

- Lấy mẫu thời gian logit-normal: hạng trung bình 1.25 ở 5 bước trong SD3, so với 5.67 của phân bố đều. JiT dịch về phía nhiễu.
- EMA: MotionLCM 0.467 vs 0.505; EDM2 post-hoc EMA (abstract).
- Dừng sớm/chọn checkpoint: thời điểm bắt đầu thuộc lòng tỉ lệ với cỡ dữ liệu (Bonnaire et al., abstract). Nội bộ: val loss thấp nhất ở epoch ~160 (0.0916), 0.0983 ở epoch 300.
- Cỡ mô hình: lớn hơn không giúp trên dữ liệu nhỏ (MotionCraft, HandX, HY-Motion; ACMDM khi không có CFG).
- Self-conditioning (Chen et al., abstract): tự nhiên với x-prediction.

### 2.7 Những thứ bị loại và vì sao

| Hướng | Lý do loại |
|---|---|
| Minibatch OT coupling | Làm xấu mô hình có điều kiện (ImageNet caption FID 3.48→5.36); C²OT chỉ gỡ hòa (Cheng & Schwing, 2025) |
| MeanFlow / consistency / shortcut | Giải bài toán chi phí; ở đây 1 bước đã rẻ và đã trả về kỳ vọng |
| Latent AE / token rời rạc | Bằng chứng trái chiều; pilot nội bộ mất B-T; thêm một giai đoạn huấn luyện |
| Mô hình lớn hơn, LLM text encoder | Không giúp trên 7k–31k clip; DTW phẳng khi đổi mBART→Qwen3 trong bảng MaDiS |
| Huấn luyện đa ngôn ngữ | Gần như không đổi với SignNMFlow-M |
| Retrieval từ điển, keyframe | Đổi track so sánh (không còn text-only); gain của MV-Sign không đáng kể |
| Dispersive / Contrastive FM | Phụ thuộc batch lớn, nhạy λ, chỉ có trên ImageNet; để dự phòng cho RQ4 |

### 2.8 Khoảng trống

1. Chưa ai quy gain của SignNMFlow về từng thành phần dưới evaluator độc lập.
2. Chưa ai trong SLP báo đường cong distortion–perception; các bảng trộn mô hình ở các điểm khác nhau trên đường cong.
3. Chưa có kết quả loss FK cho flow x1-prediction trên đa tạp quaternion trong SLP.
4. Tác động của CFG lên metric có cặp chưa được đo.

---

## 3. Proposal: PMA-Flow

### 3.1 Giả thuyết trung tâm

Trong chế độ dữ liệu nhỏ và text yếu của SLP, chất lượng trên DTW do **độ chính xác của kỳ vọng có điều kiện** quyết định, còn chất lượng trên B-T do **độ sắc nét và mức phụ thuộc text** quyết định. Một mô hình tách hai việc này (hồi quy kỳ vọng trong không gian khớp, rồi flow ngắn từ kỳ vọng tới dữ liệu) sẽ trội Pareto so với một flow từ nhiễu đều.

### 3.2 Thành phần

**A. Neo kỳ vọng (Stage A).** Mạng `μ_ψ(c, T) ∈ (S³)^{J×T}`, cùng backbone 8 lớp, đầu vào là positional query thay cho trạng thái nhiễu. Loss:

```
L_A = λ_q · Σ_j ‖μ^j − x1^j‖²                       (chordal, như SignNMFlow)
    + λ_p · Huber( FK(μ) − FK(x1) )                   (khớp thân căn root; khớp tay căn cổ tay)
    + λ_v · Huber( Δ_t FK(μ) − Δ_t FK(x1) )           (khớp vận tốc với đích, không phạt vận tốc tuyệt đối)
    + λ_e · ‖e − e1‖²
```

FK dùng chuỗi động học với offset khớp ở shape trung tính (rẻ, không cần mesh). Huber thay L2 vì DTW-JPE đo chuẩn L2 không bình phương theo khớp và MotionLCM báo Huber tốt hơn L2. Đây là "huấn luyện đúng cái được đo".

**B. Flow dư trên đa tạp (Stage B).** Nguồn `x0 = Exp_μ(σ0·ξ)`, `ξ` Gauss trong không gian tiếp tuyến, `σ0` nhỏ (quét 0.1–0.3 rad). Đường đi: geodesic `x_t = Exp_{x0}(t·Log_{x0}x1)`; vì `x0` gần `x1`, Log luôn xa cut locus và đường đi nằm trong vùng có dữ liệu. Generator `ϕ(x_t, t, c, μ)` giữ x1-prediction + chiếu radial, cùng loss như Stage A. `μ` đóng băng khi huấn luyện B. Sampler: Euler dạng vận tốc suy từ `x̂1` (cho mẫu thật) và sampler "dự đoán lại" của SignNMFlow (cho kỳ vọng ở bước cuối), cả hai đều báo cáo.

**C. Núm α.** Đầu ra `y_α = Exp_μ(α·Log_μ(x_sample))`, `α ∈ [0, 1]`. `α = 0` là kỳ vọng (distortion nhỏ nhất), `α = 1` là mẫu. Hàng trong bảng chính dùng **một** α chọn trên dev; đường cong đầy đủ báo riêng.

**D. Tăng phụ thuộc text (bật sau, từng cái một).**
- CFG: bỏ text 10% khi huấn luyện; `x̂ = x̂_u + w(x̂_c − x̂_u)` rồi chiếu radial; quét `w ∈ {1, 1.5, 2.5, 4}`; thử khoảng guidance (Kynkäänniemi et al., 2024).
- Loss ngữ nghĩa trên `x̂1` ở các mức nhiễu thấp (kiểu PixelGen): cosine tới đặc trưng của một encoder ký hiệu đóng băng. **Teacher phải khác evaluator**: huấn luyện teacher contrastive text–pose trên train split; chấm bằng back-translator mBART của repo và SiCLIP của MaDiS.

**E. Vệ sinh huấn luyện (áp cho mọi biến thể, kể cả baseline matched).** EMA; chọn checkpoint theo val; lấy mẫu `t` logit-normal; lr riêng thấp hơn hoặc đóng băng mBART trong các epoch đầu (v1 ghi nhận đặc trưng text sụp ở epoch 2–4); self-conditioning tùy chọn.

**F. Tùy chọn.** Hiệu chỉnh "naturalness" `ε_θ` trên đường đi của Stage B (RQ5).

### 3.3 Khác gì so với tiền lệ

| Thành phần | Tiền lệ | Phần còn lại để tuyên bố |
|---|---|---|
| x1-prediction trên (S³)ᴶ | SignNMFlow | Không tuyên bố |
| Nguồn = kỳ vọng + nhiễu | PMRF (ảnh, Euclid) | Chuyển sang đa tạp xoay và sinh có điều kiện text |
| Loss FK | MDM, SALAD, DART | Không tuyên bố mới; là thành phần kỹ thuật |
| Núm distortion–perception | Freirich et al. (lý thuyết) | Giao thức báo cáo đường cong cho SLP |
| CFG, loss đặc trưng | Phổ biến; PixelGen | Đo tác động lên metric có cặp trong SLP |

Tính mới ở mức vừa phải. Giá trị khoa học chính là **phân tích quy kết + giao thức đánh giá theo đường cong**, không phải một kiến trúc mới.

### 3.4 Ràng buộc repo

- Phương pháp mới tạo bằng `python scripts/new_method.py pma_flow`, chỉ có `fit/predict`.
- Không import từ `methods/signmflow`. Hình học S³ (exp/log/project) và FK khả vi dùng chung phải chuyển vào `src/slp/` kèm test.
- Metric mới (bootstrap có cặp, tỉ lệ phụ thuộc text, SiCLIP) chỉ thêm vào `src/slp/evaluation/metrics.py`, mỗi cái một unit test.
- Mọi lần chạy qua `scripts/run.py`, split `val`.

**Hiện thực (2026-11-16):** `methods/pma_flow` (README liệt kê các lựa chọn chưa có thí nghiệm), dùng chung `src/slp/geometry.py` và `src/slp/skeleton_fk.py`; script P1/P2 ở `configs/experiments/pma_flow/`. Chưa hiện thực: loss ngữ nghĩa (D, phần 2) và `ε_θ` trong Stage B (F).

---

## 4. Research questions, giả thuyết và tiêu chí bác bỏ

Đơn vị quyết định: Phoenix dev (519 clip), seed 0, bootstrap có cặp theo clip, khoảng tin cậy 95%. Ngưỡng có ý nghĩa thực tế đặt trước: **≥ 0.20 DTW-JPE** hoặc **≥ 1.0 B-T BLEU-4**, khoảng tin cậy không chứa 0. Cấu hình thắng cuối cùng mới chạy 3 seed.

| RQ | Câu hỏi | Giả thuyết | Bác bỏ khi |
|---|---|---|---|
| **RQ1** | Điểm của SignNMFlow đến từ đâu: hồi quy về kỳ vọng, đường đi đa tạp, hay text? | (a) NFE 1 ≤ NFE 5 trên DTW; (b) geodesic và naturalness chênh nhau < 0.20 JPE dưới evaluator của ta; (c) xáo text làm DTW đổi < 0.20 nhưng B-T sụp | Nếu (b) sai: naturalness cho gain ≥ 0.20 → giữ `ε_θ` làm mặc định và ưu tiên RQ5 |
| **RQ2** | Loss FK + vệ sinh huấn luyện có cải thiện DTW mà không giảm B-T, ở cùng NFE? | Giảm ≥ 0.20 body JPE và ≥ 0.20 hand JPE so với replicate matched; B-T không giảm quá 0.5 | Không đạt ngưỡng trên cả body lẫn hand → bỏ FK loss, chỉ giữ phần vệ sinh nếu nó tự có ích |
| **RQ3** | Nguồn neo kỳ vọng có trội Pareto so với nguồn đều không? | Tồn tại một α mà cả DTW và B-T đều tốt hơn mô hình RQ2 (nguồn đều) | Đường cong α nằm dưới hoặc trùng điểm của mô hình RQ2 → báo kết quả âm, giữ Stage A làm mô hình |
| **RQ4** | CFG và loss ngữ nghĩa có tăng phụ thuộc text không, và giá phải trả trên DTW là bao nhiêu? | B-T tăng ≥ 1.0 ở `w > 1`; DTW xấu đi; loss ngữ nghĩa tăng B-T trên evaluator *không* dùng làm teacher | B-T không tăng trên evaluator độc lập → gain là do khớp teacher, bỏ |
| **RQ5** (tùy chọn) | Metric tự nhiên có ích hơn khi nguồn gần dữ liệu không? | Gain của `ε_θ` trong Stage B lớn hơn gain của nó với nguồn đều | Không khác → bỏ thành phần, đơn giản hóa phương pháp |

---

## 5. Kế hoạch và ngân sách (một GPU, tối đa hai process)

Mốc thời gian từ log v1: Phoenix, generator 300 epoch ≈ 10,5 giờ khi chạy một mình; CSL ≈ 2,6×, How2Sign ≈ 4,4× theo số clip. Ước lượng dưới đây giả định hai process chạy song song chậm đi khoảng 1,5×; chưa đo.

| Pha | Việc | Chi phí ước lượng | Cổng |
|---|---|---|---|
| **P0** chỉ đánh giá, không huấn luyện | Trên checkpoint v1/v2 Phoenix: B-T và DTW ở NFE 1/5/20; dự đoán với text xáo trộn và text rỗng; đường cong nội suy giữa NFE 1 và NFE 20 | Vài giờ | **G1:** B-T ở NFE 5 hoặc 20 cao hơn NFE 1 ≥ 1.0? Nếu không, B-T không thưởng độ sắc nét → bỏ RQ3, dồn sức cho RQ2 + RQ4 |
| **P0b** hiệu chuẩn evaluator | Chạy checkpoint công khai của MaDiS qua metric của repo, so với số họ công bố (Phoenix 4.97/6.59) | Vài giờ; **cần test split → chỉ chạy khi bạn cho phép** | **G0:** khớp → khoảng cách tới SignNMFlow nằm ở mô hình; lệch → nằm ở giao thức, và không thể so trực tiếp với số công bố |
| **P1** Phoenix | RQ1: geodesic vs naturalness vs linear (3 run, đã có trong hàng đợi `jobs.py` cho CSL; đề nghị chạy trên Phoenix cho rẻ). RQ2: +vệ sinh, +FK loss (2 run). Stage A thuần (1 run) | 6 run ≈ 3–4 ngày | **G2:** RQ2 đạt ngưỡng? |
| **P2** Phoenix | RQ3: Stage B với 2 giá trị σ0 (2 run). RQ4: CFG (1 run, quét w khi suy luận), loss ngữ nghĩa (teacher + 1 run) | 4–5 run ≈ 3 ngày | **G3:** có điểm trội Pareto? |
| **P3** mở rộng | Cấu hình thắng + baseline matched trên CSL và How2Sign | ≈ 6–8 ngày | — |
| **P4** xác nhận | 3 seed trên Phoenix; test split **một lần, khi bạn yêu cầu** | ≈ 2 ngày | — |

Tổng: khoảng 2,5–3 tuần GPU nếu mọi cổng đều qua; dừng sớm sau P1 (≈ 4 ngày) nếu RQ2 thất bại.

**Báo cáo.** Bảng A: so matched (replicate SignNMFlow vs PMA-Flow, cùng dữ liệu, text encoder, cỡ mô hình, độ dài dự đoán, cùng evaluator). Bảng B: so với số công bố, ghi rõ khác biệt giao thức. Luôn kèm hàng "pose trung bình tĩnh", "GT của câu khác", và B-T của ground truth làm trần.

---

## 6. Review phản biện

### 6.1 Từng research question có đáng chạy không

Xác suất dưới đây là phán đoán chủ quan của tôi, không phải ước lượng thống kê.

| RQ | Đáng chạy? | Xác suất đạt giả thuyết | Lập luận phản biện mạnh nhất | Kết luận |
|---|---|---|---|---|
| RQ1 | **Có, chạy đầu tiên** | (a) ~85%, (b) ~55%, (c) ~60% | Kết quả phụ thuộc bản replicate; nếu replicate sai ở đâu đó thì quy kết cũng sai. v1 có hai lỗi đã sửa ở v2 | Rẻ nhất, quyết định hướng đi; dù ra sao cũng là phát hiện dùng được. Phải chạy trên v2 |
| RQ2 | **Có** | ~65% cho DTW; ~40% rằng B-T không giảm | FK loss có thể chỉ đẩy mô hình về kỳ vọng mạnh hơn: DTW tốt hơn, motion mượt hơn và khó đọc hơn. MotionLCM thấy giám sát không gian khớp làm FID xấu đi | Đáng chạy vì trực tiếp nhắm vào metric; bắt buộc báo kèm B-T và jitter |
| RQ3 | **Có điều kiện (sau G1)** | ~40% có điểm trội Pareto | Phần dư `x1 − μ` chủ yếu là phong cách người ký và thời điểm, không phải ngữ nghĩa; thêm nó có thể không tăng B-T. Chi phí gấp đôi huấn luyện. PMRF chỉ được kiểm chứng trên phục hồi ảnh, nơi điều kiện rất mạnh | Ý tưởng mới nhất nhưng rủi ro nhất; không chạy nếu G1 thất bại |
| RQ4 | **Có, nhưng tách riêng** | CFG tăng B-T: ~60%; không hại DTW: ~20%. Loss ngữ nghĩa tăng B-T độc lập: ~35% | CFG ngoại suy ra xa kỳ vọng nên gần như chắc chắn làm DTW xấu đi. Loss ngữ nghĩa dễ thành "tối ưu theo evaluator"; với 7k clip, teacher contrastive có thể quá yếu | CFG gần như miễn phí, nên chạy. Loss ngữ nghĩa chỉ chạy khi còn ngân sách |
| RQ5 | Tùy chọn | ~25% | MFM cho gain nhỏ ngoài dữ liệu tổng hợp; thêm một giai đoạn và siêu tham số | Chỉ chạy nếu RQ1(b) bị bác bỏ |

### 6.2 Các điểm yếu của toàn bộ proposal

1. **Có thể không vượt được số công bố.** Hand JPE 5.57 của paper thấp hơn ~25% so với mọi thứ evaluator của ta từng cho ra, kể cả pose trung bình (7.51) và NFE 1 (7.06). Hoặc mô hình của họ tốt hơn nhiều so với replicate, hoặc giao thức khác (độ dài ground-truth lúc suy luận, tập khớp, chuẩn hóa DTW). Không có code để phân xử. Xác suất chủ quan vượt số công bố trên *mọi* cột của cả ba bộ: dưới 30%. Vượt bản replicate matched trên DTW: ~60%.
2. **"Thắng DTW bằng cách trung bình hóa" là thắng một metric hỏng.** Nếu α tối ưu trên dev gần 0, mô hình tốt nhất theo bảng là một bộ hồi quy mượt. Đó là kết quả trung thực nhưng không phải SLP tốt hơn. Bài báo phải nói thẳng điều này và không được tuyên bố "dễ hiểu hơn" khi thiếu đánh giá của người ký.
3. **B-T cũng không đáng tin.** Tương quan với người thấp (Jiang et al.; Cory et al.), phụ thuộc evaluator. Trần ground-truth của back-translator trong repo trên Phoenix dev chỉ 11.49. Cần ít nhất hai evaluator độc lập và chỉ số phụ thuộc text.
4. **Một seed, 519 clip.** Chênh lệch 0.2 JPE có thể nằm trong nhiễu giữa các seed; bootstrap theo clip không bắt được phương sai do seed. Chỉ kết quả 3 seed ở P4 mới được dùng để tuyên bố.
5. **Rủi ro lặp lại vết xe cũ.** Proposal 4 (CPR-Flow) ước 31–44 ngày GPU và chưa huấn luyện trên dữ liệu thật. Kế hoạch này cố ý đặt P0 không tốn huấn luyện và cổng dừng sau ~4 ngày.
6. **Tính mới vừa phải.** PMRF, FK loss, CFG đều có sẵn. Nếu RQ3 thất bại, phần còn lại là một nghiên cứu phân tích/tái lập có giá trị, chưa chắc đủ cho hội nghị hạng cao như một bài phương pháp.
7. **Giả định chưa kiểm chứng trong thiết kế.** FK với shape trung tính khác FK của metric (dùng regressor khớp của SMPL-X); kỳ vọng trên (S³)ᴶ dưới loss chordal là trung bình chiếu, không phải trung bình Fréchet; `σ0` quá nhỏ làm flow suy biến thành hàm đồng nhất, quá lớn thì mất lợi thế nguồn gần.
8. **Bằng chứng đã chọn có thiên lệch không?** Đã chủ động tìm phản chứng: Jiang et al. (DTW tương quan tốt với người), TEMOS và MotionStreamer (hồi quy thua), tác giả SOKE khuyên dùng DTW thay B-T. Chúng không lật giả thuyết trung tâm nhưng giới hạn nó: hồi quy thuần có thể sụp ở chỉ số phân bố, nên Stage A một mình không phải đích đến.

### 6.3 Đạo đức và phạm vi tuyên bố

- Không có đánh giá của cộng đồng Deaf; mọi tuyên bố giới hạn ở metric tự động trên fitting SMPL-X.
- Không tuyên bố "first" cho bất kỳ thành phần nào.
- Số của các baseline là tự báo cáo, giao thức không đồng nhất; ghi rõ trong mọi bảng.
- Dữ liệu và mã SOKE có giấy phép CC BY-NC-ND; không phát hành bản sửa đổi.

---

## 7. Khuyến nghị

1. Chờ v2 Phoenix xong, chạy **P0** (chỉ đánh giá) để có G1.
2. Xin phép bạn cho **P0b** nếu muốn biết sớm liệu số công bố có so được hay không (cần test split).
3. Chạy **RQ1 + RQ2 trên Phoenix**; quyết định RQ3/RQ4 theo cổng.
4. Không đầu tư vào OT coupling, one-step, latent/token, mô hình lớn hơn, đa ngôn ngữ.

---

## Tài liệu tham khảo (đã mở được trang; số arXiv)

**SLP và đánh giá.** He et al., SignNMFlow, 2609.33339 · Zuo et al., SOKE, ICCV 2025, 2411.17799 · Zuo et al., MaDiS, 2601.19577 · Low et al., SignSparK, 2603.10446 · Symeonidis-Herzig et al., M3T, BMVC 2026, 2603.23617 · Kim et al., BLANC, 2610.24109 · Wu et al., SeRV, 2609.05742 · Liu et al., SRVQ, 2611.06781 · Muk et al., MV-Sign, 2611.07811 · Baltatzis et al., Neural Sign Actors, CVPR 2024, 2312.02702 · Hong & Košecká, Conditional Collapse in SLP, 2606.01643 · Cory et al., BackTranslation2.0, 2606.28673 · Jiang et al., Meaningful Pose-Based Sign Language Evaluation, WMT 2025, 2510.07453 · Bensabath et al., HandMDM, 2508.15902.

**Flow matching / diffusion.** Lipman et al., 2210.02747 · Liu et al., Rectified Flow, 2209.03003 · Chen & Lipman, Riemannian FM, ICLR 2024, 2302.03660 · Kapuśniak et al., Metric Flow Matching, 2405.14780 · Cheng & Schwing, C²OT, ICCV 2025, 2503.10636 · Li & He, JiT, 2511.13720 · Esser et al., SD3, 2403.03206 · Kingma & Gao, 2303.00848 · Karras et al., EDM2, 2312.02696 · Ho & Salimans, CFG, 2207.12598 · Kynkäänniemi et al., guidance interval, NeurIPS 2024, 2404.07724 · Blau & Michaeli, CVPR 2018, 1711.06077 · Freirich et al., 2107.02555 · Ohayon et al., PMRF, ICLR 2025, 2410.00418 · Delbracio & Milanfar, InDI, TMLR 2023, 2303.11435 · Ma et al., PixelGen, 2602.02493 · Yu et al., REPA, ICLR 2025, 2410.06940 · Jia et al., LUMA, 2509.25304 · Wang & He, Dispersive Loss, 2506.09027 · Stoica et al., Contrastive FM, 2506.05350 · Chen et al., self-conditioning, ICLR 2023, 2208.04202 · Bonnaire et al., NeurIPS 2025, 2505.17638 · Geng et al., MeanFlow, 2505.13447.

**Text-to-motion và biểu diễn.** Tevet et al., MDM, 2209.14916 · Guo et al., MoMask, 2312.00063 · Guo et al., SnapMoGen, 2507.09122 · Dai et al., MotionLCM, ECCV 2024, 2404.19759 · Meng et al., MARDM, CVPR 2025, 2411.16575 · Meng et al., ACMDM, 2505.19377 · Xiao et al., MotionStreamer, ICCV 2025, 2503.15451 · Zhao et al., DART, 2410.05260 · Canales Cuba et al., FlowMotion, 2504.01338 · Hong et al., SALAD, CVPR 2025, 2503.13836 · He et al., MoLingo, CVPR 2026, 2512.13840 · Cho et al., DisCoRD, 2411.19527 · Huang et al., StableMoFusion, 2405.05691 · Hu et al., 2405.02791 · Jiang et al., MotionPCM, 2501.19083 · Liu et al., RAM, 2601.14788 · Jin & Haworth, 2512.04499 · Petrovich et al., TEMOS, ECCV 2022, 2204.14109 · Bian et al., MotionCraft, 2407.21136 · Zhang et al., HandX, CVPR 2026, 2603.28766 · Xu et al., OmniMotion-X, 2510.19789 · Wen et al., HY-Motion, 2512.23464 · Rempe et al., Kimodo, 2603.15546.

Tên hội nghị chỉ ghi khi trang arXiv/PMLR/chương trình hội nghị nêu rõ.

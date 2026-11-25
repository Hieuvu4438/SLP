# 02 — Literature review: SLP, MotionHiFlow, flow matching, biểu diễn chuyển động

**Loại review:** narrative / integrative, phục vụ thiết kế phương pháp. Đây không phải
systematic review. Tra cứu web ngày 2026-11-23 qua arXiv, CVF open access, ACL Anthology,
OpenReview và GitHub, chia theo 5 nhánh: SLP, MotionHiFlow và hierarchy, flow matching, biểu diễn
chuyển động, TTS và ngôn ngữ học ký hiệu. Khoảng 300 paper được rà soát.

Mỗi mã arXiv trong tài liệu đã được xác minh là tồn tại qua arXiv API hoặc trang paper. Các con số
lấy từ bảng hoặc văn bản của paper. Thẻ **[abs]** nghĩa là chỉ kiểm tra ở mức abstract. Danh mục
đầy đủ ở [references.md](references.md). Những gì **không tìm thấy** được ghi rõ ở cuối mỗi mục.

Tài liệu này cố ý **không** dựa vào các proposal hay method khác trong repo, theo yêu cầu của người
dùng. Chỉ hai kết quả đo trong repo được dùng làm bằng chứng nền: SignNMFlow reimpl và MotionHiFlow-SLP.

---

## 1. Sign Language Production (text → 3D)

### 1.1 Bảng so sánh theo protocol SOKE (test split)

Mỗi ô theo thứ tự: DTW-JPE body / hand, DTW-PA-JPE body / hand, B-T BLEU-4. "–" nghĩa là không báo
cáo. Hai dòng cuối dùng pseudo-GT tự fit lại nên **không so được** trực tiếp với các dòng khác.

| Method | Repr. / generator | Phoenix-2014T | CSL-Daily | How2Sign |
|---|---|---|---|---|
| Progressive Transformers (ECCV'20, 2004.14874)¹ | keypoints, AR | 15.01/31.77/13.67/11.95/4.94 | 16.30/32.63/15.98/12.91/3.07 | 14.74/30.17/14.15/11.57/2.75 |
| T2S-GPT (ACL'24, 2406.07119)¹ | dynamic VQ + GPT | 11.65/19.09/10.38/6.47/9.06 | 12.32/15.43/11.94/5.93/8.94 | 12.65/18.44/11.48/6.39/11.20 |
| S-MotionGPT¹ (MotionGPT do SOKE train lại) | VQ + LM | 10.42/9.08/9.45/3.41/9.68 | 11.58/11.31/10.81/3.78/8.82 | 12.41/13.74/11.23/4.39/11.45 |
| SignFlow (ICMI'25)² | CFM + OT | 5.40/6.21/4.56/1.20/– | 7.56/8.82/6.43/1.65/– | 7.98/10.52/6.92/2.27/– |
| SOKE (ICCV'25, 2411.17799) | 3 part-VQ + mBART AR + dictionary retrieval | 6.04/7.72/4.77/1.38/11.87 | 7.38/9.68/6.24/1.71/11.30 | 7.75/10.08/6.82/2.35/14.48 |
| GARD (2606.21234) | gloss-wise AR diffusion (**cần gloss**) | 6.69/6.90/5.03/1.21/13.16 | 7.73/9.32/6.98/1.56/12.73 | – |
| SignSparK-GF (ECCV'26, 2603.10446) | CFM sparse keyframes, ~1.4B/stream | 4.40/7.10/5.24/1.52/– | 6.27/10.63/7.26/2.00/– | **6.30**/11.43/7.26/2.72/– |
| **SignSparK-SR** (retrieval qua text→gloss→dictionary) | như trên | **3.51/4.82/3.68**/1.18/– | **4.89/6.56/4.87**/1.37/– | – |
| MaDiS (2601.19577) | masked-diffusion LM (Qwen3-0.6B) trên token SOKE | 4.97/6.59/–/–/– | 5.99/7.96/–/–/– | 6.59/9.45/–/–/– |
| SeRV (2609.05742) | RVQ gióng với text + hierarchical GPT | – | – | 6.44/**9.16**/–/–/3.67³ |
| SRVQ (WACV'27, 2611.06781) | semantic → kinematic RVQ (**cần gloss** để train quantizer) | 5.33/6.92/6.00/1.35 | 6.79/8.54/7.61/1.54 | – |
| **SignNMFlow** (2609.33339) | manifold FM trên $(S^3)^{41}$, NFE 5 | 4.37/5.57/3.85/**1.07**/16.15 | 5.73/7.43/5.10/**1.30**/15.14 | 6.76/9.72/6.01/**2.07**/16.88 |
| M3T⁴ (2603.23617) | FLAME + FSQ, AR mBART + SLT aux | 4.02/8.55/3.41/1.46/10.58 | 4.63/9.28/3.97/1.51/5.80 | 4.86/9.44/4.28/2.28/1.72 |
| BLANC⁴ (2610.24109) | multi-rate VQ (body ×4, hands ×2), block diffusion | 7.33/7.00/7.58/1.48/9.85 | 9.54/8.25/9.75/1.71/2.13 | – |

Chú thích:

1. Số do SOKE tái lập.
2. Lấy từ bảng của SignNMFlow; paper SignFlow là closed access và chỉ báo cáo CSL, nên nguồn không
   kiểm chứng được.
3. BLEU chấm bằng back-translator Uni-Sign; GT đạt 8.2 với cùng model.
4. Chấm trên pseudo-GT tự fit lại, không so được.

**Đọc bảng.**

- **SignNMFlow không phải SOTA tuyệt đối.**
  - SignSparK-SR vượt nó ở 3/4 cột DTW trên Phoenix và CSL, nhưng cần text→gloss.
  - Trên How2Sign, MaDiS, SeRV và SignSparK-GF (body) đều tốt hơn ở DTW-JPE.
  - Điểm mạnh riêng của SignNMFlow là **PA-hand** (1.07 / 1.30 / 2.07), tức hình dạng bàn tay, và
    **B-T**. B-T của nó dùng back-translator của SOKE, mô hình này không công khai.
- **Ai đứng đầu phụ thuộc vào protocol.** Các bản tái lập SOKE lệch nhau tới khoảng 0.4 DTW
  (SRVQ tái lập SOKE: Phoenix 5.91/7.32, so với 6.04/7.72 SOKE tự báo cáo). Chênh lệch dưới khoảng
  0.3 có thể chỉ là nhiễu. Không paper nào báo cáo nhiều seed hay khoảng tin cậy.

### 1.2 Những gì đã thực sự làm thay đổi con số (bằng chứng từ ablation)

1. **Prior lexical (retrieval / dictionary).** Đây là đòn bẩy lớn nhất.
   - SOKE: retrieval giảm DTW trung bình 19.9 / 19.6 / 23.9% (H2S / CSL / Ph).
   - SignSparK GF → SR trên Phoenix: PA-body 5.24 → 3.68, JPE-hand 7.10 → 4.82.
   - Người thắng SLRTP-2025 (2508.06951) cũng dùng retrieval.
2. **Tách bộ phận và có cơ chế hợp nhất.**
   - SOKE 1 → 3 VQ: Phoenix PA 8.03/3.23 → 4.77/1.38.
   - MaDiS dense mixture-of-parts: −0.36/−0.54 trên CSL.
3. **Pretraining nhiều dataset hoặc có physical loss.**
   - MaDiS: hand 9.27 → 8.39 → 7.96 sau fine-tune.
   - SOKE đa ngôn ngữ tốt hơn đơn ngôn ngữ (Phoenix avg 3.83 → 2.13).
   - Phản ví dụ: SignNMFlow-M (đa ngôn ngữ, *không* fine-tune riêng) kém SignNMFlow trên Ph và CSL.
4. **Coarse-to-fine.**
   - RobotSTAR (2609.32250): next-scale −40% PA-hand.
   - MaDiS temporal checkpoints: −0.62/−1.09.
   - SRVQ, SeRV: các tầng residual.
5. **Mô hình thời lượng tường minh.**
   - T2S-GPT: bỏ duration transformer làm B4 tụt 11.87 → 9.39.
   - BLANC: duration clock thích nghi tốt hơn block cố định.
6. **Chống exposure bias.** RobotSTAR self-conditioning: PA-hand 1.85 → 1.54.
7. **Target-prediction thay vì velocity.** SignNMFlow Table 5, CFM baseline trên CSL: All 21.62 → 20.08.
8. **Regularizer ngữ nghĩa** (SLT auxiliary loss, code đối sánh với text) làm **tăng BLEU/R@1 nhưng
   có thể làm tăng DTW.**
   - M3T: How2Sign hand 8.94 → 9.44 khi thêm SLT loss.
   - SRVQ: bản chỉ có semantic đạt DTW tốt hơn bản đầy đủ.
   - Đây là đánh đổi thật, và nó là động cơ trực tiếp cho thiết kế Estimate-then-Transport (05-T2).

### 1.3 Đánh giá: metric và back-translation

**DTW-JPE là chi phí tích luỹ, không chuẩn hoá.**

- Code SOKE `mGPT/metrics/dtw.py` trả về `D1[-1,-1]`; appendix của SeRV xác nhận; repo này cũng cài
  như vậy.
- Hệ quả:
  - Output ngắn hơn làm DTW giảm. SRVQ §4.4: cắt ngắn output đưa DTW 6.787 → 6.534 trong khi R@1
    rơi 82 → 43.
  - Motion đóng băng hoặc motion trung bình được thưởng. RobotSTAR: NAR và T2S-GPT có DTW *thấp
    hơn* SOKE dù Freeze-tail = 1.0.
  - Trong repo, một pose trung bình tĩnh đạt 4.70 / 7.51 trên Phoenix dev.
- MaDiS (5 signer CSL) báo cáo: DTW tương quan yếu với đánh giá của người, trong khi SiBLEU và SiCLIP
  tương quan mạnh.
- "Meaningful Pose-Based SL Evaluation" (2510.07453): DTW đã tinh chỉnh (DTWpp) đạt Spearman 0.76 với
  người bản ngữ, ngang mức đồng thuận giữa người với người (0.77). Back-translation likelihood chỉ
  đạt 0.42.

**Back-translation không chuẩn hoá giữa các paper.**

- Back-translator của SOKE không công khai; SignNMFlow nói dùng "same as SOKE".
- Trần BLEU-4 của GT dao động 3.37 – 19.45 tuỳ evaluator.
- Dưới các evaluator SMPL-X khác, SOKE rơi từ 11.87 xuống 7.05 (TwoStream) hoặc 7.90 (MSKA) trên Phoenix.
- Công khai: SignSparK_BT (trên pose tự fit lại), MaDiS SiCLIP, Uni-Sign (pose-only, có checkpoint
  H2S), MSKA (checkpoint Ph và CSL, input 2D HRNet).

**Đánh giá với người dùng.** Các nghiên cứu dùng n = 4–16 signer, chủ yếu chấm điểm hoặc chọn bắt
buộc. Chưa có nghiên cứu nào kiểm tra mức hiểu.

### 1.4 Ghi chú từng nhóm phương pháp

- **SignNMFlow.**
  - Thiết kế: geodesic trên $(S^3)^J$ cộng một độ lệch học được. Độ lệch này tối thiểu hoá kinetic
    energy dưới một "naturalness measure" dựng từ K-means prototype (K=500 cho mỗi khớp).
  - Generator: target prediction, transformer decoder 8 lớp kèm cross-attn tới mBART, NFE 5, mỗi
    bước sampler dự đoán lại $\hat x_1$.
  - Bằng chứng ablation (CSL, All): MR không giúp (20.08 → 21.25), GE giúp (→18.31), NG giúp thêm
    (→16.86).
  - Điểm yếu (reader-inferred):
    1. Không có code; paper không nói cách chọn thời lượng.
    2. Bản reimpl trong repo chỉ đạt Phoenix test 4.93/6.54/4.35(ua)/1.30, so với 4.37/5.57/3.85/1.07
       trong paper. Paper không tiết lộ đủ chi tiết để tìm ra khoảng chênh.
    3. Cột "All" trong ablation không tái tạo được theo định nghĩa thông thường.
    4. Baseline 6D trong Table 3 không có FK loss, nên mức chênh 6D → quaternion có thể bị phóng đại.
- **SOKE.** Decoupled tokenizer: 96 code cho body, 192 cho mỗi tay. mBART đa đầu, cộng sign token
  lấy từ dictionary. Hạn chế: dừng khi *bất kỳ* đầu nào phát EOS; dictionary cần độ phủ từ → ký hiệu.
- **SignSparK.** Keyframe (onset / mid / offset) lấy từ bộ segmenter FAST. CFM có mask
  $C=M\odot x_1+(1-M)\odot x_t$, loss CFM : recon : velocity = 2 : 1 : 1, khoảng 10 bước. Mục tiêu
  được nêu rõ là chống regression-to-mean. Mỗi stream một UNet riêng, nên đồng bộ body–tay không
  được mô hình hoá chung.
- **MaDiS.** Bidirectional masked diffusion với 3 tầng pretraining (token / latent / physical), mở
  dần từ thô tới mịn. Phát hành code, checkpoint và evaluator SiCLIP. Không báo cáo PA, với lý do
  "PA có thể phóng đại hiệu năng".
- **GARD, SRVQ.** Mạnh nhưng **cần gloss** (theo đúng gloss hoặc để giám sát quantizer). Vì vậy
  không cùng hạng mục gloss-free với SOKE và SignNMFlow.
- **PIDiffSign (2607.14836)**, trên keypoint, không theo protocol SOKE:
  - FK head (bone direction + length) tốt hơn output toạ độ: WER 93.47 → 88.70.
  - Thay ngón tay bằng GT giúp nhiều hơn thay cánh tay bằng GT, tức tay mang nhiều nghĩa hơn.
  - Loss gia tốc Euclid dồn 93% gradient vào các khớp ngón.
  - 21% frame tay trong GT Phoenix bị phẳng theo chiều sâu, nghĩa là pseudo-GT có giới hạn.

**Không tìm thấy:** SignMotionGPT, M2S, Sign-Stitcher.

---

## 2. MotionHiFlow và sinh nhiều độ phân giải

### 2.1 MotionHiFlow (2604.23264, CVPR 2026), đọc đầy đủ cả paper và supplementary

**Động cơ.** Tác giả lấy GT HumanML3D, hạ mẫu xuống 0.2× mà R-precision vẫn ổn định. Kết luận của họ:
ngữ nghĩa nằm ở scale thô, chi tiết nằm ở scale mịn.

**Công thức.** $K$ stage; stage $k$ có scale $r_k$ và chiếm khoảng $[t_{k-1},t_k]$ trên trục thời gian
flow. $f$ là resampling tuyến tính.

- Trạng thái đầu stage $k$ (Eq. 4): $(1-t_{k-1})f(x_0,r_k)+t_{k-1}f(f(x_1,r_{k-1}),r_k/r_{k-1})$.
- Trạng thái cuối (Eq. 5): $(1-t_k)f(x_0,r_k)+t_kf(x_1,r_k)$.
- Mọi stage dùng cùng nhiễu $x_0$ đã resample ("noise consistency").
- Inference: denoise → upsample → renoise, bằng một ODE tất định.

**Cấu hình.**

- VAE: GCN + TCN, hạ thời gian 4×, gộp 22 khớp thành 6 latent joint, mỗi joint d=16. Có loss "aug":
  decode của latent đã resample phải khớp motion đã resample.
- TMDiT: 3 double-stream + 6 single-stream block, width 384, CLIP, joint RoPE.
- Scales {1/3, 2/3, 1}, CFG 4.5, 12 bước.

**Kết quả HumanML3D.** R@1 0.563, FID 0.032.

- Paper tuyên bố R-precision cao nhất, nhưng chính bảng của nó cho thấy SALAD cao hơn (0.581).
- FID ngang MoGenTS trong khoảng tin cậy.

**Ablation (supplementary).**

| Thay đổi | Kết quả |
|---|---|
| Scales: [1] → [1/2, 1] → [1/3, 2/3, 1] → 4 stage | FID .051 → .038 → .032 → .035 |
| Kiến trúc: baseline → +TMDiT → +topology VAE | FID .074 → .045 → .032, R@1 .511 → .557 → .563. **Lợi ích từ kiến trúc lớn hơn lợi ích từ hierarchy** |
| Cách chuyển tiếp: upsample-only → upsample + renoise → denoise-upsample-renoise | FID .304 → .057 → **.032** |
| Số double-stream block (Xd = số block double-stream, Ys = single-stream) | 3d-6s tốt nhất |
| Steps | bão hoà quanh 12 bước |

**Những gì không được ablate:** pool loss, half-diff, $\gamma$, cách phân bổ bước giữa các stage.

**Hạn chế (reader-inferred).**

1. Lợi ích của hierarchy nhỏ và chỉ được đo bằng evaluator T2M, vốn bị MARDM chỉ ra là thiên lệch.
2. Scale cố định, không phụ thuộc độ dài chuỗi, không có SNR shift.
3. **Nhiễu ở stage thô không trắng.** Agent khảo sát chạy `interpolate` (half-diff, linear) lên nhiễu
   trắng và đo được phương sai ≈0.64 / 2.9 ở $r=0.3$. Số này cần chạy lại để xác nhận.
4. Chỉ có body, 22 khớp. Không có hand.

### 2.2 Sinh nhiều độ phân giải và coarse-to-fine: bài học

- **Cần một cơ chế chống lỗi tích luỹ.**
  - Cascaded DM (2106.15282): conditioning augmentation là "crucial".
  - Pyramid Flow (2410.05954): renoise đúng cách là bắt buộc.
  - MoScale (2604.03799): thêm bước refinement, FID .176 → .046.
- **Lợi ích lớn khi baseline yếu, nhỏ khi baseline mạnh.**
  - GUESS: .485 → .109. sMDM: .544 → .130.
  - MotionHiFlow, vốn đã mạnh ở một scale: .051 → .032.
- **Trục và thứ tự của hierarchy quan trọng.** ScaleMoGen (2605.11704): thứ tự temporal-first cho
  FID .021, xen kẽ cho .105.
- **Thêm tầng không phải lúc nào cũng tốt hơn.** MoMask V > 5, MotionHiFlow K = 4 và GUESS 5 scale
  đều kém đi.
- **Hierarchy theo tần số thuần, không có nhiễu, kém hơn.** IHDM; còn Blurring Diffusion (2209.05557)
  chỉ cải thiện nhỏ.
- **Phân rã theo tần số hoặc tách từng scale riêng đã có cho ảnh và video:**
  - Decomposable Flow Matching (NeurIPS'25, 2506.19839): flow độc lập cho từng tầng Laplacian, nhanh
    hơn tới 50%.
  - DCT spectral FM, WaiT (2607.28760), FREPix, DeCo, LapFlow (2602.19461).
- **Tốc độ khác nhau theo articulator đã có tiền lệ trong SLP:** BLANC, hand gấp 2 body. Ý tưởng này
  không còn mới.

**Không tìm thấy:** "MotionPyramid"; công trình nào làm anti-aliasing khi hạ mẫu thời gian trong
motion tokenizer; công trình nào chứng minh renoise chính xác cho toán tử trực chuẩn với độ dài
thay đổi.

---

## 3. Text-to-motion: thiết kế latent, biểu diễn, mục tiêu

| Chủ đề | Bằng chứng chính |
|---|---|
| **Raw so với latent** | ACMDM (2505.19377): raw 0.121 / causal AE 0.109 / non-causal VAE 0.178 (v-pred). MLD (2212.04048): latent nhỏ nhất tốt nhất (1×256: .473; 10×256: 1.716). InterCMDM (2607.01743): nén 4× → 8× làm MPJPE tái tạo tăng 7×, FID gần như không đổi. **Metric độ chính xác (như DTW-JPE) bị lỗi codec giới hạn từ dưới.** |
| **Latent có cấu trúc** | SALAD (2503.13836): latent skeleto-temporal FID .433 → .076. MotionHiFlow: d=16 tốt hơn d=8. Latent ít chiều, nhiều token, có cấu trúc thắng latent lớn. |
| **Biểu diễn** | MARDM (2411.16575): bỏ kênh dư thừa, FID nội bộ 2.196 → 0.116. ACMDM: toạ độ tuyệt đối. HY-Motion (2512.23464): 6D + vị trí, bỏ vận tốc và contact. |
| **Mục tiêu dự đoán** | MARDM: MDM với ε sụp (31.27 cosine) so với x0 0.518. ACMDM: x0 / ε / v = .133 / .125 / .121 (chênh nhỏ). FlowMotion (2504.01338): x1-pred, 1 bước FID 2.5 so với 100 bước 0.28. **Đổi target một mình không phải đòn bẩy lớn trong motion, nhưng x-pred cần cho FK loss.** |
| **SNR / noise scale** | MotionSpaceFlow (2609.34190, không dùng VAE): source noise $s=5$ đưa FID .111 → .046. SD3 / simple diffusion: cần dịch SNR theo kích thước. |
| **Kiến trúc và text** | SALAD: bỏ cross-attn thì FID .274. MotionHiFlow: word token cải thiện MM-Dist 3.043 → 2.849. HY-Motion: Qwen3-8B + refiner, chất lượng bão hoà quanh 0.46B tham số. PRISM: RoPE theo cây động học tốt hơn RoPE 2D. |
| **Retrieval cho T2M** | ReMoMask (2508.02605), RAG-T2M: cải thiện FID; mạnh nhất khi dữ liệu nhỏ. |

---

## 4. Flow matching: những tiến bộ liên quan

### 4.1 Mục tiêu, trọng số, SNR
- JiT (2511.13720): x-prediction cần thiết cho token raw nhiều chiều (768-D). Với token khoảng 48-D
  thì mọi target đều ổn. Token 60–90-D của chúng ta nằm ở vùng giữa.
- SD3 logit-normal (2403.03206): có hạng trung bình tốt nhất, kể cả khi chỉ dùng 5 bước.
- EDM / EDM2: weighting và preconditioning.

### 4.2 Perception–distortion, ước lượng rồi sinh, và decoding (nền của Estimate-then-Transport)

| Công trình | Nội dung |
|---|---|
| Blau & Michaeli (1711.06077) | Định lý: distortion và chất lượng perceptual xung đột. Sampler hoàn hảo chịu gấp đôi MSE của posterior mean. |
| **Freirich, Michaeli & Meir** (NeurIPS'21, 2107.02555) | Với distortion là MSE và perception là $W_2$, đường D–P có dạng đóng. Các ước lượng tối ưu nằm trên geodesic giữa posterior mean và một transport của nó tới $p_x$; MSE dưới ràng buộc perception hoàn hảo là MMSE + $W_2^2(p_\mu,p_x)$. |
| **PMRF** (2410.00418) | Dự đoán posterior mean, rồi chạy rectified flow **không có điều kiện** từ mean cộng nhiễu tới phân phối dữ liệu. Xấp xỉ ước lượng MSE nhỏ nhất dưới ràng buộc perceptual hoàn hảo. |
| **CorrDiff** (2309.15214), **StoRM** (2212.11851), **DiffSinger** (2105.02446) | Ước lượng hồi quy, rồi diffusion **có điều kiện** để sửa phần dư, hoặc xuất phát từ ước lượng cộng nhiễu. |
| **Grad-TTS** (2105.06337), **PriorGrad** (2106.06406) | Prior phụ thuộc điều kiện: hội tụ nhanh hơn, nhưng không đổi phân phối đích. |
| **MBR decoding** | Freitag (2111.09388), AMBR với góc nhìn medoid (2401.02749), MBR cho ASR (2510.19471), trajectory-MBR cho diffusion LM (NeurIPS'26). |
| Ensemble cho tác vụ hồi quy | D3DP + JPMA (2303.11579): gộp hypothesis theo từng khớp cho pose. GenCast (2312.15796): trung bình ensemble diffusion thắng trên RMSE. |
| soft-DTW (1703.01541), DILATE (1909.09020) | DTW khả vi; barycenter. |

**Không tìm thấy:** MBR với utility DTW cho sinh motion hay SLP; RL với reward là back-translation
cho SLP.

### 4.3 Few-step
- Shortcut models (2410.12557); MeanFlow (2505.13447); iMF (2512.02012): 1-NFE FID 1.72.
- SplitMeanFlow (2507.16884): không cần JVP, đã triển khai cho TTS 1–2 bước.
- Align Your Flow (2506.14603); IMM (2503.07565).
- Distributional diffusion với energy score (2502.02483): cho cả mẫu đã hiệu chỉnh lẫn trung bình của chúng.
- Motion: MotionLCM, MixiMotion (2609.23010): 1 bước đạt 0.835 so với 0.858 của teacher 50 bước.

### 4.4 Gióng biểu diễn và regularization
- REPA (2410.06940): huấn luyện SiT nhanh hơn 17.5×.
- REG (2507.01467): denoise thêm một token ngữ nghĩa toàn cục.
- Contrastive FM / ΔFM (2506.05350): FID 20.01 → 16.32, nhưng đẩy dự đoán ra xa trung bình, nên có
  rủi ro cho metric L2.
- Dispersive loss (2506.09027).
- A-DMA (2505.19595, TTS): CTC ở tầng 8 cộng gióng tầng 16 với HuBERT, WER 2.68 → 1.97. Gióng cả hai
  ở cùng một tầng thì kém hơn.

### 4.5 Guidance
- CFG (2207.12598). Guidance interval (2404.07724): ImageNet-512 FID 1.81 → 1.40.
- APG (2410.02416); CFG-Zero* (2503.18886); Autoguidance (2406.02507).
- Model-guidance (2502.12154); Guidance-free training (2501.15420).
- Lưu ý cho metric một reference: CFG với $w>1$ đẩy mẫu ra xa $\mathbb E[x\mid c]$. Nó cải thiện
  FID, không cải thiện DTW-JPE.

### 4.6 Post-training với reward
- Flow-GRPO (2505.05470); ReFL (2304.05977); DRaFT (2309.17400); Adjoint Matching (2409.08861);
  ORW-CFM-W2 (2502.06061).
- Cho motion:
  - HY-Motion: DPO rồi Flow-GRPO, reward từ TMR và vật lý.
  - EasyTune (2602.07967): +8.2% MM-Dist so với DRaFT-50.
- Rủi ro: tối ưu "lừa" model SLT dùng làm reward. Phải dùng evaluator khác với reward model.

### 4.7 Hình học
- Riemannian FM (2302.03660); FrameFlow (2310.05297).
- RMG (2603.15016): quaternion + Riemannian FM, nhưng không có đối chứng Euclid.
- PoseRieFlow (2611.09692): lợi về động học; metric phân phối và retrieval chỉ "tương đương".
- Kết luận: hình học cải thiện động học, không cải thiện ngữ nghĩa.

---

## 5. Biểu diễn chuyển động và bàn tay

- **Rotation.**
  - Zhou 2019 (1812.07035): 6D liên tục.
  - Inter-X++ (2608.20312): 6D cho text alignment tốt nhất (Top-1 .389). Quaternion cho FID tốt nhất
    (0.748). **Axis-angle kém** (FID 3.70).
  - vMDM (2512.04499): toạ độ khớp tốt hơn mọi biểu diễn rotation. Geometric loss: 42.86 → 27.87.
  - Brégier (2103.16317): rotation vector chấp nhận được khi góc nhỏ (ngón tay).
- **Tay.**
  - SOKE: 1 → 3 VQ, PA-hand H2S 3.56 → 2.35.
  - H2VQ (HumanTOMATO): hand MPJPE 46.45 → 37.20.
  - EMAGE: 1 → 4 VQ, FGD 9.79 → 7.40.
  - NSA: trọng số hand ×2, không ablate.
  - MV-Sign (2611.07811): fusion đa góc nhìn giảm 20.9% PA-hand, cho thấy pseudo-GT đơn góc làm mất
    chi tiết tay.
- **Tần số.**
  - FreqMo (2610.03012): biểu diễn trên lưới đều mất năng lượng trên khoảng 2 Hz. HF-RMSE .045 →
    .012 khi dùng wavelet.
  - Free-T2M (2501.18232): loss nhất quán tần thấp giúp ích; loss nhất quán tần cao làm hỏng huấn
    luyện (FID 11.58).
- **Độ mượt của latent trong SLP.** Nghiên cứu thiết kế VAE cho SLP (2606.22959): latent có vận tốc
  cao hơn đi kèm BLEU tốt hơn; latent "quá mượt" có hại.
- **Dữ liệu ngoài.**
  - SignAvatars (2310.20436): 70K video SMPL-X.
  - BOBSL3DT / HandMDM (2508.15902).
  - SignSparK_data (HF).
  - **Không tìm thấy** bằng chứng trực tiếp rằng pretraining bằng pose không nhãn giúp sinh ký hiệu 3D.

---

## 6. TTS: alignment, thời lượng, flow matching

Text → sign giống text → speech về cấu trúc: một chuỗi rời rạc ngắn ánh xạ sang một chuỗi liên tục
dài hơn nhiều, thời lượng không biết trước. Nó khác ở một điểm quan trọng: **trật tự ký hiệu khác
trật tự từ.**

- **Alignment tường minh**:
  - FastSpeech 2: dùng aligner ngoài.
  - Glow-TTS / VITS (MAS); Badlani 2108.10447 (forward-sum).
  - Matcha-TTS: MAS cộng decoder OT-CFM.
  - **Không dùng được trực tiếp cho SLP.** MAS và CTC giả định đơn điệu; giữa từ nói và ký hiệu thì
    không đơn điệu (Camgoz 2018).
- **Alignment ngầm**:
  - E2-TTS / F5-TTS (2406.18009 / 2410.06885).
  - Chỉ hoạt động ở quy mô 50k–100k giờ.
  - E2 hội tụ chậm hơn nhiều. F5 nếu bỏ ConvNeXt refinement thì "không học được alignment". E2 bản
    nhỏ hỏng trên khoảng 7% mẫu tiếng Trung.
  - MegaTTS 3 (2502.18924): sparse anchor tốt hơn alignment ngầm (WER 1.82 so với 2.14).
- **Thời lượng tổng luôn được dự đoán bên ngoài model chính.**
  - F5: theo tỉ lệ. E1 TTS (2409.09351): L1 trên tổng thời lượng tốt hơn loss theo âm vị.
  - MaskGCT: flow trên log thời lượng. DMOSpeech 2: phân loại kết hợp GRPO. DiTTo: dự đoán độ dài còn lại.
- **Hai giai đoạn.** CosyVoice (2407.05407): LLM sinh semantic token có giám sát, rồi CFM. Token có giám
  sát tốt hơn token không giám sát về độ nhất quán nội dung. Liên hệ: μ-net và Track R của SignHiFlow.
- **Sway sampling** (F5): không cần train lại; WER 2.84 → 2.41.

---

## 7. Ngôn ngữ học ký hiệu: từ tham số âm vị học tới đại lượng FK

| Tham số | Lý thuyết | Đại lượng FK (SMPL-X) |
|---|---|---|
| Handshape | Stokoe; ASL-LEX 2.0: 58 giá trị handshape | 15 góc khớp ngón; vị trí đầu ngón trong khung cổ tay; độ gập và độ xoè; khoảng cách ngón cái tới các ngón |
| Location / contact | | Vị trí cổ tay và đầu ngón trong khung đầu và khung ngực; khoảng cách tới mặt, thân, tay kia |
| Orientation | | Pháp tuyến lòng bàn tay = (MCP trỏ − cổ tay) × (MCP út − cổ tay), tính trong khung thân |
| Movement | Liddell & Johnson: chuỗi Hold–Move–Hold | Quỹ đạo cổ tay, profile tốc độ (hold ↔ cực tiểu vận tốc), lặp lại |
| Sign type | Battison: điều kiện đối xứng và điều kiện trội | Tương quan tốc độ hai tay, đối xứng gương, độ giống handshape |
| Non-manual | Mouthing: >80% phát ngôn DGS (nguồn thứ cấp) | jaw + expression; gật đầu |

**Bằng chứng thực nghiệm.**

- **Handshape là tham số khó nhất**, nhưng không phải lúc nào cũng vậy:
  - Người học L2: handshape > movement > orientation > location (Ortega & Morgan 2015).
  - Sem-Lex (2310.00196): handshape là đặc trưng yếu thứ hai (77.4%), sau path movement (75.6%).
- **Thời lượng** (Wilbur 2009, ASL): 1.55 / 1.95 / 2.43 ký hiệu/s ở tốc độ chậm / thường / nhanh;
  mỗi ký hiệu trung bình 0.66 / 0.51 / 0.42 s. Thay đổi tốc độ làm thay đổi **thời lượng từng ký
  hiệu, không thay đổi số ký hiệu**.
- **Co-articulation.**
  - Handshape kế tiếp bắt đầu hình thành trước khi ký hiệu hiện tại kết thúc. Nghiên cứu EMA trên
    LSF (Mertz et al.).
  - Co-articulation trong ký hiệu yếu hơn trong tiếng nói (Neidle). Nó mạnh nhất ở fingerspelling.
- **Miệng.** Trong TwoStream ablation trên Phoenix, thêm 10 keypoint miệng làm WER giảm 36.88 → 28.20.

---

## 8. Tổng hợp: các khoảng trống mà SignHiFlow nhắm tới

| # | Khoảng trống | Bằng chứng | SignHiFlow trả lời bằng |
|---|---|---|---|
| G1 | Các flow và diffusion SLP **sample** toàn bộ motion, nhưng metric một-reference lại thưởng cho **ước lượng**. Lĩnh vực so sánh các phương pháp ở những điểm khác nhau trên một frontier mà không nói ra. | Blau & Michaeli; Freirich et al.; NFE1 > NFE5 trong repo; SRVQ; RobotSTAR | **Estimate-then-Transport**: μ-net + T-net không nhận text + $\Lambda$ (Mệnh đề 1); giao thức matched kinematics và frontier |
| G2 | Hierarchy của MotionHiFlow chỉ được dùng như sampler từ nhiễu, với toán tử resampling không trực chuẩn | Đo nhiễu không trắng; lợi ích nhỏ (FID .051 → .032) | Hierarchy dùng làm **transport** có ràng buộc band trên DCT trực chuẩn; cross-fitting |
| G3 | Loss trên rotation không khớp metric vị trí. Hình dạng bàn tay là điểm phân biệt chính. | PIDiffSign, vMDM; SignNMFlow mạnh nhất ở PA-hand | **PES**: FK theo đúng khung căn chỉnh của metric, tách theo tham số âm vị học |
| G4 | Tín hiệu ngữ nghĩa (B-T) và DTW kéo ngược nhau; ngữ nghĩa thường được thêm bằng loss toàn cục, và evaluator thường bị coupling với những gì được tối ưu. | M3T, SRVQ; BT của SOKE không công khai | Chọn điểm vận hành tường minh bằng $\Lambda$; dùng evaluator ngữ nghĩa bên ngoài (SiCLIP) không tham gia train; L-REPA được hoãn lại |
| G5 | Retrieval là đòn bẩy lớn nhất nhưng các bản tốt nhất cần gloss. | SignSparK-SR, SOKE | **Track R**: retrieval chỉ từ text, đưa vào plan |
| G6 | Thời lượng bị bỏ ngỏ trong các flow SLP, trong khi metric phụ thuộc thời lượng. | SignSparK, NSA, SignNMFlow không mô tả cách chọn độ dài | Duration head Laplace; báo cáo cả nDTW lẫn DTW tích luỹ |
| G7 | Đánh giá yếu: BT không chuẩn hoá, DTW dễ bị lợi dụng. | §1.3 | Protocol 06: nDTW, độ động, SiCLIP, DTWpp, nhiều BT kèm trần GT, ≥3 seed, bootstrap CI |

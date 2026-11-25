# 06 — Kế hoạch thực nghiệm (v2)

Tài liệu này chỉ là **kế hoạch**: không chạy gì. Người dùng tự chạy, trên một GPU dùng chung, mỗi lần
một tiến trình.

Mỗi thí nghiệm có các mục: **mục đích, vì sao nên hoặc không nên chạy, cấu hình, chi phí, cách đọc kết
quả và gate**. Code cần viết nằm ở 08.

Quy tắc của repo (CLAUDE.md, `docs/experiment_protocol.md`):

- Mọi tuning làm trên `--split val`.
- `--split test` **chỉ** chạy ở E7, khi người dùng yêu cầu, từ một tree sạch, và được ghi log.
- Báo cáo ≥ 3 seed; bảng chính dùng 5 seed.
- Baseline được tuning với cùng ngân sách.

---

## 1. Protocol đánh giá

### 1.1 Metric

Mọi metric nằm trong `src/slp/evaluation`. Metric mới phải có unit test.

| Nhóm | Metric | Trạng thái |
|---|---|---|
| So sánh với paper | `soke_dtw_jpe_{body,hand}`, `soke_dtw_pa_jpe_{body,body_ua,hand}`: DTW tích luỹ | đã có |
| Kiểm soát độ dài | `soke_ndtw_*`: chia tổng chi phí cho độ dài path; cộng sai số thời lượng $\overline{\lvert\hat T-T\rvert/T}$ | **mới** |
| Độ động | `soke_speed_ratio_*`, `soke_accel_ratio_*` | đã có |
| Độ động, chi tiết | speed/accel ratio **riêng tay trái, tay phải, mặt**, và **trong khung cổ tay**; median theo chuỗi; tỉ lệ chuỗi có ratio < 0.5 | **mới** (cần cho matched kinematics) |
| Phân nhóm | báo cáo riêng hai nhóm: text **đã thấy** (trùng hoặc gần trùng train) và **chưa thấy** | **mới** (script chia nhóm) |
| Theo band | DTW-JPE tính trên phần low-pass ($\Pi_{T/4}$) của cả output lẫn GT, và tỉ lệ năng lượng dải cao so với GT, cho từng articulator | **mới** (chẩn đoán) |
| **Ngữ nghĩa chính (tách khỏi train)** | SiCLIP score và R@k (MaDiS, công khai) | **mới**: wrapper |
| Phân phối | FD trên embedding motion của SiCLIP (`frechet_distance` đã có) | **mới**: wrapper |
| Ngữ nghĩa phụ | B-T BLEU-4 với back-translator làm lại trong repo (Phase 0), có báo cáo trần GT; tuỳ chọn dùng thêm Uni-Sign hoặc MSKA qua chiếu 2D | một phần |
| Gần với người đánh giá | DTWpp (`pose-evaluation`, 2510.07453); đây là metric **tự động**, không thay thế người | **mới** (tuỳ chọn) |
| Thống kê | mean ± std theo seed; `paired_bootstrap_ci` theo clip | đã có |

### 1.2 Giao thức matched kinematics

Giao thức này được đăng ký trước, để không ai lợi dụng được metric.

- **Mọi** phương pháp, kể cả baseline, được báo cáo ở cùng hai điểm vận hành:
  1. **Matched kinematics (điểm chính).** **Speed** ratio của body, và speed ratio của tay trong khung cổ
     tay, đều bằng 1.00 ± 0.05 so với GT. Accel ratio chỉ báo cáo, không ràng buộc.
  2. **Native.** Cấu hình mặc định của từng phương pháp.
- **Mọi phương pháp dùng chung một quy trình:**
  - hai tham số $(\lambda_B,\lambda_H)$, với $\lambda_F=\lambda_B$ và $\lambda_L=\lambda_R$;
  - slerp theo từng khớp;
  - grid search trên $[0,1]^2$, bước 0.05;
  - quy tắc **không có nghiệm**: báo cáo $\Lambda=I$ kèm cờ, không ngoại suy (04 §7).
- **Cách áp vào baseline:** dùng shrink-to-mean $X=\mu_{\rm base}+\Lambda(S-\mu_{\rm base})$, trong đó
  $\mu_{\rm base}$ là output 1-NFE và $S$ là mẫu nhiều bước của chính baseline đó.
- **Cách áp vào SignHiFlow:** thay $S$ bằng $G$.
- **Đường cong:** báo cáo frontier (nDTW-hand theo SiCLIP, và nDTW-hand theo FD) với 5 giá trị $\lambda$.
- **Không đọc kết quả test** trước khi đã chốt điểm vận hành trên dev.

### 1.3 Tiêu chí "cải thiện thật"

Một thay đổi được giữ lại khi thoả cả bốn điều kiện trên dev:

- (i) ở matched kinematics, nDTW-hand hoặc PA-hand giảm, CI 95% không chứa 0;
- (ii) SiCLIP không giảm quá 1 điểm;
- (iii) FD không tăng quá 10%;
- (iv) sai số thời lượng không tăng.

Ablation rẻ dùng 1 seed kèm bootstrap; khẳng định chính dùng ≥ 3 seed.

---

## 2. E0: chẩn đoán, không cần train (chạy trước)

| Mã | Mục đích | Cách làm | Chi phí | Đọc kết quả / gate |
|---|---|---|---|---|
| **E0.1** | Trần codec của MotionHiFlow-SLP | Chấm `decode(encode(GT_dev))` bằng `scripts/score_predictions.py` | vài phút | PA-hand ≥ 0.4 hoặc JPE-hand ≥ 2.0 nghĩa là codec là nút thắt, và ủng hộ việc dùng không gian thô |
| **E0.2** | Đường cong perception–distortion trên model đã có | MotionHiFlow-SLP: steps ∈ {1, 3, 6, 12} × CFG ∈ {1, 1.5, 2, 3}. SignNMFlow-reimpl: NFE ∈ {1, 2, 5, 10}. Thêm shrink-to-mean với λ ∈ {0, 0.25, 0.5, 0.75, 1} | 1–2 giờ GPU | Hình 1 của paper. Xác nhận rằng mẫu "giống thật" hơn thì DTW tệ hơn |
| **E0.3** | Phân tích phổ theo articulator và mức độ dự đoán được | Phổ DCT theo băng (0–0.5, 0.5–1, 1–2, 2–3, 3–6, > 6 Hz) cho B, L, R, F trên GT train. Phần dư theo băng của output 1-NFE so với GT, tính **sau khi gióng bằng DTW** (để tách lệch nhịp khỏi tính không dự đoán được). Thêm độ nhất quán pha giữa dải thấp và dải cao | < 1 giờ CPU | Hình 2. Quyết định có bật grid riêng cho tay hay không, và cho giá trị khởi đầu của $\Lambda$ |
| **E0.4** | Thống kê góc quay | Phân bố $\lVert\omega\rVert$ theo khớp (sau khi cộng `hands_mean`); tỉ lệ khớp có $\lVert\omega\rVert>0.8\pi$ | vài phút | Quyết định giữ axis-angle làm ablation, hay dùng 6D là bắt buộc |
| **E0.5** | Evaluator | Wrapper SiCLIP và FD. Làm lại back-translator (CE + CTC gloss phụ trợ, kiến trúc khác mọi thứ dùng để train). Kiểm tra độ nhạy trên: GT, GT bị xáo, pose trung bình, GT đóng băng nửa sau, GT có chèn pose nghỉ | 4–8 giờ GPU mỗi dataset | SiCLIP phải tách rõ GT với GT bị xáo. Trần BT trên Phoenix ≥ 15 BLEU |
| **E0.6** | Baseline đơn giản và độ nhạy của metric | (a) pose trung bình; (b) **NN-retrieval**: chép motion của câu train gần nhất, resample về $\hat T$; (c) GT co hoặc giãn ±20%; (d) GT cắt còn 70%; (e) GT làm mượt σ = 2 và 4 frame; (f) GT chèn 20% pose nghỉ. Mỗi baseline đều chấm bằng DTW tích luỹ, nDTW, speed-ratio, SiCLIP và FD | chỉ chấm điểm | Bảng 4 của paper. (b) là baseline cần vượt; nếu (b) mạnh thì cần Track R |
| **E0.7** | MBR với cùng ngân sách | Hai model đã có, N = 8, chọn medoid theo nDTW. So với 1 mẫu và với shrink-to-mean | 8 lần inference | Nếu MBR không trội shrink-to-mean ở matched kinematics thì bỏ MBR khỏi mọi bảng |
| **E0.8** | **Gate không cần train cho ý tưởng cũ (E→S-v1, A20)** | Trên SignNMFlow-reimpl và MotionHiFlow-SLP: lấy mean 1-NFE, renoise tới $t^*$, rồi tích phân phần còn lại bằng sampler có text (kiểu SDEdit), có và không có MBR. So với shrink-to-mean ở matched kinematics, bằng SiCLIP và nDTW | vài giờ GPU | Nếu không cải thiện frontier, ta có bằng chứng thực nghiệm cho 05-T3 (anchor có text không giúp) và có thêm động cơ cho T-net. Nếu **có** cải thiện, nó là một baseline mạnh mà SignHiFlow phải vượt |

| **E0.9** | **Gelbrich headroom: có chỗ cho transport hay không?** | Lấy output 1-NFE của SignNMFlow-reimpl, hoặc μ out-of-fold sau E1, làm μ trên train. Tính $\Sigma_\mu$ (giữa các câu) và $\Sigma_n=\mathrm{Cov}(x-\mu)$, dùng shrinkage Ledoit–Wolf. Tính $\eta=W_2^2(\mathcal N(m,\Sigma_\mu),\mathcal N(m,\Sigma_\mu+\Sigma_n))/\mathrm{tr}\,\Sigma_n$ cho từng band DCT × articulator (05-T3) | vài giờ CPU | **η ≈ 1 ở mọi band thì không có headroom**: bỏ T-net, chuyển sang kế hoạch dự phòng (07 §4). Các band có η nhỏ cho biết transport có thể giúp ở đâu |

**Gate G0** (sau 1–2 ngày). Phải chốt được:

- biểu diễn (6D hay axis-angle);
- có bật grid riêng cho tay hay không;
- baseline cần vượt;
- evaluator dùng được;
- trạng thái của MBR và E→S-v1;
- **headroom η**, từ đó quyết định có làm T-net hay không.

---

## 3. E1: μ-net, bộ ước lượng (nền móng)

- **Mục đích:** đo xem một bộ hồi quy mạnh (MMDiT, mBART fine-tune, PES chính xác, 6D, duration head)
  đạt bao nhiêu khi dùng một mình ($\Lambda=0$).
- **Cấu hình:** Phoenix, 40k update, batch 64.
- **Ablation con** (1 seed + bootstrap):

| Mã | Thay đổi |
|---|---|
| E1-a | bỏ PES |
| E1-b | PES xấp xỉ (cây khớp, không blendshape) thay cho PES chính xác (A21) |
| E1-c | mBART đóng băng |
| E1-d | width 384 / 9 block |
| E1-e | axis-angle |
| E1-f | thêm soft-DTW (A9) |

- **Chi phí:** khoảng 6–10 giờ GPU mỗi run. Đây là ước tính: hãy đo throughput ở smoke test trước.
  μ-net không có vòng lặp flow nên nhanh hơn một flow model cùng cỡ.
- **Gate G1:** μ-net ($\Lambda=0$) phải tốt hơn mean 1-NFE của SignNMFlow-reimpl (cùng chế độ "mean")
  về JPE-hand và PA-hand trên dev. Nếu không đạt, sửa PES và text trước khi làm T-net.
- **Vì sao có thể không đáng chạy:** không có rủi ro thật. μ-net là một nửa của phương pháp, đồng thời
  là baseline "flat regressor" bắt buộc.

---

## 4. E2: T-net và frontier (đóng góp chính)

**E2-pilot, gate G1.5. Chạy trước mọi run E2 khác.**

- **Cấu hình:** chỉ stage 1 của T-net, Phoenix, khoảng 20k update, μ cross-fit.
- **Đo trên dev**, với $\sigma_1\in\{0.05,0.1,0.2,0.4\}$:
  - $\rho=\mathbb E\|G-\mu\|^2/\mathbb E\|Y-\mu\|^2$;
  - số hạng bias $\mathbb E\langle\hat\mu-\mu^\star,\mathbb E[G\mid c]-\hat\mu\rangle$, ước lượng bằng 8 lần rút ε mỗi câu và lấy $Y$ làm proxy cho $\mu^\star$ (05-T2);
  - x-pred loss trên dev ở mỗi snapshot, để phát hiện ghi nhớ.
- **Gate, đăng ký trước:** $\rho<0.8$ ở ít nhất một giá trị $\sigma_1$ mà vẫn giữ được SiCLIP của $G$ ≥ SiCLIP của μ.
- Không đạt nghĩa là T-net đang hành xử như sampler (05-T3). Khi đó không chạy T-net đầy đủ; chuyển sang 07 §4.
- **Chi phí:** khoảng 3–5 giờ GPU. Nó tránh được việc tiêu khoảng 2 ngày GPU vào Phase B khi kết quả có
  thể không có giá trị.

| Run | Mô tả |
|---|---|
| E2-0 | **PMRF phẳng**: T-net với K = 1, transport toàn dải từ μ + σε (A5) |
| E2-1 | **SignHiFlow mặc định**: T-net K = 3, ràng buộc band, không text, μ cross-fit |
| E2-2 | E2-1 dùng μ in-fold (A19) |
| E2-3 | E2-1 với T-net nhận text + CFG (A6) |
| E2-4 | E2-1 với K = 2 |
| E2-5 | E2-1 dùng pseudo-plan thay cho μ thật (A19) |
| E2-6 | E2-1 dùng anchor là output thật được cache thay cho GT + aug (A14) |

**Baseline quy thuộc, cùng ngân sách, cùng PES, cùng điều kiện pretrain:**

| Mã | Baseline |
|---|---|
| (i) | μ-net riêng, tức flat regressor |
| (ii) | **CorrDiff / StoRM style**: μ-net cộng một flow có text sinh phần dư $x-\mu$ từ nhiễu |
| (iii) | MotionHiFlow-SLP chuyển sang x-pred, có shrink-to-mean |
| (iv) | SignNMFlow-reimpl, có shrink-to-mean |
| (v) | E→S-v1 / SDEdit (từ E0.8) |

**Cách đọc kết quả.** Vẽ frontier (nDTW-hand theo SiCLIP, và theo FD) cho mọi phương pháp, rồi so sánh
ở điểm matched kinematics.

**Gate G2 (đăng ký trước).** Ở matched kinematics, E2-1 phải đạt cả ba điều kiện sau so với **tất cả**
baseline (i)–(v) và so với E2-0:

- nDTW-hand thấp hơn, CI không chứa 0;
- SiCLIP không thấp hơn quá 1 điểm;
- FD không cao hơn quá 10%.

Ngoài ra E2-1 phải tốt hơn E2-2 (cross-fitting có tác dụng) và E2-3 (T-net không nhận text có lý do).

- Nếu E2-1 không thắng E2-0, hierarchy không giúp gì cho transport. Khi đó vẫn có thể công bố
  Estimate-then-Transport phẳng, nhưng mất phần "dựa trên MotionHiFlow". Xem 07 §4.
- Nếu E2-1 không thắng (ii), novelty của "transport không text" so với CorrDiff không đứng vững.
  Khi đó chuyển sang kế hoạch dự phòng ở 07 §4.

**Chi phí:** Phase A gồm 1 model toàn phần và 2 model cross-fit, tổng khoảng 1.2× một μ-net. Thêm T-net
mỗi run khoảng 8–12 giờ. Mọi quét $\sigma_k$, $\Lambda$ và số bước đều chỉ là inference.

**Vì sao có thể không đáng chạy E2-4, E2-5, E2-6 khi hết thời gian:** kỳ vọng tác động thấp. Hãy ưu
tiên E2-0, E2-1, E2-2, E2-3 và các baseline (i)–(iv).

---

## 5. E3–E6

**E3: giám sát.**

- PES: bỏ từng thành phần; PES chính xác so với xấp xỉ (A21).
- $\Lambda$ chung so với $\Lambda$ theo articulator (A22).
- Soft-DTW (A9).
- Kỳ vọng: $\ell_{\rm shape}$ chủ yếu tác động PA-hand; $\ell_{\rm hand}$ chủ yếu tác động JPE-hand.
  $\Lambda$ theo articulator sẽ chọn $\lambda_B<\lambda_H$, vì body dự đoán được còn handshape cần sắc.
  Đây là **giả thuyết**, E0.3 và A22 sẽ kiểm tra.

**E4: inference.** $\sigma_1\in\{0.05,0.1,0.2,0.4\}$, $\sigma_{2,3}\in\{0.2,0.5,0.8,1.0\}$, steps ∈ {(2,2,2), (4,3,3), (8,6,6)}.
MBR (nếu E0.7 ủng hộ) chỉ dùng với N bằng nhau cho mọi phương pháp.

Thêm hai ablation:

- **A23:** slerp so với trộn tuyến tính trong 6D; có hoặc không có DTW-warp $G$ theo nhịp của $\hat\mu$.
  Phát hiện "bóng đôi" bằng tỉ lệ năng lượng dải cao và bằng quan sát định tính.
- **A24:** input của T-net lúc test là μ từ fold model, μ toàn phần, hay trung bình các fold.

**E5: Track R** (retrieval trong μ-net). Chỉ chạy nếu E0.6 cho thấy NN-retrieval mạnh, hoặc nếu sau E4
vẫn chưa đạt mục tiêu ở 03 §5. Báo cáo theo nhóm độ tương đồng retrieval.

**E6: đa ngôn ngữ, CSL và How2Sign.**

- A0 (pretrain μ-net đa ngôn ngữ) so với không pretrain (A13).
- T-net gộp 3 dataset (không cần text) so với train riêng từng dataset.
- How2Sign: miền mở, mọi model dễ sụp (Conditional Collapse, 2606.01643). Ưu tiên SiCLIP và FD.

---

## 6. E7: bảng cuối và đánh giá với người dùng

Chỉ chạy khi người dùng yêu cầu chạy test.

- **Bảng chính:** 3 dataset × 5 seed, gồm SignHiFlow (và SignHiFlow-R nếu có), các baseline (i)–(iv).
  Mọi phương pháp chạy test **một lần**, ở hai điểm vận hành đã chốt trên dev.
- **Tier A và Tier B** (03 §5).
  - Tier B so với số của SignNMFlow paper dùng điểm vận hành composite của ta, ghi rõ rằng điểm
    vận hành của SignNMFlow không được biết.
- **Nghiên cứu với người Điếc dùng ký hiệu.** Bắt buộc trước mọi khẳng định về articulation hay độ
  dễ hiểu.
  - Quy mô: ≥ 5 signer mỗi ngôn ngữ có thể tiếp cận (ưu tiên DGS hoặc CSL).
  - Nội dung: so sánh từng cặp (SignHiFlow so với SignNMFlow-reimpl, so với GT), cộng một câu hỏi hiểu
    nghĩa dạng chọn 1 trong 4 câu.
  - Tính tương quan Spearman giữa mỗi metric và đánh giá của người.

**Vì sao chọn mục tiêu ở 03 §5.**

- Mục tiêu không dùng retrieval chỉ đơn giản là vượt bảng của SignNMFlow.
- Tính khả thi dựa vào hai điểm:
  - (i) ở điểm "mean", bản reimpl đã cho thấy NFE 1 tốt hơn NFE 5 khoảng 0.5 body và 0.4 hand. μ-net
    có thêm PES chính xác nhắm thẳng vào JPE-hand và PA-hand.
  - (ii) Mệnh đề 1 (05) cho thấy ở mọi $\Lambda$, transport chịu rủi ro thấp hơn shrink-to-mean.
- Không có gì bảo đảm. G1 và G2 là các điểm kiểm tra.
- Mục tiêu khi có retrieval dựa trên mức giảm khoảng 32% của SignSparK-SR. Ta giả định chỉ đạt khoảng
  một nửa mức đó, vì không dùng gloss.

---

## 7. Thứ tự ưu tiên và cây quyết định

```
P0  E0.1, E0.4, E0.6 (chỉ chấm điểm)          ─┐
P0  E0.2, E0.7, E0.8 (inference, model đã có) ─┤ khoảng 1–2 ngày, không train method mới
P0  E0.3, E0.9 (Gelbrich headroom, CPU)       ─┘──► G0 (gồm: có headroom cho transport không?)
P1  E0.5 evaluator (SiCLIP, FD, back-translator làm lại, metric độ động chi tiết)
P2  E1 μ-net (+E1-a, b, c)                    ──► G1: μ-net tốt hơn mean của SignNMFlow-reimpl
P2b Phase A cross-fit + E2-pilot (stage 1, 20k update) ──► G1.5: ρ < 0.8
P3  E2-0, E2-1, E2-2, E2-3 + baseline (i)–(iv) ──► G2: trội frontier
P4  E4 quét inference (σ, steps, Λ)
P5  E3 ablation PES / Λ
P6  E6 đa ngôn ngữ + CSL + How2Sign
P7  E5 Track R (nếu cần)
P8  E7 test + user study (khi người dùng yêu cầu)
```

**Ước tính GPU cho Phoenix tới hết P5:** khoảng 8–12 run × 8–12 giờ, tức khoảng 4–6 ngày GPU. Phần
còn lại (P6) nhân 2–4 lần cho mỗi dataset.

**Cách giảm chi phí:**

- Ablation dùng 1 seed + bootstrap, và chạy ở 50% ngân sách update.
- Mọi quét $\sigma$, $\Lambda$, steps và MBR đều là inference.
- T-net không cần text, nên có thể train một lần trên dữ liệu gộp rồi fine-tune ngắn cho từng dataset.

---

## 8. Bảng và hình dự kiến cho paper

1. **Hình 1:** frontier perception–distortion (nDTW-hand theo SiCLIP/FD) cho mọi phương pháp, có đánh
   dấu điểm matched kinematics.
2. **Hình 2:** phổ theo articulator, và mức độ dự đoán được sau khi gióng bằng DTW (E0.3).
3. **Bảng 1/2:** so sánh SOTA. Tier B (bảng của SignNMFlow) và Tier A (có kiểm soát, matched
   kinematics), cả hai có nDTW, SiCLIP, FD.
4. **Bảng 3:** quy thuộc: flat regressor → +PMRF phẳng → +hierarchy (SignHiFlow) → +cross-fit →
   +PES chính xác → +$\Lambda$ theo articulator; kèm các baseline (ii)–(v).
5. **Bảng 4:** độ nhạy của metric (E0.6), là đóng góp phân tích.
6. **Bảng 5:** chi phí (tham số, NFE, độ trễ).
7. **Hình 3:** ví dụ định tính về handshape, so sánh μ, G và X.

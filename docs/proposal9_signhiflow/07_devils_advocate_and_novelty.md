# 07 — Phản biện, định vị novelty, rủi ro và kế hoạch dự phòng

Tài liệu tổng hợp ba nguồn:

- **(A)** tự phản biện trong lúc thiết kế;
- **(B)** phản biện độc lập. Một reviewer riêng, đóng vai devil's advocate, đọc 03/04/05 bản v1 và
  đối chiếu với code metric (`metrics.py`, `smplx_fk.py`) cùng code MotionHiFlow. Tóm tắt nằm ở §5;
  mọi thay đổi do nó gây ra đã được đưa vào bản v2;
- **(C)** đối chiếu với các công trình gần nhất.

---

## 1. Định vị novelty (v2)

| Công trình | Đã làm | Khác biệt của SignHiFlow |
|---|---|---|
| **MotionHiFlow** (2604.23264) | Flow phân cấp theo scale thời gian; DiT dùng chung; sample từ nhiễu ở mọi tầng; chỉ body | Flow phân cấp được dùng làm **transport từ một ước lượng**, không nhận text. Mỗi stage chỉ sinh một band DCT trực chuẩn mới (không renoise). Có thêm μ-net, PES khớp metric, cross-fitting và $\Lambda$ |
| **PMRF** (2410.00418) + **Freirich et al.** (2107.02555) | Posterior mean, rồi rectified flow không có điều kiện tới phân phối dữ liệu; lý thuyết frontier D–P | Phục hồi ảnh, một tầng, không có điều kiện ngữ nghĩa. Ta đưa ý tưởng sang **sinh có điều kiện ngôn ngữ**, với transport **phân cấp có ràng buộc band**, **$\Lambda$ theo articulator** (đồng nhất thức rủi ro ở 05-T2), **cross-fitting** cho dữ liệu nhỏ, và một giao thức đánh giá matched kinematics |
| **CorrDiff** (2309.15214), **StoRM** (2212.11851), **DiffSinger** (2105.02446) | Ước lượng hồi quy, rồi diffusion có điều kiện trên phần dư hoặc xuất phát từ ước lượng cộng nhiễu | Phần sinh của họ **có điều kiện** (05-T3: ở nghiệm tối ưu, đó là sampler). Phần sinh của ta là transport không điều kiện. CorrDiff-style là baseline bắt buộc (ii) ở 06 |
| **Grad-TTS / PriorGrad** (2105.06337, 2106.06406) | Prior phụ thuộc dữ liệu | Chính là cơ chế "anchor = offset" ở 05-T3, tức lỗi của bản v1 |
| **Cascaded diffusion** (2106.15282), **Decomposable FM** (2506.19839) | Sinh thô → mịn; mỗi scale có flow riêng, xuất phát từ nhiễu | Của ta xuất phát từ **ước lượng**, không nhận text, dùng DCT trực chuẩn cho chuỗi có độ dài thay đổi |
| **SignSparK** (2603.10446), **MaDiS**, **RobotSTAR**, **SRVQ** | Keyframe hoặc coarse-to-fine để chống regression-to-mean | Họ *sample* hoặc *tra từ điển* keyframe. Ta *ước lượng* rồi *transport*, có lý thuyết rủi ro và đánh giá ở matched kinematics |
| **BLANC** (2610.24109) | Tốc độ token khác nhau theo articulator | **Không** tuyên bố là mới |
| **MBR** (2111.09388, 2401.02749) | Chọn medoid | Chỉ là tuỳ chọn, **không** phải đóng góp |

### Novelty có thể bảo vệ

Đã thu hẹp lại sau phản biện (B):

1. **Phân tích và đánh giá.** Lĩnh vực SLP đang so sánh các phương pháp ở những điểm khác nhau trên
   một frontier perception–distortion mà không ai nói rõ. Ta cung cấp ba thứ:
   - bằng chứng: NFE 1 > 5 > 20; output tĩnh hoặc ngắn thắng DTW; bảng độ nhạy E0.6;
   - một **giao thức matched kinematics** cùng frontier SiCLIP/FD, để so sánh công bằng;
   - phân tích phổ theo articulator về mức độ dự đoán được từ text, sau khi gióng bằng DTW.
2. **Phương pháp.** **Estimate-then-Transport** cho sinh chuỗi có điều kiện ngôn ngữ, gồm:
   - một transport **phân cấp (MotionHiFlow), ràng buộc band, không nhận text**, đi từ ước lượng tới
     phân phối dữ liệu;
   - **nội suy theo articulator**, có đồng nhất thức rủi ro chính xác (05-T2, Mệnh đề 1);
   - **cross-fitting** để tránh anchor bị overfit trên dữ liệu nhỏ.

   Luận điểm được kiểm chứng bằng **trội frontier** so với flat regressor, CorrDiff-style, PMRF phẳng,
   MotionHiFlow-SLP và SignNMFlow-reimpl, ở cùng ngân sách (Gate G2).
3. **Kỹ thuật.** PES tái tạo **chính xác** FK của metric: LBS có pose blendshapes trên khoảng 1.5k
   vertex, kiểm tra bằng unit test.

### Những tuyên bố đã bỏ sau phản biện

- "Cơ chế PMRF" cho stage có text.
- "Tổng quát hoá chính xác hiệu chỉnh $\gamma$ của MotionHiFlow" (05-T5 chỉ còn là ghi chú).
- "Giám sát khớp tầng lexical/âm vị học". Band tần số không trùng tầng âm vị học.
- MBR+DTW như một đóng góp.
- T7 như một phản bác SignNMFlow (trừ khi A3 xác nhận).
- Track B (reward fine-tune), vì coupling với evaluator.
- L-REPA trong lõi; hoãn lại.

### Rủi ro bị coi là "bag of tricks"

- Phiên bản v2 có **ba khối** (μ-net, T-net, $\Lambda$), đều suy ra từ một luận điểm.
- R và L-REPA là tuỳ chọn, có dòng báo cáo riêng.
- Ablation chính rút về Bảng 3 (06 §8).

---

## 2. Tự phản biện (A), cập nhật cho v2

| # | Phản bác | Trả lời |
|---|---|---|
| 1 | "Vẫn là lợi dụng metric: trả về μ mượt rồi gọi là ước lượng." | Điểm vận hành chính là **matched kinematics**: tốc độ và gia tốc phải khớp GT. Mọi baseline được đưa về cùng điều kiện bằng shrink-to-mean. Thêm FD, SiCLIP và user study bắt buộc |
| 2 | "T-net không có text thì sẽ chọn sai handshape khi μ nhoè giữa hai lựa chọn." | Đúng là có rủi ro. μ là hàm của text, nên về nguyên tắc nó vẫn mang thông tin; chất lượng phụ thuộc vào μ. A6 (T-net có text) đo đánh đổi này. Nếu A6 tốt hơn ở matched kinematics, chọn A6 và mô tả trung thực là conditional sampler có prior dịch chuyển |
| 3 | "Cross-fitting tốn thêm compute." | Khoảng +20% cho Phase A. A19 đo lợi ích. Nếu không có lợi, dùng pseudo-plan |
| 4 | "Mệnh đề 1 là cho L2, còn metric là DTW không bình phương." | Đúng; 05-T2.2 nói rõ đây chỉ là phép tương tự. Kết luận định tính được kiểm chứng thực nghiệm ở E0.2 và E2 |
| 5 | "Không so được với số của SignNMFlow." | So sánh hai tầng (03 §5); Tier A là khẳng định khoa học |
| 6 | "Lợi ích đến từ các bản vá (x-pred, mBART fine-tune, PES), không từ ý tưởng." | Baseline (i)–(iv) dùng **cùng** backbone, PES và điều kiện pretrain. G2 so sánh frontier ở cùng ngân sách |
| 7 | "How2Sign sụp với mọi model." | Báo cáo trung thực; trọng tâm là Phoenix và CSL |
| 8 | "μ-net giống Progressive Transformers." | Đúng, μ-net là một regressor. Ở $\Lambda=0$ nó là dòng "flat regressor". Đóng góp nằm ở cách kết hợp với transport và ở frontier. Một regressor mạnh, có PES khớp metric, là cần thiết nhưng không đủ để làm paper |

---

## 3. Sổ rủi ro

| Rủi ro | Xác suất | Tác động | Dấu hiệu sớm | Giảm thiểu |
|---|---|---|---|---|
| G1 thất bại: μ-net không tốt hơn mean của SignNMFlow-reimpl | thấp–trung bình | cao | E1 ở 20k update | PES chính xác, fine-tune text, scale model |
| G2 thất bại vì CorrDiff-style (ii) ngang SignHiFlow | trung bình | cao (mất novelty) | E2 ở 30k | §4 |
| T-net không text làm hỏng ngữ nghĩa | trung bình | trung bình | SiCLIP của G thấp hơn của μ | dùng A6 |
| Evaluator yếu (SiCLIP không phân biệt được) | thấp–trung bình | cao | E0.5 | thêm Uni-Sign hoặc MSKA qua chiếu 2D, DTWpp |
| PES chính xác chậm | trung bình | thấp | throughput | lấy mẫu 48 frame, fp32, tập vertex thưa |
| Thiếu GPU | trung bình | cao | — | ưu tiên P0–P3, ablation ở 50% ngân sách |

---

## 4. Kế hoạch dự phòng

- **E0.9 cho η ≈ 1 ở mọi band, hoặc G1.5 thất bại ($\rho\ge0.8$).** Transport không có headroom, nên bỏ
  T-net. Phương pháp còn lại là μ-net (PES chính xác) cộng shrink-to-mean với sampler tốt nhất trong các
  baseline, kèm giao thức frontier. Lúc này trọng tâm paper chuyển sang đóng góp phân tích; tiếp tục
  như trường hợp dưới đây.
- **G2 thất bại so với (ii) CorrDiff-style.** Paper chuyển trọng tâm sang ba thứ:
  1. đóng góp phân tích và giao thức matched kinematics;
  2. μ-net có PES chính xác cộng phương pháp tốt nhất trong các baseline;
  3. bảng frontier.

  Novelty khi đó yếu hơn, phù hợp với workshop hoặc track về đánh giá.
- **G2 thất bại so với E2-0 (PMRF phẳng).** Bỏ hierarchy, công bố Estimate-then-Transport phẳng. Nói
  rõ rằng MotionHiFlow-style hierarchy không giúp gì cho transport.
- **G1 thất bại.** Dừng lại và debug μ-net (PES, text, dung lượng) trước mọi thứ khác.
- **Không đạt mục tiêu Tier B khi không dùng retrieval.** Báo cáo cả SignHiFlow-R, và nhấn mạnh Tier A
  cùng frontier.

---

## 5. Phản biện độc lập (B): tóm tắt và phản hồi

Reviewer đọc bản v1. Kết luận chung của reviewer: *"As written, the method does not do what its theory
says."* Bảng dưới liệt kê từng phát hiện và phản hồi trong v2.

| Mức | Phát hiện | Phản hồi trong v2 |
|---|---|---|
| **CRITICAL** | T4c (bản v1): anchor là hàm tất định của $c$ thì chỉ là một offset. Một stage có text, khi tối ưu, sample $p(y\mid c)$ với mọi $\sigma^*$. Điều này mâu thuẫn với bảng $V_L+2V_H$ ở T5 | **Chấp nhận.** Thiết kế lại thành Estimate-then-Transport với T-net không nhận text (05-T3), cộng Mệnh đề 1 (05-T2) |
| CRITICAL (D1) | Cách trộn anchor lúc train (GT sắc + self-anchor mờ) dạy model "bỏ qua anchor mờ", nên lúc inference nó thành sampler đầy đủ | Bỏ self-anchoring. Stage ≥ 2 giữ cố định band (đúng ngay từ cấu trúc). Augmentation trộn plan, có $\kappa$ là điều kiện (04 §5.2) |
| MAJOR (D2) | Network không biết $\sigma_s$ | T-net nhận $\sigma_k$ và $u$ (04 §5) |
| MAJOR (D3) | Plan, thành phần quan trọng nhất, chỉ được 10% số update, và không tất định | μ-net riêng, tất định, train ở mọi bước (04 §4) |
| MAJOR (D4) | Self-anchor ở stage 3 sai đặc tả; chi phí thật khoảng +50–60% | Bỏ. Dùng GT + aug làm mặc định; cache output thật là ablation A14, đã ghi đúng chi phí |
| MAJOR (D5) | Band tần số không phải tầng âm vị học (hold nằm ở dải thấp). Phần dư ở E0.3 trộn lẫn lệch nhịp | Bỏ tuyên bố về tầng âm vị học. Lựa chọn theo articulator chuyển sang $\Lambda$. E0.3 gióng bằng DTW trước, có thêm độ nhất quán pha |
| MAJOR (D6) | PES không khớp metric: khớp tay của metric là `J_regressor` × **mesh đã pose** (khoảng 734 vertex cho một tay, có pose blendshapes); trong feature không có "upper spine"; PES ở nhiễu cao làm hỏng phép đổi x→v | PES chính xác: LBS + blendshapes trên khoảng 1.5k vertex, kèm unit test. Sửa danh sách khớp. PES của T-net chỉ áp khi $u\ge0.7$, trọng số $u^2$ |
| MAJOR (D7) | MBR yếu (các ứng viên dùng chung plan, $q\ne p$) và đắt | MBR chỉ còn là tuỳ chọn, với N bằng nhau cho mọi phương pháp; luôn báo cáo N = 1 |
| MAJOR (D8) | Coupling: teacher SUE, evaluator BT, reward và chọn model cùng tối ưu về một họ evaluator | Bỏ SUE, L-REPA và Track B khỏi lõi. SiCLIP (bên ngoài) là metric ngữ nghĩa chính; back-translator làm lại với mục tiêu khác |
| MINOR (D9) | CFG: nhánh "unconditional" vẫn thấy anchor sinh từ text; $w=1.8$ đẩy ra xa ước lượng | Mặc định không dùng CFG (μ tất định, T-net không có text) |
| MINOR (D10) | Chuẩn hoá thừa theo articulator; lịch $p_{sa}$ mâu thuẫn; noise scale làm sai các công thức σ; khoảng thời lượng | Đã sửa hoặc ghi chú (04 §1, §3; 05-T1) |
| Toán T1 | Ringing Gibbs; giữ biên độ ≠ giữ SNR | Đã thêm (05-T1) |
| Toán T3 | Phép tính đúng (0.4626 so với 0.485), nhưng cách đặt vấn đề sai: sampler của MotionHiFlow trừ đúng phần nhiễu đã biết; kết quả không mới (Pyramid Flow) | 05-T5 chỉ còn là ghi chú, không tuyên bố là mới |
| Toán T5 | Metric là khoảng cách **không bình phương**, nên ước lượng tối ưu theo frame là geometric median; tỉ lệ $\sqrt2$ chứ không phải 2 | 05-T2.2; μ-net thêm loss khoảng cách không bình phương |
| Toán T6 | "Thời lượng tối ưu nằm dưới median" không được chứng minh; nDTW có thể bị lợi dụng bằng cách chèn pose nghỉ | 05-T6 đã sửa; E0.6(f) kiểm tra việc chèn pose |
| Toán T7 | 6D (Gram–Schmidt) cũng là toàn ánh; T7 đang bênh vực axis-angle trong khi mặc định là 6D | 05-T7 đã sửa cách diễn đạt |
| Novelty | E→S đã có (CorrDiff, StoRM, DiffSinger, PriorGrad, PMRF, cascaded); nếu giữ nguyên dễ bị đọc là "bag of tricks" | §1: thu hẹp tuyên bố; trội frontier là luận điểm chính; CorrDiff-style là baseline bắt buộc |
| Lợi dụng metric | Các biện pháp bảo vệ chưa đủ: nDTW, speed-ratio, BT có coupling, DTWpp không phải người | Thêm: giao thức matched kinematics; DTW theo band; FD trên đặc trưng bên ngoài; MBR cùng N và mean mode cho mọi baseline; điểm vận hành đăng ký trước; user study với người Điếc là bắt buộc |
| **Top-1 của reviewer** | "Gate không cần train trên checkpoint đã có: mean 1-NFE → renoise → tích phân (SDEdit) ± MBR, so với shrink-to-mean ở matched speed" | Đã thêm thành E0.8 |

---

## 5b. Phản biện độc lập vòng 2 (trên bản v2)

Kết luận của reviewer: *"The algebra is right … The weak point is the main justification of v2."*

| Mức | Phát hiện | Phản hồi trong v2.1 |
|---|---|---|
| **CRITICAL** | "$\sigma\to0$ thì transport tất định" là sai ở đây. Mỗi câu chỉ có một μ nên $p_\mu$ rời rạc, và nhiễu trong khoảng $10^4$ chiều gần như trực giao với $\mu_i-\mu_j$. Ở những hướng có $a\ll b$ (nhịp, biên độ, signer, dải cao), $W_2^2/\mathrm{MMSE}=(\sqrt{a+b}-\sqrt a)^2/b\to1$, tức T-net lý tưởng chính là sampler. Cái học được là rectified flow của coupling, không phải $T^\star$, nên chỉ bảo đảm hoà với sampling | **Chấp nhận.** 05-T3 viết lại phần giới hạn. Thêm **E0.9 Gelbrich headroom** (chạy trên CPU) và **E2-pilot**, gate $\rho<0.8$, cả hai chạy **trước** Phase B. Nếu không qua gate thì không train T-net đầy đủ |
| MAJOR | Mệnh đề 1 giả định $\mu=\mathbb E[Y\mid c]$, còn μ-net (dùng PES, dữ liệu hữu hạn) có bias. Kết luận "trội ở mọi $\Lambda$" chỉ đúng với $\lambda$ vô hướng. Ghi chú: độc lập $G\perp Y\mid c$ vẫn đúng trên test dù có cross-fit hay không; ngoại lệ là các câu trùng text | Mệnh đề 1' có thêm số hạng bias, ước lượng trên dev. Hệ quả giới hạn về $\lambda$ vô hướng. Báo cáo tách nhóm text đã thấy / chưa thấy |
| MAJOR | Cross-fitting tạo mismatch theo chiều ngược lại: μ toàn phần lúc test tốt hơn μ out-of-fold mà T-net được train | Input của T-net lúc test lấy từ một fold model; $\hat\mu$ trong công thức trộn vẫn dùng model toàn phần (A24) |
| MAJOR | Lúc inference đặt $\kappa=1$ nghĩa là coi $B$ sắc như GT, trong khi output thật của stage 1 mượt hơn. Augmentation trộn về plan mô hình hoá sai loại lỗi | Stage 2 dùng **output thật của stage 1**, rollout no-grad 4 NFE ở grid $T/4$. Bỏ $\kappa$ |
| MAJOR | Dải cao được sample có điều kiện trên $B$, nên "transport" chỉ đúng cho band $c_1$ | Đã sửa cách diễn đạt ở 04 §0, 05-T4 và Hệ quả 1 |
| MINOR | Nhiễu $0.03\eta$ phải là $0.03\Pi\eta$; pseudo-code §8 cộng $B$ hai lần | Đã sửa |
| MINOR→MAJOR | Gram–Schmidt của một phép trộn tuyến tính là nlerp, suy biến khi $\|a_1\|$ nhỏ. Có "bóng đôi" khi $G$ lệch pha với $\hat\mu$ | Mặc định dùng slerp theo từng khớp; DTW-warp $G$ là tuỳ chọn (A23); log norm 6D |
| MAJOR | Matched kinematics: speed **không đơn điệu**; có thể không có nghiệm; $\lambda_F$ không bị ràng buộc; các part tương tác (tay giữ hướng thế giới); speed và accel không khớp được cùng lúc; tỉ lệ trên trung bình dataset có thể bị lợi dụng; baseline chỉ được 1 tham số | Giải đồng thời $(\lambda_B,\lambda_H)$ bằng grid search; $\lambda_F=\lambda_B$; chỉ ràng buộc speed; quy tắc không có nghiệm; thêm metric độ động theo từng tay và trong khung cổ tay; median theo chuỗi; baseline dùng **cùng** $\Lambda$ theo articulator |
| MAJOR | Nguy cơ ghi nhớ: mỗi $\mu_i$ chỉ có một $x_i$, T-net có thể học thuộc | Theo dõi $\rho$ và x-pred loss trên dev ở mỗi snapshot; dừng sớm |
| **Thứ tự reviewer đề xuất** | Gelbrich headroom → E0.8 → pilot stage 1 (quét ρ theo σ) → kiểm tra bias → chỉ train K = 3 nếu η và ρ đều qua | Đã áp dụng vào 06 §7 |

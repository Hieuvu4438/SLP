# 04 — SignHiFlow (v2): Estimate-then-Transport, đặc tả end-to-end

**Lịch sử sửa đổi.** Bản v1 dùng các stage mịn "anchored", có nhận text, xuất phát từ plan cộng nhiễu.
Phản biện độc lập (07 §5) chỉ ra rằng ở nghiệm tối ưu thiết kế đó suy biến thành sampling đầy đủ
(05-T3). Bản v2 dưới đây làm cho "ước lượng rồi mới tạo chi tiết" **đúng ngay từ cấu trúc**, và có một
đồng nhất thức rủi ro chính xác (05-T2).

**Bản v2.1**, sau vòng phản biện thứ hai (07 §5b), thay đổi năm điểm:

- Lợi ích của transport phải qua gate trước khi train đầy đủ: Gelbrich headroom và pilot đo $\rho$.
- Input của T-net lúc test là μ từ fold model.
- Stage 2 dùng anchor là output thật của stage 1, sinh lúc train.
- Trộn bằng slerp; tuỳ chọn DTW-warp $G$ theo nhịp của $\hat\mu$ trước khi trộn.
- Giải matched kinematics đồng thời trên $(\lambda_B,\lambda_H)$, và baseline cũng được dùng $\Lambda$ theo articulator.

## 0. Tóm tắt

SignHiFlow gồm ba khối:

1. **μ-net, bộ ước lượng.** Một MMDiT điều kiện trên text (backbone của MotionHiFlow) dự đoán
   **tất định** motion "trung tâm" $\mu(c)$, tức posterior mean hoặc median trong không gian FK, chỉ
   với 1 forward. Loss được tính trong đúng không gian mà metric đo.
2. **T-net, transport phân cấp không nhận text.** Đây là chính hierarchical flow của MotionHiFlow,
   gồm DiT dùng chung, các stage thời gian từ thô tới mịn và scale embedding, được dùng làm
   **transport** kiểu PMRF đi từ $\mu+\sigma\varepsilon$ tới phân phối motion thật. Có hai điểm khác
   MotionHiFlow: mỗi stage chỉ sinh **một band tần số mới** và giữ nguyên các band đã sinh, và
   T-net **không nhận text**. Output là $G=T(\mu)$: motion sắc, nhất quán với plan.
3. **Nội suy theo articulator.** $X=\hat\mu+\Lambda(G-\hat\mu)$ với
   $\Lambda=\mathrm{diag}(\lambda_B,\lambda_L,\lambda_R,\lambda_F)$. Theo 05-T2 (Mệnh đề 1' và Freirich et
   al. 2021), rủi ro bằng $\mathbb E\|\hat\mu-Y\|^2+\sum_p\lambda_p^2\,\mathbb E\|G_p-\hat\mu_p\|^2$ cộng một
   số hạng bias. Mức rủi ro đó **chỉ phụ thuộc độ toả của $G$ quanh $\hat\mu$**.

   Hai giới hạn của kết quả "transport tốt hơn sampling":
   - nó chỉ được bảo đảm cho $\lambda$ vô hướng, và chỉ ở band $c_1$;
   - nó chỉ đúng ở những hướng có headroom: mỗi câu chỉ có một μ, nên $p_\mu$ là rời rạc (05-T3).

   Vì vậy lợi ích này phải **đo trước** khi train đầy đủ, bằng hai gate ở 06: Gelbrich headroom (E0.9)
   và pilot stage 1 (E2-pilot, $\rho<0.8$).
   - **Điểm vận hành chính được đăng ký trước:** $\lambda_p$ được chọn trên dev sao cho tỉ lệ tốc độ
     và gia tốc của từng articulator khớp với GT ("matched kinematics").
   - Mọi baseline cũng được đưa về cùng điều kiện này bằng shrink-to-mean (06 §1).

```
text ─► mBART-large-cc25 enc (fine-tune) ─► E ─► duration head ─► T̂
                                         │
                                         ▼
          μ-net (MMDiT: query motion tokens ⟷ text, 4 double + 10 single blocks)   [Track R: + memory retrieved]
                                         │  1 forward, tất định
                                         ▼
                                    μ ∈ R^{T̂×256}
                                         │
      T-net (DiT không nhận text, 12 single blocks), band-constrained hierarchy:
        stage 1  grid T̂/4 : (D̃μ + σ1·ε) ──► ŷ1               (transport dải thấp)
        stage 2  grid T̂/2 : giữ Ũŷ1, sinh band (T̂/4, T̂/2]   (n2 bước)
        stage 3  grid T̂   : giữ Ũŷ2, sinh band (T̂/2, T̂]     (n3 bước)
                                         │
                                         ▼  G
                     X = μ + Λ(G − μ),  Λ theo articulator  ──► 6D → axis-angle → SOKE 133-D ──► evaluator chung
```

**Giữ lại từ MotionHiFlow:**

- MMDiT (double-stream rồi single-stream, joint attention text–motion, adaLN, RoPE theo thời gian
  và articulator), dùng cho μ-net;
- hierarchical flow theo scale thời gian với một DiT dùng chung và scale embedding, dùng cho T-net;
- x/velocity flow matching tuyến tính.

**Thay đổi so với MotionHiFlow:**

- (i) Flow là **transport từ ước lượng** chứ không phải sampler từ nhiễu.
- (ii) T-net không nhận text.
- (iii) Mỗi stage sinh đúng một band DCT trực chuẩn, các band thấp hơn giữ cố định, không renoise.
- (iv) Không dùng VAE latent; sinh trực tiếp trên rotation 6D.
- (v) Có loss FK chính xác theo metric (PES).
- (vi) μ dùng để train T-net được lấy bằng cross-fitting.
- (vii) Có nội suy $\Lambda$ và giao thức matched kinematics.

---

## 1. Biểu diễn chuyển động

| Articulator | Khớp | Chiều (6D) |
|---|---|---|
| **B** | neck, L/R collar, head, L/R shoulder, L/R elbow, L/R wrist (đúng 10 khớp thân trên trong feature SOKE, tức SMPL-X joint 12–21; **không** có spine) | 60 |
| **L**, **R** | 15 khớp ngón mỗi tay | 90 + 90 |
| **F** | jaw (6D) + 10 expression | 16 |

**Chuyển đổi.**

- Đầu vào: SOKE 133-D chuẩn hoá. Khử chuẩn hoá ra axis-angle $\omega$.
- Với hai tay, cộng `hands_mean` (SMPL-X `flat_hand_mean=False`; smplx cộng giá trị này trong
  axis-angle trước Rodrigues).
- Tính $R=\exp(\omega)$, rồi 6D = hai cột đầu của $R$.
- Chuẩn hoá theo từng kênh bằng thống kê của train. Không cần thêm hệ số chuẩn hoá theo articulator,
  vì chuẩn hoá theo kênh đã đủ.
- Chiều ngược: Gram–Schmidt → $R$ → $\log$ → trừ `hands_mean` → chuẩn hoá theo thống kê SOKE.
  Test round-trip phải khớp FK.

**Vì sao không dùng VAE latent.** Metric đo độ chính xác, nên lỗi tái tạo của codec là sàn cứng
(InterCMDM, ACMDM; xem 02 §3). E0.1 đo trần codec của MotionHiFlow-SLP. 6D là mặc định vì bằng chứng
Inter-X++ cho thấy axis-angle kém hơn. Ablation A3 so với axis-angle và quaternion. Theo 05-T7, cả hai
chart đều hợp lệ về hình học.

---

## 2. Toán tử và cấu trúc band

- $\tilde D$, $\tilde U$, $\Pi$ theo 05-T1: DCT-II trực chuẩn theo từng clip, có mask độ dài, cache ma
  trận theo độ dài.
- **Grid của T-net:** $c_k=\max(4,\mathrm{round}(r_kT))$, $r=(1/4,1/2,1)$.
  - Ở 25 fps, grid 1/4 là 6.25 Hz. Mỗi ký hiệu dài khoảng 0.42–0.66 s (Wilbur 2009), nên được biểu
    diễn bởi ít nhất 2–4 mẫu.
  - $K=2$ là ablation A5. MotionHiFlow Table 2 cho thấy 2 tầng mang phần lớn lợi ích, 4 tầng kém hơn 3.
- **Grid theo articulator** ($r_1^{L,R}=1/2$) là tuỳ chọn, chỉ bật nếu E0.3 ủng hộ. Không tuyên bố
  điều này là mới: BLANC đã dùng token tay với tốc độ gấp 2 body.
- **Phân tích theo band.** Hold (handshape, location) nằm chủ yếu ở dải **thấp**, transition ở dải cao.
  Vì vậy band tần số *không* trùng với tầng âm vị học. Việc chọn ước lượng hay transport cho từng
  articulator được điều khiển bằng $\Lambda$ (§7), không phải bằng band.

---

## 3. Text encoder và thời lượng (thuộc μ-net)

- **Encoder:** mBART-large-cc25 encoder, checkpoint reduced-vocab của SOKE, để so sánh công bằng.
  Fine-tune với LR 2e-5 và layer-decay 0.9, **đóng băng 5k update đầu**. Trong repo, fine-tune ở
  LR 2e-4 ngay từ đầu làm đặc trưng text sụp: cosine giữa các câu lên 0.999 (methods/signmflow README).
- **Text refinement:** chiếu 1024 → 512, rồi 2 khối ConvNeXt-1D, cộng language embedding. Ablation
  A10. Bằng chứng: F5-TTS không học được alignment khi thiếu khối refinement.
- **PA-RoPE** (A11): text token $j$ được đặt toạ độ thời gian $(j+0.5)/N\cdot T$. Đây là thiên hướng chéo
  *mềm*, attention vẫn đảo trật tự được. Không dùng MAS hay CTC, vì trật tự ký hiệu khác trật tự từ.
- **Duration head:** attention-pooling trên text token cộng language embedding, ra $(\mu_{\log T},\log b)$,
  loss Laplace NLL.
  - Inference dùng median, $\hat T=\mathrm{clip}(\mathrm{round}_4 e^{\mu_{\log T}},40,400)$. Loader crop
    mọi clip GT vào [40, 400] theo quy tắc MaDiS, nên khoảng này khớp với dữ liệu.
  - Một biến thể "quantile" chỉ được báo cáo thành dòng riêng; 05-T6 giải thích vì sao không suy ra
    được hướng tối ưu.
  - A12 so sánh tổng thời lượng với cách tách units × rate (Wilbur).

---

## 4. μ-net: bộ ước lượng tất định

**Kiến trúc** (MMDiT của MotionHiFlow, phóng to):

| Thành phần | Mặc định |
|---|---|
| Width / heads | 512 / 8 |
| Blocks | 4 double-stream + 10 single-stream; QK-RMSNorm; adaLN-Zero; MLP ×4; dropout 0.1 |
| Motion stream | **query token** tất định: mỗi frame có 4 token (B, L, R, F) = articulator embedding + embedding học được (tuỳ chọn), **không có input nhiễu**. Dùng time patch 2 để giảm token, tức T̂/2 × 4 token |
| RoPE | thời gian 1/2 · articulator 1/4 · chỉ số text 1/4 (+ PA-RoPE) |
| Điều kiện adaLN | pooled text + language |
| Output | Linear theo part → 6D chuẩn hoá (time patch 2 → 2 frame mỗi token), zero-init |
| [Track R] | memory token (motion của top-k câu train, $\tilde D$ về 64 frame) nối vào luồng text |

Khoảng 70M tham số, cộng mBART encoder khoảng 170M.

**Mục tiêu.** Loss là:

$$\mathcal L_\mu=\mathcal L_x+\lambda_{\rm PES}\mathcal L_{\rm PES}+\lambda_{\rm dur}\mathcal L_{\rm dur}\;[+\lambda_{\rm sdtw}\mathcal L_{\rm sDTW}]$$

- $\mathcal L_x=\sum_pw_p\,\mathrm{mean}(\hat\mu_p-x_p)^2$, với $w=(1,1.5,1.5,0.5)$. Loss bình phương làm
  cho $\hat\mu$ tiến về mean.
- $\mathcal L_{\rm PES}$ (§6) là khoảng cách **không bình phương** trên khớp FK, đúng với metric. Nó kéo
  $\hat\mu$ về phía geometric median theo khớp (05-T2.2).
- $\lambda_{\rm PES}$ được chọn sao cho $\lambda_{\rm PES}\mathcal L_{\rm PES}\approx\mathcal L_x$ ở 1k bước
  đầu. Hai loss có thang khác nhau, một bình phương một không, nên log cả hai và kiểm tra lại tỉ lệ
  mỗi 10k bước.
- $\mathcal L_{\rm sDTW}$ là tuỳ chọn (A9): soft-DTW, $\gamma=0.01$, trên khớp FK của $\tilde D_{T\to T/4}\hat\mu$.
  Mục đích là biến $\mu$ thành xấp xỉ DTW-barycenter thay vì mean theo frame (05-T2.2).

μ-net chính là một mô hình SLP hồi quy mạnh. Riêng nó, ở $\Lambda=0$, đã là một dòng baseline quan
trọng: "flat regressor" trong 06.

---

## 5. T-net: transport phân cấp có ràng buộc band (MotionHiFlow trong vai trò transport)

**Kiến trúc:**

- DiT **không nhận text**, 12 single-stream block, width 512, 8 head, khoảng 40M tham số.
- Token: mỗi frame của grid $c_k$ có 4 token articulator. Input mỗi token là Linear của
  $[z_p;\ \mathbb 1_{\text{band mới}}]$.
- RoPE thời gian dùng toạ độ $i\cdot T/c_k$ (đơn vị frame gốc), cộng RoPE articulator.
- Điều kiện adaLN: stage $k$, $r_k$, $u$, $\sigma_k$.
- **Không** có text, pooled text hay language. Tính tất định khi $\sigma\to0$ phụ thuộc vào điều này
  (05-T3).

**Interpolant** (05-T4):

| Stage | Grid | Phần giữ cố định | Phần sinh: $u=0\to1$ | Target x-pred |
|---|---|---|---|---|
| 1 | $T/4$ | — | $(1-u)(\mu_1+\sigma_1\varepsilon)+u\,y_1$ | $y_1$ |
| 2 | $T/2$ | $B_2=\tilde U\hat y_1$ (train: **output thật của stage 1**, rollout no-grad 4 NFE) | $(1-u)((I-\Pi)\mu_2+\sigma_2\varepsilon_H)+u(I-\Pi)y_2$ | $(I-\Pi)y_2$ |
| 3 | $T$ | $B_3=\tilde U\hat y_2$ (train: $\Pi y_3+0.03\,\Pi\eta$; A14: output thật được cache) | như trên với grid $T$ | $(I-\Pi)y_3$ |

- Velocity: $\hat v=(\hat h-h_u)/(1-u)$. Euler với $n_k$ bước.
- Mặc định $\sigma_1=0.2$, $\sigma_2=\sigma_3=0.5$. Lúc train, $\sigma_1\sim\mathcal U[0.05,0.4]$ và
  $\sigma_{2,3}\sim\mathcal U[0.2,1.0]$, đưa vào điều kiện, rồi quét lúc inference.
- $\sigma_1$ nhỏ giữ transport dải thấp gần tất định (PMRF). $\sigma_{2,3}$ lớn hơn vì dải cao gần như
  không có thông tin trong $\mu$.

### 5.1 Cross-fitting $\mu$ (quan trọng với dữ liệu nhỏ, 05-T3)

- Chia train thành 2 fold (theo câu; các câu trùng text đặt chung một fold).
- Huấn luyện $\mu$-net$^{(1)}$ trên fold 2 để dự đoán $\mu$ cho fold 1, và ngược lại.
- T-net được huấn luyện trên các $\mu$ out-of-fold này, cache sẵn trên đĩa, tính một lần sau Phase A.
- **Lúc test:** input của T-net là μ từ **một fold model chọn ngẫu nhiên**, nên khớp với phân phối lúc
  train. Thành phần ước lượng $\hat\mu$ trong công thức trộn dùng μ-net toàn phần, vốn tốt hơn; điều này
  vẫn hợp lệ theo Mệnh đề 1'.
  - Không dùng μ toàn phần làm input cho T-net: μ đó tốt hơn mọi thứ T-net từng thấy, và T-net sẽ sửa quá tay.
  - K = 5 là ablation.
- Chia fold sao cho tỉ lệ text trùng giữa các fold xấp xỉ tỉ lệ trùng giữa train và dev.
- Ablation A19: dùng μ in-fold, hoặc pseudo-plan (GT làm mượt Gaussian σ = 3 frame, co 30% về mean
  của dataset).

### 5.2 Phần giữ cố định ở stage ≥ 2: khớp train với test

**Stage 2 (mặc định).** $B_2$ là output thật của stage 1, sinh lúc train bằng rollout no-grad, 4 NFE,
dùng EMA model. Rollout chạy trên grid $T/4$ nên rẻ: dưới +20% thời gian mỗi batch stage 2. Cách này
khớp đúng với inference.

Phiên bản trước dùng $\kappa$-augmentation, $B'=\kappa\Pi y+(1-\kappa)\Pi\mu$, và đã bị loại vì hai lý do:

- Lúc inference, $\kappa=1$ báo cho T-net rằng $B$ sắc như GT, trong khi output thật của stage 1 mượt hơn.
- Phép trộn mô hình hoá sai loại lỗi. Lỗi thật của stage 1 là lệch nhịp, hoặc chọn một phương án khác
  vẫn hợp lý, chứ không phải bị nhoè về phía plan.

**Stage 3.** $B_3=\Pi y_3+0.03\,\Pi\eta$. Nhiễu nằm **trong dải thấp** để không rò sang band đang sinh.
A14 thay bằng output thật của stage 2, cache theo clip và làm mới mỗi 5k update; chi phí thêm khoảng
+50–60%.

### 5.3 Loss của T-net

$$\mathcal L_T=\sum_pw_p\,\mathrm{mean}\big((I-\Pi)(\hat y_{k,p}-y_{k,p})\big)^2+\lambda_{\rm PES}\,\mathbb 1[u\ge0.7]\,u^2\,\mathcal L_{\rm PES}(\tilde U(B+\hat h))$$

- PES chỉ áp ở SNR cao ($u\ge0.7$). Ở nhiễu cao, x-pred buộc phải mờ; nếu ép FK ở đó thì $\hat y$ bị kéo
  khỏi $\mathbb E[y\mid z]$, và công thức đổi x-pred sang velocity không còn đúng.
- Ở stage 1, $\Pi$ được thay bằng 0, nên toàn bộ grid được sinh.

---

## 6. PES: giám sát FK **chính xác theo metric**

Mỗi thành phần của PES phải tái tạo **đúng** pipeline của `src/slp/evaluation/smplx_fk.py`:

- **Body joints:** `output.joints` của SMPL-X với shape SOKE cố định, root và chân bằng 0. Khớp thân là
  biến đổi FK của khớp ở rest pose. Các khớp 55–59 (mũi, mắt, tai) lấy từ vertex đã pose.
- **Hand joints:** 21 khớp mỗi tay. Mỗi khớp là một hàng `J_regressor` nhân với **mesh đã pose**, cộng
  5 vertex đầu ngón. Theo phản biện, riêng tay trái đã cần khoảng 734 vertex.
- **Cách cài:** LBS kèm **pose blendshapes** trên đúng tập khoảng 1.5k vertex cần thiết. Cài trong
  `src/slp/geometry/sparse_fk.py`, có unit test khớp với `smplx_fk.default_fk()` tới 1e-5 m.
- **Để tiết kiệm bộ nhớ:** mỗi bước chỉ tính PES trên 48 frame lấy ngẫu nhiên mỗi clip. Chạy ở fp32.

**Các thành phần.** Mỗi thành phần là trung bình khoảng cách Euclid không bình phương, đơn vị cm, giống
metric:

| Thành phần | Đại lượng | Căn chỉnh (như metric) | Metric tương ứng |
|---|---|---|---|
| $\ell_{\rm body}$ | 12 khớp body `UPPER_BODY_IDX` | trừ pelvis | DTW-JPE body |
| $\ell_{\rm hand}$ | 2×21 khớp tay, hướng thế giới | trừ cổ tay | DTW-JPE hand |
| $\ell_{\rm shape}$ | 2×21 khớp tay trong **khung cục bộ của cổ tay** (khử xoay, xấp xỉ Procrustes) | khung cổ tay | DTW-PA-JPE hand |
| $\ell_{\rm loc}$ | cổ tay trong khung đầu và khung ngực | — | vị trí (location) |
| $\ell_{\rm vel}$ | sai phân bậc 1 của cổ tay và đầu ngón | khung thân | độ động |

$$\mathcal L_{\rm PES}=\ell_{\rm body}+\ell_{\rm hand}+\ell_{\rm shape}+0.5\,\ell_{\rm loc}+0.5\,\ell_{\rm vel}.$$

- μ-net: $\ell_{\rm vel}=0$. Đạo hàm của một ước lượng mean không cần khớp với đạo hàm của GT.
- **Cân bằng gradient:** log norm gradient theo nhóm khớp. PIDiffSign ghi nhận 93% gradient dồn vào
  ngón tay khi dùng loss gia tốc.

---

## 7. Nội suy theo articulator và điểm vận hành

$X=\hat\mu+\Lambda(G-\hat\mu)$.

**Cách trộn.**

- **Mặc định: slerp theo từng khớp** giữa $\mathrm{GS}(\hat\mu)$ và $\mathrm{GS}(G)$ với tham số $\lambda_p$.
  Phần expression trộn tuyến tính.
- Lý do không trộn tuyến tính trong 6D rồi Gram–Schmidt: cách đó tương đương nlerp trên cột thứ nhất,
  không đều theo góc và suy biến khi góc lớn. Ngoài ra $\hat\mu$, là một trung bình, có thể có
  $\|a_1\|$ nhỏ ở các khớp có nhiều mode (ví dụ sấp/ngửa cổ tay), làm $\mathrm{GS}(\hat\mu)$ không ổn định.
  Log histogram của $\|a_1\|$ và $\|a_2\|$.
- **Tuỳ chọn, chống "bóng đôi":** stage 1 có thể dịch nhịp, khi đó $G$ lệch pha so với $\hat\mu$, và trộn
  theo từng frame ở $\lambda$ trung gian tạo ra chuyển động hai lần. Cách xử lý là warp $G$ theo nhịp của
  $\hat\mu$ bằng DTW trên khớp FK trước khi trộn. Mọi $G'=f(G,\hat\mu)$ vẫn thoả $G'\perp Y\mid c$, nên
  Mệnh đề 1' vẫn đúng. Đây là ablation A23.

**Điểm vận hành được đăng ký trước** (06 §1.2):

1. **Chính: matched kinematics.**
   - Tham số: $\lambda_L=\lambda_R=\lambda_H$ và $\lambda_F=\lambda_B$ (cố định trước).
   - Giải **đồng thời** $(\lambda_B,\lambda_H)$ trên dev sao cho *speed ratio* của body và speed ratio
     *trong khung cổ tay* của tay đều bằng $1.00\pm0.05$. Accel ratio chỉ báo cáo, vì một tham số không
     khớp được cả speed lẫn accel.
   - Lý do giải đồng thời: điểm tay trong metric bị trừ vị trí cổ tay nhưng vẫn giữ hướng thế giới,
     nên $\lambda_B$ cũng làm đổi tốc độ tay.
   - Speed **không** đơn điệu theo $\lambda$: hàm lồi, và có thể giảm trước khi tăng khi $G$ lệch pha với
     $\hat\mu$. Dùng grid search trên $[0,1]^2$ với bước 0.05.
   - **Không có nghiệm** (ví dụ speed(1) < 1) thì báo cáo $\Lambda=I$ kèm cờ, **không** ngoại suy.
2. **Phụ: composite.** $\Lambda$ cực tiểu hoá $\overline{\rm nDTW}_{\rm hand}+\overline{\rm nDTW}_{\rm body}-\beta\cdot\mathrm{SiCLIP}$
   trên dev.
3. **Đường cong:** quét $\lambda$ chung $\in\{0,0.25,0.5,0.75,1\}$ để vẽ frontier.

**Chống lợi dụng tỉ lệ trung bình.** Tỉ lệ speed tính trên trung bình toàn dataset có thể bằng 1 trong
khi một số chuỗi rất nhanh hoặc rung, còn số khác gần như đứng yên; chuỗi đứng yên lại có lợi về DTW.
Vì vậy báo cáo thêm median của tỉ lệ theo từng chuỗi, và tỉ lệ số chuỗi có ratio < 0.5.

**Baseline ở cùng điều kiện.** Với SignNMFlow-reimpl, MotionHiFlow-SLP và các baseline khác:

- $\mu_{\rm base}$ là output 1-NFE của chính model đó; $S$ là mẫu nhiều bước;
- dựng $X=\mu_{\rm base}+\Lambda(S-\mu_{\rm base})$ **với $\Lambda$ theo articulator giống hệt của ta**, cùng
  quy tắc slerp và cùng cách giải;
- chọn $\Lambda$ theo đúng quy tắc 1–2.

---

## 8. Inference

```python
def generate(text, lang, sigma=(0.2, 0.5, 0.5), steps=(4, 3, 3), Lam=None):
    E = text_encoder(text)
    T = duration_head(E, lang)                        # median
    mu_hat = mu_net_full(E, lang, T)                  # ước lượng: model toàn phần
    mu_in = random.choice(mu_net_folds)(E, lang, T)   # input của T-net: một fold model (§5.1)
    y = euler(T_net, z0=Dt(mu_in, T // 4) + sigma[0] * randn(), stage=1, steps=steps[0])
    for k, c in [(2, T // 2), (3, T)]:
        B = Ut(y, c)                                  # band thấp đã sinh, giữ cố định
        h0 = (I - Pi) @ Dt(mu_in, c) + sigma[k - 1] * (I - Pi) @ randn(c)
        h = euler_band(T_net, B=B, h0=h0, stage=k, steps=steps[k - 1])   # chỉ trả về band mới (I−Π)
        y = B + h
    G = y
    X = slerp_per_joint(mu_hat, G, Lam)               # Lam theo articulator (§7); tuỳ chọn: DTW-warp G theo mu_hat trước
    return to_soke133(X)
```

- **Chi phí:** 1 forward μ-net (khoảng 70M, có text) cộng 10 forward T-net (khoảng 40M, ít token ở
  stage thô).
- **Không cần CFG:** μ-net tất định, T-net không nhận text.
- **Không cần MBR:** $N=1$. MBR là tuỳ chọn ở §10.
- Rẻ hơn hoặc tương đương SignNMFlow (decoder 100M × 5 NFE). Bảng chi phí: 06 §8.

---

## 9. Paradigm huấn luyện

| Phase | Nội dung | Dữ liệu | Ngân sách mặc định |
|---|---|---|---|
| **0** | Evaluator (không bao giờ dùng để train): SiCLIP (MaDiS, công khai) làm metric ngữ nghĩa chính; back-translator trong repo được làm lại (CE dịch + CTC gloss phụ trợ, kiến trúc khác hẳn mọi thứ dùng trong train); DTWpp | train split | khoảng 4–8 giờ GPU mỗi dataset (back-translator) |
| **A0** (tuỳ chọn) | Pretrain μ-net đa ngôn ngữ (Ph + CSL + H2S), language token, lấy mẫu theo nhiệt độ α = 0.5 | khoảng 56k clip | 150k update, batch 64 |
| **A** | μ-net theo từng dataset: 1 model trên toàn bộ train (dùng ở test) và 2 model cross-fit (dùng để sinh μ out-of-fold) | từng dataset | Ph 40k / CSL 60k / H2S 80k update cho model toàn phần; model cross-fit 60% ngân sách đó |
| **A→B** | Cache $\mu$ out-of-fold cho toàn bộ train | — | chỉ inference |
| **B0 (gate)** | **Pilot stage 1**, chạy khi E0.9 cho thấy có headroom: chỉ stage 1, một dataset, khoảng 20k update. Đo $\rho$ theo $\sigma_1$ trên dev | Phoenix | khoảng 3–5 giờ GPU |
| **B** | T-net đầy đủ, **chỉ khi** $\rho<0.8$. **Không cần text**, nên có thể gộp cả 3 dataset; cũng có thể thêm motion ký hiệu không nhãn với pseudo-plan (A19). Theo dõi $\rho$ và x-pred loss trên dev ở mỗi snapshot, **dừng sớm** để tránh ghi nhớ (05-T3) | union, rồi fine-tune từng dataset | tối đa 150k + 30k update, batch 64 |
| **C** | Chọn $\Lambda$ trên dev theo §7 | dev | chỉ inference |

**Cài đặt chung.**

- Tối ưu: AdamW (0.9, 0.99), LR 2e-4 (DiT) / 2e-5 (mBART), warmup 5k, cosine về 10%, wd 0.01,
  grad-clip 1, bf16 (FK ở fp32), EMA 0.9999.
- Snapshot EMA mỗi 5k update. Snapshot được chọn bởi script trong `configs/experiments/`, dùng
  nDTW-hand + SiCLIP trên dev; method không tự tính metric.

**Ràng buộc chống rò rỉ.**

- Mọi model, chỉ mục và thống kê chỉ dùng train split.
- Evaluator không chia dữ liệu hay mục tiêu với bất kỳ thành phần nào được tối ưu.
- Test chỉ chạy ở E7, khi người dùng yêu cầu.

---

## 10. Module tuỳ chọn (mặc định tắt, mỗi cái là một dòng ablation)

| Module | Mô tả | Lý do để tắt mặc định |
|---|---|---|
| **R** (retrieval cho μ-net) | Top-k = 4 câu train theo embedding câu → memory token. Làm giảm MMSE trực tiếp | Novelty thấp; báo cáo thành dòng riêng (SignHiFlow-R); kèm phân tích theo nhóm độ tương đồng |
| **T-text** (A6) | T-net nhận text qua cross-attn và có CFG | Biến T-net thành conditional sampler (05-T3); dùng để so sánh |
| **E→S-v1 / SDEdit** (A20) | Không có T-net: lấy sampler có text của chính baseline, xuất phát từ μ cộng nhiễu $\sigma^*$ | Đây là thiết kế cũ, dùng làm baseline. E0.8 đo trên checkpoint đã có |
| **L-REPA** | Gióng hidden state của μ-net với encoder hiểu ký hiệu | Hoãn lại: rủi ro coupling với evaluator, lợi ích không chắc |
| **MBR** | N ứng viên (khác nhau ở nhiễu T-net), chọn medoid theo nDTW | Chỉ dùng khi mọi baseline có cùng N; luôn báo cáo N = 1 |
| Reward fine-tune (Track B) | — | **Bỏ**, vì rủi ro lợi dụng evaluator |

---

## 11. Ablation

| ID | Thay đổi | Câu hỏi |
|---|---|---|
| A1 | VAE latent theo MotionHiFlow thay cho 6D thô | codec có phải sàn cứng không? |
| A3 | axis-angle / quaternion thay cho 6D | biểu diễn |
| A5 | T-net phẳng (K = 1, PMRF thuần) / K = 2 / K = 3 | hierarchy có giúp transport không? |
| A6 | T-net có text (+CFG) | text-free có cần không? (05-T3) |
| A9 | soft-DTW cho μ-net | ước lượng mean hay barycenter |
| A10 / A11 | bỏ ConvNeXt / bỏ PA-RoPE | text |
| A12 | duration: tổng / units × rate / độ dài GT (chẩn đoán) | thời lượng |
| A13 | không có A0 (không pretrain đa ngôn ngữ) | dữ liệu |
| A14 | anchor ở stage ≥ 2: GT + aug / cache output thật | nhất quán train/test |
| A16 | noise scale, $\sigma_k$ | SNR |
| A17 | kích thước μ-net và T-net | dung lượng |
| A19 | μ in-fold / cross-fit / pseudo-plan | cross-fitting |
| A20 | E→S-v1 (SDEdit có text) thay cho T-net | transport so với anchored sampling |
| A21 | bỏ từng thành phần PES; PES chính xác so với FK xấp xỉ (cây khớp, không blendshape) | giám sát |
| A22 | $\Lambda$ chung so với $\Lambda$ theo articulator | phân bổ theo articulator |
| A23 | trộn: slerp so với trộn tuyến tính 6D; có hoặc không DTW-warp $G$ về nhịp của $\hat\mu$ | bóng đôi, ổn định của Gram–Schmidt |
| A24 | input của T-net lúc test: μ từ fold model / μ toàn phần / trung bình các fold | khớp phân phối khi cross-fit |

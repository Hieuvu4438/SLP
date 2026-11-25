# 05 — Cơ sở lý thuyết của SignHiFlow (bản v2, đã sửa sau phản biện)

Tài liệu chứa các dẫn xuất mà [04_method_signhiflow.md](04_method_signhiflow.md) dựa vào.

**Lịch sử sửa đổi.** Bản v1 tuyên bố rằng một stage mịn có điều kiện text, xuất phát từ plan cộng nhiễu
("anchored"), sẽ thực hiện "ước lượng dải thấp, sample dải cao". **Tuyên bố đó sai** tại nghiệm tối ưu.
Phản biện độc lập (07 §5) chỉ ra lỗi này và nó đã được sửa ở T3. Thiết kế hiện tại là
**Estimate-then-Transport**: ước lượng posterior mean, rồi transport *không nhận text* tới phân phối dữ
liệu, rồi nội suy theo articulator.

**Bản v2.1**, sau vòng phản biện thứ hai (07 §5b), bổ sung bốn điểm:

- Mệnh đề 1' có thêm số hạng bias của $\hat\mu$.
- Hệ quả chỉ đúng với $\lambda$ vô hướng.
- Giới hạn "$p_\mu$ rời rạc": transport chỉ có lợi ở những hướng có headroom. Từ đó có các gate đo
  headroom và $\rho$ trước Phase B.
- Stage 2 dùng anchor là output thật của stage 1, sinh on-the-fly lúc train.

Quy ước:

- Thời gian flow $u\in[0,1]$, với $u=0$ là nguồn và $u=1$ là dữ liệu.
- $c$ là câu text, $Y\sim p(\cdot\mid c)$ là motion tham chiếu **duy nhất**.
- $\mu(c)=\mathbb E[Y\mid c]$.
- $x\in\mathbb R^{T\times d}$, các toán tử thời gian áp độc lập cho từng kênh.
- **[cần kiểm chứng]** đánh dấu chỗ chưa có bằng chứng thực nghiệm.

---

## T1. Toán tử resampling phổ trực chuẩn

$C_n$ là DCT-II trực chuẩn: $C_n[k,i]=\alpha_k\cos(\pi(i+\frac12)k/n)$, $\alpha_0=\sqrt{1/n}$, $\alpha_{k>0}=\sqrt{2/n}$.
Với $c\le T$:

$$D_{T\to c}=C_c^\top[I_c\ 0]C_T,\quad \tilde D_{T\to c}=\sqrt{c/T}\,D_{T\to c},\quad \tilde U_{c\to T}=\sqrt{T/c}\,D_{T\to c}^\top,\quad \Pi_c=\tilde U\tilde D=D^\top D .$$

**Tính chất.**

- (P1) $DD^\top=I_c$.
- (P2) $\Pi_c$ là phép chiếu trực giao lên $c$ tần số DCT thấp nhất.
- (P3) $\tilde D_{T\to c}=\tilde D_{c'\to c}\tilde D_{T\to c'}$ với mọi $c<c'<T$; không cần điều kiện chia hết.
- (P4) Giữ biên độ, và $\tilde D x$ là mẫu chính xác của nội suy band-limited theo DCT.
- (P5) Với $\xi\sim\mathcal N(0,I_c)$ thì $\mathrm{Cov}(\tilde U\xi)=(T/c)\Pi_c$.

**Hai lưu ý.**

1. **Hiện tượng Gibbs.** Low-pass lý tưởng gây ringing quanh các điểm khởi đầu ký hiệu hay các
   đoạn hold sắc nét. Vì vậy ta chỉ dùng $\tilde D$ để *định nghĩa band* và để tạo grid, không dùng nó
   để hiển thị kết quả trung gian.
2. **Giữ biên độ không có nghĩa là giữ SNR.** Nếu thêm nhiễu đơn vị trên grid thô thì SNR của mỗi tần
   số khác với SNR trên grid mịn, theo tỉ lệ $c/T$. Cùng một $\sigma$ không có nghĩa là cùng SNR hiệu
   dụng giữa các stage. Điều này cần tính đến khi quét noise scale (A16).

---

## T2. Rủi ro một-reference và họ ước lượng nội suy

### T2.1 Đồng nhất thức rủi ro (Mệnh đề 1)

Cho $G$ là một biến ngẫu nhiên bất kỳ, độc lập với $Y$ khi biết $c$. Ví dụ: một mẫu posterior, hoặc
$T(\mu+\sigma\varepsilon)$. Cho $\Lambda=\mathrm{blockdiag}(\lambda_p I_{d_p})$ là ma trận đường chéo khối
theo articulator $p\in\{B,L,R,F\}$, với $\lambda_p\in[0,1]$. Đặt $X_\Lambda=\mu+\Lambda(G-\mu)$. Khi đó

$$\mathbb E\|X_\Lambda-Y\|^2=\underbrace{\mathbb E\|\mu-Y\|^2}_{\mathrm{MMSE}}+\sum_p\lambda_p^2\,\mathbb E\|G_p-\mu_p\|^2 .$$

*Chứng minh.* Khai triển bình phương. Số hạng chéo là
$\mathbb E\langle\mu-Y,\Lambda(G-\mu)\rangle=\mathbb E\big[\langle\mathbb E[\mu-Y\mid c],\,\Lambda\,\mathbb E[G-\mu\mid c]\rangle\big]=0$,
vì $G\perp Y\mid c$ và $\mathbb E[Y\mid c]=\mu$. Các khối articulator trực giao trong không gian đặc trưng. ∎

**Mệnh đề 1' (với μ̂ thật của model, có bias).** μ-net không cho đúng $\mu^\star=\mathbb E[Y\mid c]$, vì
hai lý do: dữ liệu hữu hạn, và PES cố ý kéo nghiệm về phía median. Khi thay $\mu$ bằng $\hat\mu$,

$$\mathbb E\|X_\Lambda-Y\|^2=\mathbb E\|\hat\mu-Y\|^2+\sum_p\lambda_p^2\,\mathbb E\|G_p-\hat\mu_p\|^2+2\,\mathbb E\big\langle\hat\mu-\mu^\star,\ \Lambda(\mathbb E[G\mid c]-\hat\mu)\big\rangle .$$

Số hạng cuối có thể âm hoặc dương. Nếu T-net dịch kỳ vọng có điều kiện của nó về phía $\mu^\star$, số hạng
này có lợi; nếu dịch theo hướng ngược lại, nó có hại. Số hạng được ước lượng trên dev bằng 8 lần rút
$\varepsilon$ cho mỗi câu (06, E2-pilot).

**Điều kiện độc lập.** $G\perp Y\mid c$ đúng trên dev và test **bất kể có cross-fit hay không**. Lý do:
$G$ chỉ phụ thuộc vào dữ liệu train và nhiễu, còn $Y$ là một lần rút mới. Đây chính là điều kiện Markov
$X$–$Y$–$\hat X$ mà Freirich et al. cần. Có một ngoại lệ: các câu gần trùng hoặc cùng signer xuất hiện ở
cả train lẫn test sẽ làm $G$ và $Y$ tương quan. Vì vậy phải báo cáo tách riêng hai nhóm: text đã thấy
và text chưa thấy.

**Hệ quả 1.** Rủi ro chỉ phụ thuộc vào **độ toả của $G$ quanh mean**, $\mathbb E\|G_p-\mu_p\|^2$.
Để đơn giản, phát biểu cho $\hat\mu=\mu^\star$.

- Với một mẫu posterior $G=S\sim p(\cdot\mid c)$, đại lượng này là $V_p$. Khi đó $X_\Lambda$ chính là
  "shrink-to-mean".
- Trong mọi $G$ có phân phối biên đúng bằng $p_x$, $G=T^\star(\mu)$ cực tiểu hoá **tổng** độ toả
  $\mathbb E\|G-\mu\|^2=W_2^2(p_\mu,p_x)\le\mathrm{MMSE}$.
- Kết luận: với **$\lambda$ vô hướng**, transport trội sampling ở mọi $\lambda$.
- Với $\Lambda$ theo articulator, kết luận **không** còn được bảo đảm. OT plan kết hợp chỉ cực tiểu tổng,
  nên độ toả của một khối riêng lẻ có thể vượt $V_p$.
- Cùng giá trị $\lambda$ cũng **không** có nghĩa là cùng chất lượng perception theo kinematics:
  frontier của Freirich được định nghĩa bằng $W_2$.

Đây là kết quả của Freirich, Michaeli & Meir (NeurIPS 2021, arXiv 2107.02555). Họ chứng minh đường
distortion–perception (MSE so với $W_2$) được đạt bởi các ước lượng nằm trên **geodesic** giữa
$\mu$ và $T^\star(\mu)$, tức đúng họ $X_\lambda$ với một $\lambda$ chung.

**Mở rộng theo articulator.** $\Lambda$ có nhiều phần tử, và đồng nhất thức ở Mệnh đề 1 đúng chính xác.
Việc *tối ưu* dưới các ràng buộc perception riêng cho từng articulator chỉ là **giả thuyết**. Lý do:
OT map của phân phối kết hợp nhìn chung không tách được theo articulator.

### T2.2 Metric thật không phải L2 bình phương

`_pairwise_jpe` và `pairwise_part_cost` lấy **trung bình khoảng cách Euclid không bình phương** của các
khớp, cộng dọc theo đường DTW. Hệ quả:

- Ước lượng Bayes theo từng frame là **geometric median** của từng khớp, không phải mean.
- Với một phân phối Gaussian, tỉ lệ rủi ro giữa sampler và ước lượng tối ưu là $\sqrt2$, không phải 2.
- DTW tha thứ cho lệch nhịp, nên ước lượng tối ưu tiến gần một DTW-barycenter (soft-DTW, 1703.01541).

Mệnh đề 1 vì vậy là **phép tương tự định lượng**, không phải định lý cho DTW-JPE. Kết quả định tính vẫn
giữ: transport tốt hơn sampling ở cùng độ chân thực, và rủi ro tăng đơn điệu theo $\lambda$. Nhưng các
con số chính xác phải được đo (E0.8, E2). Đây cũng là lý do μ-net được huấn luyện thêm bằng loss khoảng
cách *không bình phương* trong không gian FK (04 §5.2), để xấp xỉ median theo đúng metric.

---

## T3. Transport theo cặp kiểu PMRF, và vì sao **không** cho stage mịn nhận text

**Cách huấn luyện.** Cho $\mu_i$ là dự đoán của μ-net trên clip $i$, và $x_i$ là motion tương ứng.
Nguồn là $z_0=\mu_i+\sigma\varepsilon$, đích là $z_1=x_i$, đường đi tuyến tính
$z_u=(1-u)z_0+u\,z_1$. Network $f_\theta(z_u,u)$ dự đoán $\hat x$ (x-pred), từ đó suy ra velocity
$\hat v=(\hat x-z_u)/(1-u)$.

- Nếu $f_\theta$ **không** nhận $c$, ODE thu được là một transport từ $p(\mu+\sigma\varepsilon)$ tới $p_x$.
- Đây là cơ chế của PMRF (Ohayon et al., 2410.00418).
- Tuy vậy, cái học được là **rectified flow của coupling dữ liệu** $(\mu,Y)$, **không** phải $T^\star$. Rectified
  flow (Liu et al., 2209.03003) chỉ bảo đảm chi phí vận chuyển không tăng so với coupling ban đầu, tức
  $\mathbb E\|G-\mu\|^2\lesssim\mathbb E\|Y-\mu\|^2+d\sigma^2$: chỉ đảm bảo **hoà** với sampling, không đảm bảo thắng.

**Giới hạn lớn: $p_\mu$ là rời rạc.** Mỗi câu chỉ có một $\mu$. Định lý Brenier, cùng lập luận tối ưu của
PMRF, đều cần nguồn liên tục tuyệt đối, điều kiện này ở đây không thoả. Trong không gian khoảng $10^4$
chiều, nhiễu đẳng hướng $\sigma\varepsilon$ gần như trực giao với mọi hiệu $\mu_i-\mu_j$, nên với hầu hết
$\sigma\lesssim1$, nguồn $\mu+\sigma\varepsilon$ vẫn "nhận ra" được câu nào đang được xét.

**Mô hình Gaussian theo từng hướng.** Xét một hướng, với
- $a$: phương sai giữa các câu của $\mu$;
- $b$: phương sai trong cùng một câu (phần dư).

Khi đó
$$\frac{W_2^2}{\mathrm{MMSE}}=\frac{(\sqrt{a+b}-\sqrt a)^2}{b}\ \xrightarrow{a/b\to0}\ 1 .$$
Ở những hướng có $a\ll b$, gồm nhịp, biên độ, phong cách signer và các dải cao, cách duy nhất để đẩy
$\mathrm{law}(\mu+\sigma\varepsilon)$ thành $p_x$ là khuếch đại $\varepsilon$ lên khoảng $\sqrt b/\sigma$. Nghĩa là
T-net lý tưởng **cũng hành xử như một sampler** ở những hướng đó, giống hệt lỗi đã chỉ ra cho phiên
bản có text. Khi $\sigma\to0$, hằng số Lipschitz tăng vô hạn. Với 4/3/3 bước Euler, kết quả thực tế sẽ
là một output nhoè, tức "ước lượng do tình cờ", lặp lại ảo giác của v1.

**Hệ quả cho kế hoạch.** Lợi ích của transport phải được **đo trước** Phase B, bằng hai phép đo:

1. **Gelbrich headroom.**
   $\eta=W_2^2\big(\mathcal N(m,\Sigma_\mu),\mathcal N(m,\Sigma_\mu+\Sigma_n)\big)/\mathrm{tr}\,\Sigma_n$, tính theo từng
   band DCT và từng articulator. Dùng $\mu$ out-of-fold, hoặc output 1-NFE của SignNMFlow-reimpl làm
   $\mu$; $\Sigma_n=\mathrm{Cov}(x-\mu)$, co hệ số bằng Ledoit–Wolf. Cận Gelbrich là cận *dưới* của $W_2$,
   nên $\eta\approx1$ ở một band nghĩa là band đó **không có headroom**: transport không thể tốt hơn
   sampling ở đó.
2. **Pilot stage 1.**
   $\rho=\mathbb E\|G-\mu\|^2/\mathbb E\|Y-\mu\|^2$ đo trên dev, quét theo $\sigma_1$. $\rho\approx1$ nghĩa là
   T-net đang là sampler. Gate được đăng ký trước: $\rho<0.8$.

Đây là phép thử **chính** cho đóng góp phương pháp của v2 (06, E0.9 và E2-pilot).

**Nguy cơ ghi nhớ.** Mỗi $\mu_i$ chỉ có đúng một $x_i$. Một T-net 40M tham số, train khoảng 170 epoch,
hoàn toàn có thể học thuộc ánh xạ tất định $\mu_i\mapsto x_i$, tức không có độ toả nào được học. Phải
theo dõi $\rho$ và x-pred loss trên dev ở mỗi snapshot, và dừng sớm khi cần.

**Cảnh báo. Đây là lỗi của bản v1.** Nếu $f_\theta$ được điều kiện hoá trên $c$, hoặc trên $\mu$, thì
nguồn $\mu+\sigma\varepsilon$ chỉ còn là một *offset đã biết* của điều kiện. Khi đó:

- $\mathbb E[x\mid z_u,c]=\mathbb E[x\mid z_u-(1-u)\mu,c]$;
- tại nghiệm tối ưu, flow sample $p(x\mid c)$ với **mọi** $\sigma$.

Đây là trường hợp của Grad-TTS và PriorGrad (2106.06406): prior phụ thuộc dữ liệu giúp hội tụ nhanh
hơn, nhưng không thay đổi phân phối đích. Hệ quả là stage mịn có text, dù xuất phát từ plan cộng nhiễu,
**không** cho "ước lượng dải thấp" ở nghiệm tối ưu. Nó chỉ trông như vậy khi model học chưa hoàn hảo
và dùng ít bước. Thiết kế mặc định vì vậy dùng transport **không nhận text**. Biến thể có text giữ lại
làm ablation (A6), và được mô tả trung thực là một conditional sampler có prior dịch chuyển.

**Rủi ro thực tế, quan trọng với dữ liệu nhỏ.** Trên tập train, μ-net đã thấy từng clip, nên $\mu_i$ gần
$x_i$ hơn hẳn so với trên tập test. Nếu train transport trên $\mu_i$ in-fold, nó học một ánh xạ gần đồng
nhất, và lúc test sẽ không đủ mạnh để làm sắc một plan mờ.

- **Cách khắc phục:** cross-fitting. Chia train thành $K$ fold (mặc định $K=2$). μ-net của fold $j$
  được huấn luyện trên các fold còn lại, rồi dự đoán $\mu$ cho fold $j$. Transport được huấn luyện trên
  các $\mu$ out-of-fold này.
- **Input của T-net lúc test** phải là μ từ **một fold model**, chọn ngẫu nhiên. Nếu dùng μ từ model
  huấn luyện trên toàn bộ train, nó sẽ tốt hơn mọi μ mà T-net từng thấy, và T-net sẽ sửa quá tay.
- Thành phần "ước lượng" $\hat\mu$ trong $X=\hat\mu+\Lambda(G-\hat\mu)$ vẫn có thể dùng model toàn phần.
  Mệnh đề 1' vẫn đúng với bất kỳ $\hat\mu$ nào, miễn $G\perp Y\mid c$.
- Có thể chọn K = 5 thay vì 2. Nên chia fold sao cho tỉ lệ text trùng lặp giữa các fold giống tỉ lệ
  trùng lặp giữa train và dev (kiểm tra CSL-Daily).
- Phương án rẻ hơn là pseudo-plan, tức GT làm mượt và co về mean. Phương án này lệch thống kê so với
  plan thật (ablation A19).

Trong SignNMFlow-reimpl của repo, val loss tăng sau epoch 100. Đó là dấu hiệu overfitting đủ lớn để
cross-fitting quan trọng **[cần kiểm chứng: A19]**.

---

## T4. Transport phân cấp có ràng buộc band (MotionHiFlow trong vai trò transport)

Stage $k=1..K$ dùng grid $c_1<\dots<c_K=T$, với $y_k=\tilde D_{T\to c_k}x$ và $\mu_k=\tilde D_{T\to c_k}\mu$.

- **Stage 1 (transport dải thấp):** $z_u=(1-u)(\mu_1+\sigma_1\varepsilon)+u\,y_1$, đích là $y_1$.
- **Stage $k\ge2$ (sinh band mới).** Gọi $\Pi=\Pi_{c_{k-1}}$, xét trên grid $c_k$.
  - Phần dải thấp được **giữ cố định**: $B=\tilde U_{c_{k-1}\to c_k}\hat y_{k-1}$. Lúc train, $B$ là
    $\Pi y_k$ cộng augmentation.
  - Chỉ phần bù được sinh: $h_u=(1-u)\big((I-\Pi)\mu_k+\sigma_k(I-\Pi)\varepsilon\big)+u\,(I-\Pi)y_k$.
  - Trạng thái đầy đủ là $z_u=B+h_u$, và output được chiếu lên $(I-\Pi)$.
- Ở mọi stage, x-pred cho $\hat v=(\hat h-h_u)/(1-u)$, nên bước Euler cuối rơi đúng vào $\hat h$.
- Mỗi stage đi **tới sạch** (như Decomposable Flow Matching, 2506.19839). Không có renoise giữa các
  stage, vì phần đã sinh được giữ nguyên.

**Phân rã ra sao.**

- **Dải thấp** đến từ transport của plan. Nó chỉ gần tất định ở những hướng có headroom (T3).
- **Các dải cao** được *sample có điều kiện* trên dải thấp đã có, tức $p(h\mid B,(I-\Pi)\mu)$, vì stage
  $k$ nhìn thấy $B$. Độ toả của dải cao là $\mathbb E\,\mathrm{Var}(h\mid B,\cdot)$, tức độ toả của một
  **sampler**, và không bị chặn bởi $W_2$.

Vì vậy chữ "transport" chỉ áp dụng cho band $c_1$. Hệ quả 1 cũng chỉ áp dụng cho band đó. Mệnh đề 1
vẫn cho rủi ro chính xác, với độ toả đo được.

**Tính nhất quán train/test ở stage $k\ge2$.**

- **Stage 2 (mặc định).** Lúc train, $B$ là **output thật của stage 1**, sinh on-the-fly: no-grad,
  4 NFE, trên grid $T/4$ nên chỉ tốn khoảng 1/4 số token. Cách này loại bỏ sai lệch train/test. Một
  phương án bị loại là dùng GT có $\kappa=1$ lúc inference: lúc đó T-net được báo rằng $B$ sắc như GT,
  trong khi output stage 1 thường mượt hơn. Ngoài ra, augmentation kiểu "trộn về phía plan" mô hình hoá
  sai loại lỗi: lỗi thật của stage 1 là lệch nhịp, hoặc chọn một phương án hợp lý khác.
- **Stage 3.** $B=\Pi y_3$ (GT) cộng nhiễu chỉ trong dải thấp ($0.03\,\Pi\eta$). Ablation A14 dùng
  output thật của stage 2, cache theo clip, chi phí thêm khoảng +50–60%.
- Đây là *xấp xỉ* theo lập luận cascaded diffusion, không có bảo đảm chính xác.

**Vì sao hierarchy lại hợp với transport.** Plan mờ sai khác với motion thật chủ yếu ở hai chỗ:
(i) dải thấp bị mượt và lệch nhịp so với một ký hiệu cụ thể, và (ii) dải cao gần như bằng 0. Transport
phân cấp sửa (i) trên grid thô, nơi lệch nhịp nhỏ so với bước grid, rồi sinh (ii) theo từng band.

- Đây là *động cơ*, không phải định lý.
- Bằng chứng gián tiếp: MotionHiFlow, MoScale, Decomposable Flow Matching. Phần lớn lợi ích của
  hierarchy đến từ 2 tầng đầu.
- Phải đo bằng ablation A5 (phẳng so với $K=2$ so với $K=3$).

---

## T5. Ghi chú về renoise chính xác (không phải đóng góp)

Chỉ áp dụng cho biến thể *giữ nhiễu* (không clamp), ablation A2. Sau khi upsample bằng $\tilde U$ (P5),
nhiễu ở dải thấp có độ lệch chuẩn $r\sigma_e$ với $r=\sqrt{c'/c}$. Trạng thái bắt đầu đúng phân phối lúc
train khi và chỉ khi $\frac{\sigma_s}{1-\sigma_s}\ge r\frac{\sigma_e}{1-\sigma_e}$. Lúc đó phải thêm nhiễu
$\sqrt{\sigma_s^2-a^2r^2\sigma_e^2}$ vào dải thấp ($a=(1-\sigma_s)/(1-\sigma_e)$) và $\sigma_s$ vào dải cao.

Với scales $[0.3,0.6,1]$, $\sigma_e=0.4$, $r=1.291$, ta có $\sigma_s^{\min}=0.463$. Hiệu chỉnh $\gamma=1$
mà MotionHiFlow dùng cho $\sigma_s=0.485$.

Tuy nhiên, sampler của MotionHiFlow **trừ đúng phần nhiễu đã biết** rồi renoise (denoise–upsample–renoise).
Vì vậy $\gamma$ không "sửa" một sai lệch hiệp phương sai nào. Nó chỉ chọn $\sigma_s$, và lựa chọn đó
nhất quán giữa train và test. Pyramid Flow (2410.05954) đã dẫn xuất kết quả tương tự. **Không** tuyên
bố điều này là mới.

---

## T6. DTW tích luỹ và thời lượng

`dtw_cost` trả về tổng chi phí trên đường warping có độ dài $L$, với
$\max(T_X,T_Y)\le L\le T_X+T_Y-1$. Ba hệ quả:

1. **Output ngắn hoặc tĩnh được lợi.** Các phân tích SRVQ và RobotSTAR (02 §1.3) đã ghi nhận điều này.
2. **Không suy ra được thời lượng tối ưu dưới median.** Nếu chi phí mỗi bước không đổi theo độ dài,
   lập luận chỉ dẫn tới output ngắn nhất có thể, và điều đó vô lý. Thực tế chi phí mỗi bước tăng khi
   $T_X\ll T_Y$, nên điểm tối ưu nằm ở bên trong khoảng và phải đo bằng thực nghiệm (A12).
3. **nDTW cũng có thể bị lợi dụng theo chiều ngược lại.** Chèn các frame pose nghỉ làm tăng $L$ với chi
   phí thấp.

Do đó mọi bảng phải báo cáo cùng lúc: DTW tích luỹ, nDTW, sai số thời lượng, tỉ lệ tốc độ và gia tốc,
và một metric phân phối (FD trên đặc trưng của evaluator ngoài). Không metric đơn lẻ nào là đủ.

---

## T7. Hình học quay (đã sửa cách diễn đạt)

Cả axis-angle (exp map) lẫn 6D (Gram–Schmidt) đều là ánh xạ **toàn ánh** vào $SO(3)$. Với 6D, điều này
đúng trừ một tập độ đo 0. Mọi điểm của một flow tuyến tính trong hai chart này đều decode ra một pose
hợp lệ.

Vì vậy lập luận "rời manifold" của SignNMFlow (Property 1) không áp dụng cho các thiết kế dùng output
decode qua chart. Lập luận đó chỉ đúng khi loss hoặc path được định nghĩa trên chính phép nhúng có ràng
buộc. Chọn chart nào là câu hỏi **thực nghiệm** (A3).

Chúng tôi không tuyên bố đây là phản bác SignNMFlow, trừ khi A3 cho thấy 6D/AA cộng FK loss đạt ngang
quaternion + geodesic.

---

## T8. MBR (tuỳ chọn, không phải đóng góp)

$\hat h=\arg\min_i\frac1{N-1}\sum_{j\ne i}\mathrm{DTW}(X_i,X_j)$ là ước lượng không chệch của rủi ro Bayes
**chỉ khi** các ứng viên được lấy từ $p$ thật. Transport và CFG đều làm $q\ne p$ một cách có chủ đích,
và các ứng viên dùng chung plan nên ít đa dạng. Vì vậy MBR chỉ là một tuỳ chọn decode. Nếu dùng, mọi
baseline phải được áp **cùng N**, và luôn báo cáo kèm $N=1$.

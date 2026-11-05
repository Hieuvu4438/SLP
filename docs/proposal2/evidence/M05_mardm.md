# M05 — MARDM: representation, evaluator và masked continuous generation

**D có phạm vi, chưa code.** Đã đọc §§2–4, Tables 1–6, App. A/C/D/G và Tables A1–A6. Chưa xem hết video hoặc độc lập xác nhận mọi derivation. Nguồn: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.html), [arXiv 2411.16575](https://arxiv.org/abs/2411.16575). Version/hash trong registry, snapshot `sources/mardm/`. MARDM khác MaDiS về SLG.

## Câu hỏi và representation

Paper phân tích vì sao VQ methods có điểm tốt hơn direct diffusion trên HumanML3D, bằng cách thay cả representation và evaluator. Vì thế cần tách generator quality khỏi evaluator compatibility.

263D features gồm root trajectory/height, local positions, velocities, 6D rotations và foot contacts. Paper giữ `3J+1` dimensions: 67 HumanML3D, 64 KIT-ML, gọi phần còn lại redundant vì không cần để visualize joint positions. “Redundant” phụ thuộc output: joint positions chưa xác định mọi palm orientation hoặc finger articulation cần cho avatar/SLG.

§2.2/Table 1 còn cho thấy redundancy có ích cho VQ: MoMask full rFID .029/gFID .116, essential-only .030/.200; T2M-GPT cũng giảm chất lượng khi bỏ features. Không suy compact input luôn tốt hơn.

§2.3/Table 3: evaluator nhạy với noise ở redundant features dù visualized joint trajectory không đổi. Hai nhóm khác số chiều/distribution; chưa là equal-energy/perceptual perturbation test hoặc bằng chứng mọi evaluator cũ vô giá trị.

## Method, train và inference

Deterministic temporal ResNet AE, downsample 4, L1 reconstruction trên essential features. Continuous latents được mask theo cosine schedule; bidirectional AdaLN Transformer tạo condition cho stochastic latent head. “Autoregressive” là thứ tự unmask, chưa nhất thiết temporal left-to-right.

Diffusion MLP tạo distribution cho mỗi latent thay vì deterministic regression về conditional mean. Có DDPM noise-prediction và SiT velocity-prediction variant, với path `(1−t)latent+t noise`. Decode sau khi điền các positions; editing bằng mask một vùng latent và giữ context.

Đây là adaptation của image MAR: context Transformer + continuous stochastic output head. Không mới nếu chỉ đổi domain sang SLG.

App. C: AE hidden 512, train trên 64-frame crops; context một layer/hidden 1024/16 heads; diffusion MLP 16 layers/hidden 1792. Generator tối đa 196 frames, train 500 epochs, AdamW, EMA, CFG 4.5/2.5 tùy variant. Epoch/iteration milestones còn cần code để exact reproduction. Inference cần length/mask shape; chưa xác minh predicted-duration protocol.

## Kết quả và confounds

Baselines được retrain full 263D rồi evaluate essential-only; main method train essential-only. Shared evaluator chưa đồng nghĩa shared training representation. App. A3 bổ sung baseline reform để tách một phần ảnh hưởng.

| Kiểm tra | Author-reported evidence | Diễn giải |
|---|---|---|
| Evaluator mới | Table 5 SiT FID .114/R@1 .500; MoMask .116/.490 | FID rất gần và CI chồng nhau |
| Representation/masking | Table 6 full .116/.492; no-reform 2.196/.387; no-AR .551/.435 | Đổi nhiều điều kiện, chưa cô lập từng nguyên nhân |
| Reform baseline | Table A3 MDM noise/linear 1.574/.279→essential .753/.436→latent .327/.475 | Representation góp đáng kể |
| Reconstruction | Table A1 AE rFID .004/MPJPE 1.0; RVQ .029/31.5 | Inputs khác, chưa quy toàn gap cho quantization |
| Evaluator cũ | Table A4 MARDM .061/.523; MoMask reported .045/.521, reproduced .093/.508 | Không chỉ chọn reproduction yếu hơn để nói thắng FID |
| Conversion floor | GT FID .002→round-trip .015 | Representation conversion tự tạo sai số |
| Scaling/latency | A5 tăng MLP capacity giúp; A6 2.4 s so MoMask .04/MMM .06 trên 4090 | Capacity/compute là confounds thực chất |

20 evaluation repeats/CI không là 20 training seeds. Multimodality cao chưa xác nhận variation giữ nguyên meaning, đặc biệt với sign language.

## Các diễn giải toán cần thận trọng

Gaussian corruption không đòi input distribution Gaussian: nếu terminal signal coefficient bằng 0, output là noise bất kể data distribution. Với finite terminal SNR, mismatch phụ thuộc schedule/scaling. Không dùng non-Gaussian input làm lý do bắt buộc bỏ rotations/contact.

Noise-prediction error đổi sang x0 có SNR scaling, nhưng các SD-ratio/error conventions trong paper còn cần code nếu kế thừa. Deterministic VQ embedding lookup cũng không đảm bảo uniform coordinate error sau nonlinear decoder; AE decoder cũng deterministic. Empirical diagnostics có giá trị trong scope, mạnh hơn chúng chưa được xác lập.

## Chuyển giao

Đối chứng representation/evaluator trước khi xếp hạng generator; dùng stochastic output heads khi có one-to-many targets. Không suy phải bỏ VQ, dùng SiT hoặc giữ 67D cho SLG. Cần same fitting/rotation information, matched decoder/capacity, conversion-floor measurement và independent handshape/orientation/comprehension evaluation.

# M12 — MotionStreamer: causal latent generation và learned stopping

**D có phạm vi, chưa code.** Đã đọc §3, §4/Tables 1–4, App. A/E/F; chưa mọi latent/architecture sweep hoặc video. Nguồn: [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_MotionStreamer_Streaming_Motion_Generation_via_Diffusion-based_Autoregressive_Model_in_Causal_ICCV_2025_paper.html), [arXiv 2503.15451v3](https://arxiv.org/html/2503.15451v3). Snapshot `sources/motionstreamer/`.

## Task và representation

Nhận prompts lần lượt, sinh online và giữ prefix đã xuất. Đây là temporal causality của generation, không phải causal linguistic understanding. SLG nhận cả câu từ đầu có thể cần future context cho translation/coarticulation; streaming chưa là ràng buộc bắt buộc.

Representation 272D/30 FPS gồm root planar velocity/angular features và positions/velocities/rotations của 22 SMPL joints. Rotations lấy từ AMASS; App. E chỉ ra inverse kinematics từ positions có thể bỏ twist. Có liên quan orientation fidelity, nhưng chưa có detailed fingers/face.

## Kiến trúc và training

Causal temporal encoder/decoder downsample 4 thành 16D latents. Dù tên TAE, model có Gaussian latents, reparameterization và KL theo σ-VAE, không phải deterministic AE. Thêm root loss weight 7. Causal decoder xuất frames mới mà không decode lại toàn sequence.

Causal LLaMA-style Transformer 12 layers/12 heads/hidden 768 tạo context cho diffusion MLP chín layers/hidden 1792. T5-XXL encode text; appendix dùng 50 DDPM sampling steps cho next continuous latent.

Two-Forward training: pass đầu tạo latents, thay một phần reference history bằng chúng; pass sau backprop với mixed history. Cosine schedule tăng replacement 0→1. Điều này giảm mismatch nhưng chưa đồng nhất với full free-running error distribution.

Mixed training gồm HumanML3D text-motion không history và BABEL adjacent segments có history. Gain continuation còn có thể do thêm transition data/supervision, không riêng architecture.

## Duration và inference

Encode all-zero “impossible pose” thành end latent, dừng khi generated latent gần nó theo threshold rồi chuyển prompt. Đây là learned stopping, khác external length predictor.

Threshold có thể stop nhầm hoặc kéo dài. Các phần đã đọc chưa đủ stop-error calibration và chưa code-audit Table 1 có dùng cùng stopping/max-length path như demo. Không tự điền unknown length protocol thành predicted-length evaluation.

## Evidence và trade-offs

Dùng HumanML3D/BABEL với 272D/30 FPS và evaluator do tác giả train; baselines retrain theo báo cáo. Không so trực tiếp FID 11.790 của protocol này với MoMask .045 ở protocol khác.

| Kiểm tra | Author-reported evidence | Giới hạn |
|---|---|---|
| Codec learnability | Table 3 AE rFID .001/MPJPE 1.7 mm, generation FID 43.828; causal TAE .661/22.9 và 11.790 | Near-identity reconstruction chưa dễ sinh |
| Causal representation | Noncausal VAE gFID 19.902 so 11.790 | Chỉ hỗ trợ trong causal AR setup |
| Stochastic head | Table 4 bỏ diffusion head FID 59.296 | Không chứng minh mọi regression ở task khác vô ích |
| Two-Forward | No strategy 12.944/R@3 .842; full 11.790/.859 | Cần same-cost history-corruption controls |
| Text backbone | CLIP 14.234/.791; T5-XXL 11.790/.859 | Không quy toàn gain cho continuous latents |
| Transitions | Table 2 PJ .04/AUJ .90; FlowMDM .06/.51 | PJ gần GT .03 hơn nhưng AUJ kém hơn |

First-frame latency khác end-to-end clip latency. Continuous latents vẫn có compression error. App. F thừa nhận causal streaming khó in-betweening/local editing; prefix cố định có thể hạn chế sửa early translation decisions.

## Chuyển giao

Causal decoding, inference-like history training và learned stopping là ba cơ chế riêng; có thể chuyển một phần mà không giữ toàn streaming pipeline. SLG cần full-sentence/streaming controls, generated/reference history, EOS/end-latent, duration/coverage errors, orientation tests và same transition-data budget. Chưa có evidence sign-order hoặc NMM correctness từ body benchmarks này.

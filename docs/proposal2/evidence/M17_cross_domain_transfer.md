# M17 — Chuyển giao từ image, speech và robotics theo cơ chế cụ thể

Các paper dưới đây có **D theo phạm vi riêng**, chưa code hoặc reproduction. Chúng giải thích kỹ thuật gốc; không là evidence cùng effect size khi chuyển sang SLG.

## MAR: autoregression không bắt buộc discrete vocabulary

Nguồn: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/66e226469f20625aaebddbe47f0ca997-Abstract-Conference.html), [arXiv 2406.11838v3](https://arxiv.org/html/2406.11838v3). Đọc §§3–5/Tables 1–2, architecture/representation ablations, App. D.3/limitations; chưa mọi system-benchmark entry.

Transformer tạo context z, conditional diffusion MLP mô hình hóa mỗi next continuous latent; denoising gradients về Transformer. Continuous AR khác deterministic next-latent MSE. Random-order masked generation cho known/unknown positions tương tác bidirectionally, loss ở unknowns; sinh nhiều positions mỗi round giảm sequential rounds nhưng per-token heads chưa tự đảm bảo articulation coordination.

ImageNet 256², pretrained KL/VQ codecs stride 16, Large Transformer khoảng 400M; default 100 diffusion steps/head và 64 masked iterations. Không cùng low-data SLG regime. Table 1 diffusion-loss FID 1.98 so CE 3.69 có CFG, nhưng còn khác KL/VQ tokenizers; Table 2 pre-quantized VQ latents giúp tách một phần confound. Diffusion head không tự xóa codec ceiling.

Image L2 regression baseline FID trên 100 chưa chứng minh L2 vô ích cho geometry correction đã có strong conditional input. Spatial random order cũng chưa phù hợp temporal order/NMM. Cơ chế đã chuyển sang motion qua MARDM/MoLingo/MotionStreamer, nên chỉ dùng diffusion/flow head chưa là novelty SLG.

## Glow-TTS: train-only alignment và predicted duration

Nguồn: [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/5c3b99e8f92532e5ad1556e53ceea00c-Abstract.html), [arXiv 2005.11129v2](https://arxiv.org/html/2005.11129v2). Đọc §3/§4/§5.1–5.3/App. B.1; chưa deep voice conversion.

Text encoder tạo Gaussian statistics mỗi phoneme; invertible normalizing flow ánh xạ mel→latent để tính likelihood. Monotonic Alignment Search chỉ ở lại text index i hoặc tiến từ i−1, với surjective constraint: không skip/reorder, mỗi unit có frame. Đây là giả định đọc phoneme order, chưa phù hợp spoken-to-sign translation.

Train tìm hard best alignment bằng reference audio/current model rồi update likelihood; frame counts làm duration targets, predictor log-MSE và stop-gradient tới encoder. Inference dự đoán durations, sample latents và parallel decode, không dùng reference alignment.

LJSpeech 24 giờ/LibriTTS 54 giờ, phoneme inputs, WaveGlow vocoder. MOS 4.01±.08 so Tacotron2 3.88±.08, nhưng attention-error test 100 sentences có bốn errors so ba. Constraint chưa cải thiện mọi metric. Normalizing flow có invertibility/likelihood khác rectified flow.

**SLG:** có thể học train-only timing supervision rồi tự dự đoán ở test, nhưng cần target-ordered units/reordering hoặc alignment class khác. Không áp MAS trực tiếp lên spoken-word order rồi coi đã giải translation.

## F5-TTS: bỏ explicit local alignment vẫn cần total duration

Nguồn: [ACL 2025](https://aclanthology.org/2025.acl-long.313/), [arXiv 2410.06885v3](https://arxiv.org/html/2410.06885v3). Venue verified; deep reading dùng v3, không claim byte identity. Đọc §§3–5/App. B.2; chưa mọi solver/data/human table.

Train text-guided mel infilling từ noisy mel, unmasked audio prompt và transcript. Characters/pinyin được pad bằng filler tới mel length; ConvNeXt refine text trước concatenation, DiT AdaLN-zero học conditional flow matching. Không có phoneme-level forced alignment/duration model nhưng **global duration vẫn ước lượng** từ audio-reference length và target/reference character ratio (§3.1).

Base khoảng 95K giờ English/Chinese, 335.8M parameters, 1.2M updates/tám A100/hơn một tuần. Small 155M/945 giờ dùng architecture ablation. Sway Sampling đổi integration grid ở inference, không cần retrain trong setting này.

App. B.2 small-model estimated-duration WER 4.17 so GT-duration 3.87. Bỏ audio prompt WER 3.22 nhưng speaker similarity .21 so .54. Pure DiT không text refinement khó alignment; thêm ConvNeXt sang E2TTS không tự giúp. Conditioning/architecture/training tương tác; không có universal module gain.

**SLG:** có thể khảo sát modality-specific conditioning và auxiliary infilling, nhưng characters/speech cùng order, audio prompt cung cấp rate/identity còn text-only SLG không có. “Không explicit alignment” không là “không cần duration/order”.

## ACT: action chunks và ensemble cho cùng timestamp

Nguồn: [RSS 2023](https://roboticsproceedings.org/rss19/p016.html), [arXiv 2304.13705v1](https://arxiv.org/html/2304.13705v1). Đọc IV–VII/App. F; chưa mọi hyperparameter table.

Nhận cameras/current joints, dự đoán k future actions. Train CVAE posterior nhìn reference action chunk/proprioception; decoder nhận observations/style latent, L1+KL. Test bỏ posterior, đặt z=0, không dùng GT future actions.

Query mỗi timestep tạo overlapping chunks; ensemble trộn predictions **cho cùng timestamp**, không trực tiếp average adjacent frames. Cần observation feedback và thêm inference cost. Đây là closed-loop 14-DoF bimanual robot control, chưa articulated fingers/NMM.

VI-A không ensemble: success k=1 khoảng 1%, k=100 khoảng 44%, rồi giảm với chunks dài hơn. Ensemble thêm khoảng 3.3 points cho ACT nhưng hại VINN. VI-B CVAE human demonstrations 35.3% so 2% không CVAE; scripted data ít khác. Generative objective có lợi phụ thuộc data distribution.

Real-world comparisons một training seed/25 trials; simulation ba seeds/50 evaluations. Không gộp thành multi-seed robustness cho mọi task. Hardware/demonstrations/feedback là điều kiện của kết quả.

**SLG:** chunking, generated-context training và same-time consistency là khả năng chuyển giao. Offline SLG không có closed-loop environment feedback; averaging rotations/handshapes có thể sai manifold hoặc meaning. Cần phase alignment, geometric projection và semantic preservation, không coi smoothing là coarticulation đúng.

## Tổng hợp

Chuyển được các nguyên lý distribution modeling, train-only alignment, condition refinement và temporal consistency. Không copy image spatial order, phoneme monotonicity, audio-reference rate hoặc robot feedback. Main proposal phải giải thích rõ các khác biệt input/supervision trước khi gọi một kỹ thuật là nguồn method.

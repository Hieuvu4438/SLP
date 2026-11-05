# M16 — Composition, duration và executable plans là những quyết định khác nhau

## FlowMDM: joint transition realization với timeline cho sẵn

**D có phạm vi:** §3, §4/Tables 1–4, supplementary A/B; chưa mọi sweep/video/code. Nguồn: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Barquero_Seamless_Human_Motion_Composition_with_Blended_Positional_Encodings_CVPR_2024_paper.html), [arXiv 2402.15509v1](https://arxiv.org/html/2402.15509v1). Snapshot `sources/flowmdm/`.

Task nhận descriptions **cùng durations/boundaries**, không tự tạo action order/duration. Dù tên FlowMDM, đây là diffusion x0 prediction, 1.000 steps, không rectified flow.

Bidirectional denoising cho trước/sau cùng thích nghi ở transitions. Blended positional encoding dùng absolute positions giai đoạn đầu, relative RoPE/local attention về sau. Training xen kẽ hai chế độ; sampling chọn lịch chuyển. Attention ban đầu giới hạn subsequence nên cross-segment intention planning còn hạn chế.

Pose-centric cross-attention đặt pose+condition vào query nhưng pose vào key/value, nhằm giảm unseen condition combinations tại boundaries. Đây là bias cho một train–test mismatch, không quy tắc text luôn phải rời K/V.

BABEL 135D/30 FPS, HumanML3D 263D/20 FPS, CLIP 512D, tám Transformer blocks, L2 x0/Adam 1e-4. BABEL training có adjacent-action clips, nên không gọi mọi result là không dùng transition information. Composition protocol tới 32 descriptions/sequence, durations đã có.

Table 4 HumanML3D: absolute-only R@3 .689/transition FID 3.90/AUJ 3.40; relative-only .531/2.53/.58; blended .685/1.38/.51. Blending cân bằng, không thắng mọi trục. Table 3 BABEL PCCAT R@3 .702 thấp hơn CAT .719; CAT còn tốt hơn một số jerk metrics. Multiple conditions trong BABEL training là một cách giải thích, chưa proof của nguyên nhân duy nhất.

Subsequence/transition metrics được báo riêng; jerk thấp chưa là linguistic coarticulation đúng. Main/supplement ký hiệu DoubleTake * không hoàn toàn nhất quán; exact reproduction cần source/config, không suy từ một ký hiệu.

**Chuyển giao:** phân biệt đúng timeline với realization liên tục theo timeline đó. SLG cần predicted-plan main inference, oracle-plan riêng. Smooth seams không đủ nếu mất holds/strokes/sign identity.

## Plan, Don’t Pose / Text2BFM: sinh executable behavioral programs

**D có phạm vi:** §3, §4/Tables 1–6, App. A; chưa độc lập chứng minh propositions App. B, code/simulator. [ArXiv 2605.29906v2](https://arxiv.org/html/2605.29906v2), preprint/venue chưa xác minh. Snapshot `sources/plan_dont_pose/`.

Pretrained behavioral foundation model nhận state/behavior latent để xuất controls; environment rollout tạo motion. Frozen backward map/future lookahead chuyển train motion sang policy latents; variational bottleneck nén program sequence. Loss gồm latent MSE, KL, policy-distribution KL và text–program contrastive alignment. Generator dùng conditional flow matching, frozen policy thực thi.

Planning tách low-level execution theo một executable prior. Motion ngoài reachable policy/action space không tự xuất hiện khi thêm text plan. Vì thế phải biết prior có đủ finger/face control trước khi dùng cho SLG.

Policy pretrain trên HY-Motion qua MetaMotivo là extra data/compute so HumanML3D-only baselines. Bottleneck/generator report 17/28 giờ A100 80GB, chưa toàn policy pretraining. Inference còn cần initial humanoid state và program shape/length; chưa audit duration prediction từ text-only.

Table 1 R@3 .876/matching distance 2.498 nhưng FID 1.172 kém MoMask .045. Table 3 average pooling FID 2.373, uncompressed 1.689, bottleneck no-semantic 1.272, full 1.172: matched frozen policy mới tách được compression/alignment effects. Table 5 flow 16 steps FID 1.172, diffusion 50 steps 1.672, nhưng diffusion R@3 .885 cao hơn flow khoảng .876. Không coi flow thắng semantics vô điều kiện.

Composition order accuracy N=3 .671 so joint .395, N=4 .509 so .322. Scoring dùng known boundaries cho Compose và uniform partitions cho joint; khác boundary information/conditioning là confound, chưa riêng latent representation.

**Chuyển giao:** executable prior nhắc phải kiểm tra representational reachability. Future-state teacher chỉ dùng training; inference không được nhận nó. Body policy thiếu articulators thì chưa là backbone SLG khả thi, dù plan text đúng.

## Tổng hợp

Cần cả đúng nội dung/order/timing, realization liên tục và bảo toàn distinctions. Given-boundary benchmarks chỉ xác minh một phần; physical priors có thể giới hạn behavior coverage. Hai paper hỗ trợ tách các câu hỏi, chưa xác lập một planner chung tốt nhất cho SLG.

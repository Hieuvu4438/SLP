# M06 — Motion tokenizers có giữ gesture semantics không?

**D có phạm vi, chưa code.** Đã đọc main/appendix, nhất là §§3–6, Tables 1–5 và limitations của [Do Motion Tokenizers for Co-Speech Gesture Generation Encode Gesture Semantics?](https://arxiv.org/abs/2610.03765). Preprint, chưa xác minh venue. Snapshot `sources/gesture_token_semantics/`.

## Bài toán và thiết kế

Đây là representation probing, không phải một generator SOTA. Dataset có 778 semantic gesture clips TED/Semantix, 19 descriptors thuộc dynamics, geometry, handedness, handshape và communicative properties. Nhãn từ skeleton và manual annotation là hai loại supervision khác nhau.

Tokenizer có hai VQ streams cho 38 joints: tám body, 30 finger; hand coordinates wrist-relative. Encoders/codebooks riêng, shared decoder. Train position reconstruction + commitment β=1, EMA .99 và dead-code reset trên BEAT+TED. Sweep 27 configurations: latent 64/128/256, codebook 1024/2048/4096, downsample 4/8/16. Main dùng best-reconstruction L256/K2048/d4; đối chiếu thêm EMAGE tokenizer.

Probes nhận mean-pooled embeddings hoặc normalized code histograms, logistic regression với balanced weights, five-fold CV/macro-F1; thêm one-hidden-layer MLP width 128. Code activation analysis dùng JS divergence, permutation tests và BH correction. Sum code-presence MI không phải mutual information của toàn sequence.

## Evidence

Table 4: handshape bảy classes có linear F1 .155, one-hot .197, reported random .143; handedness .734, một số depth descriptors trên .89. Motion category 19 classes .105 so .053, communicative intent 15 classes .121 so .067. Không so F1 khác class count như cùng difficulty; manual labels sparse/subjective.

Table 5 có motion-category code trace q=.006 dù pooled probe yếu; handshape q=.697 và communicative-intent q=.063 chưa qua threshold. Weak pooled probe vì vậy không có duy nhất một cách giải thích là thông tin đã mất. Reconstruction error cũng chưa là proxy đủ cho mọi communicative property.

Limitations thừa nhận chỉ hai architectures, 778 clips, chưa corpus-mixture ablation hoặc temporal probes. Appendix dùng lời mạnh hơn “genuinely absent”; review giữ mức thận trọng: chưa chứng minh thông tin hoàn toàn vắng hoặc sequence-aware model không truy xuất được.

## Confounds và giới hạn

Mean pooling/histograms bỏ order; shallow probes chưa đọc được mọi temporal property. Cần raw-input, pre-quantization và sequence-aware probes để định vị loss do fitting, encoder, quantizer hay pooling.

Train BEAT+TED và probe TED chưa được mô tả đủ source-video/identity grouping để xác nhận signer-independent generalization. Không tự suy leakage, cũng không coi five-fold CV là đã loại mọi overlap.

Mốc 1/C chỉ là random-F1 heuristic; expected macro-F1 còn phụ thuộc priors và random prediction distribution. Empirical permuted-label controls cùng folds phù hợp hơn. MI/JS có thể tận dụng corpus/part correlations; chọn stream MI cao nhất cần kiểm tra selection/multiple-testing effects.

Không có SLG generation hoặc human comprehension study. Không dùng nguồn này để nói SOKE/MaDiS thất bại vì VQ bottleneck.

## Chuyển giao

Giá trị chính là thiết kế diagnostics cho handshape, orientation, location và bimanual coordination bên cạnh MPJPE/FID. Muốn kết luận về SLG phải đo cùng pipeline ở fitted input, continuous latent, quantized latent và decoded motion. Nếu raw input đã không giữ contrast, đổi tokenizer không tự khôi phục được; nếu temporal probes đọc được, pooled-probe failure không đủ kết luận mất semantic information. Đây là giả thuyết để kiểm tra, chưa là lý do chọn discrete hoặc continuous generation.

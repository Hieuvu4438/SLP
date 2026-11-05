# M07 — MoMask: residual quantization và masked generation

**D trên arXiv v1, chưa code.** Đã đọc main §§3–5, Tables 1–2, limitations; chưa mọi video. Nguồn: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_MoMask_Generative_Masked_Modeling_of_3D_Human_Motions_CVPR_2024_paper.html), [arXiv 2312.00063v1](https://arxiv.org/html/2312.00063v1). Snapshot `sources/momask/`.

## Task, representation và generator

Sinh body motion theo caption HumanML3D/KIT-ML, dùng T2M-style features; không có detailed fingers/face trong setting này. Generator nhận target length.

Temporal convolutional encoder downsample 4. RVQ base quantizer xấp xỉ latent; mỗi tầng sau encode residual còn lại; decoder nhận tổng embeddings. Main dùng sáu layers, mỗi codebook 512 entries×512D. L1 reconstruction, commitment, straight-through, EMA/reset; quantization dropout .2 bỏ một số residual layers cuối để các tầng đầu giữ thông tin hữu ích.

Masked Transformer dự đoán base tokens với CLIP text và bidirectional context. Mask ratio theo cosine; selected tokens được thay 80% MASK, 10% random, 10% giữ nguyên, CE tại selected positions. Residual Transformer dự đoán song song tầng code tiếp theo từ text, layer index và các tầng trước. Đây là hierarchy theo residual depth, không temporal AR.

Inference all-MASK→predict→giữ confidence cao/remask confidence thấp theo schedule; 10 base iterations cộng năm residual iterations. Số rounds không tăng theo length nhưng attention cost vẫn phụ thuộc length. Token probability confidence chưa là semantic correctness probability.

## Training và điều kiện inference

Hai Transformers: sáu layers/sáu heads/hidden 384. Codec batch 512; generator batch 64 HumanML3D, 32 KIT-ML; LR 2e-4/warmup 2.000 iterations. CFG M/R 4/5 và 2/5 tương ứng; text dropout .1.

§5 nói cần target length, có thể thêm external text-to-length sampler. Không coi reported metrics tự là predicted-duration results. Inpainting còn nhận reference motion ngoài vùng mask, khác text-only generation.

## Evidence và phản bằng chứng

Table 1 HumanML3D: base-only FID .082/R@1 .504; full RVQ .045/.521. KIT-ML full FID .204 vẫn kém ReMoDiffuse .155 dù semantic metrics tốt hơn. “SOTA” phải gắn protocol/metric.

Table 2 bỏ residual quantization cho rFID .091/MPJPE 58.7 mm/gFID .093; full .019/29.5/.051. Residual codes có ích trong setting. Nhưng depth V5→V6→V7 có rFID .019→.014→.014 và gFID .051→.076→.084: decoder reconstruction tốt hơn mà generator kém đi. Counterexample này có trước LG-Tok.

Bỏ quantization dropout làm gFID .091 thay .051; bỏ replacing/remasking .063. Training details có hiệu quả đáng kể, cần giữ khi xây strong baseline.

FID full .045 ở Table 1 khác .051 ở Table 2; KIT abstract .228 khác Table 1 .204. Không lấy số khác bảng như cùng ablation. 20 evaluation repeats/CI chưa là 20 training seeds. User study 42 AMT users, 50 motions/method, ba raters/comparison đo realism/text match trên body motion, không sign comprehension.

Limitations nêu diversity thấp hơn và khó với rapid root motion như spinning. Không tự suy cùng failure mechanism cho handshape của SLG.

## Chuyển giao

RVQ, hierarchical generation và masking là các lựa chọn riêng. Coarse-to-fine reconstruction energy chưa là coarse-to-fine linguistic importance: một finger change nhỏ có thể đổi nghĩa. Cần đo articulation theo residual depth, matched bitrate/capacity, predicted/oracle length, unmask schedules và semantic reliability độc lập với token confidence. Không dùng FID body-motion để quyết định discrete/masked generation sẽ hơn continuous SLG.

# M01 — MoLingo: semantic alignment trong continuous latent generation

**D+C có phạm vi.** Đã đọc §§3–5, supplement §§6–7, Tables 1–3 và 5–10; static code audit conditioning/masking, semantic pooling và length protocol. Chưa chạy model hoặc xem hết video. Nguồn: [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/He_MoLingo_Motion-Language_Alignment_for_Text-to-Human_Motion_Generation_CVPR_2026_paper.html), [arXiv 2512.13840v3](https://arxiv.org/html/2512.13840v3). V3 ngày 13/07/2026 ghi sửa tên, content unchanged. Snapshot/hash tại `sources/molingo/`.

## Bài toán, representation và supervision

MoLingo sinh body motion theo caption HumanML3D. Hai quyết định được khảo sát riêng là latent nào dễ sinh và đưa text vào generator bằng cách nào. Semantic autoencoder (SAE) dùng thêm **BABEL frame-level labels** ở phần giao với HumanML3D; vanilla AE không nhận supervision đó. Không quy mọi gain thành architecture gain với cùng thông tin.

Causal temporal convolutional encoder/decoder nén N frames thành N/h continuous latents; cấu hình chính h=4, d=16. Paper xét các protocol 263D/20 FPS, MARDM 67D và MotionStreamer 272D/30 FPS. Không so FID tuyệt đối giữa các evaluators khác nhau.

SAE lấy local BABEL labels, encode bằng frozen text encoder, average/project thành teacher κ. Cosine alignment `mean(1−cos(m_i,κ_i))` chỉ giữ positive alignment; lọc teachers gần giống liên tiếp để giảm việc nhiều motion latents bị ép vào một nhãn lặp. Reconstruction gồm features, joints, velocity; thêm KL và semantic loss. Appendix dùng joint/velocity/KL/semantic weights 1/10/1e-5/.001, filter threshold .995. Teacher label không có ở inference.

## Generator, training và inference

Frozen T5-Large giữ nhiều text tokens, qua adapter sáu Transformer blocks. Motion Transformer dùng self-attention và motion-query/text-key-value cross-attention; output condition cho MLP rectified-flow head. Với path `(1−t)m+tε`, target velocity là `ε−m`; sampling đi từ noise về clean latent. Các positions được sinh theo thứ tự masking/unmasking, không nhất thiết trái sang phải.

Train codec trước rồi generator. Setup chính: batch 256, codec 5.000 epochs, generator khoảng 800 epochs, text dropout .1, CFG 5.5, EMA. Appendix: Transformer 16 layers/16 heads/hidden 1024; flow MLP tám residual blocks, width 1280; text length 128. Default 16 masked iterations ×32 flow steps. Report khoảng 10 giờ trên bốn H100, không phải runtime đo trong repo này.

Table 7 cho 2.83 s/motion ở cấu hình chất lượng cao; 5×32 chỉ .83 s nhưng FID .105 thay vì .066. Có noise refresh cho diversity. Không trích riêng latency dưới một giây như thể cùng quality. Evaluation và demo có duration policies khác nhau, xem audit.

## Bằng chứng và phản bằng chứng

| Câu hỏi | Author-reported evidence | Giới hạn |
|---|---|---|
| Semantic supervision | Table 2, cùng T5/cross-attention: AE FID .051/R@1 .533; SAE .066/.544 | Alignment tăng, FID giảm chất lượng; SAE thêm labels |
| VAE so SAE | Tables 1–2: VAE .049/.528; SAE .066/.544 | Không thắng mọi trục |
| Text encoder/conditioning | CLIP+AdaLN .114/.500; T5+AdaLN .077/.508; T5+cross-attention .051/.533 | Text backbone đã góp gain trước conditioning |
| Lọc nhãn lặp | Table 6: không lọc .096/.524, có lọc .066/.544 | Hiệu quả phụ thuộc xử lý supervision |
| Cosine so InfoNCE | Table 3: cosine+KL .066/.544; InfoNCE không KL .129/.523 | Khác cả KL, chưa cô lập một biến |
| Adapter depth | Table 5: không adapter .057/.524; depth 6 .049/.528; depth 9 .053/.527 | Capacity lớn hơn chưa tốt hơn |

20 lần evaluation/CI95% không phải 20 training seeds. Table 1 SAE R@1 .544, đoạn §4.3 có .542; giữ locator/version khi trích số. Fig. 5 và diễn giải về compression cũng không hỗ trợ một quy luật đơn điệu trên mọi metric.

Human study báo preferences 77.70–84.70% tùy đối thủ, với 15 users/comparison và 20 pairs/baseline trong main; appendix lại nói 15 samples/trial. Thiết kế chi tiết chưa được xác minh bằng questionnaire. Đây là body-motion preference, không là SLG comprehension.

## Static code audit

Official clone [hynann/MoLingo](https://github.com/hynann/MoLingo), commit `52e3b4c309256ce1bb29c8c038fe34b88682db08`; không dùng internal SLP code.

1. [molingo.py L134–175](https://github.com/hynann/MoLingo/blob/52e3b4c309256ce1bb29c8c038fe34b88682db08/mogen/models/molingo/molingo.py#L134): Transformer không nhận target causal mask. Training random masks; sampling L198–240 theo random ranks. Causal codec không đồng nghĩa causal target attention.
2. [eval.py L40–74](https://github.com/hynann/MoLingo/blob/52e3b4c309256ce1bb29c8c038fe34b88682db08/mogen/core/eval.py#L40) truyền batch reference length vào `sample_tokens`. [demo.py L188–213](https://github.com/hynann/MoLingo/blob/52e3b4c309256ce1bb29c8c038fe34b88682db08/mogen/demo.py#L188) có length estimator và đổi 20→30 FPS. Demo tồn tại không biến table results thành predicted-duration evaluation.
3. [sae_trainer.py L117–161](https://github.com/hynann/MoLingo/blob/52e3b4c309256ce1bb29c8c038fe34b88682db08/mogen/core/sae_trainer.py#L117) pool teacher trong `[stride*i−4*stride, stride*i+stride)`: tối đa 20 frames khi stride 4; mask padding/mẫu không BABEL.
4. Filter dùng dot product sau learned linear projection, không normalize trước dot; final alignment loss mới dùng cosine similarity. Flatten batch còn có thể tạo adjacency qua sample boundary. Không gọi filter implementation là cosine chỉ từ mô tả paper.
5. [Codec source](https://github.com/hynann/MoLingo/blob/52e3b4c309256ce1bb29c8c038fe34b88682db08/mogen/models/vae/encdec.py) có dilated residual convolutions. Tính receptive field tĩnh theo Table 10 cho đường dài nhất 176 frames, lớn hơn teacher window 20; chưa có gradient-based verification. Local teacher pooling là heuristic, không bao hết mọi input ảnh hưởng latent.

## Giới hạn và chuyển giao

§5 thừa nhận chưa sinh detailed hands. Không có bằng chứng NMM hoặc sign grammar. Coarse BABEL alignment có thể hữu ích nhưng không xác nhận giữ handshape/orientation contrasts. Retargeting qua controller cũng chưa chứng minh generator tự có physics guarantee.

Điều chuyển giao là thiết kế representation/conditioning theo thông tin cần giữ, không mặc định dùng flow. Đối chứng SLG cần same data/text encoder/latent capacity, shuffled labels, local so sentence supervision, teacher-independent semantics, articulation và predicted-length evaluation. Bỏ quantization vẫn còn compression bottleneck; chưa xác lập continuous hay discrete tốt hơn SLG.

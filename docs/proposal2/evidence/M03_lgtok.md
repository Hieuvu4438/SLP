# M03 — LG-Tok: language-guided discrete tokenizer

**D có phạm vi, chưa code.** Đã đọc §3, §4, reconstruction/generation ablations, App. B/D/E/G/I/K; chưa chi tiết KIT-ML App. C hoặc mọi video. Nguồn: [arXiv 2602.08337v2](https://arxiv.org/html/2602.08337v2), [official repo](https://github.com/eanson023/LG-Tok). Metadata/tác giả ghi ECCV 2026, chưa đối chiếu proceedings entry; giữ author-reported acceptance. Snapshot `sources/lgtok/`.

## Vấn đề và kiến trúc

Tokenizer cần vừa reconstruct motion vừa tạo targets mà conditional generator học được. Table 1 không text guidance: tăng 104→160→236 tokens làm rFID .143→.110→.090 nhưng gFID .230→.205→.257. Reconstruction tốt hơn chưa đủ chọn representation.

LG-Tok đưa text vào cả tokenizer và detokenizer. Motion frames được linear-project, nối latent queries và frozen LLaMA embeddings; Transformer self-attention rồi lấy outputs tại latent queries. Multi-scale residual quantizer downsample residual, quantize, upsample và trừ dần. Latent queries không bị buộc tương ứng một temporal window cục bộ.

Generator kế thừa MoSa next-scale autoregression: sinh cả scale tiếp theo cùng lúc, 10 scale steps, không phải một temporal AR step cho từng token. Finest query counts 25/36/49 tương ứng tổng multi-scale code counts 104/160/236. Decoder dùng frame queries cross-attend riêng code embeddings và text rồi regress motion.

Quan trọng: §4.1/App. D dùng 64–67D **body-only**, bỏ fingers/face ngay cả trên Motion-X. Paper lý giải captions ít nhắc tay/mặt, khoảng 2.74% có từ liên quan. Không dùng thí nghiệm này như whole-body articulation evidence cho SLG.

## Loss và protocol

Reconstruction Smooth L1; main chưa đủ chi tiết mọi codebook/commitment update. Text dropout .1 ở tokenizer/decoder nhằm hạn chế shortcut. Hai stages: train codec rồi next-scale generator. Generator dùng CLIP ViT-B/32, LLaMA3.2-1B ở codec path; gain có thêm text pathway/compute, chưa phải chỉ đổi generator backbone.

Inference text→codes→decoder có latent/text. Hai decoder passes tạo conditional/unconditional motions, extrapolate `m=(1+g)m_c−g m_u`, scale 2 HumanML3D, 1 Motion-X. Nếu chuyển sang rotations cần projection hợp lệ; feature extrapolation không tự giữ SO(3)/contact/anatomy.

Setup codec chín layers, bốn heads, hidden 256/FF 1024; batch 128, 200 epochs, LR 2e-4→2e-5, clip .01. Không velocity loss trong reduced body representation. App. K báo khoảng hai ngày trên RTX 4090 và gần 10× tokenizer memory so conv baseline. Table 10 latency .121 s so MoSa .045 và MoMask .062 trên cùng GPU/100 samples. Không thắng mọi efficiency trục. Output frame queries có length F, train giới hạn 196/200 frames; chưa audit predicted-duration protocol.

## Kết quả và phản bằng chứng

| Kiểm tra | Evidence | Điều chưa được suy ra |
|---|---|---|
| Text hay architecture? | Table 3 HumanML3D gFID .062→.057, Motion-X .257→.088 khi thêm text | Phần lớn HumanML3D gain đã có ở non-text architecture |
| Thay codec cho MoMask | HumanML3D reproduction .180→.111; reported MoMask từ nguồn khác .116 | Không dùng .116 thay .180 trong same-run effect |
| Text dropout | Table 4 g=0: HumanML3D .061 so no-drop .132; Motion-X .139 so .136 | Motion-X gain .088 còn cần guidance; drop riêng chưa cải thiện |
| Token budget | Table 6 Motion-X rFID .094→.051→.041, gFID .071→.076→.088 | Thêm tokens có thể làm generation khó hơn |
| Training windows | App. E/Table 9 thêm long-sequence windows: rFID .086→.041 | Data augmentation góp phần; cần split/source audit |
| Architecture sweeps | Table 5 chỉ tiny ba-layer/5K samples | Effect size không tự giữ ở full scale |

Table 5 còn có RoPE base 1000 rFID .042 tốt hơn base 100 .049 dù prose khen base 100; MPJPE có thứ hạng khác. Motion-X evaluator được retrain, vocabulary/GloVe và scoring batches khác; không so raw FID cross-paper. 20 repeats/CI là evaluation repetitions.

Perplexity, t-SNE hoặc attention maps không chứng minh word-level causal alignment hay giữ mọi semantic detail. No-drop reconstruction retrieval có thể cao mà rFID kém; đó là evidence cần kiểm tra shortcut, chưa là chẩn đoán duy nhất.

## Chuyển giao

Language-conditioned codec là prior art, với tiền thân TxtTok/TiTok trong image tokenization; nếu dùng làm đóng góp phải đọc nguồn gốc. SLG cần kiểm tra code giữ articulation nào, text path có làm bỏ qua motion codes không, và predicted motion giữ sign order ra sao. Các controls gồm caption ablation/swap trên cùng motion, same data/text encoder/capacity, independent handshape/orientation tests và valid-rotation decoding. Không sao chép token counts hoặc bỏ tay/mặt để đi theo benchmark body-only này.

# S02 — MaDiS: masked diffusion, tri-level supervision và temporal checkpoints

**D+C có phạm vi.** Đã đọc main method/setup, Tables 1–6, human evaluation và ablations. HTML acquired không có supplementary riêng. Code audit giới hạn ở generation, reconstruction losses và SiBLEU; chưa audit toàn bộ SiCLIP pipeline hoặc chạy model.

Nguồn: [arXiv 2601.19577v2](https://arxiv.org/html/2601.19577v2), [official repo](https://github.com/2000ZRL/MaDiS). Peer-review status chưa xác minh, giữ preprint. Code commit `071828ea1b94530dc7ff0a6c009c0a7c6d1f0784`; fresh snapshot tại `sources/madis/`.

## Bài toán và phần kế thừa

MaDiS sinh 133D sign motion từ text, giữ fitting và frozen tokenizer/decoders của SOKE, ba streams B/LH/RH. Không dùng external dictionary trong main method. Pretrain trên combined training data của ba datasets rồi fine-tune per dataset; phải tính multilingual pretraining khi so target-only models.

Mapping đúng là CSL-Daily/CSL, PHOENIX-2014T/DGS, How2Sign/ASL. §4 có đoạn liệt kê languages sai thứ tự; review không sao chép typo đó.

## Masked diffusion LM, UTC và MoP

Qwen3-0.6B-Base được chuyển attention từ causal sang bidirectional. Mỗi motion index gồm ba part tokens; mask một index thì mask cả ba. Forward mask probability là t; training masked CE có trọng số 1/t. Reverse process dự đoán nhiều positions mỗi step và giữ các positions đã unmask.

Default canvas 100 indices, downsample 4 tương ứng tối đa 400 frames, 25 generation steps. EOS quyết định phần output thực tế. Không có linguistic duration planner tường minh. Paper nói canvas cover khoảng 98% training videos; long clips được uniform sample về 400 frames. Đây là giới hạn phải báo khi đánh giá câu dài.

Uniform Temporal Checkpoints ưu tiên phủ timeline theo quarter/half grids rồi điền phần còn lại, vẫn dùng confidence trong mỗi stage. Đây là unmask-order prior, không phải gloss boundaries hoặc sign-order plan học từ text. Combinatorial order-count/pruning khoảng 10^41 trong paper không là measured compute speedup; stage sizes còn cần xử lý khi không chia hết cho k.

Mixture of Parts project codebook embeddings rồi học softmax gate để fuse vào LM input. Heads dự đoán từng part ở cùng index. Gate tương quan hand dominance không tự chứng minh causal coordination hoặc minimal-pair discrimination.

## Tri-level supervision

Pretraining có ba tầng:

1. Token objective mask cả text/sign; fine-tuning giữ text visible.
2. Latent heads khớp frozen codebook embeddings bằng Smooth L1.
3. Continuous predictions đi qua frozen VQ decoders để khớp motion reference; gradient qua decoder về generator dù decoder weights không đổi.

Không được gọi MaDiS là pure token CE hoặc không có motion-space grounding. Ngược lại, “physical space” trong đường code đã audit là decoded pose parameters, không tự là FK, contact, anatomy guarantee hoặc physics simulation. Paper fine-tuning giữ latent/pose losses với α=.5.

## Kết quả và ablations

Table 1, **author-reported, chưa tái lập**. SiBLEU column dùng nhãn của paper; implementation concern ở phần sau.

| Dataset | DTW-JPE body/hand | SiBLEU body/hand | SiCLIP R@1 |
|---|---|---|---|
| CSL-Daily | 5.99 / 7.96 | 4.87 / 4.08 | 38.61 |
| PHOENIX-2014T | 4.97 / 6.59 | 4.54 / 3.33 | 57.94 |
| How2Sign | 6.59 / 9.45 | 4.41 / 3.46 | 34.27 |

- Table 2: no-pretraining CSL JPE 6.93/9.27 so với full 5.99/7.96. Có thêm objectives/stages/supervision, không chỉ đổi attention mask.
- Table 3: no-checkpoint 6.61/9.05, contiguous checkpoint 6.28/8.44, uniform full 5.99/7.96. Hỗ trợ coverage prior trong study, chưa chứng minh grid points là linguistic landmarks.
- Table 4: codebook-conditioned average đã hơn random embedding average; dense MoP thêm gain. Baseline mới cần giữ initialization/capacity tương đương.
- SOKE-Qwen3 giúp giảm backbone confound nhưng main model vẫn đổi nhiều thành phần. Không gán toàn bộ gain cho masked diffusion hoặc một auxiliary loss.
- Paper báo 132 tokens/s, xấp xỉ .76 s cho 100 tokens; quality đổi theo 10/25/100 steps. Không suy real-time comprehension hay latency cố định cho mọi length/hardware.

Human study có 5 professional CSL signers, kinh nghiệm 4–13 năm, 15 samples/model, ratings 1–10: MaDiS 6.9, SOKE 4.6, reference motion 8.4. Đây là text–motion match rating, không phải blind translation accuracy; không so SOKE 4.6 với SOKE 6.45 của một study khác để kết luận model đổi chất lượng.

SiCLIP được train từ paired train motion/text. Retrieval tốt hơn chưa xác nhận nhạy với negation, handshape, order hoặc NMM. SiBLEU theo codec indices cũng không độc lập khi thay tokenizer; cần common pose-space evaluation hoặc fixed reference retokenization.

## Code audit và synthetic diagnostic

### SiBLEU release không phải BLEU-4 theo công thức đã kiểm tra

[generation.py, L16–35](https://github.com/2000ZRL/MaDiS/blob/071828ea1b94530dc7ff0a6c009c0a7c6d1f0784/mGPT/metrics/generation.py#L16) dùng `Counter` để tính clipped unigram overlap, nhân brevity penalty và 100. Không có bigram→4-gram geometric mean. Tên/docstring SiBLEU-4 không thay công thức.

Đã thực thi hai hàm được trích nguyên qua AST trên CPU, không load model/data. Reference `[1,2,3,4]`, prediction `[4,3,2,1]` cho score 100 dù zero matching 4-grams. [JSON, hash và kết quả](madis_metric_diagnostic.json). Đây là metric-only diagnostic, không là tái lập Table 1; không chứng minh paper đã dùng đúng released function này.

### Length không được kết luận từ signature

[models/madis.py, L172](https://github.com/2000ZRL/MaDiS/blob/071828ea1b94530dc7ff0a6c009c0a7c6d1f0784/mGPT/models/madis.py#L172) truyền reference token lengths vào adapter. Tuy nhiên [archs/madis.py, L255](https://github.com/2000ZRL/MaDiS/blob/071828ea1b94530dc7ff0a6c009c0a7c6d1f0784/mGPT/archs/madis.py#L255) dùng default caption-only template không có `Frame_Placeholder`; actual `generate_direct` chỉ nhận text/max length. Vì vậy chưa có căn cứ nói default length leakage chỉ vì adapter signature có lengths. Custom template có placeholder là điều kiện khác cần audit riêng.

### Sampling confidence và decoded-pose objective

[masked_diffusion.py, L336](https://github.com/2000ZRL/MaDiS/blob/071828ea1b94530dc7ff0a6c009c0a7c6d1f0784/mGPT/archs/masked_diffusion.py#L336): canvas 100, 25 steps, quarter/half/full grids. Index confidence lấy từ **body logits**, rồi điền cả ba heads; cuối cùng body EOS cắt cả ba streams. Không gọi confidence này là joint semantic reliability của mọi articulator.

`models/madis.py`, L140–169 gọi frozen VQ decoders trên continuous MLP outputs và khớp normalized motion targets; `losses/mgpt.py` dùng Smooth L1 reconstruction. Latent loss có valid-motion masks. Đường đã audit chưa có FK/contact/dynamics constraints, không được mô tả như physics engine hoặc loss chỉ trên masked 3D joints.

## Ý nghĩa cho proposal

MaDiS là baseline mạnh phải so, không là backbone mặc định. Tri-level supervision đã giải quyết một phần vấn đề token-only training; UTC tổ chức thứ tự sinh nhưng chưa là linguistic planning. Codec, fitting, semantic fidelity và duration vẫn cần diagnostics riêng, không tự chọn một bottleneck từ tên phương pháp.

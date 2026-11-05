# M08 — HumanTOMATO: body–hand hierarchy và semantic alignment

**D có đối chiếu version, chưa code.** Nguồn chính: [ICML 2024/PMLR 235](https://proceedings.mlr.press/v235/lu24b.html), [official PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/lu24b/lu24b.pdf). Đã đọc main §§3–4, App. B/C/K; không toàn bộ 39 pages/video. Đã đọc arXiv v1 trước đó, nhưng dùng published version cho final architecture/numbers.

Snapshot `sources/human_tomato/published_icml2024.pdf`, SHA256 `1d006cfad2fdc4b7c44ad0a6cbf288d49f06cc662ade061ec910caa2b94c4399`. PyMuPDF extraction có vài lỗi glyph/null. PDF preflight **UNAVAILABLE do thiếu pypdf**, không phải PASS; chỉ dùng section/table locators, không validated page anchors.

## Version và task

[ArXiv 2310.12978v1](https://arxiv.org/html/2310.12978v1) năm 2023 dùng facial cVAE và tên TMR. Bản ICML dùng **facial RVQ+GPT**, cVAE là baseline; aligned encoder gọi TMA. Baselines/table numbering cũng đổi. Không trộn hai versions hoặc coi chúng là hai studies độc lập.

Sinh whole-body activities/expressions trên Motion-X, phân tích thêm GRAB và HumanML3D body ablations. Motion-X report 95.642 motion/caption pairs; không đồng nhất với 37.751 body-only sequences của LG-Tok. Representation mở rộng H3D gồm root motion, local positions/velocities; face 50 FLAME expressions. Position reconstruction chưa tự giữ mọi palm/finger rotation cần cho SLG.

## H²VQ, generator và objective

Body/hand có encoders/codebooks riêng. Quantize hand trước, transform hand embeddings rồi nối body latents trước body quantizer; shared body–hand decoder nhận cả hai. Đây là capacity riêng **có coupling**, không hoàn toàn independent parts.

Squared reconstruction và commitment terms, EMA/reset; hai codebooks 512×512D. App. B.2 nói body downsample 2, hand 4, nhưng token/interleaving descriptions chưa đủ nhất quán để copy implementation; cần source nếu kế thừa exact rate.

Hierarchical GPT sinh body token rồi hand token tại timestep, điều kiện shared history/body hiện tại, causal attention, teacher-forced CE; text embedding là first token, có end token. Hand→body conditioning trong tokenizer và body→hand factorization trong generator là hai phân phối khác nhau, chưa phải logic contradiction.

Published face codec dùng two-level RVQ, downsample 4, facial GPT sinh các levels. Face sinh riêng, cùng text, rồi ghép output. Không suy independence của linguistic NMM từ factorization này.

TMA pretrain text/motion encoders bằng contrastive paired data. Generation dùng TMA text embedding và thêm decoded sequence alignment vào token CE. Đây là semantic prior/loss, không phải test-time exemplar retrieval. Gradient qua predicted discrete indices chưa đủ rõ trong phần paper đã đọc; không tự thêm Gumbel/soft-argmax và gán cho authors.

## Train/inference và compute

Tokenizer→GPT, TMA pretrain/freeze. Mô tả inference nhận text; actual EOS, face/body length synchronization và evaluation duration policy chưa code-audit. Không tự coi end-token formulation là proof không dùng oracle length trong mọi evaluation path.

AdamW 1e-4, batch 256, bốn A100 80GB; main ghi 6.000 tokenizer epochs/2.000 GPT epochs. App. B: GPT 18 layers, hidden 1024, 16 heads, max code length 149; report 8 giờ tokenizer/60 giờ GPT. Đây là paper-reported cost.

## Evidence và trade-offs

| Câu hỏi | Published evidence | Giới hạn |
|---|---|---|
| Body–hand hierarchy | Table 7 MPJPE: vanilla VQ 140.66, RVQ 110.94, H²VQ 92.97; hand 46.45→40.01→37.20 | Aggregation/alignment khác, chưa cùng metric SLG |
| Coupled so separate | Table 5 separate FID 2.209/R@1 .359; proposed 1.174/.416 | Cần matched capacity/data khi chuyển domain |
| Tách face | Table 6 body–hand error 108.31→92.97, face vertex error 1.368→.605 | Không chứng minh NMM độc lập với hands |
| Face generator | Table 4 RVQ-GPT FID 1.044/R@1 .200; cVAE 1.530/.084; diffusion 3.342/.064 | Tuning/adaptation khác; không có NMM comprehension |
| Có thắng mọi metric? | Table 1 FID 1.174 kém MotionDiffuse 1.129, R@1 .416 hơn .391 | Semantic/quality trade-off |
| TMA/semantic loss | Table 8 CLIP/no-align 1.086/.405; TMA/no-align 1.290/.416; full 1.174/.416 | Encoder tăng alignment, FID kém hơn; loss khôi phục một phần |

Published Table 1 có thêm MotionDiffuse, nên chỉ đọc arXiv v1 có thể kết luận sai FID leader. Evaluation repeats không là independent training runs. TMA dùng cả supervision/evaluation; cần evaluator độc lập trước claim semantic fidelity toàn diện.

## Giới hạn và chuyển giao

App. K nêu physical plausibility, fine temporal descriptions, separate face và đơn giản hóa emotion labels còn hạn chế. Đây là whole-body activity generation, không translation vào sign language.

Điều chuyển giao: phân biệt capacity riêng với độc lập hoàn toàn, và semantic prior với retrieval. Các ý tưởng này đã có prior art mạnh. SLG cần matched bitrate/capacity cho joint/independent/coupled codecs, orientation/handshape/NMM-scope tests, independent comprehension và kiểm tra actual predicted-length inference.

# M04 — Event-T2M: semantic units giữa word và sentence

**D có phạm vi, chưa code.** Đã đọc §3, §§4–6, App. A.1/5–7/9/11/14–18; Table 15 chưa đối chiếu mọi hàng. Chưa xem hết video. Nguồn: [arXiv 2602.04292v2](https://arxiv.org/html/2602.04292v2), [project](https://tjswodud.github.io/EventT2M/), [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3ea322e6eb11d3852bdb40263547ae78-Abstract-Conference.html). Đã xác minh venue dù OpenReview lúc đầu có browser challenge; không khẳng định hai full-text versions giống byte. Snapshot `sources/eventt2m/`.

## Task và kỹ thuật

Gemini 2.5 Flash tách caption thành các event clauses: action/state change có nghĩa tương đối độc lập, không đơn thuần tách từng động từ. TMR encode từng event; full caption tạo global embedding G. Đây là semantic conditioning, chưa phải explicit duration planner hoặc motion-boundary predictor.

Model sinh trực tiếp root/local-joint features. Denoiser bốn blocks, hidden 256, downsample 8. Mỗi block có local depthwise/pointwise convolution (LIMM), gated global text injection (ATII), Conformer self-attention/convolution/FFN và event cross-attention (ECA). Motion queries attend event keys/values; event gate khởi tạo gần 0. Conformer là kỹ thuật chuyển từ speech, không phải primitive mới hoàn toàn.

Eq. 19 dùng squared clean-motion prediction dưới Gaussian corruption. Training 1.000 diffusion steps, linear beta 1e-4→2e-2; appendix inference dùng UniPC 10 steps dù main gọi “10-step DDPM”. CFG dropout .2/scale 4; AdamW 1e-4, batch 128, 600 epochs HumanML3D/1.000 KIT-ML, hai RTX 4090. Chọn checkpoint theo validation FID.

Train cần caption–motion và text-derived events, không cần GT motion boundaries để tách clauses. Inference tính LLM segmentation, TMR embeddings và diffusion; length protocol chưa được code-audit. Event count không tự là predicted duration hoặc sign count.

## Evidence và phản bằng chứng

HumanML3D-E là test subsets có ≥2/≥3/≥4 events, 2.622/927/260 prompts trên 4.646 test samples. Các nhóm lồng nhau, không độc lập; sample size khác còn ảnh hưởng FID. Tác giả nói dùng standard train set, không train thêm trên các subsets.

| Câu hỏi | Author-reported evidence | Giới hạn |
|---|---|---|
| Multi-event generation | Table 3, ≥4: R@1 .466/FID .265 so MoMask .441/.418 | ≥2 FID .079 lại kém MoMask .065/MoGenTS .049; không thắng mọi hàng |
| Event units cùng encoder | Table 4 TMR, ≥4: token .440/.355→event .466/.265 | Hỗ trợ event granularity trong setting này |
| Có luôn lợi? | Table 4 CLIP, ≥4: token R@1 .399→event .374, FID .468→.425 | Alignment giảm dù FID tốt hơn |
| Backbone contribution | Table 7, ≥4: Transformer .373/.533→Conformer .466/.265 | Không quy mọi gain cho segmentation; chưa matched capacity đầy đủ |
| ATII | Table 16 toàn test: no-ATII FID .052 so full .056 | Có trade-off, trái lời diễn giải FID luôn xấu khi bỏ module |
| LLM annotations | App. A.11: 300 prompts/ba annotators, 93.3% accepted; human-defined subsets còn lợi thế | Chưa xác minh mọi event được grounded trong motion |
| So với human motion | App. A.7: 20 participants/20 samples, Wilcoxon+Holm p=.089/.291/.279 | Không có significant difference chưa là equivalence test |

App. A.14 verb/event cùng R@1 .466 nhưng FID .347→.265. TMR có motion pretraining, phải tính vào data/knowledge contract. 20 repeated evaluations không là 20 training seeds.

§5 có failure case đủ events nhưng hơi đảo thứ tự; long-horizon physics/HOI còn hạn chế. App. A.18 latency trên A5000: motion .167 s + LLM 1.430 s ≈1.597 s/caption. Không bỏ online LLM cost khỏi end-to-end comparison.

## Order constraint và chuyển giao

Từ Eqs. 12–17: cross-attention vào independent event embeddings, nếu không có event positional cues, bất biến khi cùng hoán vị hàng K/V. Main chưa nêu positional encoding trong ECA. Global caption G có thể giữ order và implementation có thể thêm chi tiết; không kết luận toàn model bất biến. Nhưng đưa ordered list vào attention tự nó chưa là temporal-order constraint. Cần giữ G cố định rồi reorder events để kiểm tra.

Điều chuyển giao là structured semantic units và complexity-stratified evaluation. Clause tiếng nói không tự thành sign unit; cần translation/reordering, còn NMM có thể chồng qua nhiều signs. Convolution/smoothness cũng chưa đảm bảo contact hoặc coarticulation đúng. Controls cho SLG: word/clauses/learned units cùng encoder/capacity, event permutation, oracle/predicted durations, independent minimal pairs và human comprehension. Chưa có bằng chứng bắt buộc chọn event conditioning hoặc một backbone SLG cụ thể.

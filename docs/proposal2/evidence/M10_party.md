# M10 — ParTY: part guidance trước holistic generation

**D có phạm vi, chưa code.** Đã đọc §3, §4, App. A/E.1–E.4 về capacity, codec và supervision; chưa toàn coherence appendix/user study/video. Nguồn: [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Heo_ParTY_Part-Guidance_for_Expressive_Text-to-Motion_Synthesis_CVPR_2026_paper.html), [arXiv 2603.09611](https://arxiv.org/abs/2603.09611). Version/hash trong registry.

## Task và cơ chế

Giải trade-off giữa part expressiveness và holistic coherence trên HumanML3D/KIT-ML. Parts là arms/legs của body motion, chưa detailed fingers/face. Không coi arms là handshape.

Temporal-aware VQ-VAE dùng frame features, local weighted pooling và GCN giữa temporal groups. Train L1 reconstruction và latent approximation. Có holistic và part tokenizers, không chỉ shared codebook.

Part-aware Text Grounding đưa CLIP sentence embedding qua bốn MLP views, losses giữ gần full-text semantics đồng thời phân biệt views; gates trộn views. LLM-generated part descriptions chỉ là auxiliary L1 targets ở **training**, không cần online LLM inference. Diversity của embeddings chưa chứng minh disentangled semantic factors.

Part-Guided Network sinh part tokens trước, tổng hợp guidance G rồi holistic Transformer sinh tokens tiếp theo. History fusion dùng self/cross-attention; holistic EOS kết thúc. Final output do holistic decoder sinh; đây là token-space guidance, không continuous pose refiner. Loss gồm holistic/part CE, view diversity và part-text auxiliary.

## Setup và điều chưa rõ

App. A: holistic codebook 256×128D/window 8, part codebook 128×128D/window 12, ba GCN layers. Holistic Transformer 10 layers, parts tám layers, hidden 128, ba part-guidance steps. AdamW/batch 64/A5000, auxiliary/diversity weights .1.

Cách khớp windows 12/8 và predicted/teacher-forced part guidance trong mọi training path chưa đủ rõ ở phần đã đọc. Cần code nếu kế thừa; không tự coi token indices khác streams tương ứng cùng frame. Predicted guidance ở inference không tự loại exposure bias trong training.

## Evidence và phản bằng chứng

Table 5 baseline FID .158/R@1 .494; thêm part guidance .086/.520; thêm text grounding .051/.545; thêm history fusion .035/.550. Pipeline có ích trong study nhưng cũng thêm generators, capacity và supervision, nên cần matched-cost/information controls.

Table S14 bỏ auxiliary part-text loss cho .042/.543 so full .035/.550. Gain nhỏ hơn tổng pipeline gain. S16 annotation runs gần nhau chưa là human validation part descriptions grounded trong motion.

Table 4 port codec vào MoMask/window 4: rFID .020→.003, gFID .045→.033, params 19.44→19.86M. Windows 8/12 còn đổi granularity và latency; không tự generalize mọi backbone. Table 1 FID .035/R@1 .550 thuộc 263D protocol, không so raw numbers với MARDM essential evaluator. 20 evaluation repeats không là 20 training seeds.

Table 3 coherence gains nhỏ và CIs chồng nhau. TC dùng velocity correlations/soft lag, SC dùng centroid distances/torso angles gần dataset means. Đây là heuristic coordination, không physics theorem hoặc semantic metric. Với SLG, một tay hold trong khi tay kia hoạt động có thể hợp lệ; tối đa synchrony hoặc gần mean pose có thể phạt valid signing.

## Chuyển giao

Structured intermediate predictions, train-only structured supervision và joint realization là các ý tưởng đáng kiểm tra nhưng đã có prior art. Đổi arms/legs thành hands/face không tự mới. Cần same-capacity holistic baseline, same auxiliary labels, predicted/oracle guidance, timeline perturbations và linguistic/articulatory evaluation. LLM descriptions của body actions không cung cấp chính xác fingers hoặc NMM grammar.

# M14 — STRIDE: kiểm tra semantic evaluator bằng structural contrasts

**D có phạm vi, chưa code.** Đã đọc §§2–4, App. A–C/E, construction, controls và hard-negative intervention; chưa mọi adapted VLM test ở App. D. Nguồn: [arXiv 2609.32462v1](https://arxiv.org/html/2609.32462v1), preprint/venue chưa xác minh. Snapshot `sources/stride/`.

## Task và construction

Coarse retrieval giữa captions khác xa chưa chứng minh phân biệt trái/phải, event order hoặc các actions gần nhau. STRIDE tạo triples motion/caption đúng/caption chỉ đổi một thuộc tính trên HumanML3D và SnapMoGen: temporal order, mirror direction, action identity; random-content mismatch là control dễ hơn.

Corpus 5.869 triples/2.598 motions, gồm 4.269 structural pairs và 1.600 random references. Qwen3-VL-8B viết temporal alternatives; human audit kiểm tra fluency, single-attribute edits và khả năng phân biệt từ motion. GPT-2 Large likelihood-gap filter ≤.15 nats/token và balanced action swaps giảm textual bias; text-only human preference vẫn 56.3%, nên chưa loại hết shortcut.

Với cùng caption pair, chấm trên matched và unrelated motion. `acc` là original-caption preference với đúng motion; `bias` là preference với unrelated motion. Grounding gain `Δ=(acc−bias)/(1−bias)`. GPR đo matched-motion margin lớn hơn unrelated-motion margin, ties nửa điểm. Đây là controlled grounding diagnostic, chưa là toàn bộ semantic understanding.

## Evidence

- Table 2 HumanML3D evaluator mirror acc .463, bias .484, Δ −.041; TMR mirror Δ .424. Random-content Δ tương ứng .879/.910. Coarse matching tốt chưa đủ structural discrimination.
- Table 3 mirror GT motion: R@1 .529→.523, matching distance còn tốt hơn 2.914→2.904; FID .000→.020. Shuffle caption–motion pairings giữ FID .000 nhưng R@1 .032: marginal distribution metric không kiểm tra conditional semantics.
- Time reversal đổi dynamics, không tương đương chỉ đảo event order trong ngôn ngữ. SnapMoGen temporal Δ .927 trên native long captions không cho phép xếp hạng với native HumanML3D như cùng task.
- Cross-domain evaluators còn chuyển 24 joints/148D/30 FPS sang 22 joints/263D/20 FPS; representation/domain shift là confounds.

## Hard-negative intervention

Thêm structural negatives vào contrastive training, giữ ordinary negatives; một số variants chỉ caption edits, NegHML có paired motion supervision. App. E: AdamW 1e-5, batch 32, hai epochs, seed 42, temperature .1. Một số action-specific checkpoints riêng, không tự là một evaluator cùng thắng mọi trục.

Table 6 NegTMR temporal Δ .416→.536, mirror .424→.435; NegHML mirror −.041→.127. Gains không đồng nhất và một số action scores giảm. Chưa có multi-seed uncertainty đầy đủ cho mọi intervention, không coi chênh nhỏ là universal improvement.

## Giới hạn và chuyển giao

Unrelated-motion controls cần nhiều samples/uncertainty; bias gần 1 làm normalized Δ bất ổn, phải báo raw acc/bias. GPT-2 likelihood matching chưa loại mọi encoder/VLM text prior. Human audit vẫn có residual preferences.

Trong SLG, mirror có thể giữ nghĩa, đổi handedness hoặc làm sai sign tùy trường hợp. Không gán mọi mirror là semantic negative. STRIDE cũng chưa chứng minh các SLG back-translators có cùng mức bias.

Điều chuyển giao là minimal-pair evaluation có motion controls và targeted supervision đáng tin. Teacher dùng để train generator cần evaluator độc lập và held-out tests; retrieval/BLEU/FID tăng chưa đủ chứng minh comprehension. Các body-motion numbers chỉ hỗ trợ thiết kế phép đo, không được bê thành kết luận định lượng cho ASL/CSL/DGS.

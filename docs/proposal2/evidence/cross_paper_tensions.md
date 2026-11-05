# Các kết quả cần đối chiếu hai chiều

Inventory chọn các cặp có câu hỏi chung hoặc findings khác hướng; không phải all-pair audit, meta-analysis hay independent review. `scholar_confirmation: pending` chỉ nói chưa có human confirmation, không yêu cầu dừng hoặc xin phê duyệt.

| ID | Nguồn/locator | Tension | Diễn giải có điều kiện | Control cần thêm |
|---|---|---|---|---|
| CP01 | MLD Table 8; MARDM §3.1/A1 | VAE hơn AE so deterministic AE được ưu tiên | Khác latent granularity, architecture, loss/scaling; chưa matched cross-paper ablation | Cùng encoder/decoder/budget, chỉ đổi KL/sampling |
| CP02 | MLD Table 5; MoLingo Table 2 | Concat hơn cross-attention so multi-token cross-attention tốt | Một condition token so nhiều T5 tokens/adapter | Same encoder/token count/adapter capacity |
| CP03 | MoMask Table 2; LG-Tok Table 6 | Reconstruction tăng nhưng generation giảm khi thêm codes | Evidence hội tụ về reconstruction/learnability trade-off | Same bitrate/capacity; reference/generated codes |
| CP04 | MotionStreamer Table 3; MLD Tables 4–5 | Near-identity hoặc nhiều latents tái tạo tốt nhưng khó sinh | Reconstruction chưa đủ chọn representation | Matched generator, detail/semantic preservation |
| CP05 | MoLingo Tables 1–3; HumanTOMATO Table 8 | Alignment tăng, FID có thể xấu hơn | Không report all-metric benefit | Same supervision, independent evaluator, Pareto analysis |
| CP06 | FlowMDM §3/Table 4; MotionStreamer §3/App. F | Bidirectional realization so causal streaming | Future context/application constraints khác | Same input availability, latency/editing/transition tests |
| CP07 | DisCoRD Tables 1–2/5; MoMask | FID/sJPE tốt nhưng MPJPE hoặc retrieval có thể kém | Các metrics đo mục tiêu khác nhau | Same prior; independent semantics/geometry |
| CP08 | HumanTOMATO §3.4/Table 6; EMAGE §4 | Face riêng nhưng coordination theo task khác | Activity/emotion so speech/visemes; chưa NMM independence | Same representations/data; joint/separate/coupled |
| CP09 | HumanTOMATO evaluation; STRIDE Tables 2–3 | Retrieval accuracy cao nhưng structural blind spots còn | Construct mismatch: coarse matching khác contrast discrimination | Minimal pairs, unrelated-motion controls, comprehension |
| CP10 | Gesture probes §4/limitations; HumanTOMATO Table 7 | Hand geometry tốt chưa đảm bảo semantic information | Khác targets; weak pooled probes chưa chứng minh loss toàn bộ | Raw/pre-quantized/decoded và temporal probes |
| CP11 | Glow-TTS §3; F5-TTS §3.1/B.2 | Explicit local alignment so implicit alignment | F5 vẫn ước lượng total duration | Predicted/oracle durations, same data scale |
| CP12 | ACT VI-B; MAR/MARDM regression discussion | CVAE giúp dù test z=0, deterministic generation có thể kém | Closed-loop conditioning khác free generation | Same observations/multimodality; train/test latent policies |
| CP13 | ParTY Table 3/TC; FrankenMotion timelines | Velocity correlation so asynchronous coordination | Synchrony không định nghĩa mọi valid coordination | Hold/asynchronous controls, human judgments |
| CP14 | Text2BFM Table 1/§4; FlowMDM | Executable prior tốt semantics nhưng giới hạn motion coverage | Policy reachability khác kinematic composition | Same data/state/skeleton, articulator coverage |
| CP15 | BLANC Table 4; SegMo Table 3/§4.3 | Motion-derived partitions giúp SLG so uniform tốt hơn CPD ở body task | Khác backbone/data/objective; chưa kết luận nhiều partition tốt | Point/confidence/jitter/full-distribution, matched K/compute |
| CP16 | SIGNER Table 3; SignRR Table 4 | GT duration cải thiện geometry nhưng có thể giảm BT | Metrics/timing không đồng biến; BT chưa là comprehension | Independent semantic tests và oracle/predicted tracks |

Đây là diễn giải từ thiết lập đã đọc, chưa phải các nguyên nhân đã được cô lập bằng thí nghiệm mới. Những khác biệt table/prose/code được giữ trong notes; không tự sửa paper numbers hoặc kết luận toàn paper sai.

Nguồn: [foundations](M11_generator_foundations.md), [MARDM](M05_mardm.md), [MoLingo](M01_molingo.md), [MoMask](M07_momask.md), [LG-Tok](M03_lgtok.md), [HumanTOMATO](M08_human_tomato.md), [MotionStreamer](M12_motionstreamer.md), [DisCoRD](M13_discord.md), [STRIDE](M14_stride.md), [gesture probes](M06_gesture_token_semantics.md), [EMAGE](M09_emage.md), [ParTY](M10_party.md), [FrankenMotion](M02_frankenmotion.md), [temporal planning](M16_temporal_planning.md), [cross-domain transfer](M17_cross_domain_transfer.md), [BLANC](S09_blanc.md), [SegMo](M18_segmo.md), [SIGNER](S04_signer.md), [SignRR](S06_signrr.md).

```yaml
inventory_scope: selected_shared_questions_not_all_pairwise
review_execution: same_model_inline
scholar_confirmation: pending
selection_gate: candidate_selected_after_SLG_comparison
```

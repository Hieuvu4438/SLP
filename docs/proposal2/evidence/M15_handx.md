# M15 — HandX: bimanual articulation, contact và data confounds

**D có phạm vi, chưa code.** Đã đọc §§3–6, App. A.2–5/C/E.1–3/G; chưa toàn fitting pipeline, inference-mask equations hoặc mọi user-study result. Nguồn: [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_HandX_Scaling_Bimanual_Motion_and_Interaction_Generation_CVPR_2026_paper.html), [arXiv 2603.28766v1](https://arxiv.org/html/2603.28766v1). Snapshot `sources/handx/`.

## Task và data

Sinh bimanual motion từ descriptions riêng cho tay trái, phải và quan hệ giữa hai tay. Descriptions trực tiếp handshape/contact giàu output information hơn spoken sentences trong SLG; đây không phải sign translation.

Corpus hợp nhất public datasets và capture mới: 21 joints/hand, 30 FPS, canonical coordinates; report 54.2 giờ/5.9M frames/485.7K texts sau filtering, không phải toàn bộ mới capture. Clips 60 frames/hai giây, non-overlapping. Filter đòi cả hai tay có activity vượt ngưỡng có thể loại one-handed signs, holds và nondominant-hand anchoring hợp lệ.

Captioning trích flexion, distances/contact và stable intervals thành structured motion events rồi LLM diễn đạt. Có observed kinematics làm căn cứ, khác suy motion tự do từ action captions. Tuy nhiên articulatory descriptions chưa là sign meanings; train-only descriptors từ motion không được trở thành target-derived test inputs.

## Hai generator và giả định

Diffusion x0 prediction dùng positions và rotation scalars. T5 encode ba prompt streams với role/CLS, separate cross-attention rồi residual fusion. Typed left/right/relation conditioning giúp tránh nhầm sides khi chỉ nối text.

AR dùng CNN codec/FSQ, downsample 2, local representation có relative wrists, wrist velocity, 6D wrist orientations, local positions/velocities và rotation scalars. Text prefix bidirectional, motion causal, CE; inference greedy theo §6.1.

Hai families khác representation, codec và decoding, nên không phải ablation chỉ thay AR↔diffusion. Rotation scalar là projected bending angle, chưa biểu diễn đầy đủ mọi 3D rotation; positions bổ sung geometry nhưng không tự chứng minh scalar đủ cho thumb/palm distinctions trong SLG.

## Evidence và evaluation

§6.1 so models trên **validation split**, không tự coi là held-out test SOTA. Sweep data 5/20/100% và depth 4/8/12/16; parameter counts không gồm frozen T5.

Table 4, diffusion 12 layers: 5% data R@1 .343/contact F1 .618; 100% .427/.641. FID 1.837→1.140→1.349 không đơn điệu theo data. Table 5 AR 92.27M với 2048 codes R@1 .182/FID 2.949; 4096 codes .134/3.050; tăng model 215.31M/4096 đạt .281/1.721. Alphabet lớn cần capacity phù hợp; không có universal scaling guarantee.

Diffusion 16 layers/260.97M lớn khoảng 6.7× bản 12 layers/38.95M nhưng kém một số metrics. Caption user study có 20 participants, ít nhất ba ratings/sample; chưa đọc mọi CI/stats, không gọi toàn corpus human-verified.

Contact F1 dùng 2 cm threshold và **cùng reference frames**, chủ yếu thumb-tip/fingertip và minimum inter-hand distance. Không bao hết orientation, surfaces/penetration hoặc variable-duration valid realizations. Contrastive evaluator train trên hand captions chưa được xác nhận nhạy với sign-language contrasts.

Completion có keyframes/wrist paths/một tay là track khác caption-only generation. Không trộn reference-constraint results vào text-only claims.

## Chuyển giao

Có thể khảo sát feature-grounded supervision, typed articulator conditions và contact/articulation diagnostics. Không copy two-second clips, both-hands-active filter hoặc canonical space mà bỏ signing location so với torso/head. Hand pretraining cần same architecture/optimization controls, separate source videos, tests cho holds/one-handed/contact/orientation và linguistic minimal pairs. Chưa có evidence HandX pretraining sẽ giải SLG semantics.

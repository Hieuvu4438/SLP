# M02 — FrankenMotion: điều khiển bộ phận bằng các timeline khác nhau

**D trên main paper.** Đã đọc §§3–6, setup, Tables 1–3 và limitations; chưa supplementary/code/video. Nguồn: [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FrankenMotion_Part-level_Human_Motion_Generation_and_Composition_CVPR_2026_paper.html), [arXiv 2601.10909v1](https://arxiv.org/html/2601.10909v1). Snapshot `sources/frankenmotion/`, readable blocks 19–86.

## Task, dữ liệu và giả định

Model nhận condition ở ba mức: toàn sequence, atomic actions có time windows, và từng body part có timeline riêng. Các parts có thể đổi hoạt động ở thời điểm khác nhau. Đây là input giàu hơn một caption toàn câu.

FrankenStein kết hợp BABEL/AMASS/HumanML3D/KIT-ML. FrankenAgent dùng DeepSeek-R1-0528 suy ba tầng `(label,start,end)` từ labels/text có sẵn; cho phép unknown và sparse annotations. Table 1 ghi 39.1 giờ, khoảng 138.5K labels, 46.1K part labels; §3.3 ghi khoảng 16K sequences, 265K atomic segments. Các đơn vị đếm không hoán đổi. Split 80/10/10, nhưng source-grouping để ngăn overlap chưa audit.

§5.1 kiểm tra 50 sequences qua ba experts: annotation accuracy 93.08%, Gwet AC1 .91. Đây là sample audit, chưa xác minh mọi label hoặc detailed hands. LLM suy chuyển động plausible từ action labels có thể thêm điều không xảy ra trong clip.

## Representation, kiến trúc và objective

§4.1 dùng STMC-style features: pelvis height, planar velocity, heading angular velocity, local 6D SMPL rotations và local joint positions. Main không đặc tả detailed handshape/face/NMM representation.

CLIP encode text; part/action embeddings được PCA xuống 50D. Action embeddings lặp trong frame range; part embeddings có ở từng frame/part. Nối chúng với noisy motion rồi MLP tạo frame features. Sequence-level CLIP và diffusion time là hai tokens bổ sung. Joint Transformer dự đoán clean motion; coordination được học trong model chung, không sinh parts riêng rồi chắp.

Loss Eq. 4 là squared x0 prediction dưới DDPM corruption, cosine schedule 100 steps. AdamW 2e-4, batch 32; report 47.5 giờ trên một H100. Unknown condition dùng zero features. Label dropout có xác suất lấy từ Beta(5r,5(1−r)); đây là condition dropout, không phải masked motion-token generation. Main chưa có ablation đầy đủ cho Beta policy.

## Train/inference và kết quả

Train cần motion cùng structured temporal annotations. Inference có sequence prompt, sparse part prompt hoặc action timeline. Agent có thể biến mô tả thành structure, nhưng Tables 2–3 đánh giá điều kiện căn thời gian từ dataset; chưa là end-to-end text-only với predicted timing.

Table 2: mean part R@1 47.21 so UniMotion 45.72; sequence R@1 65.27 so 62.66; sequence FID .06 so .08. Baselines được adapt/retrain cùng pose representation. Tuy nhiên UniMotion/DART nhận serialized part descriptions, FrankenMotion nhận explicit structure; input interface có thể góp gain. STMC còn khác training formulation vì composition ở test.

Table 3 thêm part→part+action→ba tầng cho part R@3 56.34→57.74→58.97 và sequence FID .08→.07→.05. Đây vừa là bổ sung context vừa là bổ sung information, chưa cô lập backbone. FID .05 ở Table 3 khác .06 ở Table 2; không gộp thành một số. 20 repeated evaluations/CI không phải 20 training runs.

§5.2 train bảy part evaluators, một action và một sequence evaluator theo TMR. Part evaluator vẫn nhìn full-body motion; nó có thể dựa tương quan từ parts khác. MPNet filtering false negatives cũng ảnh hưởng retrieval. Cần part-only/counterfactual tests để xác minh đúng part.

## Giới hạn và chuyển giao

Main chưa chứng minh predicted duration, minute-long single-pass generalization hoặc linguistic articulation. Coarse body actions có thể suy bằng common sense, còn spoken sentence không trực tiếp xác định sign handshape/spatial grammar.

Cơ chế đáng khảo sát là asynchronous conditions kết hợp joint realization. Với SLG phải dự đoán timelines từ text, giữ coordination tay–mặt và không dùng target timing ở main test. Đối chứng cần structured so serialized input, same encoder/capacity, oracle/predicted/no plan, synchronized/asynchronous boundaries và independent comprehension. Nếu chỉ có gain với target-derived windows hoặc evaluator full-body, chưa chứng minh lợi ích text-only SLG.

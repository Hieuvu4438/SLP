# M09 — EMAGE: holistic gesture và nguồn điều kiện theo thời gian

**D có phạm vi, chưa code.** Đã đọc §3 data, §4 method, §5 results/ablations, App. B statistics và C/D/F adaptation/training/compute. Chưa toàn bộ fitting losses/video. Nguồn: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EMAGE_Towards_Unified_Holistic_Co-Speech_Gesture_Generation_via_Expressive_Masked_CVPR_2024_paper.html), [arXiv 2401.00374](https://arxiv.org/abs/2401.00374). Snapshot `sources/emage/`.

## Task và data

Co-speech generation nhận audio, transcript features và masked gesture hints. Audio cung cấp duration/rhythm và timed content features; không cùng điều kiện text-only sign translation, nơi target order/duration chưa cho sẵn.

BEAT2 fit BEAT mocap/ARKit face sang SMPL-X/FLAME: MoSh++, sửa head/finger/outliers, handcrafted face mapping. Bỏ noisy-finger speakers còn 60 giờ/25 speakers, gồm standard 27 giờ và additional 33 giờ. Main comparisons dùng BEAT2-standard Speaker 2, split 85/7.5/7.5; không chứng minh unseen-signer generalization qua 25 speakers. Data filtering/fitting là đóng góp riêng, không quy gain toàn bộ cho architecture.

## Representation và model

55 joints Rot6D, 100 FLAME expressions, bốn contacts, ba translations. Bốn VQ-VAEs cho face 106D, upper body 78D, hands 180D, lower body 58D. Các priors có capacity riêng; generator dùng shared body hints để phối hợp.

Codec losses gồm reconstruction, velocity/acceleration và commitment, dùng geodesic/L1 phù hợp từng feature. Downsample 1, codebook 256×512D; token rate khác codec stride 4 của MoMask.

Content Rhythm Attention kết hợp audio onset/amplitude với transcript embeddings, tạo time-aligned features bằng TCN/projection và frame gates. Face/body có audio encoders riêng. Eq. 5 là adaptive weighted fusion, chưa tự là full word-to-motion attention matrix.

Hai paths: masked gesture reconstruction từ visible motion để tạo body hints/latents; audio-to-gesture cross-attend audio và hints để dự đoán latent/code classes. Face fusion khác; global translation predictor dùng lower body/contact để giảm foot sliding. Loss kết hợp latent regression và classification. Tên rec/cls trong §4.3 có dấu hiệu đảo so với mô tả; cần code nếu copy exact definition.

## Training/inference và evidence

Mask ratio tăng 0→95% trong 400 epochs, Adam 2.5e-4, clip .99. **Table 4 có four-frame seed pose**; completion có thêm motion hints. Không gọi main result là không có target-derived condition.

App. F: generator một speaker khoảng 8 giờ trên RTX 4090; còn pretraining VQ/global components khoảng 22.4 giờ trên năm RTX 4090. Không bỏ chi phí stages khỏi efficiency claim.

Table 6: một VQ FGD 9.787/face MSE 1.619; bốn VQ 7.397/1.243; thêm CRA FGD 6.833; thêm masked path 5.423. Tuy nhiên LVD 8.715→9.015 xấu hơn ở bước masked path, nên không “all objective metrics improve”. Capacity/bitrate chưa matched hoàn toàn.

Table 4 BEAT2 FGD 5.512 so TalkSHOW 6.209, BC 7.724 so 6.947. Không gộp với BEAT v1.3 Table 6 hoặc SLG metrics. App. C minh bạch adapt joints/lower-body heads, đổi một số baselines sang non-AR hoặc bỏ adversarial loss; đây không còn exact reproduction mọi original algorithm.

Table 7 thêm Trinity/AMASS cải thiện FGD nhưng BC 6.769 không hơn base 6.794. Data gain có trade-off. FGD/BC đo gesture distribution/speech synchrony, không sign semantics. Vertex MSE thấp cũng chưa đồng nghĩa expressive; human believability chưa là Deaf-signer comprehension.

## Chuyển giao

Có thể dành capacity theo articulator, học coordination bằng auxiliary reconstruction và tận dụng missing-part data. Nhưng dataset-specific heads/priors chưa chứng minh mọi motion dataset chuyển được linguistic knowledge. Cần matched text-only/seed/keyframe tracks, total-capacity controls, data/auxiliary-task controls và independent NMM semantics. Speech rhythm không thay grammatical scope của non-manuals.

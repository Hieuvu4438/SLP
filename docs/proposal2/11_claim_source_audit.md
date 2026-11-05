# Claim–source audit và kiểm tra đặc tả

Audit trên tài liệu/source snapshots hiện có ngày 05/11/2026. Đây là self-audit, không independent peer review. Mục tiêu là kiểm tra các kết luận quyết định phương án và benchmark contract, không dùng một link checker như proof của mọi claim.

## 1. Các claim quyết định và mức bằng chứng

| Claim trong synthesis/proposal | Evidence đã kiểm tra | Kết luận audit |
|---|---|---|
| Reconstruction không đủ để chọn codec | MLD Tables 4–5; MoMask Table 2; LG-Tok Table 6; MotionStreamer Table 3; DisCoRD Table 6 | Nhiều counterexamples trong điều kiện riêng. Không nâng thành một compression optimum phổ quát |
| Continuous/discrete hoặc AR/diffusion không là các lựa chọn loại trừ nhau | MAR mechanism; MARDM/MoLingo; MoMask/T2M-GPT/DisCoRD | Đúng về factorization/representation/head. Không có evidence một family thắng SLG vô điều kiện |
| Các semantic gains có data/encoder/teacher confounds | MoLingo Table 2; HumanTOMATO Table 8; ParTY Table 5/S14 | Đã giữ same-information/capacity controls; không quy toàn gain cho architecture |
| Motion-derived scheduling không gloss đã có | BLANC §3 và App. C.1–C.4 | Prior art trực tiếp, không nhận novelty từ duration/EOS/point partition/counterfactual event loss |
| Uncertainty treatment đáng kiểm tra nhưng có phản bằng chứng | BLANC Table 4; SegMo §4.3/Table 3 | Chỉ là research hypothesis. Uniform/point/jitter bắt buộc ở kế hoạch |
| Distributional segmentation/DP không mới | GP-HSMM §3.2; UAI2018 §4.1–4.2, scope M | Không mượn empirical effect cho SLG; q mới là heuristic target distribution |
| Proposed expected loss khác marginal likelihood | Main equations, M19; toy values .9163 vs .6931 | Đã sửa terminology; không claim exact latent-variable marginalization |
| Head/neck còn trong SOKE active body map | Official SOKE load/decode + public SMPL-X JOINT_NAMES | 10 active body rotations + jaw; không nhầm thiếu gaze với thiếu head motion |
| SiBLEU release là unigram overlap/BP | Pinned MaDiS generation.py và synthetic diagnostic | Verified code behavior; chưa chứng minh paper dùng exact released script |
| SOKE log PA chưa đủ xác định metric | Pinned t2m.py/dtw.py comments và calls | Default align_idx=0 là root/wrist translation; accumulated DTW không path-normalize |
| Released SignSparK no-KF còn length condition | Pinned sample.py/unet_model.py và inference config | Verified default path; không suy exact original Table 3 duration policy |
| SOTA/comprehension gain của proposed method | Chưa có training hoặc human study | **Chưa chứng minh; không là kết quả của review** |

Nguồn chi tiết: [motion notes](03_motion_literature_synthesis.md), [SOKE](evidence/S01_soke.md), [MaDiS](evidence/S02_madis.md), [SignSparK](evidence/S03_signspark.md), [BLANC](evidence/S09_blanc.md), [SegMo](evidence/M18_segmo.md), [M19](evidence/M19_segmental_uncertainty.md). Notes ghi primary URLs, versions, table/section/code locators và phần chưa đọc.

## 2. Đối chiếu trực tiếp bảng benchmark chính

Đã đọc lại các headers và method rows trong fresh primary full text, không chỉ copy lại notes:

- **SOKE:** readable blocks 56–67, Table 1. Dataset order How2Sign→CSL→Phoenix; raw BT 14.48/11.30/11.87 và geometry rows khớp contract. Full model multilingual/retrieval, không T0 target-only.
- **MaDiS:** blocks 59–75, Table 1. Order CSL→Phoenix→How2Sign, raw JPE 5.99/7.96, 4.97/6.59, 6.59/9.45. SiBLEU không bị chuyển nhãn thành BT BLEU-4; typo dataset-language order trong prose đã sửa theo dataset gốc.
- **SignSparK:** blocks 83–96, Table 3. GF/no-KF và SR rows tách; raw JPE được chép đúng. SOKE no-dictionary JPE có ô trống, không tự điền từ full model. Missing How2Sign SR không là kết quả bằng 0.
- **SIGNER:** blocks 99–110, Table 4. CSL/Phoenix BT 15.60/11.46 và DTW rows đúng. Trong Table 3, GT duration hand error .448 nhưng BT 14.44 so predicted 1.189/15.60; không đổi thành câu chuyện oracle thắng mọi metric.
- **SignRR:** blocks 65–82, Tables 2–3. Dùng TEST columns: Phoenix BT 10.96, CSL 1.57; CSL reference BT 5.06. Không lấy DEV numbers hoặc so raw BT với evaluator khác.
- **SignFLIP:** blocks 106–121, Table 3. How2Sign/CSL BT 7.10/9.75 thuộc 2D pose protocol, không 3D same-metric leaderboard.
- **BLANC:** blocks 61–72, Table 1. Phoenix/CSL BT 9.85/2.13; same-study SOKE no-dictionary 5.13/2.66. BLANC không thắng CSL BT ở comparison này; chưa có How2Sign row.

Các source paths dưới `third_party/proposal2_review_20261104/sources/<key>/readable.txt` là extraction có hash/versions trong registry. Block numbers dùng tìm nội dung trong snapshot, không là validated PDF page anchors. Source-table verification xác nhận **author-reported values**, không xác nhận reproducibility hoặc shared conditions giữa papers.

## 3. Recheck official code và provenance

Read-only recheck public clones SOKE, MaDiS, SignSparK cho working trees sạch; MaDiS HEAD khớp pinned commit. Các behaviors ở bảng trên được đọc lại từ code, không chỉ suy từ names. MoLingo audit có commit/file scope trong M01; không mở internal implementations hoặc experimental configs để làm các kết luận này.

Dataset pages chính thức được mở lại: [How2Sign](https://how2sign.github.io/), [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/), [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). Chúng xác nhận domain/modalities/annotations được dùng trong contract. Actual dataset manifests chưa tải/audit; proposal ghi đó là việc thực nghiệm tương lai, không giả đã hoàn tất.

## 4. Audit input/math của proposed pipeline

- Learning representation 60+90+90+6+10=256D; common evaluation adapter về retained SOKE parameters. Không thêm root/shape freedom riêng cho candidate.
- Gaussian motion codec/segment code có losses và dimensions riêng; frozen weights vẫn cho gradient về generator qua decoder. Extra compression được kiểm tra bằng oracle/reference diagnostics, không giả information loss bằng 0.
- q phủ toàn sequence với positive durations; forward log-normalizer/backward sampling cùng score/support. [Toy enumeration](evidence/temporal_plan_math_check.json) khớp normalizer tới sai số khoảng 8.9e-16 ở ví dụ L=4/Dmax=2; không chứng minh calibration hoặc generalization.
- Planner nhận text/language/predicted plan history tại inference. Teacher partitions, content targets và GT length là training-only. Không có retrieval/target keyframes trong main inference.
- Exposure training dùng continuous interval overlaps, không rounding gây lỗi khi predicted K lớn hơn reference L. Planner không nhận reference length; ở inference không có reference-based normalization. Subcell mixing là rủi ro được khai báo, chưa “giải quyết” exposure bias.
- Có content variance bounds, EOS policy, train-only cap policy và fully predicted evaluation. Cap-stop outputs vẫn bị tính, không chọn best-of-K gần reference.
- Expected loss qua q samples là structured plan augmentation. Novelty claim còn phụ thuộc việc vượt matched jitter/confidence/K/compute controls; các modules nền tảng không được nhận là mới.

## 5. Giới hạn của audit

Audit kiểm tra source alignment, tính nhất quán đặc tả và adequacy của proposal theo goal. Nó không chứng minh model thắng, true linguistic boundaries, universally new method, mọi reading source đều full-depth hoặc independent human comprehension. Những điều đó được giữ đúng là hypotheses/limitations hoặc scope chưa đọc; không được biến thành kết luận tích cực nhờ các checks khác pass.

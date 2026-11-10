# Phạm vi đọc và provenance hiện tại

Cập nhật 06/11/2026. Không dùng experiment có sẵn trong repo SLP làm bằng chứng. Tất cả nhận định chính được đối chiếu primary papers và code tác giả tải độc lập từ bên ngoài.

## Breadth và quy tắc chọn nguồn

Hai danh mục gốc đang là Markdown, không có CSV trong proposal3. Inventory giữ 190 linked table entries, tương ứng 139 URL sau loại tracking parameters nhưng giữ query có ý nghĩa. URL không tương đương paper identity: arXiv/project/proceedings có thể là một bài. Không cộng duplicate thành số paper đã đọc.

Toàn bộ 139 URL đã có screening record ở `input_screen_decisions.csv`: 132 HTTP acquisitions, 5 failures, 2 PDF-only URLs để theo dõi riêng. Trong acquisitions có bot/cookie pages nên không được coi là nội dung đã lấy thành công. Có 11 records unusable/blocked/pending theo kiểm tra hiện tại; các primary fallback tải riêng cho paper trọng yếu (như SignPR) vẫn được ghi ở bibliography. Secondary indexes chỉ dùng tìm paper, không làm căn cứ phương pháp.

Chọn đọc sâu theo mức ảnh hưởng tới quyết định: current 3D SLP baselines, methods có thể bác bỏ decoder/refinement proposal, anatomical/semantic alignment donors, và evaluation/representation studies. Foundations, surveys, multi-person/scene/video-rendering papers được screen để định vị scope, không gán full-text/code-read tự động. Đây là focused research có breadth từ hai danh mục, không phải systematic review tuyên bố đã đọc trọn mọi appendix và mọi source file.

## Primary papers tải mới và phạm vi substantive reading

Các artifact dưới `third_party/proposal3_sources/<key>/`; URL/hash trong từng `acquisition.json`. Tải về không tự động là đã đọc. Các mục sau đã được mở nội dung method hoặc evidence liên quan trong lượt nghiên cứu; không claim trọn appendix khi chưa kiểm tra.

| Key | Paper | Phạm vi dùng cho kết luận |
|---|---|---|
| soke | Signs as Tokens | Method, Tables 1–2; số bảng đối chiếu trực tiếp lại từ HTML mới |
| signspark | SignSparK v4 | Keyframe/CFM method, GF/SR/KF2P và Tables 3–4; release README kiểm tra riêng |
| madis | MaDiS v2 | Tri-level pretraining, UTC/MoP, evaluation và Table 1 |
| signer | SIGNER | PVQ, duration, LTF/IPA và duration/evaluator distinctions; supplement excerpts ở vòng đầu |
| blanc | BLANC | LAC teacher/duration, multi-rate/refinement prior art, evaluation differences |
| discord | DisCoRD | Method §3.2, reconstruction/generation divergence và Table 6 counter-evidence |
| t2sgpt | T2S-GPT | Dynamic encoder/length regulator, budget/semantic losses, duration transformer, eval và ablations |
| hybridsign | Hybrid AR–Diffusion | §4 method, self-forcing/confidence, §5.4 2D và appendix 60-frame protocol/limitations |
| m3t | M3T v2 | §3 representation/FSQ/SLT, §4.4 ablations; production vs recognition, Tables 1–2 và fitting limitations |
| salad | SALAD | §3 skeleton VAE/denoiser/editing, §4.4 ablations |
| ram | RAM | §3 architecture/objectives/REG và §4.4 guidance ablations |
| mate | MATE | §3.2–3.4 localization/prototypes/alignment; không đưa số SOTA vào SLP |
| signpr | SignPR | §3 2D semantic/regional codes, length predictor và causal refinement; không gộp score vào 3D |
| g2pddm | G2P-DDM | Part patches, discrete mask/replace/PAD, CodeUnet, length estimation và G2P condition |
| darslp | Disentangle and Regularize | §3 articulator latent, length/KL regularization và stated limitations |
| vae_design | Impact of VAE Design | Variants, reconstruction/latent diagnostics, evaluation truncation, one-seed/four-observation limits |
| signrr | SignRR v3 | §3 dictionary construction, RVQ/joint decoder, predicted length và DP assembly |
| serv | SeRV v2 | §3 global coarse-level and token late-interaction supervision, hierarchical generation |
| irg | IRG-MotionLLM | §3 generation/assessment/refinement tasks, ignore-incorrect strategy và GRPO/reasoning loop |

Các classical donors MDM/MLD/T2M-GPT/MoMask/ReMoDiffuse/FlowMDM/MotionLCM và HandMDM/Sign-D2C đã đọc primary excerpts ở vòng đầu, nhưng không lấy metadata của report cũ làm bằng chứng đã full-read. HumanTOMATO được đối chiếu official PMLR abstract và release code scope. Những nguồn này đóng vai trò background/donor; các claim quyết định v2 dựa primarily trên 19 paper mới ở trên và source implementations.

## Source code tác giả đã đọc

`external_code_manifest.json` pin 8 official repositories và 22 tệp đã đọc tĩnh theo function/section: SOKE, MaDiS, SignSparK, MoMask, DisCoRD, SALAD, RAM, HumanTOMATO. Không claim toàn repo được validation. 13 core files của bốn baseline đầu được so lại với fresh checkouts: hash khớp, không liên quan các experiment/pilot của SLP.

M3T project code link ghi coming soon; T2S-GPT/SIGNER/HybridSign chưa tìm thấy author implementation link ở các trang đã kiểm tra. HumanTOMATO có representation/OpenTMA nhưng chưa đủ full-generator training implementation trong checkout screen. Không giả lập source reading để lấp khoảng trống. IRG/SignRR/SeRV hiện là paper-level prior-art checks, chưa code audit.

## Kết luận reading audit

Đã đủ cơ sở cho proposal nghiên cứu có nguồn ngoài, benchmark decision và counter-evidence review. V1 đã hạ ưu tiên sau DisCoRD; v2 phân biệt expected improvement với confidence và không claim first. [Critical review](../07_critical_review.md) ghi residual risks.

Không có RACE experiments, matched leaderboard, human study hoặc empirical SOTA claim. Những việc đó là future validation đã thiết kế, không là kết quả tái sử dụng từ SLP. Full-system evaluator reproduction còn cần thực hiện trước bất kỳ claim benchmark nào. Nghiên cứu hiện tại không chứng minh novelty tuyệt đối hoặc guaranteed SOTA, và không tự coi mọi paper/code trong hai danh mục là đã đọc trọn.

# Đối chiếu độc lập paper và code tác giả

Cập nhật 06/11/2026. **Không sử dụng experiment, pilot, training log, checkpoint tự huấn luyện hoặc kết luận của repo SLP làm bằng chứng nghiên cứu.** Các con số dưới đây được đọc từ paper gốc bên ngoài. Artifact local chỉ là bản lưu primary source phục vụ truy vết, không phải kết quả experiment trong SLP.

## 1. Nguồn độc lập và phạm vi

Đã tải trực tiếp paper từ arXiv/ACL/CVF/ICLR vào `third_party/proposal3_sources/`, với URL, SHA256 và parser trong từng `acquisition.json`. Đã clone độc lập official SOKE, MaDiS, SignSparK, MoMask, DisCoRD, SALAD, HumanTOMATO và RAM vào `third_party/proposal3_sources/code/`. Những repository này được đọc tĩnh, không cài environment, chạy training hay dùng checkpoint của SLP.

13 tệp core của bốn repo SOKE/MaDiS/SignSparK/MoMask trong audit đầu đã được so hash với checkout official tải mới: khớp 13/13 (`research/independent_source_comparison.json`). Vì vậy nhận xét về các source functions vẫn có cơ sở độc lập. So hash không xác nhận metric paper tái lập, tính đúng toàn repo hoặc chất lượng method.

Danh mục đầu vào có 190 linked table entries, 139 URL khác nhau; đã thu thập và screen title/abstract/preview theo từng nguồn, không đồng nhất số URL với số paper. `research/screening_acquisition.json` lưu cả failure/bot pages; HTTP 200 không có nghĩa lấy được paper. Chi tiết methods dưới đây được đọc sâu có chọn lọc theo tính liên quan, không khẳng định đã full-text/code-review mọi mục trong danh mục.

## 2. DisCoRD — counter-evidence quan trọng nhất cho post-refiner

Nguồn: [paper §3.2, §4.2, Table 6](https://arxiv.org/html/2411.19527), [official code](https://github.com/whwjdqls/DisCoRD).

DisCoRD giữ pretrained token generator và thay feed-forward decoder bằng flow từ Gaussian noise đến raw motion. Condition projection tạo frame-specific features từ code embeddings. Nó không chỉ denoise đầu ra decoder cũ. Trong Table 6 HumanML3D, RF post-refinement có generation FID 0.035; direct DisCoRD đạt 0.032. Kết quả này ủng hộ direct decoder trong setting đó, không chứng minh superiority ở signing. Full-sequence training có reconstruction FID tốt hơn window training nhưng generation FID kém hơn (0.038 vs 0.032), nhắc rằng reconstruction và generation phải đo riêng.

Code đọc: `MotionPriors/models/rf_decoder/rectified_flow.py`, `Unet1Dflow_decoder.py`. Forward dùng `noised=t*data+(1-t)*noise`, target `data-noise`, endpoint estimate `noised+(1-t)*flow`. Decoder concatenate condition với state; interface hiện assert `text_embedding is None`, temporal factor ×4 và sequence divisible by four. Vì vậy thêm token-level text hoặc SMPL-X signing là thay đổi thực sự cần implementation và ablation, không phải config toggle. Không copy 263D output dimensions sang SOKE.

**Counterargument:** mở decoder từ token rời rạc sang stochastic continuous motion có thể hallucinate một trong nhiều motion cùng token; chính paper quan sát faithfulness giảm với vanilla VQ của T2M-GPT. Đây là động cơ cho semantic preservation/selection, không phải bằng chứng RACE đã giải quyết được nó.

## 3. HybridSign — confidence và self-generated history đã có

Nguồn: [ACL paper §4 và §5.4, appendix](https://aclanthology.org/2026.acl-long.31/), PDF tải mới `hybridsign/paper.pdf`.

Model dùng ba articulator experts, causal self-attention và flow-based generation. Training history là predicted articulators, không teacher-forced GT. Confidence bias cộng vào attention logits để tăng ảnh hưởng frame có keypoint confidence cao. Body/hands/face là các nhóm; hai tay chung một expert. Paper giải thích four-expert variant kém hơn three-expert có thể do mất bilateral coupling. Đây là ablation trong setting của họ, không phải quy luật mọi dataset.

Hai giới hạn quan trọng: representation là **2D keypoints có confidence**, và latency/throughput đo trong protocol **60 frames**. “Real-time” trong bài chỉ relative low latency; không tự đồng nghĩa 25/30 FPS deployment. Không nhập DTW/FID/BT của bài này vào SOKE 3D leaderboard. Confidence từ pose representation cũng không tương đương calibrated probability rằng sửa draft sẽ có lợi. Paper không cung cấp đủ bằng chứng để kết luận c(t) là estimate expected improvement của hai decoders; đó là khác biệt cần đo của v2, không claim novelty cho confidence nói chung.

Chưa tìm thấy official source link trên ACL page/PDF trong lượt này; không dựng pseudo-implementation rồi gọi là author code. Hybridsign-style control nếu tự xây phải ghi rõ adapted.

## 4. RAM — reconstruction guidance là prior art, code đã public

Nguồn: [paper §3.1–3.3, §4.4](https://arxiv.org/html/2601.14788), [official RAM](https://github.com/feifeifeiliu/RAM).

RAM co-train motion reconstruction và text-conditioned generation với decoder chung, dùng self-regularization và asymmetric latent alignment. Inference REG so text prediction với reconstruction conditioned on previous predicted motion latent, rồi khuếch đại residual; kết hợp CFG. Do đó “reconstruction sửa lỗi generation” không là novelty của RACE.

Code tải mới đã có thật, dù một phần author project trước đó còn ghi coming soon. Đọc `diffusion/diffuser_api.py:generate_batch` và `utils/sampler_util.py:forward`: loop cập nhật `recon_latent` từ predicted motion, branch `out_motion` cùng diffusion state, rồi cộng `reg_scale*(out-out_motion)`. CFG và REG có thêm forward passes; số diffusion steps không bằng tổng model evaluations. Không lấy wrapper CFG cũ làm bằng chứng toàn REG.

**Hệ quả:** muốn dùng RAM như baseline đúng nghĩa phải co-train reconstruction branch; chỉ gọi residual giữa SOKE draft và refiner không phải reproduction RAM. Có thể thử inspiration riêng, ghi adapted và báo tổng cost.

## 5. SALAD — skeleton-aware không có nghĩa đã có hand topology

Nguồn: [paper §3.1–3.3, §4.4](https://arxiv.org/html/2503.13836), [official SALAD](https://github.com/seokhyeonhong/salad).

SALAD dùng skeleto-temporal VAE, temporal/skeleton/cross-text attention và v-prediction diffusion. Ablations cho thấy structured latent và token cross-attention có ích trong HumanML3D/KIT; không chuyển nguyên lợi ích đó thành lời hứa cho signing.

Code đọc: `models/vae/encdec.py`, `models/skeleton/pool.py`, `models/denoiser/transformer.py`. Input parser tách HumanML3D root/relative position/6D/velocity/contact; pooling maps hard-code skeleton t2m/kit xuống 12 rồi 7 joints, có legs. Vì thế reuse sang signing phải viết schema/parser và pooling giữ fingers, không thay `joints_num` là xong. Attention mirroring là editing mechanism trong generic T2M; dùng mirror trái/phải trong ký hiệu có thể đổi nội dung.

## 6. T2S-GPT — duration và semantic preservation không phải mới

Nguồn: [ACL 2024 paper](https://aclanthology.org/2024.acl-long.183/), [full primary text](https://arxiv.org/html/2406.07119), §4.2–4.3, §6.

DVQ-VAE dùng information weights và adaptive segmentation/downsampling; decoder length regulator lặp latent theo duration. Ngoài reconstruction/VQ, có budget loss và auxiliary sign-to-text loss. Stage 2 dự đoán code lẫn duration, EOS kết thúc. Điều này bác bỏ claim rằng dynamic-rate motion tokens, text-dependent duration hay semantic auxiliary task là đóng góp mới hiện nay.

BLEU-4 11.87 của T2S-GPT có GT ceiling 14.34 trong evaluator riêng; giá trị bằng SOKE PHOENIX BLEU 11.87 là trùng con số, không chứng minh hai method ngang nhau. T2S paper tự train BT vì weights prior work không public. Chưa tìm thấy author code link trên project page trong lượt khảo sát này; ưu tiên dùng paper làm căn cứ cơ chế, không nói đã audit implementation của DVQ.

## 7. M3T — phân biệt face fittings với model parameterization

Nguồn: [M3T v2 §3, §4.2–4.4 và appendix](https://arxiv.org/html/2603.23617v2), [author project](https://cogvis-cvssp.github.io/papers/m3t/).

Bản paper mới nói rõ hạn chế thuộc **fittings** chỉ regress ít expression coefficients, không phải SMPL-X về nguyên tắc không hỗ trợ rich FLAME expression space. Không lặp lại project marketing rằng model intrinsically chỉ có 10 chiều. M3T dùng SMPL-FX, richer face reconstruction, per-modality FSQ, shared mirrored hand encoder, auxiliary translation. FSQ không tự bảo đảm uniform empirical occupancy hoặc semantic coverage; chính batch/width ablations còn thay utilization và reconstruction.

Recognition trên NMFs-CSL là representation probe, không trực tiếp chứng minh generated facial grammar đúng. Paper ghi direct generated-nonmanual evaluation bởi native signers vẫn là future work. Điều này ngăn việc dùng recognition accuracy thay cho production intelligibility.

M3T đổi fitting pipeline và face space, nên face comparisons với legacy SOKE không head-to-head. Trong v2, RACE giữ original face và chỉ claim manual-motion track; full-face study phải là bảng riêng trên dữ liệu có face annotations thích hợp. Project có code link kèm trạng thái “Coming soon”; không claim đã audit M3T code.

## 8. SignPR, FGDM, G2P-DDM — cơ chế gần nhưng input/output khác

- [SignPR, CVPR 2026 primary PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SignPR_A_Progressive_Vector-Quantized_Diffusion_Framework_for_Sign_Language_Production_CVPR_2026_paper.pdf): §3 chia **2D poses** thành whole-body semantic và regional codes; structural consistency prediction, semantic-then-regional diffusion và block-causal inference refinement. Như vậy “global semantics→local details” và phối hợp parts đã là prior art. Nó có length predictor; không giải quyết native-length bằng GT. Do 2D representation/eval khác, không so trực tiếp số với SOKE 3D.
- [FGDM, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Focal-General_Diffusion_Model_with_Semantic_Consistent_Guidance_for_Sign_Language_CVPR_2026_paper.html): focal stage ASGCN/TCN mô hình local joints, general stage xử lý sequence; semantic decoder dùng CTC với gloss và supervision điều chỉnh theo noise stage. Đây là G2P, không GF-T2P. Không thêm gloss supervision rồi giữ nhãn gloss-free.
- [G2P-DDM](https://arxiv.org/html/2208.09141): Pose-VQVAE tách ba patches, discrete mask-and-replace diffusion với PAD/MASK absorbing states, CodeUnet multi-scale và sequential clustering/length estimator. Bidirectional discrete generation, anatomical parts và length learning có lịch sử trước SOKE/MaDiS; khác condition là gloss.

Những paper này được dùng để phản biện novelty và thiết kế alternative controls, không dùng scores khác task để chọn benchmark “dễ hơn”.

## 9. MATE, DARSLP và VAE design — tránh giả định alignment hoặc reconstruction là đủ

[MATE §3.2–3.4](https://proceedings.iclr.cc/paper_files/paper/2026/file/2fbfefa47aadd5d583439590693b2dcc-Paper-Conference.pdf) phân rã caption thành sub-actions, tìm partition bằng text-motion retrieval rồi dùng làm soft Gaussian prior; prototypes/cross-disentanglement căn motion semantics vào word embeddings. Adaptation sang signing cần sign-order latent hoặc predicted gloss, không mặc định spoken words tạo chronological motion segments. Không dùng ChatGPT để tự sinh phonological ground truth từ câu thường rồi gọi đó là nhãn chuẩn.

[DARSLP §3](https://arxiv.org/html/2504.06610) đã dùng articulator-specific latent dimensions, text-conditioned non-AR generation, length predictor và channel-wise KL với training priors. Learned anatomical factorization không phải chứng minh statistical disentanglement. Việc tăng hand loss có thể thay quality ở part khác; RACE phải so parameter/data-matched controls, không ghép nhiều objective không kiểm tra.

[VAE design study, GenSign 2026 §3–5](https://arxiv.org/html/2606.22959) so bốn VAE variants trên PHOENIX, cùng latent size 64 và parameter budget gần nhau. Nó phân tích temporal variation, covariance/effective dimension và downstream diffusion. Bài ghi rõ chỉ một seed, bốn observations không đủ suy ra significant correlation; không biến emerging trend thành quy luật latent geometry. Evaluation có truncate về reference length và DTW; số BT/reconstruction không được so native-length SOKE.

**Hệ quả chung:** chẩn đoán tokenizer reconstruction là cần nhưng không đủ để chọn method; phải đo downstream generated-token behavior và semantic fidelity. Không xem latent smooth hơn hoặc occupancy cao hơn là mục tiêu tự thân.

## 10. HumanTOMATO và giới hạn phát hành code

Nguồn: [ICML primary paper](https://proceedings.mlr.press/v235/lu24b.html), [official repository](https://github.com/IDEA-Research/HumanTOMATO).

Whole-body hierarchical body/hand modeling và text-motion alignment là donor liên quan. Tuy nhiên repo official tải mới chủ yếu phát hành representation processing và OpenTMA; Quick Start/Model Training cho full generator còn trống. Không tìm thấy đủ implementation hierarchical GPT/H²VQ trong checkout đã screen để nói có thể tái dùng ngay. README/architecture figure không là code verification. Có thể dùng OpenTMA alignment source, nhưng phải ghi giới hạn release.

## 11. Những kết luận bị rút hoặc giảm độ mạnh

| Claim ban đầu dễ bị hiểu quá mạnh | Kết luận sau đối chiếu nguồn ngoài |
|---|---|
| Thêm flow refiner là hướng ưu tiên | Direct token-conditioned decoder có evidence tốt hơn từ DisCoRD; phải so lại trong signing |
| Confidence/risk là novelty | HybridSign và nhiều T2M đã dùng confidence; RACE chỉ giả thuyết về expected paired improvement và coupled selection |
| Part splitting càng chi tiết càng tốt | HybridSign three-vs-four experts và VAE Factor variant cho counterexamples có điều kiện |
| SMPL-X không thể có rich face | Vấn đề quan trọng là fitting/estimation và dữ liệu; bản M3T v2 phân biệt rõ |
| Reconstruction tốt đủ chọn tokenizer | Cần downstream generation và semantic checks; VAE study/DisCoRD cho divergence |
| NFE nhỏ nghĩa là pipeline nhanh hơn | Phải cộng base generation, CFG/REG forwards, two candidate decoders, selector, retrieval và conversion |
| Có code link nghĩa là implementation đầy đủ | HumanTOMATO/M3T/T2S/SIGNER cần kiểm tra release thực; không điền thiếu bằng suy đoán |

Review này do cùng một AI thực hiện, không có independent human hoặc cross-model validation. Nó hỗ trợ proposal và counterarguments; không tạo bằng chứng thực nghiệm mới rằng method đạt SOTA.

## 12. Nearest-work search bổ sung: SignRR, SeRV và IRG-MotionLLM

[SignRR v3 §3](https://arxiv.org/html/2608.28568v3) retrieve real training-dictionary segments từ **gloss**, ghép bằng dynamic programming, rồi refine toàn sequence với part-aware RVQ và joint decoder; duration điều chỉnh trong latent. Nó bác bỏ novelty của “retrieval + whole-sequence refinement”, nhưng khác GF text-only condition và decoder selection của v2. Gloss-constrained training segmentation và predicted inference length phải được khai báo; không lấy score G2P làm bằng chứng vượt GF-T2P.

[SeRV v2 §3](https://arxiv.org/abs/2609.05742) dùng global text alignment trên coarse RVQ level và token-level late interaction trên tổng quantized latent, sau đó hierarchical GPT. Do đó “semantically supervise tokenizer + RVQ” đã có gần signing. RACE giữ tokenizer đầu tiên để nghiên cứu decoder; nếu đổi tokenizer theo hướng SeRV, đó là một experiment khác và cần attribution phù hợp.

[IRG-MotionLLM §3.2–3.3](https://arxiv.org/html/2512.10730) train generation, assessment và refinement, rồi CoT trajectories/GRPO để tự quyết định refine tiếp hay dừng. Không claim mới việc đánh giá motion trước khi sửa hoặc quyết định giữ một prediction. Distinction RACE hiện chỉ là calibrated paired-loss improvement giữa hai frozen decoders và temporal decisions phải cùng body–hai tay. Sự khác biệt này có thể chỉ là incremental contribution; phải qua controls C1–C8 mới có căn cứ publishable novelty.

Search bổ sung ngày 06/11/2026 dùng các truy vấn `motion generation selective refinement confidence uncertainty decoding`, `sign language refinement uncertainty`, `motion decoder confidence discrete continuous`, `motion expected improvement generation` trên arXiv/CVF; kiểm tra lại title primary của các hit gần. [UDE](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_UDE_A_Unified_Driving_Engine_for_Human_Motion_Generation_CVPR_2023_paper.html) cũng đã có diffusion motion decoder từ discrete tokens trước DisCoRD. Vì vậy không claim direct stochastic decoder là novelty. Kết quả search là **search-bounded prior-art assessment**, không chứng minh toàn bộ literature chưa có cùng ý tưởng. Các paper trajectory forecasting/video moment retrieval được loại khỏi evidence cho SLP vì task khác, dù có uncertainty/refinement keywords.

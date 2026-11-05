# Nhật ký tìm kiếm — nghiên cứu bắt đầu lại 04/11/2026

Không import search log, ledger hoặc kết luận của proposal trước. Search engine không cung cấp tổng corpus/hit count tin cậy; không tự dựng PRISMA counts. Kết quả tìm kiếm chỉ là discovery cho đến khi mở nguồn gốc.

| ID | Query thực chạy | Mục đích và kết quả ban đầu |
|---|---|---|
| Q001 | `site.openaccess.thecvf.com CVPR 2026 text motion generation` | Discovery hiện đại, kết quả nhiễu video; cần venue index |
| Q002 | `site.openreview.net ICLR 2026 human motion generation` | Discovery; cần mở decision thay vì chỉ tên venue trong PDF |
| Q003 | `human motion generation 2025 2026 survey text motion whole body` | Định vị taxonomy; nguồn thứ cấp không dùng làm bằng chứng cơ chế |
| Q004 | `site.openaccess.thecvf.com/content/CVPR2026 "motion" "generation"` | Tìm được trang CVF và chương trình CVPR2026; tiếp theo khai thác index |
| Q005 | `site.openaccess.thecvf.com/content/ICCV2025 "motion generation"` | Kết quả chưa đủ hệ thống; cần index |
| Q006 | `site.openreview.net "2026" "text-to-motion"` | Discovery T2MBench, MultiAct; chưa xác minh venue/đọc method |

Các tên mới như OpenT2M, MoLingo, FrankenMotion được tìm thấy từ chương trình chính thức/search nhưng chưa được nâng thành claim hoặc D. Đang thu metadata nguồn gốc để sàng lọc.

## Các vòng tiếp theo (04–05/11/2026)

| ID | Query thực chạy | Kết quả sử dụng |
|---|---|---|
| Q007 | `site.openreview.net "ICLR 2026" "motion" "autoregressive"` | Tìm OmniMotion; thứ cấp chỉ làm pointer |
| Q008 | `site.openreview.net "ICLR 2026" "motion" "representation"` | Tìm Motion Generalist; acceptance chưa xác minh |
| Q009 | `site.proceedings.mlr.press "2025" "motion generation"` | Official ICML2025 Being-M0/MotionLib; chưa đọc method |
| Q010 | `site.proceedings.neurips.cc "2025" "motion" "generation"` | Tìm thêm nguồn; phải phân biệt track |
| Q011 | `"OmniMotion" "openreview.net/forum"` | OpenReview id qKaMU1l7Lc; forum trả browser challenge |
| Q012 | `"Motion Generalist" "openreview.net"` | OpenReview id mx9jLnzQGr; forum trả browser challenge |
| Q013 | `"MoLingo" motion language alignment` | arXiv 2512.13840 và repo tác giả |
| Q014 | `"FrankenMotion" part-level generation composition` | arXiv 2601.10909 và repo tác giả |
| Q015 | `"OmniMotion" "Continuous Masked" arxiv` | arXiv 2510.14954 |
| Q016 | `"Motion Generalist" "Multimodal Motion Generation" arxiv` | Chưa có metadata arXiv xác minh, không tự tạo reference |
| Q017 | `"motion generation" "2026" "ICLR" site:openreview.net/forum` | Tìm nguồn hiện đại và Möbius; chỉ giữ nguồn gốc khi mở |
| Q018 | `"motion generation" "NeurIPS 2025" site:proceedings.neurips.cc/paper_files` | GMMotion, SoPo, VimoRAG; cần đọc và sàng lọc riêng |
| Q019 | `"text-to-motion" "2026" "benchmark" semantic evaluation arxiv` | T2MBench; official ICML2026 Zero-Shot T2M Evaluation |
| Q020 | `"whole-body motion generation" "2026" hands face arxiv` | HandX; các nguồn robotics chỉ giữ nếu có cơ chế chuyển giao |
| Q021 | `"text-to-motion" "2026" "temporal" "planning"` | Event-T2M, TPPMG; không suy duration từ tên |
| Q022 | `"text-to-motion" "2026" "tokenization"` | LG-Tok; STRIDE evaluator diagnosis |
| Q023 | `"Language-Guided Transformer Tokenizer" arxiv` | arXiv2602.08337; accepted ECCV2026 theo tác giả |
| Q024 | `"Plan, Don’t Pose" motion arxiv` | arXiv2605.29906, đã acquisition nhưng chưa đọc |
| Q025 | `"Event-T2M" arxiv` | arXiv2602.04292; official OpenReview PDF pointer |
| Q026 | `"Motion Generalist" "arxiv.org/abs"` | Chưa xác lập reference; không dùng summary thứ cấp |
| Q027 | `site.arxiv.org "co-speech gesture" "semantics" "tokenizers" "2026"` | arXiv2610.03765, đã đọc sâu sau đó |
| Q028 | `site.arxiv.org "motion generation" "2026" "hands" "face"` | FUSION/HandX; SignMimic là video driving, chưa xem là text-to-motion SLG |
| Q029 | `site.proceedings.mlr.press "2026" "motion generation" "alignment"` | ICML2026 nguồn transfer và evaluation |
| Q030 | `site.aclanthology.org "2026" "motion generation"` | Search coverage chưa đủACL; chưa kết luận vắng relevant work |

## Truy xuất trực tiếp và trạng thái ở vòng đầu

Các mục sau ghi trạng thái lúc thu nguồn, không phải reading status cuối. Ledger hiện tại mới là nơi tra phạm vi đã đọc.

- Official CVF indices: CVPR2026 4.042 titles, CVPR2025 2.871, ICCV2025 2.701, CVPR2024 2.716. Regex discovery có 559 records còn lẫn video/recognition; narrow title screen giữ 131 records để xem tiếp. Không phải 131 papers đã đọc/included studies. Raw indices/hash/URLs ở `venue_discovery.json`.
- ECCV2024 qua CVF trả 404, ECVA papers index truy cập được. Chưa đối chiếu ECCV2026 index nên không tự nâng author-reported acceptance thành proceedings verification.
- `official_metadata/selected_records.json` có 12 CVF metadata pages với abstract/PDF/arXiv links/hash. SLG titles xuất hiện trong discovery trước khi đọc sâu; không dùng method của chúng để chọn trước motion stage.
- `acquire_arxiv.py` lưu metadata/full text/hash, không tự nâng reading level. Chỉ explicit PROGRESS với scoped notes được gán D; unread vẫn giữ acquired/unread.
- MoLingo/LG-Tok metadata được mở để kiểm tra versions; Möbius 2611.05279 mới nộp 03/11/2026. Vòng đầu chưa đọc method; sau đó có selective method reading được ledger ghi M.
- OmniMotion/Motion Generalist OpenReview gặp browser challenge; không suy acceptance. Event-T2M lúc đầu cũng gặp challenge, sau đó được xác minh official ICLR entry. MATE/Motion-R1 mới official abstract/metadata, chưa dùng như deep mechanism evidence.
- ICLR index timeout nhưng individual entries truy cập được; timeout không chứng minh paper vắng mặt. Fresh MoLingo clone commit `52e3b4c309256ce1bb29c8c038fe34b88682db08`; scoped static audit trong M01, chưa model reproduction.

Hạn chế của vòng đầu gồm non-CVF screening, transfer sources và targeted novelty search. Các vòng sau bổ sung MAR/speech/ACT và sát temporal mechanism; review vẫn không nhận exhaustive coverage.

## Q031–Q036 và cập nhật version

Q031 `"HumanTOMATO" site:openreview.net/forum`; Q032 `"Event-T2M" site:proceedings.iclr.cc`; Q033 `"OmniMotion" site:proceedings.iclr.cc`; Q034 `"Motion Generalist" site:proceedings.iclr.cc`; Q035 `"HumanTOMATO" site:proceedings.mlr.press/v235`; Q036 `"EMAGE" "Code" "CVPR 2024"`. Event-T2M đã xác nhận entry official ICLR2026. HumanTOMATO đã xác nhận PMLR235/lu24b ICML2024 và đọc bản xuất bản; khác arXivv1 ở facial generator (RVQ+GPT thay cVAE), tên TMA và baselines/tables. Không dùng hai versions như hai independent studies.

## Q037–Q059, motion synthesis rồi chuyển sang SLG

- Q037: `"Human Motion Diffusion Model" site:openreview.net/forum`
- Q038: `"Executing your Commands via Motion Diffusion in Latent Space" site:openaccess.thecvf.com/content/CVPR2023`
- Q039: `"T2M-GPT" site:openaccess.thecvf.com/content/CVPR2023`
- Q040: `"FlowMDM" site:openaccess.thecvf.com`
- Q041: `"Autoregressive Image Generation without Vector Quantization" site:proceedings.neurips.cc`
- Q042: `"F5-TTS" site:arxiv.org/abs`
- Q043: `"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" site:roboticsproceedings.org`
- Q044: `"Human Motion Diffusion Model" "ICLR 2023"`
- Q045: `"Glow-TTS" site:proceedings.neurips.cc/paper/2020`
- Q046: `"F5-TTS" "Accepted" "ACL"`
- Q047: `"Motion-Aligned Word Embeddings for Text-to-Motion Generation" arxiv`
- Q048: `"F5-TTS" site:aclanthology.org/2025`
- Q049: `"Kimodo" motion generation NVIDIA paper`
- Q050: `"Text-to-Motion" "2026" site:proceedings.mlr.press/v306/`
- Q051: `"motion generation" "2025" site:proceedings.neurips.cc/paper_files/paper/2025/hash/`
- Q052: `"SignSparK" sign language generation`
- Q053: `"sign language generation" "2026" "SIGNER"`
- Q054: `"sign language generation" "2026" "How2Sign" "CSL-Daily"`
- Q055: `"SignFlip" sign language unified generation translation arxiv`
- Q056: `"SignRR" sign language arxiv`
- Q057: `"sign language production" "2026" "gloss-free" "3D"`
- Q058: `"SOKE" "Signs as Tokens" "github"`
- Q059: `"MaDiS" "github.com" sign language`

Q052 trở đi được thực hiện sau khi motion synthesis/cross-paper inventory đã được viết. Search results thứ cấp chỉ là discovery, không dùng số liệu như evidence. SOKE/MaDiS arXiv metadata được mở trực tiếp; fresh official clones đặt trong research namespace. Các claim benchmark từ secondary pages chưa được dùng.

## Q060–Q081 — SLG mới và kiểm tra prior art sát cơ chế (05/11/2026)

Q060–Q063 được thực hiện ở lượt đọc SignSparK/SIGNER trước khi đồng bộ log; các truy vấn được ghi từ lịch sử tool call, không dựng thêm search coverage.

| ID | Query thực chạy | Kết quả ảnh hưởng nghiên cứu |
|---|---|---|
| Q060 | `"SignPR" "sign language" site:openaccess.thecvf.com` | Chưa xác minh được primary source; không cite như paper đã đọc |
| Q061 | `"Focal-General Diffusion Model" "Sign"` | Tìm project FGDM, sau đó tải official CVPR PDF |
| Q062 | `"nonmanuals" "grammatical and prosodic roles" Pfau Quer` | Chapter metadata; không claim đã đọc chapter |
| Q063 | `"Text-Driven 3D Hand Motion Generation from Sign Language Data"` | HandMDM/BOBSL3DT; acquired full text, mới abstract/introduction |
| Q064 | `sign language nonmanual markers grammatical linguistic facial expressions primary study` | Tìm ASL empirical source, sau đó đọc abstract/introduction có phạm vi |
| Q065 | `How2Sign CSL-Daily PHOENIX 2014T dataset official splits` | Tiếp theo mở trực tiếp ba dataset pages chính thức |
| Q066 | `motion generation contrastive denoising loss hard negatives semantic alignment diffusion text sign language` | Counterevidence: semantic/contrastive supervision đã có nhiều prior art |
| Q067 | `sign language generation articulatory contrast counterfactual handshape semantic loss diffusion` | Tìm phonology-conditioned paper, PIDiffSign; chỉ xác minh abstract, chưa deep review |
| Q068 | `text motion generation latent segment planning duration gloss free sign language motion units` | Tìm thêm nguồn planning; không lấy snippet làm method evidence |
| Q069 | `"sign language generation" "non-manual" "planning"` | Tìm non-manual conditioning trong KSL |
| Q070 | `"sign language" "latent plan" generation duration` | Rà claim latent planning |
| Q071 | `"motion generation" "latent" "segment" planning duration hierarchical` | Tìm SegMo/TPPMG và phân biệt action captions với spoken translation |
| Q072 | `"sign language generation" overlapping temporal face hands semantic planning` | Giới hạn claim về overlapping scopes; chưa thấy bằng chứng đủ để gọi mới |
| Q073 | `"TPPMG" site:ijcai.org` | Xác minh official IJCAI entry; mới abstract |
| Q074 | `"Text-to-Korean" "Non-Manual Signal" "2026"` | Publisher/PubMed/PMC xác minh DOI, đọc chọn lọc methods |
| Q075 | `"sign language" "latent plan" "duration" generation -site:papernotes.org -site:researchgate.net -site:sciencedirect.com` | Phát hiện BLANC qua discovery, tiếp theo tìm primary sources |
| Q076 | `"BLANC" "Articulatory Clock"` | Primary arXiv 2610.24109 và project; ảnh hưởng trực tiếp novelty guard |
| Q077 | `"BLock-wise Adaptive diffusioN" sign` | Đối chiếu cùng công trình, không tính là study thứ hai |
| Q078 | `"Latent Articulatory Clock"` | Xác minh cơ chế đã có; acquire/read main và App B/C/D |
| Q079 | `motion generation marginalizing segmentation uncertain boundaries semi Markov text motion` | GP-HSMM, SegMo và temporal uncertainty literature; đọc sâu SegMo |
| Q080 | `"sign language generation" "segmentation" "uncertainty"` | Tìm counterevidence và kiểm tra broad novelty |
| Q081 | `"motion generation" "boundary" "marginalization"` | Nhiều kết quả video segmentation không cùng task; không nhập vào corpus như motion-generation evidence |

### Nguồn được mở trực tiếp và acquisition bổ sung

- Ba dataset pages: How2Sign, CSL-Daily, RWTH-PHOENIX-2014T. Dùng cho dataset/protocol contract; chưa audit actual downloaded manifests.
- FGDM official project/PDF. Nút arXiv của project trỏ placeholder `XXXX.XXXXX`; không dùng link đó làm reference. PDF thật lưu dưới `sources/fgdm/`.
- BLANC arXiv/main+appendices, public project under review; chưa code release audit. Không truy cập implementation nội bộ dù tên method có thể giống nguồn khác.
- SegMo WACV 2026 official PDF được tải và đọc có phạm vi; giữ phản bằng chứng uniform segmentation thắng CPD/clustering.
- KSL NMS publisher request trả 403; full text lấy từ PMC thành công. Không biến download failure thành evidence paper không tồn tại.
- ASL non-manual PLOS ONE/PMC: chỉ đọc abstract/introduction. GP-HSMM và UAI temporal uncertainty hiện chỉ xác minh abstract; cần đọc sâu nếu dùng để bảo vệ khác biệt kỹ thuật cụ thể.

Mốc review hiện tại là 05/11/2026. Đây là critical scoping review và targeted novelty search, không phải exhaustive search trên mọi venue, không có reliable corpus denominator hoặc PRISMA hit counts. Các abstract-only discoveries không được dùng thay deep evidence cho main method.

### Đối chiếu primary sources sau Q081

Không chạy thêm search query: mở trực tiếp full text/PDF GP-HSMM 2017 và UAI2018 Adams–Marlin từ official links; đọc chọn lọc method/inference và setup, ghi M19 ở mức M. Kiểm tra public SMPL-X JOINT_NAMES và SOKE load/decode để xác nhận active body map, lưu source hash. M19 không được tự nâng D; old abstract-only status ở đoạn log trên mô tả thời điểm trước bước này.

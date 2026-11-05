# Kế hoạch thực nghiệm để kiểm tra PACT-Sign

Đây là kế hoạch [P], **chưa chạy**. Không thay thế benchmark bằng pilot nhỏ; pilot chỉ dùng để quyết định có nên mở rộng full experiment hay không.

## 1. Stage 0: khóa dữ liệu và evaluator

Chốt Track A/B/C/D trong [03](03_benchmark_contract.md), snapshot hashes và provenance của mọi checkpoint. Không dùng các bảng/results nội bộ hiện tại để chọn method hoặc hyperparameters.

Kiểm tra evaluator bằng các trường hợp có diễn giải được: self-comparison, rigid rotation/translation, wrist shift, finger perturbation, sequence duplication và đảo token order. Xác nhận DTW accumulated/normalized và alignment thật sự. Với SiBLEU phải resolve unigram-vs-4gram discrepancy; giữ tên khác nhau cho hai output. Chạy lại official baseline trên cùng split và cùng code trước khi xem mức vượt bảng paper là claim tái lập.

BT của SOKE nếu không có đúng checkpoint/preprocessing thì đánh dấu unresolved; có thể báo thêm BT release mới nhưng không gọi đó là SOKE BLEU-4. Khóa SiCLIP retrieval pool, batch dependence và duplicate text policy. Lưu GT ceiling theo chính evaluator dùng trong run.

**Gate:** hoàn tất protocol manifest và có baseline reproducible trên dev/full test theo mục đích. Sai khác với paper cần giải thích bằng version, asset hoặc code; không thay target đến khi khớp con số.

## 2. Stage 1: bottleneck audit trên train/dev

| Diagnostic | Câu hỏi trả lời | Kết quả làm đổi hướng |
|---|---|---|
| GT→tokenizer→pose | Quantization mất bao nhiêu chi tiết? | Reconstruction hand/contact kém → nhánh RVQ/FSQ |
| Free-run prior→pose | Generation gap so với reconstruction? | Gap lớn về lexical/order → ưu tiên prior/alignment |
| Train/dev error stratification | Coarse confidence có dự đoán được lỗi? | Không có correlation hữu ích → bỏ q hoặc sửa estimator |
| Oracle coarse/anchors vs predicted | Ceiling của correction có thực tế không? | Chỉ thắng với oracle → main hypothesis chưa đạt |
| Text-shuffled vs correct text | Model có sử dụng text? | Output/semantic scores gần như không đổi → conditional collapse concern |
| Sign-order/minimal-pair subset | Refiner giữ nội dung hay chỉ làm mượt? | Hình học tăng, nghĩa giảm → bác bỏ phương pháp hiện tại |

Đo theo dataset, sentence-length bins, one/two-hand activity, contact-rich segments, rare training tokens và fitting-quality bins nếu có bằng chứng quality. Không gọi “rare sign” chỉ vì token ít gặp; token không bằng gloss. Những nhóm cần annotation bổ sung phải có quy trình và trạng thái missing rõ ràng.

## 3. Stage 2: controlled pilot, CSL-Daily dev

Chọn CSL-Daily làm pilot vì có dữ liệu câu và gloss để kiểm tra lỗi, nhưng **main Track A không dùng gloss**. Chọn architecture/hyperparameters trên dev; How2Sign và PHOENIX vẫn là bắt buộc của full evaluation.

| ID | Model | Điều kiểm tra |
|---|---|---|
| B0 | Official/reconciled MaDiS | Prior gốc |
| B1 | B0 + deterministic temporal refiner, FK/velocity losses | Liệu smoothing/regression đã đủ? |
| B2 | B0 + plain conditional flow, constant noise, no reliability/relations | Đối chứng ghép đơn giản mạnh nhất |
| B3 | B2 + same-frame cross-part blocks, parameter budget matched | Architecture coupling có đóng góp? |
| P0 | B3 + generated-conditioning training + reliability-conditioned source | Core preservation/repair hypothesis |
| P1 | P0 + relational geometry loss | Bimanual articulation ngoài marginal hand metrics |
| P2 | P1 + soft sparse evidence/preservation | Anchor selection có thật sự cần? |

B2/P0 phải nhận **cùng** text features, coarse samples, length và data. So sánh P0 với B2 theo cả equal updates và equal GPU-hours; không để model đề xuất thắng vì train lâu hơn. Nếu P2 không giúp, final method là P1 hoặc P0, cập nhật lại mô tả đóng góp tương ứng.

Vòng đầu không thêm retrieval, rich-face fitting, RL, new tokenizer hay semantic reward. Những nhánh này chỉ mở sau khi xác định gain của correction.

## 4. Ablation bắt buộc trước claim contribution

| Trục | Các mức | Kiểm tra giả thuyết |
|---|---|---|
| Reliability | constant / random / entropy-only / learned / oracle-train diagnostic | Learned q tốt hơn chỉ capacity hoặc ordinary confidence? |
| Source | standard Gaussian / coarse+constant noise / coarse+q noise | Lợi ích từ conditional transport hay chỉ coarse mean? |
| Conditioning | GT reconstruction only / corrupted tokens / free-running / cross-fit subset | Exposure mismatch có giải thích gain? |
| Evidence | no sparse evidence / uniform / kinematic events / oracle keyframes | Event selector có hơn uniform? |
| Adherence | no penalty / constant soft / reliability soft / hard clamp | Giữ đúng và sửa sai có cân bằng? |
| Coupling | independent parts / cross-part blocks / relational loss / cả hai | Architecture và objective có đóng góp riêng? |
| Geometry | local hands / torso placement / relational distances | Có metric blind spot khi chỉ align wrists? |
| Timing | predicted EOS / train-time length oracle diagnostic | Gain có phụ thuộc biết target length? |
| Sampling | 25/10/4 steps; 2/1 sau distillation | Chất lượng và compute trade-off |
| Representation | SOKE schema; richer face riêng Track D | Gain architecture hay fitting? |

Oracle rows chỉ là diagnostics, không được là candidate để chọn final deployment hay headline score. Confidence randomization giữ nguyên histogram q để kiểm tra vai trò vị trí của reliability, không chỉ tổng mức noise.

## 5. Stage 3: full benchmark

Train/refine trên ba datasets với nguồn pretraining đã kê khai. Evaluate full official test splits; lưu mọi sample ID, seed, predicted tokens, predicted length, coarse/final pose, confidence và timing. Không bỏ sample rỗng, dài hoặc lỗi fitting mà không báo.

Baselines: SOKE full và no-dictionary; MaDiS; B1/B2; SignSparK no-KF dưới cùng fitting khi có thể train lại. SIGNER và SignRR có gloss setup riêng; M3T có richer fitting, so theo track phù hợp. MARDM/DC-style continuous or hybrid baseline trên sign data là đối chứng transfer quan trọng nếu muốn claim hơn một adaptation của MaDiS.

Không copy model output từ một benchmark có target keyframes sang headline text-only table. Với baseline không có code/assets để tái lập, ghi reported-only và lý do; không dùng một bản tự triển khai yếu làm đại diện duy nhất cho paper mạnh.

## 6. Metrics và tiêu chí quyết định

Primary ban đầu trên dev: trung bình tương đối của **DTW-JPE body và hands** so với B0, nhưng hai thành phần vẫn được báo riêng. Đây chỉ là rule chọn checkpoint trong nghiên cứu, không tạo một benchmark metric mới thay bảng gốc. Guardrails: SiCLIP độc lập, token BLEU đúng định nghĩa, BT nếu tái lập được, và tỷ lệ semantic failure trên subset đã khóa.

Metric phụ:

- PA-DTW đúng protocol SOKE; wrist-local hand và torso-relative hand-placement errors.
- Relational-distance/contact event accuracy trên pseudo-label có kiểm tra chất lượng; không coi pseudo-contact là annotation ngôn ngữ thật.
- Velocity/acceleration distribution, jerk và duration error; so với pseudo-GT lẫn nguyên video trên subset, vì smoothing có thể cải thiện vẻ tự nhiên mà giảm khớp jitter.
- Correct-vs-shuffled text scores, paired-vs-random distances; tham khảo Conditional Collapse nhưng ghi đúng encoder/protocol, không tự nhận tái lập paper khi khác schema.
- Memory, end-to-end latency và throughput tại batch=1 và batch lớn; ghi số lần gọi cả prior và refiner.

**Pilot decision rule đề xuất:** mở rộng nếu P0/P1 cải thiện ít nhất khoảng 5% primary geometric error trên dev, không làm một part xấu quá 2%, và semantic guardrails không giảm có ý nghĩa. Các ngưỡng là *mức cải thiện thực dụng đặt trước*, không phải kết quả mong đợi hay power calculation. Nếu không đạt, phân tích bottleneck rồi đổi hypothesis; không dùng test để hạ ngưỡng.

Để claim SOTA cuối cùng, phải cải thiện so với đối thủ mạnh nhất trong track tương thích, không chỉ vượt ngưỡng pilot hay vượt SOKE. Nếu geometry tốt hơn nhưng semantics xấu hơn, báo Pareto trade-off và không claim superiority tổng thể.

## 7. Thống kê và đánh giá bởi người ký hiệu

Chạy 3 seeds cho các model chính; pilot một seed không đủ kết luận cuối. Báo mean/std và paired bootstrap CI của chênh lệch theo sentence; nếu samples thuộc cùng source video/signer, dùng cluster bootstrap phù hợp. Confidence intervals trên test examples không thay thế variation giữa training seeds. Cố định seed sampling và mẫu so sánh, không chọn seed thắng nhất.

Nếu report nhiều ablation, chọn final checkpoint/hypotheses trên dev và xác nhận test một lần; công bố số lần mở test. Không best-of-N theo GT hoặc evaluator trên test. Nếu báo N candidates, báo latency tương ứng và một rule chọn chỉ dùng input; metric oracle best-of-N để cột riêng.

Human study: cộng tác với Deaf/native signers cho từng ngôn ngữ được claim; random/blind model labels, cùng renderer/view/speed; đo sentence comprehension, lexical correctness, non-manual adequacy và naturalness riêng. Không cho text đích trong lượt đánh giá comprehension tự do; nếu cho text để đánh giá alignment, đó là task khác. Thiết kế cỡ mẫu bằng pilot variance và điều kiện tuyển người phù hợp trước study chính; chưa có dữ liệu để bịa power hay số participants tối ưu. Có trả công, consent và ghi uncertainty theo signer/sentence.

## 8. Compute và deliverable khi triển khai

Không suy từ availability phần cứng SLP hiện tại. MaDiS paper train trên 4 GH200, SOKE trên 6 RTX3090; đây là bối cảnh paper, không là ước lượng chi phí PACT-Sign. [MaDiS §4](https://arxiv.org/html/2601.19577v2#S4), [SOKE §4](https://arxiv.org/html/2411.17799v3#S4).

Trình tự tài nguyên: freeze prior, cache outputs → profile refiner 1 epoch/pilot → dự báo GPU-hours từ measured steps/sec → CSL dev ablation → ba dataset/full seeds → distillation. Cross-fit prior có thể chi phối chi phí, phải budget riêng. Ghi cả preprocessing/mesh-loss overhead; không ước lượng VRAM hay thời gian hoàn thành bằng số không đo được.

Artifacts cần cho implementation review: dataset/evaluator manifest, cache provenance, model config, training logs, per-sample outputs, aggregate metric script, seed table, ablation table, qualitative failure gallery và study protocol. Tất cả số chưa chạy để `not run`, không tạo bảng “expected SOTA” với điểm giả.

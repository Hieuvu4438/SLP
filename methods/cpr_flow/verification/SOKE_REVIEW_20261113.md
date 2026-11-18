# Material Passport

- Workflow: academic-research-suite / experiment-agent / validate, inline.
- Verification Status: ANALYZED. Đọc source và log; không tái chạy model/evaluator.
- Thời điểm: 2026-11-13T18:13:26.045744+07:00.
- Scope: standard joint3 seed17; dev panel64clips/corpus, chưa phải full dev/test.
- SOKE reference: official repository commit `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`.
- Không sửa source/config/campaign đang chạy; không dùng thêm GPU payload.

# Đối chiếu evaluator

Pipeline hiện tại chưa được chấm bằng official SOKE evaluator. `eval.official_metrics_adapter` và `eval.backtranslation_checkpoint` trong resolved config vẫn null. Các metric hình học đang mang tên `controlled_v1_*`.

| Hạng mục | Hiện trạng |
|---|---|
| DTW-JPE body/hands | Có controlled implementation; chưa tương đương official SOKE |
| DTW-PA-JPE body/hands | Chưa tích hợp official Procrustes+DTW route |
| B-T BLEU-4 | Chưa nối evaluator/checkpoint tương ứng |
| AE PA-MPJPE vật lý | Chưa có trong log train/dev hiện tại |

Khác biệt cụ thể: `src/cprflow/evaluation/dtw.py` trả cumulative cost / path_length. SOKE `mGPT/metrics/dtw.py` trả cumulative cost. `mGPT/metrics/t2m.py` dùng joint map và hands regressed từ vertices; root alignment được bật bằng `align_idx=0`, Procrustes bằng `align_idx=None`. Comment upstream cảnh báo tên biến PA không đổi khi đổi alignment. Cần pin và ghi rõ alignment/aggregation/joint selection/units, không suy protocol từ tên biến.

Code controlled hiện có hàm kabsch, nhưng geometry_metrics không gọi nó; sự hiện diện helper không có nghĩa đã eval PA-JPE. Pose fitting/release, native-left convention, duration và inference conditions cũng cần đối chiếu trước khi chép điểm paper. SOKE full dùng retrieval; bảng so sánh cần ghi rõ text-only versus retrieval.

Nguồn gốc:
- Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.pdf
- Official code: https://github.com/2000ZRL/SOKE
- DTW: https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/dtw.py
- T2M: https://github.com/2000ZRL/SOKE/blob/5cbc55d84b5a7cbf05a9cf020c468052e8d94d00/mGPT/metrics/t2m.py

# Các kết quả đã đọc

AE step34841/200000; raw train loss trung bình100updates 0.001744. EMA dev composite loss step34000 = 0.061561, từ 1.306969 ở dev đầu tiên. Theo corpus: {"csl_daily": {"dev_loss": 0.059268905548378825, "clips": 64}, "how2sign": {"dev_loss": 0.06167760380776599, "clips": 64}, "phoenix2014t": {"dev_loss": 0.06373565021203831, "clips": 64}}. Gradient/loss finite và skipped0 theo log đã kiểm tra. Đây là bằng chứng về tái dựng motion và sự ổn định tối ưu, chưa là text-conditioned generation.

Teacher step65655/100000. Macro dev Recall@1 mới nhất 30.73%; best 41.67% ở step25000. Điểm mới nhất theo corpus: CSL 54.69%, How2Sign 9.38%, Phoenix 28.12%. Best checkpoint được giữ; dev retrieval suy giảm sau peak là điều cần điều tra, chưa xác định nguyên nhân từ log.

Duration complete30000. Macro absolute log-length error 0.316009; How2Sign 0.434439 cao nhất. Không chuyển log-error thành phần trăm MAE.

AE composite loss có parameter Huber, rotation, normalized FK và scaled velocity; giá trị0.06156 không phải DTW-JPE hoặc PA-MPJPE. Training dùng raw weights, BF16 và sampled auxiliary frames; validation dùng EMA, dense auxiliary và FP32. Không lấy train/dev loss ratio làm bằng chứng overfit khi chưa giữ cùng điều kiện.

Main CFM/C0 chưa bắt đầu. Chưa có generation metrics, official BT, paired C0-versus-B1 effect hoặc uncertainty estimate. Dữ liệu hiện tại không đủ để ước lượng xác suất đạt SOTA. AE có tín hiệu tốt; semantic teacher có dấu hiệu yếu và chưa chứng minh semantic guidance tốt, nhất là How2Sign.

# Kiểm tra diễn giải — 11/11 nhóm

| Nhóm | Đánh giá phạm vi này |
|---|---|
| Simpson | Đã đọc riêng3corpora; AE giảm trên cả3; không suy chỉ từ macro |
| Ecological | Không suy khả năng hiểu từng câu từ macro loss/Recall |
| Berkson/selection | Panel64clip/corpus cố định, chưa đại diện full split được kiểm chứng |
| Collider | Không có mô hình điều chỉnh nhân quả; không áp dụng |
| Base rate | Retrieval có nhiều positive khi text trùng; không gán chance=1/64 |
| Regression to mean | Báo cả best checkpoint và latest; không lấy peak làm dự báo |
| Survivorship | Không bỏ các job đang chờ; dữ liệu train có ledger giới hạn chiều dài |
| Look-elsewhere | Nhiều checkpoint dev được quan sát; chưa có phép kiểm định xác nhận |
| Forking paths | Cấu hình campaign đóng băng; run vẫn exploratory,1seed |
| Correlation/causation | Loss giảm không chứng minh method sẽ thắng SOTA |
| Reverse causality | Không phát biểu nhân quả từ các log này |

Các kiểm tra này là mô tả giới hạn bằng chứng. Chưa chạy significance test, bootstrap hay tái lập evaluator, không có kết luận statistically significant.

# Bằng chứng cần có tiếp theo

AE reconstruction trong physical JPE/PA-MPJPE theo cùng SOKE backend trên dev; text-to-motion full dev sau khi CFM được train; paired C0/B1 metrics và semantic BT dùng cùng checkpoint; cùng pose fitting/split/alignment/duration protocol; test chỉ sau khi protocol/checkpoint được đóng băng. Các số đo này mới tạo cơ sở để đánh giá khả năng cạnh tranh với SOKE/SOTA.

# M19 — Temporal uncertainty và segmental inference đã có prior art

## GP-HSMM, Frontiers in Neurorobotics 2017

**M:** đọc §3, Algorithm 2, setup chọn lọc §§4.1–4.2 và §5; chưa audit đầy đủ mọi bảng/code. [Nguồn](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2017.00067/full), PDF/hash trong `sources/gp_hsmm/`.

Model dùng GP emissions, class transitions và explicit duration; forward filtering/backward sampling suy cả segment lengths lẫn classes từ observed motion. Đây không phải text-to-motion generator. Các experiments là body exercise/karate với reduced features; chưa hỗ trợ sign articulation. Chi phí và số classes cho trước là hạn chế được nêu rõ. Candidate B kế thừa tư tưởng giữ nhiều segmentations, không kế thừa GP model hoặc nhận sampling partitions là mới.

## Temporally Imprecise Labels, UAI 2018

**M:** đọc phần semi-CRF background, §§4.1–4.2, sleep setup và chọn lọc HNS results/runtime; chưa supplementary DP hoặc mọi experiment. [Official entry](https://proceedings.mlr.press/r16/adams18a.html), PDF/hash trong `sources/imprecise_semicrf/`. PMLR reissue năm 2026 không biến study thành nghiên cứu mới năm 2026.

Paper mô hình hóa noisy timestamps và tối ưu marginal likelihood, tổng qua latent segmentations/observation assignments; MAP thay sum bằng max. Task là activity detection. Main proposal B hiện dùng expected NLL theo một frozen heuristic q, không phải objective này. Không được nhận exact marginal learning hoặc nhân kết quả activity detection thành bằng chứng SLG.

## Kết luận đối chiếu cho proposal

Cả distributional segmentation lẫn DP inference đều có lịch sử rõ. Claim ứng viên chỉ còn tác động của structured pseudo-plan supervision lên fully predicted SLG so với point/confidence/jitter. Phép so này chưa được hai nguồn trên kiểm tra. Việc chưa thấy exact combination trong corpus không chứng minh novelty toàn literature; hiệu quả cần experiment.

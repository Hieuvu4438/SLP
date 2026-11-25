# 01 — Câu hỏi nghiên cứu, phạm vi và quy trình

## 1. Bối cảnh

- `methods/motionhiflow_slp` chuyển MotionHiFlow (CVPR 2026, arXiv 2604.23264) sang SLP gần như
  nguyên dạng. Theo người dùng, nó tốt hơn SOKE và SignSparK trên Phoenix test, nhưng thua SignNMFlow
  (arXiv 2609.33339) ở mọi metric trong bảng của SignNMFlow.
- Một bản port nguyên dạng không đủ novelty cho một venue lớn.
- Người dùng yêu cầu bỏ qua mọi proposal hoặc method khác đang có trong repo, và đề xuất từ nghiên
  cứu bên ngoài cùng suy luận độc lập.

## 2. Research Question Brief

**RQ chính.** Có thể thiết kế một phương pháp flow matching phân cấp, xây trên MotionHiFlow nhưng có
cơ sở cho đặc thù của SLP, sao cho:

- (i) vượt SignNMFlow trên Phoenix-2014T, CSL-Daily và How2Sign theo protocol SOKE (DTW-JPE và
  DTW-PA-JPE cho body/hand, B-T BLEU-4), khi so có kiểm soát;
- (ii) có đóng góp phương pháp luận đủ mới để công bố?

**Câu hỏi con.**

| Mã | Câu hỏi |
|---|---|
| SQ1 | (chẩn đoán) Khoảng cách MotionHiFlow-SLP → SignNMFlow đến từ lựa chọn kỹ thuật (target, sampling, codec, text encoder) hay từ giới hạn của ý tưởng hierarchy? |
| SQ2 | (lý thuyết) Với metric một-reference và DTW tích luỹ, mỗi tầng của hierarchy nên đóng vai trò gì: ước lượng hay lấy mẫu? |
| SQ3 | (thiết kế) Làm sao để "ước lượng phần dự đoán được, rồi tạo chi tiết chân thực" đúng ngay từ cấu trúc, chứ không chỉ do model học chưa hoàn hảo? Và giám sát nào khớp với thứ metric đo (FK chính xác, tách theo tham số âm vị học)? |
| SQ4 | (đánh giá) Làm sao chứng minh lợi ích không đến từ việc lợi dụng metric (output ngắn, tĩnh hoặc trung bình)? |

**Chấm FINER** (tự đánh giá, 1–5).

| Tiêu chí | Điểm | Lý do |
|---|---|---|
| Feasible | 4 | Dữ liệu và evaluator có sẵn trong repo; một GPU; ước tính 2–3 tuần GPU cho Phoenix + CSL + H2S |
| Interesting | 5 | Đánh đổi perception–distortion trong SLP chưa được phân tích tường minh, dù đã có nhiều dấu hiệu (NFE1 > NFE5; SRVQ; RobotSTAR) |
| Novel | 3 | "Ước lượng rồi sinh" đã có nhiều tiền lệ (PMRF, CorrDiff, StoRM, DiffSinger, cascaded). Novelty nằm ở ba thứ: (i) phân tích frontier và giao thức matched kinematics cho SLP; (ii) transport phân cấp, không nhận text, có ràng buộc band, cộng $\Lambda$ theo articulator và cross-fitting; (iii) PES chính xác theo metric. Xem 07 §1 |
| Ethical | 5 | Dữ liệu công khai; user study tự nguyện |
| Relevant | 5 | Các metric của lĩnh vực đang bị lợi dụng; đóng góp đánh giá có giá trị độc lập |

## 3. Phạm vi

**Trong phạm vi:**

- SLP gloss-free (text → SMPL-X thân trên + tay + jaw/expression), trên 3 dataset chuẩn với fitting
  SMPL-X của SOKE và NSA.
- Flow matching phân cấp, biểu diễn, loss, decode và đánh giá.
- Retrieval chỉ từ train split (Track R) và reward fine-tuning (Track B), cả hai là tuỳ chọn và được
  báo cáo riêng.

**Ngoài phạm vi:**

- Dùng gloss khi train generator. Gloss chỉ được phép dùng ở evaluator.
- Sinh video hoặc avatar.
- Dữ liệu SMPL-X ngoài (SignAvatars, BOBSL3DT), trừ khi làm một dòng bổ sung có nhãn rõ ràng.
- Fit lại pseudo-GT.

## 4. Tiêu chí thành công

| Mức | Tiêu chí |
|---|---|
| S1, khoa học | Tier A (03 §5): SignHiFlow tốt hơn SignNMFlow-reimpl và MotionHiFlow-SLP trên cả 3 dataset, ở JPE-hand và PA-hand, có CI. Đồng thời không kém ở nDTW, B-T và speed-ratio |
| S2, bảng paper | Tier B: vượt số SignNMFlow báo cáo ở ≥ 3/4 cột DTW trên mỗi dataset, không dùng retrieval. Có retrieval thì báo cáo riêng |
| S3, novelty | Gate G2 (06 §4) thành công: ở matched kinematics, SignHiFlow trội frontier so với flat regressor, CorrDiff-style, PMRF phẳng, MotionHiFlow-SLP và SignNMFlow-reimpl |
| S4, phân tích | Hình perception–distortion, phân tích phổ theo articulator và bảng độ nhạy metric đều tái lập được |

## 5. Quy trình nghiên cứu đã thực hiện

Quy trình theo skill `academic-research-skills:deep-research`, mode full, có điều chỉnh để thiết kế
phương pháp.

1. **Scoping:** đọc code MotionHiFlow (`third_party/MotionHiFlow`), port `methods/motionhiflow_slp`,
   evaluator SOKE trong `src/slp/evaluation`, và toàn văn paper SignNMFlow.
2. **Investigation:** 5 nhánh khảo sát song song, gồm SLP; MotionHiFlow và hierarchy cùng T2M; flow
   matching; biểu diễn và bàn tay; TTS và ngôn ngữ học. Khoảng 300 công trình được rà soát. Mã arXiv
   được kiểm tra qua arXiv API; những gì không tìm thấy được ghi rõ.
3. **Analysis:** tổng hợp ở 02, chẩn đoán ở 03, lý thuyết ở 05.
4. **Composition:** 04 (phương pháp), 06 (thực nghiệm), 08 (cài đặt).
5. **Review:** một reviewer độc lập đóng vai devil's advocate đọc 03, 04 và 05 bản v1.
   - Reviewer tìm ra một lỗi **CRITICAL**: stage mịn "anchored" có nhận text, ở nghiệm tối ưu, suy biến
     thành sampling đầy đủ.
   - Ngoài ra còn nhiều lỗi MAJOR: plan chưa phải ước lượng thật, FK của PES lệch với metric, evaluator
     bị coupling, và novelty bị tuyên bố quá mức.
   - **Revision:** phương pháp được thiết kế lại thành Estimate-then-Transport (v2). Lý thuyết, kế hoạch
     thực nghiệm và các tuyên bố novelty được viết lại; chi tiết ở 07 §5.

**Sai khác so với quy trình chuẩn của skill.** Mode full thường dừng sau Phase 1 để người dùng xác
nhận RQ, và hỏi loại literature review. Mục tiêu (`/goal`) yêu cầu không dừng lại hỏi, nên RQ ở trên
là bản tự đề xuất, và review được làm theo kiểu narrative/integrative. Người dùng có thể yêu cầu
systematic review sau.

**Tuyên bố sử dụng AI.** Toàn bộ khảo sát, phân tích và đề xuất trong thư mục này do một trợ lý AI
(Claude) thực hiện theo yêu cầu của người dùng. Các con số trích dẫn đã được đối chiếu với nguồn ở
mức ghi trong 02. Trước khi dùng cho bản thảo, người dùng nên đọc lại các nguồn chính: SignNMFlow,
MotionHiFlow, SOKE, SignSparK, MaDiS, PMRF.

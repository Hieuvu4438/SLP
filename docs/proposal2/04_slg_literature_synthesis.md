# SLG: từ cơ chế motion generation đến những câu hỏi cần kiểm chứng

Bản tổng hợp này được viết sau [motion review](03_motion_literature_synthesis.md) và sau khi đọc các công trình SLG trọng tâm. **Chưa chọn phương pháp chính ở bước này.** Mọi con số bên dưới là số tác giả báo cáo, trừ diagnostic metric được ghi riêng; chưa có tái lập model trong goal này.

## 1. Đổi loại generator chưa đủ xác định bài toán

Một caption như “a person raises an arm” mô tả chuyển động. Một câu như “Ngày mai có mưa không?” đòi hỏi dịch sang một ngôn ngữ ký hiệu cụ thể rồi thể hiện nội dung đó bằng chuyển động. Không thể coi từng từ tiếng Việt/Anh/Đức là một motor command theo cùng thứ tự. ASL, CSL và DGS cũng không phải ba style của một ngôn ngữ chung. Đây là lý do cần target-language ID khi train đa ngôn ngữ; English input của ASL và BSL không xác định cùng một output language. [NSA §4](evidence/S05_nsa.md), [SignSparK data ablation](evidence/S03_signspark.md), [How2Sign](https://how2sign.github.io/).

Motion literature cung cấp kỹ thuật học phân phối, biểu diễn và tổ chức thời gian. Nó chưa cung cấp sẵn một bộ quy tắc dịch đúng câu spoken language thành signing. Đặc biệt, BABEL frame labels, HandX hand descriptions hoặc timed part prompts của FrankenMotion không tự có sẵn trong ba benchmark SLG. Chuyển giao phải giải thích nguồn supervision, không chỉ thay tên input. [Motion synthesis §§4–6](03_motion_literature_synthesis.md).

## 2. Ma trận các phương pháp và nguồn thông tin

| Công trình | Representation và cơ chế | Input/supervision đáng chú ý | Kết luận có thể giữ | Điều chưa chứng minh |
|---|---|---|---|---|
| NSA, CVPR 2024 | Continuous SMPL-X; GNN, diffusion và LSTM | Text transcript; fitting mới trên How2Sign | Anatomy-aware diffusion và hand weighting là baseline thực chất | Không chứng minh sentence embedding luôn tốt hơn word tokens |
| SOKE, ICCV 2025 | Ba VQ streams, mBART AR multi-head | Multilingual data; full model dùng dictionary retrieval | Token modeling, data và retrieval cùng tạo một hệ mạnh | Không tách được mọi gain thành lợi ích riêng của AR hoặc VQ |
| MaDiS, preprint | Frozen SOKE tokenizer; masked diffusion LM | Text; tri-level token/latent/decoded-pose loss | Đã có supervision vượt token CE và temporal unmasking prior | Temporal grids chưa là linguistic planning; “physical” loss chưa là constraint vật lý |
| SignSparK, author-reported ECCV 2026 | Continuous conditional flow; sparse anchors | No-keyframe, retrieval và completion là ba chế độ | Anchors giàu thông tin giúp realization và control | Không quy kết quả có anchors thành text-only translation |
| SIGNER, author-reported ECCV 2026 | Four-part VQ; discrete diffusion, IPA/LTF | Predicted glosses và durations; gloss supervision | Cách đưa semantic condition theo thời gian ảnh hưởng mạnh trong study | Chưa chứng minh một hệ không gloss bắt buộc cần cùng timeline |
| SignRR, preprint | Real segment assembly; part RVQ và joint decoder | Gloss input; train dictionary; predicted total length | Retrieval và learned global refinement bổ trợ nhau trong protocol | Không có open-vocabulary solution chỉ bằng edit distance |
| SignFLIP, preprint | Shared mT5; GCN, NAR pose decoding | 600–700K pretraining pairs; SLT transfer | Representation transfer có ích; human ratings còn thấp | Chưa tách data/capacity gain khỏi kiến trúc; 2D chưa đủ cho fine articulation |
| BLANC, preprint tháng 10/2026 | Multi-rate VQ; adaptive block diffusion | Motion-derived teacher; duration/EOS từ text và history | Gloss-free scheduling và counterfactual event loss đã có prior art | Chưa có How2Sign/human study; không thắng mọi semantic metric |

Deep notes: [NSA](evidence/S05_nsa.md), [SOKE](evidence/S01_soke.md), [MaDiS](evidence/S02_madis.md), [SignSparK](evidence/S03_signspark.md), [SIGNER](evidence/S04_signer.md), [SignRR](evidence/S06_signrr.md), [SignFLIP](evidence/S07_signflip.md), [BLANC](evidence/S09_blanc.md). Các trạng thái venue không được nâng từ lời tác giả thành xác minh proceedings độc lập.

FGDM được đọc chọn lọc: semantic CTC trên denoiser features đã có trong CVPR 2026. KSL non-manual conditioning cũng đã có, nhưng nguồn đọc hiện chỉ xác minh sample-level conditions. Hai nguồn này giới hạn novelty claim; không dùng làm evidence xếp hạng ngoài phần đã đọc. [FGDM](evidence/S08_fgdm.md), [scope checks](evidence/S10_scope_checks.md).

## 3. Ba tầng sai số cần phân biệt

**Sai số quan sát:** video→fitted motion có thể làm sai ngón tay, orientation hoặc mặt. Nếu reference đã mất contrast, generator học rất chính xác reference vẫn có thể khó hiểu. NSA, SOKE, SignSparK và BLANC dùng những fitting pipelines khác nhau; không coi fitting tốt hơn là generator tốt hơn. Đây là một confound đã xác định, còn việc fitting là bottleneck chính của từng dataset phải được đo. [NSA](evidence/S05_nsa.md), [SOKE](evidence/S01_soke.md), [BLANC App. A](evidence/S09_blanc.md).

**Sai số biểu diễn:** codec có thể mất thông tin trước khi text predictor học. Motion review đã có nhiều phản ví dụ reconstruction tốt hơn nhưng generation kém hơn. Trong SLG cần đo raw reference, codec reconstruction và free generation bằng cùng evaluator. Chỉ đổi VQ sang continuous không loại hết compression loss. [Motion synthesis §§2–3](03_motion_literature_synthesis.md).

**Sai số conditional generation:** model có thể tạo poses hợp lệ nhưng thiếu nội dung, sai thứ tự, sai timing hoặc phối hợp tay/mặt. SIGNER và BLANC cung cấp bằng chứng về lợi ích của temporal conditions; tuy nhiên teacher-derived boundaries và gloss timelines là hai loại supervision khác nhau. Không được gọi mọi motion change point là sign boundary. [SIGNER](evidence/S04_signer.md), [BLANC](evidence/S09_blanc.md).

Đây là khung chẩn đoán, không phải kết luận rằng một tầng chắc chắn thống trị hai tầng còn lại.

## 4. Articulation, coordination và non-manuals

Handshape, orientation, location và movement có thể mang lexical distinctions. Wrist-relative geometry giúp đo handshape nhưng loại bỏ vị trí tay so với mặt/cơ thể. Procrustes có thể xóa sai orientation hoặc scale. Do đó phải đo cả local articulation lẫn body-relative hand placement; một metric PA tốt không đủ xác nhận cả hai. [ASL linguistic source](https://doi.org/10.1371/journal.pone.0086268), [SOKE metric audit](evidence/S01_soke.md), [HandX](evidence/M15_handx.md).

Hai tay cũng không nên được mô hình hóa như hai chuỗi hoàn toàn độc lập. Capacity riêng có thể giữ chi tiết; shared context và joint losses mới kiểm tra được coordination. IPA của SIGNER và body-conditioned hand refinement của BLANC đã xử lý một phần câu hỏi này. Vì vậy “thêm cross-part attention” chưa là novelty. [SIGNER](evidence/S04_signer.md), [BLANC](evidence/S09_blanc.md).

Non-manuals không đồng nghĩa emotion. Nguồn ngôn ngữ học ASL mô tả tương tác thời gian giữa facial components và cấu trúc câu; điều đó không được suy rộng thành một grammar chung cho mọi sign language. Tăng số expression coefficients cũng chưa đảm bảo giữ gaze, mouthing và scope đúng. SOKE 133D giữ một tập body rotations, jaw và 10 expressions; phải kiểm tra những degrees of freedom thực sự được decode trước khi hứa full non-manual grammar. [Scope checks](evidence/S10_scope_checks.md), [SOKE representation audit](evidence/S01_soke.md).

Coarticulation bao gồm sự thích nghi với context, không chỉ giảm jerk tại seam. Smoothing quá mức có thể xóa stroke hoặc hold. SignRR hỗ trợ learned whole-sequence adjustment, nhưng chưa có human comprehension để chứng minh mọi thay đổi giữ sign identity. FlowMDM hỗ trợ joint transition realization khi descriptions/durations đã cho; chưa giải quyết việc tự dịch và tự đặt timeline. [SignRR](evidence/S06_signrr.md), [FlowMDM](evidence/M16_temporal_planning.md).

## 5. Những phản bằng chứng phải giữ trong lựa chọn

1. SIGNER dùng GT duration làm geometry tốt hơn nhưng BT thấp hơn; SignRR có hiện tượng tương tự. Timing khớp reference không đồng nghĩa nội dung đúng hơn theo evaluator.
2. BLANC vượt SOKE no-dictionary trên PHOENIX BLEU-4 trong study của nó, nhưng thấp hơn trên CSL. Một cơ chế có ích không có nghĩa thắng toàn bộ benchmark.
3. SignSparK no-keyframe không tốt hơn SOKE full trên mọi hand metric; retrieval/completion có lợi thế input phải được tách riêng.
4. SignFLIP có tự động metric tốt hơn baseline nhưng human ratings vẫn khoảng 1.7–1.9/5. Tính dễ hiểu cho người dùng còn là khoảng cách thực tế.
5. MaDiS gọi loss là physical không có nghĩa đã có FK/contact constraints; SiBLEU implementation đã audit là unigram overlap có brevity penalty, chưa phải BLEU-4.

Nguồn lần lượt: [SIGNER](evidence/S04_signer.md), [SignRR](evidence/S06_signrr.md), [BLANC](evidence/S09_blanc.md), [SignSparK](evidence/S03_signspark.md), [SignFLIP](evidence/S07_signflip.md), [MaDiS](evidence/S02_madis.md).

## 6. Các câu hỏi còn đủ lý do để xây candidate

| Câu hỏi nghiên cứu | Cơ sở | Phản bằng chứng/rủi ro | Điều cần kiểm tra trước claim |
|---|---|---|---|
| Supervision có giữ được những contrast ngôn ngữ cụ thể mà global alignment bỏ sót không? | STRIDE, gesture probes; human gap của SignFLIP | Hard negatives có thể sai; frozen teacher có blind spots | Contrast annotations đáng tin, evaluator độc lập, cùng data/capacity |
| Giữ nhiều temporal interpretations khi teacher boundaries nhiễu có tốt hơn học một partition không? | BLANC dùng một partition/confidence; SIGNER nhạy cách fuse timing | Boundary uncertainty có thể chỉ làm mờ nội dung; duration distribution đã có ở BLANC | So full posterior với point target và confidence weighting; predicted-plan evaluation |
| Retrieval có thể sửa timing/coordination mà giữ articulation đáng tin không? | SignRR và SignSparK | Prior art rất gần; dictionary coverage và T2G dependency | Equal dictionary; leave-source-out retrieval; preservation/insertion/deletion tests |

Đây là ba hướng khác nhau về đối tượng học: semantic supervision, temporal latent structure và exemplar reuse. Chúng sẽ được đặc tả ngang mức rồi mới so sánh. Không hướng nào đã được xác nhận là bottleneck chính hoặc đủ novelty. Điều kiện benchmark chung nằm trong [benchmark contract](05_benchmark_contract.md).

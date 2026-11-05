# Tự phản biện: novelty, tính khả thi và giới hạn của kết luận

Đây là self-review trong cùng phiên nghiên cứu, có dùng hướng dẫn phản biện của academic-research-suite. Không phải independent peer review, không có cross-model verification hoặc human-read attestation. Không mô phỏng rằng nhiều reviewers độc lập đã xác nhận proposal.

## 1. Claim mạnh nhất có thể bảo vệ hiện tại

Literature đủ hỗ trợ việc **kiểm tra** structured temporal supervision cho SLG, nhưng chưa hỗ trợ kết luận nó là bottleneck chính, là method hoàn toàn mới hoặc sẽ thắng SOTA. Claim đóng góp được giới hạn thành một comparison có thể bác bỏ, trong khi mục tiêu benchmark ba datasets được giữ nguyên: distribution trên các pseudo partitions so với point/confidence/jitter/uniform, đánh giá bằng fully predicted inference.

Điểm mạnh của đề xuất hiện tại là một research question rõ, input audit đầy đủ và controls tách các nguyên nhân thay thế. Giá trị thuật toán và lợi ích thực tế vẫn là các giả thuyết.

## 2. Audit novelty theo từng claim

| Claim có thể bị nói quá | Prior art / bằng chứng | Phán quyết hiện tại |
|---|---|---|
| Continuous motion generation là mới | MLD, MARDM, MoLingo, MotionStreamer | Bác bỏ claim; kế thừa |
| Part-aware body–hands–face hoặc cross-part attention là mới | HumanTOMATO, EMAGE, SIGNER | Bác bỏ claim; adaptation |
| Temporal semantic units/duration trước realization là mới | Event-T2M, FlowMDM, SIGNER; TPPMG discovery | Bác bỏ claim rộng |
| Gloss-free motion-derived partition/duration/EOS là mới | BLANC §§3/App. C | Bác bỏ claim; prior art trực tiếp |
| Counterfactual event loss để buộc dùng plan là mới | BLANC App. C.3 | Bác bỏ claim; không dùng làm novelty |
| Non-manual presence conditioning là mới | Lee & Park, Sensors 2026 §§3.3/4.3 | Bác bỏ claim; còn khác interval scope |
| Semi-Markov DP hoặc uncertainty về motion segmentation là mới | [GP-HSMM](https://www.frontiersin.org/articles/10.3389/fnbot.2017.00067/pdf), [UAI temporal uncertainty](https://proceedings.mlr.press/r16/adams18a.html) | Bác bỏ claim nền tảng; đã đọc chọn lọc method/inference, scope M trong M19; không mượn empirical claims ngoài phần đã đọc |
| Structured distribution supervision cải thiện downstream text-only SLG hơn point/confidence/jitter | Chưa thấy phép so đúng như vậy trong BLANC/SIGNER/SegMo đã đọc | Đóng góp ứng viên trong phạm vi corpus; chưa phải xác nhận không có prior art ngoài corpus |

Nguồn deep và locators: [motion synthesis](03_motion_literature_synthesis.md), [BLANC](evidence/S09_blanc.md), [SIGNER](evidence/S04_signer.md), [SegMo](evidence/M18_segmo.md), [scope checks](evidence/S10_scope_checks.md). Không coi “search không thấy” là bằng chứng tuyệt đối về novelty. Đối chiếu kỹ thuật bổ sung ở [M19](evidence/M19_segmental_uncertainty.md). Search log ghi cụm truy vấn và phạm vi thật; không tuyên bố systematic exhaustive review.

**Đánh giá độ mới:** rủi ro cao, mức đóng góp dự kiến nhỏ hơn một generative paradigm mới. Hướng chỉ đủ sức thành đóng góp nghiên cứu nếu có kết quả cho thấy structured uncertainty mang lợi ích vượt augmentation, cùng một phân tích đáng tin về khi nào nó hữu ích hoặc gây hại. Nếu không, nên trình bày là negative finding hoặc dùng baseline đơn giản hơn.

## 3. Phản biện mạnh nhất với phương án chính

> Model đang học từ một heuristic distribution không được xác nhận là linguistic uncertainty. Expected loss qua sampled partitions có thể chỉ là data augmentation. Những content vectors lấy từ reference chứa chi tiết signer mà text không dự đoán được. Realizer vẫn nhìn toàn text nên có thể bỏ qua plan. Khi đó mọi module thêm vào chỉ làm tăng cost, còn SOTA gain nếu có có thể do capacity hoặc tuning.

Phản biện này chưa bị bác bỏ bằng dữ liệu. Các controls trong E2/E4 mới là cách kiểm tra; viết thêm loss hoặc tên module không giải quyết nó. [Kế hoạch](08_experiments.md).

## 4. Các checkpoints phản biện trong audit hiện tại

Các mục này kiểm tra lại scoping, synthesis và proposal ở trạng thái hiện tại; không giả lập rằng các lần review độc lập đã xảy ra theo thứ tự lịch sử.

### Scoping: RQ có trả lời được không?

**Major:** “đạt SOTA cho SLG” là mục tiêu ứng dụng, chưa là một causal research question. Đã tách thành H1–H3 có controls và H4–H5 giữ mục tiêu ba datasets. Chưa có empirical evidence cho H1–H5; không đánh dấu chúng được xác nhận.

**Major:** giới hạn thời gian/NMM có thể làm RQ quá rộng. Main method hiện chỉ claim temporal-supervision mechanism; output vẫn có jaw/expression/head channels nhưng không claim grammatical scope đã được giải quyết. Expanded non-manual evaluation phải báo phần representation thiếu.

### Synthesis: có chọn lọc bằng chứng thuận lợi không?

**Major:** SIGNER có thể tạo neo về temporal conditioning; BLANC là prior art rất gần. Đã bổ sung SegMo và uniform baseline, đồng thời ghi rõ BLANC không thắng SOKE trên CSL BLEU-4 trong study của nó. MaDiS không bị mô tả sai thành pure token CE; SignSparK no-KF/retrieval/completion không bị trộn.

**Minor:** các papers có venue verification khác nhau, số liệu ở bảng không cùng protocol. Đã tách author-reported acceptance, preprint và official proceedings; benchmark table không dùng để xếp hạng trực tiếp.

### Proposal: logic và tính khả thi có chỗ đứt không?

**Major:** q(τ|X) là pseudo target distribution, không phải posterior đã calibrated trên true signs. Phải kiểm tra entropy/annotation disagreement và teacher corruption. Một distribution đẹp về toán chưa đảm bảo targets có giá trị.

**Major:** expected NLL khác marginal log-likelihood. Method hiện mô tả đúng expected loss/structured plan augmentation, không nhận exact latent-variable inference hay phát minh marginalization.

**Major:** teacher-plan→predicted-plan gap có thể thống trị. Exposure stage dùng length-normalization chỉ trong training không loại hết mismatch. E4 phải đo fully predicted branch; nếu nó thất bại, main claim thất bại, không dùng oracle length để thay.

**Major:** multi-part codec + segment codec là hai bước compression. Phải đo reference→codec→phase-code→oracle realization, không giả định continuous latents tự giữ handshape. Nếu loss đã xảy ra ở đầu, không đổ cho temporal planning.

**Major:** K, plan samples và auxiliary modules làm tăng compute/capacity. Cần K-matched, same-sample-count, parameter/compute-matched controls; nếu không causal interpretation yếu.

**Major:** language grammar chưa được model đảm bảo. Non-monotonic text attention cho phép reorder nhưng không đảm bảo reorder đúng. Không gọi motor partitions là sign units chỉ vì khớp CSLR pseudo boundaries.

**Minor:** Gaussian mixture content prior và overlap windows là thiết kế khởi đầu, chưa tối ưu. Cần variance bounds, inference policy, EOS threshold và cap policy pin trước test. Không chọn mỗi policy khác nhau sau khi biết từng metric thắng.

## 5. Stress tests về lập luận

| Kiểm tra | Kết quả đánh giá hiện tại |
|---|---|
| Bỏ SIGNER, luận cứ còn đứng không? | BLANC và motion composition vẫn hỗ trợ câu hỏi timing, nhưng evidence về linguistic conditioning giảm; không suy bắt buộc cần plan |
| Bỏ BLANC, còn cơ sở motion-first không? | Có alignment/segmental modeling trong motion/speech, nhưng không còn được claim đã có empirical gloss-free scheduling tương tự trong SLG |
| Đảo giả thuyết: uniform/point có thể tốt hơn không? | Có; SegMo và training-variance argument làm giả thuyết đối lập đáng tin |
| Chuyển sang How2Sign có được đảm bảo không? | Không; method không cần gloss nhưng khác miền, length, fitting và annotation coverage |
| Thắng một metric có đủ ý nghĩa không? | Không; cần semantics, articulation, motion quality và comprehension trong matched tracks |
| Reviewer nói đây chỉ là augmentation có đúng không? | Có thể đúng; phải dùng matched jitter/entropy/K controls mới phân biệt được |

## 6. Kết quả vòng sửa và phạm vi điều chưa biết

Các thiếu sót về bản viết đã được xử lý: evidence notes được biên tập lại; motion synthesis/cross-paper inventory/search log được làm rõ; main method bổ sung codec/segment objectives, variance/EOS/cap policy và overlap rule cho exposure training. M19 đối chiếu foundations, còn [claim–source audit](11_claim_source_audit.md) kiểm tra các kết luận quyết định và raw benchmark rows. Ledger giữ M/V/unread thay vì nâng acquisition thành deep reading.

[Diagnostic đại số](evidence/temporal_plan_math_check.json) xác nhận recurrence/overlap trong ví dụ nhỏ và phân biệt expected loss với marginal likelihood. Nó không xác nhận teacher calibration, sign boundaries hoặc downstream gains. SiBLEU diagnostic cũng chỉ xác nhận released scoring behavior.

Những major empirical risks ở §4 vẫn chưa được giải bằng dữ liệu: q có useful không, content prior có generalize không, exposure gap lớn đến đâu, và semantic/articulatory gains có xuất hiện trên cả ba datasets không. Đây là hypotheses và limits của research proposal, với experiments/criteria bác bỏ đã đặc tả; không được đổi thành positive findings khi bàn giao.

Kết luận self-review: proposal có thể được đánh giá và triển khai để kiểm chứng, với contribution claim có giới hạn và rủi ro novelty cao được công khai. Chưa có bằng chứng để nhận method đã đạt SOTA, có grammar guarantee hoặc đã vượt independent human evaluation. Việc thực hiện training, benchmark và human study nằm ở kế hoạch nghiên cứu tiếp theo, không được giả báo trong goal literature review này.

## 7. Công bố phương thức nghiên cứu và giới hạn review

Bộ tài liệu được hỗ trợ bởi công cụ AI để tìm kiếm, đọc full text có phạm vi, đối chiếu nguồn và viết tổng hợp. Code audit là static inspection trên official public snapshots; các kiểm tra thực chạy gồm synthetic SiBLEU diagnostic và toy algebra check cho temporal plans, không phải model evaluation. Không có independent human verification, participant study hoặc tái lập số liệu model.

Review là critical scoping review với citation chaining và targeted novelty searches; không có corpus denominator đủ để nhận systematic exhaustive coverage hoặc meta-analysis. Nguồn body-motion không được dùng như evidence trực tiếp cho grammatical sign accuracy. Các preprint mới có thể còn thay đổi; claim và phiên bản phải giữ tương ứng trong ledger.

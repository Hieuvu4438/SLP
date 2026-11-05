# Literature review: motion generation và các cầu nối liên miền

## 1. Representation quyết định điều gì generator có thể học

**[E]** T2M-GPT cho thấy recipe tokenizer như EMA, code reset và reconstruction/velocity supervision có vai trò lớn trong pipeline VQ→AR. MotionGPT mở rộng vocabulary để biểu diễn motion như một ngôn ngữ và học nhiều nhiệm vụ. Hai bài cung cấp nền tảng cho việc tái sử dụng LM, nhưng token motion là cụm latent học được, không mặc nhiên là từ hay gloss. [T2M-GPT, §3](https://arxiv.org/html/2301.06052), [MotionGPT, §3](https://arxiv.org/html/2306.14795).

**[E]** MoMask dùng RVQ để giữ phần sai số còn lại sau tầng lượng tử đầu tiên; masked transformer sinh base tokens, residual transformer sinh các tầng tiếp theo. MoMask++ trong SnapMoGen chuyển sang các độ phân giải thời gian, codebook dùng chung và một masked transformer. Đây là hai thiết kế khác nhau: *residual depth* không đồng nghĩa *temporal scale*. [MoMask, §3](https://arxiv.org/html/2312.00063), [SnapMoGen, §4](https://arxiv.org/html/2507.09122v2).

**[I]** Với ký hiệu, “thêm RVQ” có thể giảm lỗi tái tạo nhưng đồng thời tăng số quyết định cần suy ra từ một câu văn ít thông tin. Vì vậy cần đo **reconstruction ceiling và generation gap riêng**. Nếu token GT tái tạo tốt nhưng token sinh vẫn sai handshape, sửa tokenizer đơn thuần chưa chắc giải đúng nút thắt. Ngược lại, nếu autoencoding đã làm mất ngón tay/contact thì prior hoàn hảo cũng không khôi phục được bằng decoder rời rạc đó.

**[E]** FSQ thay nearest-neighbor codebook học được bằng lưới scalar có giới hạn và phép làm tròn qua straight-through estimator. Nó bỏ các cơ chế cập nhật/reset codebook của VQ, nhưng không phải định lý rằng mọi tổ hợp scalar đều được sử dụng đồng đều. [FSQ, §3–4](https://arxiv.org/html/2309.15505). **[P]** Nên kiểm tra FSQ khi có collapse thực đo, đặc biệt face, thay vì mặc định đổi tokenizer của benchmark ngay từ đầu.

## 2. Continuous generation: hình học, compression và evaluation

**[E]** MDM dự đoán clean motion thay vì chỉ noise, cho phép đặt loss FK, velocity và contact trực tiếp lên motion ước lượng. MLD nén motion bằng VAE trước khi diffusion, giảm chi phí nhưng đưa reconstruction bottleneck vào pipeline. [MDM, §3](https://arxiv.org/html/2209.14916), [MLD, §3](https://arxiv.org/html/2212.04048).

**[I]** Cầu nối quan trọng với SLG là **supervise trong không gian có ý nghĩa hình học**, không phải sao chép foot-contact loss. Contact chân thường gắn với chống trượt; contact tay–tay hoặc tay–thân có thể mang nội dung của dấu. Phải bảo toàn cả contact đúng và non-contact đúng. Một penalty đẩy mọi bề mặt xa nhau sẽ phá những dấu cần chạm.

**[E]** MARDM kiểm tra ảnh hưởng của các chiều motion dư thừa tới noise schedule và evaluator; phương pháp dùng representation thiết yếu, AE liên tục và masked autoregression với diffusion head cho từng latent. Công trình phản bác cách suy luận “FID của VQ thấp hơn nên discretization luôn tốt hơn continuous modeling”. [MARDM, §2–3](https://arxiv.org/html/2411.16575).

**[P]** Với SLG, cần giữ rotation→FK nhất quán: joint coordinates tính từ cùng mesh model khi đánh giá, chứ không dự đoán rotation và position độc lập rồi chọn kênh có điểm tốt hơn. Báo cả hand articulation trong wrist frame và hand placement trong torso frame, vì evaluator chỉ dùng wrist alignment có thể bỏ qua đặt tay sai vị trí.

## 3. Không cần chọn tuyệt đối giữa token và flow

**[E]** MAR dùng diffusion loss để mô hình hóa phân phối continuous token có điều kiện trên hidden state của sequence model; không cần biến mỗi latent thành class trong một codebook. MARDM đưa hướng này sang motion. [MAR, §3](https://arxiv.org/html/2406.11838), [MARDM, §3.2](https://arxiv.org/html/2411.16575).

**[E]** DC-Motion đã có cấu trúc rất gần một proposal dễ nghĩ ra: discrete structure từ masked prediction, continuous quantization residual từ conditional diffusion, cùng một autoencoding representation. [DC-Motion, §III-B–E](https://arxiv.org/html/2606.14721).

**[I]** Vì vậy “semantic discrete, detail continuous” chỉ là **nguyên lý kiến trúc**, chưa phải novelty. Hơn nữa, số thứ tự token không đảm bảo mã hóa “semantics” và residual không đảm bảo chỉ là “detail”; cần probe, ablation hoặc ràng buộc thực nghiệm. Một lỗi tay nhỏ về tọa độ vẫn có thể là lỗi ngữ nghĩa lớn.

**[P]** Đối chứng bắt buộc là MaDiS + plain continuous refiner, có cùng text, capacity và compute. PACT-Sign chỉ có ý nghĩa nếu reliability và coupled geometry giúp thêm khi prior dự đoán sai, vượt được đối chứng ghép đơn giản này.

## 4. Flow matching và tốc độ: đừng đánh đồng công thức với số bước

**[E]** Flow Matching học vector field theo probability path; với path thẳng từ noise \(x_0\) tới data \(x_1\), velocity target là \(x_1-x_0\). Nó không mặc nhiên bằng \(x_1-x_t\). [Flow Matching, §3–4](https://arxiv.org/html/2210.02747). Chi tiết này quan trọng khi chuyển công thức reconstruction-based training của SignSparK sang code mới.

**[E]** MotionLCM distill motion latent diffusion thành consistency model và thêm Motion ControlNet với supervision sau decoder. MeanFlow học *average velocity* phụ thuộc hai mốc thời gian, dùng quan hệ với instantaneous velocity và JVP cho objective một bước. Đây là những cách khác nhau để giảm số lần gọi mạng. [MotionLCM, §3](https://arxiv.org/html/2404.19759), [MeanFlow, §4](https://arxiv.org/html/2505.13447).

**[P]** Ưu tiên teacher 10–25 bước có chất lượng trước. Sau đó mới thử 4/2/1 bước bằng consistency/distillation. “CFM thì dùng một Euler step” không thay thế cho huấn luyện model có khả năng một bước. NFE phải đếm cả unconditional branch khi dùng CFG.

**[E]** HY-Motion 1.0 kết hợp DiT flow matching với pipeline dữ liệu và huấn luyện nhiều giai đoạn ở quy mô lớn, gồm refinement bằng feedback/reward. [HY-Motion, §2–4](https://arxiv.org/html/2512.23464). **[I]** Bằng chứng đó ủng hộ chất lượng dữ liệu và conditioning tốt; không chứng minh một lab với vài chục nghìn sentence-sign pairs nên tăng model lên hàng tỷ tham số. Skeleton output và dữ liệu của model cũng không mặc nhiên bao phủ ký hiệu ngón tay/mặt.

## 5. Composition và streaming là hai bài toán khác nhau

**[E]** FlowMDM dùng bidirectional diffusion cùng blended absolute/relative positional encodings để nối chuỗi hành động, không phải một flow-matching model chỉ vì tên có “Flow”. DART sinh các motion primitive dựa trên history và text. MotionStreamer dùng causal continuous AE, diffusion head và two-forward training để giảm exposure bias trong sinh online. [FlowMDM, §3](https://arxiv.org/html/2402.15509), [DART, §3](https://arxiv.org/html/2410.05260), [MotionStreamer, §3](https://arxiv.org/html/2503.15451).

**[I]** Coarticulation của dấu cần thông tin trước/sau; causal model đổi chất lượng lấy latency có thể không thích hợp làm lựa chọn đầu tiên để tối ưu benchmark sinh cả câu. Tuy nhiên, bài học huấn luyện trên **history/conditioning do model sinh** rất phù hợp cho refiner nhận coarse motion còn lỗi. Streaming nên là mở rộng có protocol riêng, không trộn vào claim chính.

## 6. Whole-body và co-speech gesture: tách stream nhưng vẫn phối hợp

**[E]** HumanTOMATO dùng hierarchical body–hand quantization và text–motion retrieval prior. EMAGE dùng các VQ-VAEs theo vùng, content/rhythm fusion và masked gesture reconstruction. TalkFlow dùng structured continuous latent streams và tương tác có gate giữa các stream trong AE lẫn flow generator. [HumanTOMATO, §2](https://arxiv.org/html/2310.12978), [EMAGE, §4](https://arxiv.org/html/2401.00374), [TalkFlow, §3](https://arxiv.org/html/2610.40972).

**[I]** Các bài này ủng hộ representation theo vùng để không cho motion lớn lấn át chi tiết nhỏ. Nhưng co-speech gesture không tương đương ký hiệu: acoustic rhythm có sẵn timeline; text→sign phải học cả trật tự ngôn ngữ và timing. Không chuyển giả định “face yếu liên quan body” thành ràng buộc cứng cho SLG.

**[E]** Modular Body-Part Phase Control dùng periodic phase codes và ControlNet để chỉnh biên độ, tốc độ, phase theo vùng. [Body-Part Phase Control, §4](https://arxiv.org/html/2603.19795). **[P]** Có thể thử adapter chỉnh timing; không ép toàn bộ dấu lên sinusoidal manifold vì hold, contact và chuyển handshape có thể không tuần hoàn. Tính đúng phải đo trên dấu tĩnh và dấu lặp riêng.

## 7. Retrieval, speech và robotics: chuyển nguyên lý có điều kiện

**[E]** ReMoDiffuse dùng retrieval cùng semantic/kinematic relevance và selective fusion; TMR kết hợp synthesis với contrastive learning cho retrieval. [ReMoDiffuse, §3](https://arxiv.org/html/2304.01116), [TMR](https://arxiv.org/abs/2305.00976). **[P]** Dùng retrieval như điều kiện có thể sai hoặc thiếu, với dropout và gating, thay vì xem exemplar là target bắt buộc. Đánh giá riêng external dictionary và train-only retrieval.

**[E]** F5-TTS dùng text-guided infilling, ConvNeXt để xử lý text trước fusion với DiT và Sway Sampling. Nó vẫn cần chiều dài speech lúc suy luận; trong setup zero-shot có audio reference để ước lượng tốc độ. [F5-TTS, §3](https://arxiv.org/html/2410.06885). **[I]** Có thể học conditioning và infilling, nhưng không lấy giả định thứ tự character gần timeline speech áp thẳng lên text→sign, nơi word order có thể khác. Không được lấy GT duration khi benchmark là text-only.

**[E]** Diffusion Policy sinh action chunks có điều kiện trên observations; RDT-1B mô hình hóa bimanual action distribution và xử lý các nhóm tín hiệu không đồng nhất. [Diffusion Policy, §2](https://arxiv.org/html/2303.04137), [RDT-1B, §4](https://arxiv.org/html/2410.07864). **[I]** Hai tay nên được mô hình hóa như một hệ có quan hệ, không như hai bản sao độc lập. Nhưng robotics có closed-loop observation và success feedback; pose-only SLG không có cơ chế tự sửa qua môi trường. Không sử dụng robot success rate để suy luận SLG intelligibility.

## 8. Tổng hợp thiết kế

**[P]** Chọn prior ngôn ngữ mạnh đã có, giữ representation của benchmark trong vòng đầu, rồi kiểm tra continuous correction với conditioning thực tế. Đưa reliability và hình học tương tác vào một refiner chung là giả thuyết cần thử. Các nhánh đổi tokenizer, thêm face fitting, thêm retrieval và distillation phải được mở theo ablation, tránh một model có quá nhiều thay đổi khiến không xác định được nguyên nhân cải thiện.

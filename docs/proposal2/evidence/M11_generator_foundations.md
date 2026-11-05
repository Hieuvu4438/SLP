# M11 — Ba nền tảng để tách representation khỏi generator

Ba paper dưới đây đều được đọc sâu trong phạm vi nêu riêng. Đây là các nền tảng 2022–2023 để hiểu những thay đổi của các bài 2024–2026, không phải shortlist phương án cuối. Chưa kiểm tra code của ba bài và chưa chạy lại số liệu.

## MDM: sinh trực tiếp trong không gian chuyển động

Nguồn: [Human Motion Diffusion Model, arXiv 2209.14916v2](https://arxiv.org/html/2209.14916v2). **D:** method §3; text-to-motion setup/results §4.1; geometric ablation trên action-to-motion §4.2, Tables 1–4; editing và limitation §§5–6. Snapshot `sources/mdm/`.

MDM nhận cả chuỗi motion bị nhiễu, diffusion timestep và điều kiện; transformer dự đoán chuỗi sạch x0. Điều kiện text là frozen CLIP embedding, được cộng với time embedding thành một token; mỗi motion frame có projection và positional embedding. Khác mô hình latent, MDM không cần train một tokenizer/autoencoder trước. Diffusion sampling vẫn mô hình hoá phân phối nhiều kết quả, dù loss tại một timestep là squared error của x0.

Objective gồm x0 reconstruction; khi biểu diễn bằng rotations có thể thêm forward-kinematics position loss, velocity matching và foot-contact loss. Foot-contact loss dùng nhãn contact của reference trong **training**, phạt chuyển động của chân đang tiếp đất. Điều này không tương đương bảo đảm contact vật lý cho mọi output hoặc dùng contact oracle ở inference.

Một chi tiết quan trọng: §4.1 nói **không dùng các geometric losses bổ sung trong thí nghiệm text-to-motion**, vì representation HumanML3D đã có joint/contact features. Không được gán gain trong bảng text-to-motion cho những loss chỉ được thử ở action-to-motion.

Training dùng 1.000 cosine diffusion steps, text dropout 10% để học classifier-free guidance, transformer 8 layers/hidden 512, batch 64, 500K steps cho text-to-motion. Inference đi từ Gaussian noise về motion sạch và cần kích thước chuỗi đầu ra. **Table 1 ghi rõ dùng ground-truth motion length.** Khả năng nhận một length bất kỳ không đồng nghĩa tự dự đoán duration từ text.

Bằng chứng và giới hạn:

- Table 1 HumanML3D: MDM FID 0.544 tốt hơn T2M 1.067, nhưng R-precision top-3 0.611 kém 0.740 và matching distance cũng kém. Motion quality và text fidelity là hai trục có thể đi ngược nhau ngay từ nền tảng này.
- Table 3 HumanAct12: bỏ foot-contact loss có FID 0.080 so với 0.100 khi giữ loss; tác giả lại quan sát video có sliding/shakiness nếu bỏ. Đây là ví dụ metric phân phối không đủ phản ánh chất lượng contact, không phải bằng chứng loss vô ích.
- Architecture comparison trong Table 1 không chứng minh backbone nào cũng tương đương: GRU có FID 4.569, trong khi các transformer variants khoảng 0.544–0.767. Claim “agnostic với architecture” cần giới hạn phạm vi.
- Editing/in-betweening ghi đè vùng motion đã biết ở từng sampling step. Điều kiện là reference prefix/suffix hoặc body parts, khác generation chỉ từ text.
- Bản paper này dùng khoảng 1.000 forward passes và báo latency lớn. Không dùng con số đó để kết luận mọi direct diffusion hiện đại đều chậm như vậy.

**Suy luận chuyển giao:** direct continuous generation tránh reconstruction ceiling của tokenizer và cho phép đặt loss trên articulated geometry. Đổi lại, phải học phân phối nhiều chiều từ ít dữ liệu và kiểm tra conditioning, normalization, duration. Chưa có bằng chứng rằng direct pose diffusion tự giữ đúng handshape hoặc ngữ nghĩa SLG; các khả năng này cần đối chứng riêng.

## MLD: giảm độ khó bằng latent, nhưng latent tốt để tái tạo chưa chắc dễ sinh

Nguồn: [CVPR 2023 proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.html), [arXiv 2212.04048v3](https://arxiv.org/html/2212.04048v3). **D:** method §3, setup §4.1–4.2, ablations §5/Tables 4–6, Appendix B.3–B.4, C, H.3. Không dựa vào các bảng main-result chưa đối chiếu chi tiết để xếp hạng hiện tại.

VAE encoder dùng learnable distribution tokens để nén cả chuỗi thành n latent vectors; decoder dùng L frame queries và latent làm memory. VAE được train bằng reconstruction và KL; sau đó đóng băng, train conditional noise-prediction diffusion trong latent. Frozen CLIP-ViT-L/14 tạo một condition embedding. Sau diffusion, chỉ cần một decoder pass để trở về motion.

Cấu hình điển hình dùng latent 1×256, VAE/denoiser transformer 9 layers/4 heads, concatenated condition, 1.000 training diffusion steps và 50 inference steps. Nhận L output queries vẫn cần duration; các phần đã đọc không đủ xác nhận predicted-length benchmark. Dataset là HumanML3D/KIT-ML body motion, chưa có detailed hands/face.

Bằng chứng đối chứng quan trọng:

| So sánh | Kết quả tác giả báo cáo | Điều không được suy ra |
|---|---|---|
| Nhiều latent hơn | Table 4: n=1 reconstruction MPJPE 54.4 mm; n=7: 14.7 mm | Reconstruction tốt hơn chưa chứng minh generation tốt hơn |
| Generation cùng họ | Table 5: n=1 FID 0.473; n=7: 1.011 | Một latent toàn chuỗi có thể dễ học hơn trong thiết lập này, không phải tối ưu phổ quát cho chuyển động dài hoặc SLG |
| AE và VAE | Table 8: latent diffusion với AE FID 5.033; với VAE 0.473 | Ngược chiều với câu chuyện AE tốt hơn ở một số bài sau; phải xét architecture, latent dimension, loss và scaling |
| Conditioning | Table 5: one-token cross-attention FID 1.922; concatenation 0.473 | Không phản bác MoLingo dùng nhiều text tokens với T5; hai thí nghiệm không giữ cùng thông tin điều kiện |
| Prediction target | Table 9: noise prediction FID 0.473, x0 prediction 0.513 | Không phải noise luôn tốt hơn x0 trong raw-pose space hoặc với schedule khác |

Appendix H.3 dùng số epochs khác nhau theo latent size và suy đoán nhiều dữ liệu có thể giúp latent lớn hơn. Phần “cần million-level data” là giả thuyết giải thích của tác giả, không phải scaling theorem. Table 10 so tốc độ theo tổng 2.048 clips và solver; không được nhầm tổng thời gian đó với latency một mẫu hoặc so với MDM mới hơn dùng sampler khác.

**Suy luận chuyển giao:** phải tối ưu latent cho cả detail preservation lẫn learnability và semantic structure. Với SLG, một vector nén toàn câu có nguy cơ làm khó việc giữ thứ tự và dấu hiệu nhỏ; đây là giả thuyết cần đo, không phải kết luận từ Table 4. Nên có matched latent-capacity ablation và kiểm tra từng articulator, thay vì chọn AE/VAE chỉ theo tên họ mô hình.

## T2M-GPT: discrete AR, EOS và exposure bias

Nguồn: [CVPR 2023 proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html), [arXiv 2301.06052v4](https://arxiv.org/html/2301.06052v4). **D:** method §3; setup §4.1; results/discussion liên quan training recipes; Tables 3–7 và Appendix A–D. Chưa kiểm tra code hoặc toàn bộ failure videos.

CNN VQ-VAE nén motion 4× theo thời gian, lượng tử hoá thành code indices; GPT học next-token distribution bằng cross-entropy và causal attention, điều kiện là CLIP text embedding. Inference sample từng code cho đến End token hoặc giới hạn sequence, rồi decode về motion. Codebook có 512 entries×512 dimensions; tokenizer train 64-frame crops. GPT 18 layers, hidden 1.024, 16 heads; maximum motion 196 frames, code-index length tối đa 50.

Tokenizer dùng smooth-L1 reconstruction cộng velocity loss, commitment loss, EMA và code reset. Train generator thay ngẫu nhiên một phần history codes để giảm lệch giữa teacher-forced training và inference dùng codes do model sinh. **Corruption history là một cơ chế training, không đồng nghĩa model đã học chính xác mọi loại lỗi rollout.**

Khác MDM/MoMask cần length input trong các benchmark đã đọc, T2M-GPT học EOS. Nhưng EOS cũng có thể dừng sớm, kéo dài hoặc chạm max length; “không dùng reference length” không tự chứng minh duration đúng ngữ nghĩa.

Các ablation cần giữ khi chọn baseline:

- Table 3: naive VQ không EMA/reset cho generation FID 42.797; EMA+reset 0.116. Một baseline VQ train kém không đại diện giới hạn lý thuyết của discrete representation.
- Table 6: không history corruption cho top-1 0.417/FID 0.140; corruption τ=0.5 cho 0.491/0.116. Semantic score thay đổi đáng kể dù architecture giữ nguyên.
- Table 5: smooth-L1 với velocity weight 0.5 cho rFID 0.070; dùng L1 và cùng weight lại tệ hơn L1 không velocity. Không rút kết luận loss bổ sung luôn hữu ích.
- Table 4: tăng capacity có lợi trên nhiều cấu hình nhưng không đơn điệu trên mọi metric; 18×768 có FID 0.130, tốt hơn 18×1.024 có 0.141, trong khi top-1 của bản lớn hơn cao hơn. Phải báo compute và mục tiêu lựa chọn checkpoint.
- Paper phân tích ít dữ liệu có thể vẫn cho motion trông hợp lý nhưng không khớp text. Điều đó không xác định sẵn SLG bottleneck nằm ở tokenizer hay generator.

**Suy luận chuyển giao:** EOS và training với corrupted history là đối chứng bắt buộc nếu đề xuất planner/generator tự quyết định độ dài. Cần kiểm tra semantic coverage theo độ dài câu và lỗi dừng, cùng tokenization detail. Ghép EOS vào mô hình mới không phải đóng góp nghiên cứu độc lập; ưu thế phải được chứng minh khi giữ data, text encoder và capacity công bằng.

## Tổng hợp có điều kiện

Ba bài thay đổi nhiều trục khác nhau: raw versus latent/discrete representation, distribution head, temporal factorization, duration input và training recipe. Chênh lệch bảng số liệu không cho phép kết luận “diffusion”, “AR”, “VQ” hay “VAE” là nguyên nhân duy nhất.

Để so các phương án SLG sau này, cần có cả: direct continuous generator; latent continuous generator; discrete generator được train đúng recipe; oracle-duration và predicted-duration báo riêng. Kết luận lựa chọn chỉ được đưa ra sau khi đối chiếu evidence hiện đại, task SLG và các phương án cụ thể.

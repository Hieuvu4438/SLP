# M13 — DisCoRD: stochastic flow decoder cho discrete motion tokens

**D có phạm vi, chưa code.** Đã đọc §3, §4/Tables 1–6, App. A và C.4; chưa mọi auxiliary-task table/video. Nguồn: [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Cho_DisCoRD_Discrete_Tokens_to_Continuous_Motion_via_Rectified_Flow_Decoding_ICCV_2025_paper.html), [arXiv 2411.19527v4](https://arxiv.org/html/2411.19527v4). Snapshot `sources/discord/`.

## Cơ chế và điều kiện

Giữ pretrained quantizer/token predictor như MoMask, T2M-GPT hoặc BAMM, thay deterministic feedforward decoder bằng token-conditioned rectified flow. Vì thế discrete planning→continuous realization đã có prior art; hybrid pipeline tự nó chưa mới.

Condition projection tạo features riêng cho q frames từ mỗi code embedding bằng repeat/stack/linear/unstack. Cần token–frame mapping của codec; chưa tạo linguistic alignment. Flow path `X_t=(1−t)X_0+tX_1`, noise `X_0`, target velocity `X_1−X_0`; inference Euler, thường 16 steps. Decoder bắt đầu từ noise, không từ coarse pose của decoder cũ, nên khác residual post-refiner.

Train trên reference quantized tokens và motion trong sliding 64-frame windows; test tokens từ predictor. Paper dùng local windows/U-Net không attention để generalize sang generated codes. HumanML3D output 263D, condition 256D, projected input 512D, U-Net 66.9M parameters, batch 768, LR 5e-4, EMA .999. Main không thêm text-semantic objective hoặc thay EOS/length policy của token prior.

## Kết quả và phản bằng chứng

| Câu hỏi | Author-reported evidence | Diễn giải |
|---|---|---|
| Reconstruction | Table 1 MoMask rFID .019→.011, sJPE .512→.385; MPJPE 29.5→33.3 mm | Naturalness/dynamics gain có paired geometry trade-off |
| Generation | Table 2 MoMask FID .045→.032, R@1 .521→.524 | Semantic gain nhỏ, CI overlap; chưa sign precision |
| Coarse codes | T2M-GPT FID .116→.095 nhưng R@1 .491→.476; MoMask R0 cũng giảm alignment | Decoder plausible hơn chưa giữ đủ intended content |
| So post-refinement | Table 6 RF refiner gFID .035/sJPE .489; DisCoRD .032/.385 | Hơn refiners đã thử, không bác bỏ mọi refiner khác |
| Architecture | Repeat+linear rFID .011/sJPE .342 nhưng gFID .038 so .032 | Reconstruction tốt chưa là generated-code robustness |
| Full sequence/local | Full-sequence train rFID .008/gFID .038; windowed .011/.032 | Chọn decoder theo GT reconstruction có thể sai |

SHOW co-speech/AIST++ dance cũng được thử, nhưng khác SLG conditions. §5 còn coi detailed hands/whole-body extension là future work.

## Metric assumptions

sJPE so paired-reference third derivatives bằng normalized jerk difference, tách noise/static components, nhằm phạt cả jitter và under-dynamics. App. C.4 report correlation .483 với naturalness ranks so MPJPE .181, sau khi loại 10% difficult-to-distinguish cases.

Jerk phụ thuộc FPS, smoothing và temporal alignment. Một valid realization khác timing có thể bị phạt dù meaning đúng; fitted reference noise cũng tạo error floor. Naturalness correlation chưa là linguistic comprehension validity. Không dùng sJPE như semantic metric.

## Chuyển giao

Cần tách token prediction, token information và decoder distribution. Nếu prior bỏ nội dung, token-only decoder thiếu bằng chứng để khôi phục đúng nội dung đó. Câu này suy từ conditioning pipeline, không phải SLG experiment của paper.

Nếu khảo sát hybrid decoder, giữ same prior/codes/text data; so deterministic decoder, conditional stochastic decoder, post-refiner được tuning và direct continuous generator. Đo semantic drift, articulation minimal pairs, reference/generated-code gap và compute. FID giảm chưa đủ ưu tiên hybrid phương án.

# S03 — SignSparK: sparse anchors, flow và input contract

**D+C có phạm vi.** Đã đọc main v4, method/setup, Tables 1–8, ablations/human/facial evaluation; static code audit training, sampling, length conditioning và release notes. Chưa chạy model hoặc đọc supplementary riêng. PDF v4 có 23 pages, cuối là references; chưa tìm được separate supplement trong những đường đã kiểm tra, không tính là đã đọc.

Nguồn: [arXiv 2603.10446v4](https://arxiv.org/html/2603.10446v4), [project](https://cogvis-cvssp.github.io/papers/signspark/), [official code](https://github.com/JianHe0628/SignSparK). ArXiv/repo ghi ECCV 2026, chưa đối chiếu proceedings entry. Code commit `916402d0ecd6322fb4a9e5f513229a94a57bf4b6`, snapshot `sources/signspark/`.

## Task và representation

| Chế độ | Input inference theo paper | Phạm vi kết quả |
|---|---|---|
| Gloss-free/no-KF | Spoken text, không pose anchors; duration policy cần kiểm tra riêng | Generation không sparse spatial guidance |
| Retrieval T2P | Text→predicted gloss→dictionary, ba keyframes/gloss đặt cách tám frames | Generation có exemplar articulation |
| Keyframe-to-Pose | Sparse reference/edited anchors và timeline, cùng text | Completion/control, không text-only translation |

FAST ở training đọc video/motion, không đọc text. Không được dùng target-test keyframes do FAST trích rồi gọi text-only result. FAST có hai WiLoR 6D-MANO hand streams, fuse trước Transformer BIO prediction; train frame CE/sign-level CTC trên MeineDGS boundaries. Lấy onset/midpoint/offset mỗi segment. Table 1 F1 .860/IoU .772 so Hands-On .857/.760; gain nhỏ về accuracy, còn có efficiency/scalability.

Representation 296D: body 60, mỗi hand 90, face 56 gồm 50 expressions+6D jaw. Fitting NLF/WiLoR/TEASER khác SOKE. More expression coefficients chưa chứng minh NMM grammar. Paper train multilingual How2Sign/CSL/Phoenix +10% BOBSL. Release sinh các streams riêng; hand sampler split L/R thành batch samples với shared model, chưa có evidence explicit joint hand–face linguistic constraint.

## Flow objective và training

Path giữa noise x0 và reference x1: `xt=(1−t)x0+t x1`, giữ clean anchors tại observed mask, model nhận state/control/time và multilingual CLIP. Drop text/keyframes hỗ trợ no-KF mode.

Eq. 3 viết target `x1−Ct`, khác constant derivative `x1−x0` của linear path ở unobserved positions. Code dùng `vf_gt=x1−keyframed_x0`, bằng 0 ở anchors. Nếu kế thừa phải ghi công thức code-consistent, không copy hai expressions như tương đương.

Reconstruction estimate `x1_est=xt+(1−t)v` có residual `(1−t)[v−(x1−x0)]`; dưới giả định linear path này, pose MSE tương ứng time-weighted flow error. Không là supervision mới hoàn toàn. Velocity/dynamics terms có thể thêm tác dụng, nhưng few-step claim cần matched time-weighting control.

Paper flow/reconstruction/velocity weights 2/1/1; release defaults .5/1/.5 cùng acceleration .1, text/keyframe dropout .1. Không coi current config là exact paper reproduction.

## Kết quả theo đúng tracks

Table 3, author-reported, chưa tái lập:

| Setting/dataset | DTW-PA-JPE body/hand | DTW-JPE body/hand |
|---|---|---|
| No-KF How2Sign | 7.26 / 2.72 | 6.30 / 11.43 |
| No-KF CSL | 7.26 / 2.00 | 6.27 / 10.63 |
| No-KF Phoenix | 5.24 / 1.52 | 4.40 / 7.10 |
| Retrieval CSL | 4.87 / 1.37 | 4.89 / 6.56 |
| Retrieval Phoenix | 3.68 / 1.18 | 3.51 / 4.82 |

Không có retrieval How2Sign row vì pipeline cần gloss annotations. SOKE lookup written words là retrieval contract khác. MaDiS không được same-protocol comparison trong bảng này. No-KF How2Sign body JPE 6.30 thấp hơn SOKE full 7.75 nhưng hand 11.43 cao hơn 10.08; không claim thắng mọi body/hand metric.

Table 4 completion How2Sign JPE 1.68/2.80, BT 2.99, GT BT 3.37; CSL 2.02/2.71, BT 5.85/GT 6.83. Có anchors và evaluator khác SOKE; không dùng để thay main no-KF semantics hoặc so cross-paper BT.

Table 6 segment anchors tốt hơn random, full losses tốt trong keyframe reconstruction; chưa tự chứng minh cùng gain ở zero-anchor generation. Table 7 extra data có trade-offs; language ID giải một phần khi thêm BOBSL. ASL/BSL cùng English input nhưng ngôn ngữ đích khác.

Human study 16 BSL signers, sáu Deaf/mười hearing, forced-choice naturalness/visual alignment: report 86.68% preference so SOKE/S-MotionGPT, 72.63% so pseudo reference. Chưa là blind comprehension hoặc validation mọi target languages. Chi tiết rater/task allocation trong separate supplement chưa kiểm tra được.

Face PA-VPE .07±.02 How2Sign, .06±.02 CSL là geometry với TEASER pseudo reference dưới sparse conditions; không đo grammatical scope của text-only face generation.

## Static release audit

- [Pinned README](https://github.com/JianHe0628/SignSparK/blob/916402d0ecd6322fb4a9e5f513229a94a57bf4b6/README.md): reoptimized fitting/contact, checkpoints train trên khoảng 15× data; current corpus CSL/How2Sign/BOBSL/CSL-News, không Phoenix. Release weights không là equal-data paper reproduction.
- [sample.py L115–150](https://github.com/JianHe0628/SignSparK/blob/916402d0ecd6322fb4a9e5f513229a94a57bf4b6/sample.py#L115): reference length/keyframes vào sampling. `keyframes_mask_all=True` chỉ zero observation mask. [unet_model.py L173–185](https://github.com/JianHe0628/SignSparK/blob/916402d0ecd6322fb4a9e5f513229a94a57bf4b6/signspark/unet_model.py#L173) concat length mask khi flag default bật. **Released no-KF defaults vẫn reference-length-conditioned.** Không suy ngược exact duration policy của Table 3; primary text-only adaptation cần predicted length.
- [rflow.py L226–242](https://github.com/JianHe0628/SignSparK/blob/916402d0ecd6322fb4a9e5f513229a94a57bf4b6/signspark/rflow.py#L226) xác nhận derivative target nêu trên. Clamping ở từng ODE evaluation chưa đủ proof final update giữ exact anchors nếu chưa kiểm tra final state.
- Release thường 50 steps/length 304; paper speed claim 10 steps/.01 s chưa là matched end-to-end text→gloss→retrieval→mọi streams→render latency.

## Chuyển giao

Sparse articulatory conditions có thể giúp realization, còn training với anchors có thể giúp no-anchor generation; hai giả thuyết cần tách bằng no/predicted/retrieved/oracle anchors và equal-data controls. Nếu dùng anchors phải đặc tả ai dự đoán content/timing từ text và sửa sai anchors thế nào. Không mặc định phải dùng flow, cũng không nhận onset/midpoint/offset là đủ bảo đảm sign identity.

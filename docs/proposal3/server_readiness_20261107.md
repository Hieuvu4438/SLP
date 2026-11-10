# Dataset readiness và ước lượng một seed — 07/11/2026

**CSL-Daily có đủ pose + text + pretrained assets để chuẩn bị train RACE. Chưa có cache/manifest RACE nên chưa thể chạy train trực tiếp.** Đây là kiểm tra mới trên filesystem và profiling mới; không suy ra từ kết quả các experiment cũ.

## Dataset kiểm tra thực tế

Đã đọc schema và kiểm tra finite toàn bộ **63.912 file pose**. So với annotation hiện có trên server: không thiếu file, không sai shape các trường SMPL-X, không NaN/Inf, không caption rỗng.

| Dataset | Train | Dev/val | Test |
|---|---:|---:|---:|
| CSL-Daily | 18.399 | 1.077 | 1.176 |
| PHOENIX-2014T, theo annotation hiện có | 7.092 | 519 | 642 |
| How2Sign | 30.965 | 1.724 | 2.318 |

Files: `data/processed/madis/{CSL-Daily,Phoenix_2014T,How2Sign}`. Packed files chứa raw param dictionaries; không được nhầm với normalized 133D output của loader cũ khi export sang RACE. How2Sign có ít nhất 26 train clips dưới bốn raw frames; cần quyết định preprocessing/filter cùng FPS/length policy, không coi tất cả là cache RACE dùng được ngay.

Author MaDiS CSL tokenizer/generator, Qwen model, author mean/std và licensed SMPL-X đều tồn tại. Chưa tìm thấy `data/**/manifest.jsonl` của RACE. Còn phải: export raw133, gán source-video groups vào các role, tạo teacher-token/text caches, generated selector pairs, correspondence/rejection audit và candidate cache. Cache MaDiS đang có không có đầy đủ contract RACE và không được dùng trực tiếp. Dung lượng còn khoảng 200 GiB, đủ dự trù cache CSL; chưa chạy chuẩn bị toàn bộ dataset trong lần kiểm tra này.

## Điều kiện máy và cách đo

GPU là RTX 5880 Ada 48GB, đang có ba process khác chiếm khoảng 45–46 GiB và 94–99% utilization. Không dừng hay sửa các process đó. Probe được giới hạn allocator VRAM, dùng cấu hình đầy đủ 256-width/six-block/FP32 và SMPL-X thật. B8 decoder chạm safety cap của probe; đây không phải bằng chứng B8 không chạy được khi GPU rảnh. Đo được decoder B4, selector/validation B8.

GPU probe dùng pose train CSL thật, resample để kiểm soát tensor shape, code/text condition tổng hợp, warmup rồi đo 3–5 bước. Không lưu checkpoint hay dùng làm kết quả chất lượng. Timing đã tính SMPL-X theo implementation hiện tại, gồm baseline geometry trước window crop; loader/cache IO không nằm trong từng step. CPU preparation dùng author checkpoint thật trên các mẫu CSL train; có một quan sát generation và ba teacher observations, nên ETA preprocessing còn thô.

| Phép đo mới | Median |
|---|---:|
| Decoder train B4, full shape 128, window64+halo8 | 1,58 s/step |
| Decoder train B4, full shape 256, window64+halo8 | 1,94 s/step |
| Decoder train B4, full shape 400, window64+halo8 | 3,11 s/step |
| Decoder validation B8, 128 / 256 frames | 4,05 / 6,95 s/batch |
| Selector training B8, 128 / 256 frames | 7,84 / 15,28 s/batch |
| RACE candidate cache B1, 128 frames, eight flow steps | 0,27 s/sample |
| CPU author generation, 100 motion slots, eight threads | 5,56 s/sample, một quan sát |

Các phép đo **không cô lập GPU**; không thể chia/multiply bằng utilization để suy ra tốc độ GPU dành riêng.

## Một-seed planning scenario

Ước lượng cho **CSL-Daily**, sử dụng pretrained baseline, không train lại SOKE/MaDiS. Giả định chia official train thành 14.719 decoder-train + 1.840 selector-train + 920 selector-val + 920 calibration; giữ original dev 1.077 cho decoder selection và test 1.176 cho external evaluation. Đây là kế hoạch tính thời gian, chưa phải split manifest đã tạo.

Giữ recipe **100 decoder epochs + 30 selector epochs**, decoder batch4 đo được, selector/validation batch8; nội suy theo distribution chiều dài thật. Chiều dài selector generated chưa có toàn bộ, đang dùng distribution pose CSL làm proxy và giả định đủ số cặp qua correspondence guard.

| Giai đoạn | Ngoại suy trong điều kiện GPU đang chạy chung |
|---|---:|
| Decoder training 100 epochs | ~174 giờ |
| Decoder validation qua 100 epochs | ~19 giờ |
| Selector training + validation + calibration | ~34 giờ |
| Chuẩn bị teacher/generated pairs, alignment, cache và startup | ~8 giờ, độ chắc chắn thấp hơn |
| Tổng | **~236 giờ ≈ 9,8 ngày** |

Dự trù engineering **khoảng 7–13 ngày nếu tải cạnh tranh tương tự duy trì**, không phải confidence interval. Bật thêm synthetic robustness stage B 20 epochs cộng khoảng 39 giờ theo cùng ngoại suy. Epochs chỉ là recipe cấu hình, chưa được chứng minh tối ưu/converged. Không có ETA đủ tin cậy cho GPU dành riêng hay default decoder batch8; cần đo lại khi đủ tài nguyên. Thời gian chờ các job hiện tại hoàn tất không nằm trong tổng này.

## Điểm cần xử lý trước khi chạy dài

Chi phí lớn hiện nằm ở geometry, không chỉ decoder ~10,8M parameters. Decoder tính lại baseline geometry ở full padded length trước mỗi window; selector tính lại geometry và paired surrogate labels từ các candidate đã frozen qua mọi epoch. Cache các đại lượng tĩnh và nhãn selector có thể loại bỏ công việc lặp mà vẫn giữ objective, nhưng **chưa benchmark bản tối ưu nên không đưa số speedup giả định**. Không khuyến nghị launch full seed ngay với trạng thái dataset/cache/GPU hiện tại.

Evidence chi tiết: `outputs/race_readiness_20261107/{data_audit.json,batch_shapes.json,profile_*.json,preparation_cpu.json,one_seed_estimate.json}`. Source probes được lưu cùng thư mục. Các số ở đây là dự trù compute, không phải kết quả SOTA hoặc đánh giá semantic quality.

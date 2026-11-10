# Chọn benchmark từ bằng chứng paper và code

Ngày khảo sát: 06/11/2026. Đây là nghiên cứu thiết kế; chưa có kết quả training của method mới. Số liệu được đối chiếu lại từ paper tải trực tiếp bên ngoài trong lượt tiếp theo; không dùng experiment có sẵn của repo SLP. Xem [external evidence](06_external_evidence_review.md).

## Quyết định

Ưu tiên **protocol pose của SOKE, đánh giá chính CSL-Daily, xác nhận thêm PHOENIX-2014T**, và chạy SignSparK như baseline trên cùng dữ liệu khi adapter được kiểm chứng. Lý do là tính khả thi của việc cố định pose/evaluator, không phải bằng chứng rằng benchmark nào “dễ đạt SOTA”. PHOENIX nhỏ hơn phù hợp kiểm tra pipeline nhưng domain thời tiết không đủ chứng minh generalization. Nếu chỉ đủ tài nguyên một dataset, giữ CSL-Daily làm kết quả chính.

Dùng hai track tách biệt: **GF** chỉ text và ID ngôn ngữ, **dictionary** thêm đúng dictionary được cố định. Model pretrained đa ngôn ngữ phải khai báo ba ngôn ngữ và mọi dữ liệu pretraining. Không gọi model fine-tune CSL là monolingual từ đầu nếu tokenizer/base đã học multilingual. Với benchmark SOKE gốc, dictionary track mới so trực tiếp hàng SOKE cuối cùng; GF phải so hàng no-dictionary. [SOKE §4, Table 1–2](https://arxiv.org/html/2411.17799), [SignSparK §4.2, Table 3](https://arxiv.org/html/2603.10446v4).

## Các số paper cần vượt — không gộp thành leaderboard chung

Thứ tự mỗi bộ bốn số: **PA-body / PA-hand / JPE-body / JPE-hand**, tất cả càng thấp càng tốt. Giữ nguyên đơn vị báo cáo của paper; không tự suy ra mm từ tên metric.

| Dataset | SOKE, dictionary, Table 1 | SignSparK GF, Table 3 | SignSparK SR, Table 3 |
|---|---|---|---|
| How2Sign | 6.82 / 2.35 / 7.75 / 10.08 | 7.26 / 2.72 / 6.30 / 11.43 | Không báo cáo |
| CSL-Daily | 6.24 / 1.71 / 7.38 / 9.68 | 7.26 / 2.00 / 6.27 / 10.63 | 4.87 / 1.37 / 4.89 / 6.56 |
| PHOENIX | 4.77 / 1.38 / 6.04 / 7.72 | 5.24 / 1.52 / 4.40 / 7.10 | 3.68 / 1.18 / 3.51 / 4.82 |

Nguồn: [SOKE Table 1](https://arxiv.org/html/2411.17799), [SignSparK Table 3](https://arxiv.org/html/2603.10446v4). Bản text có cấu trúc tại `third_party/proposal3_sources/{soke,signspark}/paper.html`, Table 1 và Table 3 tải trực tiếp từ arXiv; không lấy kết quả experiment SLP.

SOKE no-dictionary có PA-body/hand CSL = 7.58/2.17, PHOENIX = 6.16/1.85; không thay bằng hàng dictionary để quảng cáo GF. MaDiS Table 1 báo cáo JPE-body/hand CSL = 5.99/7.96, PHOENIX = 4.97/6.59, How2Sign = 6.59/9.45; MaDiS không dùng PA và thêm SiBLEU/SiCLIP. Các số này dùng để định vị đối thủ, chỉ được so sau khi khớp preprocessing/evaluator. [MaDiS](https://2000zrl.github.io/madis/).

**Vượt SOKE chưa đủ tuyên bố SOTA năm 2026.** Phải tái đánh giá MaDiS, SIGNER và các baseline mới đủ tương thích. BLANC tự fit pose và tái huấn luyện baseline nên các số trong bài đó không được nhập vào bảng này. [SIGNER](https://arxiv.org/abs/2506.07460), [BLANC §4 và App D](https://arxiv.org/html/2610.24109v1).

## Ba chế độ SignSparK cần phân biệt

- GF-T2P: không keyframe. Paper §4.2 dùng zero-keyframe inference.
- SR-T2P: keyframe từ sign retrieval; paper phân biệt với GF và không báo cáo How2Sign vì thiếu gloss. Phải xác minh nguồn gloss/retriever, không mặc định bằng dictionary lemma retrieval của SOKE.
- KF2P: keyframe quan sát từ motion đích. Table 4 đánh giá completion/control, không chứng minh text-only generation. BLEU-4 PHOENIX 11.70 với GT ceiling 14.77 thuộc track này, không so trực tiếp SOKE text-to-sign BLEU 11.87.

Nguồn: [SignSparK §4.2, Tables 3–4](https://arxiv.org/html/2603.10446v4). Baseline phát hành `sample.py` tạo `obs_x0=embed`; tắt `obs_mask` mới loại bỏ keyframe pose. Nó vẫn dùng shape, padding mask và `lengths` từ loader. **Tắt keyframe không tự chứng minh inference tự dự đoán độ dài.** Muốn text-only native-length phải thay độ dài bằng length predictor train-only, hoặc báo cáo rõ known-length track cho tất cả baseline. Xem [code audit](02_code_audit.md).

## Sai khác phát hành làm thay đổi kết luận

README local SignSparK cho biết pose đã reoptimize contact, checkpoint học lượng dữ liệu lớn hơn (15× theo README), official checkpoint loại PHOENIX, và BT retrain trên LMDB mới. Vì thế cần giữ hai bảng: paper-reported và matched-reproduction. Không lấy checkpoint mới, pose mới, BT mới rồi kết luận vượt paper cũ do kiến trúc. [Repository chính thức](https://github.com/JianHe0628/SignSparK).

SOKE paper có PA và JPE, nhưng `mGPT/metrics/t2m.py` tại checkout hiện tại dùng `align_idx=0` cho cả body và từng hand rồi lưu vào biến có tên `...PA...`. Hàm `dtw` trả tổng cost đường đi; `compute` lấy trung bình theo sequence, không thấy chia path length ở đường code này. Do đó **tên log không chứng minh metric PA**, và không được tự thay công thức bằng mean-per-frame. Chưa xác nhận pipeline postprocess nào dùng để tạo đủ bốn cột paper. Đây là gate tái lập phải giải quyết trước tuyên bố so số tuyệt đối; không phải lý do dừng thiết kế method.

## Hợp đồng đánh giá đề xuất

1. Ghim split IDs, pose SHA256, FPS, body model/version/shape, normalization, dictionary index và checkpoint; loại train/test overlap theo video gốc nếu phát hiện.
2. Ghim evaluator bằng code và fixture thực thi: identity, root translation, rigid rotation, một tay sai hình, lệch wrist, sequence bị lặp và khác chiều dài. Kiểm tra PA thật, root/wrist alignment, DTW normalization và unit bằng fixture có nghiệm tính được.
3. Lưu riêng native predicted length và GT length. Không crop/resample prediction theo GT ở inference. Nếu tái lập known-length, gắn nhãn rõ và dùng cùng length cho mọi baseline.
4. Một sample/câu là kết quả chính. Nhiều seed là lặp đánh giá, không chọn sample có DTW tốt nhất bằng GT. Chọn hyperparameter bằng dev; khóa trước test.
5. Báo cáo cả bốn cột hình học khi evaluator đã khớp, BT dùng cùng checkpoint/tokenization và GT ceiling, cùng metric wrist-to-body/hand-to-hand bổ sung. Không huấn luyện reward bằng chính BT test evaluator.
6. Báo mean, CI bootstrap paired theo câu (cluster video khi nhiều câu cùng video), ba training seeds nếu đủ compute; kiểm định trên chênh lệch paired và khai báo correction khi nhiều primary endpoints.
7. Chỉ gọi “SOTA trên track X, split Y, evaluator Z” sau matched reproduction và cải thiện metric đã đăng ký, không suy rộng thành SOTA mọi ngôn ngữ hoặc khả năng dùng thực tế.

Các mục 1–7 là thiết kế thí nghiệm được đề xuất, chưa được thực hiện.

## Cập nhật đối thủ 2026 từ primary source mới

[M3T v2 Table 1](https://arxiv.org/html/2603.23617v2) báo hàng w/ SLT, thứ tự PA-body/hand rồi JPE-body/hand: CSL = **3.97 / 1.51 / 4.63 / 9.28**, PHOENIX = **3.41 / 1.46 / 4.02 / 8.55**, How2Sign = **4.28 / 2.28 / 4.86 / 9.44**. Đây là paper-reported trên richer fitting/representation của M3T, không gán vào matched SOKE run. Nó tốt hơn một số cột nhưng không mọi cột của SignSparK/MaDiS; không có căn cứ lấy một con số làm mục tiêu “SOTA chung”.

Để viết paper mới, đăng ký metric/track và so Pareto giữa body, hands, semantics, latency trên matched data. Vượt SOKE cũ là milestone, không bằng chứng SOTA 2026. Các con số trong tài liệu này đều từ paper ngoài; không dùng experiment có sẵn của SLP.

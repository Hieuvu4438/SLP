## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate — source, data manifest, and selected CPU numerical checks
- Origin Date: 2026-11-04, Asia/Ho_Chi_Minh
- Verification Status: ANALYZED
- Version Label: how2sign_benchmark_audit_v1
- Scope: audit existing a25/a50 test run; no training, no regeneration of predictions, no full benchmark rerun; original outputs unchanged.
- Evidence: [evidence.json](evidence.json). The diagnostic script is preserved as [audit.py](audit.py).

## Kết luận

**Công thức và tiền xử lý DTW của lần chạy a25/a50 khớp implementation MaDiS đang phát hành, nhưng lần chạy hiện tại chưa tương đương hoàn toàn với protocol paper.** Có một lỗi xác định được khiến 8/2.308 mẫu hợp lệ bị mất, retrieval dùng thêm 281 clip train bị bộ lọc chính thức loại ra, và reranker dùng các câu input của cả tập test làm distractors. BLEU qua Uni-Sign không phải SiBLEU/SiCLIP của paper.

Điều có thể khẳng định từ artifact hiện tại: a50 có DTW thấp hơn MaDiS chạy lại trên **cùng 2.300 mẫu** trong evaluator tương thích MaDiS. Các số a50 6.2003/8.9588 thấp hơn số paper 6.59/9.45 về mặt số học. Chưa đủ cơ sở để gọi đây là kết quả khớp protocol paper hoàn toàn hay tuyên bố SOTA.

## 1. Source đã đối chiếu

| Repository | Commit |
|---|---|
| MaDiS | `071828ea1b94530dc7ff0a6c009c0a7c6d1f0784` |
| SOKE — Signs as Tokens | `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` |
| Snapshot chạy test của SLP | `778f71cbc61ff76b75cc48205caec2fbc3305e83` |

Hai checkout upstream không có tracked source modifications tại thời điểm audit; MaDiS có untracked assets. Tải lại bốn file từ raw GitHub tại đúng commit và xác nhận bytes giống file local: MaDiS `generation.py`, `sign_loader.py`; SOKE `dtw.py`, `load_data.py`. URL và SHA256 nằm trong evidence JSON. Không dùng việc code public tự nhận là “paper evaluator” làm bằng chứng duy nhất; đã đọc công thức và kiểm tra số học độc lập.

Nguồn paper:

- [MaDiS v2, §3.4, §4 và Table 1](https://arxiv.org/html/2601.19577v2#S4): How2Sign DTW-JPE body/hands **6.59 / 9.45**, SiBLEU body/hands **4.41 / 3.46**, SiCLIP R@1 **34.27**.
- [Signs as Tokens v3, §4 và Table 1](https://arxiv.org/html/2411.17799v3#S4): báo cáo cả DTW-PA-JPE lẫn DTW-JPE; không được so nhầm hai cột. SOKE How2Sign DTW-JPE **7.75 / 10.08**; DTW-PA-JPE **6.82 / 2.35**.

## 2. Dataset, split và nguồn fitting

Lần chạy thực sự dùng **How2Sign test**, không phải validation: runner truyền `--split test` hoặc `TEST.SPLIT=test`; tên config `.dev.yaml` và prefix lưu trữ `val__` không quyết định split thực tế. Manifest tên mẫu chứng minh các dự đoán nằm trong test annotations.

CSV raw SOKE và CSV repack test có cùng SHA256:

`5715d2f12210ba8601c0d98ad89c5cdd8a7eac5023d9b4b3b25bc04779f93a6b`

Fitting How2Sign được đóng gói từ archive SOKE/Neural-Sign-Actors, phù hợp nguồn fitting hai paper mô tả. Đã so sánh frame đầu/giữa/cuối của 3 clip giữa packed pickle và raw archive: cả **9 frame × 7 nhóm tham số** trùng chính xác. Đây là kiểm tra chọn mẫu, không phải hash toàn bộ corpus đối chiếu một manifest do tác giả xuất bản. Cả SOKE và MaDiS dùng cùng thư mục SMPL-X assets tại máy này, qua symlink.

### Bộ lọc chính thức

SOKE `dataset_t2m.py:58–71` và MaDiS `sign_dataset.py:148–168` cùng loại:

1. Câu có `END_REALIGNED - START_REALIGNED >= 30` giây.
2. Các ID nằm trong blacklist; hai blacklist trùng nhau.

MaDiS còn kiểm tra số frame sau subsample và token file khi nạp dữ liệu. Trên test local, không có mẫu thiếu pose/token sau bộ lọc annotation.

| Split | CSV / loader retrieval hiện tại | Sau bộ lọc annotation chính thức | Retrieval dùng thêm |
|---|---:|---:|---:|
| Train | 30.965 | 30.684 | 281 |
| Val | 1.724 | 1.717 | 7 |
| Test | 2.318 | 2.308 | 10 trước khi giao với dự đoán MaDiS |

30.684 + 1.717 + 2.308 = 34.709, phù hợp số token files của release. Không có trùng tên clip giữa train và test. Việc retrieval dùng 281 clip bổ sung là khác biệt về tập dữ liệu trong cùng split train, **không tự nó là test leakage**. Tuy nhiên, một so sánh kiểm soát dữ liệu cần thống nhất bộ lọc hoặc công bố khác biệt này.

## 3. Lỗi làm mất 8 mẫu test

| Giai đoạn | Số mẫu |
|---|---:|
| Test annotations gốc | 2.318 |
| Sau bộ lọc MaDiS/SOKE | **2.308** |
| MaDiS predictions đã sinh | **2.308**, danh sách trùng hoàn toàn bộ lọc |
| Candidate retrieval mỗi phương án | **2.308** |
| Keypoints sau projection loader | **2.300** |
| R1 / a25 / a50 đã materialize và chấm | **2.300** |

Tám clip bị mất đều có dấu `__` ngay trong YouTube ID:

```text
FzWvE__PamM_1-2-rgb_front
FzWvE__PamM_2-2-rgb_front
FzWvE__PamM_3-2-rgb_front
FzWvE__PamM_4-2-rgb_front
FzWvE__PamM_5-2-rgb_front
FzWvE__PamM_6-2-rgb_front
FzWvE__PamM_7-2-rgb_front
FzWvE__PamM_8-2-rgb_front
```

Nguyên nhân tại `smplx_to_cocowb.py:55`:

```python
return {k.split("__")[-1]: z[k] for k in z.files ...}
```

Ví dụ `val__FzWvE__PamM_1-2-rgb_front` bị biến thành `PamM_1-2-rgb_front`; vòng lặp đối chiếu tên đầy đủ nên bỏ clip. Cùng lỗi parsing có trong `blend_systems.py:10` và `score_predictions.py:39`. Chỉ sửa projection vẫn chưa đủ; cần sửa toàn bộ nơi xử lý prefix bằng cách bỏ đúng prefix một lần, bảo toàn phần còn lại, và assert tập tên bằng manifest yêu cầu.

Tỷ lệ mất là **0,3466%** của 2.308 mẫu. Không có bằng chứng đây là chọn mẫu theo điểm số, nhưng vẫn là missingness có hệ thống theo tên và tập con hiện tại không trùng đầy đủ release. Do có bước rerank chọn candidate, không thể chỉ sửa metadata trong JSON để khôi phục 8 kết quả.

## 4. Đối chiếu DTW-JPE

| Thành phần | SLP hiện tại | MaDiS release | Kết luận |
|---|---|---|---|
| Features | 133-D, bỏ root/lower-body và shape | Cùng layout | Khớp |
| Mean/std | SOKE `deps/stats` | MaDiS `hf_assets` | Trùng tensor và hash file |
| Downsample FPS | `np.linspace(0,n-1,int(24*n/fps),dtype=int)` khi fps > 24 | Cùng công thức | Khớp MaDiS |
| Reference length | Resample vào [40,400], còn lại center crop bội 4 | Cùng quy tắc | Khớp |
| Conversion | SMPL-X neutral, lower body zero, fixed shape 10-D | Cùng shape, slicing và assets | Khớp |
| Body joints | `[12,16,17,18,19,20,21,59,58,57,56,55]` | Cùng indices | Khớp |
| Hands | 21 joints/hand từ mesh regressor, có fingertip vertices | Cùng regressors | Khớp |
| Spatial alignment | Dịch pelvis với body, dịch wrist độc lập với mỗi hand | `align_idx=0` | Khớp; không Procrustes |
| Frame cost | Mean Euclidean joint error | `pairwise_jpe` | Khớp |
| Temporal aggregation | Tổng cost trên đường DTW tối ưu | Cùng | Khớp; không chia path length |
| Dataset aggregation | Mean trên clip; hands = mean hai tay | Cùng | Khớp trên cùng manifest |

**Bằng chứng thực thi:**

- 5 trường hợp synthetic, bao gồm độ dài không bằng nhau và 1 frame: DTW của helper SLP trùng MaDiS, max absolute difference **0**.
- Cùng synthetic inputs so với SOKE NumPy DTW: sai khác lớn nhất **2,742e-6** trên accumulated cost, phù hợp sai số float32 khác thứ tự phép tính.
- 3 clip a50 thực tế: nạp GT bằng loader MaDiS, đổi SMPL-X bằng `H2SDataModule.feats2joints`, chấm bằng chính `GenerationMetrics.update`. Tiền xử lý GT, joints và DTW body/hands đều trùng script SLP: max absolute difference **0**.

| Clip kiểm tra | DTW body — SLP = official | DTW hands — SLP = official |
|---|---:|---:|
| `-fZc293MpJk_2-1-rgb_front` | 7.4953867923 | 13.4546283744 |
| `G42xKICVj9U_6-10-rgb_front` | 4.2262264621 | 6.3128109071 |
| `g3ushtMfLiY_9-3-rgb_front` | 6.6380160525 | 10.6092382139 |

Đây là kiểm tra tương đương số học trên các trường hợp đã chạy, không phải chạy lại đủ 2.308 mẫu qua Lightning evaluator.

### Điểm khác giữa source SOKE và MaDiS

SOKE `load_data.py:24–27` dùng helper `sample`, với chỉ số `floor(i*n/count)` để xuống 24 fps. MaDiS `sign_loader.py:78–81` dùng `linspace(0,n-1,count,dtype=int)`. Hai cách lấy cùng số frame nhưng có thể lấy frame khác nhau. Ví dụ clip đầu trong bảng trên khác **106 chỉ số frame** trước crop. SLP khớp **MaDiS** ở điểm này; không được khẳng định toàn bộ tiền xử lý SOKE và MaDiS giống hệt nhau chỉ vì cùng fitting.

SOKE metric state còn giữ tên `DTW_MPJPE_PA_*`, nhưng tại `t2m.py:146–170` đang gọi `align_idx=0`, tức DTW-JPE không PA. Paper MaDiS §4 cũng yêu cầu không PA. Audit xác nhận lần chạy SLP so đúng loại DTW với cột MaDiS 6.59/9.45.

## 5. Điều kiện inference và dữ liệu ngoài generator

- MaDiS generation thực tế nhận GT token lengths trong chữ ký hàm, nhưng `tasks=None` tạo prompt chỉ có caption; `generate_conditional` không đưa length vào prompt mặc định. `generate` dùng max length 100 và EOS để quyết định chiều dài. Retrieval retime theo **chiều dài MaDiS đã sinh**, không theo GT length. Không thấy rò rỉ GT length vào generation ở đường code đã kiểm tra.
- Candidate motions lấy từ split train, không từ test GT motions. Text input của chính câu test là input hợp lệ cho text-to-sign.
- Tuy nhiên, `unisign_rerank.py:50–67` tạo distractor pool từ **toàn bộ input texts của tập test được nạp**. Đây là điều kiện transductive, không phải độc lập từng câu như generator MaDiS. Không đồng nghĩa dùng nhãn pose test, nhưng cần công bố hoặc thay bằng pool train cố định và đóng băng từ validation để so theo điều kiện inference giống nhau.
- Uni-Sign reranker là một model SLT How2Sign bổ sung, còn evaluator BLEU là Uni-Sign OpenASL. Chi phí, data provenance và tác dụng của reranker phải được báo cáo; phép trộn a50 không phải một MaDiS checkpoint train theo kiến trúc mới.

## 6. Metrics ngữ nghĩa không tương đương

Paper MaDiS yêu cầu SiBLEU-4 trong sign-token space và SiCLIP R@1. Lần chạy này dùng back-translation của Uni-Sign OpenASL trên keypoints chiếu từ SMPL-X rồi tính text BLEU. **0.2708 BLEU-4 của a50 không so trực tiếp với 4.41/3.46 SiBLEU hoặc 34.27 SiCLIP của paper.** SOKE có BT BLEU-4, nhưng cùng tên BLEU không đủ: cần cùng BT model/checkpoint, projection, preprocessing và decoding; các điều kiện đó chưa được tái lập theo SOKE paper.

Lưu ý thêm từ source MaDiS public: `generation.py:16–36` dùng clipped unigram overlap + brevity penalty, không tạo n-gram 2/3/4. Paper gọi metric SiBLEU-4. Đây là mismatch paper/release cần ghi rõ khi triển khai đánh giá; không có căn cứ từ audit này để khẳng định tác giả đã tính số paper bằng công thức release đó. Không được dùng token của prior MaDiS để đại diện output a50: cần re-encode chính chuyển động cuối bằng cùng frozen tokenizer.

## 7. Diễn giải các con số hiện có

| Hệ thống | Manifest | DTW body | DTW hands | BT BLEU-4 Uni-Sign OpenASL |
|---|---|---:|---:|---:|
| MaDiS paper | Paper test protocol; không có per-ID manifest paper trong artifact hiện tại | 6.59 | 9.45 | Không báo metric này |
| MaDiS chạy lại | 2.300 mẫu chung | 6.5817 | 9.4240 | 0.1357 |
| a25 | 2.300 mẫu chung | 6.5117 | 9.0738 | 0.8127 |
| a50 | 2.300 mẫu chung | 6.2003 | 8.9588 | 0.2708 |

Baseline gần paper hỗ trợ tính hợp lý của pipeline, nhưng không chứng minh tự động rằng mọi chi tiết protocol đều giống paper. Cải thiện DTW không tự chứng minh ký hiệu đúng ngữ nghĩa, và hai metric DTW không đủ để kết luận SOTA toàn diện. Chưa có paired bootstrap DTW trong artifact này; không lấy CI/p của BT BLEU để suy ra ý nghĩa thống kê của DTW.

## 8. Các điều kiện cần hoàn tất trước kết luận theo paper

1. Sửa parser prefix ở toàn bộ đường đọc NPZ; khôi phục generation/rerank/blend cho 8 mẫu và assert chính xác manifest 2.308 mẫu, không giao tập im lặng.
2. Nếu yêu cầu matched-data comparison: retrieval chỉ dùng 30.684 clip train sau bộ lọc chính thức. Nếu giữ 281 clip bổ sung, phải ghi rõ chế độ dữ liệu và có ablation tương ứng.
3. Nếu yêu cầu inference độc lập từng câu: dùng distractor pool train cố định, lựa chọn cấu hình trên val; giữ kết quả transductive hiện tại như một setting riêng. Việc thay pool có thể thay candidate được chọn, nên cần rerank lại thay vì chỉ sửa bảng số.
4. Chấm DTW trên toàn manifest bằng evaluator official; đối chiếu score per-clip với helper trước aggregation.
5. Re-encode output cuối để đo token metrics; đánh giá SiCLIP bằng checkpoint release và batch/candidate protocol phù hợp, phân biệt SiBLEU-release với định nghĩa SiBLEU-4 paper nếu chưa được giải thích.
6. Lưu đầy đủ hashes/assets, seed, commands và code dependency closure. Snapshot test hiện tại vẫn import `scripts/madis_oracle_eval.py` và `decode_sign_tokens.py` từ main workspace; hai file có mtime trước run nhưng không nằm trong snapshot dependency closure. Điều này không phủ nhận numerical checks, nhưng hạn chế tính tự chứa của artifact tái lập.

Không cần train lại MaDiS để sửa lỗi tên mẫu hoặc bổ sung evaluator. Thay training retrieval pool và distractor pool là thay setting inference, cần giữ version rõ ràng và tách khỏi số đã báo.

## 9. Fallacy scan và giới hạn xác minh

Coverage: **11/11 loại đã xét** theo workflow; chỉ khẳng định kiểm tra đủ loại, không khẳng định có đủ dữ liệu để loại trừ mọi loại bias.

| Loại | Kết quả trong audit này |
|---|---|
| Simpson's paradox | Chưa có phân tích theo signer/video; chưa loại trừ khác biệt theo nhóm. |
| Ecological fallacy | Không suy mean toàn tập thành thắng trên mọi câu hoặc đúng mọi ký hiệu. |
| Berkson / selection bias | Có mất mẫu theo tên và train filtering khác; ảnh hưởng định lượng chưa tái tính. |
| Collider bias | Không fitting mô hình hồi quy có covariates; không áp dụng trực tiếp. |
| Base-rate neglect | Không phải bài toán diagnostic sensitivity/specificity; không áp dụng trực tiếp. |
| Regression to the mean | Không suy từ một lần chạy thành bảo đảm ổn định nhiều seed. |
| Survivorship bias | 8 mẫu mất đã được liệt kê; không gọi tập còn lại là full official test. |
| Look-elsewhere effect | Có nhiều biến thể nghiên cứu; không suy cải thiện vài metric thành thắng toàn diện. |
| Garden of forking paths | Freeze a25/a50 trước test được log; vẫn có khác biệt implementation cần công bố. |
| Correlation/causal overreach | Không suy DTW giảm là semantics tốt hơn; không quy cải thiện riêng cho renderer chưa đo. |
| Reverse causality | Không phải suy luận nhân quả quan sát theo thời gian; không áp dụng trực tiếp. |

Reproducibility verdict: **CANNOT_VERIFY full paper-equivalent benchmark** từ artifact hiện tại. Đã xác minh source hashes, manifests và numerical equivalence trong phạm vi nêu trên; chưa full rerun, chưa sửa dữ liệu hay predictions và chưa có đầy đủ paper semantic metrics.

# S10 — Các nguồn dùng để giới hạn claim về tay và non-manuals

## Non-manual signals trong KSL: có condition riêng chưa phải novelty

**M:** đọc §3.3 và §§4.1–4.4 của [Lee & Park, Sensors 2026](https://www.mdpi.com/1424-8220/26/13/4245), qua [bản full text PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13363759/). Chưa đọc sâu toàn bộ results/ablation; không dùng để xếp hạng benchmark.

Paper đã dự đoán non-manual conditions từ text và dùng projection/FiLM cho pose decoder. Nhãn bốn chiều được gộp từ metadata thành sample-level multi-hot; chúng không phải temporal scopes chi tiết. Do đó không thể nhận novelty từ việc thêm một nhánh dự đoán non-manual presence, đồng thời cũng chưa có căn cứ coi mô hình này đã giải quyết overlapping interval planning. Tài liệu lưu mới tại `sources/ksl_nms/`; MDPI trả 403 qua HTTP request, PMC retrieval thành công. Không che việc nguồn truy cập khác nhau.

## Ngôn ngữ học: khuôn mặt có cấu trúc thời gian

**M:** đọc abstract và introduction của [Benitez-Quiroz et al., PLOS ONE 2014](https://doi.org/10.1371/journal.pone.0086268), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3916328/). Paper phân tích ASL non-manuals và quan hệ thời gian giữa các facial components cho một số loại câu. Nguồn này hỗ trợ yêu cầu kiểm tra mặt và timing; không chứng minh mọi ngôn ngữ ký hiệu có cùng quy tắc, cũng không xác nhận một vector expression SMPL-X cụ thể giữ đủ grammar. Không dùng các tỷ lệ trong abstract làm benchmark cho CSL/DGS.

## HandMDM và bài phonology: khác task với spoken-text translation

**V + introduction:** [Text-Driven 3D Hand Motion Generation from Sign Language Data](https://arxiv.org/abs/2508.15902) dùng mô tả đặc điểm chuyển động bàn tay. Phần đã đọc giải thích BOBSL3DT xây từ BOBSL, sign attributes và LLM descriptions; chưa đọc sâu method/results nên không dùng như bằng chứng chọn backbone. Text mô tả handshape có nhiều thông tin hơn câu spoken-language đầu vào của SLG.

**V:** [Toward Phonology-Guided Sign Language Motion Generation, CVPR Workshops 2026](https://openaccess.thecvf.com/content/CVPR2026W/GenSign/html/Hong_Toward_Phonology-Guided_Sign_Language_Motion_Generation_A_Diffusion_Baseline_and_CVPRW_2026_paper.html) đã khảo sát conditioning bằng phonological attributes. Chỉ xác minh official abstract; chưa dùng kết quả như evidence quyết định. Một candidate dùng attribute conditioning phải đọc sâu nguồn này trước khi nhận novelty.

## Transition completion và temporal planning

**M:** đọc chọn lọc §3 và các bảng setup của [Discrete-to-Continuous Transition](https://arxiv.org/abs/2411.16810). Bài sinh các đoạn thiếu từ observed sign motions; đây là completion, không phải text-only SLG. Không sử dụng code hoặc các thử nghiệm cùng tên trong repo nội bộ.

**V:** [TPPMG, IJCAI 2026](https://www.ijcai.org/proceedings/2026/197) đã có text→stage-duration script và progressive diffusion. Chưa đọc sâu, vì vậy chỉ dùng để bác bỏ claim quá rộng “planning duration trước khi sinh motion chưa từng có”; không dùng để chứng minh chi tiết architecture hoặc hiệu quả candidate.

Các nguồn M/V ở đây chỉ thiết lập những điều trực tiếp thấy trong phần đã đọc. Chúng không thay cho deep notes của các paper quyết định phương án chính.

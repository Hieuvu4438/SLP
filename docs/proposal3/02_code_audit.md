# Audit source code phục vụ proposal

Đọc tĩnh ngày 06/11/2026, chưa chạy inference/training. **Các tệp bên dưới đã được so khớp lại với fresh official checkouts từ bên ngoài, 13/13 hash khớp.** Dùng `third_party/proposal3_sources/code/` và `research/external_code_manifest.json` làm nguồn hiện tại; không dùng experiment của SLP. Checkout local là bằng chứng cho implementation đang có, không suy ra mọi chi tiết đều trùng bản paper. Các vị trí dưới đây có thể mở bằng tên file và symbol.

| Repo | HEAD local | Tệp/symbol đọc | Kết luận và hệ quả |
|---|---|---|---|
| SOKE | `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00` | `mGPT/models/mgpt.py: val_t2m_forward` (khoảng 163–295) | Decode body và hai hand riêng; upsample ×4; lấy max length các stream và pad replicate. Proposal có thể lấy draft sau decode; phải giữ length dự đoán. |
| SOKE | như trên | `mGPT/archs/mgpt_lm.py:generate_conditional` (362–400) | `with_len=False` đặt lengths thành zero trong text-to-motion template. Việc caller truyền GT length chưa đủ kết luận GT length vào generation; phải trace callee. |
| SOKE | như trên | `mGPT/metrics/t2m.py:TM2TMetrics.update/compute` (88–174); `metrics/dtw.py:dtw/l2_dist_align` | Tên PA không khớp alignment đang gọi; DTW là tổng cost; hand mesh regressor rồi wrist translation alignment. Hand score có thể không nhạy với sai vị trí tay tương đối thân. Cần thêm relational diagnostic. |
| SignSparK | `916402d0ecd6322fb4a9e5f513229a94a57bf4b6` | `sample.py` (125–166), `signspark/keyframes.py:get_keyframes_mask` | Sampling mặc định có pose anchor từ loader; `keyframes_mask_all` bỏ observed-pose mask nhưng còn length/mask của loader. Hai tay được tách thành batch items với cùng text. Đây không phải joint bilateral interaction module. |
| SignSparK | như trên | `signspark/rflow.py:training_losses_poses`, `_interp_xt_and_mask`, `sflow_sampler` | Code dùng target vector `x_1-keyframed_x0`, trong khi paper Eq.3 viết residual `x_1-C_t`; không được chép cả hai thành cùng một objective. Code còn dựng endpoint bằng `x_t+(1-t)v`, clamp anchor mỗi bước ODE. |
| MaDiS | `071828ea1b94530dc7ff0a6c009c0a7c6d1f0784` | `mGPT/archs/masked_diffusion.py`: UTC constants, `_bidirectional_attention_mask`, mask sampler, MoP fusion, generation | Có bidirectional mask, pivot .75/.5, gate softmax theo part, fixed 25 steps. Không gọi “part gating + coarse-to-fine” là đóng góp mới của proposal. |
| MaDiS | như trên | `mGPT/archs/madis.py`, `connectors.py` | Wrapper map token IDs và pretrained codebooks; thay tokenizer phá contract ID/embedding. Giữ tokenizer khi thử refiner giúp attribution rõ hơn. |
| MoMask | `94a6636c9c463b7a9414c3401a6f1b67e6c51824` | `models/mask_transformer/transformer.py:generate` và residual generate (khoảng 893–963); `models/vq/residual_vq.py` | Base masked prediction và residual quantizer có vai trò khác nhau; generation nhận `m_lens`. Reuse masked sampler không tự giải quyết length và không biến motion residual thành semantic error correction. |

Đường dẫn tính từ `third_party/<repo>/`. README các repo cũng đã được đối chiếu. MaDiS có thư mục assets untracked; chưa sửa source. Không chỉnh sửa bất kỳ third-party code nào trong lượt khảo sát này.

## Một chi tiết toán học đáng kiểm tra ở SignSparK

Với path `x_t=(1-t)x_0+t x_1`, đạo hàm đúng là `x_1-x_0` (ở tọa độ không bị anchor). `x_1-x_t=(1-t)(x_1-x_0)` là residual endpoint, cần chia `1-t` nếu dùng nó như velocity theo cùng time parameterization. Code local dùng hiệu hai endpoint nên phù hợp path tuyến tính; paper Eq.3 viết khác. Chưa thể kết luận lỗi thực nghiệm từ bất nhất ký hiệu này. Proposal sẽ dùng objective endpoint-consistent riêng và kiểm tra nghiệm tuyến tính trước training.

## Những phần chưa đủ bằng chứng

Chưa có runtime proof cho metric PA/JPE đầy đủ, adapter 6D↔axis-angle, left-hand convention, và BT paper cũ. Chưa audit code đầy đủ của mọi paper trong hai danh mục. `research/input_inventory.csv` giữ toàn bộ URL đã trích từ bảng để tiếp tục, không tự gán trạng thái “đã đọc paper/code”. Không lấy báo cáo proposal2 làm bằng chứng thay cho việc mở source hiện tại.

# Trạng thái bàn giao — 05/11/2026

Bộ literature review và proposal đã qua audit tám yêu cầu tại [10](10_completion_audit.md). Không có model training/reproduction hoặc human study. Những empirical hypotheses còn mở được giữ rõ trong method/experimental plan; chúng không phải kết quả đã đạt.

## No-progress audit và kết quả lượt cuối

Lượt goal trước là **progress**: tạo SLG/benchmark/candidates/main method/experiments/critical-review artifacts. Lượt cuối tiếp tục progress: đọc lại attachment, biên tập M01–M17 và S03/S04, motion synthesis/cross-paper inventory/search log; kiểm tra primary tables/code; sửa những phần method chưa đủ cụ thể. Không có blocker hoặc verified wait còn phải xử lý.

## Nội dung bàn giao

- 03 motion synthesis trước SLG; 04 deep SLG; 05 benchmark contract cho How2Sign/CSL-Daily/PHOENIX-2014T.
- 06 ba candidates ngang cấu trúc: semantic contrast supervision, temporal-target distribution, selective retrieval.
- 07 chọn B với q trên pseudo partitions, content/duration/EOS prior và joint diffusion. Full train/inference specification, no target-reference condition ở primary inference.
- 08 baseline/ablation/data/compute/evaluator/human design và falsification/fallback; 09 self-review; 10 requirement audit; 11 claim–source audit.
- Ledger có 53 acquisitions, 32 scoped deep readings. Đây là registry description, không bằng chứng chất lượng thay nội dung; M/V/unread giữ nguyên scope.

## Những giới hạn phải giữ nếu tiếp tục thực nghiệm

q là heuristic pseudo-target distribution, chưa calibrated linguistic uncertainty. Expected NLL là structured plan augmentation, không exact marginal likelihood. BLANC đã có motion-derived partition/confidence/duration/EOS/counterfactual event loss; semi-Markov inference cũng không mới. Candidate chỉ đáng giữ nếu hơn matched point/confidence/jitter/uniform/K/compute controls ở **fully predicted** generation.

SOKE-compatible 256D learning representation vẫn có limits với gaze/facial grammar; head/neck rotations còn trong active body map. Proposed input policy không nhận target gloss/length/keyframes tại inference. T0 phải retrain codecs/teachers trên target train; extra multilingual data/annotations/dictionaries có tracks riêng.

Mục tiêu ba datasets và nhiều trục semantics/articulation/motion/comprehension được giữ. Không dùng oracle-plan gains, một metric thuận lợi hoặc một dataset để gọi đã hoàn thành mục tiêu SOTA.

## Verification thực chạy

- SiBLEU synthetic diagnostic: official functions trên CPU, reversed unigram sequence vẫn 100; không model/data benchmark.
- Toy algebra: partition normalizer enumeration/DP khớp, overlap cells hợp lệ; expected loss khác marginal likelihood. Không xác nhận method efficacy.
- Rechecked primary benchmark table headers/rows và public code paths, theo 11.
- Source acquisition integrity checks: 102 artifacts, gồm selected published/supplementary PDFs, khớp hash; kiểm tra này không chứng nhận mọi PDF page anchors hoặc source truth.
- 19 pinned public-code links tồn tại trong git objects; local links/fences kiểm tra ở evidence JSON. Các checks không thay content audit.

## Ranh giới và tooling

Không đọc proposal1/archive/methods/configs/outputs/results/logs nội bộ. Fresh public code: MoLingo 52e3b4c…, SOKE 5cbc55d…, MaDiS 071828e…, SignSparK 916402d…; notes có full commits và scopes. Không có jobs do lượt cuối tạo còn live; các acquisition sessions trước đều terminal, không restart từ state file.

`acquire_arxiv.py` chỉ acquisition; `make_readable.py` tạo extraction; `build_ledger.py` dùng explicit PROGRESS. Đã freeze source versions/hash trong registry. PDF locators dùng sections/tables; HumanTOMATO preflight UNAVAILABLE không bị nâng PASS. Không giả human-read/independent review.

Nếu người dùng yêu cầu triển khai, bắt đầu từ 05/07/08 và cập nhật latest baselines trước experiment. Không suy toàn bộ hypotheses đã đúng chỉ vì research goal đã bàn giao.

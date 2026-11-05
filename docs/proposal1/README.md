# Proposal 1 — FLARE-Sign: flow-refined, lexicon-augmented, reward-aligned sign language production

Status: **draft v2 (2026-11-03)**. It is based on:
- a literature survey (≈60 SLP and ≈70 text-to-motion papers);
- a code audit of SOKE / MaDiS / SignSparK;
- a full reproduction of the released MaDiS on the dev splits of all three benchmarks;
- 19 diagnostic findings (F1–F19) in [03](03_diagnostics.md).

No test-split data was used. Method folder: `methods/d2c_sign/`.

## Tóm tắt (Vietnamese summary)

### Bối cảnh

SOTA SLP 2026 chia hai nhánh:
- **LM trên sign token:** SOKE (ICCV'25) → MaDiS (masked-diffusion Qwen3-0.6B).
- **Sinh liên tục:** SignSparK (flow matching, cần gloss + từ điển keyframe), cùng M3T / SeRV / BLANC / GARD.

Chưa có công trình nào dùng RL / preference optimisation cho sign *production*, và chưa có protocol đánh giá có hiệu chuẩn.

### Phát hiện chính (đo trên dev; chi tiết ở [03](03_diagnostics.md))

**A. Protocol đánh giá hiện tại không đáng tin**

1. **MaDiS tái lập đúng paper trên cả 3 benchmark** (Phoenix 5.19 / 6.94, CSL 5.78 / 7.65, How2Sign 6.61 / 9.61 DTW-JPE body / hands). Nhưng DTW-JPE *cộng dồn* không phân biệt được với output tầm thường:
   - **How2Sign:** MaDiS thua cả mean pose tĩnh về body (6.61 vs 5.99) và thua GT đảo thời gian ở mọi biến thể DTW.
   - **Phoenix:** MaDiS thua GT đảo thời gian về body (5.19 vs 5.11).
   - **CSL:** MaDiS thắng nhưng hoà về DTW theo từng bước.

   (F1, F7, F11, F13)
2. **Bàn tay của MaDiS chỉ chuyển động bằng 35–54% tốc độ GT** (regression-to-mean). Tỉ lệ này **thừa hưởng từ tokenizer** (cũng 0.35), không phải từ LM. (F7, F9)
3. **Lỗi trong asset phát hành của MaDiS:** token tham chiếu của split val/dev chỉ trùng 0.6–7% với pose GT dùng cho DTW (train trùng 100%). SiBLEU trên dev vì vậy không có nghĩa. Test chưa kiểm tra; cần báo tác giả. (F8)
4. **SiBLEU trong code là unigram** (không quan tâm thứ tự). **SiCLIP xếp hạng trong batch 45.** (§5)
5. **Back-translation (BT)** phân biệt mean pose tốt (0.4 BLEU), nhưng có hai vấn đề:
   - ít nhạy thứ tự: GT đảo thời gian giữ 80% trần;
   - bị **bias do evaluator và generator train cùng train split:** MaDiS + refiner đạt R@1 BT-retrieval *vượt* GT, trong khi decode token GT hoàn hảo thấp hơn nhiều.

   Đã loại trừ giả thuyết "GT nhiễu" bằng thí nghiệm làm mượt GT. **Bằng chứng trực tiếp:** refiner *hồi quy* (output "trung bình" nhất) đạt BT 12.50 BLEU / 23.3 R@1, **vượt GT** (11.49 / 20.6). (F14, F16–F20)

   ⇒ **Kết luận:** với DTW, SiBLEU và BT cùng-split, protocol hiện tại *không thể* chứng nhận một cải tiến SOTA. Đóng góp đầu tiên phải là C0 (evaluator dữ liệu rời + người đánh giá).
6. **Evaluator độc lập đầu tiên (C0, đã xây xong).** Dùng Uni-Sign train trên **OpenASL** (chưa từng thấy How2Sign), chạy trên SMPL-X chiếu thành keypoint 2-D. Kết quả BLEU-4 trên How2Sign val (n = 1,717):

   | | BLEU-4 |
   |---|---|
   | GT (trần) | 2.50 |
   | mean pose | 0.04 |
   | GT đảo thời gian | 0.42 |
   | **MaDiS** | **0.24** (≈10% trần) |

   Metric này không bão hoà, nhạy thứ tự, và không bị fit. SOTA hiện tại còn khoảng trống rất lớn trên How2Sign, nên đây là thước đo chính cho mục tiêu SOTA. (F21)

   Evaluator độc lập thứ hai (Uni-Sign train trên How2Sign video thật) cho kết quả tương tự: MaDiS đạt 0.79 BLEU-4 (≈13% trần 6.14), gần mean pose (0.63). (F22)

   **CSL-Daily** (Uni-Sign CSL, có trần video thật): SMPL-X chiếu giữ 88% chất lượng video thật (20.92 vs 23.82 BLEU-4). MaDiS chỉ đạt 3.35, tức **16% trần**, ngang GT đảo thời gian. (F23)

   ⇒ **Mục tiêu SOTA cụ thể, kiểm chứng được:** nâng BLEU-4 của evaluator ngoài trên How2Sign từ ≈10–13% trần (MaDiS) lên gần trần.

**B. Giới hạn kỹ thuật của các phương pháp token**

6. Tokenizer dùng chung (96 / 192 code) có **sàn lượng tử hoá** 3.12 / 4.80. **≈97% sàn "hands" đến từ stream body** (hướng cổ tay / cẳng tay), không phải codebook ngón tay (0.89). Nên tăng codebook bàn tay (hướng của M3T / SeRV) là không đủ. (F3, F5)
7. **Refiner liên tục C1 phá sàn này:** với GT token, 3.12 / 4.80 → **2.75 / 3.96** (−12% / −17%). (F9)
8. **Trên output MaDiS:**
   - DTW gần như không đổi, vì lỗi chủ yếu là *planning* (≈2.1 trên sàn).
   - BT tăng **+1.5 BLEU-4 dưới hai evaluator kiến trúc khác nhau** (B1 mBART, B2 Qwen).
   - Refiner chỉ tăng khả năng nhận diện câu (R@1) khi nội dung token đúng.

   (F10, F14, F15, F18)

**C. Thử nghiệm phương pháp (pilot, đánh giá bằng evaluator ngoài)**

9. **Rerank best-of-4 mẫu MaDiS** (reward cùng-split yếu, hoặc reward Uni-Sign-H2S mạnh) **không cải thiện có ý nghĩa thống kê** (bootstrap: mọi CI 95% chứa 0). Mẫu MaDiS "sai đều nhau" ở mức câu. (F24, F25, F27)
10. **Retrieval câu gần nhất (copy clip train, không sinh gì) vượt MaDiS +58% BLEU-4** dưới evaluator dữ liệu rời OpenASL. Trong khi đó DTW xếp hạng ngược lại. (F26)
11. **Ghép retrieval trong không gian token (re-mask + MaDiS khử nhiễu) thất bại**, còn tệ hơn MaDiS: tokenizer làm mất ngữ nghĩa. ⇒ Retrieval phải ở **không gian chuyển động liên tục**. (F28)

12. ★ **FLARE-R0 (retrieval top-8 + rerank bằng Uni-Sign-H2S, không train) vượt SOTA đã công bố có ý nghĩa thống kê** dưới evaluator dữ liệu rời (Uni-Sign-OpenASL), How2Sign val (n = 1,717).

    | | BLEU-4 | BLEU-1 |
    |---|---|---|
    | GT (trần) | 2.50 | 19.50 |
    | **FLARE-R0** | **0.90** (36% trần) | **14.24** |
    | MaDiS | 0.24 (10% trần) | 8.04 |

    Chênh lệch BLEU-4 so với MaDiS: **+0.65, CI 95% [0.44, 0.87], p < 0.001** (gấp 3.7×). Hạn chế: chỉ ghép được câu đã có trong train. **DTW tệ hơn** (8.71 / 11.71 vs MaDiS 6.61 / 9.61), vì độ dài và nhịp lấy từ câu khác. Lỗi bàn tay theo từng bước ngang MaDiS, và tốc độ bàn tay tự nhiên (1.01 × GT so với 0.43). Metric ngữ nghĩa đáng tin và DTW xếp hạng **ngược nhau hoàn toàn**. (F29, F30)

    **Sửa nhịp đơn giản**: resample về độ dài mà MaDiS dự đoán (không dùng độ dài GT). Kết quả:
    - DTW còn 7.32 / 9.93, gần MaDiS (6.61 / 9.61);
    - BLEU-4 giữ ở 0.82, giảm không đáng kể;
    - vẫn **+0.58 so với MaDiS, CI [0.37, 0.80], p < 0.001**. (F31)

13. **Kiểm tra trên CSL lộ ra artefact của evaluator.** Uni-Sign-CSL đã train trên chính các clip train được retrieve, nên nó "đọc lại" câu train đã nhớ (BLEU-4 97 so với câu train). Điểm 19.07 của retrieval trên CSL chỉ bằng việc copy câu text (19.59), tức **không có giá trị**. CSL hiện chưa có evaluator độc lập cho hệ retrieval. Ngược lại, OpenASL trên How2Sign *đọc* thật, không nhớ (BLEU-4 ≈ 3 so với câu train, ngang mức đọc GT). Vì vậy **kết quả vượt SOTA chỉ được khẳng định trên How2Sign**. (F32, F33)

24. **Reward mạnh (Uni-Sign-CSL) trên CSL không giúp thêm** (−0.03, n.s.). Điểm gần nhất là α = 0.4: DTW chỉ kém MaDiS 0.08 / 0.04, ngữ nghĩa +0.10 (p = 0.06), tức chưa trội. Giới hạn là evaluator CSL độc lập quá yếu, và retrieval trên CSL có DTW kém hơn về bản chất. (F47)
23. **Quét α trên CSL và Phoenix:** α ≥ 0.5 vượt MaDiS về DTW, nhưng trên CSL thì lợi thế ngữ nghĩa biến mất ở đúng vùng đó. **Không có điểm vận hành nào trội hơn MaDiS trên CSL.** Khác biệt so với How2Sign nằm ở độ mạnh của reward. (F46)
22. **Renderer R7 v0 thất bại (sụp về trung bình).** Train trên cặp "ghép clip của câu khác → GT của câu này", vốn không nhất quán nội dung. Model bỏ qua input: tay gần đứng yên (0.13×), BLEU-4 chỉ 0.06, nhưng DTW body lại *tốt nhất* (5.96), thêm một minh chứng DTW thưởng việc lấy trung bình. ⇒ R7 phải train trên cặp nhất quán nội dung (đoạn của chính câu đó, cắt bằng aligner) và có bảo vệ ngữ nghĩa. FLARE-R1b vẫn là hệ tốt nhất. (F45)
21. **Aligner học được (R5)** đạt R@1 27.6% (pool 100, ngẫu nhiên 1%). Cắt đoạn bằng aligner giảm mất mát ngữ nghĩa còn một nửa so với cắt theo vị trí, nhưng không cải thiện DTW. (F44)
20. ★★★ **Xác nhận trên tập TEST How2Sign** (do user yêu cầu, đã ghi trong `results/test_access_log.jsonl`; cấu hình cố định từ val; n = 2,300). FLARE-R1b tốt hơn MaDiS trên mọi metric:

    | | FLARE-R1b | MaDiS |
    |---|---|---|
    | DTW body | **6.51** | 6.58 |
    | DTW tay | **9.07** | 9.42 |
    | per-step body / tay (cm) | **4.64 / 6.14** | 4.69 / 6.50 |
    | BLEU-1 (OpenASL) | **14.06** | 8.42 |
    | BLEU-4 (OpenASL) | **0.81** | 0.14 |

    Chênh lệch BLEU-4: **+0.68, CI [0.51, 0.85], p < 0.001**. (F43)
19. **Thử lại trên CSL và Phoenix.**
    - **CSL**, evaluator rời CSL-News: R1 vượt MaDiS **+0.49 BLEU-4 (3.8×), p < 0.001**; R1b +0.19, p = 0.002. Nhưng DTW vẫn kém MaDiS.
    - **Phoenix**, chỉ có BT cùng-split nên không đủ tin cậy: DTW kém MaDiS.
    - ⇒ Lợi thế ngữ nghĩa lặp lại trên 2 benchmark có evaluator độc lập. Ưu thế cả DTW (Pareto) hiện chỉ có trên How2Sign, nơi reward mạnh. (F41, F42)
18. ★★ **FLARE-R1b = 75% FLARE-R1 + 25% MaDiS (trộn đều mọi stream): tốt hơn SOTA đã công bố (MaDiS) trên MỌI metric** (How2Sign val, n = 1,717).

    | | FLARE-R1b | MaDiS |
    |---|---|---|
    | DTW body | **6.45** | 6.61 |
    | DTW tay | **9.09** | 9.61 |
    | per-step body / tay (cm) | **4.57 / 6.10** | 4.70 / 6.62 |
    | BLEU-1 (OpenASL, dữ liệu rời) | **13.79** | 8.04 |
    | BLEU-4 (OpenASL, dữ liệu rời) | **0.72** | 0.24 |

    Chênh lệch BLEU-4: +0.48, CI [0.27, 0.69], p < 0.001. α là núm Pareto giữa ngữ nghĩa và DTW. (F40)
17. ★ **FLARE-R1: ghép 2 clip thật (chia câu làm đôi, top-4 mỗi nửa, rerank bằng Uni-Sign-H2S, chỉnh nhịp).**
    - BLEU-4 1.08, tức **43% trần**, gấp 4.5× MaDiS: **+0.84, CI [0.65, 1.04], p < 0.001**.
    - Vượt FLARE-R0: +0.27, p = 0.008.
    - BLEU-1 đạt 80% của GT.
    - DTW tay 9.56, **tốt hơn MaDiS** (9.61); body 7.07 so với 6.61.
    - Tốc độ tay hơi cao (1.72×), cần cắt đoạn và chuyển tiếp (R5, R7). (F37, F38)
16. **Ngữ nghĩa cần sự nhất quán toàn thân.** Ghép tay MaDiS với ngón retrieval, hoặc ngược lại, đều mất toàn bộ lợi ích (BLEU-4 0.19 / 0.14, ngang MaDiS). Vì vậy renderer phải giữ nguyên đoạn retrieval (mọi stream) và chỉ chỉnh nhịp và chuyển tiếp. (F36)
15. **Ổn định theo seed:** 3 seed distractor cho BLEU-4 0.888 ± 0.013. Cả 3 đều vượt MaDiS với p < 0.001. (F35)
14. **Evaluator CSL độc lập (Uni-Sign chỉ pre-train trên CSL-News):** tuy yếu (trần GT chỉ 0.78 BLEU-4), vẫn cho thấy cùng mẫu với How2Sign:
    - MaDiS ≤ mean pose (0.18 vs 0.24);
    - retrieval chuyển động thật **vượt MaDiS +0.26 BLEU-4 (≈2.5×), p < 0.001**.

    Đây là bằng chứng hỗ trợ trên benchmark thứ hai. (F34)

### Đề xuất FLARE-Sign ([05](05_proposed_method.md))

**Kiến trúc cuối:** *"retrieve real, plan with LM, blend with flow"*.
- Lexicon đoạn chuyển động thật (C2, liên tục).
- LM masked-diffusion lập kế hoạch: chọn đoạn, thứ tự, độ dài.
- Flow matching sinh chuyển tiếp và inpaint (C1).
- RL trên planner (C3).

Baseline bắt buộc phải vượt: MaDiS và text-NN retrieval.


Ưu tiên sau khi đo: **C3 > C2 > C1**, cùng với C0 bắt buộc.

* **C0 — protocol hiệu chuẩn:**
  - hàng tầm thường (mean pose, GT đảo, NN) và sàn tokenizer;
  - DTW theo từng bước và tỉ lệ tốc độ;
  - SiBLEU với tham chiếu re-encode;
  - BT-retrieval R@k;
  - evaluator **train trên dữ liệu rời** (signer-held-out, SLT ngoài như Uni-Sign);
  - đánh giá bởi người;
  - gắn cờ mọi điểm "vượt GT".
* **C3 — RL đầu tiên cho sign production:** diffu-GRPO / RAFT trên masked-diffusion LM, với reward = **BT likelihood dạng contrastive** + phạt đảo thứ tự + kinematic. Reward model và evaluator khác dữ liệu train.
* **C2 — retrieval từ lexicon tự cảm sinh** từ train split, cộng condition-mixture CFG (MaDiS đã bỏ retrieval).
* **C1 — refiner flow:**
  - C1a: hiệu chỉnh trung bình có điều kiện cho chuỗi tay / cổ tay;
  - C1b: articulation lấy mẫu;
  - giữ vai trò khuếch đại độ trung thực.

**Hướng phụ:** benchmark 3-D SLP tiếng Việt đầu tiên từ `OurVSL_small` (~15k câu).

## Documents

| File | Content |
|---|---|
| [01_literature_review.md](01_literature_review.md) | SLP evolution 2020–2026, transferable T2M techniques, leaderboard, evaluation critique, reading list |
| [02_codebase_and_data_audit.md](02_codebase_and_data_audit.md) | What the three repos actually implement; local assets; datasets on this machine |
| [03_diagnostics.md](03_diagnostics.md) | All measurements, findings F1–F19 |
| [04_research_gaps.md](04_research_gaps.md) | Gaps G1–G7 with evidence, research questions RQ1–RQ7 |
| [05_proposed_method.md](05_proposed_method.md) | FLARE-Sign C0–C3, revisions after each experiment, novelty search, risks |
| [06_experiment_plan.md](06_experiment_plan.md) | Phases, ablations, compute budget, job policy, deliverables |
| [_raw_lit_slp.md](_raw_lit_slp.md), [_raw_lit_motion.md](_raw_lit_motion.md) | Raw survey appendices |

## Code and artefacts produced

| What | Where |
|---|---|
| Data packers (SOKE fittings → MaDiS layout) | `scripts/madis_build_{phoenix,csl,how2sign}.py` |
| Shared loaders `phoenix_smplx` / `csl_smplx` / `how2sign_smplx` | `src/slp/data/sign_smplx.py`, `configs/data/*_smplx.yaml` (tested) |
| Shared metrics `dtw_jpe`, `dtw_jpe_norm`, `sibleu_unigram`, `sibleu4` | `src/slp/evaluation/metrics.py` (tested) |
| Calibration and oracle rows | `scripts/madis_oracle_eval.py`, `scripts/score_predictions.py` |
| MaDiS dev configs and generation (incl. train split) | `configs/madis/*.dev.yaml`, `scripts/madis_generate.py` |
| Back-translators B1 (mBART) / B2 (Qwen-LoRA), BLEU / chrF / likelihood / BT-retrieval | `scripts/train_backtranslator.py`, `outputs/bt/` |
| C1 refiner (flow / regress, self-gen targets) | `methods/d2c_sign/refiner.py`, `scripts/{decode_sign_tokens,build_selfgen_targets}.py` |
| GPU job queue (≤ 2–3 processes) | `scripts/jobq.py`, `outputs/jobs/` |
| Raw result JSONs | `outputs/diagnostics/` |

## Reproducing

```bash
python scripts/madis_build_phoenix.py      # also _csl / _how2sign
python scripts/jobq.py add <name> "<cmd>"; python scripts/jobq.py run --workers 1
# released MaDiS on dev
cd third_party/MaDiS && .venv/bin/python test.py --cfg /home/haipd/SLP/configs/madis/madis_phoenix.dev.yaml --nodebug
# paired DTW scoring with calibration rows
.venv/bin/python /home/haipd/SLP/scripts/score_predictions.py --source phoenix --split dev madis=<pred_dir> @mean_pose @gt_reversed
# back-translation incl. BT-retrieval
cd /home/haipd/SLP && third_party/MaDiS/.venv/bin/python scripts/train_backtranslator.py --arch mbart \
    --out outputs/bt/phoenix_mbart --retrieval --score @gt @mean_pose madis=<pred_dir>
```

## Next steps (in priority order)

1. **Evaluator independence (C0).** Build a signer-held-out or sentence-disjoint
   back-translator and an external SLT evaluator. Without these, no SOTA ranking is
   possible (F17–F19).
2. **C3 pilot.**
   - RAFT (best-of-N + SFT) with the contrastive BT reward on Phoenix, using the
     released MaDiS SFT checkpoint and LoRA.
   - Then diffu-GRPO.
3. **C2:** train-split lexicon induction and retrieval-conditioned SFT continuation.
4. **Replicate on CSL-Daily and How2Sign.** Use 3 seeds, a human study, and test
   only on request.

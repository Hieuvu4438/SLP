---
status: exploring   # exploring | promising | final | abandoned
created: 2026-11-03
---

# d2c_sign (FLARE-Sign, component C1 + pilots)

Working name "discrete-to-continuous". This is the code for FLARE-Sign component C1,
the token-conditioned flow refiner, plus the tokenizer pilot. The full method and
its context are in `docs/proposal1/05_proposed_method.md`.

## Hypothesis
LM-based SLP (SOKE / MaDiS) is capped by the decoupled VQ tokenizer. A continuous
residual refiner R(x̃ = D(c)) → x should:
1. lower the quantization floor;
2. raise articulation and back-translation of LM outputs, without retraining the LM.

## Method
- `refiner.py`:
  - residual rectified flow on r = (x − x̃)/s (per-dim residual scale);
  - transformer d = 384 × 6 with a conv front-end;
  - `--mode regress` gives the deterministic L1 baseline.
- Training inputs:
  - GT-token decodes, plus 3 corrupted variants (code replacement p ≤ 0.3, ±1 hand
    shift);
  - optionally MaDiS self-generated train outputs with DTW-warped GT targets
    (`--selfgen_npz`).
- Inference: 8 Euler steps, temperature τ.
- `tokenizer_pilot.py`: right-hand VQ / RVQ pilot. It was stopped after finding F5:
  the hand floor lives in the body stream.

## How to run (all through `scripts/jobq.py`)
```bash
third_party/MaDiS/.venv/bin/python methods/d2c_sign/refiner.py train --mode flow --out outputs/refiner/flow_v0
third_party/MaDiS/.venv/bin/python methods/d2c_sign/refiner.py infer --ckpt outputs/refiner/flow_v0/model.pt \
    --madis_dir third_party/MaDiS/results/madis/madis_phoenix_dev/val_rank_0 --temp 0 --out outputs/refiner/flow_v0/dev_madis_t0.npz
```
This method does not yet go through `scripts/run.py` (fit/predict). Porting it is a
todo, and all numbers below come from the shared `scripts/score_predictions.py` and
`scripts/train_backtranslator.py`.

## Results (Phoenix dev, 1 seed; see `docs/proposal1/03_diagnostics.md`)
| Experiment | Input | DTW body / hands | B1 BLEU-4 | B2 BLEU-4 | Notes |
|---|---|---|---|---|---|
| tokenizer floor | GT tokens | 3.12 / 4.80 | 7.82 | 6.87 | – |
| flow_v0 τ = 0 | GT tokens | **2.75 / 3.96** | 9.62 | 9.49 | floor −12% / −17% |
| MaDiS released | MaDiS | 5.19 / 6.94 | 9.76 | 8.48 | – |
| flow_v0 τ = 0 | MaDiS | 5.16 / 6.93 | **11.25** | **10.00** | BT +1.5 under both evaluators |
| flow_v0 τ = 1 | MaDiS | 5.41 / 7.08 | 9.94 | – | hand speed 0.35 → 0.87 of GT |
| flow_v1 self-gen τ = 0 | MaDiS | 5.15 / 6.96 | 10.48 | 9.96 | self-gen does not fix planning |
| regress_v0 | GT tokens | 2.87 / 4.04 | 10.52 | 9.43 | flow τ = 0 slightly better on DTW |
| regress_v0 | MaDiS | 5.22 / 6.94 | **12.50** | 10.56 | **above GT under B1**: evaluator saturation (F20) |

## Notes / log
- 2026-11-03: created. The pilot showed finger codebooks are not the bottleneck (F5).
- 2026-11-03: v0 first looked negative because the released Phoenix-dev tokens
  mismatch the local poses (F8). It was fixed with `--reencode`.
- 2026-11-03: the BT gains need disjoint-data evaluators before they can be claimed
  (F17–F19).

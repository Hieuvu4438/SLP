---
status: exploring   # exploring | promising | final | abandoned
created: 2026-11-20
---

# signmflow_plus

Experiment variants of SignNMFlow (arXiv:2609.33339) for the plan in `docs/proposal7/07_lo_trinh_uu_tien.md`.

The core modules are a self-contained copy of `methods/signmflow` (methods never import each other).
With the default config, behaviour is identical to the paper replica (`methods/signmflow`, v2).
Every new feature is behind a config flag whose default reproduces the paper.

## Hypothesis
See `docs/proposal7/04_ke_hoach_thuc_nghiem.md` and `07_lo_trinh_uu_tien.md`.

## How to run
```bash
python scripts/run.py --method signmflow_plus --data phoenix_smplx --name <experiment> --seeds 0
```

## Results
| Experiment | Data | Seeds | Metric (mean ± std) | Run dir / notes |
|---|---|---|---|---|

## Notes / log
- 2026-11-20: created as a copy of `methods/signmflow` (v2).

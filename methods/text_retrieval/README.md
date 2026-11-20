---
status: exploring   # exploring | promising | final | abandoned
created: 2026-11-20
---

# text_retrieval

Trivial baselines that calibrate the DTW metrics (docs/proposal7, experiment E0.4).

## Hypothesis
DTW-JPE rewards regression to the mean, and Phoenix repeats many sentences, so two text-free or
lookup baselines set the floor any generator has to clear:
- `mode=mean_pose`: the mean pose for every frame (no text, no motion);
- `mode=retrieval`: the motion of the most similar training sentence (character TF-IDF).
If SignNMFlow does not beat retrieval, its text conditioning adds little beyond a lookup.

## How to run
```bash
python scripts/run.py --method text_retrieval --data phoenix_smplx --name e04_retrieval
python scripts/run.py --method text_retrieval --data phoenix_smplx --name e04_mean_pose method.mode=mean_pose
```
No training; the runs take the evaluation time only (CPU is enough).

## Results
| Experiment | Data | Seeds | Metric (mean ± std) | Run dir / notes |
|---|---|---|---|---|

## Notes / log
- 2026-11-20: created for docs/proposal7 E0.4 (not run yet).

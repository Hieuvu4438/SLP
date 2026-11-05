"""Paired bootstrap for corpus BLEU-4 differences between two systems scored by scripts/unisign_bt_eval.py.

  python scripts/paired_bootstrap.py outputs/diagnostics/X.json best random [--n 1000]
Uses sacrebleu (13a) on the saved per-sentence predictions (CSL outputs are already char-split).
Reports delta BLEU-4, 95% CI, and p(delta <= 0).
"""
import argparse, json
import numpy as np
import sacrebleu

ap = argparse.ArgumentParser()
ap.add_argument("json"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
x = ap.parse_args()
d = json.load(open(x.json))
A, B = d[x.a], d[x.b]
assert A["names"] == B["names"]
refs, pa, pb = A["refs"], A["preds"], B["preds"]
bleu = lambda h, r: sacrebleu.corpus_bleu(h, [r], tokenize="13a").score
base = bleu(pa, refs) - bleu(pb, refs)
rng = np.random.default_rng(x.seed); N = len(refs); deltas = []
for _ in range(x.n):
    i = rng.integers(0, N, N)
    r = [refs[j] for j in i]
    deltas.append(bleu([pa[j] for j in i], r) - bleu([pb[j] for j in i], r))
deltas = np.array(deltas)
print(json.dumps({"a": x.a, "b": x.b, "delta_bleu4": round(base, 3),
                  "ci95": [round(float(np.percentile(deltas, 2.5)), 3), round(float(np.percentile(deltas, 97.5)), 3)],
                  "p_delta_le_0": round(float((deltas <= 0).mean()), 4), "n": N}))

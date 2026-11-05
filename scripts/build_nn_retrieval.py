"""Text-nearest-neighbour retrieval baseline: copy the TRAIN clip whose sentence is most similar.

For each val clip, pick the train clip with the highest TF-IDF similarity (word 1-2 grams for
How2Sign / Phoenix, char 1-3 grams for CSL) and output its normalised 133-D motion. No
generation; it measures how much "whole-sentence retrieval" conveys under external evaluators.
Train data only (no val/test leakage in the index).

  python scripts/build_nn_retrieval.py --source how2sign --out outputs/retrieval/how2sign_val_nn.npz
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/haipd/SLP/src")
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop
from sklearn.feature_extraction.text import TfidfVectorizer

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="how2sign")
ap.add_argument("--split", default="val")
ap.add_argument("--out", required=True)
ap.add_argument("--topk", type=int, default=1, help=">1: write <out_dir>/r{i}.npz for the i-th nearest train clip")
a = ap.parse_args()
roots = {"phoenix": "Phoenix_2014T", "csl": "CSL-Daily", "how2sign": "How2Sign"}
root = Path(f"/home/haipd/SLP/data/processed/madis/{roots[a.source]}")
mean, std = load_stats("/home/haipd/SLP/third_party/SOKE/deps/stats")
tr, va = LOADERS[a.source](root, "train"), LOADERS[a.source](root, a.split)
vec = (TfidfVectorizer(analyzer="char", ngram_range=(1, 3)) if a.source == "csl"
       else TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)).fit([r[0]["text"] for r in tr])
Xt, Xv = vec.transform([r[0]["text"] for r in tr]), vec.transform([r[0]["text"] for r in va])
S = (Xv @ Xt.T).toarray()
order = np.argsort(-S, axis=1)[:, : a.topk]


def motion(t):
    f = feats133(pickle.load(open(t[1], "rb")))
    if t[2] is not None and t[2] > 24:
        f = f[np.linspace(0, len(f) - 1, int(24 * len(f) / t[2]), dtype=int)]
    return madis_crop((f - mean) / std).astype(np.float32)


if a.topk == 1:
    out = {"val__" + Path(r[0]["name"]).name: motion(tr[order[i, 0]]) for i, r in enumerate(va)}
    np.savez(a.out, **out)
else:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for k in range(a.topk):
        np.savez(Path(a.out) / f"r{k}.npz", **{"val__" + Path(r[0]["name"]).name: motion(tr[order[i, k]]) for i, r in enumerate(va)})
print("clips", len(va), "exact-text top1 matches", sum(va[i][0]["text"] == tr[order[i, 0]][0]["text"] for i in range(len(va))))

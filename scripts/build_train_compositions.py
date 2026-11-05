"""R7 training data: for each TRAIN sentence, compose two OTHER train clips (top-1 per half by TF-IDF,
excluding clips with the same sentence text) retimed to the GT length; target = GT DTW-warped onto the
composition timeline (keeps composition timing, learns pose adaptation). Refiner npz format."""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/haipd/SLP/src"); sys.path.insert(0, "/home/haipd/SLP/scripts")
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop
from build_selfgen_targets import dtw_path
from sklearn.feature_extraction.text import TfidfVectorizer
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=12000); ap.add_argument("--out"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
root = Path("/home/haipd/SLP/data/processed/madis/How2Sign"); mean, std = load_stats("/home/haipd/SLP/third_party/SOKE/deps/stats")
tr = LOADERS["how2sign"](root, "train"); rng = np.random.default_rng(a.seed)
idx = rng.choice(len(tr), min(a.limit, len(tr)), replace=False)
texts = [r[0]["text"] for r in tr]
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts); Xt = vec.transform(texts)
def halves(t):
    w = t.split(); m = max(1, len(w) // 2); return " ".join(w[:m]), " ".join(w[m:]) or " ".join(w[:m])
def motion(j):
    t = tr[j]; f = feats133(pickle.load(open(t[1], "rb")))
    if t[2] and t[2] > 24: f = f[np.linspace(0, len(f) - 1, int(24 * len(f) / t[2]), dtype=int)]
    return madis_crop((f - mean) / std).astype(np.float32)
def top_other(q, i):
    s = (vec.transform([q]) @ Xt.T).toarray()[0]
    for j in np.argsort(-s)[:20]:
        if texts[j] != texts[i]: return j
    return None
w = np.ones(133, dtype=np.float32); w[30:120] = 2.0; out = {}; n = 0
for i in idx:
    h1, h2 = halves(texts[i]); j1, j2 = top_other(h1, i), top_other(h2, i)
    if j1 is None or j2 is None: continue
    gt = motion(i); c = np.concatenate([motion(j1), motion(j2)])
    c = c[np.linspace(0, len(c) - 1, len(gt)).round().astype(int)]
    cost = np.sqrt((((c[:, None] - gt[None]) ** 2) * w).sum(-1))
    tgt = np.zeros_like(c); cnt = np.zeros(len(c))
    for p, q in dtw_path(cost): tgt[p] += gt[q]; cnt[p] += 1
    key = f"train__{Path(tr[i][0]['name']).name}"
    out[f"gt::{key}"] = (tgt / cnt[:, None]).astype(np.float32); out[f"tok::{key}::0"] = c; n += 1
    if n % 1000 == 0: print(n, flush=True)
np.savez(a.out, **out); print("clips", n)

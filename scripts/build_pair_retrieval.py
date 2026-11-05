"""Compositional retrieval (R5-lite): split each input sentence into two halves (by words), retrieve the
top-k TRAIN clips per half (TF-IDF), and write all k*k concatenations (first-half clip ++ second-half clip),
each retimed to the MaDiS-generated length for that sentence (never the GT length).
Output: <out>/r{i}.npz, i = a*k + b, keys val__<name>."""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/haipd/SLP/src")
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop
from sklearn.feature_extraction.text import TfidfVectorizer
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="how2sign", choices=["phoenix", "csl", "how2sign"]); ap.add_argument("--trim", action="store_true", help="keep only the time span of words overlapping the query half (proportional heuristic)"); ap.add_argument("--aligner", default=None, help="aligner ckpt: trim each clip to the span its query-half tokens align to (R5)"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--len_dir"); ap.add_argument("--out")
a = ap.parse_args()
root = Path("/home/haipd/SLP/data/processed/madis/" + {"phoenix": "Phoenix_2014T", "csl": "CSL-Daily", "how2sign": "How2Sign"}[a.source])
mean, std = load_stats("/home/haipd/SLP/third_party/SOKE/deps/stats")
tr, va = LOADERS[a.source](root, "train"), LOADERS[a.source](root, "val")
vec = (TfidfVectorizer(analyzer="char", ngram_range=(1, 3)) if a.source == "csl"
       else TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)).fit([r[0]["text"] for r in tr])
Xt = vec.transform([r[0]["text"] for r in tr])
def halves(t):
    if a.source == "csl":  # Chinese: split by characters
        m = max(1, len(t) // 2); return t[:m], t[m:] or t[:m]
    w = t.split(); m = max(1, len(w) // 2); return " ".join(w[:m]), " ".join(w[m:]) or " ".join(w[:m])
H1, H2 = zip(*[halves(r[0]["text"]) for r in va])
O1 = np.argsort(-(vec.transform(H1) @ Xt.T).toarray(), 1)[:, : a.k]
O2 = np.argsort(-(vec.transform(H2) @ Xt.T).toarray(), 1)[:, : a.k]
cache = {}
def motion(j):
    if j not in cache:
        t = tr[j]; f = feats133(pickle.load(open(t[1], "rb")))
        if t[2] and t[2] > 24: f = f[np.linspace(0, len(f) - 1, int(24 * len(f) / t[2]), dtype=int)]
        cache[j] = ((f - mean) / std).astype(np.float32)
    return cache[j]
import re
tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())
AL = None
if a.aligner:
    import torch
    sys.path.insert(0, "/home/haipd/SLP/methods/d2c_sign")
    from aligner import Aligner, LANG
    torch.backends.cuda.enable_cudnn_sdp(False)
    AL = Aligner(LANG[a.source]).cuda().eval()
    AL.load_state_dict(torch.load(a.aligner, map_location="cuda"), strict=False)


def aligned_span(m, half):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(m[:400])[None].cuda(); mk = torch.ones(1, x.shape[1], dtype=torch.bool, device="cuda")
        T, tm, _ = AL.encode_text([half], "cuda"); M, mm = AL.menc(x, mk)
        s = torch.einsum("ld,fd->lf", T[0].float(), M[0].float())[tm[0]]
    if s.shape[0] == 0: return m
    fr = s.argmax(-1).cpu().numpy() * 2  # encoder runs at half frame rate
    lo, hi = np.percentile(fr, 10), np.percentile(fr, 90)
    s_, e_ = max(0, int(lo) - 4), min(len(m), int(hi) + 6)
    return m[s_:e_] if e_ - s_ >= 8 else m


def seg(j, half):
    m = motion(j)
    if AL is not None: return aligned_span(m, half)
    if not a.trim: return m
    w = tok(tr[j][0]["text"]); q = set(tok(half))
    hit = [i for i, x in enumerate(w) if x in q]
    if not w or not hit: return m
    n = len(m); lo = max(0.0, hit[0] / len(w) - 0.1); hi = min(1.0, (hit[-1] + 1) / len(w) + 0.1)
    s_, e_ = int(lo * n), max(int(lo * n) + 8, int(hi * n))
    return m[s_:min(e_, n)]
outs = [dict() for _ in range(a.k * a.k)]
for i, r in enumerate(va):
    n = Path(r[0]["name"]).name; p = Path(a.len_dir) / f"{n}.pkl"
    if not p.exists(): continue
    L = len(pickle.load(open(p, "rb"))["feats_rst"])
    for x in range(a.k):
        for y in range(a.k):
            c = np.concatenate([seg(O1[i, x], H1[i]), seg(O2[i, y], H2[i])])
            outs[x * a.k + y]["val__" + n] = c[np.linspace(0, len(c) - 1, L).round().astype(int)]
    if len(cache) > 6000: cache.clear()
Path(a.out).mkdir(parents=True, exist_ok=True)
for i, d in enumerate(outs): np.savez(Path(a.out) / f"r{i}.npz", **d)
print("clips", len(outs[0]), "pairs", len(outs))

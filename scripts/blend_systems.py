"""Uniform blend of two equal-length systems in the 133-D (normalised axis-angle) space: (1-a)*X + a*Y."""
import argparse, pickle
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--x"); ap.add_argument("--y"); ap.add_argument("--alpha", type=float); ap.add_argument("--out")
a = ap.parse_args()
def load(p):
    p = Path(p)
    if p.is_dir(): return {f.stem: pickle.load(open(f, "rb"))["feats_rst"].astype(np.float32) for f in p.glob("*.pkl")}
    z = np.load(p); return {k.split("__")[-1]: z[k] for k in z.files}
X, Y = load(a.x), load(a.y); out = {}
for n in X.keys() & Y.keys():
    L = min(len(X[n]), len(Y[n])); out["val__" + n] = (1 - a.alpha) * X[n][:L] + a.alpha * Y[n][:L]
np.savez(a.out, **out); print("clips", len(out))

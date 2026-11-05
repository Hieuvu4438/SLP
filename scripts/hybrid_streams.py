"""Stream-swap hybrids between two systems with equal per-clip length (e.g. retimed retrieval vs MaDiS).
A = body(+face) from --body_src, fingers from --hand_src. Layout: body 0:30, hands 30:120, face 120:133."""
import argparse, pickle
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--body_src"); ap.add_argument("--hand_src"); ap.add_argument("--out")
a = ap.parse_args()
def load(p):
    p = Path(p)
    if p.is_dir():
        return {f.stem: pickle.load(open(f, "rb"))["feats_rst"].astype(np.float32) for f in p.glob("*.pkl")}
    z = np.load(p); return {k.split("__")[-1]: z[k] for k in z.files}
B, H = load(a.body_src), load(a.hand_src); out = {}
for n in B.keys() & H.keys():
    b, h = B[n], H[n]; L = min(len(b), len(h))
    x = b[:L].copy(); x[:, 30:120] = h[:L, 30:120]
    out["val__" + n] = x
np.savez(a.out, **out); print("clips", len(out))

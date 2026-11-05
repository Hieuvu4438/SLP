"""Resample each motion in an npz to the length of another system's output (e.g. MaDiS-predicted length).
Only *generated* lengths are used (never the GT length), so this stays a valid text-to-motion system."""
import argparse, pickle
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--len_dir"); ap.add_argument("--out")
a = ap.parse_args()
z = np.load(a.src); out = {}
for k in z.files:
    p = Path(a.len_dir) / f"{k.split('__')[-1]}.pkl"
    if not p.exists():
        continue
    L = len(pickle.load(open(p, "rb"))["feats_rst"]); f = z[k]
    out[k] = f[np.linspace(0, len(f) - 1, L).round().astype(int)]
np.savez(a.out, **out); print("clips", len(out))

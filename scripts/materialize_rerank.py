"""Build the feats npz of a rerank selection: for each clip pick r{argmax reward}.npz (from build_nn_retrieval --topk)."""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--rewards"); ap.add_argument("--retr_dir"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out")
a = ap.parse_args()
rw = json.load(open(a.rewards))
R = [np.load(f"{a.retr_dir}/r{k}.npz") for k in range(a.n)]
out = {}
for name, r in rw.items():
    key = "val__" + name
    out[key] = R[int(np.argmax(r))][key]
np.savez(a.out, **out); print("clips", len(out))

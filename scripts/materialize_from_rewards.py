"""Pick r{argmax reward}.npz per clip (rewards json keyed by bare name; npz keys val__<name>)."""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--rewards"); ap.add_argument("--dir"); ap.add_argument("--n", type=int); ap.add_argument("--out")
a = ap.parse_args(); rw = json.load(open(a.rewards)); Z = [np.load(f"{a.dir}/r{k}.npz") for k in range(a.n)]
np.savez(a.out, **{"val__" + n: Z[int(np.argmax(r))]["val__" + n] for n, r in rw.items()}); print("clips", len(rw))

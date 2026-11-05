"""FLARE C3 step 1: best-of-N reranking of MaDiS samples by a contrastive back-translation reward.

reward(x̂) = ℓ(y | x̂) − log Σ_{y' ∈ {y} ∪ D} exp ℓ(y' | x̂),
where ℓ is the length-normalised log-likelihood under the REWARD back-translator (B1,
trained on the generator's train split) and D are K random distractor sentences.
The evaluator (Uni-Sign, data-/modality-disjoint) is never used here.

Outputs npz files {name: feats}, all with the same clips:
  best.npz (argmax reward), random.npz (uniform sample, seeded), s0.npz (first sample),
  rewards.json (per-clip rewards of every sample).

  third_party/MaDiS/.venv/bin/python scripts/bt_rerank.py --source how2sign --bt outputs/bt/how2sign_mbart \
      --samples outputs/madis_samples/how2sign_val500_t08 --out outputs/rerank/how2sign_val500_t08
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/haipd/SLP/scripts")
import train_backtranslator as bt  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="how2sign", choices=["phoenix", "csl", "how2sign"])
    ap.add_argument("--bt", required=True, help="dir with best.pt of the reward back-translator (mBART)")
    ap.add_argument("--samples", default=None, help="dir with s0, s1, ... MaDiS sample dirs")
    ap.add_argument("--npz_dir", default=None, help="alternative: dir with r0.npz, r1.npz, ... (e.g. top-k retrieval)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    model = bt.MBartBT(bt.LANG[a.source]).to(dev)
    model.load_state_dict(torch.load(Path(a.bt) / "best.pt", map_location=dev))
    model.eval()
    if a.npz_dir:
        sys.path.insert(0, "/home/haipd/SLP/src")
        from slp.data.sign_smplx import LOADERS
        roots = {"phoenix": "Phoenix_2014T", "csl": "CSL-Daily", "how2sign": "How2Sign"}
        rows = LOADERS[a.source](Path(f"/home/haipd/SLP/data/processed/madis/{roots[a.source]}"), a.split)
        all_texts = {Path(r[0]["name"]).name: r[0]["text"] for r in rows}
        zs = [np.load(p) for p in sorted(Path(a.npz_dir).glob("r*.npz"), key=lambda p: int(p.stem[1:]))]
        names = sorted({k.split("__")[-1] for k in zs[0].files} & set(all_texts))
        texts = {n: all_texts[n] for n in names}
        key = {n: k for k in zs[0].files for n in [k.split("__")[-1]]}
        get = lambda n: [z[key[n]] for z in zs]
        n_s = len(zs)
    else:
        sdirs = sorted(p for p in Path(a.samples).iterdir() if p.is_dir() and p.name.startswith("s"))
        names = sorted(set.intersection(*[{f.stem for f in d.glob("*.pkl")} for d in sdirs]))
        texts = {n: pickle.load(open(sdirs[0] / f"{n}.pkl", "rb"))["text"] for n in names}
        get = lambda n: [pickle.load(open(d / f"{n}.pkl", "rb"))["feats_rst"].astype(np.float32) for d in sdirs]
        n_s = len(sdirs)
    pool = sorted(set(texts.values()))
    best, rand, s0, rewards = {}, {}, {}, {}
    for n in names:
        y = texts[n]
        distract = [pool[i] for i in rng.choice(len(pool), size=min(a.k + 1, len(pool)), replace=False) if pool[i] != y][: a.k]
        cands = [y] + distract
        feats = [np.asarray(f, dtype=np.float32) for f in get(n)]
        r = []
        for f in feats:
            x = torch.from_numpy(f[:400])[None].to(dev)
            m = torch.ones(1, x.shape[1], dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = np.array(bt.seq_logprob(model, x.expand(len(cands), -1, -1), m.expand(len(cands), -1), cands))
            r.append(float(lp[0] - np.log(np.exp(lp - lp.max()).sum()) - lp.max()))
        rewards[n] = r
        k_ = f"{a.split}__{n}"
        best[k_] = feats[int(np.argmax(r))]
        rand[k_] = feats[int(rng.integers(len(feats)))]
        s0[k_] = feats[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for lab, d in (("best", best), ("random", rand), ("s0", s0)):
        np.savez(out / f"{lab}.npz", **d)
    json.dump(rewards, open(out / "rewards.json", "w"))
    r = np.array(list(rewards.values()))
    print(json.dumps({"clips": len(names), "n_samples": n_s, "mean_reward_s0": float(r[:, 0].mean()),
                      "mean_reward_best": float(r.max(1).mean())}))


if __name__ == "__main__":
    main()

"""FLARE C3 best-of-N with a strong external REWARD: Uni-Sign (How2Sign-trained) contrastive likelihood.

The reward and evaluator never share data:
- reward    = Uni-Sign fine-tuned on How2Sign (real-video RTMPose);
- evaluator = Uni-Sign fine-tuned on OpenASL (never saw How2Sign); used only afterwards.

reward(x̂) = ℓ(y | x̂) − logsumexp_{y' ∈ {y} ∪ D} ℓ(y' | x̂), with K random distractor val sentences D.

Input : <kp_dir>/s0.npz … s{N-1}.npz (COCO-WholeBody keypoints of the N MaDiS samples, from
        scripts/smplx_to_cocowb.py) and texts.pkl
Output: <kp_dir>/best_us.npz, random_us.npz, rewards_us.json
Run in the Uni-Sign venv.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

UNISIGN = "/home/haipd/Sign_Representations/third_party/Uni-Sign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="How2Sign")
    ap.add_argument("--kp_dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _s1 for distractor seed 1)")
    a = ap.parse_args()
    kp_dir = Path(a.kp_dir).resolve(); ckpt = Path(a.ckpt).resolve()
    os.chdir(UNISIGN); sys.path.insert(0, UNISIGN)
    from models import Uni_Sign
    from pose_preprocessing import load_part_kp
    args = SimpleNamespace(dataset=a.dataset, hidden_dim=256, label_smoothing=0.2, rgb_support=False,
                           mt5_path=f"{UNISIGN}/pretrained_weight/mt5-base", language="")
    model = Uni_Sign(args=args)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=True)
    model = model.cuda().eval()
    texts = pickle.load(open(kp_dir / "texts.pkl", "rb"))
    samples = [np.load(kp_dir / f"s{k}.npz") for k in range(a.n)]
    names = sorted(set.intersection(*[set(z.files) for z in samples]))
    pool = sorted({texts[n] for n in names})
    rng = np.random.default_rng(a.seed)

    def encode(arr):
        idx = np.linspace(0, len(arr) - 1, min(len(arr), 256), dtype=int)
        parts = load_part_kp([arr[i:i + 1, :, :2] for i in idx], [arr[i:i + 1, :, 2] for i in idx], force_ok=True)
        src = {k: v[None].cuda() for k, v in parts.items()}
        src["attention_mask"] = torch.ones(1, len(idx), dtype=torch.long).cuda()
        src["src_length_batch"] = torch.tensor([len(idx)])
        return model.encode(src)

    best, rand, rewards = {}, {}, {}
    with torch.no_grad():
        for n in names:
            y = texts[n]
            d = [pool[i] for i in rng.choice(len(pool), size=min(a.k + 1, len(pool)), replace=False) if pool[i] != y][: a.k]
            cands = [y] + d
            r = []
            for z in samples:
                lp = model.score_candidate_labels(encode(z[n]), cands)[0].float().cpu().numpy()
                r.append(float(lp[0] - (np.log(np.exp(lp - lp.max()).sum()) + lp.max())))
            rewards[n] = r
            best[n] = samples[int(np.argmax(r))][n]
            rand[n] = samples[int(rng.integers(a.n))][n]
    np.savez(kp_dir / f"best_us{a.tag}.npz", **best)
    np.savez(kp_dir / f"random_us{a.tag}.npz", **rand)
    json.dump(rewards, open(kp_dir / f"rewards_us{a.tag}.json", "w"))
    R = np.array(list(rewards.values()))
    print(json.dumps({"clips": len(names), "mean_reward_s0": float(R[:, 0].mean()), "mean_reward_best": float(R.max(1).mean()),
                      "argmax_hist": np.bincount(R.argmax(1), minlength=a.n).tolist()}))


if __name__ == "__main__":
    main()

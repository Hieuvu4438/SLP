"""Diagnostic baselines for the SOKE/MaDiS SLP protocol (val/dev or test split).

Computes, with MaDiS's own DTW-JPE code (root/wrist translation aligned,
accumulated DTW cost, averaged over sequences):

  oracle_tok : GT poses -> SOKE decoupled tokenizer -> decode   (quantization floor)
  mean_pose  : the training-set mean pose repeated for the GT length
  nn_text    : pose of the training clip whose text is most similar (TF-IDF char n-gram)
  gt_shift   : GT sequence time-reversed (a semantically wrong but natural motion)

Also reports per-frame (path-normalised) DTW so that the accumulated values can be
interpreted, and codebook usage of the tokenizer.

Run from third_party/MaDiS with its venv:
  .venv/bin/python /home/haipd/SLP/scripts/madis_oracle_eval.py --split dev
"""

import argparse
import gzip
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from mGPT.archs.mgpt_vq import VQVae  # noqa: E402
from mGPT.metrics.dtw import pairwise_jpe  # noqa: E402
from mGPT.utils.human_models import get_coord, smpl_x  # noqa: E402

KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
        "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr")
SHAPE = [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
         0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]


def feats133(p):
    f = np.concatenate([p[k] for k in KEYS], 1)[:, 36:]
    return np.concatenate([f[:, :-20], f[:, -10:]], 1).astype(np.float32)


def dtw_with_len(cost):
    """Accumulated DTW cost and warping-path length for a [Tx,Ty] matrix."""
    c = cost.double()
    r, k = c.shape
    acc = torch.full((r + 1, k + 1), float("inf"), dtype=c.dtype, device=c.device)
    ln = torch.zeros_like(acc)
    acc[0, 0] = 0
    for d in range(2, r + k + 1):
        i = torch.arange(max(1, d - k), min(r, d - 1) + 1, device=c.device)
        j = d - i
        cand = torch.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        lens = torch.stack([ln[i - 1, j - 1], ln[i - 1, j], ln[i, j - 1]])
        best = cand.argmin(0)
        acc[i, j] = c[i - 1, j - 1] + cand.gather(0, best[None])[0]
        ln[i, j] = lens.gather(0, best[None])[0] + 1
    return acc[r, k].item(), ln[r, k].item()


class Joints:
    def __init__(self, mean, std, device):
        self.mean, self.std, self.device = mean.to(device), std.to(device), device

    def __call__(self, feats_norm):  # [T,133] normalised
        f = feats_norm.to(self.device) * self.std + self.mean
        T = f.shape[0]
        f = torch.cat([torch.zeros(T, 36, device=self.device), f], -1)
        shape = torch.tensor(SHAPE, device=self.device).expand(T, -1)
        verts, joints = get_coord(f[:, :3], f[:, 3:66], f[:, 66:111], f[:, 111:156],
                                  f[:, 156:159], shape, f[:, 159:169])
        lh = smpl_x.orig_hand_regressor["left"].to(verts) @ verts
        rh = smpl_x.orig_hand_regressor["right"].to(verts) @ verts
        return joints, lh, rh


def scores(gen, ref):
    gj, gl, gr = gen
    rj, rl, rr = ref
    body = pairwise_jpe(gj, rj, smpl_x.joint_part2idx["upper_body"], 0)
    lh = pairwise_jpe(gl, rl, None, 0)
    rh = pairwise_jpe(gr, rr, None, 0)
    out = {}
    for name, c in (("body", body), ("lhand", lh), ("rhand", rh)):
        a, n = dtw_with_len(c)
        out[name] = a
        out[name + "_perframe"] = a / n
    out["hands"] = (out["lhand"] + out["rhand"]) / 2
    out["hands_perframe"] = (out["lhand_perframe"] + out["rhand_perframe"]) / 2
    return out


def load_vae(path):
    from mGPT.utils.load_checkpoint import _state_dict
    sd = _state_dict(path)
    parts = {"body": OrderedDict(), "lh": OrderedDict(), "rh": OrderedDict()}
    for k, v in sd.items():
        if "motion_vae" in k:
            parts["body"][k.replace("motion_vae.", "")] = v
        elif "rhand_vae" in k:
            parts["rh"][k.replace("rhand_vae.", "")] = v
        elif "hand_vae" in k:
            parts["lh"][k.replace("hand_vae.", "")] = v
        elif k.startswith("vae."):
            parts["body"][k.replace("vae.", "")] = v
    cfg = dict(down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, code_dim=512,
               output_emb_width=512)
    body = VQVae(nfeats=43, code_num=96, **cfg)
    lh = VQVae(nfeats=45, code_num=192, **cfg)
    rh = VQVae(nfeats=45, code_num=192, **cfg)
    body.load_state_dict(parts["body"], strict=True)
    lh.load_state_dict(parts["lh"], strict=True)
    rh.load_state_dict(parts["rh"], strict=True)
    return [m.eval() for m in (body, lh, rh)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/haipd/SLP/data/processed/madis/Phoenix_2014T")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tokenizer", default="/home/haipd/SLP/third_party/SOKE/checkpoints/tokenizer.ckpt")
    ap.add_argument("--stats", default="/home/haipd/SLP/third_party/SOKE/deps/stats")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--recon_npz", nargs="*", default=[],
                    help="label=path.npz of right-hand reconstructions (keys: name with / -> __); "
                         "scored as GT with only the right hand replaced")
    ap.add_argument("--only_rh", action="store_true", help="skip the slow full baselines")
    ap.add_argument("--out", default="/home/haipd/SLP/outputs/diagnostics/phoenix_oracles.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)

    mean = torch.load(f"{args.stats}/mean.pt", weights_only=True)
    std = torch.load(f"{args.stats}/std.pt", weights_only=True)
    mean, std = mean[36:], std[36:]
    mean = torch.cat([mean[:-20], mean[-10:]]).float()
    std = torch.cat([std[:-20], std[-10:]]).float()
    J = Joints(mean, std, dev)

    def load(split):
        ann = pickle.load(gzip.open(root / f"phoenix14t.{split}", "rb"))
        items = []
        for a in ann:
            p = root / "poses_video_level" / f"{a['name']}.pkl"
            if not p.exists():
                continue
            f = feats133(pickle.load(open(p, "rb")))
            if len(f) < 40 or len(f) > 400:  # MIN/MAX_MOTION_LEN of the protocol
                continue
            items.append((a["name"], a["text"], torch.from_numpy((f - mean.numpy()) / std.numpy())))
        return items

    train = load("train")
    test = load(args.split)
    if args.limit > 0:
        test = test[: args.limit]
    print(f"train={len(train)} {args.split}={len(test)}", flush=True)

    body_vq, lh_vq, rh_vq = [m.to(dev) for m in load_vae(args.tokenizer)]
    mean_pose = torch.zeros(133)  # normalised space => mean pose

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([t for _, t, _ in train])
    Xtr = vec.transform([t for _, t, _ in train])
    Xte = vec.transform([t for _, t, _ in test])
    nn_idx = (Xte @ Xtr.T).toarray().argmax(1)

    recon = {}
    for spec in args.recon_npz:
        label, path = spec.split("=", 1)
        recon[label] = np.load(path)

    usage = {"body": set(), "lh": set(), "rh": set()}
    agg = {}
    for i, (name, text, x) in enumerate(test):
        T = (x.shape[0] // 4) * 4
        x = x[:T]
        ref = J(x)
        xd = x.unsqueeze(0).to(dev)
        body_in = torch.cat([xd[..., :30], xd[..., 120:]], -1)
        cb, _ = body_vq.encode(body_in)
        cl, _ = lh_vq.encode(xd[..., 30:75])
        cr, _ = rh_vq.encode(xd[..., 75:120])
        usage["body"].update(cb.flatten().tolist())
        usage["lh"].update(cl.flatten().tolist())
        usage["rh"].update(cr.flatten().tolist())
        b, l, r = body_vq.decode(cb)[0], lh_vq.decode(cl)[0], rh_vq.decode(cr)[0]
        b, l, r = b.cpu(), l.cpu(), r.cpu()
        rec = torch.cat([b[:, :30], l, r, b[:, 30:]], -1)
        rh_only = x.clone(); rh_only[:, 75:120] = r.cpu()
        body_only = x.clone(); body_only[:, :30] = b[:, :30]; body_only[:, 120:] = b[:, 30:]
        hands_only = x.clone(); hands_only[:, 30:75] = l; hands_only[:, 75:120] = r
        cands = {"soke_tok_rh_only": rh_only, "soke_tok_body_only": body_only,
                 "soke_tok_hands_only": hands_only}
        for label, z in recon.items():
            g = x.clone(); g[:, 75:120] = torch.from_numpy(z[name.replace("/", "__")][:T])
            cands[f"{label}_rh_only"] = g
        if not args.only_rh:
            cands.update({
            "oracle_tok": rec,
            "mean_pose": mean_pose.expand(T, -1).clone(),
            "nn_text": train[nn_idx[i]][2],
            "gt_reversed": x.flip(0),
            })
        for k, g in cands.items():
            s = scores(J(g), ref)
            for m, v in s.items():
                agg.setdefault(k, {}).setdefault(m, []).append(float(v))
        if i % 50 == 0:
            print(i, {k: round(np.mean(v["body"]), 3) for k, v in agg.items()}, flush=True)

    res = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in agg.items()}
    res["_meta"] = {"split": args.split, "n": len(test),
                    "codebook_usage": {k: len(v) for k, v in usage.items()},
                    "mean_len_frames": float(np.mean([t[2].shape[0] for t in test]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

"""Project SOKE/MaDiS 133-D SMPL-X sequences to 2-D COCO-WholeBody-133 keypoints.

This is the input format of RTMPose-based SLT models (Uni-Sign). Pipeline: 133-D
(normalised) → denormalise → SMPL-X FK (144 joints, face contour) → frontal
orthographic projection (u = X, v = -Y; root orientation is zero, so the body
faces the camera) → COCO-WholeBody indices. Lower body / feet get confidence 0
(unused by Uni-Sign).

Sources, as in scripts/score_predictions.py: label=<MaDiS pred dir | npz>, @gt,
@mean_pose, @gt_reversed. Output: one npz per label, {name: [T,133,3]} (x, y, conf).

  cd third_party/MaDiS && .venv/bin/python /home/haipd/SLP/scripts/smplx_to_cocowb.py --source csl --split val \
      --out_dir /home/haipd/SLP/outputs/cocowb/csl_val @gt @mean_pose @gt_reversed madis=<pred_dir>
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "/home/haipd/SLP/src")
from mGPT.utils.human_models import get_coord  # noqa: E402
from slp.data.sign_smplx import LOADERS, feats133, load_stats, madis_crop  # noqa: E402

SHAPE = [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
         0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]


def cocowb_index_map():
    """COCO-WholeBody index -> SMPL-X (144-joint) index, for the points Uni-Sign uses (+ more)."""
    m = {0: 55, 1: 57, 2: 56, 3: 59, 4: 58, 5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21}  # body
    for k in range(68):  # face: iBUG-68 -> COCO 23..90
        m[23 + k] = 127 + k if k < 17 else 76 + (k - 17)
    def hand(wrist, base, tips):  # OpenPose hand order: wrist, thumb, index, middle, ring, pinky
        thumb, index, middle, pinky, ring = [base + 12, base + 13, base + 14], [base, base + 1, base + 2], \
            [base + 3, base + 4, base + 5], [base + 6, base + 7, base + 8], [base + 9, base + 10, base + 11]
        order = [wrist] + thumb + [tips[0]] + index + [tips[1]] + middle + [tips[2]] + ring + [tips[3]] + pinky + [tips[4]]
        return order
    for i, j in enumerate(hand(20, 25, [66, 67, 68, 69, 70])):
        m[91 + i] = j
    for i, j in enumerate(hand(21, 40, [71, 72, 73, 74, 75])):
        m[112 + i] = j
    return m


def load_source(path, split):
    p = Path(path)
    if p.is_dir():
        return {f.stem: pickle.load(open(f, "rb"))["feats_rst"].astype(np.float32) for f in p.glob("*.pkl")}
    z = np.load(p)
    return {k.split("__")[-1]: z[k] for k in z.files if not k.startswith("coarse::")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--source", default="csl", choices=["phoenix", "csl", "how2sign"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--stats", default="/home/haipd/SLP/third_party/SOKE/deps/stats")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args()
    roots = {"phoenix": "Phoenix_2014T", "csl": "CSL-Daily", "how2sign": "How2Sign"}
    rows = LOADERS[a.source](Path(f"/home/haipd/SLP/data/processed/madis/{roots[a.source]}"), a.split)
    if a.limit > 0:
        rows = rows[: a.limit]
    mean, std = load_stats(a.stats)
    mt, st = torch.from_numpy(mean).cuda(), torch.from_numpy(std).cuda()
    imap = cocowb_index_map()
    coco_idx = np.array(sorted(imap)); smplx_idx = np.array([imap[i] for i in coco_idx])

    def gt_feats(r):
        raw = feats133(pickle.load(open(r[1], "rb")))
        if r[2] is not None and r[2] > 24:
            raw = raw[np.linspace(0, len(raw) - 1, int(24 * len(raw) / r[2]), dtype=int)]
        return madis_crop((raw - mean) / std).astype(np.float32)

    def project(f):
        x = torch.from_numpy(f).cuda() * st + mt
        T = len(x)
        x = torch.cat([torch.zeros(T, 36, device="cuda"), x], -1)
        shape = torch.tensor(SHAPE, device="cuda").expand(T, -1)
        _, j = get_coord(x[:, :3], x[:, 3:66], x[:, 66:111], x[:, 111:156], x[:, 156:159], shape, x[:, 159:169],
                         return_vert=False)
        j = j.cpu().numpy()
        out = np.zeros((T, 133, 3), dtype=np.float32)
        out[:, coco_idx, 0] = j[:, smplx_idx, 0]
        out[:, coco_idx, 1] = -j[:, smplx_idx, 1]
        out[:, coco_idx, 2] = 1.0
        return out

    preds = {}
    for spec in a.sources:
        if not spec.startswith("@"):
            lab, path = spec.split("=", 1)
            preds[lab] = load_source(path, a.split)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = [s[1:] if s.startswith("@") else s.split("=", 1)[0] for s in a.sources]
    res = {lab: {} for lab in labels}
    for r in rows:
        name = Path(r[0]["name"]).name
        gt = None
        for lab in labels:
            if lab in preds:
                if name not in preds[lab]:
                    continue
                f = preds[lab][name]
            else:
                gt = gt_feats(r) if gt is None else gt
                f = {"gt": gt, "mean_pose": np.zeros_like(gt), "gt_reversed": gt[::-1].copy()}[lab]
            res[lab][name] = project(f)
    for lab, d in res.items():
        np.savez(out_dir / f"{lab}.npz", **d)
        print(lab, len(d), flush=True)
    with open(out_dir / "texts.pkl", "wb") as h:
        pickle.dump({Path(r[0]["name"]).name: r[0]["text"] for r in rows}, h)


if __name__ == "__main__":
    main()

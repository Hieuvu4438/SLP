"""SMPL-X sign-language datasets in the SOKE/MaDiS 133-D representation.

Official splits only (Phoenix-2014T train/dev/test, CSL-Daily train/val/test,
How2Sign train/val/test); nothing is re-split. Data must first be packed with
scripts/madis_build_{phoenix,csl,how2sign}.py (see data/README.md).

Split.inputs  : list of dicts {name, text, src, lang}
Split.targets : lazy sequence of normalised [T,133] float32 arrays (MaDiS crop rule)
"""

from __future__ import annotations

import csv
import gzip
import pickle
from pathlib import Path

import numpy as np

from slp.data.base import DatasetSplits, Split

SMPLX_KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
              "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr")
# 133-D layout: upper body 0:30 | lhand 30:75 | rhand 75:120 | jaw 120:123 | expr 123:133
PARTS = {"body": list(range(30)) + list(range(120, 133)),
         "lhand": list(range(30, 75)), "rhand": list(range(75, 120))}


def feats133(poses: dict) -> np.ndarray:
    """Stacked SMPL-X params -> SOKE 133-D feature (drop root+lower body and shape)."""
    f = np.concatenate([np.asarray(poses[k]) for k in SMPLX_KEYS], 1)[:, 36:]
    return np.concatenate([f[:, :-20], f[:, -10:]], 1).astype(np.float32)


def madis_crop(f: np.ndarray, lo: int = 40, hi: int = 400, unit: int = 4) -> np.ndarray:
    """MaDiS evaluation crop: resample into [lo, hi] frames, else centre-crop to a multiple of unit."""
    n = len(f)
    if n < lo:
        return f[np.linspace(0, n - 1, lo, dtype=int)]
    if n > hi:
        return f[np.linspace(0, n - 1, hi, dtype=int)]
    u = (n // unit) * unit
    s = (n - u) // 2
    return f[s:s + u]


def load_stats(stats_dir: str) -> tuple[np.ndarray, np.ndarray]:
    import torch  # stats are torch tensors in the SOKE release
    out = []
    for n in ("mean", "std"):
        t = torch.load(Path(stats_dir) / f"{n}.pt", weights_only=True)[36:]
        out.append(torch.cat([t[:-20], t[-10:]]).float().numpy())
    return out[0], out[1]


# How2Sign clips whose SMPL-X fitting failed; SOKE drops them from every split
# (third_party/SOKE/mGPT/data/humanml/dataset_t2m_token.py, ``bad_how2sign_ids``).
SOKE_BAD_HOW2SIGN_IDS = frozenset("""
0DU7wWLK-QU_0-8-rgb_front 0ICZi26jdaQ_28-5-rgb_front 0vNfEYst_tQ_11-8-rgb_front 13X0vEMNm7M_8-5-rgb_front
14weIYQswlE_23-8-rgb_front 1B56XMJ-j1Q_13-8-rgb_front 1P0oKY4FNyI_0-8-rgb_front 1dpRaxOTfZs_0-8-rgb_front
1ei1kVTw23A_29-8-rgb_front 1spCnuBmWYk_0-8-rgb_front 2-vXO7MMLJc_0-5-rgb_front 21PbS6wnHtY_0-5-rgb_front
3tyfxL2wO-M_0-8-rgb_front BpYDl3AO4B8_0-1-rgb_front CH7AviIr0-0_14-8-rgb_front CJ8RyW9pzKU_6-8-rgb_front
D0T7ho08Q3o_25-2-rgb_front Db5SUQvNsHc_18-1-rgb_front Eh697LCFjTw_0-3-rgb_front F-p1IdedNbg_23-8-rgb_front
aUBQCNegrYc_13-1-rgb_front cvn7htBA8Xc_9-8-rgb_front czBrBQgZIuc_19-5-rgb_front dbSAB8F8GYc_11-9-rgb_front
doMosV-zfCI_7-2-rgb_front dvBdWGLzayI_10-8-rgb_front eBrlZcccILg_26-3-rgb_front 39FN42e41r0_17-1-rgb_front
a4Nxq0QV_WA_9-3-rgb_front fzrJBu2qsM8_11-8-rgb_front g3Cc_1-V31U_12-3-rgb_front""".split())


def fps24_indices(n: int, fps: float, rule: str = "madis") -> np.ndarray:
    """Frame indices that resample an n-frame clip recorded at fps (> 24) to 24 fps.

    madis: np.linspace(0, n-1, count) (MaDiS sign_loader); soke: floor(i * n / count)
    (SOKE load_data.sample). count = int(24 * n / fps) in both."""
    count = int(24 * n / fps)
    if rule == "soke":
        ss = float(n) / count
        return np.array([int(np.floor(i * ss)) for i in range(count)], dtype=int)
    if rule != "madis":
        raise ValueError(rule)
    return np.linspace(0, n - 1, count, dtype=int)


class LazyPoses:
    """Sequence of pose arrays loaded on access (How2Sign does not fit in RAM comfortably)."""

    def __init__(self, files, mean, std, crop=True, fps=None, subsample="madis"):
        self.files, self.mean, self.std, self.crop, self.fps = files, mean, std, crop, fps
        self.subsample = subsample

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = feats133(pickle.load(open(self.files[i], "rb")))
        if self.fps is not None and self.fps[i] > 24:  # How2Sign: subsample to 24 fps
            f = f[fps24_indices(len(f), self.fps[i], self.subsample)]
        f = (f - self.mean) / self.std
        return madis_crop(f) if self.crop else f


def _phoenix(root, split):
    tag = {"train": "train", "val": "dev", "test": "test"}[split]
    rows = []
    for a in pickle.load(gzip.open(root / f"phoenix14t.{tag}", "rb")):
        p = root / "poses_video_level" / f"{a['name']}.pkl"
        if p.exists():
            rows.append(({"name": a["name"], "text": a["text"], "src": "phoenix", "lang": "de_DGS",
                          "gloss": a.get("gloss")}, p, None))
    return rows


def _csl(root, split):
    rows = []
    for a in pickle.load(gzip.open(root / f"csl_clean.{split}", "rb")):
        p = root / "poses_video_level" / f"{a['name']}.pkl"
        if p.exists():
            rows.append(({"name": a["name"], "text": a["text"], "src": "csl", "lang": "zh_CSL",
                          "gloss": a.get("gloss")}, p, None))
    return rows


def _how2sign(root, split, exclude=frozenset()):
    rows = []
    fn = root / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"
    with open(fn, newline="") as h:
        for r in csv.DictReader(h):
            name = r["SENTENCE_NAME"]
            p = root / "poses_video_level" / split / "poses" / f"{name}.pkl"
            if p.exists() and name not in exclude:
                rows.append(({"name": name, "text": r["SENTENCE"], "src": "how2sign", "lang": "en_ASL"},
                             p, float(r["fps"])))
    return rows


LOADERS = {"phoenix": _phoenix, "csl": _csl, "how2sign": _how2sign}


def load_sign_smplx(cfg: dict) -> DatasetSplits:
    d = cfg["data"]
    root = Path(d["root"])
    mean, std = load_stats(d["stats"])
    splits = {}
    # protocol: "madis" (default) or "soke" (How2Sign: SOKE's 24-fps subsampling and failed-fit
    # exclusions; identical to madis for Phoenix / CSL)
    protocol = d.get("protocol", "madis")
    if protocol not in ("madis", "soke"):
        raise ValueError(f"unknown data.protocol {protocol!r}")
    for split in ("train", "val", "test"):
        if d["source"] == "how2sign":
            rows = _how2sign(root, split, SOKE_BAD_HOW2SIGN_IDS if protocol == "soke" else frozenset())
        else:
            rows = LOADERS[d["source"]](root, split)
        fps = [r[2] for r in rows] if d["source"] == "how2sign" else None
        splits[split] = Split([r[0] for r in rows],
                              LazyPoses([r[1] for r in rows], mean, std, fps=fps, subsample=protocol),
                              meta={"source": d["source"], "split": split})
    return DatasetSplits(splits["train"], splits["val"], splits["test"],
                         info={"nfeats": 133, "parts": PARTS, "source": d["source"]})

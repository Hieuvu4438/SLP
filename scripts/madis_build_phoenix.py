"""Pack SOKE per-frame Phoenix-2014T SMPL-X pickles into MaDiS video-level files.

Input : data/raw/soke/Phoenix_2014T/{train,dev,test}/<name>/imagesXXXX.pkl
Output: data/processed/madis/Phoenix_2014T/poses_video_level/<split>/<name>.pkl
        (dict of stacked arrays, one key per SMPL-X parameter) plus copies of
        the phoenix14t.{train,dev,test} annotation files.
Raw data is never modified.
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np

KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
        "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/soke/Phoenix_2014T")
    ap.add_argument("--dst", default="data/processed/madis/Phoenix_2014T")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    for split in ("train", "dev", "test"):
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f"phoenix14t.{split}", dst / f"phoenix14t.{split}")
        out_dir = dst / "poses_video_level" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for vid in sorted((src / split).iterdir()):
            out = out_dir / f"{vid.name}.pkl"
            if out.exists():
                continue
            frames = sorted(vid.glob("*.pkl"))
            if not frames:
                continue
            per = [pickle.load(open(f, "rb")) for f in frames]
            packed = {k: np.stack([p[k] for p in per]).astype(np.float32) for k in KEYS}
            with open(out, "wb") as h:
                pickle.dump(packed, h)
            n += 1
        print(split, "packed", n)


if __name__ == "__main__":
    main()

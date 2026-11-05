"""Stream SOKE CSL-Daily per-frame SMPL-X fittings (zip) into MaDiS video-level pickles.

Input : data/raw/soke/_downloads/csl_fit/csl-daily_pose.zip  (csl-daily_pose/<name>/<frame>.pkl)
Output: data/processed/madis/CSL-Daily/poses_video_level/<name>.pkl  + csl_clean.{train,val,test}
The raw archive is read-only; nothing is extracted to disk.
"""

import argparse
import pickle
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
        "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/soke/_downloads/csl_fit/csl-daily_pose.zip")
    ap.add_argument("--splits", default="data/raw/soke/CSL-Daily")
    ap.add_argument("--dst", default="data/processed/madis/CSL-Daily")
    a = ap.parse_args()
    dst = Path(a.dst)
    out = dst / "poses_video_level"
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        shutil.copy(Path(a.splits) / f"csl_clean.{s}", dst / f"csl_clean.{s}")
    zf = zipfile.ZipFile(a.zip)
    groups = defaultdict(list)
    for info in zf.infolist():
        parts = info.filename.split("/")
        if len(parts) == 3 and parts[2].endswith(".pkl"):
            groups[parts[1]].append(info.filename)
    print("videos", len(groups), flush=True)
    for n, (name, files) in enumerate(sorted(groups.items())):
        target = out / f"{name}.pkl"
        if target.exists():
            continue
        frames = [pickle.loads(zf.read(f)) for f in sorted(files)]
        packed = {k: np.stack([fr[k] for fr in frames]).astype(np.float32) for k in KEYS}
        with open(target, "wb") as h:
            pickle.dump(packed, h)
        if n % 2000 == 0:
            print(n, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

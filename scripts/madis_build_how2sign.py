"""Stream SOKE How2Sign (Neural-Sign-Actors) per-frame SMPL-X fittings into MaDiS layout.

Input : data/raw/soke/_downloads/h2s_fit/{train,val,test}_poses.zip  (poses/<name>/<name>_<idx>_3D.pkl)
        data/raw/soke/How2Sign/how2sign_realigned_<split>_preprocessed_fps.csv
Output: data/processed/madis/How2Sign/poses_video_level/<split>/poses/<name>.pkl
        data/processed/madis/How2Sign/<split>/re_aligned/how2sign_realigned_<split>_preprocessed_fps.csv
Raw archives are read-only; frames are stacked at their native fps (MaDiS subsamples to 24 fps).
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
    ap.add_argument("--src", default="data/raw/soke/_downloads/h2s_fit")
    ap.add_argument("--csv", default="data/raw/soke/How2Sign")
    ap.add_argument("--dst", default="data/processed/madis/How2Sign")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    a = ap.parse_args()
    for split in a.splits:
        csv = f"how2sign_realigned_{split}_preprocessed_fps.csv"
        (Path(a.dst) / split / "re_aligned").mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(a.csv) / csv, Path(a.dst) / split / "re_aligned" / csv)
        out = Path(a.dst) / "poses_video_level" / split / "poses"
        out.mkdir(parents=True, exist_ok=True)
        zf = zipfile.ZipFile(Path(a.src) / f"{split}_poses.zip")
        groups = defaultdict(list)
        for info in zf.infolist():
            parts = info.filename.split("/")
            if len(parts) == 3 and parts[2].endswith("_3D.pkl"):
                groups[parts[1]].append(info.filename)
        print(split, "videos", len(groups), flush=True)
        for n, (name, files) in enumerate(sorted(groups.items())):
            target = out / f"{name}.pkl"
            if target.exists():
                continue
            files = sorted(files, key=lambda f: int(f.rsplit("_", 2)[-2]))
            frames = [pickle.loads(zf.read(f)) for f in files]
            packed = {k: np.stack([fr[k] for fr in frames]).astype(np.float32) for k in KEYS}
            with open(target, "wb") as h:
                pickle.dump(packed, h)
            if n % 5000 == 0:
                print(split, n, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

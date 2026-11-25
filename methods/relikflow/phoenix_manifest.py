"""Convert the repository's packed PHOENIX train/dev poses for RelIKFlow."""

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

import numpy as np

from slp.data.sign_smplx import feats133

from .data import digest, sha256
from .runtime import npz_write, write_atomic


def export(source: Path, destination: Path):
    """Keep official clip splits and use broadcasts as groups within train for OOF."""
    rows = []
    for split in ("train", "dev"):
        # The repository's trusted SOKE/MaDiS export uses gzipped pickle for annotations.
        with gzip.open(source / f"phoenix14t.{split}", "rb") as handle:
            annotations = pickle.load(handle)
        for annotation in annotations:
            clip_id = annotation["name"]
            raw = source / "poses_video_level" / f"{clip_id}.pkl"
            with raw.open("rb") as handle:
                pose = feats133(pickle.load(handle))
            if pose.ndim != 2 or pose.shape[1] != 133 or len(pose) == 0:
                raise ValueError(f"Invalid SOKE-133 input for {clip_id}")
            path = destination.parent / "poses" / f"{digest(clip_id)}.npz"
            npz_write(path, pose133=pose.astype(np.float32, copy=False))
            broadcast = re.sub(r"-\d+$", "", Path(clip_id).name)
            rows.append(
                {
                    "clip_id": clip_id,
                    "video_id": clip_id,
                    "oof_group": broadcast,
                    "corpus": "phoenix_2014t",
                    "split": split,
                    "source_language": "de_DE",
                    "target_sign_language": "DGS",
                    "text": annotation["text"],
                    "pose_path": str(path.relative_to(destination.parent)),
                    "pose_layout": "soke133_physical",
                    "pose_fit_id": "soke_phoenix_packed",
                    "fps": 25.0,
                    "timebase_source": "phoenix_2014t_release",
                    "num_frames": len(pose),
                    "annotation_num_frames": int(annotation["num_frames"]),
                    "sha256": sha256(path),
                }
            )
        print(f"PHOENIX {split}: {len(annotations)} clips", flush=True)
    write_atomic(
        destination,
        lambda out: out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)

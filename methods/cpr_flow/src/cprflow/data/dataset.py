from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from cprflow.data.cache import file_hash
from cprflow.data.manifest import read_manifest


class MotionDataset(Dataset):
    def __init__(self, manifest, split, datasets=None, max_frames=None, verify_hash=True):
        self.rows = read_manifest(manifest, split)
        if datasets:
            self.rows = [r for r in self.rows if r["dataset_id"] in datasets]
        if max_frames is not None:
            excluded = [r for r in self.rows if r["frames"] > max_frames]
            if excluded:
                import json

                ledger = Path(manifest).parent / "long_train_exclusions.jsonl"
                if split != "train" or not ledger.is_file():
                    raise ValueError("Length filtering requires explicit train exclusion ledger")
                recorded = {
                    json.loads(line)["uid"]
                    for line in ledger.read_text().splitlines()
                    if line.strip()
                }
                if not {r["uid"] for r in excluded} <= recorded:
                    raise ValueError("Unrecorded long train exclusions")
            self.rows = [r for r in self.rows if r["frames"] <= max_frames]
        self.verify_hash = verify_hash
        self._verified_signatures = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["processed_path"])
        if self.verify_hash:
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if self._verified_signatures.get(row["uid"]) != signature:
                if file_hash(path) != row["processed_sha256"]:
                    raise ValueError(f"Processed hash mismatch: {row['uid']}")
                self._verified_signatures[row["uid"]] = signature
        with np.load(path, allow_pickle=False) as data:
            x, times = data["motion"], data["timestamps"]
        if x.shape != (row["frames"], 296) or times.shape != (len(x),):
            raise ValueError(f"Processed shape mismatch: {row['uid']}")
        return {**row, "x": x, "timestamps": times}

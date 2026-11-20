"""Write a fixed-length table {name: frames} for ``method.sampler.lengths_file`` (task C1).

    # reference (GT) lengths of the val split: an oracle-length protocol that removes the length
    # head from every comparison (same lengths for every arm)
    python -m methods.signmflow_plus.tools.lengths --data phoenix_smplx --out outputs/signmflow_plus/lengths/phoenix_val_gt.json

    # lengths of an existing run (e.g. the baseline's predicted lengths), no GT access
    python -m methods.signmflow_plus.tools.lengths --from-npz outputs/runs/<run>/predictions.npz --out <file>.json

Only ``--split val`` is accepted for GT lengths; the test split stays untouched (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def lengths_from_npz(path: str | Path) -> dict[str, int]:
    """Keys stay in the predictions.npz spelling ('/' -> '__'); read_lengths accepts both."""
    with np.load(path) as z:
        return {k: int(len(z[k])) for k in z.files}


def lengths_from_split(split) -> dict[str, int]:
    return {str(x["name"]): int(len(t)) for x, t in zip(split.inputs, split.targets)}


def main(argv=None):
    sys.path.insert(0, str(ROOT / "src"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", help="dataset config name (GT lengths of the val split)")
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--from-npz", help="predictions.npz of an existing run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if bool(a.data) == bool(a.from_npz):
        ap.error("give exactly one of --data / --from-npz")
    if a.from_npz:
        table = lengths_from_npz(a.from_npz)
    else:
        from slp import config as config_lib
        from slp.data import load_dataset

        table = lengths_from_split(load_dataset(config_lib.compose("signmflow_plus", a.data)).val)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(out, "w"))
    print(f"{len(table)} lengths -> {out} (mean {np.mean(list(table.values())):.1f} frames)")


if __name__ == "__main__":
    main()

"""Bound periodic snapshots while preserving best, last, and scientific milestones."""

import json
import re
from pathlib import Path

PATTERN = re.compile(r"^step_(\d+)_ema\.pt$")


def prune_checkpoints(directory, keep_steps, keep_recent=1):
    if keep_recent < 1:
        raise ValueError("At least one recent periodic checkpoint must be retained")
    directory = Path(directory)
    snapshots = sorted(
        (int(m.group(1)), p)
        for p in directory.glob("step_*_ema.pt")
        if (m := PATTERN.fullmatch(p.name))
    )
    protected = set(keep_steps) | {step for step, _ in snapshots[-keep_recent:]}
    removed = []
    for step, path in snapshots:
        if step not in protected:
            path.unlink()
            Path(str(path) + ".json").unlink(missing_ok=True)
            removed.append(path.name)
    if removed:
        with (directory / "retention.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"removed": removed, "protected_steps": sorted(protected)}) + "\n"
            )
    return removed

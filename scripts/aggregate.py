#!/usr/bin/env python
"""Aggregate final metrics over seeds -> mean ± std table.

    python scripts/aggregate.py --name main_table [--split test] [--out results/main_table]

Writes <out>.csv and <out>.md. Runs from a dirty git tree are flagged with '*'.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slp.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402


def collect(experiment: str, split: str | None):
    groups = defaultdict(list)
    for mfile in sorted((RUNS_DIR / experiment).glob("*/*/metrics.json")):
        meta = json.loads((mfile.parent / "meta.json").read_text())
        if split and meta["eval_split"] != split:
            continue
        key = (meta["method"], meta["data"], meta["eval_split"])
        groups[key].append((meta, json.loads(mfile.read_text())))
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="experiment name")
    p.add_argument("--split", default=None, choices=["val", "test"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    groups = collect(args.name, args.split)
    if not groups:
        sys.exit(f"No finished runs under {RUNS_DIR / args.name}")
    metric_names = sorted({m for runs in groups.values() for _, r in runs for m in r})

    rows = []
    for (method, data, split), runs in sorted(groups.items()):
        row = {"method": method, "data": data, "split": split, "n_seeds": len(runs),
               "seeds": ",".join(str(m["seed"]) for m, _ in runs),
               "dirty": any(m["git"]["dirty"] for m, _ in runs),
               "commits": ",".join(sorted({(m["git"]["commit"] or "?")[:8] for m, _ in runs}))}
        for name in metric_names:
            vals = [r[name] for _, r in runs if name in r]
            row[f"{name}_mean"] = statistics.mean(vals)
            row[f"{name}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)

    out = Path(args.out) if args.out else RESULTS_DIR / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    header = ["method", "data", "split", "n"] + metric_names
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        flag = "*" if r["dirty"] else ""
        cells = [r["method"] + flag, r["data"], r["split"], str(r["n_seeds"])]
        cells += [f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}" for m in metric_names]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n`*` = at least one run launched from a dirty git tree (not reportable).")
    table = "\n".join(lines)
    out.with_suffix(".md").write_text(f"# {args.name}\n\n{table}\n")
    print(table)
    print(f"\nSaved {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

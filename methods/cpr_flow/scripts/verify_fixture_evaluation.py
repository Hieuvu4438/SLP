#!/usr/bin/env python
"""Evaluate all three synthetic fixture routes with the actual pinned mT5 encoder."""

import argparse
import json
from pathlib import Path

import torch

from cprflow.cli.common import save_json
from cprflow.cli.evaluate import evaluate
from cprflow.evaluation.aggregate import aggregate_reports
from cprflow.sampling.generator import MotionGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    generator = MotionGenerator.from_checkpoint(args.checkpoint)
    if not generator.config["run"]["synthetic"]:
        raise ValueError("This verification script is only for synthetic=true fixtures")
    output = Path(args.output)
    reports = [
        evaluate(generator, dataset, output=output / f"{dataset}.json")
        for dataset in generator.config["data"]["datasets"]
    ]
    save_json(output / "macro.json", aggregate_reports(reports))
    summary = {
        "synthetic": True,
        "real_data_evaluated": False,
        "mt5_encoder_executed": True,
        "per_dataset": {
            r["dataset"]: {
                "requested": r["num_requested"],
                "successful": r["num_successful"],
                "coverage": r["coverage"],
            }
            for r in reports
        },
        "reports": str(output.resolve()),
        "unavailable_metrics": reports[0]["unavailable_metrics"],
    }
    save_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

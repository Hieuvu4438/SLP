import argparse
import json
from pathlib import Path

from cprflow.cli.common import print_json, save_json
from cprflow.evaluation.aggregate import aggregate_reports


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate compatible per-corpus controlled reports"
    )
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate_reports([json.loads(Path(p).read_text()) for p in args.reports])
    save_json(args.output, result)
    print_json(result)


if __name__ == "__main__":
    main()

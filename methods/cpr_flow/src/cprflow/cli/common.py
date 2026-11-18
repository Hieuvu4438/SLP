import argparse
import json
from pathlib import Path

from cprflow.config import load_config


def config_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    return parser


def resolved(args):
    return load_config(args.config, args.set)


def print_json(obj):
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False, allow_nan=False))

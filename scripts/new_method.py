#!/usr/bin/env python
"""Scaffold a new method folder from methods/_template.

    python scripts/new_method.py my_new_idea
"""

from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "methods"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    name = sys.argv[1]
    if not name.isidentifier() or name.startswith("_"):
        sys.exit(f"{name!r} must be a snake_case Python identifier not starting with '_'")
    dst = METHODS / name
    if dst.exists():
        sys.exit(f"{dst} already exists")
    shutil.copytree(METHODS / "_template", dst, ignore=shutil.ignore_patterns("__pycache__"))
    for f in dst.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".yaml"}:
            f.write_text(
                f.read_text().replace("__NAME__", name).replace("__DATE__", str(date.today()))
            )
    print(f"Created {dst.relative_to(ROOT)}/ - implement method.py, then:\n"
          f"  python scripts/run.py --method {name} --name debug")


if __name__ == "__main__":
    main()

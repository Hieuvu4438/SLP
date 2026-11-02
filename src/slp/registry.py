"""Discover methods: every ``methods/<name>/`` folder with a ``method.py``
exposing a class ``Method`` (subclass of BaseMethod) is a runnable method.
Folders starting with ``_`` (e.g. ``_template``) are ignored.
"""

from __future__ import annotations

import importlib
import re
import sys

from slp.method import BaseMethod
from slp.paths import METHODS_DIR, PROJECT_ROOT


def _ensure_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def list_methods() -> list[str]:
    return sorted(
        p.name
        for p in METHODS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "method.py").exists()
    )


def method_status(name: str) -> str:
    readme = METHODS_DIR / name / "README.md"
    if readme.exists():
        m = re.search(r"^status:\s*(\S+)", readme.read_text(), re.MULTILINE)
        if m:
            return m.group(1)
    return "unknown"


def get_method(name: str) -> type[BaseMethod]:
    if name not in list_methods():
        raise KeyError(f"Unknown method {name!r}. Available: {list_methods()}")
    _ensure_root_on_path()
    module = importlib.import_module(f"methods.{name}.method")
    cls = getattr(module, "Method", None)
    if cls is None or not issubclass(cls, BaseMethod):
        raise TypeError(f"methods/{name}/method.py must define `class Method(BaseMethod)`")
    return cls

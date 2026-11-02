"""Config composition.

Resolution order (later wins):
    configs/base.yaml
    -> configs/data/<data.name>.yaml
    -> methods/<method>/config.yaml
    -> CLI overrides  (key.sub=value)

The fully resolved config is saved into every run directory.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from slp.paths import CONFIGS_DIR, METHODS_DIR


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_overrides(items: list[str]) -> dict:
    """Turn ["a.b=1", "c=[1,2]"] into a nested dict. Values parsed as YAML."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def compose(method: str, data: str | None = None, overrides: list[str] | None = None) -> dict:
    cli = parse_overrides(overrides or [])
    if data is not None:
        cli = deep_merge(cli, {"data": {"name": data}})

    base = load_yaml(CONFIGS_DIR / "base.yaml")
    method_path = METHODS_DIR / method / "config.yaml"
    method_cfg = load_yaml(method_path) if method_path.exists() else {}

    data_name = (
        get(cli, "data.name") or get(method_cfg, "data.name") or get(base, "data.name")
    )
    if not data_name:
        raise ValueError("No dataset specified (data.name).")
    data_path = CONFIGS_DIR / "data" / f"{data_name}.yaml"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data config: {data_path}")

    cfg = base
    for layer in (load_yaml(data_path), method_cfg, cli):
        cfg = deep_merge(cfg, layer)
    cfg["method_name"] = method
    return cfg


def dump(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

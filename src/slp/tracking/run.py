"""Run directory + provenance.

outputs/runs/<experiment>/<method>/seed<k>_<timestamp>/
    config.yaml            fully resolved config
    meta.json              git commit/branch/dirty, command, host, python, time, split
    code.diff              uncommitted changes at launch (if any)
    log.txt                full log
    metrics_history.jsonl  per-step metrics logged by the method (training curves)
    metrics.json           FINAL metrics from slp.evaluation
    checkpoints/
"""

from __future__ import annotations

import json
import platform
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path

from slp import config as config_lib
from slp.paths import RUNS_DIR
from slp.utils.git import git_diff, git_info
from slp.utils.logging import get_logger


@dataclass
class RunContext:
    run_dir: Path
    seed: int
    logger: Logger

    @property
    def ckpt_dir(self) -> Path:
        p = self.run_dir / "checkpoints"
        p.mkdir(exist_ok=True)
        return p

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """For training curves / diagnostics (NOT the final reported numbers)."""
        rec = {"step": step, "time": time.time(), **metrics}
        with open(self.run_dir / "metrics_history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.logger.info(f"step={step} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

    def save_final_metrics(self, metrics: dict) -> None:
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


def create_run(cfg: dict, seed: int, experiment: str, split: str) -> RunContext:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = RUNS_DIR / experiment / cfg["method_name"] / f"seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = {**cfg, "seed": seed, "experiment": experiment, "eval_split": split}
    config_lib.dump(cfg, run_dir / "config.yaml")

    git = git_info()
    meta = {
        "experiment": experiment,
        "method": cfg["method_name"],
        "data": cfg["data"]["name"],
        "seed": seed,
        "eval_split": split,
        "git": git,
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if git["dirty"]:
        (run_dir / "code.diff").write_text(git_diff())

    logger = get_logger(f"run.{run_dir.name}", run_dir / "log.txt")
    logger.info(f"Run dir: {run_dir}")
    if git["dirty"]:
        logger.warning("Working tree is dirty - diff saved to code.diff")
    return RunContext(run_dir=run_dir, seed=seed, logger=logger)

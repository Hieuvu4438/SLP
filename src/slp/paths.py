"""Canonical project paths. Use these instead of hard-coded relative paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
METHODS_DIR = PROJECT_ROOT / "methods"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_ACCESS_LOG = RESULTS_DIR / "test_access_log.jsonl"

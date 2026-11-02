import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_script(tmp_path, monkeypatch):
    """Import scripts/run.py with outputs redirected to a temp dir."""
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import slp.tracking.run as tracking_run

    monkeypatch.setattr(tracking_run, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(mod, "TEST_ACCESS_LOG", tmp_path / "test_access_log.jsonl")
    return mod

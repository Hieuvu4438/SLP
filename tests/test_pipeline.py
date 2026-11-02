import json


def _runs(tmp_path):
    return sorted((tmp_path / "runs").glob("*/*/*/metrics.json"))


def test_end_to_end_and_artifacts(run_script, tmp_path):
    run_script.main(["--method", "baseline_majority", "--data", "toy",
                     "--name", "t", "--seeds", "0,1"])
    runs = _runs(tmp_path)
    assert len(runs) == 2
    d = runs[0].parent
    for f in ("config.yaml", "meta.json", "log.txt", "metrics.json"):
        assert (d / f).exists(), f
    meta = json.loads((d / "meta.json").read_text())
    assert meta["eval_split"] == "val" and "commit" in meta["git"]


def test_same_seed_is_deterministic(run_script, tmp_path):
    for _ in range(2):
        run_script.main(["--method", "baseline_majority", "--name", "det", "--seeds", "3"])
    a, b = (json.loads(p.read_text()) for p in _runs(tmp_path))
    assert a == b


def test_test_split_is_logged(run_script, tmp_path):
    run_script.main(["--method", "baseline_majority", "--name", "t", "--split", "test",
                     "--allow-dirty"])
    log = (tmp_path / "test_access_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 1 and json.loads(log[0])["method"] == "baseline_majority"

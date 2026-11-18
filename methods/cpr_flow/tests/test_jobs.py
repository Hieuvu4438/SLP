"""Real subprocess tests of admission, dependency ordering, retries and live recovery."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cprflow.jobs.runner import Runner, live
from cprflow.jobs.schema import atomic_json, source_digest, validate_plan
from cprflow.jobs.store import Store
from cprflow.training.retention import prune_checkpoints


def payload(tmp, name, duration=0.2, fail_once=False):
    script = f"""
import fcntl,json,time
from pathlib import Path
root=Path({str(tmp)!r}); name={name!r}
def event(phase):
 with (root/'events.jsonl').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX); f.write(json.dumps({{'name':name,'phase':phase,'time':time.time()}})+'\\n')
event('start')
time.sleep({duration})
if {fail_once!r} and not (root/(name+'.retry')).exists():
 (root/(name+'.retry')).touch();event('end');raise SystemExit(1)
(root/(name+'.done')).write_text(json.dumps({{'status':'complete'}}))
event('end')
"""
    return {
        "id": name,
        "argv": [sys.executable, "-c", script],
        "deps": [],
        "gates": [],
        "outputs": [{"path": str(tmp / (name + ".done")), "equals": {"status": "complete"}}],
        "slots": 1,
        "gpu_gib": 0,
        "disk_gib": 0,
        "locks": [],
        "max_attempts": 3,
        "retry_seconds": 0.05,
        "priority": 0,
    }


def plan(tmp, jobs):
    root = Path(__file__).parents[1]
    value = {
        "version": 1,
        "id": "test",
        "root": str(root),
        "state_dir": str(tmp / "state"),
        "global_lock": str(tmp / "global.lock"),
        "max_concurrent": 2,
        "poll_seconds": 0.05,
        "disk": {
            "min_free_gib": 0,
            "budget_gib": 10,
            "tracked_paths": [str(tmp)],
            "scan_seconds": 0.1,
        },
        "environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        "jobs": jobs,
        "source_digest": source_digest(root),
        "config_hashes": {},
        "scope": "test",
        "seeds": [17],
    }
    path = tmp / "plan.json"
    atomic_json(path, value)
    return path, value


def test_dependency_concurrency_retry_and_two_process_hard_limit(tmp_path):
    a, b = payload(tmp_path, "a", 0.5), payload(tmp_path, "b", 0.5)
    c = payload(tmp_path, "c", 0.1, True)
    c["deps"] = ["a", "b"]
    path, value = plan(tmp_path, [a, b, c])
    # run with watch so retry backoff is honored, not treated as queue completion.
    result = subprocess.run(
        [sys.executable, "-m", "cprflow.cli.jobs", "run", "--plan", str(path), "--watch"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rows = Store(value).rows()
    assert all(r["state"] == "succeeded" for r in rows.values())
    assert rows["c"]["attempt"] == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    active = 0
    peak = 0
    for event in sorted(events, key=lambda e: e["time"]):
        active += 1 if event["phase"] == "start" else -1
        peak = max(peak, active)
    assert peak == 2 and active == 0
    assert min(e["time"] for e in events if e["name"] == "c") >= max(
        e["time"] for e in events if e["name"] in ("a", "b")
    )
    invalid = copy.deepcopy(value)
    invalid["max_concurrent"] = 3
    with pytest.raises(ValueError, match="two"):
        validate_plan(invalid)
    invalid = copy.deepcopy(value)
    invalid["jobs"][0]["deps"] = ["c"]
    with pytest.raises(ValueError, match="cycle"):
        validate_plan(invalid)


def test_missing_gate_waits_without_spawning_or_faking_success(tmp_path):
    a = payload(tmp_path, "gated")
    a["gates"] = [{"kind": "exists", "path": str(tmp_path / "human_review.json")}]
    path, value = plan(tmp_path, [a])
    Runner(path).run(watch=False)
    row = Store(value).rows()["gated"]
    assert row["state"] == "waiting" and row["attempt"] == 0
    assert "human_review.json" in row["reason"] and not (tmp_path / "events.jsonl").exists()


def test_restart_adopts_live_payload_without_duplicate_training(tmp_path):
    job = payload(tmp_path, "long_job", 2.0)
    path, value = plan(tmp_path, [job])
    first = subprocess.Popen(
        [sys.executable, "-m", "cprflow.cli.jobs", "run", "--plan", str(path), "--watch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (tmp_path / "events.jsonl").exists():
                break
            time.sleep(0.05)
        rows = Store(value).rows()
        pid = rows["long_job"]["pid"]
        token = rows["long_job"]["pid_start"]
        assert live(pid, token)
        first.kill()
        first.wait()
        assert live(pid, token)
        second = subprocess.run(
            [sys.executable, "-m", "cprflow.cli.jobs", "run", "--plan", str(path), "--watch"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert second.returncode == 0, second.stderr
        row = Store(value).rows()["long_job"]
        assert row["state"] == "succeeded" and row["attempt"] == 1
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert sum(e["phase"] == "start" for e in events) == 1
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()
        row = Store(value).rows()["long_job"]
        if live(row["pid"], row["pid_start"]):
            os.killpg(row["pid"], signal.SIGKILL)


def test_checkpoint_retention_preserves_resume_best_and_pinned_fork(tmp_path):
    for step in (5000, 10000, 50000, 55000, 100000, 200000, 205000):
        (tmp_path / f"step_{step}_ema.pt").write_bytes(b"fixture")
    for name in ("last.pt", "best_ema.pt", "best.pt"):
        (tmp_path / name).write_bytes(b"keep")
    removed = prune_checkpoints(tmp_path, [50000, 100000, 200000], 1)
    assert set(removed) == {"step_5000_ema.pt", "step_10000_ema.pt", "step_55000_ema.pt"}
    assert all(
        (tmp_path / name).exists()
        for name in (
            "last.pt",
            "best_ema.pt",
            "best.pt",
            "step_200000_ema.pt",
            "step_205000_ema.pt",
        )
    )


def test_foreign_live_claims_survive_scheduler_crash_and_bound_next_campaign(tmp_path):
    jobs = [payload(tmp_path, "old_a", 2.0), payload(tmp_path, "old_b", 2.0)]
    old_path, old_plan = plan(tmp_path, jobs)
    owner = subprocess.Popen(
        [sys.executable, "-m", "cprflow.cli.jobs", "run", "--plan", str(old_path), "--watch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (tmp_path / "events.jsonl").exists() and len(
                (tmp_path / "events.jsonl").read_text().splitlines()
            ) >= 2:
                break
            time.sleep(0.05)
        assert all(r["state"] == "running" for r in Store(old_plan).rows().values())
        owner.kill()
        owner.wait()
        second_root = tmp_path / "other"
        second_root.mkdir()
        new_path, new_plan = plan(
            second_root, [payload(tmp_path, "new_a", 0.1), payload(tmp_path, "new_b", 0.1)]
        )
        new_plan["global_lock"] = old_plan["global_lock"]
        atomic_json(new_path, new_plan)
        result = subprocess.run(
            [sys.executable, "-m", "cprflow.cli.jobs", "run", "--plan", str(new_path), "--watch"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        active = 0
        peak = 0
        for e in sorted(events, key=lambda e: e["time"]):
            active += 1 if e["phase"] == "start" else -1
            peak = max(peak, active)
        assert peak == 2 and active == 0
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        for r in Store(old_plan).rows().values():
            if live(r["pid"], r["pid_start"]):
                os.killpg(r["pid"], signal.SIGKILL)


def test_compile_full_dag_frozen_configs_shared_cache_and_resume(tmp_path):
    import shutil
    from cprflow.jobs.compiler import create_plan
    from cprflow.jobs.schema import load_plan
    from cprflow.config import load_config

    shutil.copytree(Path(__file__).parents[1] / "configs", tmp_path / "configs")
    path = create_plan(tmp_path, "standard_test", "standard", (17,), device="cpu")
    compiled = load_plan(path)
    jobs = {j["id"]: j for j in compiled["jobs"]}
    assert compiled["max_concurrent"] == 2
    assert {
        "train.ae",
        "train.duration",
        "train.teacher",
        "s17.train.c0",
        "s17.train.full",
        "smoke.cfm_first500",
        "smoke.cfm_resume1000",
        "eval.c0_s17.test.how2sign",
    } <= jobs.keys()
    assert jobs["smoke.cfm_resume1000"]["deps"] == ["smoke.cfm_first500"]
    assert jobs["s17.train.c0"]["warm_start"].endswith("step_200000_ema.pt")
    assert jobs["s17.train.full"]["warm_start"].endswith(
        "relation_joint3_s17/checkpoints/best_ema.pt"
    )
    cache_roots = set()
    for file in (Path(compiled["state_dir"]) / "configs").glob("*.yaml"):
        cfg = load_config(file)
        assert cfg["data"]["workers_per_rank"] == 0 and cfg["train"]["checkpoint_keep_recent"] == 1
        if not cfg["run"]["id"].startswith("smoke_real48"):
            cache_roots.add(cfg["paths"]["cache"])
    assert len(cache_roots) == 1
    assert all(
        "--final-test" in j["argv"]
        for j in compiled["jobs"]
        if j["id"].startswith("eval.") and ".test." in j["id"]
    )
    assert any(g["kind"] == "clean_git" for g in jobs["release.c0_s17"]["gates"])


def test_training_interrupt_during_validation_saves_exact_logical_boundary(tmp_path):
    import torch
    from cprflow.config import DEFAULTS
    from cprflow.training.engine import Trainer, seed_everything
    from cprflow.data.sampler import BalancedBatchSampler
    from cprflow.training.checkpoint import load_checkpoint

    cfg = copy.deepcopy(DEFAULTS)
    cfg["run"].update(out_dir=str(tmp_path), id="interrupted", synthetic=True)
    cfg["data"]["workers_per_rank"] = 0
    cfg["train"].update(
        effective_batch=2,
        microbatch_per_rank=1,
        accumulate=2,
        max_updates=20,
        checkpoint_every=100,
        dev_loss_every=2,
        precision="fp32",
    )

    class Dataset:
        rows = [
            {"uid": str(i), "frames": 2, "dataset_id": "csl_daily", "value": i / 10}
            for i in range(8)
        ]

        def __getitem__(self, i):
            return self.rows[i]

    ds = Dataset()
    seed_everything(17, True)
    trainer = Trainer(torch.nn.Linear(1, 1), cfg, {"fixture": True})
    sampler = BalancedBatchSampler(ds.rows, 2, 17)

    def loss(model, rows, step, logical_rows):
        x = torch.tensor([[r["value"]] for r in rows])
        return model(x).square().flatten(), {}

    def interrupted_validate(t):
        t.stop_requested = True
        torch.rand(20)
        raise InterruptedError("Simulated graceful stop during dev sampling")

    trainer.run(ds, sampler, loss, interrupted_validate)
    checkpoint = load_checkpoint(trainer.directory / "checkpoints/last.pt")
    assert trainer.step == checkpoint["successful_optimizer_step"] == 2
    assert sampler.cursor == 2 and checkpoint["sampler"]["cursor"] == 2
    assert (trainer.directory / "checkpoints/last.pt.json").is_file()


def test_storage_scan_does_not_follow_shared_external_symlinks(tmp_path):
    from cprflow.jobs.storage import directory_bytes, check_free_space

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 1000)
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "small").write_bytes(b"x" * 7)
    (owned / "alias").symlink_to(outside, target_is_directory=True)
    assert directory_bytes([owned]) == 7
    with pytest.raises(OSError, match="reserve"):
        check_free_space(tmp_path, min_gib=10**9)


def test_operator_requested_run_waives_manual_gates_without_faking_approval(tmp_path):
    import shutil
    from cprflow.jobs.compiler import create_plan
    from cprflow.jobs.schema import load_plan
    from cprflow.config import load_config
    from cprflow.training.gates import require_geometry_review

    shutil.copytree(Path(__file__).parents[1] / "configs", tmp_path / "configs")
    p = create_plan(tmp_path, "operator_run", "standard", (17,), device="cpu", operator_waived=True)
    compiled = load_plan(p)
    jobs = {j["id"]: j for j in compiled["jobs"]}
    assert "data.geometry_audit" not in jobs and "data.resolve_timing" in jobs
    assert jobs["gate.geometry"]["deps"] == ["data.canonicalize"]
    assert jobs["release.c0_s17"]["gates"][0]["kind"] == "source_snapshot"
    config = load_config(Path(compiled["state_dir"]) / "configs/s17.train.c0.yaml")
    result = require_geometry_review(config)
    assert result["status"] == "operator_waived" and result["approved"] is False
    assert config["data"]["source_profile"] == "signspark_body126_hands90_or96_wristdup_face56"

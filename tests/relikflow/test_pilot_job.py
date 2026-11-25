import json
import subprocess

import pytest

from scripts import relikflow_pilot_job as job


def test_gate_sequential_stages_and_resume(tmp_path, monkeypatch):
    output = tmp_path / "pilot"
    monkeypatch.setattr(job, "JOB_DIR", tmp_path / "locks")
    monkeypatch.setattr(job, "stages", lambda *_: [("first", ["one"]), ("second", ["two"])])
    readings = iter([11000, 24575, 24576, 12000, 32000])
    monkeypatch.setattr(job, "query_memory", lambda _: next(readings))
    checked = []

    def sleep(seconds):
        state = json.loads((output / "job_state.json").read_text())
        assert state["status"] == "waiting_for_vram"
        assert state["free_mib"] < job.MIN_FREE_MIB
        checked.append((state["stage"], state["free_mib"]))

    ran = []

    def run(command, env, lock, log_path):
        assert log_path.is_file()
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert "WORLD_SIZE" not in env
        if command == ["one"]:
            with pytest.raises(RuntimeError, match="Another RelIKFlow pilot"):
                job.run("GPU-test", tmp_path / "duplicate", 1)
        ran.append(command)

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(job.time, "sleep", sleep)
    monkeypatch.setattr(job, "run_stage", run)
    assert job.run("GPU-test", output, 1) == 0
    assert checked == [("first", 11000), ("first", 24575), ("second", 12000)]
    assert ran == [["one"], ["two"]]
    assert job.run("GPU-test", output, 1) == 0
    assert ran == [["one"], ["two"]]


def test_retry_cuda_oom_and_recheck_gpu_threshold(tmp_path, monkeypatch):
    output = tmp_path / "pilot"
    monkeypatch.setattr(job, "JOB_DIR", tmp_path / "locks")
    monkeypatch.setattr(job, "stages", lambda *_: [("first", ["one"]), ("second", ["two"])])
    readings = iter([27000, 21000, 28000])
    monkeypatch.setattr(job, "query_memory", lambda _: next(readings))
    states = []
    monkeypatch.setattr(
        job.time,
        "sleep",
        lambda _: states.append(json.loads((output / "job_state.json").read_text())["status"]),
    )
    attempts = []

    def run(command, env, lock, log_path):
        attempts.append(command)
        if len(attempts) == 1:
            with log_path.open("a") as log:
                log.write("RelIKFlow: CUDA out of memory\n")
            raise subprocess.CalledProcessError(1, command)
        if command == ["two"]:
            with log_path.open("a") as log:
                log.write("AttributeError: unknown method\n")
            raise subprocess.CalledProcessError(1, command)
        monkeypatch.setattr(job, "query_memory", lambda _: 28000)

    monkeypatch.setattr(job, "run_stage", run)
    with pytest.raises(subprocess.CalledProcessError):
        job.run("GPU-test", output, 1)
    assert states == ["retry_gpu_oom", "waiting_for_vram"]
    assert attempts == [["one"], ["one"], ["two"]]
    assert json.loads((output / "job_state.json").read_text())["completed"] == ["first"]


def test_legacy_recovery_only_restarts_once_on_latest_cuda_oom(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    state = {"status": "failed", "pid": 17, "stage": "fit-planner"}
    state_path = tmp_path / "job_state.json"
    state_path.write_text(json.dumps(state))
    log = logs / "fit-planner.log"
    log.write_text(
        "t0 fit-planner: start (free=26000 MiB)\nCUDA out of memory\n"
        "t1 fit-planner: start (free=26000 MiB)\nAttributeError\n"
    )
    calls = []
    monkeypatch.setattr(job.subprocess, "run", lambda cmd, **_: calls.append(cmd))
    unit = "relikflow-phoenix-pilot.service"
    job.recover_legacy_worker(tmp_path, unit)
    assert not calls
    with log.open("a") as handle:
        handle.write("t2 fit-planner: start (free=26000 MiB)\nCUDA out of memory\n")
    job.recover_legacy_worker(tmp_path, unit)
    job.recover_legacy_worker(tmp_path, unit)
    state_path.write_text(json.dumps({**state, "pid": 18, "attempt": 3}))
    job.recover_legacy_worker(tmp_path, unit)
    assert calls == [["systemctl", "--user", "--no-block", "start", unit]]

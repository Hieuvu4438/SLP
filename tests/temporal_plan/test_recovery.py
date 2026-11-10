import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from methods.temporal_plan.checkpoint import load_checkpoint
from methods.temporal_plan.data import MotionDataset
from methods.temporal_plan.progress import TrainingProgress, AtomicState, rng_state, restore_rng
from methods.temporal_plan.training import TrainingRunner, TrainingConfig
from methods.temporal_plan.system import STAGES
from methods.temporal_plan.pilot.evaluator import checked_optimizer_step
from methods.temporal_plan.pilot import queue


def data():
    return MotionDataset(
        [dict(name="train", text="hello rain", source_id="train", lang="en_ASL", fps=25.0)],
        [np.random.default_rng(9).normal(0, 0.05, (8, 133)).astype(np.float32)],
        split="train",
    )


def config(**kwargs):
    return TrainingConfig(
        batch_size=1,
        accumulation=1,
        epochs={s: 20 for s in STAGES},
        patience=None,
        checkpoint_every_updates=1,
        **kwargs,
    )


def test_training_recovery_restores_optimizer_noise_and_data_cursor(tiny_components, tmp_path):
    initial, tokenizer = tiny_components
    baseline = copy.deepcopy(initial)
    broken = copy.deepcopy(initial)
    train = MotionDataset(
        [
            dict(
                name=f"train{i}", text="hello rain", source_id=f"train{i}", lang="en_ASL", fps=25.0
            )
            for i in range(5)
        ],
        [
            np.random.default_rng(i + 10).normal(0, 0.05, (8, 133)).astype(np.float32)
            for i in range(5)
        ],
        split="train",
    )
    cfg = config(stage_max_updates={"codec": 8})
    TrainingRunner(baseline, tokenizer, cfg, tmp_path / "baseline").fit(train, stop_after="codec")
    runner = TrainingRunner(broken, tokenizer, cfg, tmp_path / "recovered")
    original = runner._loss
    calls = 0

    def interrupt(batch, generator):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("simulated power interruption")
        return original(batch, generator)

    runner._loss = interrupt
    with pytest.raises(RuntimeError, match="power interruption"):
        runner.fit(train, stop_after="codec")
    _, saved = TrainingProgress(tmp_path / "recovered").load()
    assert saved["runtime"]["updates"] == 4
    assert saved["runtime"]["epoch"] == 0 and saved["runtime"]["window_index"] == 4
    restored = copy.deepcopy(initial)
    finish = TrainingRunner(restored, tokenizer, cfg, tmp_path / "recovered")
    finish.fit(train, stop_after="codec", resume=True)
    assert finish.history[-1]["updates"] == 8
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)


def test_matched_weights_restart_equals_same_reset_in_continuous_branch(tiny_components, tmp_path):
    initial, tokenizer = tiny_components
    common = TrainingRunner(
        initial, tokenizer, config(stage_max_updates={s: 1 for s in STAGES}), tmp_path / "common"
    )
    shared = common.fit(data(), stop_after="segment")
    full, tokens, meta = load_checkpoint(shared)
    cfg = config(
        stage_max_updates={"planner": 1, "generator": 5, "exposure": 1},
        stage_restart_updates={"generator": [2]},
    )
    run = TrainingRunner(full, tokens, cfg, tmp_path / "continuous")
    run.fit(data(), start_stage="planner", parent_metadata=meta)
    phase = tmp_path / "continuous/phase_generator_2.pt"
    warm, tokens, meta = load_checkpoint(phase)
    assert warm.completed_stages.tolist() == [True, True, True, False, False]
    warm_cfg = replace(cfg, stage_initial_updates={"generator": 2})
    recover = TrainingRunner(warm, tokens, warm_cfg, tmp_path / "warm")
    recover.history = meta["history"]
    recover.fit(data(), start_stage="generator", parent_metadata=meta)
    for name, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, warm.state_dict()[name], rtol=0, atol=0)
    assert recover.history[-2]["updates"] == 5


def test_progress_pair_survives_uncommitted_slot_and_rejects_corruption(tiny_components, tmp_path):
    model, tokens = tiny_components
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    progress = TrainingProgress(tmp_path)
    runtime = {"updates": 0, "phase_start": 0, "best": float("inf")}
    progress.save(model, tokens, runtime, optimizer, torch.Generator().manual_seed(3), {})
    (progress.root / "b").mkdir()
    (progress.root / "b/model.pt").write_bytes(b"incomplete")
    _, payload = progress.load()
    assert payload["runtime"]["updates"] == 0
    (progress.root / "a/model.pt").write_bytes(b"corrupt committed file")
    with pytest.raises(ValueError, match="checksum"):
        progress.load()


def test_atomic_evaluator_state_restores_rng_and_optimizer(tmp_path):
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3)).sum().backward()
    opt.step()
    progress = AtomicState(tmp_path)
    progress.save({"state": model.state_dict(), "optimizer": opt.state_dict(), "rng": rng_state()})
    expected = torch.rand(4)
    payload = progress.load()
    restore_rng(payload["rng"])
    torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
    restored = torch.nn.Linear(3, 1)
    restored.load_state_dict(payload["state"])
    opt2 = torch.optim.AdamW(restored.parameters())
    opt2.load_state_dict(payload["optimizer"])
    assert next(iter(opt2.state.values()))["step"].item() == 1


def test_float64_gradient_norm_handles_large_finite_gradients_and_rejects_nan():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([p], lr=0.01)
    p.grad = torch.tensor([1e30, 1e30])
    norm = checked_optimizer_step([p], optimizer)
    assert norm is not None and np.isfinite(norm) and torch.isfinite(p).all()
    old = p.detach().clone()
    step = optimizer.state[p]["step"].clone()
    p.grad = torch.tensor([float("nan"), 1.0])
    assert checked_optimizer_step([p], optimizer) is None
    torch.testing.assert_close(old, p)
    torch.testing.assert_close(step, optimizer.state[p]["step"])


def test_unlimited_job_and_probe_have_no_deadline(tmp_path, monkeypatch):
    plan = json.loads(Path("configs/pilots/phoenix72_20261106.json").read_text())
    plan.update(output=str(tmp_path / "out"), min_free_gib=0)
    plan["jobs"] = [
        dict(id="prepare", needs=[], gpu=False, timeout_hours=None),
        dict(id="report", needs=["prepare"], gpu=False, timeout_hours=None),
    ]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calls = []

    def child(command, log, **kwargs):
        calls.append(command[-1])
        assert kwargs["deadline"] is None
        assert "PILOT_JOB_DEADLINE_EPOCH" not in kwargs["env"]
        assert kwargs["env"]["PILOT_QUEUE_MANAGED"] == "1"
        kwargs["on_start"](123)
        return 0, None

    monkeypatch.setattr(queue, "run_child", child)
    monkeypatch.setattr(queue, "check_gate", lambda _: None)
    queue.dispatch(path)
    state = json.loads((Path(plan["output"]) / "queue_state.json").read_text())
    assert state["status"] == "completed" and calls == ["prepare", "report"]
    assert all(j["deadline_epoch"] is None for j in state["jobs"].values())


def test_real_worker_without_deadline_can_finish(tmp_path):
    import os
    import sys

    code, reason = queue.run_child(
        [sys.executable, "-c", "import time; time.sleep(.05)"],
        tmp_path / "job.log",
        deadline=None,
        env=os.environ.copy(),
        tick=0.01,
    )
    assert code == 0 and reason is None


def test_runtime_oom_waits_and_relaunches_recoverable_job(tmp_path, monkeypatch):
    plan = json.loads(Path("configs/pilots/phoenix72_20261106.json").read_text())
    plan.update(output=str(tmp_path / "out"), min_free_gib=0)
    plan["jobs"] = [dict(id="map", needs=[], gpu=True, timeout_hours=None)]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))

    def admit(plan, path, root, state, job, save, resource):
        state["gpu_wait"] = {"total_bytes": 48 * 2**30}
        state.setdefault("memory_profiles", {}).setdefault(
            "map", {"peak_reserved_bytes": 4 * 2**30}
        )
        return 12 * 2**30

    calls = []

    def child(command, log, **kwargs):
        calls.append(1)
        kwargs["on_start"](123)
        return (76, None) if len(calls) == 1 else (0, None)

    monkeypatch.setattr(queue, "admit_gpu_job", admit)
    monkeypatch.setattr(queue, "run_child", child)
    monkeypatch.setattr(queue, "check_gate", lambda _: None)
    queue.dispatch(path)
    state = json.loads((Path(plan["output"]) / "queue_state.json").read_text())
    assert calls == [1, 1] and state["jobs"]["map"]["status"] == "completed"


def test_service_restart_recovers_only_committed_interrupted_training(tmp_path):
    (tmp_path / "evaluator_resume").mkdir()
    (tmp_path / "evaluator_resume/current.json").write_text("{}")
    (tmp_path / "map/resume").mkdir(parents=True)
    (tmp_path / "map/resume/current.json").write_text("{}")
    state = {
        "jobs": {
            "evaluator": {"status": "failed", "reason": "dispatcher interrupted", "pid": 123},
            "map": {"status": "failed", "reason": "nonfinite gradients"},
            "sample": {"status": "running", "pid": 456},
        }
    }
    queue.recover_interrupted_jobs(tmp_path, state)
    assert state["jobs"]["evaluator"]["status"] == "pending"
    assert "pid" not in state["jobs"]["evaluator"]
    assert state["jobs"]["map"]["status"] == "failed"
    assert state["jobs"]["sample"]["status"] == "running"  # no committed optimizer checkpoint


def test_interrupted_service_terminal_status_does_not_block_resume(tmp_path, monkeypatch):
    plan = json.loads(Path("configs/pilots/phoenix72_20261106.json").read_text())
    plan.update(output=str(tmp_path / "out"), min_free_gib=0)
    plan["jobs"] = [dict(id="evaluator", needs=[], gpu=False, timeout_hours=None)]
    root = Path(plan["output"])
    (root / "evaluator_resume").mkdir(parents=True)
    (root / "evaluator_resume/current.json").write_text("{}")
    queue.atomic_json(
        root / "queue_state.json",
        {
            "started_epoch": 1,
            "status": "inconclusive",
            "jobs": {"evaluator": {"status": "failed", "reason": "dispatcher interrupted"}},
        },
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calls = []

    def child(command, log, **kwargs):
        calls.append(command[-1])
        kwargs["on_start"](123)
        return 0, None

    monkeypatch.setattr(queue, "run_child", child)
    monkeypatch.setattr(queue, "check_gate", lambda _: None)
    queue.dispatch(path)
    state = json.loads((root / "queue_state.json").read_text())
    assert calls == ["evaluator"] and state["status"] == "completed"

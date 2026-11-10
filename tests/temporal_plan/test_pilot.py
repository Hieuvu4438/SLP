import copy
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

from test_smplx_adapter import official_synthetic_smplx
from methods.temporal_plan.pilot import queue
from methods.temporal_plan.pilot.worker import packed_to_raw, dtw_native
from methods.temporal_plan.training import TrainingConfig, TrainingRunner
from methods.temporal_plan.system import STAGES
from methods.temporal_plan.checkpoint import load_checkpoint, read_checkpoint
from methods.temporal_plan.data import MotionDataset, validate_train_val


def tiny_data(split):
    return MotionDataset([dict(name=split, text="hello rain", source_id="same_recording",
        fps=25., lang="en_ASL")], [np.random.default_rng(7).normal(0,.05,(8,133)).astype(np.float32)], split=split)


def test_absolute_time_floor_cannot_be_weakened_by_plan():
    plan = {"not_before": "2020-01-01T00:00:00+07:00"}
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        queue.check_gate(plan, queue.EARLIEST-1)
    queue.check_gate(plan, queue.EARLIEST)
    plan["not_before"] = "2026-11-06T01:00:00+07:00"
    with pytest.raises(RuntimeError):
        queue.check_gate(plan, queue.EARLIEST+30)


def test_real_child_wait_timeout_exit_and_no_overlap(tmp_path):
    # These are CPU-only seconds-long scheduler tests, never experiment jobs.
    for i in range(2):
        started = time.time()
        code, reason = queue.run_child([sys.executable, "-c", "import time; time.sleep(.05)"],
            tmp_path/f"{i}.log", deadline=time.time()+5, env=os.environ.copy(), tick=.01)
        assert code == 0 and reason is None and time.time()-started >= .05
    code, reason = queue.run_child([sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path/"timeout.log", deadline=time.time()+.05, env=os.environ.copy(), tick=.01)
    assert code != 0 and reason == "hard_timeout"
    code, reason = queue.run_child([sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path/"exit.log", deadline=time.time()+5, env=os.environ.copy(), tick=.01)
    assert code == 7 and reason is None
    pids = []
    def broken_record(pid):
        pids.append(pid)
        raise OSError("state write failed")
    with pytest.raises(OSError, match="state write"):
        queue.run_child([sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path/"record_failure.log", deadline=time.time()+5, env=os.environ.copy(),
            on_start=broken_record, tick=.01)
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_queue_dependencies_failures_and_no_retry(tmp_path, monkeypatch):
    source = Path("configs/pilots/phoenix72_20261106.json")
    plan = json.loads(source.read_text())
    plan["output"] = str(tmp_path/"output")
    plan["min_free_gib"] = 0
    plan["jobs"] = [dict(id="prepare", needs=[], gpu=False, timeout_hours=.1),
                    dict(id="preflight", needs=["prepare"], gpu=True, timeout_hours=.1),
                    dict(id="report", needs=[], gpu=False, timeout_hours=.1)]
    path = tmp_path/"plan.json"; path.write_text(json.dumps(plan))
    monkeypatch.setattr(queue, "check_gate", lambda p: None)
    invoked = []
    def fake(command, log, **kwargs):
        invoked.append(command[-1]); kwargs["on_start"](123)
        return (1, None) if command[-1] == "prepare" else (0, None)
    monkeypatch.setattr(queue, "run_child", fake)
    monkeypatch.setattr(queue, "gpu_status", lambda *args: pytest.fail("dependent GPU job must not run"))
    queue.dispatch(path)
    state = json.loads((Path(plan["output"])/"queue_state.json").read_text())
    assert invoked == ["prepare", "report"]
    assert state["jobs"]["preflight"]["status"] == "skipped"
    assert state["status"] == "inconclusive"
    queue.dispatch(path)
    assert len(invoked) == 2


def test_unlimited_wall_time_keeps_job_timeouts_and_runs_all_jobs(tmp_path, monkeypatch):
    plan = json.loads(Path("configs/pilots/phoenix72_20261106.json").read_text())
    plan.update(hours=None, output=str(tmp_path/"output"), min_free_gib=0)
    plan["jobs"] = [dict(id="prepare", needs=[], gpu=False, timeout_hours=.1),
                    dict(id="report", needs=["prepare"], gpu=False, timeout_hours=.1)]
    path = tmp_path/"plan.json"; path.write_text(json.dumps(plan))
    root = Path(plan["output"]); root.mkdir()
    queue.atomic_json(root/"queue_state.json", {"started_epoch": 1, "deadline_epoch": 2,
                                              "jobs": {}, "status": "running"})
    monkeypatch.setattr(queue, "check_gate", lambda p: None)
    invoked = []
    def fake(command, log, **kwargs):
        invoked.append(command[-1])
        assert time.time() < kwargs["deadline"] < time.time()+400
        assert float(kwargs["env"]["PILOT_JOB_DEADLINE_EPOCH"]) == kwargs["deadline"]
        kwargs["on_start"](123)
        return 0, None
    monkeypatch.setattr(queue, "run_child", fake)
    queue.dispatch(path)
    state = json.loads((root/"queue_state.json").read_text())
    assert state["status"] == "completed" and state["deadline_epoch"] is None
    assert invoked == ["prepare", "report"]
    assert not queue.wall_expired(state, now=1e20)
    assert queue.wall_expired({"deadline_epoch": 10}, now=10)


def test_gpu_telemetry_is_not_a_fixed_memory_or_utilization_gate(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="49140, 12000, 99", stderr=""))
    snapshot = queue.gpu_status(0)
    assert snapshot["free_bytes"] == 12000*2**20
    assert snapshot["utilization_percent"] == 99
    assert "ready" not in snapshot


def test_measured_requirement_admits_busy_gpu_with_less_than_40_gib(monkeypatch):
    monkeypatch.setattr(queue, "gpu_status", lambda *_: dict(
        total_bytes=48*2**30, free_bytes=12*2**30, utilization_percent=100,
        free_gib=12, total_gib=48))
    plan = {"gpu":0,"gpu_policy":{"mode":"measured_memory","reserve_fraction":.1}}
    state = {"deadline_epoch":None}; saved=[]
    budget = queue.wait_for_gpu(plan,state,lambda:saved.append(copy.deepcopy(state)),
        required_bytes=5*2**30, sleep=lambda _:pytest.fail("enough memory must not wait"))
    assert budget == int(12*2**30*.9)
    assert not state["gpu_wait"]["utilization_is_gate"]


def test_memory_wait_releases_process_slot_until_resources_increase(monkeypatch):
    sizes=iter([4,7,12])
    def sample(*args):
        n=next(sizes)
        return dict(total_bytes=48*2**30,free_bytes=n*2**30,utilization_percent=99,free_gib=n,total_gib=48)
    monkeypatch.setattr(queue,"gpu_status",sample)
    plan={"gpu":0,"gpu_policy":{"mode":"measured_memory","reserve_fraction":.1,"poll_seconds":30}}
    state={"deadline_epoch":None}; sleeps=[]
    budget=queue.wait_for_gpu(plan,state,lambda:None,required_bytes=8*2**30,sleep=sleeps.append)
    assert sleeps==[30,30] and budget >= 8*2**30


def test_probe_oom_waits_for_more_memory_then_success_is_cached(tmp_path,monkeypatch):
    plan=json.loads(Path("configs/pilots/phoenix72_20261106.json").read_text())
    state={"deadline_epoch":None,"jobs":{}}
    budgets=iter([10*2**30,13*2**30,13*2**30,13*2**30])
    required=[]; commands=[]
    def admit(plan,state,save,*,required_bytes):
        required.append(required_bytes)
        state["gpu_wait"]={"total_bytes":48*2**30}
        return next(budgets)
    monkeypatch.setattr(queue,"wait_for_gpu",admit)
    def child(command,log,**kwargs):
        commands.append(command)
        assert "--memory-probe" in command
        kwargs["on_start"](123)
        report=Path(command[-1])
        assert "PILOT_GPU_MEMORY_BUDGET_BYTES" in kwargs["env"]
        if len(commands)==1:
            queue.atomic_json(report,{"status":"oom","required_lower_bound_bytes":10*2**30+1})
            return 75,None
        queue.atomic_json(report,{"status":"fit","peak_reserved_bytes":9*2**30,
            "peak_allocated_bytes":8*2**30,"experimental_weights_saved":False})
        return 0,None
    monkeypatch.setattr(queue,"run_child",child)
    job={"id":"common"}
    budget=queue.admit_gpu_job(plan,tmp_path/"plan.json",tmp_path,state,job,lambda:None,lambda:None)
    assert len(commands)==2 and budget==13*2**30
    assert required==[1,12*2**30,9*2**30]
    assert "common" not in state["jobs"]  # no real training job was started/retried
    queue.admit_gpu_job(plan,tmp_path/"plan.json",tmp_path,state,job,lambda:None,lambda:None)
    assert len(commands)==2  # cached successful measurement, not repeated optimization


def test_non_memory_probe_failure_is_not_retried(tmp_path,monkeypatch):
    plan=json.loads(Path("configs/pilots/phoenix72_20261106.json").read_text())
    state={"deadline_epoch":None,"jobs":{}}
    monkeypatch.setattr(queue,"wait_for_gpu",lambda *a,**k:10*2**30)
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1);kwargs["on_start"](123)
        return 1,None
    monkeypatch.setattr(queue,"run_child",fail)
    assert queue.admit_gpu_job(plan,tmp_path/"plan.json",tmp_path,state,{"id":"common"},lambda:None,lambda:None) is None
    assert calls==[1] and state["jobs"]["common"]["status"]=="failed"


def test_disposable_memory_diagnostics_train_real_tiny_modules_without_saving(tiny_components,tmp_path):
    from methods.temporal_plan.pilot.memory import diagnostic_batch, diagnostic_updates, maximum_targets
    model,tokenizer=tiny_components
    train=tiny_data("train")
    cfg=TrainingConfig(batch_size=1,accumulation=1,epochs={s:1 for s in STAGES})
    TrainingRunner(model,tokenizer,cfg,tmp_path/"common").fit(train,stop_after="segment")
    maximum_targets(model)
    batch=diagnostic_batch(train,2)
    assert batch.raw.shape==(2,8,133) and batch.lengths.tolist()==[8,8]
    generator=torch.Generator().manual_seed(0)
    runner=TrainingRunner(model,tokenizer,cfg,tmp_path/"must_not_exist")
    for stage in ("planner","generator","exposure"):
        model.set_stage(stage)
        diagnostic_updates(runner,batch,generator)
        model.mark_stage_completed()
    assert not runner.output_dir.exists()
    assert model.completed_stages.all()


def test_raw_layout_uses_retained_body_hands_jaw_expression_and_no_shape():
    pose = {"smplx_body_pose": np.arange(2*63).reshape(2,63),
            "smplx_lhand_pose": np.ones((2,45))*100,
            "smplx_rhand_pose": np.ones((2,45))*200,
            "smplx_jaw_pose": np.ones((2,3))*300,
            "smplx_expr": np.ones((2,10))*400,
            "smplx_shape": np.ones((2,10))*999}
    raw = packed_to_raw(pose)
    assert raw.shape == (2,133) and raw.dtype == np.float32
    np.testing.assert_array_equal(raw[:,:30], pose["smplx_body_pose"][:,33:])
    assert (raw[:,30:75] == 100).all() and (raw[:,123:] == 400).all()
    assert not (raw == 999).any()


def test_explicit_official_source_overlap_exception_preserves_sample_guard():
    train, val = tiny_data("train"), tiny_data("val")
    with pytest.raises(ValueError, match="source_id"):
        validate_train_val(train,val)
    validate_train_val(train,val,allow_shared_source_groups=True)
    val.records[0]["name"] = "train"
    with pytest.raises(ValueError, match="name"):
        validate_train_val(train,val,allow_shared_source_groups=True)


def test_stage_boundary_fork_same_weights_stats_and_matched_update_targets(tiny_components,tmp_path):
    model, tokenizer = tiny_components
    train = tiny_data("train")
    cfg = TrainingConfig(batch_size=1, accumulation=1, epochs={s: 5 for s in STAGES},
        patience=None, stage_max_updates={s: 2 for s in STAGES})
    common = TrainingRunner(model,tokenizer,cfg,tmp_path/"common")
    path = common.fit(train, stop_after="segment")
    assert common.model.completed_stages.tolist() == [True,True,False,False,False]
    starts, ends = [], []
    for mode in ("map","sample"):
        m,t,meta = load_checkpoint(path)
        starts.append({k:v.clone() for k,v in m.state_dict().items()})
        m.config = replace(m.config,teacher_mode=mode)
        runner = TrainingRunner(m,t,cfg,tmp_path/mode)
        end = runner.fit(train,start_stage="planner",parent_metadata=meta)
        assert [h["updates"] for h in runner.history] == [2,2,2]
        assert m.completed_stages.all()
        ends.append(read_checkpoint(end))
    for key in starts[0]:
        torch.testing.assert_close(starts[0][key],starts[1][key])
    for key in ("motion_norm.mean","latent_norm.std","content_norm.mean"):
        torch.testing.assert_close(ends[0]["state_dict"][key],ends[1]["state_dict"][key])
    wrong, t, meta = load_checkpoint(path)
    meta["train_fingerprint"] = "wrong"
    with pytest.raises(ValueError,match="same training"):
        TrainingRunner(wrong,t,cfg,tmp_path/"wrong").fit(train,start_stage="planner",parent_metadata=meta)


def test_unmet_update_budget_never_marks_stage_complete(tiny_components,tmp_path):
    model,tokenizer=tiny_components
    cfg=TrainingConfig(batch_size=1,accumulation=1,epochs={s:1 for s in STAGES},
        stage_max_updates={"codec":2})
    with pytest.raises(RuntimeError,match="matched update target"):
        TrainingRunner(model,tokenizer,cfg,tmp_path/"short").fit(tiny_data("train"))
    assert not model.completed_stages.any()
    ckpt=read_checkpoint(tmp_path/"short/incomplete_codec.pt")
    assert ckpt["metadata"]["actual_updates"] == 1


def test_fast_native_fk_real_engine_positions_and_gradients(official_synthetic_smplx):
    from methods.temporal_plan.kinematics import SMPLXForward
    rig=official_synthetic_smplx
    torch.manual_seed(9)
    with torch.no_grad():
        rig.shapedirs.copy_(torch.randn_like(rig.shapedirs)*.001)
        rig.expr_dirs.copy_(torch.randn_like(rig.expr_dirs)*.002)
        rig.v_template.add_(torch.randn_like(rig.v_template)*.01)
    fast=SMPLXForward(rig,native_only=True)
    full=SMPLXForward(rig)
    raw=(torch.randn(3,133)*.1).requires_grad_()
    a,b=fast(raw),full(raw)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    da=torch.autograd.grad(a.square().sum(),raw)[0]
    db=torch.autograd.grad(b.square().sum(),raw)[0]
    torch.testing.assert_close(da,db,atol=1e-5,rtol=1e-4)


def test_independent_evaluator_actual_mt5_forward_gradient_and_checkpoint():
    from transformers import MT5Config,MT5ForConditionalGeneration
    from methods.temporal_plan.pilot.evaluator import PoseTranslator,corpus_bleu,calibration
    config=MT5Config(vocab_size=16,d_model=8,d_kv=4,d_ff=16,num_layers=1,
        num_decoder_layers=1,num_heads=2,dropout_rate=0,decoder_start_token_id=0,pad_token_id=0,eos_token_id=2)
    model=PoseTranslator(MT5ForConditionalGeneration(config))
    x=torch.randn(2,9,256)
    loss=model(x,torch.tensor([9,5]),torch.tensor([[3,4,2],[4,3,2]]))
    loss.backward()
    assert torch.isfinite(loss) and model.conv1.weight.grad.abs().sum()>0
    assert model.lm.shared.weight.grad is None
    assert any(p.grad is not None for n,p in model.lm.named_parameters() if "EncDecAttention" in n)
    other=copy.deepcopy(model)
    state=model.compact_state(); other.restore_compact(state)
    assert not any(k.startswith("lm.shared") for k in state)
    model.eval(); other.eval()
    torch.testing.assert_close(model(x,torch.tensor([9,5]),torch.tensor([[3,4,2],[4,3,2]])),
                               other(x,torch.tensor([9,5]),torch.tensor([[3,4,2],[4,3,2]])))
    class Tokens:
        def batch_decode(self, tokens, **kwargs):
            return [" ".join(map(str, row.tolist())) for row in tokens]
    generated = model.translate(x, torch.tensor([9,5]), Tokens())
    assert len(generated) == 2 and all(isinstance(t, str) for t in generated)
    assert corpus_bleu(["a b c d"],["a b c d"]) == pytest.approx(100)
    assert corpus_bleu(["wrong wrong wrong wrong"],["a b c d"]) == 0
    assert not calibration(["x"],["x"],["a b c d"])["usable"]


def test_native_dtw_identity_shift_and_variable_length():
    a=np.zeros((3,41,3));b=np.zeros((5,41,3))
    assert dtw_native(a,b) == 0
    b[:,:,0]=.01
    assert dtw_native(a,b) == pytest.approx(10)

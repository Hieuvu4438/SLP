"""Supervisor: SignRAE jobs, one process at a time, evaluated on DEV only.

    nohup setsid python -u -m methods.morae_slp.job [--plan v2] > <job dir>/supervisor.out 2>&1 &
    python -m methods.morae_slp.job [--plan v2] --status

Plan v1 (outputs/morae_slp/job; finished 2026-11-25, kept as the control):
  jepa   Sign-JEPA on the union of the three train splits
  ae     compact latent (d=64) on the union, AE checkpoint chosen on val
  diag   gate-G1 diagnostics on Phoenix (spectrum, reconstruction ceiling, gloss probe)
  flow   latent flow on Phoenix train + scoring on Phoenix dev (scripts/run.py --split val)
  sweep  guidance grid of the trained flow on 256 dev clips

Plan v2 (outputs/morae_slp/job_v2; PLAN_V2.md): the v1 flow memorised the 7k Phoenix sentences
(dev flow loss with text > without text), so the generator is pre-trained on the union of the
three train splits (56k sentences, mBART fine-tuned, text-token dropout, best-val selection) and
fine-tuned per dataset; sweeps add pose-space ensembles (point estimate of E[pose | text]).
Stages are skipped once done; an interrupted stage restarts from scratch.

A file lock allows a single supervisor, and stages run strictly one after another, so this job
never has more than one training process. Before each stage it only waits until the GPU has
MIN_FREE_MIB free (out-of-memory guard); other jobs on the GPU are not otherwise considered.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from slp.paths import PROJECT_ROOT

JOB = PROJECT_ROOT / "outputs" / "morae_slp" / "job"  # set per plan in main()
PRE = PROJECT_ROOT / "outputs" / "morae_slp" / "pretrain"
DIAG = PROJECT_ROOT / "outputs" / "morae_slp" / "diagnostics"
UNION = "phoenix_smplx,csl_smplx,how2sign_smplx_soke"
EXPERIMENT = "morae_phoenix_main"
MIN_FREE_MIB = 14 * 1024
PY = sys.executable


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(JOB / "job.log", "a") as f:
        f.write(line + "\n")


V2 = {  # dataset -> (experiment name, fine-tune updates)
    "phoenix_smplx": ("morae_v2_phoenix", 20000),
    "csl_smplx": ("morae_v2_csl", 30000),
    "how2sign_smplx_soke": ("morae_v2_how2sign", 40000),
}
V2_COMMON = ["method.text.finetune=true", "method.text.lr_scale=0.1", "method.flow.token_drop=0.1",
             "method.flow.select=flow"]
# Fine-tune stages run next to other users' jobs: half the micro-batch halves the activation peak
# (same optimiser batch of 64; only the frame weighting inside a micro-batch differs slightly).
V2_FT = ["method.flow.microbatch=16"]
OOM_RETRIES = 4


def stages_v2(workers: int) -> list[tuple[str, callable]]:
    ae = PRE / "ae_union_jepa_d64" / "ae.pt"
    union = PRE / "flow_union_v2"
    w = f"method.workers={workers}"
    out = [("flow_union", lambda: ([PY, "-u", "-m", "methods.morae_slp.pretrain", "--stage", "flow",
                                    "--datasets", UNION, "--out", str(union), "--init-ae", str(ae),
                                    w, *V2_COMMON, "method.flow.updates=100000",
                                    "method.flow.snapshot_steps=[40000,70000]"], union)),
           ("zeroshot_phoenix", lambda: ([PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                                          str(union), "--data", "phoenix_smplx", "--cfg", "1,2.5",
                                          "--ensemble", "1,8", "--limit", "256"], None))]
    for data, (name, updates) in V2.items():
        short = name.removeprefix("morae_v2_")
        out.append((f"ft_{short}", lambda data=data, name=name, updates=updates: (
            [PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", data, "--name", name,
             "--seeds", "0", "--split", "val", w, *V2_COMMON, *V2_FT, f"method.init_ae={ae}",
             f"method.init_flow={union / 'checkpoints' / 'model.pt'}",
             "method.latent.recalibrate=false", f"method.flow.updates={updates}",
             "method.flow.lr=1.0e-4", "method.flow.warmup_updates=500",
             "method.flow.eval_every=1000", "method.flow.ema_decay=0.999",
             "method.sampler.cfg_scale=2.0"], None)))
        out.append((f"sweep_{short}", lambda name=name: (
            [PY, "-u", "-m", "methods.morae_slp.sweep", "--run", str(latest_run(name)),
             "--cfg", "1,1.5,2.5,3.5", "--ensemble", "1,8", "--limit", "256"], None)))
    # Control: the same v2 recipe on Phoenix alone (no union pre-training of the flow).
    out.append(("ctrl_phoenix_only", lambda: (
        [PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data", "phoenix_smplx", "--name",
         "morae_v2ctrl_phoenix_only", "--seeds", "0", "--split", "val", w, *V2_COMMON, *V2_FT,
         f"method.init_ae={ae}", "method.flow.updates=30000", "method.sampler.cfg_scale=2.0"],
        None)))
    out.append(("sweep_ctrl", lambda: (
        [PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
         str(latest_run("morae_v2ctrl_phoenix_only")), "--cfg", "1,1.5,2.5", "--ensemble", "1,8",
         "--limit", "256"], None)))
    return out


PLANS = {"v1": ("job", "stages"), "v2": ("job_v2", "stages_v2")}


def latest_run(name: str) -> Path | None:
    runs = sorted((PROJECT_ROOT / "outputs" / "runs" / name / "morae_slp").glob("seed0_*"))
    done = [r for r in runs if (r / "metrics.json").is_file()]
    return done[-1] if done else None


def stages(workers: int) -> list[tuple[str, callable]]:
    """(name, function returning (argv, output dir to clear before a fresh attempt or None))."""
    jepa_dir, ae_dir = PRE / "jepa_union", PRE / "ae_union_jepa_d64"
    w = f"method.workers={workers}"
    return [
        ("jepa", lambda: ([PY, "-u", "-m", "methods.morae_slp.pretrain", "--stage", "jepa",
                           "--datasets", UNION, "--out", str(jepa_dir), w], jepa_dir)),
        ("ae", lambda: ([PY, "-u", "-m", "methods.morae_slp.pretrain", "--stage", "ae",
                         "--datasets", UNION, "--out", str(ae_dir),
                         "--init-jepa", str(jepa_dir / "jepa.pt"), w], ae_dir)),
        ("diag", lambda: ([PY, "-u", "-m", "methods.morae_slp.diagnostics", "--data",
                           "phoenix_smplx", "--ae", str(ae_dir / "ae.pt"), "--spectrum",
                           "--recon", "519", "--probe",
                           "--out", str(DIAG / "g1_jepa_d64_phoenix.json")], None)),
        ("flow", lambda: ([PY, "-u", "scripts/run.py", "--method", "morae_slp", "--data",
                           "phoenix_smplx", "--name", EXPERIMENT, "--seeds", "0", "--split",
                           "val", f"method.init_ae={ae_dir / 'ae.pt'}", w,
                           "method.flow.snapshot_steps=[20000,40000]"], None)),
        ("sweep", lambda: ([PY, "-u", "-m", "methods.morae_slp.sweep", "--run",
                            str(latest_run(EXPERIMENT)), "--solver", "rk4", "--steps", "5",
                            "--cfg", "1.5,2.5,3.5,4.5,6.5", "--temperature", "1.0",
                            "--limit", "256"], None)),
    ]


def load_state() -> dict:
    path = JOB / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(state: dict) -> None:
    tmp = JOB / ".state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(JOB / "state.json")


def free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return max(int(x) for x in out.split())


def wait_for_gpu(poll: int) -> None:
    """Only guards against OOM: wait until the GPU has MIN_FREE_MIB free."""
    last = 0.0
    while (free := free_mib()) < MIN_FREE_MIB:
        if time.time() - last > 1800:
            log(f"waiting for GPU memory: {free} MiB free < {MIN_FREE_MIB}")
            last = time.time()
        time.sleep(poll)


def run_stage(name: str, argv: list[str], clear: Path | None) -> int:
    if clear is not None and clear.exists():
        import shutil
        shutil.rmtree(clear)  # interrupted attempt; pretrain needs an empty directory
    out = JOB / f"{name}.log"
    log(f"start {name}: {' '.join(argv)}")
    with open(out, "a") as f:
        f.write(f"\n===== {now()} {' '.join(argv)}\n")
        f.flush()
        proc = subprocess.Popen(argv, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1",
                                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

        def stop(signum, _frame):
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log(f"stopped by signal {signum} during {name}")
            sys.exit(1)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        return proc.wait()


def ended_in_oom(name: str) -> bool:
    """The last stage attempt died of CUDA out-of-memory (other jobs share this GPU)."""
    try:
        text = (JOB / f"{name}.log").read_text(errors="replace")
    except OSError:
        return False
    last = text.rsplit("=====", 1)[-1]
    return "OutOfMemoryError" in last or "CUDA out of memory" in last


def status(build) -> None:
    state = load_state()
    for name, _ in build(0):
        print(f"{name:6s} {state.get(name, {}).get('status', 'pending')}  "
              f"{state.get(name, {}).get('finished', '')}")
    for exp in [EXPERIMENT] if build is stages else [n for n, _ in V2.values()]:
        run = latest_run(exp)
        if run:
            print("flow run:", run)
            print(json.dumps(json.loads((run / "metrics.json").read_text()), indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="skip the free-memory check")
    p.add_argument("--min-free-gb", type=float, default=None,
                   help="free GPU memory required before a stage starts (default 14)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--plan", choices=sorted(PLANS), default="v1")
    args = p.parse_args(argv)
    global JOB, MIN_FREE_MIB
    if args.min_free_gb is not None:
        MIN_FREE_MIB = int(args.min_free_gb * 1024)
    folder, builder = PLANS[args.plan]
    JOB = PROJECT_ROOT / "outputs" / "morae_slp" / folder
    build = globals()[builder]
    JOB.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    if args.status:
        status(build)
        return
    with open(JOB / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("Another SignRAE job supervisor is already running")
        log(f"supervisor pid {os.getpid()} started (plan {args.plan})")
        state = load_state()
        for name, make in build(args.workers):
            if state.get(name, {}).get("status") == "done":
                continue
            if not args.no_wait:
                wait_for_gpu(args.poll)
            argv, clear = make()
            state[name] = {"status": "running", "started": now()}
            save_state(state)
            code = run_stage(name, argv, clear)
            attempts = 0
            while code != 0 and attempts < OOM_RETRIES and ended_in_oom(name):
                attempts += 1
                log(f"{name}: CUDA OOM (another job took the memory); retry {attempts}/{OOM_RETRIES}")
                time.sleep(args.poll)
                if not args.no_wait:
                    wait_for_gpu(args.poll)
                code = run_stage(name, argv, clear)
            state[name].update(status="done" if code == 0 else f"failed({code})",
                               finished=now())
            save_state(state)
            log(f"{name} exit {code}")
            if code != 0:
                log(f"stopping: see {JOB / (name + '.log')}")
                sys.exit(code)
        log(f"all stages done (plan {args.plan})")


if __name__ == "__main__":
    main()

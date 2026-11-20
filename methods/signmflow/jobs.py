"""Job queue for every SignNMFlow experiment (Tables 1-5, 7, 8 + B-T), at most N processes at a time.

    python -m methods.signmflow.jobs status            # table of all jobs
    python -m methods.signmflow.jobs plan              # launch order and resolved commands
    nohup python -m methods.signmflow.jobs run >> outputs/signmflow/jobs/runner.log 2>&1 &

Scheduling
  * Dependency graph: ablations / NFE / sigma reuse the checkpoints of the CSL-Daily main run (the
    paper's "full" row), SignNMFlow-M evaluations reuse the multilingual model, B-T scores the
    predictions of finished runs.
  * At most ``--max-procs`` (default 1) processes: every running ``scripts/run.py --method signmflow``
    process counts, including runs started by hand (they are adopted, not restarted), plus the
    runner's own command jobs.
  * Ready jobs start in priority order: short dependents (eval-only, B-T, NFE, benchmark) first
    so they do not wait behind multi-day trainings, then main seed 0, Tables 3/5, SignNMFlow-M,
    Table 4, extra seeds of the main table, extra seeds of the ablations.
  * A GPU-memory gate (``--min-free-gb``) is checked before each launch; one launch per tick.
  * Failures: CUDA OOM -> retry with half the generator micro-batch (the 64-sample batches and
    the effective batch 256 are unchanged) after a back-off; any failure resumes from the previous attempts of the
    same job (prototypes.pt, correction.pt / correction_final.pt, generator.pt), so no finished
    stage is recomputed. A job is failed after ``MAX_ATTEMPTS`` attempts.
  * State is the file system: a run job is done when one of its run dirs has metrics.json, a
    command job when its output file exists. ``outputs/signmflow/jobs/state.json`` only keeps
    attempt counters and batch sizes, so the runner can be stopped and restarted at any time.

All evaluation is on the val split (CLAUDE.md); the test split is never touched here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"
JOBS_DIR = ROOT / "outputs" / "signmflow" / "jobs"
STATE = JOBS_DIR / "state.json"
BT_PY = os.environ.get("BT_PYTHON", str(ROOT / "third_party/MaDiS/.venv/bin/python"))
MAX_ATTEMPTS, MAX_OOM_ATTEMPTS = 3, 8
GEN_MICRO, MIN_GEN_MICRO = 16, 2       # generator micro-batch (memory only; batch 64 per "GPU" is kept)
# v2 = after the 2026-11-16 review (K-means convergence, sum-reduced loss, exact 64-sample batches,
# SOKE How2Sign protocol); v1 runs (signmflow_main_*) are kept for reference only
PREFIX = "signmflow_v2"
SRC = {"phoenix_smplx": "phoenix", "csl_smplx": "csl", "how2sign_smplx_soke": "how2sign"}
DATASETS = list(SRC)


# ----------------------------------------------------------------------------- job model
@dataclass
class Job:
    id: str
    prio: int
    deps: tuple = ()
    name: str | None = None            # run job: experiment name for scripts/run.py
    data: str | None = None
    seed: int = 0
    over: Callable[[], list] | list = field(default_factory=list)
    trains: bool = True                # has training stages (OOM batch halving, stage resume)
    cmd: Callable[[], list] | None = None   # command job
    out: Path | None = None            # command job: done when this file exists
    heavy: bool = True                 # needs the GPU-memory gate

    @property
    def is_run(self):
        return self.name is not None


def run_dirs(name, seed):
    return sorted((RUNS / name / "signmflow").glob(f"seed{seed}_*"))


def done_dir(name, seed):
    d = [p for p in run_dirs(name, seed) if (p / "metrics.json").exists()]
    return d[-1] if d else None


JOBS: dict[str, Job] = {}


def add(job: Job):
    assert job.id not in JOBS, job.id
    JOBS[job.id] = job
    return job


def ck(job_id) -> Path:
    j = JOBS[job_id]
    return done_dir(j.name, j.seed) / "checkpoints"


def init_from(job_id, stages=("prototypes", "correction", "generator")):
    files = {"prototypes": "prototypes.pt", "correction": "correction_final.pt", "generator": "generator.pt"}
    return lambda: [f"method.init.{s}={ck(job_id) / files[s]}" for s in stages]


def run_job(id, prio, data, name, seed, over=(), deps=(), trains=True, heavy=True):
    return add(Job(id, prio, tuple(deps), name=name, data=data, seed=seed, over=over,
                   trains=trains, heavy=heavy))


def bt_job(run_id, prio=0):
    """B-T BLEU-4 of a finished run with the repo's mBART back-translator of its language."""
    r = JOBS[run_id]
    src = SRC[r.data]
    out = ROOT / "outputs" / "bt" / f"{src}_mbart" / f"{r.name}_seed{r.seed}.json"

    def cmd():
        pred = done_dir(r.name, r.seed) / "predictions.npz"
        return [BT_PY, "scripts/train_backtranslator.py", "--arch", "mbart", "--source", src,
                "--out", f"outputs/bt/{src}_mbart", "--score", f"seed{r.seed}={pred}",
                "--score_name", out.name]
    return add(Job(f"bt_{run_id}", prio, (run_id,), cmd=cmd, out=out, heavy=False))


# ----------------------------------------------------------------------------- experiment plan
def build(seeds_extra=(), tables=("1", "2"), multilingual=False):
    """Register the jobs of the selected tables. Tables 1/2 = main runs, SignNMFlow-M and B-T;
    3/5 = ablations, 4 = measure, 7 = benchmark, 8 = NFE (all four on CSL-Daily)."""
    JOBS.clear()
    tables = {str(t) for t in tables}
    seeds = [0, *seeds_extra]
    csl = "csl_smplx"
    for s in seeds:
        p_main = 10 if s == 0 else 50
        p_abl = (20, 40) if s == 0 else (60, 60)
        # Tables 1/2: one model per dataset (the CSL run is also the "full" row of Tables 3-5, 8)
        need_main = tables & {"1", "2"}
        need_csl = need_main or tables & {"3", "4", "5", "7", "8"}
        for D in DATASETS:
            if need_main or (D == csl and need_csl):
                run_job(f"main_{SRC[D]}_s{s}", p_main, D, f"{PREFIX}_main_{D}", s)
                if need_main:
                    bt_job(f"main_{SRC[D]}_s{s}")
        main_csl = f"main_csl_s{s}"
        if tables & {"3", "5"}:   # Row "full / target" is the CSL main run itself.
            run_job(f"abl_full_velocity_s{s}", p_abl[0], csl, f"{PREFIX}_abl_full_velocity", s,
                    over=lambda m=main_csl: ["method.flow.prediction=velocity",
                                             *init_from(m, ("prototypes", "correction"))()],
                    deps=[main_csl])
            run_job(f"abl_mr_geodesic_s{s}", p_abl[0], csl, f"{PREFIX}_abl_mr_geodesic", s,
                    over=["method.flow.path=geodesic"])
            run_job(f"abl_cfm6d_target_s{s}", p_abl[0], csl, f"{PREFIX}_abl_cfm6d_target", s,
                    over=["method.flow.representation=6d", "method.flow.path=linear"])
            run_job(f"abl_mr_linear_s{s}", p_abl[0], csl, f"{PREFIX}_abl_mr_linear", s,
                    over=["method.flow.path=linear"])
            run_job(f"abl_cfm6d_velocity_s{s}", p_abl[0], csl, f"{PREFIX}_abl_cfm6d_velocity", s,
                    over=["method.flow.representation=6d", "method.flow.path=linear",
                          "method.flow.prediction=velocity"])
        if "8" in tables:         # NFE, sampling only (NFE 5 = main run)
            for nfe in (10, 20):
                run_job(f"nfe{nfe}_s{s}", 0, csl, f"{PREFIX}_nfe{nfe}", s,
                        over=lambda m=main_csl, n=nfe: [f"method.sampler.steps={n}", *init_from(m)()],
                        deps=[main_csl], trains=False)
        if "4" in tables:         # measure (K = 500, sigma = 0.5 is the main run)
            for sig in (0.25, 0.75):
                run_job(f"sigma{sig}_s{s}", p_abl[1], csl, f"{PREFIX}_measure_sigma{sig}", s,
                        over=lambda m=main_csl, v=sig: [f"method.measure.sigma={v}",
                                                         *init_from(m, ("prototypes",))()],
                        deps=[main_csl])
            for K in (250, 750):
                run_job(f"K{K}_s{s}", p_abl[1], csl, f"{PREFIX}_measure_K{K}", s, over=[f"method.measure.K={K}"])
    if multilingual and tables & {"1", "2"}:   # dropped from the default plan (user, 2026-11-18)
        # SignNMFlow-M (seed 0): one model on Phoenix + CSL + How2Sign, evaluated on each
        run_job("M_phoenix_s0", 30, "phoenix_smplx", f"{PREFIX}_M_phoenix_smplx", 0, over=["method.multilingual=true"])
        bt_job("M_phoenix_s0")
        for D in ("csl_smplx", "how2sign_smplx_soke"):
            j = run_job(f"M_{SRC[D]}_s0", 0, D, f"{PREFIX}_M_{D}", 0,
                        over=lambda: ["method.multilingual=true", *init_from("M_phoenix_s0")()],
                        deps=["M_phoenix_s0"], trains=False)
            bt_job(j.id)
    if "7" in tables:             # parameters / latency (shared GPU: latency is indicative only)
        add(Job("benchmark_s0", 0, ("main_csl_s0",), heavy=False,
                cmd=lambda: [sys.executable, "-m", "methods.signmflow.benchmark", "--ckpt", str(ck("main_csl_s0")),
                             "--data", "csl_smplx", "--n", "100"],
                out=JOBS_DIR / "benchmark_s0.log.done"))


# ----------------------------------------------------------------------------- runtime state
def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {}


def save_state(st):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def scan_runs():
    """(name, seed) of every live `scripts/run.py --method signmflow` process (any user of this box)."""
    live = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            a = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        a = [x.decode(errors="ignore") for x in a if x]
        if not any(x.endswith("scripts/run.py") for x in a) or "signmflow" not in a:
            continue
        get = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None
        name, seeds = get("--name"), get("--seeds") or "0"
        for s in seeds.split(","):
            live[(name, int(s))] = int(p.name)
    return live


def gpu_free_gb():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        return min(float(x) for x in o.split()) / 1024
    except Exception:
        return 1e9


def is_done(j: Job):
    return done_dir(j.name, j.seed) is not None if j.is_run else j.out.exists()


def resume_overrides(j: Job, base: list):
    """Re-use finished / partial stages of earlier attempts of the same job."""
    have = {o.split("=")[0] for o in base}
    found = {}
    for d in reversed(run_dirs(j.name, j.seed)):
        c = d / "checkpoints"
        for key, f in (("method.init.prototypes", "prototypes.pt"), ("method.init.correction", "correction_final.pt"),
                       ("method.correction.resume_from", "correction.pt"),
                       ("method.generator.resume_from", "generator.pt")):
            if key not in found and (c / f).exists():
                found[key] = c / f
    if "method.init.correction" in found:
        found.pop("method.correction.resume_from", None)
    if "method.init.generator" in have:
        found.pop("method.generator.resume_from", None)
    return [f"{k}={v}" for k, v in found.items() if k not in have]


def command(j: Job, st):
    if not j.is_run:
        return j.cmd()
    base = list(j.over() if callable(j.over) else j.over)
    o = list(base)
    if j.trains:
        o.append(f"method.generator.micro_batch={st.get(j.id, {}).get('micro', GEN_MICRO)}")
        o += resume_overrides(j, base)
    return [sys.executable, "scripts/run.py", "--method", "signmflow", "--data", j.data, "--name", j.name,
            "--seeds", str(j.seed), *o]


def statuses(st, live, children):
    out = {}
    for j in JOBS.values():
        js = st.get(j.id, {})
        if is_done(j):
            out[j.id] = "done"
        elif (j.is_run and (j.name, j.seed) in live) or j.id in children:
            out[j.id] = "running"
        elif js.get("failed"):
            out[j.id] = "failed"
        elif any(st.get(d, {}).get("failed") and not is_done(JOBS[d]) for d in j.deps):
            out[j.id] = "dead"      # a dependency was given up
        elif any(not is_done(JOBS[d]) for d in j.deps):
            out[j.id] = "blocked"
        elif js.get("retry_after", 0) > time.time():
            out[j.id] = "backoff"
        else:
            out[j.id] = "ready"
    return out


def order():
    return sorted(JOBS.values(), key=lambda j: (j.prio, list(JOBS).index(j.id)))


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- runner
def run(a):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    pidf = JOBS_DIR / "runner.pid"
    if pidf.exists():
        try:
            os.kill(int(pidf.read_text()), 0)
            sys.exit(f"runner already alive (pid {pidf.read_text()})")
        except (OSError, ValueError):
            pass
    pidf.write_text(str(os.getpid()))
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    children: dict[str, tuple] = {}   # job id -> (Popen, log path, start)
    st = load_state()
    log(f"runner started: {len(JOBS)} jobs, max {a.max_procs} processes")
    try:
        while True:
            # reap finished children
            for jid, (p, lp, t0) in list(children.items()):
                if p.poll() is None:
                    continue
                del children[jid]
                j, js = JOBS[jid], st.setdefault(jid, {})
                if not j.is_run and p.returncode == 0 and not j.out.exists():
                    j.out.write_text(lp.read_text()[-5000:])   # command jobs without own output file
                if is_done(j):
                    log(f"DONE {jid} ({(time.time() - t0) / 3600:.1f} h)")
                    if j.is_run:
                        subprocess.run([sys.executable, "scripts/aggregate.py", "--name", j.name],
                                       cwd=ROOT, capture_output=True)
                    continue
                tail = lp.read_text(errors="ignore")[-20000:]
                oom = "OutOfMemoryError" in tail or "CUDA out of memory" in tail
                js["attempts"] = js.get("attempts", 0) + 1
                js["oom"] = js.get("oom", 0) + int(oom)
                if oom and j.trains:
                    js["micro"] = max(MIN_GEN_MICRO, js.get("micro", GEN_MICRO) // 2)
                lim = MAX_OOM_ATTEMPTS if oom else MAX_ATTEMPTS
                js["failed"] = js["attempts"] >= lim
                js["retry_after"] = time.time() + (900 if oom else 300)
                log(f"FAIL {jid} rc={p.returncode} oom={oom} attempt={js['attempts']} "
                    f"{'-> given up' if js['failed'] else '-> will resume'} (log {lp})")
                save_state(st)
            live = scan_runs()
            stat = statuses(st, live, children)
            n_busy = len(live) + sum(1 for jid in children if not JOBS[jid].is_run)
            pending = [j for j in JOBS.values() if stat[j.id] in ("ready", "blocked", "backoff", "running")]
            if not pending:
                log("all jobs finished (done or failed)")
                break
            if n_busy < a.max_procs:
                for j in order():
                    if stat[j.id] != "ready":
                        continue
                    free = gpu_free_gb()
                    need = a.min_free_gb if j.heavy else 3.0
                    if free < need:
                        break   # wait for memory, keep the priority order
                    cmd = command(j, st)
                    n = st.setdefault(j.id, {}).get("attempts", 0)
                    lp = JOBS_DIR / "logs" / f"{j.id}.{n}.log"
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(lp, "w"), stderr=subprocess.STDOUT,
                                         start_new_session=True)
                    children[j.id] = (p, lp, time.time())
                    st[j.id]["last_cmd"] = " ".join(cmd)
                    save_state(st)
                    log(f"START {j.id} (prio {j.prio}, free {free:.1f} GB) -> {lp}")
                    break   # one launch per tick, memory settles before the next decision
            time.sleep(a.tick)
    finally:
        pidf.unlink(missing_ok=True)


def status(a):
    st, live = load_state(), scan_runs()
    stat = statuses(st, live, {})
    for j in order():
        js = st.get(j.id, {})
        extra = ""
        if j.is_run:
            d = done_dir(j.name, j.seed) or (run_dirs(j.name, j.seed) or [None])[-1]
            if d is not None and (d / "log.txt").exists():
                lines = (d / "log.txt").read_text(errors="ignore").strip().splitlines()
                extra = lines[-1][:110] if lines else ""
        att = f"att={js.get('attempts', 0)} mb={js.get('micro', GEN_MICRO)}" if j.is_run and j.trains else ""
        print(f"{stat[j.id]:8s} p{j.prio:<3d} {j.id:28s} {att:14s} {extra}")
    from collections import Counter
    print(dict(Counter(stat.values())), f"| live run.py: {len(live)} | GPU free {gpu_free_gb():.1f} GB")


def plan(a):
    st = load_state()
    for j in order():
        try:
            c = " ".join(command(j, st)) if all(is_done(JOBS[d]) for d in j.deps) else f"(after {', '.join(j.deps)})"
        except Exception as e:
            c = f"(unresolved: {e})"
        print(f"p{j.prio:<3d} {j.id:28s} {c}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "status", "plan"])
    ap.add_argument("--max-procs", type=int, default=1)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--tick", type=int, default=60)
    ap.add_argument("--extra-seeds", default="", help="seeds after 0 (lower priority), e.g. 1,2")
    ap.add_argument("--tables", default="1,2", help="paper tables to run: any of 1,2,3,4,5,7,8")
    ap.add_argument("--multilingual", action="store_true", help="also run SignNMFlow-M (Tables 1/2 M rows)")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    build(tuple(int(s) for s in a.extra_seeds.split(",") if s), tuple(t for t in a.tables.split(",") if t),
          multilingual=a.multilingual)
    {"run": run, "status": status, "plan": plan}[a.action](a)


if __name__ == "__main__":
    main()

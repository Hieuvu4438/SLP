"""One dispatcher + one worker, persistent serial queue; stdlib only."""

from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# An absolute floor independent of editable plan/timer settings.
EARLIEST = datetime.fromisoformat("2026-11-06T00:00:00+07:00").timestamp()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if datetime.fromisoformat(plan["not_before"]).tzinfo is None:
        raise ValueError("not_before needs an explicit timezone")
    if plan["max_processes"] != 2 or plan["seed"] != 0:
        raise ValueError("pilot is fixed to one seed and two processes")
    hours = plan["hours"]
    if hours is not None and (
        isinstance(hours, bool)
        or not isinstance(hours, (int, float))
        or not math.isfinite(hours)
        or hours <= 0
    ):
        raise ValueError("wall limit must be null (unlimited) or finite positive hours")
    policy = plan.get("gpu_policy", {})
    if policy.get("mode") != "measured_memory":
        raise ValueError("GPU admission must use measured_memory, not a fixed GiB/utilization gate")
    for key, default in (("reserve_fraction", 0.1), ("retry_growth_fraction", 0.2)):
        value = policy.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("invalid GPU memory fraction")
    if policy.get("poll_seconds", 30) <= 0 or (
        policy.get("probe_timeout_hours") is not None and policy["probe_timeout_hours"] <= 0
    ):
        raise ValueError("invalid memory probe timing")
    seen = set()
    total = 0
    for job in plan["jobs"]:
        if job["id"] in seen or set(job["needs"]) - seen:
            raise ValueError("duplicate job or nonsequential dependency")
        if job["timeout_hours"] is not None and job["timeout_hours"] <= 0:
            raise ValueError("invalid job timeout")
        if job["timeout_hours"] is not None:
            total += job["timeout_hours"]
        seen.add(job["id"])
    if hours is not None and total > hours:
        raise ValueError("job allocations exceed wall budget")
    return plan


def check_gate(plan, now=None):
    now = time.time() if now is None else now
    floor = max(EARLIEST, datetime.fromisoformat(plan["not_before"]).timestamp())
    if now < floor:
        raise RuntimeError("FORBIDDEN: no pilot job before 2026-11-06 00:00 Asia/Ho_Chi_Minh")


def gpu_status(index=0):
    # Called only when no worker exists: this query is the sole second process.
    result = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError("GPU resource query failed: " + result.stderr[:500])
    try:
        total, free, utilization = map(float, result.stdout.strip().split(","))
    except ValueError as error:
        raise RuntimeError("invalid GPU resource response") from error
    if not all(math.isfinite(x) for x in (total, free, utilization)) or not 0 <= free <= total:
        raise RuntimeError("invalid GPU resource values")
    return {
        "total_bytes": int(total * 2**20),
        "free_bytes": int(free * 2**20),
        "free_gib": free / 1024,
        "total_gib": total / 1024,
        "utilization_percent": utilization,
    }


def wall_expired(state, now=None):
    deadline = state.get("deadline_epoch")
    return deadline is not None and (time.time() if now is None else now) >= deadline


def wait_for_gpu(plan, state, save, *, required_bytes=1, sleep=time.sleep):
    policy = plan["gpu_policy"]
    while not wall_expired(state):
        snapshot = gpu_status(plan["gpu"])
        budget = int(snapshot["free_bytes"] * (1 - policy.get("reserve_fraction", 0.1)))
        state["gpu_wait"] = {
            **snapshot,
            "budget_bytes": budget,
            "required_budget_bytes": required_bytes,
            "checked_epoch": time.time(),
            "admission": "measured_memory",
            "utilization_is_gate": False,
        }
        if budget >= required_bytes:
            state.pop("waiting_for", None)
            save()
            return budget
        if required_bytes > int(
            snapshot["total_bytes"] * (1 - policy.get("reserve_fraction", 0.1))
        ):
            raise RuntimeError("measured/probed memory requirement exceeds total GPU capacity")
        state["waiting_for"] = (
            f"VRAM: usable budget {budget / 2**30:.2f} GiB, "
            f"job/probe requires {required_bytes / 2**30:.2f} GiB; "
            f"utilization {snapshot['utilization_percent']:.0f}% is informational only"
        )
        save()
        sleep(policy.get("poll_seconds", 30))
    return None


def worker_environment(plan, gpu, deadline, budget=None):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(plan["gpu"]) if gpu else "",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_ENABLE_PARALLEL_LOADING="false",
        HF_HUB_DISABLE_XET="1",
        PYTHONUNBUFFERED="1",
        PILOT_QUEUE_MANAGED="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    env.pop("PILOT_JOB_DEADLINE_EPOCH", None)
    if deadline is not None:
        env["PILOT_JOB_DEADLINE_EPOCH"] = str(deadline)
    if budget is not None:
        env["PILOT_GPU_MEMORY_BUDGET_BYTES"] = str(budget)
    return env


def admit_gpu_job(plan, plan_path, root, state, job, save, resource_check):
    identity = job["id"]
    state["waiting_job"] = identity
    profiles = state.setdefault("memory_profiles", {})
    profile = profiles.get(identity)
    required = 1 if profile is None else profile["peak_reserved_bytes"]
    attempt = 0
    while not wall_expired(state):
        budget = wait_for_gpu(plan, state, save, required_bytes=required)
        if budget is None:
            return None
        if profile is not None:
            return budget
        resource_check()
        attempt += 1
        folder = root / "memory_probes" / identity
        folder.mkdir(parents=True, exist_ok=True)
        # Unique attempt IDs preserve failures across dispatcher restarts.
        record = folder / f"{time.time_ns()}.json"
        probe_hours = plan["gpu_policy"].get("probe_timeout_hours")
        deadline = None if probe_hours is None else time.time() + probe_hours * 3600
        if state["deadline_epoch"] is not None:
            deadline = (
                state["deadline_epoch"]
                if deadline is None
                else min(deadline, state["deadline_epoch"])
            )
        state["memory_probe"] = {
            "job": identity,
            "attempt_this_dispatcher": attempt,
            "status": "running",
            "budget_bytes": budget,
            "report": str(record),
            "started_epoch": time.time(),
        }

        def on_start(pid):
            state["memory_probe"]["pid"] = pid
            save()

        code, reason = run_child(
            [
                sys.executable,
                "-m",
                "methods.temporal_plan.pilot.worker",
                "--plan",
                str(plan_path),
                "--job",
                identity,
                "--memory-probe",
                "--probe-report",
                str(record),
            ],
            record.with_suffix(".log"),
            deadline=deadline,
            env=worker_environment(plan, True, deadline, budget),
            resource_check=resource_check,
            on_start=on_start,
        )
        state["memory_probe"].update(exit_code=code, ended_epoch=time.time(), reason=reason)
        payload = json.loads(record.read_text()) if record.exists() else {}
        if code == 0 and reason is None and payload.get("status") == "fit":
            profile = {**payload, "report": str(record)}
            profiles[identity] = profile
            state["memory_probe"]["status"] = "fit"
            required = max(1, profile["peak_reserved_bytes"])
            save()
            # Recheck free memory after the probe has released its CUDA context.
            continue
        if code == 75 and reason is None and payload.get("status") == "oom":
            maximum = int(
                state["gpu_wait"]["total_bytes"]
                * (1 - plan["gpu_policy"].get("reserve_fraction", 0.1))
            )
            if budget >= maximum:
                state["jobs"][identity] = {
                    "status": "failed",
                    "reason": "memory diagnostic cannot fit total GPU capacity",
                    "exit_code": code,
                }
                state["memory_probe"]["status"] = "failed"
                save()
                return None
            required = min(
                maximum,
                max(
                    int(budget * (1 + plan["gpu_policy"].get("retry_growth_fraction", 0.2))),
                    payload.get("required_lower_bound_bytes", 0),
                ),
            )
            state["memory_probe"].update(
                status="waiting_for_memory", next_required_budget_bytes=required
            )
            save()
            # No immediate retry at the same memory budget. No real training was saved.
            continue
        state["memory_probe"]["status"] = "failed"
        state["jobs"][identity] = {
            "status": "failed",
            "reason": reason or "memory_probe_failed",
            "exit_code": code,
            "probe_report": str(record),
        }
        save()
        return None
    return None


def stop_worker(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
    return child.wait()


def run_child(
    command, log, *, deadline, env, resource_check=lambda: None, on_start=lambda pid: None, tick=5
):
    with Path(log).open("ab", buffering=0) as output:
        child = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )

        def interrupt(signum, frame):
            stop_worker(child)
            raise KeyboardInterrupt

        previous = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
        reason = None
        try:
            on_start(child.pid)
            while child.poll() is None:
                if deadline is not None and time.time() >= deadline:
                    reason = "hard_timeout"
                    stop_worker(child)
                    break
                try:
                    resource_check()
                except Exception as error:
                    reason = "resource_guard: " + str(error)
                    stop_worker(child)
                    break
                time.sleep(tick)
            code = child.wait()  # a real process handle, never a guessed sleep
        finally:
            if child.poll() is None:
                stop_worker(child)
            for s, old in previous.items():
                signal.signal(s, old)
        return code, reason


def recover_interrupted_jobs(root, state):
    """Resume committed training after an explicit service restart, not unknown failures."""
    for identity, item in state["jobs"].items():
        interrupted = item["status"] in ("running", "waiting_for_memory") or (
            item["status"] == "failed"
            and item.get("reason")
            in ("dispatcher interrupted", "dispatcher interrupted; no automatic retry")
        )
        pointer = (
            root / "evaluator_resume/current.json"
            if identity == "evaluator"
            else root / identity / "resume/current.json"
        )
        if interrupted and identity in ("map", "sample", "evaluator") and pointer.exists():
            item.update(
                status="pending",
                reason="service restarted; resume committed optimizer/RNG/data cursor",
            )
            item.pop("pid", None)


def dispatch(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_plan(plan_path)
    check_gate(plan)  # before even creating a run/claiming a GPU
    root = Path(plan["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "dispatcher.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / "queue_state.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"started_epoch": time.time(), "jobs": {}, "status": "running"}
    )
    recover_interrupted_jobs(root, state)
    if state.get("status") in ("completed", "inconclusive", "expired") and not any(
        item["status"] == "pending" for item in state["jobs"].values()
    ):
        return
    state["deadline_epoch"] = (
        None if plan["hours"] is None else state["started_epoch"] + plan["hours"] * 3600
    )
    state["dispatcher_pid"] = os.getpid()
    state["status"] = "running"
    # A service restart may continue pending jobs; a crashed job is never retried.
    for item in state["jobs"].values():
        if item["status"] == "running":
            item.update(status="failed", reason="dispatcher interrupted; no automatic retry")
    atomic_json(path, state)

    def resources():
        free = shutil.disk_usage(root).free / 2**30
        if free < plan["min_free_gib"]:
            raise RuntimeError(f"free disk {free:.1f} GiB below reserve")
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 2**30
        if plan.get("max_output_gib") is not None and size > plan["max_output_gib"]:
            raise RuntimeError(f"pilot outputs {size:.1f} GiB exceed cap")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for job in plan["jobs"]:
            identity = job["id"]
            if identity in state["jobs"] and state["jobs"][identity]["status"] not in (
                "waiting_for_memory",
                "pending",
            ):
                continue
            if wall_expired(state):
                state["status"] = "expired"
                break
            failed = [
                n for n in job["needs"] if state["jobs"].get(n, {}).get("status") != "completed"
            ]
            if failed:
                state["jobs"][identity] = {
                    "status": "skipped",
                    "reason": "dependency",
                    "needs": failed,
                }
                atomic_json(path, state)
                continue
            while True:
                budget = None
                if job.get("gpu", False):
                    try:
                        budget = admit_gpu_job(
                            plan,
                            plan_path,
                            root,
                            state,
                            job,
                            lambda: atomic_json(path, state),
                            resources,
                        )
                    except RuntimeError as error:
                        state["jobs"][identity] = {"status": "failed", "reason": str(error)}
                        atomic_json(path, state)
                        break
                    if budget is None and not wall_expired(state):
                        break
                if wall_expired(state):
                    state["status"] = "expired"
                    break
                resources()
                started = time.time()
                deadline = (
                    None if job["timeout_hours"] is None else started + job["timeout_hours"] * 3600
                )
                if state["deadline_epoch"] is not None:
                    deadline = (
                        state["deadline_epoch"]
                        if deadline is None
                        else min(state["deadline_epoch"], deadline)
                    )
                state.pop("waiting_job", None)
                env = worker_environment(plan, job.get("gpu", False), deadline, budget)
                state["jobs"][identity] = {
                    "status": "running",
                    "started_epoch": started,
                    "deadline_epoch": deadline,
                }

                def on_start(pid):
                    state["jobs"][identity]["pid"] = pid
                    atomic_json(path, state)

                code, reason = run_child(
                    [
                        sys.executable,
                        "-m",
                        "methods.temporal_plan.pilot.worker",
                        "--plan",
                        str(plan_path),
                        "--job",
                        identity,
                    ],
                    root / f"{identity}.log",
                    deadline=deadline,
                    env=env,
                    resource_check=resources,
                    on_start=on_start,
                )
                if code == 76 and reason is None:
                    # The worker will restore optimizer/RNG/cursor from its last atomic
                    # commit. Do not retry incomplete legacy runs without recovery.
                    maximum = int(
                        state["gpu_wait"]["total_bytes"]
                        * (1 - plan["gpu_policy"].get("reserve_fraction", 0.1))
                    )
                    if budget >= maximum:
                        state["jobs"][identity].update(
                            status="failed", reason="CUDA OOM at full device budget", exit_code=code
                        )
                        atomic_json(path, state)
                        break
                    required = min(
                        maximum,
                        int(budget * (1 + plan["gpu_policy"].get("retry_growth_fraction", 0.2))),
                    )
                    state.setdefault("resource_retries", []).append(
                        {
                            "job": identity,
                            "at": time.time(),
                            "exit_code": code,
                            "previous_budget_bytes": budget,
                            "required_budget_bytes": required,
                            "recovery": "optimizer/RNG/data cursor",
                        }
                    )
                    state["memory_profiles"][identity]["peak_reserved_bytes"] = max(
                        state["memory_profiles"][identity]["peak_reserved_bytes"], required
                    )
                    state["jobs"][identity].update(
                        status="waiting_for_memory",
                        exit_code=code,
                        reason="recoverable CUDA OOM; waiting before resuming committed optimizer state",
                    )
                    atomic_json(path, state)
                    continue
                state["jobs"][identity].update(
                    status="completed" if code == 0 and reason is None else "failed",
                    exit_code=code,
                    reason=reason,
                    ended_epoch=time.time(),
                )
                atomic_json(path, state)
                break
        if state["status"] != "expired":
            state["status"] = (
                "completed"
                if all(r["status"] == "completed" for r in state["jobs"].values())
                else "inconclusive"
            )
    except BaseException as error:
        state.update(status="inconclusive", error=f"{type(error).__name__}: {error}")
        for item in state["jobs"].values():
            if item["status"] == "running":
                item.update(
                    status="failed", reason="dispatcher interrupted", ended_epoch=time.time()
                )
        raise
    finally:
        atomic_json(path, state)
        if state["status"] != "completed":
            (root / "QUEUE_RESULT.md").write_text(
                "Pilot INCONCLUSIVE / incomplete. Inspect queue_state.json and per-job logs. "
                "No automatic retry, no scientific rejection of the method from an incomplete run.\n"
            )
        lock.close()
        for s, handler in old_handlers.items():
            signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "status", "validate"])
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read_plan(args.plan)
    if args.command == "run":
        dispatch(args.plan)
    elif args.command == "status":
        path = Path(plan["output"]) / "queue_state.json"
        print(path.read_text() if path.exists() else "ARMED/PREPARED: no job has started")
    else:
        print(
            json.dumps(
                {
                    "valid": True,
                    "jobs": len(plan["jobs"]),
                    "wall_limit_hours": plan["hours"],
                    "allocated_hours": (
                        None
                        if any(j["timeout_hours"] is None for j in plan["jobs"])
                        else sum(j["timeout_hours"] for j in plan["jobs"])
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()

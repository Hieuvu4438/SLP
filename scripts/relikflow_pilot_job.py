"""Wait for GPU headroom and run one PHOENIX RelIKFlow pilot stage at a time."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB_DIR = ROOT / "outputs" / "jobs"
CONFIG = ROOT / "configs" / "experiments" / "relikflow" / "phoenix_pilot.yaml"
SOURCE = ROOT / "data" / "processed" / "madis" / "Phoenix_2014T"
DEFAULT_OUTPUT = ROOT / "outputs" / "relikflow" / "phoenix_pilot_seed17"
MIN_FREE_MIB = 24 * 1024
MAX_GPU_RETRIES = 2
DISTRIBUTED_ENV = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
}


class Stopped(Exception):
    pass


def utc_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def interrupt(signum, frame):
    raise Stopped(f"signal {signum}")


def query_memory(uuid):
    result = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    return int(result.stdout.strip())


def stages(output, config):
    base = [sys.executable, "-u", "-m", "methods.relikflow"]
    manifest = output / "input" / "phoenix.jsonl"
    common = [
        "--profile",
        "pilot",
        "--config",
        str(config),
        "--output",
        str(output),
        "--seed",
        "17",
        "--resume",
    ]
    plan = [
        ("prepare", ["--manifest", str(manifest)]),
        ("fit-planner", ["--oof", "3", "--fit-full"]),
        ("build-sources", ["--arms", "C"]),
        ("fit-bank", ["--split", "train"]),
        ("train-correction", ["--arms", "C"]),
        ("train-generator", ["--arms", "C"]),
        ("sample", ["--split", "dev", "--arms", "C", "--nfe", "5", "--debug"]),
        ("evaluate", ["--split", "dev", "--arms", "C", "--nfe", "5"]),
    ]
    return [
        (
            "export-manifest",
            [
                sys.executable,
                "-u",
                "-m",
                "methods.relikflow.phoenix_manifest",
                "--source",
                str(SOURCE),
                "--output",
                str(manifest),
            ],
        )
    ] + [(name, [*base, name, *common, *args]) for name, args in plan]


def child_env(uuid):
    env = {key: value for key, value in os.environ.items() if key not in DISTRIBUTED_ENV}
    env.update(
        CUDA_VISIBLE_DEVICES=uuid,
        HF_HUB_OFFLINE="1",
        PYTHONUNBUFFERED="1",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    return env


def run_stage(command, env, lock, log_path):
    # Inherit the lock so a supervisor crash cannot allow another training process.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open("a") as log,
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            pass_fds=(lock.fileno(),),
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as proc,
    ):
        try:
            code = proc.wait()
        except Stopped:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def run(uuid, output, interval):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "job_state.json"
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    completed = []
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config_sha256"] != config_hash:
            raise ValueError(
                "Job config changed. Resume with its original config or a new output path."
            )
        completed = saved["completed"]
    state = {
        "pid": os.getpid(),
        "gpu_uuid": uuid,
        "min_free_mib": MIN_FREE_MIB,
        "seed": 17,
        "config_sha256": config_hash,
        "completed": completed,
    }

    def record(status, **fields):
        state.update(fields, status=status, updated_at=utc_now())
        tmp = path.with_name(".job_state.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(path)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with (JOB_DIR / ".relikflow-pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another RelIKFlow pilot is waiting or running") from None
        try:
            for name, command in stages(output, CONFIG):
                if name in completed:
                    continue
                log_path = output / "logs" / f"{name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                for attempt in range(1, MAX_GPU_RETRIES + 2):
                    last_log = 0
                    while True:
                        free = None
                        try:
                            free = query_memory(uuid)
                        except (OSError, ValueError, subprocess.SubprocessError) as exc:
                            print(f"{utc_now()} GPU query failed: {exc}", flush=True)
                        if free is not None and free >= MIN_FREE_MIB:
                            break
                        record("waiting_for_vram", stage=name, free_mib=free, attempt=attempt)
                        now = time.monotonic()
                        if not last_log or now - last_log >= 300:
                            print(
                                f"{utc_now()} {name}: waiting for {MIN_FREE_MIB} MiB, free={free}",
                                flush=True,
                            )
                            last_log = now
                        time.sleep(interval)
                    with log_path.open("a") as log:
                        log.write(f"\n{utc_now()} {name}: attempt {attempt} (free={free} MiB)\n")
                    start = log_path.stat().st_size
                    record(
                        "running",
                        stage=name,
                        free_mib=free,
                        stage_log=str(log_path),
                        attempt=attempt,
                        error=None,
                    )
                    print(
                        f"{utc_now()} {name}: start (free={free} MiB), log={log_path}", flush=True
                    )
                    try:
                        run_stage(command, child_env(uuid), lock, log_path)
                    except subprocess.CalledProcessError:
                        with log_path.open("rb") as log:
                            log.seek(start)
                            oom = any(b"cuda out of memory" in line.lower() for line in log)
                        if not oom or attempt > MAX_GPU_RETRIES:
                            raise
                        record("retry_gpu_oom", error="CUDA out of memory")
                        time.sleep(interval)
                        continue
                    break
                completed.append(name)
                record("stage_complete", stage=name)
            record("completed", stage=None)
        except Stopped as exc:
            record("interrupted", error=str(exc))
            return 130
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            record("failed", error=str(exc))
            raise
    return 0


def recover_legacy_worker(output, unit):
    """Hand off one failed pre-retry worker to the current restart-aware runner."""
    state_path = output / "job_state.json"
    if not state_path.is_file():
        return
    state = json.loads(state_path.read_text())
    if state["status"] != "failed" or "attempt" in state:
        return
    stage = state.get("stage")
    if stage not in {name for name, _ in stages(output, CONFIG)}:
        return
    log_path = output / "logs" / f"{stage}.log"
    if not log_path.is_file():
        return
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - 128 * 1024))
        tail = log.read()
    last_attempt = tail.rfind(f" {stage}: start (free=".encode())
    if last_attempt < 0 or b"cuda out of memory" not in tail[last_attempt:].lower():
        return
    marker = output / f".gpu-recovery-{int(state['pid'])}"
    try:
        with marker.open("x") as handle:
            handle.write(utc_now() + "\n")
    except FileExistsError:
        return
    try:
        subprocess.run(["systemctl", "--user", "--no-block", "start", unit], check=True)
    except (OSError, subprocess.SubprocessError):
        marker.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gpu-uuid")
    mode.add_argument("--recover-unit", metavar="SERVICE")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if args.recover_unit:
        recover_legacy_worker(args.output.resolve(), args.recover_unit)
        return 0
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    return run(args.gpu_uuid, args.output.resolve(), args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

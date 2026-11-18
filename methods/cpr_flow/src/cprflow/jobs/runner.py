"""Persistent DAG scheduler. A global flock enforces the two-job cap across invocations."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cprflow.data.cache import file_hash
from cprflow.jobs.gates import check_outputs, evaluate_gate
from cprflow.jobs.schema import atomic_json, load_plan, source_digest
from cprflow.jobs.storage import check_free_space, directory_bytes
from cprflow.jobs.store import Store


def process_token(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def live(pid, token):
    return bool(pid and token and process_token(pid) == str(token))


def resume_checkpoint(directory):
    if not directory:
        return None
    paths = list(Path(directory).glob("*.pt"))
    candidates = []
    for path in paths:
        meta_path = Path(str(path) + ".json")
        try:
            meta = json.loads(meta_path.read_text())
            if meta["size"] != path.stat().st_size or meta["mtime_ns"] != path.stat().st_mtime_ns:
                raise ValueError("stale checkpoint header")
        except (OSError, ValueError, KeyError):
            from cprflow.training.checkpoint import load_checkpoint

            state = load_checkpoint(path)
            meta = {"successful_optimizer_step": state["successful_optimizer_step"]}
        candidates.append((meta["successful_optimizer_step"], path.name == "last.pt", str(path)))
    return max(candidates)[2] if candidates else None


class Runner:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.plan = load_plan(self.path)
        self.store = Store(self.plan)
        self.by_id = {j["id"]: j for j in self.plan["jobs"]}
        self.handles = {}
        self.stopping = False
        self.root = Path(self.plan["state_dir"])
        self.usage_time = 0.0
        self.usage = 0

    def disk_reason(self, job=None, running=()):
        disk = self.plan["disk"]
        try:
            free = check_free_space(self.plan["root"], 0)
            reserve = disk["min_free_gib"] + (job["disk_gib"] if job else 2 * len(running))
            if free < reserve:
                return f"Disk reserve: {free:.1f}GiB free; require {reserve:.1f}GiB"
            if time.monotonic() - self.usage_time > disk.get("scan_seconds", 60):
                self.usage = directory_bytes(disk["tracked_paths"]) / 2**30
                self.usage_time = time.monotonic()
            future = (
                (job["disk_gib"] + sum(self.by_id[r["id"]]["disk_gib"] for r in running))
                if job
                else 2 * len(running)
            )
            if self.usage + future >= disk["budget_gib"]:
                return f"Workspace budget: {self.usage:.1f}/{disk['budget_gib']}GiB; no files automatically deleted"
        except OSError as exc:
            return str(exc)
        return None

    def claims(self):
        path = Path(self.plan["global_lock"] + ".claims.json")
        claims = json.loads(path.read_text()) if path.exists() else {}
        return {
            k: v
            for k, v in claims.items()
            if live(v.get("pid"), v.get("token"))
            or not v.get("pid")
            and v["lease_until"] > time.time()
        }

    def write_claims(self, claims):
        atomic_json(self.plan["global_lock"] + ".claims.json", claims)

    def resources(self, job, running):
        claims = self.claims()
        occupied = list(claims.values())
        if sum(r["slots"] for r in occupied) + job["slots"] > self.plan["max_concurrent"]:
            return "Waiting for process slots"
        held = {k for r in occupied for k in r["locks"]}
        if held & set(job["locks"]):
            return "Waiting for exclusive artifact/run lock"
        if job["gpu_gib"]:
            try:
                gpu = self.plan["environment"].get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={gpu}",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                )
                free = float(result.splitlines()[0]) / 1024
                requested = job["gpu_gib"] + sum(r["gpu_gib"] for r in occupied)
                if free < requested + 2:
                    return f"Waiting for VRAM reserve: free {free:.1f}GiB, reservations {requested:.1f}+2GiB"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return f"GPU unavailable: {exc}"
        return self.disk_reason(job, running)

    def finalize(self, row):
        job = self.by_id[row["id"]]
        result_path = Path(row["result"])
        code = 125
        reason = "Process ended without durable receipt; retry/resume required"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if (
                result.get("plan_hash") != self.store.hash
                or result.get("attempt") != row["attempt"]
            ):
                raise ValueError("Worker receipt provenance mismatch")
            code = result["returncode"]
            reason = result.get("error")
        if code == 0:
            try:
                check_outputs(job)
            except (ValueError, OSError) as exc:
                code = 126
                reason = str(exc)
        if code == 0:
            self.store.set(row["id"], "succeeded", reason=None)
        elif row["failures"] + 1 < job["max_attempts"] or code == 75:
            self.store.set(
                row["id"],
                "retry_wait",
                reason=f"Exit {code}: {reason}",
                next_retry=time.time() + job["retry_seconds"],
                failures=row["failures"] + (code != 75),
            )
        else:
            self.store.set(
                row["id"], "failed", reason=f"Exit {code}: {reason}", failures=row["failures"] + 1
            )
        claims = self.claims()
        claims = {
            k: v
            for k, v in claims.items()
            if not (v["plan_hash"] == self.store.hash and v["job"] == row["id"])
        }
        self.write_claims(claims)
        handle = self.handles.pop(row["id"], None)
        if handle:
            handle[0].wait()
            handle[1].close()

    def launch(self, job, row):
        attempt = row["attempt"] + 1
        while (self.root / "attempts" / job["id"] / str(attempt)).exists():
            attempt += 1
        self.store.set(job["id"], "pending", attempt=attempt, reason="Launching")
        argv = list(job["argv"])
        if job.get("resume_dir"):
            checkpoint = resume_checkpoint(job["resume_dir"])
            if checkpoint:
                argv += ["--resume", checkpoint]
            elif job.get("warm_start"):
                argv += ["--warm-start", job["warm_start"]]
        elif job.get("warm_start"):
            argv += ["--warm-start", job["warm_start"]]
        work = self.root / "attempts" / job["id"] / str(attempt)
        work.mkdir(parents=True, exist_ok=False)
        atomic_json(work / "argv.json", argv)
        result = work / "result.json"
        log_path = work / "output.log"
        stream = log_path.open("a")
        env = {
            **os.environ,
            **self.plan["environment"],
            "CPRFLOW_MIN_FREE_GIB": str(self.plan["disk"]["min_free_gib"]),
        }
        claims = self.claims()
        claim = f"{self.store.hash}:{job['id']}:{attempt}"
        claims[claim] = {
            "plan_hash": self.store.hash,
            "job": job["id"],
            "pid": None,
            "token": None,
            "lease_until": time.time() + 30,
            "slots": job["slots"],
            "gpu_gib": job["gpu_gib"],
            "locks": job["locks"],
        }
        self.write_claims(claims)
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cprflow.jobs.worker",
                "--plan",
                str(self.path),
                "--job",
                job["id"],
                "--attempt",
                str(attempt),
                "--argv-file",
                str(work / "argv.json"),
                "--result",
                str(result),
                "--admission",
                str(work / "admitted.json"),
            ],
            cwd=self.plan["root"],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        token = process_token(p.pid)
        claims[claim].update(pid=p.pid, token=token)
        self.write_claims(claims)
        self.handles[job["id"]] = (p, stream)
        self.store.set(
            job["id"],
            "running",
            attempt=attempt,
            pid=p.pid,
            pid_start=token,
            log=str(log_path),
            result=str(result),
            reason=None,
        )
        atomic_json(work / "admitted.json", {"pid": p.pid, "token": token})

    def stop_running(self, reason):
        for row in self.store.rows().values():
            if row["state"] == "running" and live(row["pid"], row["pid_start"]):
                os.killpg(row["pid"], signal.SIGTERM)
                self.store.set(row["id"], "running", reason=reason)

    def run(self, watch=False):
        lock_path = Path(self.plan["global_lock"])
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another scheduler owns the global process limit")
            atomic_json(
                self.root / "scheduler.json",
                {
                    "pid": os.getpid(),
                    "pid_start": process_token(os.getpid()),
                    "plan": str(self.path),
                },
            )
            previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

            def stop(signum, frame):
                self.stopping = True

            for sig in previous:
                signal.signal(sig, stop)
            stopping_since = None
            last_reason = None
            try:
                while True:
                    for row in self.store.rows().values():
                        if row["state"] == "running":
                            handle = self.handles.get(row["id"])
                            if handle:
                                handle[0].poll()
                            if not live(row["pid"], row["pid_start"]):
                                self.finalize(row)
                    rows = self.store.rows()
                    running = [r for r in rows.values() if r["state"] == "running"]
                    reason = self.disk_reason(running=running)
                    if self.stopping or reason:
                        if stopping_since is None:
                            stopping_since = time.monotonic()
                            self.stop_running(reason or "Scheduler stopping; checkpoint and resume")
                        if not running:
                            if self.stopping or not watch:
                                return
                            time.sleep(self.plan["poll_seconds"])
                            continue
                        if time.monotonic() - stopping_since > 180:
                            for r in running:
                                if live(r["pid"], r["pid_start"]):
                                    os.killpg(r["pid"], signal.SIGKILL)
                        time.sleep(0.2)
                        continue
                    stopping_since = None
                    if source_digest(self.plan["root"]) != self.plan["source_digest"]:
                        last_reason = "Implementation changed; no new jobs admitted. Restore source or create a new campaign."
                    elif any(file_hash(p) != h for p, h in self.plan["config_hashes"].items()):
                        last_reason = "Frozen job config changed; create a new campaign"
                    else:
                        last_reason = None
                        for job in sorted(self.plan["jobs"], key=lambda j: -j["priority"]):
                            row = self.store.rows()[job["id"]]
                            if row["state"] in ("running", "succeeded", "failed", "cancelled"):
                                continue
                            if row["next_retry"] > time.time():
                                continue
                            parents = [self.store.rows()[p] for p in job["deps"]]
                            if any(p["state"] != "succeeded" for p in parents):
                                self.store.set(
                                    job["id"],
                                    "waiting",
                                    reason="Dependencies: "
                                    + ", ".join(
                                        p["id"] + ":" + p["state"]
                                        for p in parents
                                        if p["state"] != "succeeded"
                                    ),
                                )
                                continue
                            try:
                                evidence = [evaluate_gate(g) for g in job["gates"]]
                            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                                self.store.set(job["id"], "waiting", reason=str(exc))
                                continue
                            if not job["argv"]:
                                atomic_json(
                                    self.root / "gates" / f"{job['id']}.json",
                                    {"evidence": evidence},
                                )
                                self.store.set(job["id"], "succeeded", reason=None)
                                continue
                            running = [
                                r for r in self.store.rows().values() if r["state"] == "running"
                            ]
                            blocked = self.resources(job, running)
                            if blocked:
                                self.store.set(job["id"], "waiting", reason=blocked)
                                continue
                            self.launch(job, row)
                    current = self.store.rows()
                    atomic_json(
                        self.root / "status.json",
                        {
                            "scheduler_pid": os.getpid(),
                            "scheduler_reason": last_reason,
                            "max_concurrent": self.plan["max_concurrent"],
                            "workspace_gib": self.usage,
                            "jobs": list(current.values()),
                        },
                    )
                    running = [r for r in current.values() if r["state"] == "running"]
                    if all(
                        r["state"] in ("succeeded", "failed", "cancelled") for r in current.values()
                    ):
                        return
                    if not running and not watch:
                        return
                    time.sleep(self.plan["poll_seconds"])
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

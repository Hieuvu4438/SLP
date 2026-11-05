"""Minimal file-based job queue for the shared single GPU.

Keeps the number of concurrent jobs bounded (default 1 worker) so we never
oversubscribe the shared RTX 5880 (another user's job holds ~27 GB).

  python scripts/jobq.py add  <name> "<shell command>"   # append a job (runs from repo root)
  python scripts/jobq.py run  [--workers 1]             # process the queue until empty
  python scripts/jobq.py status                          # show queue / running / done / failed
  python scripts/jobq.py cancel <name>                   # drop a pending job

State lives in outputs/jobs/: queue.jsonl (pending), state.jsonl (event log),
<name>.log (stdout+stderr of each job).
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "outputs" / "jobs"
QUEUE = JOBS / "queue.jsonl"
STATE = JOBS / "state.jsonl"
LOCK = JOBS / ".lock"


@contextmanager
def locked():
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _log(event):
    event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE, "a") as fh:
        fh.write(json.dumps(event) + "\n")


def add(name, cmd):
    with locked():
        q = _read(QUEUE)
        if any(j["name"] == name for j in q):
            sys.exit(f"job {name!r} already queued")
        q.append({"name": name, "cmd": cmd})
        QUEUE.write_text("".join(json.dumps(j) + "\n" for j in q))
        _log({"event": "queued", "name": name, "cmd": cmd})
    print("queued", name)


def pop():
    with locked():
        q = _read(QUEUE)
        if not q:
            return None
        job, rest = q[0], q[1:]
        QUEUE.write_text("".join(json.dumps(j) + "\n" for j in rest))
        _log({"event": "start", "name": job["name"], "pid": os.getpid()})
        return job


def run(workers):
    procs = {}
    while True:
        while len(procs) < workers:
            job = pop()
            if job is None:
                break
            log = open(JOBS / f"{job['name']}.log", "w")
            p = subprocess.Popen(job["cmd"], shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            procs[p] = (job, log, time.time())
            print(f"[jobq] start {job['name']} pid={p.pid}", flush=True)
        if not procs:
            print("[jobq] queue empty, exiting", flush=True)
            return
        time.sleep(5)
        for p in list(procs):
            if p.poll() is not None:
                job, log, t0 = procs.pop(p)
                log.close()
                status = "done" if p.returncode == 0 else "failed"
                with locked():
                    _log({"event": status, "name": job["name"], "rc": p.returncode,
                          "minutes": round((time.time() - t0) / 60, 1)})
                print(f"[jobq] {status} {job['name']} rc={p.returncode}", flush=True)


def status():
    events = _read(STATE)
    last = {}
    for e in events:
        last[e["name"]] = e
    print("pending:", [j["name"] for j in _read(QUEUE)])
    for name, e in last.items():
        if e["event"] != "queued":
            print(f"  {e['event']:7s} {name:28s} {e.get('minutes', '')} {e['time']}")


def cancel(name):
    with locked():
        q = [j for j in _read(QUEUE) if j["name"] != name]
        QUEUE.write_text("".join(json.dumps(j) + "\n" for j in q))
        _log({"event": "cancelled", "name": name})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("add"); a1.add_argument("name"); a1.add_argument("command")
    a2 = sub.add_parser("run"); a2.add_argument("--workers", type=int, default=1)
    sub.add_parser("status")
    a4 = sub.add_parser("cancel"); a4.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "add":
        add(a.name, a.command)
    elif a.cmd == "run":
        run(a.workers)
    elif a.cmd == "status":
        status()
    else:
        cancel(a.name)


if __name__ == "__main__":
    main()

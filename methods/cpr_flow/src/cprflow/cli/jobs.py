"""Create, inspect, start, stop and retry a bounded CPR-Flow experiment campaign."""

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cprflow.cli.common import print_json
from cprflow.jobs.compiler import create_plan
from cprflow.jobs.runner import Runner, live
from cprflow.jobs.schema import load_plan
from cprflow.jobs.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--id", default="joint3_standard")
    create.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    create.add_argument("--preset", choices=("standard", "confirmatory"), default="standard")
    create.add_argument("--seeds", default="17")
    create.add_argument("--optimized", action="store_true")
    create.add_argument("--device", default="cuda")
    create.add_argument("--budget-gib", type=float, default=100)
    create.add_argument(
        "--operator-waived-checks",
        action="store_true",
        help="Record the explicit operator request; no fake verified/approved artifacts",
    )
    for name in ("run", "start", "status", "stop", "retry"):
        command = sub.add_parser(name)
        command.add_argument("--plan", required=True)
        if name == "run":
            command.add_argument("--watch", action="store_true")
        if name == "status":
            command.add_argument("--json", action="store_true")
        if name == "retry":
            command.add_argument("--job", required=True)
    args = parser.parse_args()
    if args.action == "create":
        path = create_plan(
            args.root,
            args.id,
            args.preset,
            tuple(map(int, args.seeds.split(","))),
            args.device,
            args.budget_gib,
            args.operator_waived_checks,
            args.optimized,
        )
        plan = load_plan(path)
        print_json(
            {
                "plan": str(path),
                "jobs": len(plan["jobs"]),
                "max_concurrent": 2,
                "scope": plan["scope"],
                "seeds": plan["seeds"],
                "started": False,
            }
        )
        return
    path = Path(args.plan).resolve()
    plan = load_plan(path)
    store = Store(plan)
    owner_path = Path(plan["state_dir"]) / "scheduler.json"
    owner = json.loads(owner_path.read_text()) if owner_path.exists() else {}
    alive = live(owner.get("pid"), owner.get("pid_start"))
    if args.action == "run":
        Runner(path).run(args.watch)
    elif args.action == "start":
        if alive:
            print_json({"scheduler_alive": True, "pid": owner["pid"], "already_running": True})
            return
        logfile = Path(plan["state_dir"]) / "scheduler.log"
        with logfile.open("a") as stream:
            proc = subprocess.Popen(
                [sys.executable, "-m", "cprflow.cli.jobs", "run", "--plan", str(path), "--watch"],
                cwd=plan["root"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Scheduler exited; inspect {logfile}")
            if owner_path.exists() and json.loads(owner_path.read_text()).get("pid") == proc.pid:
                break
            time.sleep(0.05)
        print_json(
            {"scheduler_pid": proc.pid, "plan": str(path), "log": str(logfile), "max_concurrent": 2}
        )
    elif args.action == "stop":
        if alive:
            os.kill(owner["pid"], signal.SIGTERM)
        print_json(
            {
                "stop_requested": alive,
                "pid": owner.get("pid"),
                "policy": "finish current optimizer step, checkpoint, preserve queue",
            }
        )
    elif args.action == "retry":
        row = store.rows()[args.job]
        if row["state"] == "running":
            raise ValueError("Cannot reset a live job")
        store.set(args.job, "pending", failures=0, next_retry=0, reason="User requested retry")
        print_json({"retry_queued": args.job})
    else:
        rows = list(store.rows().values())
        result = {
            "scheduler_alive": alive,
            "scheduler_pid": owner.get("pid"),
            "plan": str(path),
            "max_concurrent": 2,
            "counts": dict(collections.Counter(r["state"] for r in rows)),
            "running": [
                r for r in rows if r["state"] == "running" and live(r["pid"], r["pid_start"])
            ],
            "terminal_needs_reconciliation": [
                r["id"]
                for r in rows
                if r["state"] == "running" and not live(r["pid"], r["pid_start"])
            ],
            "waiting": [
                {"id": r["id"], "reason": r["reason"]}
                for r in rows
                if r["state"] in ("waiting", "failed", "retry_wait")
            ],
        }
        if args.json:
            result["jobs"] = rows
        else:
            result["waiting"] = result["waiting"][:12]
        print_json(result)


if __name__ == "__main__":
    main()

"""Final-test access guard; the append-only access ledger is not experiment source."""

import datetime
import fcntl
import json
import subprocess
from pathlib import Path


def require_clean_tree():
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=True
    )
    dirty = [
        line
        for line in status.stdout.splitlines()
        if not (line[:2] in (" M", "??", "M ") and line[3:] == "results/test_access_log.jsonl")
    ]
    if dirty:
        raise ValueError(
            "Final-test evaluation requires a clean committed source tree: " + ", ".join(dirty[:5])
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def authorize_final_test(config, dataset, output, explicit=False):
    if not explicit:
        raise ValueError("Test is sealed; --final-test records explicit final evaluation intent")
    waived = config["run"].get("manual_gate_policy", "required") == "operator_waived"
    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if waived
        else require_clean_tree()
    )
    source_state = {"status": "clean_committed"}
    if waived:
        from cprflow.jobs.schema import source_digest

        source_state = {
            "status": "operator_authorized_worktree_snapshot",
            "source_digest": source_digest(Path(__file__).resolve().parents[3]),
            "reference": config["run"]["policy_reference"],
        }
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    target = root / "results/test_access_log.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(
            json.dumps(
                {
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "experiment": config["run"]["id"],
                    "method": "cpr_flow",
                    "data": dataset,
                    "commit": revision,
                    "output": str(output),
                    "event": "final_test_opened",
                    "source_state": source_state,
                }
            )
            + "\n"
        )

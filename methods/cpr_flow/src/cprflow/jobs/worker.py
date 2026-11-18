"""One payload process, durable completion receipt, no shell or nested job launcher."""

import argparse
import os
import runpy
import signal
import sys
import time
import traceback
from pathlib import Path

from cprflow.data.cache import digest
from cprflow.jobs.schema import atomic_json, load_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--argv-file", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--admission", required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    import json

    argv = json.loads(Path(args.argv_file).read_text())
    started = time.time()
    code = 1
    error = None

    def stop(signum, frame):
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.chdir(plan["root"])
    try:
        deadline = time.monotonic() + 15
        while not Path(args.admission).is_file():
            if time.monotonic() > deadline:
                raise RuntimeError("Uncommitted scheduler admission expired")
            time.sleep(0.05)
        if Path(argv[0]).resolve() != Path(sys.executable).resolve():
            raise ValueError("Worker interpreter must match the submitted environment")
        if argv[1] == "-m":
            sys.argv = [argv[2], *argv[3:]]
            runpy.run_module(argv[2], run_name="__main__", alter_sys=True)
        elif argv[1] == "-c":
            sys.argv = ["-c", *argv[3:]]
            exec(compile(argv[2], "<local-job>", "exec"), {"__name__": "__main__"})
        else:
            raise ValueError("Unsupported command; no implicit subprocesses")
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            error = str(exc)
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        atomic_json(
            args.result,
            {
                "job": args.job,
                "attempt": args.attempt,
                "plan_hash": digest(plan),
                "returncode": code,
                "started": started,
                "finished": time.time(),
                "error": error,
            },
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

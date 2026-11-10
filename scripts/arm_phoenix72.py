"""Freeze reviewed pilot source and arm a user systemd timer. Never starts a job directly."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from methods.temporal_plan.pilot.queue import read_plan, atomic_json

plan_path = repo/"configs/pilots/phoenix72_20261106.json"
plan = read_plan(plan_path)
root = Path(plan["output"])
source = root/"source"
if (root/"ARMED.json").exists():
    raise SystemExit("Already armed. Inspect ARMED.json; never overwrite an active pilot.")
if source.exists():
    raise SystemExit("Snapshot already exists. Inspect the previous setup; refusing to replace it.")
for required in (Path(plan["annotations"])/"phoenix14t.pami0.train.annotations_only.gzip",
                 Path(plan["poses"])/"phoenix14t.train",
                 Path(plan["smplx"])/"smplx/SMPLX_NEUTRAL.npz"):
    if not required.is_file():
        raise FileNotFoundError(required)
if shutil.disk_usage(repo).free/2**30 < plan["min_free_gib"]+8:
    raise RuntimeError("Insufficient disk for model download and reserve")
(source/"methods").mkdir(parents=True)
(source/"methods/__init__.py").write_text("")
shutil.copytree(repo/"methods/temporal_plan", source/"methods/temporal_plan",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
plan["method_config"] = str(source/"methods/temporal_plan/config.yaml")
frozen = root/"plan.json"
atomic_json(frozen, plan)
hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(source.rglob("*")) if p.is_file()}
atomic_json(root/"source_manifest.json", hashes)
zone = ZoneInfo("Asia/Ho_Chi_Minh")
now = datetime.now(zone)
# Leave a preparation margin even if setup happens after midnight.
start = max(datetime(2026,11,6,0,1,tzinfo=zone), now+timedelta(minutes=3))
start = start.replace(microsecond=0)
unit = "slp-phoenix-pilot72-20261106"
units = Path.home()/".config/systemd/user"
units.mkdir(parents=True,exist_ok=True)
service = f"""[Unit]
Description=Sequential PHOENIX-2014T experiment pilot

[Service]
Type=simple
WorkingDirectory={source}
ExecStart={sys.executable} -m methods.temporal_plan.pilot.queue run --plan {frozen}
Environment=PYTHONPATH={source}:{repo / 'src'}
Environment=OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
Environment=TOKENIZERS_PARALLELISM=false HF_ENABLE_PARALLEL_LOADING=false HF_HUB_DISABLE_XET=1
Environment=PYTHONUNBUFFERED=1
KillMode=control-group
TimeoutStopSec=45
Restart=no
RuntimeMaxSec={str(plan['hours'])+'h' if plan['hours'] is not None else 'infinity'}
TasksMax=256

[Install]
WantedBy=default.target
"""
# UTC calendar avoids depending on the server's local timezone/systemd zone parsing.
utc = start.astimezone(ZoneInfo("UTC"))
timer = f"""[Unit]
Description=Start PHOENIX pilot after permitted midnight

[Timer]
OnCalendar={utc.strftime('%Y-%m-%d %H:%M:%S')} UTC
AccuracySec=1s
Persistent=true
Unit={unit}.service

[Install]
WantedBy=timers.target
"""
(units/f"{unit}.service").write_text(service)
(units/f"{unit}.timer").write_text(timer)
for command in (["systemd-analyze","--user","verify",str(units/f"{unit}.service"),str(units/f"{unit}.timer")],
                ["systemctl","--user","daemon-reload"],
                ["systemctl","--user","enable","--now",f"{unit}.timer"],
                ["systemctl","--user","show",f"{unit}.timer","-p","NextElapseUSecRealtime","-p","ActiveState"],
                ["systemctl","--user","is-active",f"{unit}.service"]):
    result = subprocess.run(command,capture_output=True,text=True,timeout=20)
    print(result.stdout.strip() or result.stderr.strip())
    # Service must still be inactive; enabling the TIMER is not starting the SERVICE.
    if command[-1] == f"{unit}.service" and command[2] == "is-active":
        if result.stdout.strip() != "inactive":
            raise RuntimeError("Unexpected service state during setup")
    elif result.returncode:
        raise RuntimeError("systemd setup failed")
atomic_json(root/"ARMED.json", {"unit":unit,"armed_at":now.isoformat(),
    "scheduled_local":start.isoformat(),"scheduled_utc":utc.isoformat(),
    "not_before":plan["not_before"],"plan":str(frozen),"source":str(source),
    "python":sys.executable,"process_policy":"one dispatcher plus one worker; threads are not processes",
    "window_policy":("no overall wall-clock deadline; stop when queue completes" if plan['hours'] is None else
                     f"{plan['hours']} wall-clock hours from first dispatcher launch; GPU waiting counts"),
    "queue_state_exists":(root/"queue_state.json").exists()})
print(json.dumps(json.loads((root/"ARMED.json").read_text()),indent=2))

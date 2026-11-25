"""Run the gated How2Sign pilot, one CUDA subprocess at a time."""

import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

from methods.slp_multirate.data import read_jsonl, write_json
from methods.slp_multirate.evaluation import _comparison_pair

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "f417e5563320b2cc8aabe4329d986b238809067f"
ARMS = ("B", "D", "no_anchor")
MIN_CLIPS = 100


class StopPilot(Exception):
    def __init__(self, stage, report):
        self.stage, self.report = stage, report


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    path = pathlib.Path(f"/proc/{pid}")
    try:
        uid = path.stat().st_uid
        _, _, after_comm = (path / "stat").read_text().rpartition(")")
        values = after_comm.split()
        return {"pid": pid, "uid": uid, "start_ticks": int(values[19]), "state": values[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity(pid):
    value = process_identity(pid)
    return (
        {k: value[k] for k in ("pid", "uid", "start_ticks")}
        if value and value["state"] != "Z"
        else None
    )


def nvidia(*args):
    proc = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return [
        tuple(field.strip() for field in line.split(","))
        for line in proc.stdout.splitlines()
        if line.strip()
    ]


def gpu_status(gpu):
    gpus = {uuid: int(free) for uuid, free in nvidia("--query-gpu=uuid,memory.free")}
    if gpu not in gpus:
        raise ValueError(f"Configured GPU is missing: {gpu}")
    apps = nvidia("--query-compute-apps=gpu_uuid,pid")
    pids = {int(pid) for uuid, pid in apps if uuid == gpu}
    return pids, gpus[gpu]


def evaluate_gate(base, cand, stage, expected):
    _, effects, failures = _comparison_pair(base, cand)
    a, b = [read_jsonl(folder / "per_clip.jsonl") for folder in (base, cand)]

    def ids(rows):
        return {(x["clip_id"], x["sample_index"]) for x in rows}

    expected_ids = {(clip, 0) for clip in expected}
    reasons = []
    if any(failures) or ids(a) != expected_ids or ids(b) != expected_ids:
        reasons.append("Evaluation failures or missing paired predictions")
    if len({x["origin_video_id"] for x in a}) < 15:
        reasons.append("Fewer than 15 independently grouped source videos")

    def metric(name, minimum_n=MIN_CLIPS):
        value = effects[name]
        if value["n"] < minimum_n:
            reasons.append(f"{name}: only {value['n']} paired clips, need {minimum_n}")
            return None
        return value

    minimum_gain = 0.05 if stage == "p1" else 0.03
    primary = metric("event_tip_mm")
    if primary and (primary["effect"] < minimum_gain or primary["ci95"][0] <= 0):
        reasons.append(f"event_tip_mm needs gain >= {minimum_gain:.0%} and CI lower bound > 0")
    for name, max_loss in (("wrist_mm", 0.05), ("outside_tip_mm", 0.03), ("palm_degrees", 0.05)):
        score = metric(name)
        if score and score["effect"] < -max_loss:
            reasons.append(f"{name}: regression exceeds {max_loss:.0%}")
    if stage == "p1":
        excluded = metric("event_tip_excluding_anchors_mm")
        if excluded and (excluded["effect"] <= 0 or excluded["ci95"][0] <= 0):
            reasons.append("No positive paired gain outside the union of oracle neighborhoods")
    if stage == "p2_text":
        for name in ("event_recall", "event_f1"):
            score = metric(name)
            if score and score["effect"] < -0.02:
                reasons.append(f"{name}: regression exceeds two percentage points")
        score = metric("duration_relative")
        if score and score["effect"] < -0.02:
            reasons.append("Median duration error regressed by over two percentage points")
        # Timing MAE is computed only for matches, so track it beside recall and F1.
        score = metric("timing_mae_ms", min(30, MIN_CLIPS))
        if score and score["candidate"] - score["baseline"] > 40.0:
            reasons.append("Matched event timing degraded by more than one 25-fps frame")
    return {
        "status": "STOP" if reasons else "GO",
        "stage": stage,
        "reasons": reasons,
        "effects": effects,
        "paired_clips_expected": len(expected_ids),
    }


def check_gate_coverage(manifest, rig_path, cohort):
    import torch
    from methods.slp_multirate.anchors import build_oracle_anchors, clip_pose, score_scales
    from methods.slp_multirate.config import resolve
    from methods.slp_multirate.data import CanonicalDataset
    from methods.slp_multirate.geometry import Rig

    torch.set_num_threads(2)
    rig = Rig.load(rig_path)
    cfg = resolve("p1", "D")
    training = CanonicalDataset(manifest, rig, ("train",), cohort)
    evaluation = CanonicalDataset(manifest, rig, ("dev_gate",), cohort)
    if not len(training):
        return {"event": 0, "away_from_anchors": 0, "outside": 0}
    scales = score_scales(training, rig, cfg["anchors"])
    counts = {"event": 0, "away_from_anchors": 0, "outside": 0}
    for clip in evaluation:
        clock = clip["timestamps_s"]
        valid = clip_pose(clip, rig).tip_mask.any(dim=(1, 2))
        events = clock.new_tensor([e["time_s"] for e in clip["row"]["events"]])
        windows = ((clock[:, None] - events).abs() <= 3 / 25 + 1e-7).any(1)
        adjacent = torch.zeros_like(valid)
        covered = True
        for policy in ("event", "uniform"):
            a = build_oracle_anchors(clip, rig, dict(cfg["anchors"], policy=policy), scales)
            covered &= bool(a.feature.any())
            keyframes = (a.time[a.slot] * (len(clock) - 1)).round()
            adjacent |= ((torch.arange(len(clock))[:, None] - keyframes).abs() <= 1).any(1)
        if covered:
            counts["event"] += int((valid & windows).any())
            counts["away_from_anchors"] += int((valid & windows & ~adjacent).any())
            counts["outside"] += int((valid & ~windows).any())
    return counts


class Pilot:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.prepared = self.out / "prepared"
        self.manifest = self.prepared / "canonical" / "manifest.jsonl"
        self.rig = self.manifest.parent / "rig.json"
        self.cache = self.out / "text_cache"
        self.spec = json.loads(args.watch.read_text())
        self.released = set()

    def state(self, phase, **details):
        report = {
            "phase": phase,
            "time": now(),
            "launcher_pid": os.getpid(),
            "released_pids": sorted(self.released),
            **details,
        }
        write_json(self.out / "state.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)

    def wait_gpu(self, stage):
        while True:
            try:
                pids, free = gpu_status(self.spec["gpu"])
                for original in self.spec["processes"]:
                    pid = original["pid"]
                    if pid not in pids or identity(pid) != original:
                        self.released.add(pid)
                own = sorted(
                    pid
                    for pid in pids
                    if (current := identity(pid)) and current["uid"] == os.getuid()
                )
                if self.released and not own and free >= self.args.min_free_mib:
                    return
                self.state(
                    "waiting_gpu",
                    stage=stage,
                    own_gpu_pids=own,
                    watcher_pids=[p["pid"] for p in self.spec["processes"]],
                    reported_free_mib=free,
                )
            except (subprocess.SubprocessError, OSError, ValueError) as err:
                self.state("waiting_gpu_probe", stage=stage, error=str(err))
            time.sleep(self.args.poll_seconds)

    def run(self, tag, argv, *, gpu=False, cli=True, offline=True):
        if gpu:
            self.wait_gpu(tag)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=self.spec["gpu"] if gpu else "",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1" if offline else "0",
            PYTHONPATH=str(ROOT),
        )
        if cli:
            wrapper = (
                "import torch; from methods.slp_multirate.cli import main; main(); "
                "torch.cuda.synchronize() if torch.cuda.is_initialized() else None; "
                "print(f'peak_cuda_reserved_bytes={torch.cuda.max_memory_reserved()}', "
                "file=__import__('sys').stderr) if torch.cuda.is_initialized() else None"
            )
            command = [sys.executable, "-u", "-c", wrapper, *map(str, argv)]
        else:
            command = [sys.executable, "-u", *map(str, argv)]
        clock = time.monotonic()
        log = self.out / f"{tag}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{now()} {tag}\n")
            stream.flush()
            with subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            ) as child:
                self.state("running", stage=tag, child_pid=child.pid, gpu=gpu, log=str(log))
                code = child.wait()
        if code:
            raise RuntimeError(f"{tag} exited {code}. See {log}")
        self.state("stage_done", stage=tag, elapsed_seconds=round(time.monotonic() - clock, 2))

    def decided(self, name, report):
        write_json(self.out / f"{name}.json", report)
        self.state("gate", stage=name, result=report["status"], reasons=report["reasons"])
        if report["status"] != "GO":
            raise StopPilot(name, report)

    def flags(self, pilot, arm):
        return [
            "--pilot",
            pilot,
            "--variant",
            arm,
            "--manifest",
            self.manifest,
            "--cohort",
            self.prepared / f"{pilot}.json",
            "--text-cache",
            self.cache,
            "--run-dir",
            self.out / "runs",
            "--threads",
            "2",
            "--device",
            "cuda:0",
            "--set",
            f"text.revision={REVISION}",
            "--set",
            "body.stride=2",
            "--set",
            "data.num_workers=0",
            "--set",
            "train.keep_last=1",
        ]

    def train(self, pilot, arm, stage, deps=(), profile=False):
        path = self.out / "runs" / pilot / arm / "17" / "stages" / stage
        latest = path / "latest.pt"
        from methods.slp_multirate.config import resolve
        from methods.slp_multirate.pipeline import load_torch

        updates = resolve(pilot, arm)["train"]["stages"][stage]["total_updates"]
        args = ["train", *self.flags(pilot, arm), "--stage", stage, *deps]
        tag = f"{pilot}_{arm}_{stage}"
        if profile and not latest.exists():
            self.run(tag + "_profile50", [*args, "--stop-after", 50], gpu=True)
        if latest.exists():
            state = load_torch(latest)
            finished = state["update"] == updates
            del state
            if finished:
                if not (path / "best.pt").is_file():
                    raise RuntimeError(f"Missing best checkpoint in {path}")
                return path / "best.pt"
            args.append("--resume")
        self.run(tag, args, gpu=True)
        return path / "best.pt"

    def evaluate(self, pilot, arm, protocol, deps, calibration):
        path = self.out / "eval" / pilot / arm / protocol
        if (path / "summary.json").exists() and (path / "latency.json").exists():
            return path
        if path.exists():
            path.rename(path.with_name(path.name + f".incomplete-{time.time_ns()}"))
        self.run(
            f"{pilot}_{arm}_{protocol}",
            [
                "evaluate",
                *self.flags(pilot, arm),
                "--protocol",
                protocol,
                "--split",
                "dev_gate",
                "--calibration",
                calibration,
                "--out",
                path,
                *deps,
            ],
            gpu=True,
        )
        return path

    def prep(self):
        artifact = self.prepared / "prep_report.json"
        if not artifact.exists():
            self.run(
                "prepare_how2sign",
                [
                    "scripts/slp_multirate_prepare_how2sign.py",
                    "--packed",
                    "data/processed/madis/How2Sign",
                    "--source",
                    "/home/shared_data/sign_language/How2Sign",
                    "--smplx",
                    self.args.smplx,
                    "--out",
                    self.prepared,
                ],
                cli=False,
            )
        info = json.loads(artifact.read_text())
        reasons = []
        for pilot in ("p1", "p2"):
            row = info["cohorts"][pilot]
            limits = {
                "dev_gate": MIN_CLIPS,
                "gate_video_groups": 15,
                "train": 512 if pilot == "p1" else 1500,
                "dev_tune": 128 if pilot == "p1" else 200,
            }
            for field, cap in limits.items():
                if row[field] < cap:
                    reasons.append(f"{pilot}/{field} {row[field]} < {cap}")
        if (
            info["evaluated_pose_frames"] < 100
            or not info["median_wrist_elbow_error_shoulder_widths"] < 0.65
            or not info["mean_correct_side_beats_swapped"] > 0.75
        ):
            reasons.append("2D side agreement and shoulder-aligned geometry verification failed")
        if not 0 < info["mean_event_window_fraction"] <= 0.55:
            reasons.append("Independent event windows absent or too dense")
        coverage = check_gate_coverage(self.manifest, self.rig, self.prepared / "p1.json")
        if any(n < MIN_CLIPS for n in coverage.values()):
            reasons.append(
                f"Insufficient scoreable P1 event/anchor-excluded/outside clips: {coverage}"
            )
        self.decided(
            "p0_gate",
            {
                "status": "STOP" if reasons else "GO",
                "reasons": reasons,
                "body_stride": 2,
                "data_audit": info,
                "p1_coverage": coverage,
            },
        )

    def encoder(self):
        ready = self.out / "encoder_ready.json"
        if ready.exists():
            return
        self.run(
            "prepare_text_encoder",
            [
                "-c",
                """
from huggingface_hub import snapshot_download
from methods.slp_multirate.config import resolve
from methods.slp_multirate.models import TextEncoder
import sys
encoder = 'facebook/mbart-large-cc25'
revision = sys.argv[1]
path = snapshot_download(encoder, revision=revision,
                         allow_patterns=['config.json', 'pytorch_model.bin',
                                         'tokenizer.json', 'sentencepiece.bpe.model'])
model = TextEncoder(resolve('p1', 'B', overrides=(f'text.revision={revision}',))['text'])
result = model(['The key is on the table.'])
assert result.hidden.isfinite().all() and result.hidden.shape[-1] == 1024
print(path, tuple(result.hidden.shape))
""",
                REVISION,
            ],
            cli=False,
            offline=False,
        )
        write_json(
            ready,
            {
                "model": "facebook/mbart-large-cc25",
                "revision": REVISION,
                "verified_cpu_forward": True,
            },
        )

    def cache_text(self, pilot):
        flags = self.flags(pilot, "B")
        self.run(
            f"{pilot}_cache_text",
            ["cache-text", *flags, "--split", "train,dev_tune,dev_gate"],
            gpu=True,
        )

    def calibration(self, pilot, model):
        key = self.out / f"{pilot}_calibration.json"
        if not key.exists():
            flags = self.flags(pilot, "B")
            flags[flags.index("cuda:0")] = "cpu"
            self.run(
                f"{pilot}_calibrate", ["calibrate-eval", *flags, "--fine-ckpt", model, "--out", key]
            )
        return key

    def execute(self):
        self.prep()
        self.encoder()
        self.cache_text("p1")
        p1 = {}
        for arm in ARMS:
            model = self.train("p1", arm, "fine", profile=arm == "B")
            if arm == "B":
                calibration = self.calibration("p1", model)
            p1[arm] = self.evaluate("p1", arm, "oracle", ["--fine-ckpt", model], calibration)
            expected = json.loads((self.prepared / "p1.json").read_text())["clips"]["dev_gate"]
            if arm == "D":
                self.decided("p1_D_vs_B", evaluate_gate(p1["B"], p1["D"], "p1", expected))
            if arm == "no_anchor":
                self.decided("p1_D_vs_no_anchor", evaluate_gate(p1[arm], p1["D"], "p1", expected))

        self.cache_text("p2")
        shared = ["--body-ckpt", self.train("p2", "B", "body")]
        length = ["--length-ckpt", self.train("p2", "B", "length")]
        expected = json.loads((self.prepared / "p2.json").read_text())["clips"]["dev_gate"]
        runs, gt, text = {}, {}, {}
        for arm in ARMS:
            anchor = (
                ["--anchors-ckpt", self.train("p2", arm, "anchors")] if arm != "no_anchor" else []
            )
            ckpts = [*shared, *anchor]
            fine = self.train("p2", arm, "fine", ckpts)
            args = [*ckpts, "--fine-ckpt", fine]
            runs[arm] = [*args, *length]
            if arm == "B":
                calibration = self.calibration("p2", fine)
            gt[arm] = self.evaluate("p2", arm, "predicted_gt_length", args, calibration)
            if arm != "B":
                self.decided(
                    f"p2_gt_D_vs_{'B' if arm == 'D' else arm}",
                    evaluate_gate(gt["B" if arm == "D" else arm], gt["D"], "p2_gt", expected),
                )
            text[arm] = self.evaluate("p2", arm, "text_only", runs[arm], calibration)
            if arm != "B":
                self.decided(
                    f"p2_text_D_vs_{'B' if arm == 'D' else arm}",
                    evaluate_gate(text["B" if arm == "D" else arm], text["D"], "p2_text", expected),
                )
        for arm in ARMS:
            src = self.out / f"semantics_{arm}.json"
            if not src.exists():
                self.run(
                    f"semantics_{arm}",
                    [
                        "scripts/slp_multirate_pilot_semantics.py",
                        "--manifest",
                        self.manifest,
                        "--cohort",
                        self.prepared / "p2.json",
                        "--text-cache",
                        self.cache,
                        "--variant",
                        arm,
                        "--text-revision",
                        REVISION,
                        "--smplx",
                        self.args.smplx,
                        "--out",
                        src,
                        *runs[arm],
                    ],
                    cli=False,
                    gpu=True,
                )
        from scripts.slp_multirate_pilot_semantics import judge

        self.decided("p2_semantic_gate", judge(self.out))
        self.state("completed", result="P2 passed; P3 requires a separate decision")

    def launch(self):
        # Serialize all launches of this pilot by haipd, independent of output directory.
        lock = self.out.parent / ".slp_multirate_gpu.lock"
        with lock.open("a") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state("started")
            try:
                self.execute()
            except StopPilot as halt:
                self.state("stopped_by_gate", stage=halt.stage, reasons=halt.report["reasons"])
            except Exception as exc:
                self.state("failed", error=str(exc))
                raise


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--watch", type=pathlib.Path, required=True)
    p.add_argument("--smplx", type=pathlib.Path, required=True)
    p.add_argument("--poll-seconds", type=float, default=20)
    p.add_argument("--min-free-mib", type=int, default=12288)
    Pilot(p.parse_args()).launch()

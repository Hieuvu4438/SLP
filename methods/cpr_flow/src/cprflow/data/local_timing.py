"""Resolve timing from server assets; otherwise label an operator-authorized assumption."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from cprflow.data.cache import file_hash
from cprflow.data.lmdb_adapter import LMDBAdapter
from cprflow.data.sources import audit_sources

VIDEO_ROOTS = {
    "csl_daily": [Path("/home/dongvk/datasets/CSL_Daily_Sentence_Crop/videos")],
    "how2sign": [
        Path("/home/shared_data/sign_language/How2Sign") / s / "raw_videos"
        for s in ("train", "eval", "test")
    ],
    "phoenix2014t": [
        Path("/home/dongvk/datasets/phoenix14T/videos_phoenix/videos") / s
        for s in ("train", "dev", "test")
    ],
}
PHOENIX_FRAMES = Path(
    "/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
)


def resolve_timing(config):
    from cprflow.data.preparation import source_paths

    dest = Path(config["paths"]["manifests"])
    evidence = []
    datasets = {}
    for dataset in config["data"]["datasets"]:
        index = {
            p.stem: p for root in VIDEO_ROOTS[dataset] if root.is_dir() for p in root.glob("*.mp4")
        }
        source = next(path for d, s, path in source_paths(config) if d == dataset and s == "train")
        adapter = LMDBAdapter(source, file_hash(source / "data.mdb"), trusted=True)
        checked = []
        matched = []
        for uid in adapter.clip_ids[:16]:
            record = adapter.read(uid)
            frames = len(record["body_features"])
            stem = Path(uid).stem
            video = index.get(stem)
            row = {"dataset": dataset, "uid": uid, "pose_frames": frames}
            if dataset == "phoenix2014t":
                raw = PHOENIX_FRAMES / "train" / stem
                count = sum(1 for _ in raw.glob("*.png")) if raw.is_dir() else 0
                row.update(
                    original_frame_directory=str(raw), original_frames=count, original_fps=25.0
                )
                if count == frames and count > 1:
                    matched.append(25.0)
            elif video is not None:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_frames,avg_frame_rate",
                        "-of",
                        "json",
                        str(video),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                )
                info = json.loads(result.stdout)["streams"][0]
                fps = float(Fraction(info["avg_frame_rate"]))
                count = int(info.get("nb_frames", 0))
                row.update(original_video=str(video), original_frames=count, original_fps=fps)
                if count == frames and count > 1:
                    matched.append(fps)
            else:
                row["status"] = "matching original clip metadata not found"
            checked.append(row)
            if len(checked) >= 8:
                break
        adapter.close()
        evidence.extend(checked)
        verified = (
            len(matched) == len(checked)
            and len(matched) >= 3
            and max(matched) - min(matched) < 0.01
        )
        fps = matched[0] if verified else (25.0 if dataset == "phoenix2014t" else 30.0)
        if not verified and config["data"]["require_verified_feature_fps"]:
            raise ValueError(
                f"Feature timing not established for {dataset}; operator policy does not permit assumptions"
            )
        datasets[dataset] = {
            "verified": verified,
            "feature_fps": fps,
            "verification_method": "Exact pose/original frame counts across local clips and source native FPS"
            if verified
            else "Explicit exploratory assumption of native video FPS; effective pose timing NOT verified",
            "operator_reference": config["run"].get("policy_reference", ""),
            "evidence": [],
        }
    report = dest / "local_timing_evidence.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"records": evidence, "no_corpus_copy": True}, indent=2))
    for record in datasets.values():
        record["evidence"] = [{"path": str(report.resolve()), "sha256": file_hash(report)}]
    timing = {
        "revision": audit_sources(config)["dataset"]["revision"],
        "datasets": datasets,
        "policy": config["run"].get("manual_gate_policy", "required"),
    }
    path = Path(config["paths"]["source_timing"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timing, indent=2))
    return timing

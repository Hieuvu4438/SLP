"""Matched-channel back-translation and fixed-noise prompt swaps for the How2Sign pilot."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from methods.slp_multirate.cli import create_cascade
from methods.slp_multirate.config import resolve
from methods.slp_multirate.data import CanonicalDataset, file_hash, stable_seed, write_json
from methods.slp_multirate.geometry import JOINTS, Rig, merge_local, rot6d_to_matrix, so3_log
from methods.slp_multirate.data import SMPLX_BODY_21, SMPLX_HAND_15
from methods.slp_multirate.training import split_hash
from slp.data.sign_smplx import load_stats, madis_crop
from slp.evaluation.multirate import _bootstrap_indices

BT = Path("outputs/bt/how2sign_mbart/best.pt")
BODY = [JOINTS.index(key) for key in SMPLX_BODY_21[-10:]]
HANDS = [JOINTS.index(f"{side}_{key}") for side in ("left", "right") for key in SMPLX_HAND_15]


def features(local, hand_mean, mean, std):
    aa = so3_log(local).cpu().numpy()
    pose = np.concatenate(
        (
            aa[:, BODY].reshape(-1, 30),
            aa[:, HANDS].reshape(-1, 90) - hand_mean,
            np.zeros((len(aa), 13), np.float32),
        ),
        axis=-1,
    )
    size = max(1, int(len(aa) * 24 / 25))
    ids = (np.arange(size) * 25 // 24).clip(max=len(pose) - 1)
    return madis_crop(pose[ids] - mean) / std


def wrong_prompts(rows):
    words = [set(r["text"].casefold().split()) for r in rows]
    matches = []
    for i, row in enumerate(rows):

        def score(j):
            other = rows[j]
            sim = len(words[i] & words[j]) / max(1, len(words[i] | words[j]))
            return (
                sim > 0.55,
                abs(other["frames"] - row["frames"]),
                -sim,
                stable_seed("prompt-swap", other["clip_id"]),
            )

        matches.append(
            rows[
                min(
                    (
                        j
                        for j in range(len(rows))
                        if rows[j]["origin_video_id"] != row["origin_video_id"]
                        and rows[j]["text"] != row["text"]
                    ),
                    key=score,
                )
            ]["text"]
        )
    return matches


@torch.inference_mode()
def score(args):
    from scripts.train_backtranslator import MBartBT, batches, seq_logprob

    cfg = resolve(
        "p2",
        args.variant,
        overrides=(
            "body.stride=2",
            "data.num_workers=0",
            "train.keep_last=1",
            f"text.revision={args.text_revision}",
        ),
    )
    rig = Rig.load(args.manifest.parent / "rig.json")
    dataset = CanonicalDataset(args.manifest, rig, ("dev_gate",), args.cohort, 25)
    if len(dataset) < 100:
        raise ValueError("Insufficient dev gate coverage for semantic pilot")
    ckpts = {
        stage: path
        for stage in ("length", "body", "anchors", "fine")
        if (path := getattr(args, stage + "_ckpt")) is not None
    }
    gpu = torch.device("cuda:0")
    cascade = create_cascade(
        cfg,
        rig,
        gpu,
        ckpts,
        split_hash(args.manifest, args.cohort),
        args.text_cache,
        require_fine_dependencies=True,
    )
    # Semantic evaluation uses identical face masking for reference and generated poses.
    model = np.load(args.smplx, allow_pickle=False)
    hm = np.r_[model["hands_meanl"], model["hands_meanr"]]
    mean, std = load_stats("third_party/SOKE/deps/stats")
    rows, pairs = dataset.rows, wrong_prompts(dataset.rows)
    collected = {key: [] for key in ("reference", "reference_wrong_text", "correct", "swapped")}
    for clip, wrong in zip(dataset, pairs, strict=True):
        row = clip["row"]
        key, text = row["clip_id"], row["text"]
        ref = merge_local(rot6d_to_matrix(clip["body_rot6d"]), rot6d_to_matrix(clip["hands_rot6d"]))
        reference = features(ref, hm, mean, std)
        collected["reference"].append((key, text, reference))
        collected["reference_wrong_text"].append((key, wrong, reference))
        for label, prompt in (("correct", text), ("swapped", wrong)):
            local = cascade.generate([prompt], [key])["pose"].local[0]
            collected[label].append((key, text, features(local, hm, mean, std)))
    del cascade
    gc.collect()
    torch.cuda.empty_cache()
    bt = MBartBT("en_XX").eval()
    bt.load_state_dict(torch.load(BT, map_location="cpu", weights_only=True, mmap=True))
    bt.to(gpu)
    results = {}
    for label, records in collected.items():
        results[label] = {}
        for names, texts, x, mask in batches(records, 4, False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = seq_logprob(bt, x.to(gpu), mask.to(gpu), texts)
            results[label].update(zip(names, scores, strict=True))
    write_json(
        args.out,
        {
            "bt_sha256": file_hash(BT),
            "manifest_sha256": file_hash(args.manifest),
            "cohort_sha256": file_hash(args.cohort),
            "face_channels": "zeroed_in_all_arms",
            "variant": args.variant,
            "scores": results,
            "videos": {r["clip_id"]: r["origin_video_id"] for r in rows},
        },
    )


def judge(out):
    names = ("B", "D", "no_anchor")
    data = {key: json.loads((out / f"semantics_{key}.json").read_text()) for key in names}
    expected = {
        key: data["D"][key]
        for key in ("bt_sha256", "manifest_sha256", "cohort_sha256", "face_channels", "videos")
    }
    if any(d[k] != v for d in data.values() for k, v in expected.items()):
        raise ValueError("Incompatible semantic evaluation cohorts")
    keys = sorted(expected["videos"])
    draws = list(_bootstrap_indices(keys, expected["videos"], np.random.default_rng(17), 2000))

    def evaluate(a, aa, b, bb):
        x = np.asarray([data[a]["scores"][aa][k] - data[b]["scores"][bb][k] for k in keys])
        if not np.isfinite(x).all():
            raise ValueError("Non-finite back-translation scores")
        return {
            "effect": float(x.mean()),
            "ci95": np.quantile([x[idx].mean() for idx in draws], [0.025, 0.975]).tolist(),
        }

    comparisons = {
        "reference_text_vs_wrong_text": evaluate("D", "reference", "D", "reference_wrong_text"),
        "D_conditioned_vs_prompt_swap": evaluate("D", "correct", "D", "swapped"),
        "D_vs_B": evaluate("D", "correct", "B", "correct"),
        "D_vs_no_anchor": evaluate("D", "correct", "no_anchor", "correct"),
    }
    reasons = []
    for key in ("reference_text_vs_wrong_text", "D_conditioned_vs_prompt_swap"):
        if comparisons[key]["ci95"][0] <= 0:
            reasons.append(f"{key}: positive prompt discrimination not demonstrated")
    for key in ("D_vs_B", "D_vs_no_anchor"):
        if comparisons[key]["effect"] < 0 or comparisons[key]["ci95"][0] < -0.03:
            reasons.append(
                f"{key}: BT log probability regressed or noninferiority was inconclusive"
            )
    return {
        "status": "STOP" if reasons else "GO",
        "reasons": reasons,
        "n": len(keys),
        "unit": "nats_per_token",
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for path in ("manifest", "cohort", "text-cache", "smplx", "out"):
        p.add_argument(f"--{path}", type=Path, required=True)
    for stage in ("length", "body", "anchors", "fine"):
        p.add_argument(f"--{stage}-ckpt", type=Path, required=stage != "anchors")
    p.add_argument("--variant", required=True, choices=("B", "D", "no_anchor"))
    p.add_argument("--text-revision", required=True)
    score(p.parse_args())

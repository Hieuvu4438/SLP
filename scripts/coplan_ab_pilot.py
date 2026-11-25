"""Sequential CSL-Daily A/B pilot and paired dev diagnostics."""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from methods.coplan_so3.data.dataset import collate
from methods.coplan_so3.factory import ROOT, build
from methods.coplan_so3.sampling import sample_plan
from methods.coplan_so3.training import (
    finalize_plan,
    fit_feature_statistics,
    restore_checkpoint,
    save_checkpoint,
    train_stage,
)
from methods.signmflow_kinematic.checkpoints import fingerprint, write_json
from methods.signmflow_kinematic.diagnostics import archive_values, bootstrap_mean, metrics_values
from scripts.train_backtranslator import MBartBT, batches, seq_logprob
from slp.data.sign_smplx import LOADERS, feats133, madis_crop

LOG = logging.getLogger("coplan_so3.pilot")
FPS = 30.0
SAMPLING_SEEDS = (0, 1, 2)
PLAN_NFE = 2


def select_rows(root, split, size, rng):
    rows = LOADERS["csl"](root, split)
    if not 2 <= size <= len(rows):
        raise ValueError(f"Requested {size} items from {len(rows)} {split} rows")
    indices = sorted(rng.choice(len(rows), size=size, replace=False))
    return [(rows[i][0], rows[i][1]) for i in indices]


@torch.no_grad()
def load_clips(rows, codec):
    items, features = [], []
    for info, path in rows:
        with path.open("rb") as file:
            raw = feats133(pickle.load(file))
        # Keep the legacy index selection, including its actual frame times.
        # Drop inserted duplicates so short clips retain strictly increasing time.
        indices = np.unique(madis_crop(np.arange(len(raw))))
        norm = (torch.from_numpy(raw[indices]) - codec.mean) / codec.std
        rotations, expression = codec.encode(norm)
        items.append(
            {
                "text": info["text"],
                "language": 1,
                "rotations": rotations,
                "expression": expression,
                "timestamps": torch.from_numpy(indices).float() / FPS,
            }
        )
        features.append(norm.numpy())
    return items, features


def wrong_sentences(inputs, lengths):
    tokens = [set(x["text"]) for x in inputs]
    replacements = []
    for i, item in enumerate(inputs):
        candidates = []
        for j, other in enumerate(inputs):
            if item["text"] == other["text"] or item["gloss"] == other["gloss"]:
                continue
            similarity = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
            gap = abs(int(lengths[i]) - int(lengths[j]))
            candidates.append((gap > 0.2 * lengths[i], similarity > 0.8, -similarity, gap, j))
        if not candidates:
            raise ValueError(f"No distinct dev negative for {item['name']}")
        replacements.append(min(candidates)[-1])
    return replacements


def feature_values(actual, expected, times):
    report = {}
    for group in ("wrists", "handshape", "palms", "relative"):
        diff = (actual[group] - expected[group]).unflatten(-1, (-1, 3))
        report[group + "_mean_l2"] = float(diff.norm(dim=-1).mean())
    dt = times.diff()[:, None]
    for group in ("wrists", "handshape"):
        velocity = actual[group].diff(dim=0) / dt
        speed = velocity.unflatten(-1, (-1, 3)).norm(dim=-1)
        report[group + "_speed_p95_m_s"] = float(torch.quantile(speed.flatten(), 0.95))
    return report


@torch.no_grad()
def coarse_predictions(model, codec, items, inputs, swaps, static, device):
    model.eval()
    output = {key: [] for key in ("static", "oracle", "posterior")}
    for seed in SAMPLING_SEEDS:
        output.update({f"correct_{seed}": [], f"wrong_{seed}": []})
    geometry = {key: [] for key in output}
    geometry["ground_truth"] = []
    leng_errors = []
    for i, item in enumerate(items):
        batch = collate([item]).to(device)
        other_text = inputs[swaps[i]]["text"]
        lengths = batch.lengths.expand(2)
        masks = batch.mask.expand(2, -1)
        times = batch.timestamps.expand(2, -1)
        language = batch.languages.expand(2)
        text, text_mask = model.encode_text([item["text"], other_text], language)
        log_length, _ = model.length_model(text[:1], text_mask[:1], language[:1])
        predicted_length = log_length.exp().round().clamp_min(1).item()
        leng_errors.append(abs(predicted_length - int(lengths[0])))
        z, zm = model.plan.encode_raw(
            batch.rotations, batch.expression, batch.mask, batch.timestamps
        )
        motions = {"oracle": model.plan.decode_raw(z, zm, batch.mask, batch.timestamps)}
        generator = torch.Generator(device=device).manual_seed(i)
        z, zm = model.plan.posterior(
            batch.rotations,
            batch.expression,
            batch.mask,
            batch.timestamps,
            model.posterior_sigma,
            generator,
        )
        motions["posterior"] = model.plan.decode(z, zm, batch.mask, batch.timestamps)
        static_r, static_e = codec.encode(torch.from_numpy(static))
        motions["static"] = (
            static_r[None].expand(1, int(lengths[0]), -1, -1, -1).to(device),
            static_e[None].expand(1, int(lengths[0]), -1).to(device),
        )
        slot_mask = zm.expand(2, -1)
        for seed in SAMPLING_SEEDS:
            # Replay the identical latent noise for both sentences.
            noise_seed = (seed + 1) * 10_000 + i
            plans = [
                sample_plan(
                    model.prior,
                    text[j : j + 1],
                    text_mask[j : j + 1],
                    language[j : j + 1],
                    lengths[j : j + 1],
                    slot_mask[j : j + 1],
                    model.plan_dim,
                    PLAN_NFE,
                    torch.Generator(device=device).manual_seed(noise_seed),
                )
                for j in range(2)
            ]
            r, e = model.plan.decode(torch.cat(plans), slot_mask, masks, times)
            motions[f"correct_{seed}"] = (r[:1], e[:1])
            motions[f"wrong_{seed}"] = (r[1:], e[1:])
        expected = model.features.raw(batch.rotations[0], batch.expression[0])
        geometry["ground_truth"].append(feature_values(expected, expected, times[0]))
        for key, (r, e) in motions.items():
            output[key].append(codec.decode(r[0].cpu(), e[0].cpu()).numpy())
            actual = model.features.raw(r[0], e[0])
            geometry[key].append(feature_values(actual, expected, times[0]))
    return output, geometry, float(np.mean(leng_errors))


@torch.inference_mode()
def score_text(archives, inputs, checkpoint, device):
    model = MBartBT("zh_CN").to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    scores = {}
    for key, poses in archives.items():
        records = [(x["name"], x["text"], p) for x, p in zip(inputs, poses, strict=True)]
        scores[key] = [
            value
            for _, texts, x, mask in batches(records, 4, False)
            for value in seq_logprob(model, x.to(device), mask.to(device), texts)
        ]
        LOG.info("Scored text %s", key)
    return scores


def paired_delta(scores, pairs, ids):
    deltas = np.asarray([np.asarray(scores[correct]) - scores[wrong] for correct, wrong in pairs])
    per_clip = deltas.mean(axis=0)
    return {
        "delta_correct_minus_wrong": float(per_clip.mean()),
        "per_seed_delta": deltas.mean(axis=1).tolist(),
        "ci95_fixed_seed_clip_bootstrap": bootstrap_mean(per_clip, ids, 4000, 0),
    }


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "methods/coplan_so3/config.yaml").read_text())["method"]
    config["architecture"]["fps_by_language"] = [25.0, FPS, 24.0]
    config["device"] = args.device
    config["training"]["A"].update(epochs=args.epochs_a, batch_size=4, seed=args.seed)
    config["training"]["B"].update(epochs=args.epochs_b, batch_size=16, seed=args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    rng = np.random.default_rng(args.seed)
    root = ROOT / "data/processed/madis/CSL-Daily"
    train_rows = select_rows(root, "train", args.train_size, rng)
    dev_rows = select_rows(root, "val", args.dev_size, rng)
    train_ids = [info["name"] for info, _ in train_rows]
    inputs = [info for info, _ in dev_rows]
    ids = [info["name"] for info in inputs]
    if set(train_ids) & set(ids) or len(set(train_ids + ids)) != len(train_ids) + len(ids):
        raise ValueError("Duplicate IDs or overlap between train and dev")
    info = {
        "train_ids": train_ids,
        "dev_ids": ids,
        "seed": args.seed,
        "sampling_seeds": SAMPLING_SEEDS,
        "plan_nfe": PLAN_NFE,
        "config": config,
        "timing": "CSL-Daily native 30 fps, retained SOKE frame indices with duplicates removed",
        "csl_daily_source": "https://openaccess.thecvf.com/content/CVPR2021/papers/"
        "Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf",
        "length_controlled": True,
        "length_source": "dev reference for paired diagnostics",
        "scorer": str(args.scorer.resolve()),
        "scorer_sha256": fingerprint(args.scorer),
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": fingerprint(args.baseline),
    }
    manifest = args.out / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(info)):
        raise ValueError("Cannot resume a pilot with different inputs or settings")
    write_json(manifest, info)
    LOG.info("Preparing %d train and %d dev clips", len(train_rows), len(dev_rows))
    model, codec = build(config)
    train, train_features = load_clips(train_rows, codec)
    val, targets = load_clips(dev_rows, codec)
    static = np.concatenate(train_features).mean(0, keepdims=True)
    del train_features
    swaps = wrong_sentences(inputs, [len(t) for t in targets])
    write_json(args.out / "pairs.json", {key: ids[j] for key, j in zip(ids, swaps, strict=True)})
    archives = {
        "ground_truth": targets,
        "wrong_motion": [targets[j] for j in swaps],
        "signmflow_v2": archive_values(args.baseline, inputs),
    }
    model.to(device)
    checkpoints = args.out / "checkpoints"
    frozen, best_b = checkpoints / "plan_frozen.pt", checkpoints / "stage_B.pt"
    if not frozen.exists():
        stats = DataLoader(train, batch_size=4, collate_fn=collate)
        fit_feature_statistics(model, stats, device)
        LOG.info("Training stage A")
        train_stage(
            model,
            "A",
            train,
            val,
            config["training"]["A"],
            device,
            checkpoints / "stage_A.pt",
            config,
            LOG.info,
        )
        finalize_plan(model, stats, device)
        save_checkpoint(frozen, model, config, stage="B")
    if not (args.out / "stage_B_complete.json").is_file():
        restore_checkpoint(frozen, model)
        LOG.info("Training stage B")
        train_stage(
            model, "B", train, val, config["training"]["B"], device, best_b, config, LOG.info
        )
        write_json(args.out / "stage_B_complete.json", {"checkpoint": str(best_b)})
    else:
        restore_checkpoint(best_b, model)
    LOG.info("Generating oracle and paired coarse predictions")
    output, geometry, length_error = coarse_predictions(
        model, codec, val, inputs, swaps, static, device
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    archives.update(output)
    predictions = args.out / "predictions"
    predictions.mkdir(exist_ok=True)
    for key, poses in archives.items():
        np.savez_compressed(predictions / f"{key}.npz", **dict(zip(ids, poses, strict=True)))
    text_scores = score_text(archives, inputs, args.scorer, device)
    per_clip = {
        key: {
            "bt_logprob": text_scores[key],
            **{k: v.tolist() for k, v in metrics_values(poses, targets).items()},
        }
        for key, poses in archives.items()
    }
    for key, rows in geometry.items():
        per_clip[key].update({metric: [row[metric] for row in rows] for metric in rows[0]})
    control_gap = paired_delta(text_scores, [("ground_truth", "wrong_motion")], ids)
    prior_gap = paired_delta(
        text_scores, [(f"correct_{s}", f"wrong_{s}") for s in SAMPLING_SEEDS], ids
    )
    write_json(args.out / "per_clip.json", {"ids": ids, "scores": per_clip})
    write_json(
        args.out / "report.json",
        {
            "means": {
                key: {m: float(np.mean(v)) for m, v in values.items()}
                for key, values in per_clip.items()
            },
            "gt_control_bt_gap": control_gap,
            "prior_bt_gap": prior_gap,
            "predicted_length_mae_frames": length_error,
        },
    )
    LOG.info("Finished A/B pilot, report at %s", args.out / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=ROOT / "outputs/bt/csl_mbart/best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--dev-size", type=int, default=256)
    parser.add_argument("--epochs-a", type=int, default=20)
    parser.add_argument("--epochs-b", type=int, default=30)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Score the exact shared dev split with a frozen local mBART back-translator."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import fingerprint, write_json
from .diagnostics import archive_values, bootstrap_mean, development


def bleu_stats(metric, hypothesis, reference):
    score = metric.sentence_score(hypothesis, [reference])
    return [score.sys_len, score.ref_len, *score.counts, *score.totals]


def summarize(rows, tokenizer):
    from sacrebleu.metrics import BLEU, CHRF

    hyp = [row["hypothesis"] for row in rows]
    refs = [row["reference"] for row in rows]
    bleu = BLEU(tokenize=tokenizer)
    values = bleu.corpus_score(hyp, [refs])
    return {"bleu4": values.score, "chrf": CHRF().corpus_score(hyp, [refs]).score,
            "bt_logprob": float(np.mean([row["logprob"] for row in rows])),
            "bleu_signature": str(bleu.get_signature()), "n": len(rows)}


@torch.inference_mode()
def score(run, checkpoint, predictions, output, device, batch_size=32, limit=None):
    from scripts.train_backtranslator import LANG, MBartBT, batches, seq_logprob

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    cfg, info, inputs, targets, _ = development(run, limit)
    source = cfg["data"]["source"]
    tokenizer = "zh" if source == "csl" else "13a"
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    model = MBartBT(LANG[source]).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state)
    del state
    archives = {"ground_truth": targets}
    hashes = {}
    for label, path in predictions:
        if label in archives:
            raise ValueError(f"Duplicate scoring label {label!r}")
        path = Path(path)
        poses = archive_values(path, inputs)
        if any(p.ndim != 2 or p.shape[1] != 133 or not len(p) or not np.isfinite(p).all()
               for p in poses):
            raise ValueError(f"Invalid prediction array in {path}")
        archives[label] = poses
        hashes[label] = {"path": str(path.resolve()), "sha256": fingerprint(path)}
    data = {}
    for label, poses in archives.items():
        records = [(item["name"], item["text"], pose) for item, pose in zip(inputs, poses, strict=True)]
        rows = []
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for ids, text, feat, mask in batches(records, batch_size, False):
                feat, mask = feat.to(device), mask.to(device)
                translations = model.generate(feat, mask)
                likelihood = seq_logprob(model, feat, mask, text)
                rows.extend({"id": key, "reference": r, "hypothesis": h, "logprob": lp}
                            for key, r, h, lp in zip(ids, text, translations, likelihood, strict=True))
        data[label] = {"metrics": summarize(rows, tokenizer), "translations": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"split": "val", "dataset": cfg["data"], "run": str(run.resolve()),
                        "training_seed": info["seed"], "tokenizer": tokenizer, "arch": "mbart",
                        "device_type": device.type, "batch_size": batch_size,
                        "scorer_path": str(checkpoint.resolve()), "scorer_sha256": fingerprint(checkpoint),
                        "archives": hashes, "scores": data})


def _bleu(metric, sums):
    sys_len, ref_len, *counts = map(int, sums)
    return metric.compute_bleu(counts[:4], counts[4:], sys_len, ref_len,
                               smooth_method=metric.smooth_method, smooth_value=metric.smooth_value,
                               effective_order=metric.effective_order).score


def paired_corpus_bleu(rows_by_seed, labels, tokenizer, samples, seed):
    """Recompute corpus BLEU from summed n-gram counts in each paired resample."""
    from sacrebleu.metrics import BLEU

    metric = BLEU(tokenize=tokenizer)
    per_sentence = BLEU(tokenize=tokenizer, effective_order=True)
    _, index = np.unique(labels, return_inverse=True)
    blocks = int(index.max()) + 1
    scores = []
    matrices = []
    for base, alternative in rows_by_seed:
        points = []
        per_side = []
        for rows in (base, alternative):
            if len(rows) != len(labels):
                raise ValueError("Each BLEU row needs a cluster ID")
            counts = np.asarray([bleu_stats(per_sentence, r["hypothesis"], r["reference"]) for r in rows])
            cluster = np.zeros((blocks, counts.shape[1]), np.int64)
            np.add.at(cluster, index, counts)
            per_side.append(cluster)
            points.append(_bleu(metric, counts.sum(0)))
        matrices.append(np.stack(per_side))
        scores.append(points)
    matrices = np.stack(matrices)
    random = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        picks = random.integers(blocks, size=blocks)
        resampled = matrices[:, :, picks].sum(2)
        deltas.append(float(np.mean([_bleu(metric, y) - _bleu(metric, x) for x, y in resampled])))
    point = np.asarray(scores)
    return {"per_seed_bleu4_baseline_then_candidate": point.tolist(),
            "delta_bleu4": float(np.mean(point[:, 1] - point[:, 0])),
            "ci_95_delta_bleu4_fixed_seeds": np.quantile(deltas, [.025, .975]).tolist()}


def compare(files, output, samples, seed, clusters, baseline, candidate):
    if samples < 1:
        raise ValueError("Bootstrap draws must be positive")
    results = [json.loads(f.read_text()) for f in files]
    first = results[0]
    outer_keys = ("split", "dataset", "tokenizer", "arch", "scorer_sha256", "device_type", "batch_size")
    if first["split"] != "val" or any(any(r[k] != first[k] for k in outer_keys) for r in results):
        raise ValueError("Comparisons require the same dev data, scorer and scoring settings")
    seeds = [data["training_seed"] for data in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Use one translation result per training seed")
    pairs = [(d["scores"][baseline]["translations"], d["scores"][candidate]["translations"]) for d in results]
    refs = [(r["id"], r["reference"]) for r in pairs[0][0]]
    if not refs or any([(r["id"], r["reference"]) for r in side] != refs for pair in pairs for side in pair):
        raise ValueError("Identical ordered clip IDs and references are required")
    labels = [key for key, _ in refs]
    if clusters:
        groups = json.loads(clusters.read_text())
        labels = [groups[key] for key in labels]
    stats = paired_corpus_bleu(pairs, labels, first["tokenizer"], samples, seed)
    likelihood = np.asarray([[c["logprob"] - b["logprob"] for b, c in zip(x, y)] for x, y in pairs])
    stats["delta_bt_logprob"] = float(likelihood.mean())
    stats["ci_95_delta_logprob_fixed_seeds"] = bootstrap_mean(likelihood.mean(0), labels, samples, seed)
    stats["metadata"] = {"training_seeds": seeds, "n_clips": len(refs), "n_clusters": len(set(labels)),
                         "scorer_sha256": first["scorer_sha256"], "bootstrap_draws": samples,
                         "baseline": baseline, "candidate": candidate}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    s = modes.add_parser("score")
    s.add_argument("run", type=Path)
    s.add_argument("--checkpoint", required=True, type=Path)
    s.add_argument("--prediction", nargs=2, action="append", default=[], metavar=("LABEL", "NPZ"))
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--device", default="cuda")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--limit", type=int)
    c = modes.add_parser("compare")
    c.add_argument("files", type=Path, nargs="+")
    c.add_argument("--output", required=True, type=Path)
    c.add_argument("--bootstrap", type=int, default=10_000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--clusters", type=Path)
    c.add_argument("--baseline", default="baseline")
    c.add_argument("--candidate", default="candidate")
    a = parser.parse_args(argv)
    if a.mode == "score":
        score(a.run, a.checkpoint, a.prediction, a.output, a.device, a.batch_size, a.limit)
    else:
        compare(a.files, a.output, a.bootstrap, a.seed, a.clusters, a.baseline, a.candidate)


if __name__ == "__main__":
    main()

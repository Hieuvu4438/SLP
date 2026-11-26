"""Frozen back-translation and broadcast-bootstrap corpus BLEU on matched dev clips."""

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from methods.articulatory_hierarchical_flow_matching.data import broadcast_id
from methods.articulatory_hierarchical_flow_matching.method import Method
from methods.articulatory_hierarchical_flow_matching.subspace import checksum


def corpus_scores(hypotheses, references, broadcasts, *, samples=5000):
    import sacrebleu

    bleu = sacrebleu.BLEU(tokenize="13a")
    chrf = sacrebleu.CHRF()
    clusters, inverse = np.unique(broadcasts, return_inverse=True)
    if len(clusters) < 2:
        raise ValueError("Need multiple broadcasts for the corpus bootstrap")
    draws = np.random.default_rng(20261126).integers(len(clusters), size=(samples, len(clusters)))
    scores = {}
    trials = {}
    for label, hyps in hypotheses.items():
        if len(hyps) != len(references) or len(hyps) != len(broadcasts):
            raise ValueError(f"Back-translation ID coverage differs for {label}")
        stats = np.array(bleu._extract_corpus_statistics(hyps, [references]), dtype=np.int64)
        grouped = np.zeros((len(clusters), stats.shape[1]), dtype=np.int64)
        np.add.at(grouped, inverse, stats)
        # BLEU is recalculated on the resampled corpus, so no per-sentence BLEU is averaged.
        trials[label] = np.array([bleu._compute_score_from_stats(row.tolist()).score
                                 for block in np.array_split(draws, max(1, samples // 100))
                                 for row in grouped[block].sum(1)])
        scores[label] = {"bleu4": bleu.corpus_score(hyps, [references]).score,
                         "chrf": chrf.corpus_score(hyps, [references]).score}
    contrasts = {}
    for a, b in combinations(scores, 2):
        contrasts[f"{b} vs {a}"] = {"delta": scores[b]["bleu4"] - scores[a]["bleu4"],
                                   "ci95": np.quantile(trials[b] - trials[a], [.025, .975]).tolist()}
    return {"scores": scores, "bleu_deltas": contrasts, "sacrebleu": sacrebleu.__version__,
            "broadcasts": len(clusters), "bootstrap_samples": samples}


def load_scorer(checkpoint, arch, device_name):
    import torch
    from scripts import train_backtranslator as bt

    device = torch.device(device_name if device_name != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    model = (bt.MBartBT("de_DE") if arch == "mbart" else bt.QwenBT()).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model, device


def score_archives(inputs, targets, archives, checkpoint, arch, device_name):
    import torch
    from scripts import train_backtranslator as bt

    from slp.utils.torch_training import precision_context

    keys = Method.prediction_keys(inputs)
    references = [row["text"] for row in inputs]
    metadata = {}
    systems = {"gt": targets}
    for spec in archives:
        label, sep, path = spec.partition("=")
        if not sep or not label or label in systems:
            raise ValueError("Supply unique LABEL=predictions.npz archives")
        with np.load(path, allow_pickle=False) as values:
            systems[label] = [values[key] for key in keys]
        if any(x.ndim != 2 or x.shape[-1] != 133 or not 1 <= len(x) <= 400
               or not np.isfinite(x).all() for x in systems[label]):
            raise ValueError(f"Invalid normalized pose archive for {label}")
        metadata[label] = checksum(Path(path).resolve())
    model, device = load_scorer(checkpoint, arch, device_name)
    hypotheses = {}
    with torch.inference_mode(), precision_context(device, "bf16"):
        for label, poses in systems.items():
            hypotheses[label] = [sentence
                                 for _, _, x, mask in bt.batches(
                                     list(zip(keys, references, poses, strict=True)), 32, False,
                                 )
                                 for sentence in model.generate(x.to(device), mask.to(device))]
    numbers = corpus_scores(hypotheses, references, [broadcast_id(row) for row in inputs])
    return {"arch": arch, "device": device.type, "checkpoint_sha256": checksum(checkpoint),
            "archive_sha256": metadata, "inputs": keys, "references": references,
            "hypotheses": hypotheses, "batch": 32, **numbers}


def judge_blind_bundle(bundle, checkpoint, arch, device_name, margin):
    """Two-alternative judgments from an independent pose-to-text likelihood model."""
    import torch
    from scripts import train_backtranslator as bt

    from slp.utils.torch_training import precision_context

    if margin < 0 or not math.isfinite(margin):
        raise ValueError("The abstention margin must be finite and nonnegative")
    path = bundle / "blind.json"
    manifest = json.loads(path.read_text())
    archive = bundle / "blind.npz"
    if manifest["poses_sha256"] != checksum(archive):
        raise ValueError("The blind pose archive differs from the manifest")
    options = {row["id"]: row["choices"] for row in manifest["samples"]}
    with np.load(archive, allow_pickle=False) as poses:
        if set(poses.files) != set(options) or len(options) != len(manifest["samples"]):
            raise ValueError("The blind bundle needs exactly one pose per opaque ID")
        items = [(key, "", poses[key]) for key in options]
    by_length = defaultdict(list)
    for item in items:
        if (item[2].ndim != 2 or item[2].shape[-1] != 133 or not 1 <= len(item[2]) <= 400
                or not np.isfinite(item[2]).all()):
            raise ValueError(f"Invalid pose shape or values for {item[0]}")
        by_length[len(item[2])].append(item)
    name = f"{arch}_bt_logprob"
    model, device = load_scorer(checkpoint, arch, device_name)
    votes = []
    with torch.inference_mode(), precision_context(device, "bf16"):
        # Qwen reads the first caption token from the final pose prefix position.
        # Equal-length batches avoid conditioning this logit on padded prefix states.
        batches = (batch for group in by_length.values() for batch in bt.batches(group, 32, False))
        for keys, _, x, mask in batches:
            x, mask = x.to(device), mask.to(device)
            left, right = zip(*(options[k] for k in keys), strict=True)
            a = bt.seq_logprob(model, x, mask, list(left))
            b = bt.seq_logprob(model, x, mask, list(right))
            for key, pa, pb in zip(keys, a, b, strict=True):
                if not math.isfinite(pa + pb):
                    raise RuntimeError(f"Non-finite back-translation likelihood for {key}")
                choice = "unknown" if abs(pa - pb) <= margin else ("left" if pa > pb else "right")
                votes.append({"sample": key, "annotator": name, "choice": choice,
                              "choice_gap": pa - pb})
    return {"blind_sha256": checksum(path), "votes": votes, "scorer": {
        "kind": "model", "id": name, "checkpoint_sha256": checksum(checkpoint),
        "margin_nats_per_token": margin, "device": device.type, "equal_length_batches": 32,
    }}

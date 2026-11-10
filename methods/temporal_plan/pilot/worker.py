"""Gated worker: a single process, zero DataLoader children, no subprocesses."""

from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import time

from .queue import atomic_json, check_gate, read_plan


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packed_to_raw(pose):
    import numpy as np

    raw = np.concatenate(
        (
            pose["smplx_body_pose"][:, 33:],
            pose["smplx_lhand_pose"],
            pose["smplx_rhand_pose"],
            pose["smplx_jaw_pose"],
            pose["smplx_expr"],
        ),
        axis=-1,
    ).astype(np.float32)
    if raw.ndim != 2 or raw.shape[1] != 133 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("invalid packed SOKE pose")
    return raw


def prepare(plan):
    import numpy as np
    from huggingface_hub import snapshot_download

    root = Path(plan["output"])
    data = root / "data"
    data.mkdir()
    coverage, sets = {}, {}
    for original, split in (("train", "train"), ("dev", "val")):
        annotation = (
            Path(plan["annotations"]) / f"phoenix14t.pami0.{original}.annotations_only.gzip"
        )
        with gzip.open(annotation, "rb") as f:
            official = pickle.load(f)
        with gzip.open(Path(plan["poses"]) / f"phoenix14t.{original}", "rb") as f:
            curated = {r["name"]: r for r in pickle.load(f)}
        folder = data / split
        folder.mkdir()
        missing, records, hashes = [], [], []
        for row in official:
            name = row["name"]
            if name not in curated:
                missing.append(name)
                continue
            if row["text"].strip() != curated[name]["text"].strip():
                raise ValueError("annotation text mismatch: " + name)
            pose_path = (
                Path(plan["poses"])
                / "poses_video_level"
                / original
                / (name.split("/", 1)[1] + ".pkl")
            )
            with pose_path.open("rb") as f:
                raw = packed_to_raw(pickle.load(f))
            filename = hashlib.sha256(name.encode()).hexdigest() + ".npy"
            destination = folder / filename
            np.save(destination, raw, allow_pickle=False)
            source = re.sub(r"-\d+$", "", name.split("/", 1)[1])
            records.append(
                dict(
                    name=name,
                    text=row["text"],
                    lang="de_DGS",
                    source_id=source,
                    fps=25.0,
                    motion=str(destination.resolve()),
                    split=split,
                    format="raw133",
                )
            )
            hashes.append({"name": name, "sha256": sha256(destination), "frames": len(raw)})
        if set(curated) - {r["name"] for r in official}:
            raise ValueError("curated sample is outside official split")
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        atomic_json(data / f"{split}_hashes.json", hashes)
        coverage[split] = {
            "official": len(official),
            "usable": len(records),
            "excluded": missing,
            "annotation_sha256": sha256(annotation),
        }
        sets[split] = {r["source_id"] for r in records}
    coverage.update(
        shared_recording_groups=len(sets["train"] & sets["val"]),
        official_split_preserved=True,
        test_used=False,
    )
    atomic_json(root / "coverage.json", coverage)
    if coverage["train"]["usable"] != 7092 or coverage["val"]["usable"] != 519:
        raise ValueError("unexpected PHX coverage; inspect before training")
    # Network and model assets are not touched until this explicitly gated job.
    spec = json.loads(Path(__file__).parents[1].joinpath("text_model.json").read_text())
    cached = snapshot_download(
        spec["model_id"],
        revision=spec["revision"],
        max_workers=1,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "generation_config.json",
        ],
    )
    atomic_json(
        root / "assets.json",
        {
            "text": spec,
            "cache": cached,
            "smplx": str(Path(plan["smplx"]).resolve()),
            "test_targets_accessed": False,
        },
    )
    print(json.dumps(coverage), flush=True)


def datasets(plan):
    from ..data import load_manifest, validate_train_val

    root = Path(plan["output"]) / "data"
    train = load_manifest(root / "train.jsonl", split="train")
    val = load_manifest(root / "val.jsonl", split="val")
    validate_train_val(train, val, allow_shared_source_groups=True)
    return train, val


def fk_backend(plan):
    from ..kinematics import SMPLXForward

    return SMPLXForward.from_model_dir(plan["smplx"], native_only=True)


def training_config(plan, **kwargs):
    from ..training import TrainingConfig
    from ..system import STAGES

    return TrainingConfig(
        device="cuda",
        seed=plan["seed"],
        batch_size=plan["microbatch"],
        accumulation=plan["effective_batch"] // plan["microbatch"],
        epochs={s: 100000 for s in STAGES},
        patience=None,
        allow_shared_source_groups=True,
        **kwargs,
    )


def fresh(plan, train):
    import torch
    import yaml
    from ..checkpoint import model_config
    from ..system import TemporalPlanSystem
    from ..text import TextConditioner

    torch.manual_seed(plan["seed"])
    config = yaml.safe_load(Path(plan["method_config"]).read_text())["method"]
    options = {**config["model"], "max_latents": train.max_latents}
    backend = fk_backend(plan)
    text, tokenizer = TextConditioner.from_pretrained(output_width=config["text_width"])
    model = TemporalPlanSystem(model_config(options), text, backend.rotation_offsets)
    return model, tokenizer, backend


def preflight(plan):
    import torch
    from ..kinematics import SMPLXForward
    from ..data import collate_motion

    model, tokenizer, backend = fresh(plan, datasets(plan)[0])
    # Compare real licensed asset FK and gradients against full official LBS on CPU.
    reference = SMPLXForward(backend.body_model)
    raw = (torch.randn(4, 133) * 0.05).requires_grad_()
    fast, official = backend(raw), reference(raw)
    torch.testing.assert_close(fast, official, atol=3e-6, rtol=3e-5)
    g1 = torch.autograd.grad(fast.square().sum(), raw)[0]
    g2 = torch.autograd.grad(official.square().sum(), raw)[0]
    torch.testing.assert_close(g1, g2, atol=1e-5, rtol=1e-4)
    train, _ = datasets(plan)
    model.to("cuda")
    backend.to("cuda")
    model.fit_motion_statistics(
        (
            (b.raw.to("cuda"), b.lengths.to("cuda"))
            for b in [collate_motion([train[i]]) for i in range(min(16, len(train)))]
        )
    )
    longest = sorted(range(len(train)), key=train.frame_count)[-plan["microbatch"] :]
    batch = collate_motion([train[i] for i in longest]).to("cuda")
    model.set_stage("codec")
    loss = model.codec_loss(batch.raw, batch.lengths, fk=backend)["total"]
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("preflight loss")
    atomic_json(
        Path(plan["output"]) / "preflight.json",
        {
            "real_asset_fk_position_and_gradient_parity": True,
            "loss": float(loss.detach()),
            "gpu": torch.cuda.get_device_name(),
            "total_vram": torch.cuda.get_device_properties(0).total_memory,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "longest_train_frames": int(batch.lengths.max()),
            "max_latents": train.max_latents,
            "microbatch": plan["microbatch"],
            "effective_batch": plan["effective_batch"],
            "torch": torch.__version__,
            "fk": backend.provenance(),
        },
    )


def common(plan):
    from ..training import TrainingRunner

    train, val = datasets(plan)
    model, tokenizer, backend = fresh(plan, train)
    root = Path(plan["output"])
    runner = TrainingRunner(
        model,
        tokenizer,
        training_config(plan, stage_max_updates=plan["common_updates"]),
        root / "common",
        fk=backend,
        log_callback=lambda r: print(json.dumps(r), flush=True),
    )
    path = runner.fit(train, val, stop_after="segment")
    atomic_json(
        root / "common_ready.json",
        {
            "checkpoint": str(path),
            "sha256": sha256(path),
            "history": runner.history,
            "not_a_converged_model_claim": True,
        },
    )


def shared(plan, mode=None):
    from ..checkpoint import load_checkpoint

    root = Path(plan["output"])
    ready = json.loads((root / "common_ready.json").read_text())
    if sha256(ready["checkpoint"]) != ready["sha256"]:
        raise ValueError("shared checkpoint changed")
    model, tokenizer, metadata = load_checkpoint(ready["checkpoint"], device="cuda")
    if mode:
        model.config = replace(model.config, teacher_mode=mode)
    metadata["parent_checkpoint"] = ready
    return model, tokenizer, metadata


def profile(plan):
    import torch
    from ..data import collate_motion
    from ..training import TrainingRunner, weighted_backward

    train, _ = datasets(plan)
    root = Path(plan["output"])
    timings = {}
    memory_peaks = {}
    rng = random.Random(0)
    lengths = sorted(range(len(train)), key=train.frame_count)
    # Include length extremes, not just short clips. All discarded profiling updates
    # start from the shared checkpoint and never become a branch initialization.
    indices = [lengths[-1]] + rng.sample(range(len(train)), plan["effective_batch"] - 1)
    for mode in ("map", "sample"):
        model, tokenizer, _ = shared(plan, mode)
        backend = fk_backend(plan).to("cuda")
        runner = TrainingRunner(
            model, tokenizer, training_config(plan), root / "unused", fk=backend
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        timings[mode] = {}
        memory_peaks[mode] = {}
        for stage in ("planner", "generator", "exposure"):
            model.set_stage(stage)
            torch.cuda.reset_peak_memory_stats()
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
            durations = []
            for repeat in range(2):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                began = time.monotonic()
                for first in range(0, len(indices), plan["microbatch"]):
                    batch = collate_motion(
                        [train[i] for i in indices[first : first + plan["microbatch"]]]
                    )
                    loss = runner._loss(batch, generator)["total"]
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("profiling loss")
                    weighted_backward(loss, len(batch.lengths), len(indices))
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - began)
            timings[mode][stage] = max(durations)
            memory_peaks[mode][stage] = {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            model.mark_stage_completed()
            del optimizer
        del parameters, loss, batch, runner, model, tokenizer, backend
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        root / "matched_budget.json",
        {
            "timings_seconds_per_64_examples": timings,
            "measured_memory_by_stage": memory_peaks,
            "stage_updates": plan["stage_updates"],
            "stage_updates_source": "explicit fixed update targets; no time limit",
            "same_seed_same_initial_state_same_update_count": True,
            "minimum_for_interpretation": {"planner": 100, "generator": 200, "exposure": 50},
        },
    )


def branch(plan, mode):
    from ..training import TrainingRunner
    from ..checkpoint import load_checkpoint, read_checkpoint
    from ..progress import TrainingProgress

    root = Path(plan["output"])
    train, val = datasets(plan)
    recovery = plan.get("recovery", {})
    initial = {}
    inherited = []
    if mode == "map" and recovery.get("map_generator_checkpoint"):
        if sha256(recovery["map_generator_checkpoint"]) != recovery["map_generator_sha256"]:
            raise ValueError("legacy generator recovery checksum mismatch")
        model, tokenizer, metadata = load_checkpoint(
            recovery["map_generator_checkpoint"], device="cuda"
        )
        initial = {"generator": recovery["generator_reset_update"]}
        inherited = read_checkpoint(recovery["map_planner_checkpoint"])["metadata"]["history"]
        ready = json.loads((root / "common_ready.json").read_text())
        metadata["parent_checkpoint"] = ready
        metadata["warm_start_source"] = {
            "path": recovery["map_generator_checkpoint"],
            "sha256": recovery["map_generator_sha256"],
            "completed_updates": recovery["generator_reset_update"],
        }
        start = "generator"
    else:
        model, tokenizer, metadata = shared(plan, mode)
        start = "planner"
    restart = plan.get("generator_reset_update")
    config = training_config(
        plan,
        stage_max_updates=plan["stage_updates"],
        stage_initial_updates=initial,
        stage_restart_updates={"generator": [restart]} if restart else {},
        checkpoint_every_updates=plan.get("checkpoint_every_updates", 50),
    )
    runner = TrainingRunner(
        model,
        tokenizer,
        config,
        root / mode,
        fk=fk_backend(plan),
        log_callback=lambda r: print(json.dumps(r), flush=True),
    )
    runner.history = inherited
    path = runner.fit(
        train,
        val,
        start_stage=start,
        parent_metadata=metadata,
        resume=TrainingProgress(root / mode).exists(),
    )
    atomic_json(
        root / f"{mode}_ready.json",
        {
            "checkpoint": str(path),
            "sha256": sha256(path),
            "history": runner.history,
            "parent_sha256": metadata["parent_checkpoint"]["sha256"],
            "generator_optimizer_reset_update": restart,
        },
    )


def bt_model():
    from transformers import MT5ForConditionalGeneration, AutoTokenizer
    from .evaluator import PoseTranslator

    spec = json.loads(Path(__file__).parents[1].joinpath("text_model.json").read_text())
    lm, info = MT5ForConditionalGeneration.from_pretrained(
        **{
            "pretrained_model_name_or_path": spec["model_id"],
            "revision": spec["revision"],
            "local_files_only": True,
            "output_loading_info": True,
        }
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("incomplete pinned evaluator LM")
    tokens = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"], local_files_only=True
    )
    return PoseTranslator(lm).to("cuda"), tokens


def bt_features(model, raw, lengths):
    return model.motion_norm(model.representation.encode(raw, lengths))


def evaluator(plan):
    import torch
    from torch.utils.data import DataLoader
    from ..data import collate_motion, tokenize_texts
    from ..progress import AtomicState, rng_state, restore_rng
    from .evaluator import calibration, checked_optimizer_step

    root = Path(plan["output"])
    torch.manual_seed(plan["seed"])
    train, val = datasets(plan)
    shared_model, _, _ = shared(plan)
    bt, tokenizer = bt_model()
    params = [p for p in bt.parameters() if p.requires_grad]
    options = plan["evaluator_training"]
    optimizer = torch.optim.AdamW(params, lr=options["lr"])
    target = options["target_updates"]
    progress = AtomicState(root / "evaluator_resume")
    steps, epoch, cursor, skipped, best, consecutive = 0, 0, 0, 0, float("inf"), 0

    def log(row):
        with (root / "evaluator_training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def save():
        progress.save(
            {
                "version": 1,
                "state": bt.compact_state(),
                "optimizer": optimizer.state_dict(),
                "steps": steps,
                "epoch": epoch,
                "cursor": cursor,
                "skipped": skipped,
                "best": best,
                "rng": rng_state(),
                "options": options,
                "seed": plan["seed"],
            }
        )

    if progress.exists():
        state = progress.load()
        if state["options"] != options or state["seed"] != plan["seed"]:
            raise ValueError("evaluator recovery configuration mismatch")
        bt.restore_compact(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        steps, epoch, cursor, skipped, best = [
            state[k] for k in ("steps", "epoch", "cursor", "skipped", "best")
        ]
        restore_rng(state["rng"])
        log(
            {
                "event": "resumed_checkpoint",
                "bt_updates": steps,
                "epoch": epoch,
                "data_cursor": cursor,
            }
        )
    else:
        save()
    while steps < target:
        loader = DataLoader(
            train,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_motion,
            generator=torch.Generator().manual_seed(plan["seed"] + epoch),
        )
        for index, batch in enumerate(loader):
            if index < cursor:
                continue
            batch = batch.to("cuda")
            labels, mask = tokenize_texts(tokenizer, batch.texts, max_length=128, device="cuda")
            labels = labels.masked_fill(~mask.bool(), -100)
            with torch.no_grad():
                features = bt_features(shared_model, batch.raw, batch.lengths)
            bt.train()
            optimizer.zero_grad(set_to_none=True)
            loss = bt(features, batch.lengths, labels)
            norm = None
            if bool(torch.isfinite(loss)):
                loss.backward()
                norm = checked_optimizer_step(params, optimizer)
            cursor = index + 1
            if norm is None:
                skipped += 1
                consecutive += 1
                log(
                    {
                        "event": "nonfinite_batch_rejected",
                        "names": list(batch.names),
                        "successful_updates": steps,
                        "epoch": epoch,
                        "skipped": skipped,
                    }
                )
                if consecutive >= 10:
                    save()
                    raise FloatingPointError(
                        "ten consecutive invalid evaluator batches; checkpoint preserved"
                    )
                continue
            consecutive = 0
            steps += 1
            if steps % 100 == 0:
                log(
                    {
                        "bt_updates": steps,
                        "target_updates": target,
                        "epoch": epoch,
                        "loss": float(loss.detach()),
                        "gradient_norm": norm,
                        "skipped": skipped,
                    }
                )
            if steps % options.get("checkpoint_every_updates", 250) == 0:
                save()
            if steps >= target:
                break
        bt.eval()
        total, examples = 0.0, 0
        with torch.no_grad():
            for batch in DataLoader(val, batch_size=2, num_workers=0, collate_fn=collate_motion):
                batch = batch.to("cuda")
                labels, mask = tokenize_texts(tokenizer, batch.texts, max_length=128, device="cuda")
                labels = labels.masked_fill(~mask.bool(), -100)
                value = bt(
                    bt_features(shared_model, batch.raw, batch.lengths), batch.lengths, labels
                )
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError("evaluator validation non-finite")
                total += float(value) * len(batch.lengths)
                examples += len(batch.lengths)
        objective = total / examples
        log({"event": "validation", "bt_updates": steps, "epoch": epoch, "objective": objective})
        if objective < best:
            best = objective
            temporary = root / "evaluator_best.tmp"
            torch.save(
                {
                    "state": bt.compact_state(),
                    "steps": steps,
                    "seed": plan["seed"],
                    "validation": best,
                },
                temporary,
            )
            os.replace(temporary, root / "evaluator_best.pt")
        epoch += 1
        cursor = 0
        save()
    selected = torch.load(root / "evaluator_best.pt", map_location="cpu", weights_only=True)
    bt.restore_compact(selected["state"])
    temporary = root / "evaluator.tmp"
    torch.save(
        {**selected, "total_successful_updates": steps, "skipped_nonfinite_batches": skipped},
        temporary,
    )
    os.replace(temporary, root / "evaluator.pt")
    optimizer.zero_grad(set_to_none=True)
    optimizer = None
    bt.eval()
    bt.requires_grad_(False)
    real, reconstructed, texts, names = [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(val, batch_size=2, num_workers=0, collate_fn=collate_motion):
            batch = batch.to("cuda")
            real += bt.translate(
                bt_features(shared_model, batch.raw, batch.lengths), batch.lengths, tokenizer
            )
            normalized, _ = shared_model._prepare(batch.raw, batch.lengths)
            posterior = shared_model.codec.encode(normalized, batch.lengths)
            rec = shared_model.codec.decode(posterior.mean, posterior.lengths, batch.lengths)
            reconstructed += bt.translate(rec, batch.lengths, tokenizer)
            texts += list(batch.texts)
            names += list(batch.names)
    result = calibration(real, reconstructed, texts)
    result.update(
        updates=steps,
        checkpoint_sha256=sha256(root / "evaluator.pt"),
        train_only=True,
        selection="best_real_dev_cross_entropy",
        frozen_before_generation=True,
        skipped_nonfinite_batches=skipped,
        gradient_norm_accumulation="float64",
    )
    atomic_json(root / "evaluator_calibration.json", result)
    atomic_json(
        root / "evaluator_real_captions.json",
        [
            dict(name=n, reference=t, real=r, codec=c)
            for n, t, r, c in zip(names, texts, real, reconstructed)
        ],
    )


def predict(plan, mode):
    from ..application import predict_records, export_predictions
    from ..checkpoint import load_checkpoint

    root = Path(plan["output"])
    ready = json.loads((root / f"{mode}_ready.json").read_text())
    if sha256(ready["checkpoint"]) != ready["sha256"]:
        raise ValueError("branch checkpoint changed")
    model, tokenizer, metadata = load_checkpoint(
        ready["checkpoint"], device="cuda", require_completed=True
    )
    # Reference-free manifest explicitly contains no lengths/motions.
    records = [{k: r[k] for k in ("name", "text", "lang")} for r in datasets(plan)[1].records]
    (root / f"{mode}_text_inputs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    began = time.time()

    def logged():
        for i, item in enumerate(
            predict_records(model, tokenizer, records, metadata, seed=plan["seed"])
        ):
            if i % 10 == 0:
                print(
                    json.dumps(
                        {"generated": i + 1, "total": len(records), "elapsed": time.time() - began}
                    ),
                    flush=True,
                )
            yield item

    count = export_predictions(logged(), root / f"predictions_{mode}")
    atomic_json(
        root / f"{mode}_prediction_summary.json",
        {
            "count": count,
            "seconds": time.time() - began,
            "sampling_steps": model.config.sampling_steps,
            "no_reference_duration": True,
        },
    )


def dtw_native(a, b):
    import numpy as np

    # Mean native-joint distance, fixed shape, meters -> millimeters.
    distances = np.linalg.norm(a[:, None] - b[None], axis=-1).mean(-1) * 1000
    costs = np.full((len(a) + 1, len(b) + 1), np.inf)
    counts = np.zeros_like(costs, dtype=np.int32)
    costs[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            choices = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
            x, y = min(choices, key=lambda ij: costs[ij])
            costs[i, j] = costs[x, y] + distances[i - 1, j - 1]
            counts[i, j] = counts[x, y] + 1
    return float(costs[-1, -1] / counts[-1, -1])


def evaluate(plan):
    import numpy as np
    import torch
    from .evaluator import corpus_bleu

    root = Path(plan["output"])
    _, val = datasets(plan)
    shared_model, _, _ = shared(plan)
    bt, tokenizer = bt_model()
    checkpoint = torch.load(root / "evaluator.pt", map_location="cpu", weights_only=True)
    bt.restore_compact(checkpoint["state"])
    bt.eval()
    bt.requires_grad_(False)
    backend = fk_backend(plan).to("cuda")
    captions, metrics = {}, {}
    with torch.no_grad():
        for mode in ("map", "sample"):
            folder = root / f"predictions_{mode}"
            rows = [json.loads(s) for s in (folder / "predictions.jsonl").read_text().splitlines()]
            by_name = {r["name"]: r for r in rows}
            if set(by_name) != {r["name"] for r in val.records} or len(rows) != len(val):
                raise ValueError("incomplete/duplicate dev predictions")
            scores, hypotheses, collapse, caps, durations = [], [], [], [], []
            for i in range(len(val)):
                example = val[i]
                row = by_name[example.name]
                raw = torch.from_numpy(np.load(folder / row["motion"], allow_pickle=False)).to(
                    "cuda"
                )[None]
                if not bool(torch.isfinite(raw).all()):
                    raise FloatingPointError("prediction contains NaN")
                lengths = torch.tensor([raw.shape[1]], device="cuda")
                hypotheses += bt.translate(
                    bt_features(shared_model, raw, lengths), lengths, tokenizer
                )
                pred_joints = backend(raw[0]).cpu().numpy()
                target_joints = backend(example.raw133.to("cuda")).cpu().numpy()
                scores.append(dtw_native(pred_joints, target_joints))
                speed = (
                    float(np.linalg.norm(np.diff(pred_joints, axis=0), axis=-1).mean())
                    if len(raw[0]) > 1
                    else 0.0
                )
                collapse.append(speed < 1e-4)
                caps.append(row["stop_reason"] != "eos")
                durations.append(len(raw[0]) / len(example.raw133))
            captions[mode] = hypotheses
            metrics[mode] = {
                "pilot_BT_whitespace_BLEU4": corpus_bleu(
                    hypotheses, [r["text"] for r in val.records]
                ),
                "native41_DTW_mm": float(np.mean(scores)),
                "dtw_per_clip_mm": scores,
                "collapse_fraction": float(np.mean(collapse)),
                "cap_stop_fraction": float(np.mean(caps)),
                "mean_generated_reference_length_ratio_diagnostic_only": float(np.mean(durations)),
                "examples": len(val),
            }
    atomic_json(root / "metrics.json", metrics)
    atomic_json(root / "generated_captions.json", captions)


def report(plan):
    root = Path(plan["output"])
    state = json.loads((root / "queue_state.json").read_text())
    calibration_path = root / "evaluator_calibration.json"
    calibration = (
        json.loads(calibration_path.read_text()) if calibration_path.exists() else {"usable": False}
    )
    metrics = (
        json.loads((root / "metrics.json").read_text())
        if (root / "metrics.json").exists()
        else None
    )
    budget = (
        json.loads((root / "matched_budget.json").read_text())
        if (root / "matched_budget.json").exists()
        else None
    )
    reasons, decision = [], "INCONCLUSIVE"
    if metrics and budget:
        map_m, q = metrics["map"], metrics["sample"]
        updates_ok = all(
            budget["stage_updates"][s] >= n for s, n in budget["minimum_for_interpretation"].items()
        )
        if not updates_ok:
            reasons.append("Too few profiled updates to interpret quality")
        if not calibration["usable"]:
            reasons.append("Independent semantic evaluator failed calibration")
        if updates_ok and calibration["usable"]:
            semantic = q["pilot_BT_whitespace_BLEU4"] - map_m["pilot_BT_whitespace_BLEU4"]
            geometry = q["native41_DTW_mm"] / max(map_m["native41_DTW_mm"], 1e-9)
            healthy = q["collapse_fraction"] <= 0.05 and q["cap_stop_fraction"] <= 0.05
            if not healthy or semantic <= -1 or geometry > 1.10:
                decision = "NO_GO_FOR_SCALING"
                reasons.append("Collapse/cap or material semantic/geometric regression")
            elif semantic >= 0.5 and geometry <= 1.05:
                decision = "GO_TO_MULTI_SEED_CONFIRMATION"
                reasons.append("Pilot margin passed; one seed cannot establish generality")
            else:
                decision = "NO_GO_FOR_SCALING"
                reasons.append(
                    "No prespecified useful improvement in this pilot; not a proof of impossibility"
                )
    else:
        reasons.append("Missing jobs or comparison artifacts")
    result = {
        "decision": decision,
        "reasons": reasons,
        "seed": plan["seed"],
        "dataset": "PHOENIX-2014T",
        "test_used": False,
        "semantic_calibration": calibration,
        "metrics": metrics,
        "queue_at_report": state,
        "single_seed_limit": True,
        "no_published_SOTA_comparison": True,
    }
    atomic_json(root / "report.json", result)
    (root / "REPORT.md").write_text(
        "# PHOENIX temporal-plan pilot\n\n"
        + decision
        + "\n\n"
        + "\n".join("- " + r for r in reasons)
        + "\n\n"
        + "See report.json, metrics.json, coverage.json, matched_budget.json and job logs. "
        "One seed; official train/dev; no test. Native41 DTW and locally calibrated BT are pilot diagnostics, "
        "not published SOKE metrics. Full recording groups overlap across official train/dev.\n"
    )
    print(json.dumps({"decision": decision, "reasons": reasons}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--probe-report")
    args = parser.parse_args()
    plan = read_plan(args.plan)
    check_gate(plan)  # before torch imports, data work, downloads or CUDA initialization
    jobs = {j["id"]: j for j in plan["jobs"]}
    if args.job not in jobs:
        raise ValueError("unknown job")
    if os.environ.get("PILOT_QUEUE_MANAGED") != "1":
        raise RuntimeError("worker must be launched by the serial dispatcher")
    deadline = os.environ.get("PILOT_JOB_DEADLINE_EPOCH")
    if deadline is not None and time.time() >= float(deadline):
        raise RuntimeError("expired job")
    if args.job != "prepare":
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if jobs[args.job].get("gpu") and not args.memory_probe:
            from .memory import configure_budget

            configure_budget()
        torch.manual_seed(plan["seed"])
    if args.memory_probe:
        if not args.probe_report or not jobs[args.job].get("gpu"):
            raise ValueError("memory diagnostics require a GPU job and report path")
        from .memory import probe_job, measured_memory

        budget = int(os.environ["PILOT_GPU_MEMORY_BUDGET_BYTES"])
        try:
            from .memory import configure_budget

            configure_budget()
            result = probe_job(plan, args.job)
            atomic_json(
                args.probe_report,
                {
                    **result,
                    **measured_memory(),
                    "status": "fit",
                    "job": args.job,
                    "budget_bytes": budget,
                },
            )
            print(
                json.dumps({"memory_probe": args.job, "status": "fit", **measured_memory()}),
                flush=True,
            )
            return
        except RuntimeError as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and not (
                "cuda" in str(error).lower() and "out of memory" in str(error).lower()
            ):
                raise
            # Exit releases this diagnostic's CUDA context before the queue waits.
            atomic_json(
                args.probe_report,
                {
                    "status": "oom",
                    "job": args.job,
                    "budget_bytes": budget,
                    "required_lower_bound_bytes": budget + 1,
                    "error": str(error)[:3000],
                    **measured_memory(),
                    "diagnostic_only": True,
                    "experimental_weights_saved": False,
                },
            )
            print(
                json.dumps({"memory_probe": args.job, "status": "oom", "budget_bytes": budget}),
                flush=True,
            )
            raise SystemExit(75)
    actions = {
        "prepare": prepare,
        "preflight": preflight,
        "common": common,
        "evaluator": evaluator,
        "profile": profile,
        "map": lambda p: branch(p, "map"),
        "sample": lambda p: branch(p, "sample"),
        "predict_map": lambda p: predict(p, "map"),
        "predict_sample": lambda p: predict(p, "sample"),
        "evaluate": evaluate,
        "report": report,
    }
    print(
        json.dumps({"job": args.job, "started_epoch": time.time(), "pid": os.getpid()}), flush=True
    )
    try:
        actions[args.job](plan)
    except RuntimeError as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or (
            "cuda" in str(error).lower() and "out of memory" in str(error).lower()
        ):
            print(
                json.dumps(
                    {
                        "job": args.job,
                        "recoverable_cuda_oom": True,
                        "recovery": "last committed optimizer/RNG/data cursor",
                        "error": str(error)[:2000],
                    }
                ),
                flush=True,
            )
            raise SystemExit(76)
        raise


if __name__ == "__main__":
    main()

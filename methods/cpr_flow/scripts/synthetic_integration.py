#!/usr/bin/env python
"""Explicit synthetic=true integration; random text fixtures are NOT mT5 output."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from cprflow.cli.cache import build_cache
from cprflow.cli.fit_stats import fit
from cprflow.cli.train import train
from cprflow.config import DEFAULTS
from cprflow.data.cache import NumericCache, digest, file_hash, save_numeric
from cprflow.data.collate import collate_text
from cprflow.data.dataset import MotionDataset
from cprflow.data.manifest import write_jsonl
from cprflow.data.sampler import BalancedBatchSampler
from cprflow.models.duration import DurationModel
from cprflow.sampling.generator import MotionGenerator
from cprflow.training.artifacts import CachedFeatures, load_weights, update_index
from cprflow.training.engine import Trainer, seed_everything
from cprflow.training.tasks import TaskObjective, make_model, provenance
from cprflow.types import DATASETS, LANGUAGES, PROFILE, GenerationRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smplx-model", required=True)
    parser.add_argument("--sources-lock", default="sources.lock.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cfg = copy.deepcopy(DEFAULTS)
    cfg["run"].update(id="ae_fixture", out_dir=str(root / "runs"), synthetic=True, task="ae")
    cfg["paths"].update(
        manifests=str(root / "manifests"),
        processed=str(root / "processed"),
        stats=str(root / "stats"),
        cache=str(root / "cache"),
        sources_lock=str(Path(args.sources_lock).resolve()),
        smplx_model=str(Path(args.smplx_model).resolve()),
        ae_checkpoint=str(root / "runs/ae_fixture/checkpoints/best_ema.pt"),
        duration_checkpoint=str(root / "runs/duration_fixture/checkpoints/best_ema.pt"),
    )
    cfg["model"].update(hidden_dim=24, blocks=2, heads=3, ffn_dim=48)
    cfg["train"].update(
        max_updates=2,
        effective_batch=2,
        microbatch_per_rank=1,
        accumulate=2,
        warmup_updates=1,
        dev_loss_every=1,
        dev_generation_every=2,
        checkpoint_every=1,
        dev_panel_per_dataset=2,
        dev_prompts_per_dataset=1,
        precision="fp32",
    )
    cfg["data"].update(workers_per_rank=0, pin_memory=False)
    cfg["sample"]["nfe"] = 2
    seed_everything(17, True)
    all_rows = []
    for split in ("train", "dev"):
        records = []
        for dataset, (language, spoken, _) in DATASETS.items():
            for i in range(4 if split == "train" else 2):
                frames = 4 + i % 2
                x = torch.zeros(frames, 296)
                x[:, :246] = torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(41)
                angle = torch.linspace(0, 0.1 + i * 0.01, frames)
                x[:, 60] = angle.cos()
                x[:, 61] = -angle.sin()
                x[:, 63] = angle.sin()
                x[:, 64] = angle.cos()
                x[:, 246:] = torch.linspace(0, 0.02, frames)[:, None]
                uid = f"{dataset}:synthetic_{split}_{i}"
                path = root / "processed" / f"{digest(uid)}.npz"
                save_numeric(path, motion=x.numpy(), timestamps=np.arange(frames) / 30)
                records.append(
                    dict(
                        uid=uid,
                        dataset_id=dataset,
                        split=split,
                        text=f"synthetic fixture {i}",
                        sign_language_id=LANGUAGES[language],
                        spoken_language=spoken,
                        source_clip_id=uid,
                        source_video_id=None,
                        signer_id=None,
                        frames=frames,
                        fps=30.0,
                        source_fps=30.0,
                        profile=PROFILE,
                        processed_path=str(path),
                        processed_sha256=file_hash(path),
                        synthetic=True,
                    )
                )
        write_jsonl(root / f"manifests/{split}.jsonl", records)
        all_rows += records
    fit(cfg, "motion-relation")
    build_cache(cfg, "geometry", ["train", "dev"])
    lock = json.loads(Path(cfg["paths"]["sources_lock"]).read_text())
    deps = dict(
        revision=lock["text"]["revision"],
        tokenizer_revision=lock["text"]["tokenizer_revision"],
        max_tokens=512,
        prefix="literal_square_brackets_v1",
        eval_mode=True,
        synthetic=True,
        fixture="random_features_not_mt5",
    )
    cache = NumericCache(root / "cache/text", deps)
    for row in all_rows:
        uid = digest({"text": row["text"], "language": row["sign_language_id"]})
        rng = np.random.default_rng(int(uid[:8], 16))
        cache.put(uid, hidden=rng.normal(size=(3, 768)).astype(np.float32))
    update_index(root / "cache", "text", {"dependencies": deps, "cache_key": cache.key})
    train(cfg)
    dc = copy.deepcopy(cfg)
    dc["run"].update(id="duration_fixture", task="duration")
    dc["train"]["selection"] = "macro_dev_absolute_log_length_error"
    train(dc)
    tc = copy.deepcopy(cfg)
    tc["run"].update(id="teacher_fixture", task="teacher")
    tc["train"]["selection"] = "macro_dev_retrieval_r1"
    train(tc)
    fit(cfg, "latent")
    build_cache(cfg, "latent-relation", ["train", "dev"])
    fc = copy.deepcopy(cfg)
    fc["run"].update(id="cfm_fixture", task="flow")
    fc["train"].update(max_updates=4, selection="macro_dev_hand_global_dtw_jpe")
    ds = MotionDataset(root / "manifests/train.jsonl", "train", fc["data"]["datasets"], 1024)

    def build():
        seed_everything(17, True)
        objective = TaskObjective(fc)
        trainer = Trainer(make_model(fc), fc, provenance(fc))
        sampler = BalancedBatchSampler(
            ds.rows, 2, 17, fc["data"]["datasets"], fc["data"]["q_dataset"]
        )
        return trainer, sampler, objective

    complete, sampler, objective = build()
    complete.run(ds, sampler, objective)
    expected = {key: value.clone() for key, value in complete.raw.state_dict().items()}
    partial, ps, po = build()
    partial.run(ds, ps, po, until=2)
    resumed, rs, ro = build()
    resumed.resume(partial.directory / "checkpoints/last.pt", rs)
    resumed.run(ds, rs, ro)
    for key, value in expected.items():
        torch.testing.assert_close(value, resumed.raw.state_dict()[key], atol=0, rtol=0)
    resumed.raw.load_state_dict(resumed.ema.shadow)
    duration = load_weights(DurationModel(), cfg["paths"]["duration_checkpoint"])
    text_cache = CachedFeatures(fc, "text")

    def encode(texts, languages):
        return collate_text([text_cache.get(all_rows[0])["hidden"]])

    gen = MotionGenerator(
        resumed.raw,
        ro.ae,
        duration,
        ro.motion_norm,
        ro.latent_norm,
        encode,
        fc,
        {"synthetic": True, "text_features": "synthetic_fixture_not_mt5"},
    )
    request = GenerationRequest("synthetic fixture 0", "CSL", 1234, duration_seconds=0.1)
    result = gen.generate(request, nfe=2)
    repeat = gen.generate(request, nfe=2)
    np.testing.assert_array_equal(result.canonical_motion, repeat.canonical_motion)
    save_numeric(
        root / "sample.npz",
        motion=result.canonical_motion,
        timestamps=np.arange(result.predicted_frames) / 30,
    )
    (root / "sample.json").write_text(json.dumps(result.metadata, indent=2))
    rc = copy.deepcopy(fc)
    rc["run"].update(id="relation_fixture", variant="relational_cfm")
    rc["relation"]["enabled"] = True
    rc["loss"].update(
        geom=0.1,
        cycle=0.05,
        lag=0.01,
        auxiliary_every_updates=1,
        auxiliary_frequency_compensation=1.0,
        auxiliary_ramp_start=0,
        auxiliary_ramp_end=1,
    )
    relation = make_model(rc)
    relation.warm_start(resumed.raw.state_dict())
    relation_obj = TaskObjective(rc)
    seed_everything(8, True)
    vector, terms = relation_obj(relation, [ds[0], ds[1]], 40000, [ds[0], ds[1]])
    vector.mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in relation.parameters())
    relational_trainer = Trainer(relation, rc, provenance(rc))
    relational_sampler = BalancedBatchSampler(
        ds.rows, 2, 17, rc["data"]["datasets"], rc["data"]["q_dataset"]
    )
    relational_trainer.run(ds, relational_sampler, relation_obj, until=1)
    report = {
        "synthetic": True,
        "real_data_training": False,
        "mt5_executed": False,
        "ae_updates": 2,
        "duration_updates": 2,
        "flow_updates": 4,
        "teacher_updates": 2,
        "relational_updates": 1,
        "relational_checkpoint_path": str(relational_trainer.directory / "checkpoints/last.pt"),
        "teacher_negative_pool": 2,
        "resume_exact": True,
        "deterministic_sample": True,
        "sample_frames": result.predicted_frames,
        "sample_sha256": file_hash(root / "sample.npz"),
        "sample_path": str(root / "sample.npz"),
        "checkpoint_path": str(resumed.directory / "checkpoints/last.pt"),
        "relation_aux_terms": {k: float(v.float().mean()) for k, v in terms.items()},
        "elapsed_seconds": time.perf_counter() - start,
        "device": "cpu",
        "hidden": 24,
        "blocks": 2,
    }
    (root / "resolved_fixture_config.yaml").write_text(yaml.safe_dump(fc))
    (root / "integration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

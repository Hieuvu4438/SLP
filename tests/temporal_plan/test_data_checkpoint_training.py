from dataclasses import asdict
import copy
import json

import numpy as np
import pytest
import torch

from methods.temporal_plan.checkpoint import (
    load_checkpoint, read_checkpoint, restore_into, save_checkpoint,
)
from methods.temporal_plan.data import MotionDataset, collate_motion, load_manifest, validate_train_val
from methods.temporal_plan.system import STAGES
from methods.temporal_plan.training import TrainingConfig, TrainingRunner, weighted_backward


def dataset(split, lengths, *, offset=0):
    records = [{"name": f"{split}{i}", "text": "hello rain", "source_id": f"{split}_recording{i}",
                "lang": "en_ASL", "fps": 25.} for i in range(len(lengths))]
    rng = np.random.default_rng(3 + offset)
    motions = [(rng.normal(size=(n, 133))*.05+offset).astype(np.float32) for n in lengths]
    return MotionDataset(records, motions, split=split)


def test_manifest_raw_layout_lengths_split_and_no_hidden_resampling(tmp_path):
    motion = np.arange(7*133, dtype=np.float32).reshape(7, 133)
    np.save(tmp_path / "motion.npy", motion)
    row = {"name": "clip", "text": "hello", "lang": "en_ASL", "source_id": "recording",
           "fps": 25, "motion": "motion.npy", "split": "train", "format": "raw133"}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    data = load_manifest(path, split="train")
    assert data.frame_count(0) == 7 and data.max_latents == 4
    torch.testing.assert_close(data[0].raw133, torch.from_numpy(motion))
    batch = collate_motion([data[0]])
    assert batch.lengths.tolist() == [7]
    assert batch.raw.shape == (1, 7, 133)
    with pytest.raises(ValueError, match="split"):
        load_manifest(path, split="val")
    row["format"] = "normalized133"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="raw133"):
        load_manifest(path, split="train")


def test_explicit_inverse_normalization_and_source_overlap_guards(tmp_path):
    original = dataset("train", [5, 8])
    raw = original.motions[0]
    mean, std = np.ones(133, dtype=np.float32), np.full(133, 2., dtype=np.float32)
    normalized = MotionDataset(original.records[:1], [(raw-mean)/std], split="train",
                                normalized_stats=(mean, std))
    torch.testing.assert_close(normalized[0].raw133, torch.from_numpy(raw))
    val = dataset("val", [4])
    validate_train_val(original, val)
    val.records[0]["source_id"] = original.records[0]["source_id"]
    with pytest.raises(ValueError, match="source_id"):
        validate_train_val(original, val)
    val.records[0]["source_id"] = "another"
    val.records[0]["fps"] = 30
    with pytest.raises(ValueError, match="FPS"):
        validate_train_val(original, val)
    with pytest.raises(ValueError, match="std"):
        MotionDataset(original.records, original.motions, split="train",
                       normalized_stats=(mean, np.zeros(133)))
    np.save(tmp_path / "same.npy", raw)
    (tmp_path / "nested").mkdir()
    train_file = MotionDataset(original.records[:1], [tmp_path / "same.npy"], split="train")
    val.records[0]["fps"] = 25.
    val_file = MotionDataset(val.records, [tmp_path / "nested/../same.npy"], split="val")
    with pytest.raises(ValueError, match="same motion"):
        validate_train_val(train_file, val_file)


def test_short_last_microbatch_gradient_equals_full_batch_mean():
    weight = torch.tensor(1.7, requires_grad=True)
    batches = [torch.tensor([1., 2.]), torch.tensor([10.])]
    for values in batches:
        weighted_backward((weight*values).square().mean(), len(values), 3)
    expected = 2*1.7*(1+4+100)/3
    torch.testing.assert_close(weight.grad, torch.tensor(expected))


def test_complete_runner_checkpoint_fresh_load_and_text_only_inference(tiny_components, tmp_path):
    model, tokenizer = tiny_components
    train, val = dataset("train", [5, 7, 8]), dataset("val", [4, 6])
    cfg = TrainingConfig(batch_size=2, accumulation=2, epochs={s: 1 for s in STAGES}, patience=None)
    runner = TrainingRunner(model, tokenizer, cfg, tmp_path / "run")
    checkpoint = runner.fit(train, val)
    assert [row["stage"] for row in runner.history] == list(STAGES)
    assert [row["updates"] for row in runner.history] == [1]*5
    assert model.completed_stages.all()
    assert model.motion_norm.count.item() == sum([5, 7, 8])
    expected = torch.cat([model.representation.encode(
        torch.from_numpy(raw)[None], torch.tensor([len(raw)])
    )[0] for raw in train.motions]).mean(0)
    torch.testing.assert_close(model.motion_norm.mean, expected, atol=1e-6, rtol=1e-6)
    loaded, loaded_tokenizer, metadata = load_checkpoint(checkpoint)
    assert metadata["max_latents_source"] == "ceil(max training frames/2)"
    assert metadata["fps_by_language"] == {"en_ASL": 25.}
    assert not metadata["exact_resume_supported"]
    tokens = loaded_tokenizer(["hello rain"], return_tensors="pt")
    args = (tokens["input_ids"], tokens["attention_mask"], torch.tensor([0]))
    a = model.eval().generate(*args, generator=torch.Generator().manual_seed(42))
    b = loaded.generate(*args, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(a.raw133, b.raw133)
    assert int(b.frame_lengths[0]) <= 8
    assert torch.isfinite(b.raw133).all()
    fresh = copy.deepcopy(model)
    fresh.completed_stages.zero_()
    with pytest.raises(FileExistsError):
        TrainingRunner(fresh, tokenizer, cfg, tmp_path / "run").fit(train, val)


def test_validation_beyond_training_cap_is_not_cropped_or_used_to_expand_cap(
    tiny_components, tmp_path,
):
    model, tokenizer = tiny_components
    train, val = dataset("train", [5, 8]), dataset("val", [20])
    config = TrainingConfig(batch_size=2, accumulation=1, epochs={s: 1 for s in STAGES})
    runner = TrainingRunner(model, tokenizer, config, tmp_path / "run")
    runner.fit(train, val)
    rows = [json.loads(line) for line in (tmp_path / "run/training.jsonl").read_text().splitlines()]
    assert all(not row["validation"]["available"] for row in rows)
    assert all(row["validation"]["out_of_support"] == 1 for row in rows)
    assert all(row["selection"] == "last_epoch" for row in runner.history)
    assert model.config.max_latents == train.max_latents == 4
    assert model.motion_norm.count.item() == 13
    assert len(val[0].raw133) == 20


def test_checkpoint_requires_complete_state_and_tokenizer(tiny_components, tmp_path):
    model, _ = tiny_components
    path = tmp_path / "partial.pt"
    save_checkpoint(model, path)
    restored, token, _ = load_checkpoint(path, require_tokenizer=False)
    assert token is None
    assert not restored.completed_stages.any()
    with pytest.raises(ValueError, match="tokenizer"):
        load_checkpoint(path)
    payload = read_checkpoint(path)
    payload["state_dict"].pop("motion_norm.std")
    with pytest.raises(ValueError, match="incomplete"):
        restore_into(model, payload)
    payload["model_config"] = {**asdict(model.config), "max_latents": 7}
    with pytest.raises(ValueError, match="config"):
        restore_into(model, payload)

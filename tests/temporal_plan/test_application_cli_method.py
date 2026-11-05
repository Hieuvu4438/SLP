import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from methods.temporal_plan import application
from methods.temporal_plan.application import export_predictions, predict_records
from methods.temporal_plan.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from methods.temporal_plan.data import MotionDataset
from methods.temporal_plan.method import Method
from methods.temporal_plan.system import STAGES, TemporalPlanSystem
from methods.temporal_plan.text import TextConditioner
from methods.temporal_plan.training import TrainingConfig, TrainingRunner
from slp.data.base import Split


def make_data(split, sizes):
    records = [{"name": f"{split}_{i}", "text": "hello rain", "lang": "en_ASL",
                "source_id": f"{split}_source_{i}", "fps": 25.} for i in range(len(sizes))]
    rng = np.random.default_rng(2)
    return MotionDataset(records, [rng.normal(0, .05, (n, 133)).astype(np.float32)
                                   for n in sizes], split=split)


def small_training():
    return TrainingConfig(batch_size=2, accumulation=2, epochs={s: 1 for s in STAGES}, patience=None)


def test_cli_is_lazy_and_sample_fresh_process_does_not_use_reference_fields(
    tiny_components, tmp_path,
):
    model, tokenizer = tiny_components
    checkpoint = TrainingRunner(model, tokenizer, small_training(), tmp_path / "train").fit(
        make_data("train", [5, 8]), make_data("val", [4])
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    repo = Path(__file__).resolve().parents[2]
    lazy = subprocess.run(
        [sys.executable, "-c",
         "import sys; import methods.temporal_plan.cli; assert 'torch' not in sys.modules"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=30,
    )
    assert lazy.returncode == 0, lazy.stderr
    outputs = []
    for index, reference in enumerate([1, 99999]):
        source = tmp_path / f"texts{index}.jsonl"
        source.write_text(json.dumps({
            "name": "../../outside", "text": "hello rain", "lang": "en_ASL",
            "reference_length": reference, "motion": "/does/not/exist.npy", "gloss": str(reference),
        }) + "\n")
        output = tmp_path / f"predictions{index}"
        argv = ["sample", "--checkpoint", str(checkpoint), "--inputs", str(source),
                "--output", str(output), "--seed", "7"]
        script = ("from methods.temporal_plan.cli import main; main(" + repr(argv) + "); "
                  "import torch; assert not torch.cuda.is_initialized()")
        process = subprocess.run([sys.executable, "-c", script], cwd=repo, env=env,
                                  capture_output=True, text=True, timeout=30)
        assert process.returncode == 0, process.stderr
        record = json.loads((output / "predictions.jsonl").read_text())
        raw = np.load(output / record["motion"], allow_pickle=False)
        assert raw.shape == (record["frames"], 133) and np.isfinite(raw).all()
        assert record["frames"] == 2 * sum(record["durations"]) <= 8
        assert "/" not in record["motion"] and record["fps"] == 25
        outputs.append(raw)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_prediction_export_is_transactional_and_refuses_existing_runs(tmp_path):
    prediction = {"name": "example", "lang": "en_ASL", "raw133": np.zeros((2, 133))}
    def failed_predictions():
        yield prediction
        raise RuntimeError("failed second sample")
    output = tmp_path / "predictions"
    with pytest.raises(RuntimeError, match="second"):
        export_predictions(failed_predictions(), output)
    assert not output.exists()
    assert export_predictions([prediction], output) == 1
    with pytest.raises(FileExistsError):
        export_predictions([prediction], output)
    duplicate_output = tmp_path / "duplicates"
    with pytest.raises(ValueError, match="duplicate"):
        export_predictions([prediction, prediction], duplicate_output)
    assert not duplicate_output.exists()


def test_repository_method_bridge_restores_normalization_from_checkpoint(
    tiny_components, tmp_path, monkeypatch,
):
    model, tokenizer = tiny_components
    train, val = make_data("train", [5, 8]), make_data("val", [4])
    mean, std = np.ones(133, dtype=np.float32), np.full(133, 2., dtype=np.float32)
    for name, array in [("mean", mean), ("std", std)]:
        np.save(tmp_path / f"{name}.npy", array)
    config = {"method": {
        "target_format": "normalized133", "target_preprocessing": "synthetic_uncropped_fixture",
        "target_stats": {"mean": str(tmp_path / "mean.npy"), "std": str(tmp_path / "std.npy")},
        "training": {"batch_size": 2, "accumulation": 2,
                     "epochs": {s: 1 for s in STAGES}, "patience": None},
    }}
    context = SimpleNamespace(seed=0, ckpt_dir=tmp_path / "method",
                               log_metrics=lambda *args, **kwargs: None)
    monkeypatch.setattr(application, "create_system", lambda *args, **kwargs: (model, tokenizer, None))
    method = Method(config, context)
    method.fit(Split(train.records, [(x-mean)/std for x in train.motions]),
                Split(val.records, [(x-mean)/std for x in val.motions]))
    records = [{"name": "new", "text": "hello rain", "lang": "en_ASL", "reference_length": 99}]
    actual = method.predict(records)
    raw = list(predict_records(model, tokenizer, records, method.metadata, seed=0))[0]["raw133"]
    np.testing.assert_allclose(actual[0], (raw-mean)/std, atol=1e-6)
    # Current caller config cannot silently change the checkpoint's output representation.
    other = Method({"method": {"target_format": "raw133"}}, context)
    other.load(context.ckpt_dir)
    np.testing.assert_allclose(other.predict(records)[0], actual[0], atol=1e-6)
    bad = Method({"method": {}}, context)
    with pytest.raises(ValueError, match="target_format"):
        bad._dataset(Split(train.records, train.motions), "train")


def test_pretrained_checkpoint_excludes_backbone_and_validates_full_load(
    tiny_components, tmp_path, monkeypatch,
):
    import transformers
    model, tokenizer = tiny_components
    local = tmp_path / "local_backbone"
    model.text.encoder.save_pretrained(local)
    tokenizer.save_pretrained(local)
    real_encoder_loader = transformers.MT5EncoderModel.from_pretrained
    real_tokenizer_loader = transformers.AutoTokenizer.from_pretrained
    def encoder_transport(model_id, **kwargs):
        assert kwargs["revision"] == "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f"
        assert kwargs["local_files_only"] is True and kwargs["output_loading_info"] is True
        return real_encoder_loader(local, local_files_only=True, output_loading_info=True)
    def tokenizer_transport(model_id, **kwargs):
        path = model_id if Path(model_id).exists() else local
        return real_tokenizer_loader(path, local_files_only=True)
    monkeypatch.setattr(transformers.MT5EncoderModel, "from_pretrained", encoder_transport)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_transport)
    text, tokenizer = TextConditioner.from_pretrained(output_width=7)
    system = TemporalPlanSystem(model.config, text, torch.zeros(41, 3))
    path = tmp_path / "model.pt"
    save_checkpoint(system, path, tokenizer=tokenizer)
    payload = read_checkpoint(path)
    assert not any(k.startswith("text.encoder.") for k in payload["state_dict"])
    restored, _, _ = load_checkpoint(path)
    for a, b in zip(restored.text.encoder.parameters(), system.text.encoder.parameters()):
        torch.testing.assert_close(a, b)
    monkeypatch.setattr(transformers.MT5EncoderModel, "from_pretrained", lambda *a, **k: (
        model.text.encoder, {"missing_keys": ["encoder.block.0.some_weight"]}
    ))
    with pytest.raises(ValueError, match="completely"):
        TextConditioner.from_pretrained(output_width=7)
    with pytest.raises(ValueError, match="40-character"):
        TextConditioner.from_pretrained(spec={"model_id": "google/mt5-base", "revision": "main"})


def test_tokenizer_corruption_is_not_silently_used(tiny_components, tmp_path):
    model, tokenizer = tiny_components
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, path, tokenizer=tokenizer)
    payload = read_checkpoint(path)
    token_file = tmp_path / payload["tokenizer_dir"] / "tokenizer_config.json"
    token_file.write_text(token_file.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(path)


def test_factory_fails_before_backbone_load_when_assets_or_pose_mean_are_missing(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(TextConditioner, "from_pretrained", lambda **kwargs: pytest.fail(
        "expensive backbone was touched before validating assets"
    ))
    train = make_data("train", [5, 8])
    with pytest.raises(ValueError, match="SMPL-X"):
        application.create_system(train, {})
    with pytest.raises(ValueError, match="pose-mean"):
        application.create_system(train, {"losses": {"fk": 0}})
    np.save(tmp_path / "bad_offsets.npy", np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="pose mean"):
        application.create_system(train, {"losses": {"fk": 0}},
                                   rotation_offsets=tmp_path / "bad_offsets.npy")


def test_cli_train_routes_manifests_and_forces_cpu_default(tiny_components, tmp_path, monkeypatch):
    import yaml
    from methods.temporal_plan.cli import main

    model, tokenizer = tiny_components
    for role, sizes in [("train", [5, 8]), ("val", [4])]:
        data = make_data(role, sizes)
        rows = []
        for i, (record, motion) in enumerate(zip(data.records, data.motions)):
            filename = f"{role}{i}.npy"
            np.save(tmp_path / filename, motion)
            rows.append({**record, "motion": filename, "split": role, "format": "raw133"})
        (tmp_path / f"{role}.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in rows)
        )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"method": {"training": {
        "device": "cuda:0", "epochs": {s: 1 for s in STAGES}, "batch_size": 2,
        "accumulation": 1, "patience": None,
    }}}))
    def factory(train, options, **kwargs):
        assert train.split == "train" and train.max_latents == model.config.max_latents
        return model, tokenizer, None
    monkeypatch.setattr(application, "create_system", factory)
    main(["train", "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
          "--config", str(config), "--output", str(tmp_path / "cli_run")])
    payload = read_checkpoint(tmp_path / "cli_run/final.pt")
    assert payload["metadata"]["training"]["device"] == "cpu"
    assert model.completed_stages.all() and not torch.cuda.is_initialized()


def test_generic_classification_metrics_fail_before_expensive_training(tmp_path):
    method = Method({"eval": {"metrics": ["accuracy"]}, "method": {}},
                    SimpleNamespace(seed=0, ckpt_dir=tmp_path))
    with pytest.raises(ValueError, match="SMPL-X parameters"):
        method.fit(None, None)

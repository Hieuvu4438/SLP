import pytest
import torch
from methods.race_sign.data import RecordDataset, require_disjoint
from methods.race_sign.training import train_decoder, train_selector, load_checkpoint
from methods.race_sign.preparation import cache_candidates, align_training_target


def test_full_two_stage_training_and_resume(cfg, geometry, manifest_factory, tmp_path):
    train = RecordDataset(manifest_factory("decoder_train"), "decoder_train")
    val = RecordDataset(manifest_factory("decoder_val"), "decoder_val")
    output = tmp_path / "decoder"
    _, _, history = train_decoder(
        cfg, train, val, geometry, output, epochs=1, batch_size=2, window=8, halo=2
    )
    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]["train_loss"]))
    train_decoder(
        cfg,
        train,
        val,
        geometry,
        output,
        epochs=2,
        batch_size=2,
        window=8,
        halo=2,
        resume=output / "last.pt",
    )
    roles = ("selector_train", "selector_val", "selector_calibration")
    cached = [
        cache_candidates(
            output / "best.pt", manifest_factory(role), role, geometry, tmp_path / f"cached_{role}"
        )
        for role in roles
    ]
    datasets = [RecordDataset(path, role) for path, role in zip(cached, roles)]
    _, history = train_selector(
        output / "best.pt", *datasets, geometry, tmp_path / "selector.pt", epochs=2, batch_size=2
    )
    _, _, selector, ckpt = load_checkpoint(tmp_path / "selector.pt")
    assert selector is not None and ckpt["stage"] == "selector"
    assert torch.isfinite(selector.calibration_scale)
    assert len(history) == 2


def test_split_and_alignment_guards(manifest_factory, geometry):
    a = RecordDataset(manifest_factory("decoder_train"), "decoder_train")
    with pytest.raises(ValueError):
        require_disjoint(a, a)
    with pytest.raises(ValueError):
        RecordDataset(a.manifest, "test")
    raw = torch.zeros(8, 133)
    with pytest.raises(ValueError):
        align_training_target(raw, raw, geometry, role="test")
    aligned = align_training_target(raw, raw[:4], geometry, role="selector_train")
    assert aligned.shape == (4, 133)
    torch.testing.assert_close(aligned, raw[:4])


def test_resume_matches_uninterrupted_training(cfg, geometry, manifest_factory, tmp_path):
    from dataclasses import replace

    cfg = replace(cfg, dropout=0.1, text_dropout=0.1)
    train = RecordDataset(manifest_factory("decoder_train"), "decoder_train")
    val = RecordDataset(manifest_factory("decoder_val"), "decoder_val")
    train_decoder(cfg, train, val, geometry, tmp_path / "full", epochs=2, batch_size=2, seed=5)
    train_decoder(cfg, train, val, geometry, tmp_path / "resume", epochs=1, batch_size=2, seed=5)
    train_decoder(
        cfg,
        train,
        val,
        geometry,
        tmp_path / "resume",
        epochs=2,
        batch_size=2,
        seed=5,
        resume=tmp_path / "resume/last.pt",
    )
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    for key in full["decoder"]:
        torch.testing.assert_close(full["decoder"][key], resumed["decoder"][key], atol=0, rtol=0)
    assert full["best_validation"] == resumed["best_validation"]


def test_robustness_initialization_retains_normalizer(cfg, geometry, manifest_factory, tmp_path):
    train = RecordDataset(manifest_factory("decoder_train"), "decoder_train")
    val = RecordDataset(manifest_factory("decoder_val"), "decoder_val")
    train_decoder(cfg, train, val, geometry, tmp_path / "a", epochs=1, batch_size=2)
    robustness = RecordDataset(manifest_factory("robustness_train"), "robustness_train")
    with pytest.raises(ValueError):
        train_decoder(cfg, robustness, val, geometry, tmp_path / "bad", epochs=1)
    _, norm, _ = train_decoder(
        cfg,
        robustness,
        val,
        geometry,
        tmp_path / "b",
        epochs=1,
        initialize_from=tmp_path / "a/best.pt",
        batch_size=2,
    )
    _, original, _, _ = load_checkpoint(tmp_path / "a/best.pt")
    assert torch.equal(norm.mean, original.mean) and torch.equal(norm.std, original.std)


def test_selector_rejects_teacher_cache(manifest_factory):
    manifest = manifest_factory("selector_train")
    dataset = RecordDataset(manifest, "selector_train")
    record_path = manifest.parent / dataset.rows[0]["path"]
    record = torch.load(record_path, weights_only=True)
    record["provenance"]["preparation"] = "teacher"
    torch.save(record, record_path)
    with pytest.raises(ValueError, match="generated"):
        dataset[0]


def test_selector_rejects_wrong_candidate_checkpoint(cfg, geometry, manifest_factory, tmp_path):
    from methods.race_sign.training import save_checkpoint
    from methods.race_sign.model import FlowDecoder
    from methods.race_sign.schema import MotionNormalizer
    from methods.race_sign.geometry import geometry_signature

    path = tmp_path / "decoder.pt"
    save_checkpoint(
        path,
        FlowDecoder(cfg),
        MotionNormalizer(),
        stage="decoder",
        provenance=dict(base_signature={}, geometry_signature=geometry_signature(geometry)),
    )
    datasets = [
        RecordDataset(manifest_factory(role), role)
        for role in ("selector_train", "selector_val", "selector_calibration")
    ]
    with pytest.raises(ValueError, match="Candidate cache"):
        train_selector(path, *datasets, geometry, tmp_path / "bad.pt")

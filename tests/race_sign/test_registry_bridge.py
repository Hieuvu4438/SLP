from types import SimpleNamespace
from slp.data.base import Split
from methods.race_sign.data import collate_inputs
from methods.race_sign.method import Method
import torch


def test_registry_fit_save_load_sentence_policy(
    cfg, geometry, manifest_factory, record_factory, tmp_path, monkeypatch
):
    import methods.race_sign.method as module

    monkeypatch.setattr(module, "SMPLXGeometry", lambda _: geometry)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    context = SimpleNamespace(
        ckpt_dir=ckpt_dir, run_dir=tmp_path, seed=7, log_metrics=lambda metrics, step: None
    )
    selector_paths = {
        key: str(manifest_factory(role))
        for key, role in (
            ("train", "selector_train"),
            ("val", "selector_val"),
            ("calibration", "selector_calibration"),
        )
    }
    config = dict(
        method=dict(
            model=cfg.to_dict(),
            epochs=1,
            selector_epochs=1,
            batch_size=2,
            smplx_path="explicit_fixture",
            policy="sentence",
            selector_manifests=selector_paths,
        )
    )
    method = Method(config, context)
    train = Split([], [], meta={"race_manifest": str(manifest_factory("decoder_train"))})
    val = Split([], [], meta={"race_manifest": str(manifest_factory("decoder_val"))})
    method.fit(train, val)
    inputs = collate_inputs([record_factory(8, 8)])
    before = method.predict(inputs)
    method.save()
    restored = Method(config, context)
    restored.load(ckpt_dir / "model.pt")
    after = restored.predict(inputs)
    torch.testing.assert_close(before.poses, after.poses)
    assert restored.pipeline.selector.whole_sentence

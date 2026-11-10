import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from methods.race_sign.preparation import corrupt_tokens, prepare_manifest
from methods.race_sign.data import RecordDataset


def test_nearest_corruption_excludes_self_and_repeat_uses_original_history():
    ids = torch.tensor([0, 1, 2, 1])
    part = SimpleNamespace(quantizer=SimpleNamespace(codebook=torch.tensor([[0.0], [1.0], [3.0]])))
    changed = corrupt_tokens((ids, ids, ids), (part, part, part), 1.0)
    assert all(torch.equal(x, torch.tensor([1, 0, 1, 0])) for x in changed)
    repeated = corrupt_tokens((ids, ids, ids), (part, part, part), 0.0, repeat_probability=1.0)
    assert all(torch.equal(x, torch.tensor([0, 0, 1, 2])) for x in repeated)
    assert torch.equal(ids, torch.tensor([0, 1, 2, 1]))


class PreparedFixtureAdapter:
    generator_train_groups = None
    generator_train_groups_sha256 = "fixture"

    def prepare(self, text, language, sample_id, *, target=None, mode, **kwargs):
        if mode == "generated":
            assert target is None
        return dict(
            sample_id=sample_id,
            language=1,
            baseline=torch.zeros(8, 133),
            codes=torch.ones(2, 3, 8),
            text=torch.ones(3, 12),
            provenance={"fixture": "synthetic", "preparation": mode},
        )


def test_preparation_part_dropout_and_generator_holdout(tmp_path, geometry):
    np.save(tmp_path / "pose.npy", np.zeros((8, 133), dtype=np.float32))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            dict(
                sample_id="sample",
                source_group="video",
                role="robustness_train",
                official_split="train",
                text="fixture",
                language="csl",
                pose="pose.npy",
            )
        )
        + "\n"
    )
    adapter = PreparedFixtureAdapter()
    manifest = prepare_manifest(
        raw, adapter, tmp_path / "drop", "robustness_train", part_dropout=1.0
    )
    record = RecordDataset(manifest, "robustness_train")[0]
    assert not record["codes"].any()
    assert record["provenance"]["dropped_code_parts"] == [True, True, True]
    with pytest.raises(ValueError, match="held-out"):
        prepare_manifest(
            raw,
            adapter,
            tmp_path / "unknown",
            "robustness_train",
            mode="generated",
            align=True,
            geometry=geometry,
        )
    adapter.generator_train_groups = {"another_video"}
    manifest = prepare_manifest(
        raw,
        adapter,
        tmp_path / "heldout",
        "robustness_train",
        mode="generated",
        align=True,
        geometry=geometry,
    )
    record = RecordDataset(manifest, "robustness_train")[0]
    assert record["provenance"]["generator_holdout_groups_sha256"] == "fixture"


def test_official_split_is_distinct_from_method_role():
    from methods.race_sign.data import validate_official_split

    validate_official_split("decoder_val", "dev")
    with pytest.raises(ValueError, match="test is forbidden"):
        validate_official_split("decoder_train", "test")
    with pytest.raises(ValueError):
        validate_official_split("selector_calibration", "dev")

"""All nine LMDB routing fixtures are synthetic and never represent licensed corpus scans."""

import copy
import io
import json
import pickle

import lmdb
import numpy as np
import pytest

from cprflow.config import DEFAULTS
from cprflow.data.cache import file_hash
from cprflow.data.lmdb_adapter import canonical_to_source
from cprflow.data.manifest import read_manifest
from cprflow.data.preparation import canonicalize, import_sources, verified_timing
from cprflow.types import DATASETS


@pytest.mark.requires_body_model
def test_nine_synthetic_lmdbs_timing_quarantine_and_canonicalization(
    tmp_path, monkeypatch, identity_motion, body_model
):
    config = copy.deepcopy(DEFAULTS)
    config["run"]["synthetic"] = True
    config["paths"].update(
        source_pose=str(tmp_path / "raw"),
        manifests=str(tmp_path / "manifests"),
        processed=str(tmp_path / "processed"),
        source_timing=str(tmp_path / "timing.json"),
        access_record=str(tmp_path / "access.json"),
    )
    config["paths"]["smplx_model"] = str(body_model.asset_path)
    revision = "a" * 40
    monkeypatch.setattr(
        "cprflow.data.preparation.audit_sources", lambda _: {"dataset": {"revision": revision}}
    )
    (tmp_path / "access.json").write_text(
        json.dumps(
            {d: {"authorized": True, "reference": "synthetic_test_fixture"} for d in DATASETS}
        )
    )
    for dataset, (language, _, prefix) in DATASETS.items():
        for split in ("train", "dev", "test"):
            path = tmp_path / "raw" / revision / split / f"{prefix}{split}.lmdb"
            path.parent.mkdir(parents=True, exist_ok=True)
            env = lmdb.open(str(path), map_size=1 << 20)
            clip = f"synthetic_{split}_0"
            fields = canonical_to_source(identity_motion(3).numpy())
            if dataset == "csl_daily" and split == "train":
                fields["face_features"] = np.zeros((3, 10))
            buffer = io.BytesIO()
            np.savez(
                buffer,
                **fields,
                translation=np.array(["Text Case!"]),
                language=np.array([language]),
            )
            with env.begin(write=True) as txn:
                txn.put(
                    b"__meta__",
                    pickle.dumps({"clip_ids": [clip], "num_clips": 1, "synthetic": True}),
                )
                txn.put(clip.encode(), buffer.getvalue())
            env.close()
    report = import_sources(config)
    assert len(report) == 9 and all(r["actual_clips"] == 1 for r in report)
    with pytest.raises(FileNotFoundError, match="verified pose timing"):
        canonicalize(config)
    evidence = tmp_path / "synthetic_timing_evidence.txt"
    evidence.write_text("synthetic=true; known fixture pose sampling 15 fps")
    timing = {
        "revision": revision,
        "datasets": {
            d: {
                "verified": True,
                "feature_fps": 15.0,
                "verification_method": "synthetic fixture construction",
                "evidence": [{"path": str(evidence), "sha256": file_hash(evidence)}],
            }
            for d in DATASETS
        },
    }
    (tmp_path / "timing.json").write_text(json.dumps(timing))
    result = canonicalize(config)
    assert result["quarantined"] == 1 and result["synthetic"] is True
    rows = read_manifest(tmp_path / "manifests/dev.jsonl", "dev")
    assert {r["dataset_id"] for r in rows} == set(DATASETS)
    assert all(
        r["frames"] == 5 and r["text"] == "Text Case!" and r["source_video_id"] is None
        for r in rows
    )
    with np.load(rows[0]["processed_path"], allow_pickle=False) as record:
        assert record["motion"].shape == (5, 296)
        assert record["timestamps"][-1] == pytest.approx(2 / 15)
    timing["revision"] = "b" * 40
    (tmp_path / "timing.json").write_text(json.dumps(timing))
    with pytest.raises(ValueError, match="revision mismatch"):
        verified_timing(config)

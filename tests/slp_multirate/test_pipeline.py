import json
from pathlib import Path

import pytest
import torch

from methods.slp_multirate.cli import (
    _bundle, _verify_stage, build_parser, create_cascade, dispatch,
)
from methods.slp_multirate.config import STAGES, resolve
from methods.slp_multirate.data import CanonicalDataset, collate_clips, read_jsonl
from methods.slp_multirate.geometry import Rig
from methods.slp_multirate.pipeline import load_torch, required_dependencies
from methods.slp_multirate.training import split_hash, train_stage


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("multirate")
    previous_threads = torch.get_num_threads()
    args = ["smoke", "--out", str(root), "--threads", "1"]
    for stage in STAGES:
        args += ["--set", f"train.stages.{stage}.total_updates=4"]
    args += ["--set", "eval.bootstrap_resamples=10"]
    dispatch(build_parser().parse_args(args))
    yield root
    torch.set_num_threads(previous_threads)


def _inputs(root: Path):
    canonical = root / "canonical"
    manifest = canonical / "manifest.jsonl"
    rig = Rig.load(canonical / "rig.json")
    cfg = resolve(path=root / "stages" / "fine" / "config.resolved.yaml")
    return cfg, manifest, rig, root / "cohort.json"


def test_smoke_all_stages_verify_and_text_only_guard(trained):
    cfg, manifest, rig, cohort = _inputs(trained)
    paths = _bundle(trained / "bundle.json")
    cascade = create_cascade(cfg, rig, torch.device("cpu"), paths, split_hash(manifest, cohort),
                             require_fine_dependencies=True)
    ds = CanonicalDataset(manifest, rig, ("dev_gate",), cohort)
    c = ds[0]
    batch = collate_clips([c])
    with pytest.raises(ValueError, match="cannot receive reference"):
        cascade.generate(batch["text"], [c["row"]["clip_id"]], batch=batch)
    with torch.no_grad():
        result = cascade.generate(["toy sign 1 length 13"], ["example"])
        repeat = cascade.generate(["toy sign 1 length 13"], ["example"])
    assert result["nfe"] == {"body": 2, "fine": 2}
    torch.testing.assert_close(result["pose"].local, repeat["pose"].local)
    assert result["pose"].local.shape[-3:] == (44, 3, 3)
    assert result["pose"].joints.isfinite().all()
    for stage in STAGES:
        assert _verify_stage(trained / "stages" / stage)["updates"] == 4
    assert len(read_jsonl(trained / "eval" / "text_only" / "failures.jsonl")) == 0


@pytest.mark.parametrize("stage", STAGES)
def test_interrupted_matches_uninterrupted_and_rejects_wrong_split(trained, tmp_path, stage):
    cfg, manifest, rig, cohort = _inputs(trained)
    key = split_hash(manifest, cohort)
    paths = _bundle(trained / "bundle.json")
    deps = {d: paths[d] for d in required_dependencies(stage, cfg)}
    train = CanonicalDataset(manifest, rig, ("train",), cohort)
    val = CanonicalDataset(manifest, rig, ("dev_tune",), cohort)
    c = create_cascade(cfg, rig, torch.device("cpu"), deps, key)
    train_stage(stage, c, train, val, tmp_path, cohort)
    actual = load_torch(tmp_path / "latest.pt")
    saved = load_torch(trained / "stages" / stage / "latest.pt")
    assert actual["sampler"] == saved["sampler"]
    assert torch.equal(actual["rng"]["torch"], saved["rng"]["torch"])
    assert actual["scheduler"] == saved["scheduler"]
    for name in actual["model"]:
        torch.testing.assert_close(actual["model"][name], saved["model"][name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="split_hash"):
        train_stage(stage, c, train, val, tmp_path, None, resume=True)


def test_wrong_rig_and_cli_infer(trained, tmp_path):
    cfg, manifest, rig, cohort = _inputs(trained)
    rig_path = manifest.parent / "rig.json"
    wrong = tmp_path / "rig.json"
    altered = json.loads(rig_path.read_bytes())
    altered["local_axes_id"] = "incorrect"
    wrong.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="incompatible"):
        create_cascade(cfg, Rig.load(wrong), torch.device("cpu"), _bundle(trained / "bundle.json"))
    args = build_parser().parse_args([
        "infer", "--config", str(trained / "stages" / "fine" / "config.resolved.yaml"),
        "--bundle", str(trained / "bundle.json"), "--rig", str(rig_path),
        "--text", "sign", "--out", str(tmp_path / "infer")])
    assert dispatch(args)["samples"] == 1
    assert read_jsonl(tmp_path / "infer" / "predictions.jsonl")[0]["nfe"] == {"body": 2, "fine": 2}

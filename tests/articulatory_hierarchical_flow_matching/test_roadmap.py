import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.experiments.articulatory_hierarchical_flow_matching.minimal_pairs import (
    export_pairs,
    reviewed_pairs,
    score_pairs,
    write_json,
)
from configs.experiments.articulatory_hierarchical_flow_matching.roadmap import (
    choose_snapshot,
    ci_delta,
    read,
    same_pair_contract,
)
from methods.articulatory_hierarchical_flow_matching.chart import RotationChart
from methods.articulatory_hierarchical_flow_matching.data import MotionDataset, train_monitor_indices
from methods.articulatory_hierarchical_flow_matching.endpoint import jitter_coarse, quaternion_angle
from methods.articulatory_hierarchical_flow_matching.method import Method
from methods.articulatory_hierarchical_flow_matching.planning import (
    frame_condition,
    make_planner,
    positive_allocation,
    teacher_plan,
)
from methods.articulatory_hierarchical_flow_matching.retrieval import neighbors
from methods.articulatory_hierarchical_flow_matching.semantic import (
    PoseTextTeacher,
    SemanticBank,
    build_bank,
    terms,
)
from methods.articulatory_hierarchical_flow_matching.subspace import Representation
from slp.config import compose, dump
from slp.data import Split
from slp.geometry import aa_to_quat
from slp.sign_text import TextEncoder
from slp.tracking.run import RunContext

NAME = "articulatory_hierarchical_flow_matching"


@pytest.fixture(autouse=True)
def stable_threads():
    workers = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(workers)


@pytest.fixture
def fixtures(monkeypatch):
    class FK:
        provenance = {"fixture": True}

        def __init__(self, *args):
            pass

        def __call__(self, poses):
            return poses[..., :96].roll(1, -1)

    class Endpoint:
        def __init__(self, *_):
            self.mean = torch.zeros(133)
            self.std = torch.ones(133)
            self.pose_mean = torch.linspace(-.1, .1, 165)
            self.calls = []

        def to(self, device):
            return self

        def __call__(self, x, y, lengths, samples, **options):
            self.calls.append(options.get("shoulder_relative", False))
            loss = (x[:, 0] - y[:, 0]).square().mean()
            return loss, loss

    monkeypatch.setattr(f"methods.{NAME}.method.DescriptorExtractor", FK)
    monkeypatch.setattr(f"methods.{NAME}.method.SigningEndpointLoss", Endpoint)
    rng = np.random.default_rng(21)

    def split(prefix, n):
        inputs = [{"name": f"{prefix}broadcast{i:02d}-01", "lang": "de_DGS",
                   "text": f"am montag {i + 8} grad im westen {i}"}
                  for i in range(n)]
        targets = [rng.normal(scale=.2, size=(8 + i * 2, 133)).astype(np.float32)
                   for i in range(n)]
        return Split(inputs, targets)

    def config(*opts):
        return compose(NAME, overrides=[
            "method.device=cpu", "method.precision=fp32", "method.workers=0",
            "method.text.backend=bytes", "method.text.byte_dim=8", "method.flow.width=16",
            "method.flow.heads=2", "method.flow.depth=1", "method.flow.updates=4",
            "method.flow.batch_size=4", "method.flow.microbatch=2",
            "method.flow.eval_every=4", "method.flow.snapshot_steps=[4]",
            "method.sampler.steps=[2,2]", "method.sampler.min_frames=1",
            "method.semantic.min_retrieval_gain=-1", "method.semantic.updates=3",
            "method.semantic.eval_every=2", "method.semantic.monitor_fraction=0.40",
            "method.semantic.width=16", "method.semantic.negatives=3",
            "method.semantic.calibration_microbatches=2", "method.planner.width=16",
            "method.planner.updates=2", "method.planner.eval_every=2", "method.planner.max_units=8",
            "method.sampler.max_frames=32", "method.planner.max_frames=16",
            "method.planner.target_frames=12",
            *opts,
        ])

    return SimpleNamespace(config=config, train=split("train", 10), val=split("val", 3))


@pytest.mark.parametrize("overrides, field", [
    (["method.semantic.mode=sentence"], "semantic"),
    (["method.semantic.mode=word"], "semantic"),
    (["method.planner.mode=predicted", "method.planner.duration_source=planner"], "planner"),
    (["method.planner.mode=uniform"], "planner"),
    (["method.planner.mode=shuffled"], "planner"),
    (["method.retrieval.mode=nearest"], "retrieval"),
    (["method.retrieval.mode=random"], "retrieval"),
    (["method.subspace.rotation_chart=tangent", "method.flow.coarse_rotation_weight=0.02",
      "method.flow.coarse_fk_weight=0.02"], None),
])
def test_roadmap_arms_train_sample_and_reload(fixtures, tmp_path, overrides, field):
    cfg = fixtures.config(*overrides)
    ctx = RunContext(tmp_path, 17, logging.getLogger(__name__))
    method = Method(cfg, ctx)
    method.fit(fixtures.train, fixtures.val)
    inputs = fixtures.val.inputs
    full = method.generate(inputs)
    counts = list(map(len, full))
    coarse = method.generate(inputs, lengths=counts, component="coarse")
    if field is not None:
        assert field in method.calibration
    if field == "semantic":
        assert (ctx.ckpt_dir / "semantic_teacher.pt").is_file()
        detail = method.calibration["semantic"]
        assert detail["weight"] > 0
        assert detail["weight"] * detail["semantic_grad_rms"] == pytest.approx(
            .075 * detail["coarse_cfm_grad_rms"]
        )
        assert method.teacher is None and method.bank is None
    if field == "planner":
        assert counts == list(map(len, method.generate(inputs[::-1])))[::-1]
        if cfg["method"]["planner"]["duration_source"] == "planner":
            assert counts == [min(32, 4 * sum(method.plan_cache[item["text"], 0][1]))
                              for item in inputs]
    for j, (a, b) in enumerate(zip(full, coarse, strict=True)):
        proj = method.representation.projector(torch.tensor([len(a)]), len(a))
        assert np.isfinite(a).all() and len(a) == len(b)
        # The chart inverse is canonical modulo quaternion sign.
        tensors = method.representation.encode(torch.from_numpy(np.stack((a, b))))
        if method.representation.chart is None:
            torch.testing.assert_close(proj.coarse(tensors[:1]), tensors[1:], atol=1e-4, rtol=1e-4)
        assert len(coarse[j]) <= 32
    loaded = Method(cfg, ctx)
    loaded.load(ctx.ckpt_dir)
    for a, b in zip(loaded.generate(inputs), full, strict=True):
        np.testing.assert_array_equal(a, b)


def test_reference_jitter_holds_out_whole_train_broadcasts(fixtures, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cfg = fixtures.config("method.refinement.reference_train_only=true")
    dump({**cfg, "eval_split": "val"}, source / "config.yaml")
    reference = Method(cfg, RunContext(source, 17, logging.getLogger(__name__)))
    reference.fit(fixtures.train, fixtures.val)
    cfg = fixtures.config(
        "method.refinement.jitter_scale=0.15", f"method.refinement.reference_run={source}",
        "method.refinement.reference_checkpoint=model.pt", "method.refinement.calibration_clips=3",
    )
    target = tmp_path / "target"
    target.mkdir()
    method = Method(cfg, RunContext(target, 17, logging.getLogger(__name__)))
    method.fit(fixtures.train, fixtures.val)
    info = method.calibration["refinement"]
    data = MotionDataset(fixtures.train, None)
    kept, held = train_monitor_indices(data, .4)
    assert set(info["ids"]) <= {fixtures.train.inputs[i]["name"] for i in held}
    assert reference.calibration["clips"] == len(kept)
    assert info["strength"] > 0
    assert info["strength"] == pytest.approx(.15 * info["rms_median"])
    lengths = torch.tensor([7, 2])
    op = method.representation.projector(lengths, 7)
    noise = torch.randn(2, 7, 133)
    perturbed = jitter_coarse(noise * 0, noise, op, .23)
    torch.testing.assert_close(op.coarse(perturbed), perturbed)
    torch.testing.assert_close((perturbed.square().sum((1, 2)) / op.coarse_rank).sqrt(),
                               torch.full((2,), .23))


def test_lexical_normalization_and_hard_bank(fixtures):
    assert terms("Montag, 12,5 Grad! Nicht im Süd-West.") == [
        "montag", "12,5", "grad", "nicht", "im", "süd-west",
    ]
    data = MotionDataset(fixtures.train, None)
    kept, held = train_monitor_indices(data, .4)
    assert set(held).isdisjoint(kept)
    encoder = TextEncoder("bytes", None, 128, False, 8).eval()
    cfg = fixtures.config()["method"]["semantic"]
    bank, meta = build_bank(data, kept, held, encoder, torch.device("cpu"), cfg)
    used = set(bank.candidates(torch.tensor(kept)).flatten().tolist()) - {-1}
    hold = set(bank.candidates(torch.tensor(held), monitor=True).flatten().tolist()) - {-1}
    assert used.isdisjoint(hold)
    assert meta["monitor_unseen_texts"] == len(held)
    for row in bank.candidates(torch.tensor(kept)):
        ids = row[row >= 0]
        assert len(ids) == len(ids.unique())
    assert torch.isfinite(bank.duration_proxy).all()


def test_semantic_frozen_endpoint_ignores_padding_and_condition_drop(fixtures):
    cfg = fixtures.config()["method"]["semantic"]
    teacher = PoseTextTeacher(8, {**cfg, "mode": "word"}).eval().requires_grad_(False)
    word_ids = torch.tensor([[1, 2], [2, 3], [1, 3]])
    bank = SemanticBank(torch.randn(3, 8), torch.randn(4, 8), word_ids, word_ids.gt(0).float(),
                        torch.arange(3), torch.zeros(3), torch.empty(0), torch.empty(0))
    bank = teacher.freeze_bank(bank)
    x = torch.randn(2, 15, 133, requires_grad=True)
    length = torch.tensor([11, 4])
    key = torch.tensor([[0, 2, 1], [2, 1, -1]])
    args = length, key, bank
    loss, _, _, _ = teacher.objective(x, *args, frozen=True, active=torch.tensor([True, False]))
    loss.backward()
    assert x.grad[0, :11].abs().sum() > 0
    assert not x.grad[0, 11:].any() and not x.grad[1].any()
    assert all(param.grad is None for param in teacher.parameters())
    changed = x.detach().clone()
    changed[0, 11:] = 200
    changed[1, 4:] = 200
    a = teacher.objective(x.detach(), *args, frozen=True)
    b = teacher.objective(changed, *args, frozen=True)
    for av, bv in zip(a, b, strict=True):
        torch.testing.assert_close(av, bv)


def test_planner_boundaries_controls_and_greedy(fixtures):
    cfg = fixtures.config()["method"]["planner"]
    observation = torch.zeros(29, 83)
    observation[12:, :6] = 3
    e, d = teacher_plan(observation, torch.zeros(83), torch.ones(83), cfg)
    assert sum(d) == 8 and 1 < len(d) == len(e)
    assert positive_allocation([8, 1, 3], 2) == [1, 1]
    assert sum(positive_allocation([1, 1, 3], 29)) == 29
    lengths = torch.tensor([15])
    modes = [frame_condition([(e, d)], lengths, 18, key, "cpu")
             for key in ("uniform", "predicted", "shuffled")]
    assert all(x.shape == (1, 18, 249) and not x[:, 15:].any() for x in modes)
    assert not torch.equal(modes[1], modes[2])
    planner = make_planner(8, 32, cfg).eval()
    memory = torch.randn(2, 3, 8)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    a = planner.generate_greedy(memory, valid, max_units=planner.max_latents)
    b = planner.generate(memory, valid, sample_content=False)
    a.plan.validate(planner.max_duration)
    assert a.stop_reasons == b.stop_reasons
    torch.testing.assert_close(a.plan.durations, b.plan.durations)
    torch.testing.assert_close(a.plan.contents, b.plan.contents, atol=1e-5, rtol=1e-5)


def test_quaternion_chart_and_joint_loss_at_pi():
    qa = aa_to_quat(torch.tensor([[.0, .0, .0], [torch.pi - 1e-5, .0, .0]]))
    qb = aa_to_quat(torch.tensor([[.0, .0, .0], [.0, .0, .0]], requires_grad=True))
    angle = quaternion_angle(qa, qb)
    torch.testing.assert_close(angle, quaternion_angle(qa, -qb))
    torch.autograd.grad(angle.square().mean(), qb)
    assert angle.tolist() == pytest.approx([0, torch.pi - 1e-5])
    rep = Representation("temporal", 8, 8, tangent=True).double()
    chart: RotationChart = rep.chart
    chart.rest.copy_(torch.linspace(-.2, .3, 123))
    chart.raw_mean.copy_(torch.linspace(.1, .2, 123))
    chart.reference.copy_(aa_to_quat(torch.randn(41, 3, dtype=torch.float64) * .3))
    pose = torch.randn(2, 19, 133, dtype=torch.float64) * .3
    pose[0, 0, 0] = torch.pi
    op = rep.projector(torch.tensor([19, 19]), 19)
    encoded = rep.encode(pose)
    restored = rep.decode(op.coarse(encoded) + op.detail(encoded))
    err = quaternion_angle(chart.rotations(pose), chart.rotations(restored))
    assert err.max() < 1e-10
    torch.testing.assert_close(restored[..., -10:], pose[..., -10:])


def test_retrieval_excludes_broadcast_matches_and_random_is_length_matched():
    index = {"vectors": torch.eye(3), "groups": torch.tensor([0, 0, 1, 2, 3]),
             "classes": torch.tensor([0, 0, 1, 0, 2]), "languages": torch.zeros(5, dtype=torch.long),
             "durations": torch.tensor([58, 59, 54, 33, 33])}
    a = torch.tensor([[1., 0, 0]])
    lang = torch.tensor([0])
    own = torch.tensor([0])
    assert neighbors(index, a, lang, ["Dienstag"], "nearest", 0, own).tolist() == [3]
    assert neighbors(index, a, lang, ["Dienstag"], "random", 0, own).tolist() == [4]


def test_blind_bundle_matched_lengths_and_eval_gates(tmp_path):
    items = [{"name": "broadcast-01", "text": "am montag sonnig"}]
    path = tmp_path / "manifest.json"
    write_json(path, {"phase": "tune", "pairs": [
        {"id": "m1", "clip": "broadcast-01", "a": "am montag sonnig",
         "b": "am dienstag sonnig", "kind": "weekday", "reviewed": True,
         "reviewer": "fixture", "reference": "a"},
    ]})
    with pytest.raises(ValueError, match="partition"):
        reviewed_pairs(path, items, "held")

    class Fake:
        ctx = SimpleNamespace(seed=0)
        calls = []

        def load_checkpoint(self, path):
            pass

        def generate(self, inputs, *, lengths, component):
            self.calls.append((inputs[0]["name"], tuple(lengths), component))
            return [np.zeros((lengths[0], 133))]

    model = Fake()
    check = tmp_path / "model.pt"
    check.write_bytes(b"fixture")
    export_pairs(model, path, "tune", items, [np.zeros((15, 133))], [check], [12], tmp_path)
    assert all(key == "broadcast-01" and lengths == (12,) for key, lengths, _ in model.calls)
    key = read(tmp_path / "key.json")
    votes = [{"sample": k, "annotator": "reviewer", "choice": row["truth"]}
             for k, row in key["samples"].items()]
    result = score_pairs(key, {"blind_sha256": key["blind_sha256"], "votes": votes,
                               "scorer": {"kind": "human", "id": "DGS panel"}})
    assert next(iter(result["scores"].values()))["full"]["accuracy"] == 1.0
    same_pair_contract(result, result, "tune")
    with pytest.raises(ValueError, match="phase"):
        same_pair_contract(result, result, "held")
    reference = {"hand_speed_ratio": .80, "hand_accel_ratio": .90}
    def row(hand, speed):
        return {"summary": {"hand": hand, "hand_speed_ratio": speed, "hand_accel_ratio": .88}}
    rows = {"8k": row(6.5, .73), "16k": row(7.0, .81), "24k": row(6.8, .81)}
    assert choose_snapshot(rows, reference, {"8k": True, "16k": True, "24k": False})[0] == "16k"
    assert ci_delta([1, 2, 3], [2, 3, 4], ["a", "a", "b"])["ci95"] == [-1, -1]


def test_dev_recipe_locks_snapshot_before_held_and_allows_shared_durations(tmp_path, monkeypatch):
    from configs.experiments.articulatory_hierarchical_flow_matching import roadmap
    from configs.experiments.motionhiflow_slp.select_dev import split_indices

    dev = [{"name": f"broadcast{i:03d}-01", "text": "am montag sonnig", "lang": "de_DGS"}
           for i in range(70)]
    poses = [np.zeros((12, 133), dtype=np.float32) for _ in dev]
    tune_idx, held_idx = split_indices(dev)
    loaded = []

    def load_split(cfg, phase):
        loaded.append(phase)
        assert phase == "val"
        return Split(dev, poses)

    class Fake:
        ctx = SimpleNamespace(seed=0)
        sampler = {"steps": [6, 6], "cfg_scale": 2.0, "lengths_from_npz": None,
                   "batch_size": 8, "min_frames": 1, "max_frames": 60}
        precision = "fp32"
        device = torch.device("cpu")
        use_plan = False
        calibration = {"semantic": {}}
        subspace = {"mode": "articulatory"}
        prediction_keys = staticmethod(Method.prediction_keys)
        _length_archive = Method._length_archive

        def load_checkpoint(self, snapshot):
            self.snapshot = snapshot

        def predict_durations(self, inputs):
            return [12] * len(inputs)

        def generate(self, inputs, *, lengths=None, component="full"):
            return [np.full((12, 133), 2 if "02" in self.snapshot.stem else 1.) for _ in inputs]

    def geometry(pred, ref):
        score = 1 / float(pred[0][0, 0])
        return {**{k: [score] * len(pred) for name in roadmap.GEOMETRY
                   for k in (name, f"{name}_per_step")},
                **{k: [1.] * len(pred) for k in (
                    "hand_speed", "hand_accel", "ref_hand_speed", "ref_hand_accel")},
                "duration_error": [0.] * len(pred), "generated_lengths": [12] * len(pred),
                "target_lengths": [12] * len(pred)}

    model = Fake()
    monkeypatch.setattr(roadmap, "load_sign_smplx_split", load_split)
    config = {"method": {"flow": {"snapshot_steps": [1, 2]}}}
    monkeypatch.setattr(roadmap, "load_run", lambda run: (config, model))
    monkeypatch.setattr(roadmap, "geometry", geometry)
    monkeypatch.setattr(roadmap, "rig_checksums", lambda *_: {"smplx": "fixture"})
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for i in (1, 2):
        (checkpoints / f"model_step{i:06}.pt").write_bytes(str(i).encode())
    path = str(tmp_path)
    durations = tmp_path / "lengths.npz"
    roadmap.main(["lengths", path, str(durations)])
    assert model.predict_durations(dev) == roadmap.pair_lengths(
        model, Method.prediction_keys(dev), durations,
    )
    assert model.sampler["lengths_from_npz"] is None
    roadmap.main(["tune", path])
    directory = tmp_path / "roadmap"
    selected = directory / "selection.json"
    assert read(directory / "tune.json")["ids"] == [dev[i]["name"] for i in tune_idx]
    with pytest.raises(FileNotFoundError):
        roadmap.main(["held", path])
    roadmap.main(["select", path, "--geometry-only"])
    assert read(selected)["file"] == "model_step000002.pt"
    roadmap.main(["held", path])
    assert read(directory / "held.json")["ids"] == [dev[i]["name"] for i in held_idx]
    with pytest.raises(ValueError, match="already"):
        roadmap.main(["held", path])
    assert set(loaded) == {"val"}


def test_bootstrap_recomputes_corpus_bleu():
    pytest.importorskip("sacrebleu")
    from configs.experiments.articulatory_hierarchical_flow_matching.backtranslation import corpus_scores

    refs = ["am montag gibt es sonne", "am dienstag gibt es regen", "am mittwoch gibt es wind"]
    numbers = corpus_scores({"reference": refs, "same": refs, "other": ["x"] * 3}, refs,
                            ["broadcast1", "broadcast2", "broadcast2"], samples=100)
    assert numbers["scores"]["same"]["bleu4"] == pytest.approx(100.)
    assert numbers["bleu_deltas"]["same vs reference"]["ci95"] == [0., 0.]
    assert numbers["bleu_deltas"]["other vs same"]["ci95"][1] < 0

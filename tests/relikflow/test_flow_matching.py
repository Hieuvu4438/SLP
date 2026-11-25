import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from methods.relikflow.bridge import Correction, bridge, correction_loss
from methods.relikflow.data import Request, pack133, phase_grid, resample, unpack133
from methods.relikflow.evaluation import relation_metrics
from methods.relikflow.fk import FixtureFK
from methods.relikflow.models import Generator, TextEncoder
from methods.relikflow.pipeline import Pipeline, fold_assignment
from methods.relikflow.relations import body_frame, describe, extract_relations
from methods.relikflow.rotations import (
    canon,
    multiply,
    normalize,
    physical_angle,
    so3_exp,
    sphere_exp,
    sphere_log_signed,
)
from methods.relikflow.runtime import checkpointed_records, config_load, seed_all
from methods.relikflow.sampling import replan
from methods.relikflow.source import ACTIVE, sample_source, solve_ik
from methods.relikflow.train import Process, run_stage
from slp.evaluation.flow_matching import bleu_counts, corpus_bleu4, geometry, paired_bootstrap
from slp.geometry import quat_to_matrix


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_codec_and_phase():
    seed_all(17)
    pose = torch.randn(10_000, 133)
    q, e = unpack133(pose)
    restored = pack133(q, e)
    assert torch.equal(restored[:, 123:], pose[:, 123:])
    original = quat_to_matrix(so3_exp(pose[:, :123].reshape(-1, 41, 3)))
    mapped = quat_to_matrix(unpack133(restored)[0])
    assert (mapped - original).norm(dim=(-1, -2)).max() < 1e-5
    assert [len(phase_grid(n, 4)) for n in (1, 2, 4, 5, 6, 9, 10)] == [1, 2, 2, 2, 3, 3, 4]
    values = torch.arange(6.0)
    mask = values != 3
    data, valid = resample(values, phase_grid(6, 4), valid=mask)
    assert torch.allclose(data, torch.tensor([0.0, 2.5, 5.0]))
    assert valid.tolist() == [True, False, True]
    assert torch.equal(
        canon(torch.tensor([[0.0, -1.0, 0.0, 0.0]])), torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    )


def test_mbart_token_windows_use_eos_and_source_language():
    class FastTokenizer:
        eos_token_id = 2
        pad_token_id = 1
        lang_code_to_id = {"de_DE": 250003, "zh_CN": 250025, "en_XX": 250004}

        def __call__(self, text, *, add_special_tokens, verbose):
            assert add_special_tokens is False
            return {"input_ids": [int(token) for token in text.split()]}

    encoder = TextEncoder({"backend": "fixture"}, frozen=True)
    encoder.fixture = False
    encoder.tokenizer = FastTokenizer()
    encoder.max_position = 5
    ids, valid, counts = encoder.tokenize(
        [Request("11 12 13 14", "de_DE", "DGS", "de", 17), Request("", "zh_CN", "CSL", "zh", 17)]
    )
    assert counts == [2, 1]
    assert [row[mask].tolist() for row, mask in zip(ids, valid)] == [
        [11, 12, 13, 2, 250003],
        [14, 2, 250003],
        [2, 250025],
    ]


def test_records_continue_after_last_atomic_batch(tmp_path):
    called = []

    def fail(key):
        called.append(key)
        if key == "d":
            raise RuntimeError("interruption")
        return torch.tensor(ord(key))

    keys = list("abcd")
    with pytest.raises(RuntimeError, match="interruption"):
        checkpointed_records(tmp_path, keys, fail, batch_size=2, resume=True, label="IK")
    assert called == keys
    called.clear()

    def compute(key):
        called.append(key)
        return torch.tensor(ord(key))

    results = checkpointed_records(tmp_path, keys, compute, batch_size=2, resume=True, label="IK")
    assert called == ["c", "d"]
    assert [v.item() for v in results.values()] == [ord(k) for k in keys]
    with pytest.raises(ValueError, match="Checkpoint keys"):
        checkpointed_records(tmp_path, list("zbcd"), compute, batch_size=2, resume=True, label="IK")


def test_resume_keeps_completed_artifact_bytes(tmp_path):
    pipe = Pipeline(config_load(None, "smoke"), tmp_path, resume=True)
    pipe.state = {"signature": "fixture"}
    parent = tmp_path / "parent"
    parent.write_text("first")
    path = tmp_path / "predictions.pt"
    poses = torch.zeros(2, 3)
    pipe.artifact(path.name, [poses[0], poses[1]], [parent])
    original = path.read_bytes()

    # Identical values with separate tensor storage must not invalidate descendants.
    pipe.artifact(path.name, [poses[0].clone(), poses[1].clone()], [parent])
    assert path.read_bytes() == original

    parent.write_text("second")
    pipe.artifact(path.name, [torch.ones(3)], [parent])
    assert torch.equal(pipe.read_artifact(path.name)[0], torch.ones(3))
    assert path.read_bytes() != original


def test_signed_log_and_energy_gradients():
    seed_all(5)
    q = torch.tensor([0.1, math.sqrt(0.99), 0, 0], dtype=torch.float64)
    p = q * torch.tensor([1, -1, 1, 1])
    assert sphere_log_signed(q, p).norm() > math.pi / 2
    assert physical_angle(q, p) < 0.5
    for target in (q, -q, p, normalize(-q + q.new_tensor([0, 0, 1e-9, 0]))):
        assert torch.allclose(sphere_exp(q, sphere_log_signed(q, target)), target, atol=1e-8)
    conf = {
        "correction_dims": [21, 8, 8, 8],
        "time_harmonics": 2,
        "kernel_sigma": 0.5,
        "ridge_rho": 1e-6,
    }
    module = Correction(conf, joints=2).double()
    with torch.no_grad():
        module.net[-1].weight.normal_(0, 0.1)
    q0, q1 = [canon(normalize(torch.randn(1, 1, 2, 4, dtype=torch.float64))) for _ in range(2)]
    time = q.new_full((1, 1, 1, 1), 0.37)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    bank = {
        "centers": canon(normalize(torch.randn(2, 8, 4, dtype=torch.float64))),
        "counts": torch.arange(1, 9).repeat(2, 1),
    }

    def path(t):
        return bridge(q0, q1, t, module)

    assert torch.equal(path(time * 0), q0)
    assert torch.equal(path(time * 0 + 1), q1)
    xt, vel = torch.func.jvp(path, (time,), (torch.ones_like(time),))
    assert torch.allclose(vel, (path(time + 1e-5) - path(time - 1e-5)) / 2e-5, atol=1e-6)
    assert (xt * vel).sum(-1).abs().max() < 1e-9

    def loss():
        return correction_loss(q0, q1, time, module, bank, conf, valid)[0]

    value = loss()
    delta = sphere_log_signed(xt[..., None, :], bank["centers"])
    w = bank["counts"] / bank["counts"].sum(-1, keepdim=True)
    w = w.to(q) * torch.exp(-delta.square().sum(-1) / 0.5)
    m = w.sum(-1)
    scatter = torch.einsum("...k,...ki,...kj->...ij", w, delta, delta)
    metric = m[..., None, None] * torch.linalg.inv(scatter + 1e-6 * torch.eye(4, dtype=q.dtype))
    reference = (vel * torch.linalg.solve(metric, vel)).sum()
    assert torch.allclose(reference, value, rtol=1e-6)
    weight = module.net[0].weight
    grad = torch.autograd.grad(value, weight)[0]
    index = np.unravel_index(int(grad.abs().argmax()), grad.shape)
    assert grad.abs().max() > 1e-6
    with torch.no_grad():
        weight[index] += 1e-5
        plus = loss()
        weight[index] -= 2e-5
        minus = loss()
        weight[index] += 1e-5
    torch.testing.assert_close((plus - minus) / 2e-5, grad[index], rtol=1e-3, atol=1e-6)


def test_relations_local_masks_and_ik_jacobian():
    seed_all(17)
    fk = FixtureFK()
    q = so3_exp(torch.randn(3, 41, 3, dtype=torch.float64) * 0.15)
    p = fk(q)
    reference, valid = describe(p)
    assert valid.all()
    matrix = quat_to_matrix(so3_exp(q.new_tensor([0.03, -0.7, 1.2])))
    transformed, valid = describe(3 * (p @ matrix) + q.new_tensor([1, 2, 3]))
    torch.testing.assert_close(transformed, reference, atol=1e-10, rtol=0)
    assert body_frame(p)[2].det().min() > 0.999
    missing = torch.ones_like(q[..., 0], dtype=torch.bool)
    missing[:, 10:13] = False
    _, mask = extract_relations(q, missing, fk)
    assert not mask[:, 10].any()
    assert mask[:, 18:36].all()
    duplicate = p.clone()
    duplicate[1, 3] = duplicate[1, 4]
    _, invalid = describe(duplicate, rest_points=p[0])
    assert not invalid[1].any()
    index = torch.tensor(ACTIVE)

    def local(d):
        return describe(fk(q[:1].index_copy(1, index, multiply(q[:1, index], so3_exp(d)))))[0]

    d = torch.randn(len(ACTIVE), 3, dtype=q.dtype)
    zero = torch.zeros_like(d)
    _, jv = torch.autograd.functional.jvp(local, zero, d)
    torch.testing.assert_close(
        jv, (local(d * 1e-5) - local(-d * 1e-5)) / 2e-5, atol=1e-6, rtol=1e-3
    )
    cfg = config_load(None, "smoke")
    for targets in (reference[:1], reference, reference * 30):
        center, trace, status = solve_ik(
            targets, q[0], cfg["ik"], fk, torch.ones(39), cfg["planner"]["loss_weights"]
        )
        assert torch.isfinite(center).all()
        assert len(status["accepted"]) == cfg["ik"]["sweeps"]
        assert np.max(np.diff(np.asarray(trace)[:, 0])) <= 1e-8


def test_padding_rng_and_replanning():
    seed_all(17)
    cfg = config_load(None, "smoke")
    model = Generator(TextEncoder(cfg["text"], False), cfg["generator"]).eval()
    req = Request("a scrolling test " * 5, "zh_CN", "CSL", "one", 7)
    req2 = Request("短", "zh_CN", "CSL", "two", 7)
    q, e = sample_source(
        "B", None, 2, cfg["source"], req, "csl_daily", device="cpu", dtype=torch.float32
    )
    args = ("B", None, 2, cfg["source"], req, "csl_daily")
    assert torch.equal(q, sample_source(*args, device="cpu", dtype=q.dtype)[0])
    assert not torch.equal(q, sample_source(*args, device="cpu", dtype=q.dtype, draw=(1, 7))[0])
    t = q.new_zeros(1, 1, 1, 1)
    valid = torch.ones(1, 2, dtype=torch.bool)
    pred, expr = model.predict_target(q[None], e[None], t, model.encode([req]), None, valid)
    qs = normalize(torch.randn(2, 5, 41, 4))
    es = torch.randn(2, 5, 10)
    qs[0, :2], es[0, :2] = q, e
    both = torch.ones(2, 5, dtype=torch.bool)
    both[0, 2:] = False
    pq, pe = model.predict_target(
        qs, es, t.expand(2, -1, -1, -1), model.encode([req, req2]), None, both
    )
    torch.testing.assert_close(pq[:1, :2], pred, rtol=0, atol=1e-5)
    torch.testing.assert_close(pe[:1, :2], expr, rtol=0, atol=1e-5)

    class Endpoint:
        calls = 0

        def predict_target(self, *args):
            self.calls += 1
            return pred.detach(), expr.detach()

    class CountCorrection:
        calls = 0

        def __call__(self, a, b, t):
            self.calls += 1
            return torch.zeros_like(a)

    for n in (1, 3, 5, 10):
        endpoint, correction = Endpoint(), CountCorrection()
        a, b, _ = replan(endpoint, q[None], e[None], None, None, n, correction)
        assert torch.equal(a, pred) and torch.equal(b, expr)
        assert endpoint.calls == n and correction.calls == n - 1


def test_geometry_and_clustered_score():
    q = so3_exp(torch.randn(4, 41, 3, dtype=torch.float64) * 0.1)
    fk = FixtureFK()
    e = torch.zeros(4, 10, dtype=q.dtype)
    decoded = fk.score_joints(q, e)
    clean, path = geometry(decoded, decoded)
    assert max(clean[k] for k in ("body_dtw_jpe", "hand_dtw_jpe", "hand_dtw_pa_jpe")) < 1e-8
    moved = (decoded[0], decoded[1] + 1, decoded[2])
    result, _ = geometry(moved, decoded)
    assert result["hand_dtw_jpe"] < 1e-8
    points = fk(q)
    changed = points.clone()
    changed[:, 5:26] += 1
    x, _ = describe(changed)
    y, _ = describe(points)
    assert relation_metrics(x, y, body_frame(points)[1], path)["wrist_torso_mm"] > 100
    tokens = ["one", "two", "three", "four"]
    row = {
        "bt_counts": bleu_counts(tokens, tokens),
        "clip_id": "one",
        "seed": 17,
        "wrist_torso_error": 0.0,
        "duration_mae_seconds": 0.0,
        **clean,
    }
    assert corpus_bleu4([row]) == 100
    assert all(x["ci"] == [0.0, 0.0] for x in paired_bootstrap([row], [row], 10, 2026).values())


def test_masked_accumulation_and_step_resume(tmp_path):
    seed_all(7)
    clips = [
        SimpleNamespace(
            id=f"c{i}", q=torch.randn(i + 1, 2, dtype=torch.float64), row={"corpus": "csl_daily"}
        )
        for i in range(5)
    ]
    model = torch.nn.Linear(2, 1).double()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    config = dict(epochs=2, batch=2, lr=0.01, decay=0.0)
    args = {
        "signature": "test",
        "process": Process(torch.device("cpu")),
        "seed": 13,
        "shuffle": False,
    }

    def count(items):
        return {"loss": sum(len(c.q) for c in items)}

    def train(path, micro, interrupt=False, resume=False):
        local = torch.nn.Linear(2, 1).double()
        local.load_state_dict(initial)

        def forward(items, epoch):
            if interrupt and items[0][1] >= 2:
                raise ValueError("interrupt")
            return {"loss": (local(torch.cat([c.q for c, _ in items])) - 1).square().sum()}, []

        return run_stage(
            local,
            clips,
            config,
            {"loss": 1.0},
            count,
            forward,
            None,
            path,
            microbatch=micro,
            resume=resume,
            **args,
        ).state_dict()

    combined = train(tmp_path / "combined", 2)
    micro = train(tmp_path / "micro", 1)
    with pytest.raises(ValueError, match="interrupt"):
        train(tmp_path / "resume", 1, interrupt=True)
    replay = train(tmp_path / "resume", 1, resume=True)
    for k, expected in combined.items():
        torch.testing.assert_close(micro[k], expected, rtol=0, atol=1e-12)
        assert torch.equal(micro[k], replay[k])


def test_video_group_folds():
    clips = [
        SimpleNamespace(id=str(i), row={"corpus": "csl_daily", "video_id": str(i // 2)})
        for i in range(12)
    ]
    assignment = fold_assignment(clips, 3, 17)
    assert set(assignment.values()) == {0, 1, 2}
    assert all(assignment[str(i)] == assignment[str(i + 1)] for i in range(0, 12, 2))


def test_smplx_parameter_order_and_landmark_fk(tmp_path):
    pytest.importorskip("smplx")
    from methods.relikflow.fk import SokeFK
    from slp.evaluation.smplx_fk import SMPLXJoints

    # A synthetic asset exercises the real library and hand mean without requiring
    # licensed SMPL-X data or relying on the FixtureFK skeleton.
    rng = np.random.default_rng(42)
    vertices, joints = 10475, 55
    weights = rng.random((vertices, joints)).astype("f")
    weights /= weights.sum(-1, keepdims=True)
    regressor = np.zeros((joints, vertices), dtype="f")
    regressor[np.arange(joints), np.arange(joints)] = 1
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez_compressed(
        path,
        v_template=rng.normal(0, 0.12, (vertices, 3)).astype("f"),
        shapedirs=np.zeros((vertices, 3, 20), dtype="f"),
        f=np.tile([0, 1, 2], (10, 1)),
        J_regressor=regressor,
        posedirs=np.zeros((vertices, 3, (joints - 1) * 9), dtype="f"),
        kintree_table=np.stack((np.maximum(0, np.arange(joints) // 2 - 1), np.arange(joints))),
        weights=weights,
        hands_componentsl=np.eye(45),
        hands_componentsr=np.eye(45),
        hands_meanl=rng.normal(0, 0.09, 45),
        hands_meanr=rng.normal(0, 0.09, 45),
        lmk_faces_idx=np.zeros(51, dtype="int64"),
        lmk_bary_coords=np.ones((51, 3), dtype="f") / 3,
        dynamic_lmk_faces_idx=np.zeros((79, 17), dtype="int64"),
        dynamic_lmk_bary_coords=np.ones((79, 17, 3), dtype="f") / 3,
    )
    adapter = SokeFK(path)
    scorer = SMPLXJoints("cpu", path, load_stats=False)
    pose = torch.randn(4, 133) * 0.12
    q, e = unpack133(pose)
    for actual, expected in zip(adapter.score_joints(q, e), scorer(pose, normalised=False)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    q.requires_grad_()
    full, _ = adapter.decode(q, torch.zeros_like(e))
    landmarks = adapter(q)
    torch.testing.assert_close(landmarks, full[:, adapter.indices], rtol=0, atol=1e-5)
    grad = torch.autograd.grad(landmarks[..., 6:26, :].sum(), q)[0]
    assert grad.isfinite().all()
    assert grad[:, 10:25].abs().max() > 0

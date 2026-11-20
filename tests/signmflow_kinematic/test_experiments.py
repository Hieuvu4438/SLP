
import numpy as np
import pytest
import torch

from methods.signmflow_kinematic import geometry as G
from methods.signmflow_kinematic.data import collate
from methods.signmflow_kinematic.diagnostics import bootstrap_mean, swap_indices
from methods.signmflow_kinematic.energy import DensityEnergy, KinematicEnergy, group_weights
from methods.signmflow_kinematic.flow import (
    FlowSpec,
    ar1_noise,
    auxiliary_losses,
    motion_residual,
    per_clip_sources,
    sample,
    sampling_grid,
)
from methods.signmflow_kinematic.kinematics import (
    SigningKinematics,
    TIP_VERTICES,
    axis_jacobian,
    tangent_pushforward,
)
from methods.signmflow_kinematic.naturalness import NaturalnessMeasure, kmeans_joint
from methods.signmflow_kinematic.paths import CorrectionNet, path_and_velocity, path_point
from methods.signmflow_kinematic.representation import MotionCodec, load_hand_mean, neutral_asset
from slp.data.sign_smplx import load_stats
from slp.evaluation.smplx_fk import SMPLXJoints
from slp.paths import PROJECT_ROOT

SMPLX = PROJECT_ROOT / "third_party/SOKE/deps/smpl_models"
STATS = PROJECT_ROOT / "third_party/SOKE/deps/stats"


@pytest.fixture(scope="module")
def asset():
    pytest.importorskip("smplx")
    if not neutral_asset(SMPLX).exists() or not (STATS / "mean.pt").exists():
        pytest.skip("Local licensed SMPL-X asset and normalization are needed")
    return SigningKinematics.from_smplx(SMPLX)


def test_shortest_path_uses_quotient_and_ignores_lifts_at_pi():
    angle = torch.tensor([0., 178.], dtype=torch.float64).deg2rad()
    aa = torch.stack((angle, angle * 0, angle * 0), -1)
    q0, q1 = G.aa_to_quat(aa[1]), G.aa_to_quat(-aa[1])
    assert G.relative_rotvec(q0, q1).norm().rad2deg().item() == pytest.approx(4.)
    near_cut = torch.tensor([0., -1., 0., 0.], dtype=torch.float64)
    eye = G.aa_to_quat(aa[0])
    for sign in (-1, 1):
        x = path_point("so3", sign * eye, near_cut, torch.tensor(0.5, dtype=eye.dtype))
        y = path_point("so3", eye, -near_cut, torch.tensor(0.5, dtype=eye.dtype))
        torch.testing.assert_close(G.quat_to_matrix(x), G.quat_to_matrix(y))
    torch.testing.assert_close(G.expmap(eye, G.logmap(eye, -eye)), -eye)


def test_projected_6d_fallback_is_a_rotation():
    source = torch.tensor([1., 0., 0., 0.])
    target = torch.tensor([0., 0., 0., 1.])
    rotations = [G.quat_to_matrix(path_point("projected_6d", source, target, torch.tensor(t)))
                 for t in (0., 0.5, 1.)]
    for r in rotations:
        torch.testing.assert_close(r.T @ r, torch.eye(3))
        assert torch.det(r) == 1
    torch.testing.assert_close(rotations[0], G.quat_to_matrix(source))
    torch.testing.assert_close(rotations[2], G.quat_to_matrix(target))


def test_so3_correction_endpoints_velocity_and_lift_equivariance():
    net = CorrectionNet(n_joints=2, hidden=16, layers=2, quotient=True).double()
    torch.nn.init.normal_(net.net[-1].weight, std=0.1)
    start = G.normalize(torch.randn(4, 2, 4, dtype=torch.float64))
    end = G.normalize(torch.randn_like(start))
    t = torch.rand(4, 1, 1, dtype=torch.float64)
    q, v = path_and_velocity("so3_corrected", start, end, t, net)
    q_flip = path_point("so3_corrected", -start, -end, t, net)
    torch.testing.assert_close(q, -q_flip)
    torch.testing.assert_close((q * v).sum(-1), torch.zeros_like(q[..., 0]))
    eps = 1e-5
    plus = path_point("so3_corrected", start, end, t + eps, net)
    minus = path_point("so3_corrected", start, end, t - eps, net)
    torch.testing.assert_close(v, (plus - minus) / (2 * eps), atol=1e-8, rtol=1e-8)
    for value, endpoint in ((0, start), (1, end)):
        result = path_point("so3_corrected", start, end, torch.full_like(t, value), net)
        torch.testing.assert_close(G.quat_to_matrix(result), G.quat_to_matrix(endpoint))


def test_chordal_matches_matrix_distance_and_invariant_input():
    q = G.normalize(torch.randn(1, 3, 5, 4, dtype=torch.float64))
    target = G.normalize(torch.randn_like(q))
    spec = FlowSpec(endpoint_loss="chordal", input_rotation="6d")
    physical = (G.quat_to_matrix(q) - G.quat_to_matrix(target)).square().sum((-1, -2, -3)) / 2
    torch.testing.assert_close(physical, motion_residual(q, target, spec))
    torch.testing.assert_close(motion_residual(q, target, spec), motion_residual(-q, target, spec))
    torch.testing.assert_close(spec.model_input(q), spec.model_input(-q))


def test_fk_fingertips_match_full_mesh_at_neutral_expression(asset):
    codec = MotionCodec(*load_stats(str(STATS)), load_hand_mean(SMPLX))
    quaternion = G.aa_to_quat(torch.randn(5, 41, 3, generator=torch.Generator().manual_seed(3)) * 0.15)
    neutral = (-codec.mean[123:] / codec.std[123:]).expand(5, -1)
    full_joints, left, right = SMPLXJoints("cpu", SMPLX, STATS)(codec.decode(quaternion, neutral).numpy())
    skeleton, _, tips = asset.points(G.quat_to_matrix(quaternion))
    torch.testing.assert_close(full_joints[:, :55], skeleton, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(left[:, [4, 8, 12, 16, 20]], tips[:, :len(TIP_VERTICES)//2], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(right[:, [4, 8, 12, 16, 20]], tips[:, len(TIP_VERTICES)//2:], atol=2e-6, rtol=2e-5)


def test_pullback_diagonal_matches_basis_and_full_derivative(asset):
    asset = asset.double()
    q = G.aa_to_quat(torch.zeros(2, 41, 3, dtype=torch.float64))
    r = G.quat_to_matrix(q).requires_grad_()
    v = torch.randn_like(q[..., 1:]).requires_grad_()
    full = tangent_pushforward(asset, r, v)
    jac = axis_jacobian(asset, r)
    torch.testing.assert_close(full, (jac * v[:, None]).sum((-1, -2)))
    delta = 1e-5 * v
    finite_diff = (asset(r @ G.aa_to_matrix(delta)) - asset(r @ G.aa_to_matrix(-delta))) / 2e-5
    torch.testing.assert_close(full, finite_diff, atol=1e-6, rtol=1e-6)
    grads = torch.autograd.grad(full.square().sum(), (r, v))
    assert all(torch.isfinite(g).all() for g in grads)

    weights = {"body": 1., "wrist": 1., "fingers": 1., "palm": 1.}
    metrics = [KinematicEnergy(asset, name, weights).double() for name in
               ("fk_full", "fk_diagonal", "fk_joint_block")]
    basis = torch.zeros_like(v)
    basis[:, 8, 1] = 1
    dq = 0.5 * G.quat_multiply(q, torch.cat((torch.zeros_like(q[..., :1]), basis), -1))
    for energy in metrics[1:]:
        torch.testing.assert_close(energy(q, dq), metrics[0](q, dq))
    vector = basis.clone()
    vector[:, 6, 1] = 1
    dq = 0.5 * G.quat_multiply(q, torch.cat((torch.zeros_like(q[..., :1]), vector), -1))
    assert not torch.allclose(metrics[0](q, dq), metrics[1](q, dq))
    asset.float()


@pytest.mark.parametrize("kind", ["fk_full", "fk_diagonal", "fk_joint_block", "density", "legacy_ng"])
def test_correction_energy_backprop_and_update_is_finite(asset, kind):
    torch.manual_seed(20)
    is_quotient = kind != "legacy_ng"
    correction = CorrectionNet(hidden=8, layers=2, quotient=is_quotient)
    opt = torch.optim.AdamW(correction.parameters())
    a = G.uniform_hemisphere((2, 41))
    b = a.clone()
    b[0] = G.uniform_hemisphere((1, 41))[0]
    t = torch.rand(2, 1, 1)
    learned_path = "so3_corrected" if is_quotient else "naturalness"
    if kind.startswith("fk_"):
        energy = KinematicEnergy(asset, kind, {"body": 1., "wrist": 1., "fingers": 1.})
    else:
        measure = NaturalnessMeasure(a.transpose(0, 1), torch.ones(41, 2) / 2, quotient=is_quotient)
        energy = DensityEnergy(measure, torch.zeros(41)) if is_quotient else measure.quad_form
    for _ in range(2):
        q, dq = path_and_velocity(learned_path, a, b, t, correction)
        value = energy(q, dq).mean()
        assert value.isfinite()
        opt.zero_grad(set_to_none=True)
        value.backward()
        assert all(p.grad.isfinite().all() for p in correction.parameters())
        opt.step()


def test_kmeans_accepts_repeated_frames_and_quotient_mass():
    q = torch.tensor([[1., 0, 0, 0], [-1., 0, 0, 0]]).repeat(8, 1)
    cent, counts = kmeans_joint(q, 6, quotient=True)
    assert len(cent) == 1 and counts.item() == 16
    cent, counts = kmeans_joint(q[:1].repeat(4, 1), 6)
    assert len(cent) == 1 and counts.item() == 4
    measure = NaturalnessMeasure(cent[None], torch.ones(1, 1), quotient=True)
    point = G.uniform_hemisphere((3, 1))
    torch.testing.assert_close(measure.log_mass(point), measure.log_mass(-point))


def test_per_clip_sources_and_stationary_ar():
    spec = FlowSpec(source_ar=0.5)
    p, e = per_clip_sources(spec, ["one", "two"], 41, 12, 16, "cpu")
    many, expr = per_clip_sources(spec, ["three", "two", "one"], 41, 16, 16, "cpu")
    torch.testing.assert_close(p, many[[2, 1], :12], rtol=0, atol=0)
    torch.testing.assert_close(e, expr[[2, 1], :12], rtol=0, atol=0)
    innovations = torch.randn(10000, 6, 1, generator=torch.Generator().manual_seed(2))
    z = ar1_noise(innovations, 0.5)
    torch.testing.assert_close(z.var(0), torch.ones_like(z.var(0)), rtol=0, atol=0.05)
    assert (z[:, 2] * z[:, 3]).mean().item() == pytest.approx(0.5, abs=0.04)


def test_anchored_and_current_expression_at_irregular_times():
    class Generator:
        def __init__(self):
            self.inputs = []

        def __call__(self, pose, expr, t, *_):
            self.inputs.append(expr.clone())
            target = torch.full_like(expr, float(t[0] >= 0.5))
            return pose, target
    x = torch.tensor([1., 0, 0, 0]).reshape(1, 1, 1, 4)
    e = torch.zeros(1, 1, 1)
    for policy, expect in (("current", 0.5), ("anchored", 0.75)):
        gen = Generator()
        grid = sampling_grid(3, [0., 0.5, 0.75, 1.])
        _, final, _ = sample(gen, None, FlowSpec(path="so3"), None, None, torch.ones(1, 1, dtype=torch.bool),
                             x, e, grid, policy)
        assert gen.inputs[-1].item() == expect
        assert final.item() == 1.
    with pytest.raises(ValueError):
        sampling_grid(2, [0., 0., 1.])


def test_auxiliary_loss_obeys_padding_time_and_microbatch(asset):
    x1 = G.aa_to_quat(torch.randn(2, 3, 41, 3) * 0.15)
    pose = G.aa_to_quat(torch.randn(2, 3, 41, 3) * 0.15).requires_grad_()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    inverse = torch.tensor([[25., 0.], [0., 0.]])
    spec = FlowSpec(position_weight=1., velocity_weight=1., aux_min_t=0.5)
    t = torch.tensor([0.6, 0.4])
    weights = group_weights({"wrist": 1., "fingers": 1.})
    counts = {"frames": 4, "pairs": 1}
    with torch.no_grad():
        whole = auxiliary_losses(asset, pose, x1, mask, t, inverse, spec, counts, weights)
        parts = [auxiliary_losses(asset, pose[i:i+1], x1[i:i+1], mask[i:i+1], t[i:i+1],
                                  inverse[i:i+1], spec, counts, weights) for i in range(2)]
    for value, a, b in zip(whole, *parts):
        torch.testing.assert_close(value, a + b)
        assert b == 0
    val = sum(auxiliary_losses(asset, pose, x1, mask, t, inverse, spec, counts, weights))
    val.backward()
    assert pose.grad.isfinite().all()
    assert not pose.grad[1].count_nonzero()


def test_collate_and_paired_dev_statistics():
    feat = torch.randn(3, 133)
    out = collate([(feat, torch.tensor([0., 0.04, 0.04]), "a", "csl"),
                   (feat[:1], torch.tensor([0.]), "b", "csl")])
    torch.testing.assert_close(out[-1], torch.tensor([[25., 0.], [0., 0.]]))
    values = [2., 2., 2.]
    np.testing.assert_allclose(bootstrap_mean(values, ["s1", "s1", "s2"], 99, 0), [2., 2.])
    inputs = [{"text": t, "name": t, "src": "csl"} for t in "abcd"]
    swaps = swap_indices(inputs, [10, 11, 91, 92])
    assert swaps == {0: 1, 1: 0, 2: 3, 3: 2}


def test_conditional_ot_preserves_source_bank_per_text_and_lengths():
    pytest.importorskip("scipy")
    from methods.signmflow_kinematic.coupling import pair_sources

    q = G.aa_to_quat(torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]))
    a = q.view(4, 1, 1, 4).expand(-1, 2, -1, -1)
    e = torch.arange(4.)[:, None, None].expand(-1, 2, -1)
    target = a[[1, 0, 3, 2]]
    mask = torch.tensor([[True, True], [True, True], [True, False], [True, True]])
    p, x, count = pair_sources(a, e, target, mask, ["yes"] * 4, ["csl"] * 3 + ["phoenix"], "quat")
    torch.testing.assert_close(p, a[[1, 0, 2, 3]])
    torch.testing.assert_close(x, e[[1, 0, 2, 3]])
    assert count == 2


def test_coarse_and_quantile_forward_with_short_final_bin():
    from methods.signmflow_kinematic.coarse import coarse_targets
    from methods.signmflow_kinematic.model import SignGenerator

    gen = SignGenerator(n_joints=3, expr_dim=2, latent=8, heads=2, layers=1, ff=16, time_dim=8,
                        text_dim=8, max_frames=9, coarse_stride=4, length_mode="quantile", dropout=0.)
    pose = G.normalize(torch.randn(2, 7, 3, 4))
    expression = torch.randn(2, 7, 2)
    text = torch.randn(2, 3, 8)
    mask = torch.arange(7)[None] < torch.tensor([[7], [3]])
    text_pad = torch.zeros(2, 3, dtype=torch.bool)
    targets, bins = coarse_targets(pose, expression, mask, 4)
    assert targets.shape == (2, 2, 20)
    torch.testing.assert_close(targets[0, 1, -2:], expression[0, 4:].mean(0))
    assert bins.tolist() == [[True, True], [True, False]]
    result = gen.length_outputs(text, text_pad)
    assert ((result[:, 1:] - result[:, :-1]) > 0).all()
    out, e = gen(pose, expression, torch.zeros(2), text, text_pad, ~mask)
    assert out.shape == pose.shape and e.shape == expression.shape
    torch.manual_seed(2)
    loss = (gen.coarse(text, text_pad, mask)[bins] - targets[bins]).square().mean()
    loss.backward()
    assert gen.coarse.prediction.weight.grad.isfinite().all()


def test_generator_resume_matches_uninterrupted_partial_accumulation(tmp_path):
    from types import SimpleNamespace

    from methods.signmflow_kinematic.model import SignGenerator
    from methods.signmflow_kinematic.training import train_generator

    class Corpus:
        key = "fixed-corpus"

        def __len__(self):
            return 5

        def __getitem__(self, i):
            feature = torch.randn(i % 3 + 2, 133, generator=torch.Generator().manual_seed(i)) * 0.1
            return feature, torch.arange(len(feature)) / 30, str(i), "csl"

    class FrozenText(torch.nn.Module):
        def forward(self, texts, _languages, device):
            tokens = torch.tensor([int(t) for t in texts], dtype=torch.float32, device=device)
            return tokens[:, None, None].expand(-1, 2, 8), torch.zeros(len(texts), 2, dtype=torch.bool, device=device)

    def model():
        torch.manual_seed(42)
        result = SignGenerator(latent=8, layers=1, heads=2, ff=16, time_dim=8,
                               max_frames=7, text_dim=8, dropout=0.1, coarse_stride=2)
        result.text = FrozenText()
        return result

    def context(name, hook):
        ckpt = tmp_path / name
        ckpt.mkdir()
        return SimpleNamespace(ckpt_dir=ckpt, log_metrics=hook)

    def interrupt(values, step):
        if step == 2:
            raise RuntimeError("interrupted before epoch 2 save")

    opt = dict(epochs=3, lr=1e-3, text_lr=0., weight_decay=0., warmup_steps=0, freeze_text=True,
               batch_size=2, effective_batch=4, micro_batch=1, workers=0, grad_clip=None,
               length_loss_weight=1., coarse_loss_weight=0.1, validation_batches=1, save_every=0,
               probe_gradients=False)
    data = Corpus()
    codec = MotionCodec(np.zeros(133), np.ones(133), None)
    spec = FlowSpec(path="so3", endpoint_loss="chordal")
    args = (None, spec, data, data, codec, opt, torch.device("cpu"), 0)
    signature = {"format": "test"}
    continuous = model()
    train_generator(continuous, *args, context("continuous", lambda *a, **kw: None), signature)
    stopped = context("interrupted", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_generator(model(), *args, stopped, signature)
    resumed = model()
    train_generator(resumed, *args, context("resumed", lambda *a, **kw: None), signature,
                    resume=stopped.ckpt_dir / "generator.pt")
    for key, actual in resumed.state_dict().items():
        torch.testing.assert_close(continuous.state_dict()[key], actual, rtol=0., atol=0.)


def test_paired_corpus_bleu_equals_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")

    from methods.signmflow_kinematic.backtranslation import paired_corpus_bleu

    refs = ["one two three four five", "six seven eight nine", "ten eleven twelve thirteen"]
    hyps = ["one two three four five", "six seven eight nine", "eleven twelve"]
    baseline = [{"hypothesis": text, "reference": ref} for text, ref in zip(hyps, refs)]
    candidate = [{"hypothesis": text, "reference": text} for text in refs]
    x = paired_corpus_bleu([(baseline, candidate)], ["a", "a", "b"], "13a", 99, 33)
    assert x["per_seed_bleu4_baseline_then_candidate"][0][0] == pytest.approx(
        sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score)
    assert x["per_seed_bleu4_baseline_then_candidate"][0][1] == pytest.approx(100.)
    assert x["delta_bleu4"] > 0
    same = paired_corpus_bleu([(candidate, candidate)], ["a", "a", "b"], "13a", 99, 33)
    assert same["ci_95_delta_bleu4_fixed_seeds"] == [0., 0.]

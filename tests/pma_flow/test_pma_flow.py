"""Unit tests for methods/pma_flow (docs/proposal5). CPU only, tiny networks, no mBART."""

import numpy as np
import pytest
import torch

from slp import geometry as G
from methods.pma_flow import flow as F
from methods.pma_flow.data import MotionTextDataset, collate
from methods.pma_flow.losses import LossWeights, huber, reconstruction_loss
from methods.pma_flow.model import PoseNet
from methods.pma_flow.representation import MotionCodec, flatten, unflatten
from methods.pma_flow.training import EMA, train_network

TEXT_DIM = 16


class TinyNet(PoseNet):
    """PoseNet without mBART: text features are a deterministic function of the string."""

    def __init__(self, mode="flow", cond_anchor=True):
        super().__init__(mode=mode, cond_anchor=cond_anchor, latent=32, layers=1, heads=2, ff=64,
                         time_dim=32, dropout=0.0, max_frames=64, text_model=None, text_dim=TEXT_DIM)
        self.tok = torch.nn.Embedding(97, TEXT_DIM)

    def encode_text(self, texts, srcs, device):
        L = max(len(t) for t in texts)
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        pad = torch.ones(len(texts), L, dtype=torch.bool)
        for i, t in enumerate(texts):
            ids[i, :len(t)] = torch.tensor([ord(c) % 97 for c in t])
            pad[i, :len(t)] = False
        return self.tok(ids.to(device)), pad.to(device)


def _codec():
    return MotionCodec(np.zeros(133, np.float32), np.ones(133, np.float32), np.zeros((41, 3), np.float32))


def _rand_q(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return G.sign_convention(G.normalize(torch.randn(*shape, 4, generator=g)))


def test_codec_round_trip_and_flatten():
    c = MotionCodec(np.random.randn(133).astype(np.float32) * 0.1, np.full(133, 0.5, np.float32),
                    np.random.randn(41, 3).astype(np.float32) * 0.2)
    f = torch.randn(3, 7, 133) * 0.3
    q, e = c.encode(f)
    assert torch.allclose(c.decode(q, e), f, atol=1e-4)
    q2, e2 = unflatten(flatten(q, e))
    assert torch.equal(q2, q) and torch.equal(e2, e)


def test_huber_continuous_and_linear_tail():
    d = torch.tensor([0.0, 0.005, 0.01, 0.02, 0.5])
    h = huber(d, 0.01)
    assert h[0] == 0 and abs(float(h[2]) - 0.005) < 1e-9          # both branches meet at delta
    assert torch.allclose(h[3:], d[3:] - 0.005)


def test_reconstruction_loss_zero_at_target_and_sign_invariant_positions():
    q = _rand_q(2, 5, 41)
    e = torch.randn(2, 5, 10)
    m = torch.ones(2, 5, dtype=torch.bool)
    w = LossWeights(quat=10, pos=0, vel=0, expr=0.1)
    loss, parts = reconstruction_loss(w, None, q, e, q, e, m)
    assert float(loss) == 0.0
    # micro-batch normalisation: halves with the full-batch denominator add up to the full loss
    q_hat = _rand_q(2, 5, 41, seed=1)
    full, _ = reconstruction_loss(w, None, q_hat, e, q, e, m)
    a, _ = reconstruction_loss(w, None, q_hat[:1], e[:1], q[:1], e[:1], m[:1], denom=m.sum())
    b, _ = reconstruction_loss(w, None, q_hat[1:], e[1:], q[1:], e[1:], m[1:], denom=m.sum())
    assert torch.allclose(a + b, full)


def test_flow_loss_backward_and_cfg_drop():
    torch.manual_seed(0)
    net = TinyNet()
    spec = F.FlowSpec(source="anchor", cfg_drop=0.5)
    w = LossWeights()
    q1 = _rand_q(3, 6, 41)
    e1 = torch.randn(3, 6, 10)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[2, 4:] = False
    anchor = flatten(G.normalize(G.expmap(q1, G.tangent_gaussian(q1, 0.1))), e1 + 0.1)
    text, pad = net.encode_text(["abc", "hello", "x"], ["phoenix"] * 3, "cpu")
    loss, parts = F.flow_loss(net, spec, w, None, q1, e1, mask, text, pad, anchor=anchor,
                              sigma=torch.full((41, 1), 0.1), expr_sigma=torch.full((10,), 0.1))
    loss.backward()
    assert torch.isfinite(loss)
    assert net.inp.weight.grad is not None and net.null_text.grad is not None


def test_drop_text_replaces_only_dropped_samples():
    net = TinyNet()
    text, pad = net.encode_text(["abcd", "xy"], ["phoenix"] * 2, "cpu")
    t2, p2 = net.drop_text(text, pad, torch.tensor([True, False]))
    assert torch.equal(t2[1], text[1]) and torch.equal(p2[1], pad[1])
    assert torch.allclose(t2[0], net.null_text[0].expand(4, -1))
    assert p2[0].tolist() == [False, True, True, True]


def test_source_distributions():
    spec = F.FlowSpec(source="anchor")
    torch.manual_seed(0)
    mu = _rand_q(20000, 1, 41, seed=2)
    emu = torch.randn(20000, 1, 10)
    sig = torch.linspace(0.05, 0.3, 41)[:, None]
    x0, e0 = F.sample_source(spec, (20000, 1), "cpu", flatten(mu, emu), sigma=sig,
                             expr_sigma=torch.full((10,), 0.2))
    assert torch.allclose(x0.norm(dim=-1), torch.ones(20000, 1, 41), atol=1e-5)
    d2 = G.logmap(mu, x0).pow(2).sum(-1).mean((0, 1))           # E|Log_mu x0|^2 = 3 sigma^2
    assert torch.allclose(d2, 3 * sig[:, 0] ** 2, rtol=0.05)
    assert abs(float((e0 - emu).std()) - 0.2) < 0.01
    u = F.FlowSpec(source="uniform")
    xu, eu = F.sample_source(u, (10, 5), "cpu")
    assert xu.shape == (10, 5, 41, 4) and (xu[..., 0] >= 0).all()


def test_residual_scales_recovers_known_sigma():
    torch.manual_seed(0)
    n, T = 64, 30
    mu = _rand_q(n, T, 41, seed=3)
    sig = torch.linspace(0.02, 0.25, 41)[:, None]
    x1 = G.sign_convention(G.normalize(G.expmap(mu, G.tangent_gaussian(mu, sig))))
    emu = torch.randn(n, T, 10)
    e1 = emu + 0.3 * torch.randn(n, T, 10)
    codec = _codec()
    feats = codec.decode(x1, e1)
    cache = {"frames": feats.reshape(-1, 133).numpy(), "offsets": np.arange(n + 1) * T,
             "inputs": [{"name": str(i), "text": "t", "src": "phoenix"} for i in range(n)], "key": "k"}
    ds = MotionTextDataset(cache, flatten(mu, emu).reshape(-1, 174).numpy())
    s, es = F.residual_scales(ds, True, codec, torch.device("cpu"))
    assert torch.allclose(s, sig, rtol=0.05, atol=2e-3)
    assert torch.allclose(es, torch.full((10,), 0.3), rtol=0.05)


@pytest.mark.parametrize("kind", ["euler", "replan"])
def test_sampler_returns_last_prediction(kind):
    torch.manual_seed(0)
    net = TinyNet().eval()
    spec = F.FlowSpec(source="anchor")
    text, pad = net.encode_text(["abc", "hello"], ["phoenix"] * 2, "cpu")
    L = [5, 7]
    anchor = flatten(_rand_q(2, 7, 41, seed=4), torch.zeros(2, 7, 10))
    kw = dict(anchor=anchor, sigma=torch.full((41, 1), 0.1), expr_sigma=torch.full((10,), 0.1))
    q, e, mask = F.sample(net, spec, text, pad, L, steps=3, sampler=kind,
                          generator=torch.Generator().manual_seed(1), **kw)
    assert q.shape == (2, 7, 41, 4) and mask.sum(1).tolist() == L
    assert torch.allclose(q.norm(dim=-1), torch.ones(2, 7, 41), atol=1e-5)
    # NFE 1 = the x1-prediction at t = 0 from the source
    q1, e1, _ = F.sample(net, spec, text, pad, L, steps=1, sampler=kind,
                         generator=torch.Generator().manual_seed(1), **kw)
    x0, ex0 = F.sample_source(spec, (2, 7), "cpu", anchor, kw["sigma"], kw["expr_sigma"],
                              torch.Generator().manual_seed(1))
    pose, expr = net(x0, ex0, torch.zeros(2), text, pad, ~mask, anchor=anchor)
    assert torch.allclose(q1, G.normalize(pose), atol=1e-5) and torch.allclose(e1, expr, atol=1e-5)


def test_cfg_weight_one_is_conditional_and_zero_is_unconditional():
    torch.manual_seed(0)
    net = TinyNet(cond_anchor=False).eval()
    spec = F.FlowSpec(source="uniform")
    text, pad = net.encode_text(["abc"], ["phoenix"], "cpu")
    x = _rand_q(1, 4, 41, seed=5)
    e = torch.randn(1, 4, 10)
    t = torch.zeros(1)
    m = torch.zeros(1, 4, dtype=torch.bool)
    null = net.null_like(text, pad)
    xc, ec = F._predict(net, x, e, t, text, pad, m, None, 1.0, null)
    xu, eu = F._predict(net, x, e, t, null[0], null[1], m, None, 1.0, None)
    x0, e0 = F._predict(net, x, e, t, text, pad, m, None, 0.0, null)
    x1, e1 = F._predict(net, x, e, t, text, pad, m, None, 1.0 + 1e-9, null)
    assert torch.allclose(x0, G.align_sign(xu, x0), atol=1e-5) and torch.allclose(e0, eu, atol=1e-5)
    assert torch.allclose(G.align_sign(x1, xc), xc, atol=1e-5) and torch.allclose(e1, ec, atol=1e-5)


def test_alpha_interpolation_endpoints():
    mu = _rand_q(2, 3, 41, seed=6)
    emu = torch.randn(2, 3, 10)
    q = -_rand_q(2, 3, 41, seed=7)   # any sign
    e = torch.randn(2, 3, 10)
    a = flatten(mu, emu)
    y0, ey0 = F.interpolate(a, q, e, 0.0)
    y1, ey1 = F.interpolate(a, q, e, 1.0)
    assert torch.allclose(y0, mu, atol=1e-6) and torch.allclose(ey0, emu)
    assert torch.allclose(G.quat_to_matrix(y1), G.quat_to_matrix(q), atol=1e-5)
    assert torch.allclose(ey1, e)


def test_anchor_mode_ignores_state_and_uses_relative_position():
    net = TinyNet(mode="anchor").eval()
    text, pad = net.encode_text(["abc"], ["phoenix"], "cpu")
    q, e, mask = F.predict_anchor(net, text, pad, [6])
    assert q.shape == (1, 6, 41, 4)
    q2, _, _ = F.predict_anchor(net, text, pad, [12])
    assert not torch.allclose(q2[:, :6], q)   # depends on the total length (relative position)


def test_ema_and_trainer_select_best(tmp_path):
    torch.manual_seed(0)
    n, T = 12, 8
    codec = _codec()
    q = _rand_q(n, T, 41, seed=8)
    feats = codec.decode(q, torch.randn(n, T, 10) * 0.1)
    cache = {"frames": feats.reshape(-1, 133).numpy(), "offsets": np.arange(n + 1) * T,
             "inputs": [{"name": str(i), "text": f"sentence {i}", "src": "phoenix"} for i in range(n)],
             "key": "k"}
    ds = MotionTextDataset(cache)
    net = TinyNet(mode="anchor")
    w = LossWeights(pos=0, vel=0)

    def step_loss(net, f, m, text, pad, anc, denom):
        q1, e1 = codec.encode(f)
        return F.anchor_loss(net, w, None, q1, e1, m, text, pad, denom)

    scores = iter([3.0, 1.0, 2.0])
    hp = {"batch_size": 4, "micro_batch": 2, "effective_batch": 8, "epochs": 3, "lr": 1e-3,
          "text_lr": 1e-4, "freeze_text_epochs": 1, "warmup_steps": 1, "grad_clip": 1.0,
          "ema_decay": 0.9, "val_every": 1, "workers": 0, "amp": "none", "select": "best"}
    out = train_network(net, step_loss, ds, hp, torch.device("cpu"), log=lambda *_: None,
                        ckpt=tmp_path / "a.pt", validate=lambda n_: {"score": next(scores)}, tag="a")
    assert out["best_epoch"] == 2
    best = torch.load(tmp_path / "a_best.pt", weights_only=False)["model"]
    assert all(torch.equal(best[k], v) for k, v in net.state_dict().items())
    loaded = torch.load(tmp_path / "a.pt", weights_only=False)
    assert loaded["epoch"] == 3 and loaded["best_epoch"] == 2
    ema = EMA(torch.nn.Linear(2, 2), 0.5)
    assert ema.n == 0 and len(ema.shadow) == 2


def test_collate_pads_anchors():
    batch = [(torch.ones(3, 133), "a", "phoenix", torch.ones(3, 174)),
             (torch.ones(5, 133), "b", "phoenix", torch.ones(5, 174))]
    f, m, t, s, a = collate(batch)
    assert f.shape == (2, 5, 133) and a.shape == (2, 5, 174) and m.sum().item() == 8
    assert a[0, 3:].abs().sum() == 0


def test_t_sampling_covers_t_near_zero():
    g = torch.Generator().manual_seed(0)
    ln = F.sample_t(F.FlowSpec(t_uniform_mix=0.0), 100000, "cpu", g)
    mix = F.sample_t(F.FlowSpec(t_uniform_mix=0.2), 100000, "cpu", g)
    assert float((ln < 0.05).float().mean()) < 0.005          # logit-normal alone: ~0.2 %
    assert abs(float((mix < 0.05).float().mean()) - 0.2 * 0.05) < 0.003
    assert ((mix > 0) & (mix < 1)).all()


def test_unaligned_replan_sampler_runs():
    net = TinyNet(cond_anchor=False).eval()
    spec = F.FlowSpec(source="uniform", align_pairs=False)
    text, pad = net.encode_text(["abc"], ["phoenix"], "cpu")
    q, e, m = F.sample(net, spec, text, pad, [4], steps=3, sampler="replan",
                       generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(q).all() and torch.allclose(q.norm(dim=-1), torch.ones(1, 4, 41), atol=1e-5)


def test_alpha_sweep_scores_every_prediction_file_and_refuses_test(tmp_path, monkeypatch):
    import json

    import yaml

    import slp.data
    from methods.pma_flow import sweep
    from slp.data.base import DatasetSplits, Split

    rng = np.random.RandomState(0)
    targets = [rng.randn(6, 2, 3) for _ in range(3)]
    inputs = [{"name": f"a/{i}", "text": "t", "src": "phoenix"} for i in range(3)]
    split = Split(inputs, targets)
    monkeypatch.setattr(slp.data, "load_dataset", lambda cfg: DatasetSplits(split, split, split))
    cfg = {"eval": {"metrics": ["dtw_jpe"]}, "method": {"sampler": {"alpha": 1.0}}}
    yaml.safe_dump(cfg, open(tmp_path / "config.yaml", "w"))
    json.dump({"eval_split": "val"}, open(tmp_path / "meta.json", "w"))
    names = [x["name"].replace("/", "__") for x in inputs]
    np.savez(tmp_path / "predictions.npz", **dict(zip(names, [t + 0.1 for t in targets])))
    np.savez(tmp_path / "predictions_alpha0.npz", **dict(zip(names, targets)))
    sweep.main(["--run", str(tmp_path)])
    out = json.load(open(tmp_path / "alpha_sweep.json"))
    assert set(out) == {"0", "1"} and out["0"]["dtw_jpe"] == 0.0 and out["1"]["dtw_jpe"] > 0
    json.dump({"eval_split": "test"}, open(tmp_path / "meta.json", "w"))
    with pytest.raises(SystemExit):
        sweep.main(["--run", str(tmp_path)])

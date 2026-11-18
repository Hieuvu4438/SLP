import torch

from cprflow.objectives.diffusion import CosineSchedule
from cprflow.objectives.flow import endpoint, linear_path
from cprflow.objectives.reductions import per_clip_mse


def test_flow_endpoint_and_derivative():
    data, eps = torch.randn(2, 4, 3, 128), torch.randn(2, 4, 3, 128)
    zero, velocity = linear_path(data, eps, torch.zeros(2))
    one, _ = linear_path(data, eps, torch.ones(2))
    torch.testing.assert_close(zero, eps)
    torch.testing.assert_close(one, data)
    t = torch.tensor([0.3, 0.8])
    u, v = linear_path(data, eps, t)
    torch.testing.assert_close(endpoint(u, v, t), data)
    plus, _ = linear_path(data, eps, t + 0.001)
    torch.testing.assert_close((plus - u) / 0.001, v, atol=5e-4, rtol=5e-4)


def test_per_clip_and_pad_reduction():
    pred = torch.tensor([[[2.0], [999.0], [999.0]], [[1.0], [1.0], [1.0]]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    loss = per_clip_mse(pred, torch.zeros_like(pred), mask)
    torch.testing.assert_close(loss, torch.tensor([4.0, 1.0]))
    assert loss.mean() == 2.5


def test_ddim_indexing_and_ideal_endpoint():
    sched = CosineSchedule()
    assert sched.alpha_bar[0] == 1
    assert (
        sched.indices(16)[0] == 1000 and sched.indices(16)[-2] == 1 and sched.indices(16)[-1] == 0
    )
    data, eps = torch.randn(2, 4, 3, 128), torch.randn(2, 4, 3, 128)
    state = sched.corrupt(data, eps, torch.full((2,), 1000))
    for k, next_k in zip(sched.indices(16)[:-1], sched.indices(16)[1:]):
        # Ideal epsilon corresponding to the current state and clean target.
        a = sched.alpha_bar[k]
        ideal = (state - a.sqrt() * data) / (1 - a).sqrt()
        state = sched.ddim(state, ideal, k, next_k)
    torch.testing.assert_close(state, data, atol=1e-5, rtol=1e-5)


def test_oracle_relation_is_an_explicit_sampler_diagnostic():
    from cprflow.models.dit import Prediction
    from cprflow.sampling.ode import integrate

    seen = []

    def model(z, r, t, **kwargs):
        seen.append((float(t[0]), r.clone()))
        return Prediction(torch.zeros_like(z), torch.ones_like(r) * 99, z)

    z = torch.zeros(1, 4, 2, 128)
    noise = torch.ones(1, 2, 33)
    target = torch.ones_like(noise) * 3
    result, relation, metadata = integrate(
        model,
        z,
        noise,
        {"valid": torch.ones(1, 2, dtype=torch.bool)},
        nfe=2,
        oracle_relation=target,
    )
    torch.testing.assert_close(seen[0][1], noise)
    torch.testing.assert_close(seen[1][1], noise * 0.5 + target * 0.5)
    torch.testing.assert_close(relation, target)
    assert metadata["oracle_relation"] is True

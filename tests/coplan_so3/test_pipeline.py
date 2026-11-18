import pytest
import torch

from .conftest import freeze_plan
from methods.coplan_so3.data.dataset import TrainMoments
from methods.coplan_so3.geometry import so3
from methods.coplan_so3.sampling import generate


def test_stage_objectives_and_freeze(model, batch):
    for stage in ("A", "B", "C"):
        if stage == "B":
            freeze_plan(model, batch)
        model.set_stage(stage)
        model.train()
        loss, terms = model.objective(batch, torch.Generator().manual_seed(3))
        assert torch.isfinite(loss), terms
        loss.backward()
        active = {"A": model.plan, "B": model.prior, "C": model.dense}[stage]
        grads = [p.grad for p in active.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
        model.zero_grad(set_to_none=True)
    assert not model.plan.training
    assert not any(p.requires_grad for p in model.plan.parameters())
    with pytest.raises(RuntimeError):
        model.set_stage("A")


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("budget", [5, 10, 20])
def test_inference_leakage_nfe_reproducible(model, batch, solver, budget):
    freeze_plan(model, batch)
    model.set_stage("C")
    # Median length 5 makes the main text-only route executable without training.
    with torch.no_grad():
        model.length_model.head[-1].weight.zero_()
        model.length_model.head[-1].bias.copy_(torch.tensor([5.0, 1.0]).log())
    out = generate(model, ["hello", "world"], [0, 1], seed=4, total_budget=budget, solver=solver)
    again = generate(model, ["hello", "world"], [0, 1], seed=4, total_budget=budget, solver=solver)
    assert out.metadata["total_flow_nfe"] == budget
    assert not out.metadata["length_controlled"]
    assert out.lengths.tolist() == [5, 5]
    torch.testing.assert_close(out.rotations, again.rotations, rtol=0, atol=0)
    so3.assert_valid(out.rotations)
    controlled = generate(
        model,
        ["hello", "world"],
        [0, 1],
        seed=4,
        total_budget=budget,
        solver=solver,
        lengths=[3, 1],
    )
    assert controlled.metadata["length_controlled"]
    torch.testing.assert_close(controlled.rotations[1, 1:], torch.eye(3).expand(2, 5, 3, 3))
    assert (controlled.expression[1, 1:] == 0).all()


def test_train_statistics_and_inverse():
    stats = TrainMoments(2)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
    m = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError):
        stats.update(x, m, split="val")
    stats.update(x, m, split="train")
    stats.finalize(split="train")
    torch.testing.assert_close(stats.mean, torch.tensor([2.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(stats.inverse(stats.normalize(x)), x)
    with pytest.raises(ValueError):
        stats.update(x, m, split="train")

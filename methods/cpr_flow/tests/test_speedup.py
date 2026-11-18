import copy

import pytest
import torch

from cprflow.models.ae import MotionAE
from cprflow.objectives.lag import lag_loss, lag_loss_reference


def test_masked_batch_matches_reference_outputs_latents_input_and_parameter_gradients():
    torch.manual_seed(17)
    reference = MotionAE().double()
    batched = copy.deepcopy(reference)
    batched.execution = "masked_batch"
    x = torch.randn(3, 13, 296, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([5, 8, 13])
    y, z = reference(x, lengths)
    a, b = batched(x, lengths)
    torch.testing.assert_close(y, a, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(z, b, atol=1e-9, rtol=1e-8)
    gradient = torch.autograd.grad(
        y.square().sum() + z.square().sum(), (x, *reference.parameters())
    )
    other = torch.autograd.grad(a.square().sum() + b.square().sum(), (x, *batched.parameters()))
    for old, new in zip(gradient, other):
        torch.testing.assert_close(old, new, atol=1e-8, rtol=1e-7)
    padded = x.detach().clone()
    padded[0, 5:] = 999
    changed, _ = batched(padded, lengths)
    torch.testing.assert_close(a[0, :5], changed[0, :5], atol=1e-9, rtol=1e-8)
    assert not changed[0, 5:].any()


@pytest.mark.parametrize("length", [12, 32, 67, 129])
def test_vectorized_lag_loss_gradient_and_static_target_prediction_match_reference(length):
    torch.manual_seed(19)
    target = torch.randn(length, 4)
    target[:, 3] = 0
    predicted = torch.randn(length, 4, requires_grad=True)
    reference, count = lag_loss_reference(predicted, target)
    result, actual = lag_loss(predicted, target)
    assert actual == count
    torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-6)
    g = torch.autograd.grad(reference, predicted)[0]
    h = torch.autograd.grad(result, predicted)[0]
    torch.testing.assert_close(g, h, atol=2e-6, rtol=2e-5)
    static = torch.zeros_like(predicted, requires_grad=True)
    a, c = lag_loss_reference(static, target)
    b, d = lag_loss(static, target)
    assert c == d
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.requires_body_model
@pytest.mark.parametrize("step", [0, 1])
def test_packed_fk_objective_matches_reference_loss_and_latent_gradient(
    body_model, normalizer, identity_motion, step
):
    from cprflow.data.collate import collate_motion
    from cprflow.objectives.ae import ae_objective

    rows = [
        {
            "x": identity_motion(n).numpy(),
            "timestamps": torch.arange(n).numpy() / 30,
            "sign_language_id": 0,
            "dataset_id": "csl_daily",
            "text": "synthetic fixture",
            "uid": str(n),
        }
        for n in (5, 8)
    ]
    batch = collate_motion(rows, normalizer)
    pred = (batch.x + torch.randn_like(batch.x) * 0.001).requires_grad_()
    with torch.no_grad():
        target = [body_model(torch.from_numpy(r["x"])) for r in rows]
    scales = {
        key: 1.0 for key in ("body", "left", "right", "left_local", "right_local", "expression")
    }
    reference, _ = ae_objective(
        pred, batch, normalizer, body_model, scales, step, target_geometry=target
    )
    result, _ = ae_objective(
        pred, batch, normalizer, body_model, scales, step, target_geometry=target, batched_fk=True
    )
    torch.testing.assert_close(result, reference, atol=1e-7, rtol=1e-5)
    a = torch.autograd.grad(reference.sum(), pred)[0]
    b = torch.autograd.grad(result.sum(), pred)[0]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_runtime_migration_rejects_recipe_change_and_accepts_batching_only():
    from cprflow.config import DEFAULTS
    from cprflow.training.runtime_migration import config_changes

    old = copy.deepcopy(DEFAULTS)
    new = copy.deepcopy(old)
    new["ae"]["execution"] = "masked_batch"
    new["train"].update(microbatch_per_rank=8, accumulate=8)
    new["geometry"]["fk_chunk_frames"] = 512
    assert set(config_changes(old, new)) == {
        "ae.execution",
        "train.microbatch_per_rank",
        "train.accumulate",
        "geometry.fk_chunk_frames",
    }
    new["train"]["max_updates"] = 100
    with pytest.raises(ValueError, match="Not an execution-only"):
        config_changes(old, new)


def test_checkpoint_migration_preserves_training_state_and_original(tmp_path):
    from cprflow.config import DEFAULTS
    from cprflow.training.checkpoint import atomic_save, load_checkpoint
    from cprflow.training.runtime_migration import migrate_checkpoint, tensor_tree_equal

    old = copy.deepcopy(DEFAULTS)
    new = copy.deepcopy(old)
    new["ae"]["execution"] = "masked_batch"
    new["train"].update(microbatch_per_rank=8, accumulate=8)
    dependencies = {"config": "old", "manifest": "frozen"}
    keys = ("model", "optimizer", "scheduler", "scaler", "ema", "sampler", "rank_rng")
    state = {key: {"value": torch.randn(3)} for key in keys}
    state.update(config=old, provenance=dependencies, world_size=1, successful_optimizer_step=346)
    path = tmp_path / "last.pt"
    atomic_save(state, path)
    receipt = tmp_path / "migration" / "receipt.json"
    output = migrate_checkpoint(path, new, {**dependencies, "config": "new"}, receipt)
    loaded = load_checkpoint(path)
    assert output["state_preservation_verified"]
    assert loaded["successful_optimizer_step"] == 346
    assert all(tensor_tree_equal(loaded[key], state[key]) for key in keys)
    assert tensor_tree_equal(load_checkpoint(output["backup"]), state)


@pytest.mark.requires_body_model
def test_packed_flow_auxiliary_matches_reference_loss_and_latent_gradient(
    body_model, normalizer, identity_motion
):
    from cprflow.config import DEFAULTS
    from cprflow.data.collate import collate_motion
    from cprflow.data.stats import Normalizer
    from cprflow.training.tasks import TaskObjective

    rows = [
        {
            "x": identity_motion(n).numpy(),
            "timestamps": torch.arange(n).numpy() / 30,
            "sign_language_id": 0,
            "dataset_id": "csl_daily",
            "text": "fixture",
            "uid": str(n),
        }
        for n in (33, 38, 41)
    ]
    batch = collate_motion(rows, normalizer)
    obj = TaskObjective.__new__(TaskObjective)
    obj.config = copy.deepcopy(DEFAULTS)
    obj.config["loss"].update(geom=0.1, cycle=0.05, lag=0.01, semantic=0.0)
    obj.device = torch.device("cpu")
    obj.motion_norm = normalizer
    obj.latent_norm = Normalizer(torch.zeros(4, 1, 128), torch.ones(4, 1, 128))
    obj.relation_norm = Normalizer(torch.zeros(33), torch.ones(33))
    obj.ae = MotionAE(normalizer).freeze()
    obj.body = body_model
    obj.geometry_cache = None
    obj.scales = {
        key: 1.0
        for key in (
            "body",
            "left",
            "right",
            "left_local",
            "right_local",
            "expression",
            "activity_left",
            "activity_right",
            "activity_head",
            "activity_face",
        )
    }
    z = torch.randn(3, 4, 21, 128, requires_grad=True)
    relation = torch.randn(3, 21, 33, requires_grad=True)
    args = (
        batch,
        torch.ones(3),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor([True, False, True]),
        rows,
        rows,
    )
    reference, _ = obj.auxiliary(z, relation, *args)
    a = torch.autograd.grad(reference.sum(), (z, relation))
    obj.config["ae"]["execution"] = "masked_batch"
    obj.ae.execution = "masked_batch"
    result, _ = obj.auxiliary(z, relation, *args)
    b = torch.autograd.grad(result.sum(), (z, relation))
    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-3)

import copy

import torch
import pytest

from cprflow.config import DEFAULTS
from cprflow.data.sampler import BalancedBatchSampler
from cprflow.training.engine import Trainer, seed_everything


class FixtureDataset:
    def __init__(self):
        self.rows = [
            {"uid": str(i), "dataset_id": "csl_daily", "frames": 2, "value": i / 10}
            for i in range(12)
        ]

    def __getitem__(self, index):
        return self.rows[index]


@pytest.mark.parametrize("workers", [0, 2])
def test_twenty_update_exact_resume_with_rng_ema_sampler(tmp_path, workers):
    config = copy.deepcopy(DEFAULTS)
    config["run"].update(id="uninterrupted", out_dir=str(tmp_path), synthetic=True)
    config["train"].update(
        max_updates=20,
        effective_batch=4,
        microbatch_per_rank=1,
        accumulate=4,
        checkpoint_every=100,
        precision="fp32",
        warmup_updates=3,
    )
    config["data"].update(workers_per_rank=workers, pin_memory=False)
    dataset = FixtureDataset()

    def build(cfg):
        seed_everything(17, True)
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1)
        )
        trainer = Trainer(model, cfg, {"synthetic": True, "profile": "test"})
        sampler = BalancedBatchSampler(dataset.rows, 4, 17)
        return trainer, sampler

    def objective(model, rows, step, logical_rows):
        x = torch.tensor([[r["value"]] for r in rows])
        prediction = model(x)
        loss = (prediction[:, 0] - x[:, 0].square()).square()
        return loss, {}

    complete, full_sampler = build(config)
    complete.run(dataset, full_sampler, objective)
    config2 = copy.deepcopy(config)
    config2["run"]["id"] = "resumed"
    partial, sampler = build(config2)
    partial.run(dataset, sampler, objective, until=8)
    next_ids = sampler.batch(8)
    resumed, sampler2 = build(config2)
    resumed.resume(partial.directory / "checkpoints/last.pt", sampler2)
    assert sampler2.batch(8) == next_ids
    resumed.run(dataset, sampler2, objective)
    assert resumed.step == complete.step == 20
    assert resumed.scheduler.state_dict() == complete.scheduler.state_dict()
    assert resumed.ema.updates == 20
    for key, value in complete.raw.state_dict().items():
        torch.testing.assert_close(value, resumed.raw.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(
            complete.ema.shadow[key], resumed.ema.shadow[key], rtol=0, atol=0
        )
    assert full_sampler.state_dict() == sampler2.state_dict()


def test_worker_prefetch_does_not_advance_checkpoint_sampler_cursor():
    from cprflow.data.loader import LogicalBatchLoader

    dataset = FixtureDataset()
    sampler = BalancedBatchSampler(dataset.rows, 4, 17)
    loader = LogicalBatchLoader(dataset, sampler, 3, 6, workers=2)
    rows = loader.get(3)
    assert [r["uid"] for r in rows] == [dataset.rows[i]["uid"] for i in sampler.plan(3)]
    assert sampler.cursor == 0
    assert loader.get(3) is rows
    next_rows = loader.get(4)
    assert [r["uid"] for r in next_rows] == [dataset.rows[i]["uid"] for i in sampler.plan(4)]
    loader.close()

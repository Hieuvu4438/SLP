"""Persistent per-run CPU workers with pure step plans and replayable logical batches."""

import torch
from torch.utils.data import DataLoader


def collate_cpu(rows):
    for row in rows:
        for key in ("x", "timestamps"):
            if key in row:
                row[key] = torch.as_tensor(row[key])
    return rows


def cpu_worker(_):
    torch.set_num_threads(1)


class LogicalBatchLoader:
    def __init__(
        self,
        dataset,
        sampler,
        start,
        end,
        rank=0,
        world_size=1,
        workers=4,
        pin_memory=False,
        seed=17,
    ):
        def plans():
            for step in range(start, end):
                yield sampler.shard(sampler.plan(step), rank, world_size)

        rng = torch.Generator().manual_seed(seed + rank)
        self.loader = DataLoader(
            dataset,
            batch_sampler=plans(),
            collate_fn=collate_cpu,
            num_workers=workers,
            pin_memory=pin_memory,
            worker_init_fn=cpu_worker,
            generator=rng,
            multiprocessing_context="spawn" if workers else None,
        )
        self.iterator = iter(self.loader)
        self.next_step = start
        self.cached_step = None
        self.cached_rows = None

    def get(self, step):
        if step == self.cached_step:
            return self.cached_rows
        if step != self.next_step:
            raise ValueError(
                "Logical loader requires sequential steps or exact current-batch replay"
            )
        self.cached_rows = next(self.iterator)
        self.cached_step = step
        self.next_step = step + 1
        return self.cached_rows

    def close(self):
        # Releasing the iterator shuts down its worker pool. No worker survives a run.
        self.iterator = None
        self.loader = None

import torch
from torch.nn.utils.rnn import pad_sequence

from cprflow.geometry.temporal import lengths_mask
from cprflow.types import MotionBatch, TextBatch


def collate_motion(rows, normalizer=None, device="cpu"):
    xs = [torch.as_tensor(r["x"], dtype=torch.float32, device=device) for r in rows]
    if normalizer is not None:
        xs = [normalizer(x) for x in xs]
    lengths = torch.tensor([len(x) for x in xs], device=device)
    return MotionBatch(
        pad_sequence(xs, batch_first=True),
        lengths,
        lengths_mask(lengths),
        pad_sequence(
            [torch.as_tensor(r["timestamps"], device=device, dtype=torch.float32) for r in rows],
            batch_first=True,
        ),
        torch.tensor([r["sign_language_id"] for r in rows], device=device),
        [r["dataset_id"] for r in rows],
        [r["text"] for r in rows],
        [r["uid"] for r in rows],
    )


def collate_text(features, device="cpu"):
    xs = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in features]
    lengths = torch.tensor([len(x) for x in xs], device=device)
    return TextBatch(pad_sequence(xs, batch_first=True), lengths_mask(lengths))

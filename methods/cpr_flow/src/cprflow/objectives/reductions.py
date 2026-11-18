import torch.nn.functional as F


def per_clip_mean(error, mask):
    mask = mask.to(error.device)
    if error.shape[:2] != mask.shape:
        raise ValueError("Expected [B,T,...] error and [B,T] valid mask")
    width = error[0, 0].numel() if error.shape[1] else error.shape[-1]
    expanded = mask.reshape(*mask.shape, *((1,) * (error.ndim - 2)))
    return (error.float() * expanded).flatten(1).sum(-1) / (mask.sum(-1).clamp_min(1) * width)


def per_clip_mse(pred, target, mask):
    return per_clip_mean((pred.float() - target.float()).square(), mask)


def per_clip_huber(pred, target, mask):
    return per_clip_mean(F.huber_loss(pred.float(), target.float(), reduction="none"), mask)


def logical_batch_loss(per_clip, global_count, world_size=1):
    return world_size * per_clip.sum() / global_count

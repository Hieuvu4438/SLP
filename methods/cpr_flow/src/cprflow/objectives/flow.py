import torch

from cprflow.objectives.reductions import per_clip_mse


def expand_time(t, x):
    return t.reshape(len(t), *((1,) * (x.ndim - 1)))


def linear_path(target, noise, t):
    w = expand_time(t, target)
    return (1 - w) * noise + w * target, target - noise


def endpoint(state, velocity, t):
    return state + (1 - expand_time(t, state)) * velocity


def motion_fm(pred, target, mask):
    return torch.stack([per_clip_mse(pred[:, p], target[:, p], mask) for p in range(4)]).mean(0)


def auxiliary_ramp(step, start=20000, end=40000):
    return min(1.0, max(0.0, (step - start) / max(1, end - start)))

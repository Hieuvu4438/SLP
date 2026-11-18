import os

import torch
import torch.distributed as dist


def setup(device="cpu", backend="nccl"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not dist.is_initialized():
        if device.startswith("cuda"):
            torch.cuda.set_device(local)
            device = f"cuda:{local}"
        dist.init_process_group(backend=backend if device.startswith("cuda") else "gloo")
    return rank, world, torch.device(device)


def all_sum(tensor):
    tensor = tensor.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def gather_objects(obj):
    if not dist.is_initialized():
        return [obj]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj)
    return result


def gather_grad(tensor):
    if not dist.is_initialized():
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(tensor), 0)


def coordinated_error(error):
    failures = gather_objects(None if error is None else str(error))
    if any(failures):
        raise RuntimeError(f"Coordinated data failure by rank: {failures}")

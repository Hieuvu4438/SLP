import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from cprflow.objectives.reductions import logical_batch_loss
from cprflow.training.distributed import gather_grad


def worker(rank, init_file, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(0.3)
    ddp = DistributedDataParallel(model)
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10
    for index in range(rank, 6, 2):
        prediction = ddp(x[index : index + 1])
        per_clip = prediction.square().flatten()
        logical_batch_loss(per_clip, 6, 2).backward()
    torch.save(model.weight.grad, output + str(rank) + ".pt")
    # Gradient-aware gather includes key gradients from other ranks.
    value = torch.tensor([[float(rank + 1)]], requires_grad=True)
    gathered = gather_grad(value)
    (gathered.square().sum() / 2).backward()
    torch.save(value.grad, output + "g" + str(rank) + ".pt")
    dist.destroy_process_group()


def test_two_process_gloo_accumulation_and_gather(tmp_path):
    init = str(tmp_path / "init")
    out = str(tmp_path / "gradient")
    mp.spawn(worker, args=(init, out), nprocs=2, join=True)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(0.3)
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10
    model(x).square().mean().backward()
    for rank in range(2):
        actual = torch.load(out + str(rank) + ".pt", weights_only=True)
        torch.testing.assert_close(actual, model.weight.grad)
        grad = torch.load(out + "g" + str(rank) + ".pt", weights_only=True)
        torch.testing.assert_close(grad, torch.tensor([[2.0 * (rank + 1)]]))


def contrastive_worker(rank, init_file, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    from cprflow.objectives.semantic import multi_positive_nce

    torch.manual_seed(17)
    layer = torch.nn.Linear(3, 4, bias=False)
    model = DistributedDataParallel(layer)
    values = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10
    local = model(values[rank * 3 : (rank + 1) * 3])
    global_keys = gather_grad(local)
    positive = torch.zeros(3, 6, dtype=torch.bool)
    positive[torch.arange(3), torch.arange(rank * 3, (rank + 1) * 3)] = True
    vector, _ = multi_positive_nce(
        local, global_keys, positive, torch.ones_like(positive), temperature=0.7
    )
    logical_batch_loss(vector, 6, 2).backward()
    torch.save(layer.weight.grad, output + str(rank) + ".pt")
    dist.destroy_process_group()


def test_contrastive_global_negative_pool_gradient(tmp_path):
    from cprflow.objectives.semantic import multi_positive_nce

    out = str(tmp_path / "contrastive")
    mp.spawn(
        contrastive_worker, args=(str(tmp_path / "init_contrastive"), out), nprocs=2, join=True
    )
    torch.manual_seed(17)
    layer = torch.nn.Linear(3, 4, bias=False)
    values = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10
    embedding = layer(values)
    loss, _ = multi_positive_nce(
        embedding,
        embedding,
        torch.eye(6, dtype=torch.bool),
        torch.ones(6, 6, dtype=torch.bool),
        temperature=0.7,
    )
    loss.mean().backward()
    for rank in range(2):
        torch.testing.assert_close(
            torch.load(out + str(rank) + ".pt", weights_only=True),
            layer.weight.grad,
            atol=1e-6,
            rtol=1e-5,
        )

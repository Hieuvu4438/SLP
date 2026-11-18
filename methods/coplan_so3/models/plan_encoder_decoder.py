import torch
from torch import nn

from ..data.dataset import TrainMoments, length_mask
from ..geometry import so3
from .components import AxialBlock, embedding
from .part_temporal_transformer import MotionHead, PartInput


class PlanAutoencoder(nn.Module):
    def __init__(
        self, manifest, plan_dim=256, dim=256, stride=4, layers=4, heads=8, ff=1024, dropout=0.1
    ):
        super().__init__()
        self.manifest, self.stride, self.dim = manifest, stride, dim
        self.input = PartInput(manifest, dim)
        self.encoder = nn.ModuleList([AxialBlock(dim, heads, ff, dropout) for _ in range(layers)])
        self.to_plan = nn.Linear(dim * 4, plan_dim)
        self.from_plan = nn.Linear(plan_dim, dim)
        self.query = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.decoder = nn.ModuleList([AxialBlock(dim, heads, ff, dropout) for _ in range(layers)])
        self.head = MotionHead(manifest, dim, 6)
        self.stats = TrainMoments(plan_dim)
        self.register_buffer("frozen", torch.tensor(False))
        self.register_buffer("degenerate_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("decoded_count", torch.zeros((), dtype=torch.long))

    def slot_mask(self, lengths):
        return length_mask((lengths + self.stride - 1) // self.stride)

    def encode_raw(self, r, e, mask, times):
        h = self.input(r, e, times)
        for block in self.encoder:
            h = block(h, mask)
        b, t = mask.shape
        k = (t + self.stride - 1) // self.stride
        padding = k * self.stride - t
        h = torch.nn.functional.pad(h.flatten(-2), (0, 0, 0, padding))
        valid = torch.nn.functional.pad(mask, (0, padding)).reshape(b, k, self.stride)
        h = h.reshape(b, k, self.stride, -1)
        h = (h * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]
        zmask = valid.any(2)
        return self.to_plan(h) * zmask[..., None], zmask

    def decode_raw(self, z, zmask, mask, times):
        memory = (
            self.from_plan(z)
            + embedding(torch.arange(z.shape[1], device=z.device, dtype=z.dtype), self.dim)[None]
        )
        h = self.query[None, None] + embedding(times, self.dim)[..., None, :]
        for block in self.decoder:
            h = block(h, mask, memory, zmask)
        raw, e = self.head(h)
        r, degenerate = so3.from_6d(raw, return_degenerate=True)
        with torch.no_grad():
            self.degenerate_count.add_((degenerate & mask[..., None]).sum())
            self.decoded_count.add_(mask.sum() * self.manifest.joints)
        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        return torch.where(mask[..., None, None, None], r, eye), e * mask[..., None]

    def posterior(self, r, e, mask, times, sigma=0.05, generator=None):
        if not self.frozen:
            raise RuntimeError("Freeze plan distribution before stages B/C")
        z, zmask = self.encode_raw(r, e, mask, times)
        z = self.stats.normalize(z)
        noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        return (z + sigma * noise) * zmask[..., None], zmask

    def decode(self, z, zmask, mask, times):
        return self.decode_raw(self.stats.inverse(z), zmask, mask, times)

    def freeze(self):
        if not self.stats.ready:
            raise RuntimeError("Fit latent statistics on train before freezing plan")
        self.requires_grad_(False)
        self.frozen.fill_(True)
        self.eval()

    def train(self, mode=True):
        return super().train(False if self.frozen else mode)

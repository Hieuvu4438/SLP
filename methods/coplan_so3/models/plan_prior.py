import torch
from torch import nn

from .components import Condition, LanguageAdapters, embedding, mlp


class PlanPrior(nn.Module):
    def __init__(self, text_dim, plan_dim=256, dim=256, layers=4, heads=8, ff=1024, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(plan_dim, dim)
        self.condition = Condition(dim, text_dim, plan_dim)
        self.time = mlp(dim, dim, dim)
        layer = nn.TransformerDecoderLayer(
            dim, heads, ff, dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerDecoder(layer, layers)
        self.adapter = LanguageAdapters(dim)
        self.output = nn.Linear(dim, plan_dim)
        self.dim = dim

    def forward(self, z, s, text, text_mask, language, lengths, zmask):
        pos = embedding(torch.arange(z.shape[1], device=z.device, dtype=z.dtype), self.dim)
        h = self.input(z) + pos[None] + self.time(embedding(s * 1000, self.dim))[:, None]
        memory, mask = self.condition(text, text_mask, language, lengths)
        h = self.blocks(h, memory, tgt_key_padding_mask=~zmask, memory_key_padding_mask=~mask)
        return self.output(self.adapter(h, language)) * zmask[..., None]


class LengthModel(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.language = nn.Embedding(3, text_dim)
        self.head = mlp(text_dim, 2, text_dim)

    def forward(self, text, mask, language):
        pooled = (text * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        mu, log_sigma = self.head(pooled + self.language(language)).unbind(-1)
        return mu, log_sigma.clamp(-5, 3)

    def nll(self, lengths, text, mask, language):
        mu, ls = self(text, mask, language)
        return (0.5 * ((lengths.to(mu).log() - mu) * (-ls).exp()).square() + ls).mean()

    def predict(self, text, mask, language, max_frames, generator=None, stochastic=False):
        mu, ls = self(text, mask, language)
        if stochastic:
            mu = mu + ls.exp() * torch.randn(mu.shape, device=mu.device, generator=generator)
        # Resource violations are explicit, never silently truncated.
        if (
            not torch.isfinite(mu).all()
            or (mu > torch.tensor(float(max_frames), device=mu.device).log()).any()
        ):
            raise ValueError(
                "Predicted length exceeds max_frames; increase the disclosed resource limit"
            )
        return mu.exp().round().long().clamp_min(1)

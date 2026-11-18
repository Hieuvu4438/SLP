import math

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(hidden, mask):
    return (hidden.float() * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]


class DurationModel(nn.Module):
    def __init__(self, hidden=1024, sigma_floor=0.05, sigma_max=2.0):
        super().__init__()
        self.language = nn.Embedding(3, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(800),
            nn.Linear(800, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )
        self.sigma_floor, self.sigma_max = sigma_floor, sigma_max

    def forward(self, text, mask, language):
        h = torch.cat((masked_mean(text, mask), self.language(language)), -1)
        mu, rho = self.net(h).float().unbind(-1)
        return mu, (F.softplus(rho) + self.sigma_floor).clamp(max=self.sigma_max)

    def nll(self, text, mask, language, lengths):
        mu, sigma = self(text, mask, language)
        target = lengths.float().log()
        return (
            target
            + sigma.log()
            + 0.5 * ((target - mu) / sigma).square()
            + 0.5 * math.log(2 * math.pi)
        )

    def predict(self, text, mask, language, maximum=2048):
        mu, _ = self(text, mask, language)
        median = mu.exp()
        preclamp = 2 * torch.floor((median + 1) / 2)
        return preclamp.clamp(2, maximum).long(), {
            "median": median,
            "preclamp": preclamp,
            "clipped": (preclamp < 2) | (preclamp > maximum),
        }

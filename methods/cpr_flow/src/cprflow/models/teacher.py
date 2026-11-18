"""Train-only sign motion/text representation with masked parameter reconstruction."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from cprflow.geometry.temporal import lengths_mask, pool2
from cprflow.models.duration import masked_mean
from cprflow.models.embeddings import sinusoidal
from cprflow.types import PART_DIMS


class SignTeacher(nn.Module):
    def __init__(self, hidden=256, blocks=2, heads=4, ffn=1024, dropout=0.1, embedding=256):
        super().__init__()
        self.hidden = hidden
        self.inputs = nn.ModuleList([nn.Linear(d, hidden) for d in PART_DIMS])
        self.mask_projection = nn.ModuleList([nn.Linear(1, hidden, bias=False) for _ in PART_DIMS])
        self.temporal = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        hidden, heads, ffn, dropout, batch_first=True, norm_first=True
                    ),
                    blocks,
                    enable_nested_tensor=False,
                )
                for _ in PART_DIMS
            ]
        )
        self.parts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, embedding))
                for _ in PART_DIMS
            ]
        )
        self.reconstruct = nn.ModuleList([nn.Linear(hidden, d) for d in PART_DIMS])
        self.global_motion = nn.Sequential(
            nn.Linear(embedding * 4, 512), nn.SiLU(), nn.Linear(512, embedding)
        )
        self.language = nn.Embedding(3, 32)
        self.text_net = nn.Sequential(nn.Linear(800, 512), nn.SiLU(), nn.Linear(512, embedding))

    def encode_text(self, text, mask, language):
        return F.normalize(
            self.text_net(
                torch.cat((masked_mean(text, mask), self.language(language)), -1)
            ).float(),
            dim=-1,
        )

    def forward(self, x, lengths, times, corruption=None):
        features = []
        embeddings = []
        recon = []
        for i, (part, proj, temporal, head, decoder) in enumerate(
            zip(x.split(PART_DIMS, -1), self.inputs, self.temporal, self.parts, self.reconstruct)
        ):
            sequences = []
            for b, n in enumerate(lengths):
                n = int(n)
                h = proj(part[b, :n])
                if corruption is not None:
                    h = h + self.mask_projection[i](corruption[b, :n, None].float())
                h = pool2(h[None], lengths[b : b + 1])[0]
                ts = pool2(times[b : b + 1, :n, None], lengths[b : b + 1])[0, :, 0]
                sequences.append(h + sinusoidal(ts * 30, self.hidden))
            low_lengths = (lengths + 1) // 2
            valid = lengths_mask(low_lengths)
            h = temporal(pad_sequence(sequences, batch_first=True), src_key_padding_mask=~valid)
            h = h * valid[..., None]
            features.append(h)
            embeddings.append(F.normalize(head(masked_mean(h, valid)).float(), dim=-1))
            recon.append(decoder(h).repeat_interleave(2, 1)[:, : x.shape[1]])
        parts = torch.stack(embeddings, 1)
        global_embed = F.normalize(self.global_motion(parts.flatten(1)).float(), dim=-1)
        return {
            "global": global_embed,
            "parts": parts,
            "temporal": torch.stack(features, 1),
            "reconstruction": torch.cat(recon, -1),
        }

    def freeze(self):
        self.requires_grad_(False)
        return self.eval()


def corrupt_spans(x, lengths, probability=0.3, span_min=2, span_max=10):
    corrupted = x.clone()
    mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    for i, n in enumerate(lengths):
        n = int(n)
        needed = max(1, round(n * probability))
        while int(mask[i, :n].sum()) < needed:
            start = int(torch.randint(n, (1,), device=x.device))
            width = int(torch.randint(span_min, span_max + 1, (1,), device=x.device))
            mask[i, start : min(n, start + width)] = True
    return corrupted.masked_fill(mask[..., None], 0), mask

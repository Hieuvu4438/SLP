"""Frozen mT5 backbone, trainable projection/language embeddings; no implicit download."""
from __future__ import annotations

import json
from pathlib import Path
import re

import torch
from torch import Tensor, nn

LANGUAGES = {"en_ASL": 0, "zh_CSL": 1, "de_DGS": 2}


class TextConditioner(nn.Module):
    def __init__(self, encoder: nn.Module, output_width: int = 512, num_languages: int = 3):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.projection = nn.Linear(encoder.config.d_model, output_width)
        self.language = nn.Embedding(num_languages, output_width)
        self.output_width = output_width
        self.pretrained_spec: dict | None = None

    @classmethod
    def from_pretrained(cls, *, output_width: int = 512,
                        local_files_only: bool = True, spec: dict | None = None):
        # Lazy imports: importing the method never instantiates/downloads a large model.
        from transformers import AutoTokenizer, MT5EncoderModel

        spec = (json.loads(Path(__file__).with_name("text_model.json").read_text())
                if spec is None else dict(spec))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", spec["model_id"]):
            raise ValueError("pretrained backbone references must name an immutable HF repository")
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("pin the pretrained backbone to a full 40-character commit")
        tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["revision"], local_files_only=local_files_only
        )
        encoder, info = MT5EncoderModel.from_pretrained(
            spec["model_id"], revision=spec["revision"], local_files_only=local_files_only,
            output_loading_info=True,
        )
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError("pretrained encoder did not load completely; refusing frozen random weights")
        result = cls(encoder, output_width)
        result.pretrained_spec = {"model_id": spec["model_id"], "revision": spec["revision"]}
        return result, tokenizer

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen weights alone would not disable backbone dropout
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                language_ids: Tensor) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("expected token IDs and attention mask [B,N]")
        if input_ids.dtype != torch.long or language_ids.dtype != torch.long:
            raise ValueError("tokens and language IDs must be int64")
        if language_ids.shape != (input_ids.shape[0],):
            raise ValueError("one target-language ID is required per input")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention mask must contain only binary values")
        if bool(((language_ids < 0) | (language_ids >= self.language.num_embeddings)).any()):
            raise ValueError("unknown target language")
        valid = attention_mask.bool()
        if bool((~valid.any(-1)).any()):
            raise ValueError("empty text is not a valid conditioning sequence")
        with torch.no_grad():
            encoded = self.encoder(input_ids=input_ids, attention_mask=valid.long(),
                                   return_dict=True).last_hidden_state
        features = self.projection(encoded) + self.language(language_ids)[:, None]
        return torch.where(valid[..., None], features, torch.zeros_like(features)), valid

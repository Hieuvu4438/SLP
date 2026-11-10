"""Alternative frozen MaDiS base on the same author 133D/three-part token contract."""

from pathlib import Path
import json
import hashlib
import torch
from .adapters import SOKEAdapter, import_upstream, _state, _substate, file_hash, LANGUAGES


class MaDiSAdapter(SOKEAdapter):
    @classmethod
    def from_assets(cls, config):
        cfg = (
            json.loads(Path(config).read_text())
            if isinstance(config, (str, Path))
            else dict(config)
        )
        if cfg.get("target_language") not in LANGUAGES:
            raise ValueError("MaDiS per-language checkpoint requires explicit target_language")
        token_cfg = dict(cfg)
        token_cfg.pop("generator_checkpoint", None)
        base = SOKEAdapter.from_assets(token_cfg).cpu()
        generator_state = _state(cfg["generator_checkpoint"])
        # Author evaluation loads the full SFT state AFTER tokenizer initialization.
        # Restore VQ encoders/decoders too, rather than merely comparing codebook buffers.
        for part, prefixes in zip(
            (base.body, base.left, base.right),
            (("vae.", "motion_vae."), ("hand_vae.",), ("rhand_vae.",)),
        ):
            part.load_state_dict(_substate(generator_state, prefixes), strict=True)
        repo = Path(cfg["upstream_repo"]).resolve()
        module = import_upstream(repo, "mGPT.archs.madis")
        lm = module.MaDiSLanguageModel(
            model_path=str(Path(cfg["model_path"]).resolve()),
            stage="lm_instruct",
            motion_codebook_size=base.body.code_num,
            hand_codebook_size=base.left.code_num,
            rhand_codebook_size=base.right.code_num,
            body_codebook=base.body.quantizer.codebook.detach().clone(),
            hand_codebook=base.left.quantizer.codebook.detach().clone(),
            rhand_codebook=base.right.quantizer.codebook.detach().clone(),
            max_length=cfg.get("max_text_tokens", 256),
        )
        lm.load_state_dict(_substate(generator_state, ("lm.",)), strict=True)
        for part, book in zip(
            (base.body, base.left, base.right),
            (
                lm.language_model.body_codebook,
                lm.language_model.lhand_codebook,
                lm.language_model.rhand_codebook,
            ),
        ):
            if not torch.equal(part.quantizer.codebook.cpu(), book.cpu()):
                raise ValueError(
                    "MaDiS checkpoint and motion tokenizer use different codebook values"
                )
        provenance = dict(
            base.provenance,
            adapter="official_MaDiS",
            generator_sha256=file_hash(cfg["generator_checkpoint"]),
            mapping_sha256=file_hash(Path(cfg["model_path"]) / "map_ids.pkl"),
            text_encoder_mode="frozen_Qwen_causal_hidden_states",
            effective_tokenizer_source="full_MaDiS_SFT_state_after_tokenizer_initialization",
        )
        result = (
            cls(
                base.body,
                base.left,
                base.right,
                base.mean,
                base.std,
                lm,
                factor=base.factor,
                provenance=provenance,
            )
            .float()
            .to(cfg.get("device", "cpu"))
        )
        result.target_language = cfg["target_language"]
        return result

    def prepare(self, text, language, sample_id, **kwargs):
        if language != self.target_language:
            raise ValueError("Requested language differs from the MaDiS fine-tuning checkpoint")
        return super().prepare(text, language, sample_id, **kwargs)

    @torch.no_grad()
    def text_features(self, text, language):
        if language not in LANGUAGES:
            raise ValueError("Unknown target sign language")
        ids, mask = self.lm._tokenize([text], labels=False, device=self.mean.device)
        encoded = self.lm.language_model.main_lm(
            input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
        )
        return encoded.last_hidden_state[0, mask[0].bool()]


def load_base_adapter(config, *, device=None):
    cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
    if device is not None:
        cfg["device"] = device
    kind = cfg.pop("base_type", "soke")
    if kind == "soke":
        result = SOKEAdapter.from_assets(cfg)
    elif kind == "madis":
        result = MaDiSAdapter.from_assets(cfg)
    else:
        raise ValueError(f"Unknown frozen generator {kind!r}")
    groups = cfg.get("generator_train_groups")
    if isinstance(groups, str):
        groups = json.loads(Path(groups).read_text())
    if groups is not None and (
        not isinstance(groups, list) or any(not isinstance(g, str) for g in groups)
    ):
        raise ValueError("generator_train_groups must be a list or JSON list path")
    result.generator_train_groups = None if groups is None else frozenset(groups)
    result.generator_train_groups_sha256 = (
        None
        if groups is None
        else hashlib.sha256(json.dumps(sorted(set(groups))).encode()).hexdigest()
    )
    return result

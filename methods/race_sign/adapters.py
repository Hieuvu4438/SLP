"""Frozen SOKE integration using the author's exact VQ/LM implementations.
No local SLP experiments/checkpoints are implicitly discovered or loaded.
"""

from contextlib import contextmanager
from pathlib import Path
import importlib
import json
import os
import sys
import tempfile
import threading
import hashlib
import torch
from .author_checkpoint import load_author_state
from torch import nn

LANGUAGES = {"how2sign": 0, "csl": 1, "phoenix": 2}
_LOAD_LOCK = threading.RLock()


def author_statistics_133(value):
    """Author releases stats in 179D: root/body/hands/jaw/shape/expression.
    Match H2SDataModule exactly: omit first 36 coordinates and shape[159:169].
    """
    value = torch.as_tensor(value).float()
    if value.shape == (179,):
        value = torch.cat((value[36:159], value[169:179]))
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def _empty_keyword_metadata():
    """Upstream constructor unconditionally opens name->keyword split files.
    Supply empty metadata in a temporary directory, so it never reads train/dev/test
    annotations. Retrieval, when requested, is passed explicitly as text-only prompts.
    CWD change is only for synchronous model construction; never use from worker threads.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Construct the upstream adapter on the main thread before workers")
    with _LOAD_LOCK, tempfile.TemporaryDirectory(prefix="race_soke_init_") as tmp:
        scripts = Path(tmp) / "scripts"
        scripts.mkdir()
        for name in (
            "name2kws_train.json",
            "name2kws_val.json",
            "name2kws_test.json",
            "word2code.json",
        ):
            (scripts / name).write_text("{}")
        previous = Path.cwd()
        try:
            os.chdir(tmp)
            yield
        finally:
            os.chdir(previous)


def import_upstream(repo, module):
    repo = Path(repo).resolve()
    if not (repo / "mGPT/archs/mgpt_vq.py").exists():
        raise FileNotFoundError("Expected an official SOKE checkout")
    current = sys.modules.get("mGPT")
    if current is not None and Path(current.__file__).resolve().parent != repo / "mGPT":
        raise RuntimeError("A different mGPT checkout is already imported; use a separate process")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module(module)


def _state(path):
    return load_author_state(path)


def _substate(state, prefixes):
    for prefix in prefixes:
        selected = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if selected:
            return selected
    raise ValueError(f"Checkpoint contains none of the expected prefixes {prefixes}")


class SOKEAdapter(nn.Module):
    def __init__(self, body, left, right, mean, std, lm=None, *, factor=4, provenance=None):
        super().__init__()
        self.body, self.left, self.right, self.lm = body, left, right, lm
        self.factor = factor
        mean, std = author_statistics_133(mean), author_statistics_133(std)
        if (
            mean.shape != (133,)
            or std.shape != (133,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Expected author 133D normalization statistics")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.provenance = provenance or {"adapter": "explicitly_injected_SOKE_components"}
        self.requires_grad_(False)
        self.eval()
        if lm is not None:
            # Never retrieve via sample name or split-specific keyword metadata.
            lm.num_kws_per_sen = 0
            lm.name2kws = {}
            lm.word2code = {}

    def train(self, mode=True):
        return super().train(False)

    @classmethod
    def from_assets(cls, config):
        """Explicit JSON assets; no global experiment config or dataset loader."""
        cfg = (
            json.loads(Path(config).read_text())
            if isinstance(config, (str, Path))
            else dict(config)
        )
        repo = Path(cfg["upstream_repo"]).resolve()
        vq_type = import_upstream(repo, "mGPT.archs.mgpt_vq").VQVae
        args = dict(
            code_dim=512,
            output_emb_width=512,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            norm=None,
            activation="relu",
        )
        overrides = cfg.get("vq_params", {})
        if set(overrides) - set(args):
            raise ValueError("Unknown VQ architecture parameters")
        args.update(overrides)
        factor = args["stride_t"] ** args["down_t"]
        modules = [
            vq_type(nfeats=d, code_num=k, **args).cpu() for d, k in ((43, 96), (45, 192), (45, 192))
        ]
        token_state = _state(cfg["tokenizer_checkpoint"])
        for m, prefixes in zip(modules, [("vae.", "motion_vae."), ("hand_vae.",), ("rhand_vae.",)]):
            m.load_state_dict(_substate(token_state, prefixes), strict=True)
        lm = None
        if cfg.get("generator_checkpoint"):
            module = import_upstream(repo, "mGPT.archs.mgpt_mbart")
            model_path = str(Path(cfg["model_path"]).resolve())
            with _empty_keyword_metadata():
                lm = module.Mbart_Based_MLM(
                    model_path=model_path,
                    model_type=cfg.get("model_type", "mbart_multi"),
                    num_heads=3,
                    motion_codebook_size=96,
                    hand_codebook_size=192,
                    rhand_codebook_size=192,
                    stage="lm_pretrain",
                    max_length=cfg.get("max_text_tokens", 256),
                )
            lm.load_state_dict(
                _substate(_state(cfg["generator_checkpoint"]), ("lm.",)), strict=True
            )
        mean = torch.load(cfg["soke_mean"], map_location="cpu", weights_only=True)
        std = torch.load(cfg["soke_std"], map_location="cpu", weights_only=True)
        provenance = {
            "adapter": "official_SOKE",
            "repo": str(repo),
            "tokenizer_sha256": file_hash(cfg["tokenizer_checkpoint"]),
            "generator_sha256": file_hash(cfg["generator_checkpoint"]) if lm else None,
            "mean_sha256": file_hash(cfg["soke_mean"]),
            "std_sha256": file_hash(cfg["soke_std"]),
            "mapping_sha256": file_hash(Path(cfg["model_path"]) / "map_ids.pkl") if lm else None,
            "upstream_code_sha256": {
                f: file_hash(repo / f)
                for f in (
                    "mGPT/archs/mgpt_vq.py",
                    "mGPT/archs/mgpt_mbart.py",
                    "mGPT/archs/lm_multihead.py",
                    "mGPT/archs/madis.py",
                    "mGPT/archs/masked_diffusion.py",
                    "mGPT/archs/connectors.py",
                )
                if (repo / f).exists()
            },
        }
        return (
            cls(*modules, mean, std, lm, factor=factor, provenance=provenance)
            .float()
            .to(cfg.get("device", "cpu"))
        )

    @torch.no_grad()
    def encode_pose(self, raw_pose):
        pose = (raw_pose.to(self.mean) - self.mean) / self.std
        # Exact author slicing: body VQ contains 30 body + 3 jaw + 10 expression.
        parts = (
            torch.cat((pose[..., :30], pose[..., 120:]), -1),
            pose[..., 30:75],
            pose[..., 75:120],
        )
        return tuple(
            m.encode(x[None])[0][0].long()
            for m, x in zip((self.body, self.left, self.right), parts)
        )

    @torch.no_grad()
    def decode_tokens(self, streams):
        if len(streams) != 3 or any(s.ndim != 1 or len(s) == 0 for s in streams):
            raise ValueError("Three nonempty body/LH/RH token streams required")
        poses, embeddings = [], []
        max_tokens = max(map(len, streams))
        for module, ids in zip((self.body, self.left, self.right), streams):
            ids = ids.to(self.mean.device).long()
            if (ids < 0).any() or (ids >= module.code_num).any():
                raise ValueError("Out-of-codebook IDs; EOS/special IDs must be parsed by author LM")
            pose = module.decode(ids)[0]
            if len(pose) != len(ids) * self.factor:
                raise ValueError("Tokenizer upsampling factor differs from configured contract")
            # Match author max-length assembly: decode each part BEFORE pose replication.
            pad = max_tokens * self.factor - len(pose)
            poses.append(torch.cat((pose, pose[-1:].expand(pad, -1)), 0))
            emb = module.quantizer.dequantize(ids).reshape(len(ids), -1)
            embeddings.append(torch.cat((emb, emb[-1:].expand(max_tokens - len(ids), -1)), 0))
        body, left, right = poses
        normalized = torch.cat((body[:, :30], left, right, body[:, 30:]), -1)
        return normalized * self.std + self.mean, torch.stack(embeddings, 1)

    @torch.no_grad()
    def text_features(self, text, language):
        if self.lm is None:
            raise ValueError("Text encoder/generator assets are required")
        if language not in LANGUAGES:
            raise ValueError("Unknown target sign language")
        encoding = self.lm.tokenizer(
            [text],
            padding="longest",
            truncation=True,
            max_length=self.lm.max_length,
            return_tensors="pt",
            return_attention_mask=True,
            return_length=True,
        )
        ids = encoding.input_ids.to(self.mean.device)
        module = importlib.import_module(self.lm.__class__.__module__)
        module.correct_lang_token(
            self.lm.tokenizer,
            ids,
            encoding.length.to(ids.device),
            [language],
            part=None,
            target=False,
            model_type=self.lm.model_type,
        )
        self.lm.map_ids(ids, direction="token_to_emb")
        mask = encoding.attention_mask.to(ids.device)
        encoder = self.lm.language_model.main_lm.get_encoder()
        return encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state[
            0, mask[0].bool()
        ]

    @torch.no_grad()
    def prepare(
        self,
        text,
        language,
        sample_id,
        *,
        target=None,
        mode="generated",
        max_tokens=100,
        generator_seed=0,
        prompt=None,
    ):
        if language not in LANGUAGES or mode not in ("teacher", "generated"):
            raise ValueError("Invalid language/preparation mode")
        if mode == "teacher":
            if target is None or target.ndim != 2 or target.shape[1] != 133:
                raise ValueError("Teacher preparation requires raw [T,133] training poses")
            n = len(target) // self.factor * self.factor
            if n == 0:
                raise ValueError("Training motion shorter than one token")
            streams = self.encode_pose(target[:n])
        else:
            if target is not None:
                raise ValueError("Generated preparation must never receive reference motion/length")
            if self.lm is None:
                raise ValueError("Generator assets required for text-only inference")
            devices = [self.mean.device] if self.mean.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(generator_seed)
                outputs = self.lm.generate_direct(
                    [text if prompt is None else prompt],
                    src=[language],
                    name=None,
                    max_length=max_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            streams = tuple(
                outputs[k][0]
                for k in ("outputs_tokens", "outputs_tokens_hand", "outputs_tokens_rhand")
            )
        baseline, codes = self.decode_tokens(streams)
        return dict(
            sample_id=sample_id,
            language=LANGUAGES[language],
            baseline=baseline.cpu(),
            codes=codes.cpu(),
            text=self.text_features(text, language).cpu(),
            provenance=dict(
                self.provenance,
                preparation=mode,
                prompt_track="GF" if prompt is None else "explicit_dictionary_condition",
                frame_tail_policy="teacher_trim_to_factor"
                if mode == "teacher"
                else "native_generated",
            ),
        )

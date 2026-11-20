"""SignNMFlow - Naturalness-guided Manifold Flow Matching for SLP (arXiv:2609.33339).

fit(train, val):
    1. cache the training motions; MotionCodec -> sign-aligned quaternions on (S^3)^41
    2. prototypes  : per-joint K-means, K=500 (App. C.2)   [path in naturalness, ng_parallel]
    3. correction  : eps_theta by the naturalness kinetic energy (Alg. 1)
    4. generator   : phi with the target-prediction objective along the frozen path (Alg. 2)
predict(inputs):
    text -> predicted length -> 5-step re-planned sampler (Eq. 53) -> normalised 133-D poses.

Every stage writes to ``ctx.ckpt_dir`` and can be re-used through ``method.init.<stage>``
(path to a checkpoint), e.g. to share prototypes/correction across ablations, to evaluate
the multilingual model on each dataset, or to vary only the sampler (Table 8).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from slp.data import Split, load_dataset
from slp.data.sign_smplx import load_stats
from slp.method import BaseMethod

from .data import MotionTextDataset, all_frames, build_cache
from .flow import FlowSpec, sample
from .model import SignGenerator
from .naturalness import NaturalnessMeasure, extract_prototypes
from .paths import CorrectionNet
from .representation import MotionCodec, load_hand_mean

ROOT = Path(__file__).resolve().parents[2]
CH = 1 << 18


def _abs(p):
    if p is None:
        return None
    p = Path(os.path.expandvars(str(p)))
    return p if p.is_absolute() else ROOT / p


def read_lengths(path, names, cap: int) -> list[int]:
    """Fixed lengths (C1) from a JSON table {name: frames}; keys may also use the
    predictions.npz spelling (``/`` -> ``__``)."""
    import json

    table = json.load(open(path))
    out, missing = [], []
    for n in names:
        v = table.get(n, table.get(n.replace("/", "__")))
        if v is None:
            missing.append(n)
        else:
            out.append(int(v))
    if missing:
        raise KeyError(f"{len(missing)} sequences missing from {path}, e.g. {missing[:3]}")
    if out and (min(out) < 1 or max(out) > cap):
        raise ValueError(f"lengths in {path} must lie in [1, {cap}] (positional table size)")
    return out


class Method(BaseMethod):
    # ------------------------------------------------------------------ setup
    def _setup(self):
        hp = self.hparams
        self.device = torch.device(hp.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        f = hp["flow"]
        self.spec = FlowSpec(f["representation"], f["path"], f["prediction"],
                             float(f["w_pose"]), float(f["w_expr"]), f.get("loss_reduction", "sum"),
                             shift=float(f.get("shift", 1.0)), train_shift=float(f.get("train_shift", 1.0)))
        mean, std = load_stats(str(_abs(self.cfg["data"]["stats"])))
        hand_mean = load_hand_mean(_abs(hp["smplx_dir"])) if hp.get("add_hand_mean", True) else None
        self.codec = MotionCodec(mean, std, hand_mean, self.spec.representation)
        g = hp["generator"]
        self.gen = SignGenerator(rot_dim=self.codec.rot_dim, latent=g["latent"], layers=g["layers"],
                                 heads=g["heads"], ff=g["ff"], time_dim=g["time_dim"],
                                 dropout=g["dropout"], max_frames=g["max_frames"],
                                 text_model=str(_abs(g["text_model"])),
                                 text_max_len=g.get("text_max_len", 128))
        self.corr, self.measure = None, None
        self.log = self.ctx.logger.info

    def _cache(self, split: Split, tag: str):
        return build_cache(split, _abs(self.hparams.get("cache_dir", "outputs/signmflow/cache")), tag,
                           workers=int(self.hparams.get("cache_workers", 8)), log=self.log)

    def _train_caches(self, train: Split, val: Split):
        """Single dataset, or the union of Phoenix/CSL/How2Sign for SignNMFlow-M."""
        name = self.cfg["data"]["name"]   # cache tag = dataset config (protocols differ per config)
        if not self.hparams.get("multilingual", False):
            return [self._cache(train, f"{name}_train")], [self._cache(val, f"{name}_val")]
        tr, va = [], []
        for name in self.hparams["multilingual_datasets"]:
            from slp import config as config_lib
            cfg = config_lib.deep_merge(self.cfg, config_lib.load_yaml(ROOT / "configs" / "data" / f"{name}.yaml"))
            d = load_dataset(cfg)  # official splits; only train/val are read
            tr.append(self._cache(d.train, f"{name}_train"))
            va.append(self._cache(d.val, f"{name}_val"))
        return tr, va

    def _init_path(self, stage):
        p = (self.hparams.get("init") or {}).get(stage)
        return _abs(p) if p else None

    # ------------------------------------------------------------------ fit
    def fit(self, train: Split, val: Split) -> None:
        self._setup()
        hp, dev, ck = self.hparams, self.device, self.ctx.ckpt_dir
        tr_caches, va_caches = self._train_caches(train, val)
        train_ds, val_ds = MotionTextDataset(tr_caches), MotionTextDataset(va_caches)
        if hp.get("debug_limit"):  # smoke tests only: first N sequences of each split
            n = int(hp["debug_limit"])
            train_ds.items, val_ds.items = train_ds.items[:n], val_ds.items[:n]
        self.log(f"SignNMFlow {self.spec} train={len(train_ds)} val={len(val_ds)}")

        if self.spec.needs_correction:  # naturalness / ng_parallel: stages 1-2
            m = hp["measure"]
            p_init = self._init_path("prototypes")
            if p_init is not None:
                st = torch.load(p_init, map_location="cpu", weights_only=False)
                protos, weights = st["protos"], st["weights"]
                self.log(f"prototypes loaded from {p_init}")
            else:
                frames = torch.from_numpy(all_frames(tr_caches))
                quats = torch.cat([self.codec.encode(frames[a:a + CH].to(dev))[0].cpu()
                                   for a in range(0, len(frames), CH)])
                if m.get("max_frames"):
                    g = torch.Generator().manual_seed(self.ctx.seed)
                    quats = quats[torch.randperm(len(quats), generator=g)[:int(m["max_frames"])]]
                self.log(f"K-means on {len(quats)} frames, K={m['K']}")
                protos, weights = extract_prototypes(quats, int(m["K"]), iters=int(m["kmeans_iters"]),
                                                     seed=self.ctx.seed, device=str(dev), log=self.log,
                                                     tol=float(m.get("kmeans_tol", 1e-4)))
                del quats, frames
            torch.save({"protos": protos, "weights": weights, "K": int(m["K"])}, ck / "prototypes.pt")
            self.measure = NaturalnessMeasure(protos, weights, float(m["sigma"]), float(m["rho"])).to(dev)

            c = hp["correction"]
            self.corr = CorrectionNet(hidden=c["hidden"], layers=c["layers"]).to(dev)
            c_init = self._init_path("correction")
            if c_init is not None:
                self.corr.load_state_dict(torch.load(c_init, map_location="cpu", weights_only=False)["model"])
                self.log(f"correction loaded from {c_init}")
            else:
                from .training import geodesic_energy_gap, train_correction
                rf = c.get("resume_from")
                if rf and not (ck / "correction.pt").exists():  # continue an interrupted run (same config)
                    import shutil
                    shutil.copy(_abs(rf), ck / "correction.pt")
                    self.log(f"correction: resuming from {rf}")
                train_correction(self.corr, self.measure, train_ds, self.codec, c, dev, log=self.log,
                                 ckpt=ck / "correction.pt", metrics=self.ctx.log_metrics)
                gap = geodesic_energy_gap(self.corr, self.measure, val_ds, self.codec, dev)
                self.log(f"naturalness energy on val: geodesic={gap['geodesic']:.5g} "
                         f"learned={gap['naturalness']:.5g}")
                self.ctx.log_metrics({f"val_energy_{k}": v for k, v in gap.items()})
            torch.save({"model": self.corr.state_dict()}, ck / "correction_final.pt")
            self.corr.eval().requires_grad_(False)

        g_init = self._init_path("generator")
        if g_init is not None:
            self.gen.load_state_dict(torch.load(g_init, map_location="cpu", weights_only=False)["model"])
            self.log(f"generator loaded from {g_init}")
            self.gen.to(dev).eval()
        else:
            from .training import train_generator
            rf = hp["generator"].get("resume_from")
            if rf and not (ck / "generator.pt").exists():  # continue an interrupted run (same config)
                import shutil
                shutil.copy(_abs(rf), ck / "generator.pt")
                self.log(f"generator: resuming from {rf}")
            train_generator(self.gen, self.corr, self.spec, train_ds, self.codec, hp["generator"], dev,
                            log=self.log, ckpt=ck / "generator.pt", metrics=self.ctx.log_metrics,
                            val_dataset=val_ds)
        n = sum(p.numel() for p in self.gen.parameters())
        self.log(f"generator parameters: {n / 1e6:.2f}M")

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, inputs):
        hp, dev = self.hparams, self.device
        self.gen.to(dev).eval()
        s = hp["sampler"]
        g = torch.Generator(device=dev).manual_seed(int(self.ctx.seed))
        bs = int(s.get("batch_size", 64))
        lo, hi = int(s.get("min_len", 40)), int(s.get("max_len", 400))
        texts = [x["text"] for x in inputs]
        srcs = [x["src"] for x in inputs]
        out = [None] * len(inputs)
        # lengths: fixed per sequence from sampler.lengths_file (C1, same lengths for every arm), or
        # predicted by the length head; then batch by length (less padding)
        if s.get("lengths_file"):
            lengths = self._fixed_lengths(inputs)
        else:
            lengths = []
            for a in range(0, len(inputs), bs):
                text, pad = self.gen.encode_text(texts[a:a + bs], srcs[a:a + bs], dev)
                ll = self.gen.predict_log_length(text, pad).exp().round().clamp(lo, hi).long()
                lengths += ll.tolist()
        order = np.argsort(lengths)
        for a in range(0, len(order), bs):
            idx = order[a:a + bs].tolist()
            text, pad = self.gen.encode_text([texts[i] for i in idx], [srcs[i] for i in idx], dev)
            L = [lengths[i] for i in idx]
            rot, expr, mask = sample(self.gen, self.corr, self.spec, text, pad, L,
                                     steps=int(s["steps"]), generator=g,
                                     grid_power=float(s.get("grid_power", 1.0)),
                                     grid_shift=float(s.get("shift", 1.0)))
            feats = self.codec.decode(rot, expr).float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = feats[j, :L[j]]
        exp = hp.get("export_npz")
        if exp:
            p = _abs(exp)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **{x["name"].replace("/", "__"): o for x, o in zip(inputs, out)})
        np.savez_compressed(self.ctx.run_dir / "predictions.npz",
                            **{x["name"].replace("/", "__"): o for x, o in zip(inputs, out)})
        return out

    def _fixed_lengths(self, inputs) -> list[int]:
        path = _abs(self.hparams["sampler"]["lengths_file"])
        out = read_lengths(path, [x["name"] for x in inputs], int(self.hparams["generator"]["max_frames"]))
        self.log(f"fixed lengths from {path} (mean {np.mean(out):.1f} frames)")
        return out

    def save(self) -> None:
        pass  # every stage already checkpoints into ctx.ckpt_dir

    def load(self, ckpt_dir) -> None:
        self._setup()
        ck = Path(ckpt_dir)
        if (ck / "prototypes.pt").exists() and self.spec.needs_correction:
            st = torch.load(ck / "prototypes.pt", map_location="cpu", weights_only=False)
            m = self.hparams["measure"]
            self.measure = NaturalnessMeasure(st["protos"], st["weights"], m["sigma"], m["rho"]).to(self.device)
            c = self.hparams["correction"]
            self.corr = CorrectionNet(hidden=c["hidden"], layers=c["layers"]).to(self.device)
            self.corr.load_state_dict(torch.load(ck / "correction_final.pt", map_location="cpu",
                                                 weights_only=False)["model"])
            self.corr.eval()
        self.gen.load_state_dict(torch.load(ck / "generator.pt", map_location="cpu", weights_only=False)["model"])
        self.gen.to(self.device).eval()

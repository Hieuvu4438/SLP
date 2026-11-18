"""PMA-Flow - Posterior-Mean-Anchored residual flow on (S^3)^41 for text-to-sign (proposal 5).

pipeline (method.pipeline):
    anchor_flow  (default) Stage A anchor mu_psi(c, T) -> cross-fitted training anchors ->
                 Stage B residual geodesic flow from Exp_mu(sigma xi) -> y_alpha = Exp_mu(alpha Log_mu x)
    anchor_only  Stage A only: the regression to the conditional mean (alpha = 0 by construction)
    uniform_flow Stage B with the uniform-hemisphere source: matched SignNMFlow-GE baseline with
                 the same losses / hygiene (no Stage A)

fit(train, val) uses ``val`` only for checkpoint selection (frame-aligned point error proxy,
never the reported metric) and, if cross-fitting is off, for the auto source scale.
predict(inputs): text -> predicted length -> anchor -> flow sample -> alpha blend -> 133-D.

Re-use (``method.init``): ``anchor_dir`` = checkpoints dir of an earlier run with the same Stage A
config (re-uses anchor_net.pt and its anchor caches, e.g. for sigma / CFG sweeps of Stage B);
``flow`` = a flow_net.pt (evaluate / re-sample only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from slp.data import Split
from slp.data.sign_smplx import load_stats
from slp.method import BaseMethod
from slp.skeleton_fk import SkeletonFK

from .data import MotionTextDataset, build_cache, collate
from .flow import (FlowSpec, anchor_loss, flow_loss, interpolate, predict_anchor, residual_scales,
                   sample)
from .losses import LossWeights, point_error
from .model import PoseNet
from .representation import FEAT_DIM, MotionCodec, flatten
from .training import train_network

ROOT = Path(__file__).resolve().parents[2]


def _abs(p):
    if p is None:
        return None
    p = Path(os.path.expandvars(str(p)))
    return p if p.is_absolute() else ROOT / p


class Method(BaseMethod):
    # ------------------------------------------------------------------ setup
    def _setup(self):
        hp = self.hparams
        self.pipeline = hp.get("pipeline", "anchor_flow")
        assert self.pipeline in ("anchor_flow", "anchor_only", "uniform_flow"), self.pipeline
        dev = hp.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        stats = str(_abs(self.cfg["data"]["stats"]))
        self.fk = SkeletonFK(model_dir=_abs(hp.get("smplx_dir")), stats=stats).to(self.device)
        mean, std = load_stats(stats)
        self.codec = MotionCodec(mean, std, self.fk.hand_mean.cpu().numpy())
        f = hp["flow"]
        self.spec = FlowSpec(source="uniform" if self.pipeline == "uniform_flow" else "anchor",
                             align_pairs=bool(f.get("align_pairs", True)),
                             t_sampling=f.get("t_sampling", "logit_normal"),
                             t_mean=float(f.get("t_mean", 0.0)), t_std=float(f.get("t_std", 1.0)),
                             t_uniform_mix=float(f.get("t_uniform_mix", 0.2)),
                             cfg_drop=float(f.get("cfg_drop", 0.0)))
        self.anchor_net = self.flow_net = None
        self.sigma = self.expr_sigma = None
        self.log = self.ctx.logger.info

    def _weights(self, stage: str) -> LossWeights:
        return LossWeights(**{k: float(v) for k, v in self.hparams[stage]["loss"].items()})

    def _net(self, mode: str) -> PoseNet:
        a = self.hparams["arch"]
        cond = bool(self.hparams["flow"].get("cond_anchor", True)) and self.pipeline == "anchor_flow"
        return PoseNet(mode=mode, cond_anchor=cond,
                       latent=a["latent"], layers=a["layers"], heads=a["heads"], ff=a["ff"],
                       time_dim=a["time_dim"], dropout=a["dropout"], max_frames=a["max_frames"],
                       text_model=str(_abs(a["text_model"])), text_max_len=a.get("text_max_len", 128))

    def _train_hp(self, stage: str) -> dict:
        return {**self.hparams["train"], **(self.hparams[stage].get("train") or {})}

    # ------------------------------------------------------------------ validation proxies
    def _val_items(self, ds: MotionTextDataset) -> MotionTextDataset:
        n = int(self.hparams["train"].get("val_items", 256))
        return ds.subset(range(min(n, len(ds))))

    @torch.no_grad()
    def _point_error(self, ds, predict_fn) -> dict:
        from torch.utils.data import DataLoader

        tot, n = {"body": 0.0, "hand": 0.0}, 0
        for feats, mask, texts, srcs, anc in DataLoader(ds, batch_size=32, collate_fn=collate):
            feats, mask = feats.to(self.device), mask.to(self.device)
            q1, _ = self.codec.encode(feats)
            q = predict_fn(texts, srcs, mask, None if anc is None else anc.to(self.device))
            e = point_error(self.fk, q, q1, mask)
            k = int(mask.sum())
            tot = {key: tot[key] + e[key] * k for key in tot}
            n += k
        out = {k: v / max(1, n) for k, v in tot.items()}
        out["score"] = out["body"] + out["hand"]
        return out

    def _val_anchor(self, net, ds):
        def fn(texts, srcs, mask, anc):
            text, pad = net.encode_text(texts, srcs, self.device)
            return predict_anchor(net, text, pad, mask.sum(1).tolist())[0]
        return self._point_error(ds, fn)

    def _val_flow(self, net, ds):
        s = self.hparams["sampler"]
        g = torch.Generator(device=self.device).manual_seed(1234)

        def fn(texts, srcs, mask, anc):
            text, pad = net.encode_text(texts, srcs, self.device)
            return sample(net, self.spec, text, pad, mask.sum(1).tolist(), steps=int(s["steps"]),
                          sampler=s.get("kind", "euler"), anchor=anc, sigma=self.sigma,
                          expr_sigma=self.expr_sigma, generator=g)[0]
        return self._point_error(ds, fn)

    # ------------------------------------------------------------------ Stage A
    def _train_anchor(self, train_ds, val_ds, tag) -> PoseNet:
        net = self._net("anchor")
        w = self._weights("anchor")
        vi = self._val_items(val_ds)

        def step_loss(net, feats, mask, text, text_pad, anc, denom):
            q1, e1 = self.codec.encode(feats)
            return anchor_loss(net, w, self.fk, q1, e1, mask, text, text_pad, denom)

        train_network(net, step_loss, train_ds, self._train_hp("anchor"), self.device, log=self.log,
                      ckpt=self.ctx.ckpt_dir / f"{tag}.pt", metrics=self.ctx.log_metrics,
                      validate=lambda n: self._val_anchor(n, vi), tag=tag)
        return net

    @torch.no_grad()
    def _anchor_cache(self, net, ds: MotionTextDataset, out: np.ndarray, bs: int = 64):
        """Write mu (GT lengths) of every item of ``ds`` into ``out`` (frame layout of the cache)."""
        net.to(self.device).eval()
        order = np.argsort([ds.span(i)[1] - ds.span(i)[0] for i in range(len(ds))])
        for a in range(0, len(order), bs):
            idx = order[a:a + bs].tolist()
            items = [ds.cache["inputs"][ds.indices[i]] for i in idx]
            spans = [ds.span(i) for i in idx]
            text, pad = net.encode_text([x["text"] for x in items], [x["src"] for x in items], self.device)
            q, e, _ = predict_anchor(net, text, pad, [b - s for s, b in spans])
            fl = flatten(q, e).float().cpu().numpy()
            for j, (s, b) in enumerate(spans):
                out[s:b] = fl[j, :b - s]

    def _fit_stage_a(self, train_ds, val_ds):
        ck, hp = self.ctx.ckpt_dir, self.hparams["anchor"]
        src = _abs((self.hparams.get("init") or {}).get("anchor_dir"))
        if src is not None and src.resolve() != ck.resolve():
            for pat in ("anchor_net.pt", "anchor_fold*_net.pt", "anchors_*.npy"):
                for f in src.glob(pat):
                    if not (ck / f.name).exists():
                        os.symlink(f.resolve(), ck / f.name)
            self.log(f"Stage A re-used from {src}")
        self.anchor_net = self._net("anchor")
        if (ck / "anchor_net.pt").exists():
            self.anchor_net.load_state_dict(torch.load(ck / "anchor_net.pt", map_location="cpu",
                                                       weights_only=False)["model"])
            self.anchor_net.to(self.device).eval()
        else:
            self.anchor_net = self._train_anchor(train_ds, val_ds, "anchor")
            torch.save({"model": self.anchor_net.state_dict()}, ck / "anchor_net.pt")
        if self.pipeline != "anchor_flow":
            return None, None
        # ---- training anchors (cross-fitted: each item predicted by a model that never saw it)
        K = int(hp.get("cross_fit_folds", 2))
        n_tr = int(train_ds.cache["offsets"][-1])
        tr_path = ck / f"anchors_train_{train_ds.cache['key']}_k{K}_n{len(train_ds)}.npy"
        if not tr_path.exists():
            arr = np.lib.format.open_memmap(tr_path.with_suffix(".tmp.npy"), "w+", np.float16,
                                            (n_tr, FEAT_DIM))
            if K >= 2:
                perm = np.random.RandomState(self.ctx.seed).permutation(len(train_ds))
                for k, held in enumerate(np.array_split(perm, K)):
                    rest = np.setdiff1d(perm, held)
                    fold_ck = ck / f"anchor_fold{k}_net.pt"
                    net = self._net("anchor")
                    if fold_ck.exists():
                        net.load_state_dict(torch.load(fold_ck, map_location="cpu", weights_only=False)["model"])
                    else:
                        net = self._train_anchor(train_ds.subset(rest), val_ds, f"anchor_fold{k}")
                        torch.save({"model": net.state_dict()}, fold_ck)
                    self._anchor_cache(net, train_ds.subset(held), arr)
                    del net
                    torch.cuda.empty_cache()
            else:
                self.log("WARNING: cross_fit_folds < 2 - training anchors are in-sample predictions")
                self._anchor_cache(self.anchor_net, train_ds, arr)
            arr.flush()
            del arr
            os.replace(tr_path.with_suffix(".tmp.npy"), tr_path)
        va_path = ck / f"anchors_val_{val_ds.cache['key']}_n{len(val_ds)}.npy"
        if not va_path.exists():
            arr = np.lib.format.open_memmap(va_path.with_suffix(".tmp.npy"), "w+", np.float16,
                                            (int(val_ds.cache["offsets"][-1]), FEAT_DIM))
            self._anchor_cache(self.anchor_net, val_ds, arr)
            arr.flush()
            del arr
            os.replace(va_path.with_suffix(".tmp.npy"), va_path)
        return np.load(tr_path, mmap_mode="r"), np.load(va_path, mmap_mode="r")

    def _calibrate_sigma(self, train_ds, val_ds):
        f = self.hparams["flow"]
        K = int(self.hparams["anchor"].get("cross_fit_folds", 2))
        ref = train_ds if K >= 2 else val_ds   # in-sample train residuals would be far too small
        sig, esig = residual_scales(ref, True, self.codec, self.device)
        scale = float(f.get("sigma_scale", 1.0))
        if f.get("sigma", "auto") == "auto":
            self.sigma = sig.to(self.device) * scale
        else:
            self.sigma = torch.full((41, 1), float(f["sigma"]), device=self.device)
        if f.get("expr_sigma", "auto") == "auto":
            self.expr_sigma = esig.to(self.device) * scale
        else:
            self.expr_sigma = torch.full((10,), float(f["expr_sigma"]), device=self.device)
        self.log(f"residual scale ({'cross-fit train' if K >= 2 else 'val'}): sigma mean "
                 f"{float(sig.mean()):.4f} rad [min {float(sig.min()):.4f}, max {float(sig.max()):.4f}], "
                 f"expr {float(esig.mean()):.4f}; used sigma mean {float(self.sigma.mean()):.4f}")

    # ------------------------------------------------------------------ fit
    def fit(self, train: Split, val: Split) -> None:
        self._setup()
        hp, ck = self.hparams, self.ctx.ckpt_dir
        name = self.cfg["data"]["name"]
        cache_dir = _abs(hp.get("cache_dir", "outputs/pma_flow/cache"))
        workers = int(hp.get("cache_workers", 8))
        tr_c = build_cache(train, cache_dir, f"{name}_train", workers, self.log)
        va_c = build_cache(val, cache_dir, f"{name}_val", workers, self.log)
        train_ds, val_ds = MotionTextDataset(tr_c), MotionTextDataset(va_c)
        if hp.get("debug_limit"):  # smoke tests only
            n = int(hp["debug_limit"])
            train_ds, val_ds = train_ds.subset(range(min(n, len(train_ds)))), val_ds.subset(range(min(n, len(val_ds))))
        self.log(f"PMA-Flow pipeline={self.pipeline} {self.spec} train={len(train_ds)} val={len(val_ds)}")

        if self.pipeline != "uniform_flow":
            tr_a, va_a = self._fit_stage_a(train_ds, val_ds)
            if self.pipeline == "anchor_flow":
                train_ds, val_ds = train_ds.with_anchors(tr_a), val_ds.with_anchors(va_a)
                self._calibrate_sigma(train_ds, val_ds)
        if self.pipeline != "anchor_only":
            self.flow_net = self._net("flow")
            init = _abs((hp.get("init") or {}).get("flow"))
            if init is not None:
                self.flow_net.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model"])
                self.log(f"flow net loaded from {init}")
            else:
                w = self._weights("flow")
                vi = self._val_items(val_ds)

                def step_loss(net, feats, mask, text, text_pad, anc, denom):
                    q1, e1 = self.codec.encode(feats)
                    return flow_loss(net, self.spec, w, self.fk, q1, e1, mask, text, text_pad,
                                     anchor=anc, sigma=self.sigma, expr_sigma=self.expr_sigma, denom=denom)

                train_network(self.flow_net, step_loss, train_ds, self._train_hp("flow"), self.device,
                              log=self.log, ckpt=ck / "flow.pt", metrics=self.ctx.log_metrics,
                              validate=lambda n: self._val_flow(n, vi), tag="flow")
            torch.save({"model": self.flow_net.state_dict()}, ck / "flow_net.pt")
        self._save_meta()
        for n, net in (("anchor", self.anchor_net), ("flow", self.flow_net)):
            if net is not None:
                self.log(f"{n} net parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    def _save_meta(self):
        meta = {"pipeline": self.pipeline,
                "sigma": None if self.sigma is None else self.sigma.flatten().tolist(),
                "expr_sigma": None if self.expr_sigma is None else self.expr_sigma.tolist()}
        json.dump(meta, open(self.ctx.ckpt_dir / "pma_meta.json", "w"), indent=1)

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, inputs):
        hp, dev = self.hparams, self.device
        s = hp["sampler"]
        g = torch.Generator(device=dev).manual_seed(int(self.ctx.seed))
        bs = int(s.get("batch_size", 64))
        lo, hi = int(s.get("min_len", 40)), int(s.get("max_len", 400))
        texts, srcs = [x["text"] for x in inputs], [x["src"] for x in inputs]
        len_net = self.anchor_net if self.anchor_net is not None else self.flow_net
        lengths = []
        for a in range(0, len(inputs), bs):
            text, pad = len_net.encode_text(texts[a:a + bs], srcs[a:a + bs], dev)
            lengths += len_net.predict_log_length(text, pad).float().exp().round().clamp(lo, hi).long().tolist()
        alpha = float(s.get("alpha", 1.0))
        extra = [float(a) for a in (s.get("export_alphas") or [])] if self.pipeline == "anchor_flow" else []
        outs = {a: [None] * len(inputs) for a in [alpha] + [a for a in extra if a != alpha]}
        order = np.argsort(lengths)
        for a in range(0, len(order), bs):
            idx = order[a:a + bs].tolist()
            bt, bsrc = [texts[i] for i in idx], [srcs[i] for i in idx]
            L = [lengths[i] for i in idx]
            anchor = None
            if self.anchor_net is not None:
                text, pad = self.anchor_net.encode_text(bt, bsrc, dev)
                mq, me, _ = predict_anchor(self.anchor_net, text, pad, L)
                anchor = flatten(mq, me)
            if self.pipeline == "anchor_only":
                res = {a_: (mq, me) for a_ in outs}
            else:
                text, pad = self.flow_net.encode_text(bt, bsrc, dev)
                q, e, _ = sample(self.flow_net, self.spec, text, pad, L, steps=int(s["steps"]),
                                 sampler=s.get("kind", "euler"),
                                 anchor=anchor if self.spec.source == "anchor" else None,
                                 sigma=self.sigma, expr_sigma=self.expr_sigma,
                                 cfg_w=float(s.get("cfg_w", 1.0)),
                                 cfg_interval=tuple(s.get("cfg_interval", (0.0, 1.0))), generator=g)
                res = {a_: ((q, e) if a_ == 1.0 or anchor is None else interpolate(anchor, q, e, a_))
                       for a_ in outs}
            for a_, (q, e) in res.items():
                feats = self.codec.decode(q, e).float().cpu().numpy()
                for j, i in enumerate(idx):
                    outs[a_][i] = feats[j, :L[j]]
        names = [x["name"].replace("/", "__") for x in inputs]
        for a_, o in outs.items():
            suffix = "" if a_ == alpha else f"_alpha{a_:g}"
            np.savez_compressed(self.ctx.run_dir / f"predictions{suffix}.npz", **dict(zip(names, o)))
        return outs[alpha]

    def save(self) -> None:
        pass  # every stage checkpoints into ctx.ckpt_dir

    def load(self, ckpt_dir) -> None:
        self._setup()
        ck = Path(ckpt_dir)
        meta = json.load(open(ck / "pma_meta.json"))
        if meta.get("sigma") is not None:
            self.sigma = torch.tensor(meta["sigma"], device=self.device)[:, None]
            self.expr_sigma = torch.tensor(meta["expr_sigma"], device=self.device)
        for name, mode, attr in (("anchor_net.pt", "anchor", "anchor_net"), ("flow_net.pt", "flow", "flow_net")):
            if (ck / name).exists():
                net = self._net(mode)
                net.load_state_dict(torch.load(ck / name, map_location="cpu", weights_only=False)["model"])
                setattr(self, attr, net.to(self.device).eval())


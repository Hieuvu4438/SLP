"""SignNMFlow experiments with quotient paths and kinematic supervision."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from slp.data.sign_smplx import load_stats
from slp.method import BaseMethod
from slp.paths import PROJECT_ROOT

from .checkpoints import fingerprint, read_checkpoint, save_atomic, write_json
from .data import build_cache
from .energy import DensityEnergy, KinematicEnergy, group_weights
from .flow import FlowSpec, per_clip_sources, sample, sample_source, sampling_grid
from .kinematics import SigningKinematics
from .model import SignGenerator
from .naturalness import NaturalnessMeasure, extract_prototypes
from .paths import CorrectionNet, path_point
from .representation import EXPR_DIM, N_JOINTS, MotionCodec, load_hand_mean, neutral_asset
from .training import correction_diagnostics, rng, train_correction, train_generator


def absolute(path):
    if path is None:
        return None
    return (PROJECT_ROOT / Path(os.path.expandvars(os.path.expanduser(str(path))))).resolve()


class Method(BaseMethod):
    def _setup(self):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
                torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1):
            raise RuntimeError("Use the single-process runner; configure effective_batch for accumulation")
        hp = self.hparams
        self.device = torch.device(hp["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Set method.device=cpu for CPU experiments.")
        self.spec = FlowSpec(**hp["flow"])
        self.correction = self.fk = self.metric_weights = None
        self.sources = {}
        mode = hp["measure"]["kind"]
        required = {
            "naturalness": {"legacy_ng"},
            "so3_corrected": {"fk_full", "fk_diagonal", "fk_joint_block", "density"},
        }
        if self.spec.path in required and mode not in required[self.spec.path]:
            raise ValueError(f"{self.spec.path} requires one of {sorted(required[self.spec.path])}")
        self.corrected = self.spec.path in required
        if self.corrected and hp["correction"]["max_angle"] is not None and self.spec.path != "so3_corrected":
            raise ValueError("correction.max_angle applies to SO(3) correction only")
        stats = absolute(self.cfg["data"]["stats"])
        self.smplx_dir = absolute(hp["smplx_dir"])
        if self.spec.uses_fk or hp["measure"]["kind"].startswith("fk_"):
            from slp.evaluation.smplx_fk import smplx_dir, stats_dir

            # The shared evaluator resolves its assets independently of the method.
            if neutral_asset(smplx_dir()).resolve() != neutral_asset(self.smplx_dir).resolve() or stats_dir().resolve() != stats:
                self.ctx.logger.warning("Method and shared evaluator resolve different SMPL-X assets or stats.")
        asset = neutral_asset(self.smplx_dir)
        self.asset_identity = {"shape": fingerprint(asset), "mean": fingerprint(stats / "mean.pt"),
                               "std": fingerprint(stats / "std.pt"), "hand_mean": hp["add_hand_mean"]}
        mean, std = load_stats(str(stats))
        if not np.isfinite(mean).all() or not (np.isfinite(std).all() and (std > 0).all()):
            raise ValueError("Invalid SMPL-X normalization statistics")
        hand_mean = load_hand_mean(self.smplx_dir) if hp["add_hand_mean"] else None
        self.codec = MotionCodec(mean, std, hand_mean, self.spec.representation)
        g = hp["generator"]
        self.model_signature = {name: g[name] for name in
                                ("latent", "layers", "heads", "ff", "time_dim", "dropout", "max_frames",
                                 "text_max_len", "length_mode", "length_quantiles", "coarse_stride")}
        text_dir = absolute(g["text_model"])
        if text_dir is None:
            raise ValueError("generator.text_model is required")
        text_signature = {name: fingerprint(text_dir / name) for name in
                          ("config.json", "sentencepiece.bpe.model", "map_ids.pkl", "tokenizer.json")
                          if (text_dir / name).is_file()}
        self.generator_signature = {"stage": "generator", "assets": self.asset_identity,
                                    "dataset": self.cfg["data"]["name"], "architecture": self.model_signature,
                                    "tokenizer": text_signature,
                                    "transport": {name: getattr(self.spec, name) for name in (
                                        "representation", "path", "prediction", "input_rotation",
                                        "endpoint_loss", "source_ar", "coupling")}}
        self.gen = SignGenerator(rot_dim=self.spec.rot_dim,
                                 input_rot_dim=6 if self.spec.input_rotation == "6d" else self.spec.rot_dim,
                                 text_model=str(text_dir), **self.model_signature)
        self.grid = sampling_grid(hp["sampler"]["steps"], hp["sampler"]["time_grid"])
        s = hp["sampler"]
        if not 0 < s["min_len"] <= s["max_len"] <= self.gen.max_frames or s["batch_size"] < 1:
            raise ValueError("Invalid sampler batch size or length bounds")
        if s["noise"] not in {"clip", "legacy"} or s["expression_update"] not in {"current", "anchored"}:
            raise ValueError("Unknown sampler noise or expression policy")
        if s["noise"] == "legacy" and self.spec.source_ar:
            raise ValueError("Legacy inference uses the original independent source")
        if s["expression_update"] == "anchored" and self.spec.prediction != "target":
            raise ValueError("Anchored expression requires endpoint prediction")
        self.manifest = {"assets": self.asset_identity, "generator_signature": self.generator_signature,
                         "sources": self.sources, "seed": self.ctx.seed}

    def _data(self, split):
        hp = self.hparams
        return build_cache(split, absolute(hp["cache_dir"]), hp["fps"],
                           limit=hp["debug_limit"], workers=int(hp["cache_workers"]))

    def _get_fk(self):
        if self.fk is None:
            self.fk = SigningKinematics.from_smplx(self.smplx_dir).to(self.device)
        return self.fk

    def _make_correction(self):
        c = self.hparams["correction"]
        return CorrectionNet(hidden=c["hidden"], layers=c["layers"], quotient=self.spec.path == "so3_corrected",
                             max_angle=c["max_angle"]).to(self.device)

    def _correction_signature(self):
        c = self.hparams["correction"]
        return {"stage": "correction", "assets": self.asset_identity, "dataset": self.cfg["data"]["name"],
                "path": self.spec.path, "coupling": self.spec.coupling, "architecture": {k: c[k] for k in ("hidden", "layers", "max_angle")},
                "measure": self.hparams["measure"]}

    def _legacy_ok(self, component):
        if not self.hparams["init"]["legacy_signmflow"]:
            return False
        basic = self.spec.representation == "quat" and self.spec.path == "naturalness"
        if component == "correction":
            return self.spec.coupling == "independent" and basic and self.hparams["measure"]["kind"] == "legacy_ng"
        return (basic and self.spec.prediction == "target" and self.spec.input_rotation == "raw"
                and self.spec.endpoint_loss == "lift" and not self.spec.source_ar
                and self.spec.coupling == "independent" and self.gen.coarse is None
                and self.gen.length_mode == "baseline")

    def _validate_legacy_source(self, file, state):
        if state.get("signature") is not None:
            return
        metadata = file.parent.parent / "meta.json"
        if not metadata.is_file():
            self.ctx.logger.warning(f"Legacy checkpoint has no adjacent run metadata: {file}")
            return
        run = json.loads(metadata.read_text())
        if run.get("method") != "signmflow" or run.get("data") != self.cfg["data"]["name"]:
            raise ValueError(f"Legacy checkpoint run does not match the expected method and dataset: {file}")

    def _record_source(self, key, file):
        self.sources[key] = {"path": str(file), "sha256": fingerprint(file)}

    def _load_correction(self, file):
        state = read_checkpoint(file, self._correction_signature(), self._legacy_ok("correction"))
        self._validate_legacy_source(file, state)
        self.correction = self._make_correction()
        self.correction.load_state_dict(state["model"])
        self.correction.eval().requires_grad_(False)
        self._record_source("correction", file)

    def _fit_energy(self, data, prototypes_path):
        m = self.hparams["measure"]
        if m["kind"].startswith("fk_"):
            return KinematicEnergy(self._get_fk(), m["kind"], m["fk_groups"], m["fk_base"],
                                   m["fk_strength"], m["jacobian_chunk"]).to(self.device)
        quotient = m["kind"] == "density"
        signature = {"stage": "prototypes", "assets": self.asset_identity, "dataset": self.cfg["data"]["name"],
                     "data": data.key, "quotient": quotient, "K": m["K"], "max_frames": m["max_frames"],
                     "iterations": m["kmeans_iters"], "tolerance": m["kmeans_tol"]}
        if prototypes_path:
            state = read_checkpoint(prototypes_path, signature, self._legacy_ok("correction") and not quotient)
            self._record_source("prototypes", prototypes_path)
            centroids, weights = state["protos"], state["weights"]
        else:
            raw = data.frames
            count = len(raw) if m["max_frames"] is None else min(len(raw), int(m["max_frames"]))
            if count < 1:
                raise ValueError("Expected positive measure.max_frames")
            idx = np.arange(len(raw)) if count == len(raw) else np.sort(
                np.random.default_rng(self.ctx.seed).choice(len(raw), size=count, replace=False))
            quats = []
            for a in range(0, len(idx), 1 << 15):
                f = torch.from_numpy(raw[idx[a:a + (1 << 15)], :133].copy()).to(self.device)
                quats.append(self.codec.encode(f)[0].cpu())
            centroids, weights = extract_prototypes(torch.cat(quats), m["K"], iters=m["kmeans_iters"],
                                                    seed=self.ctx.seed, device=str(self.device),
                                                    log=self.ctx.logger.info, tol=m["kmeans_tol"],
                                                    quotient=quotient)
            save_atomic({"protos": centroids, "weights": weights, "signature": signature},
                        self.ctx.ckpt_dir / "prototypes.pt")
        measure = NaturalnessMeasure(centroids, weights, m["density_sigma_rad"] if quotient else m["sigma"],
                                     m["rho"], quotient=quotient).to(self.device)
        if quotient:
            noise = rng(self.ctx.seed + 724, self.device)
            n = min(1024, len(data.frames))
            idx = np.random.default_rng(self.ctx.seed + 724).choice(len(data.frames), size=n, replace=False)
            frames = torch.from_numpy(data.frames[idx, :133].copy()).to(self.device)
            target = self.codec.encode(frames)[0]
            source = sample_source(self.spec, (n, 1, N_JOINTS), self.device, noise)[:, 0]
            t = torch.rand(n, 1, 1, device=self.device, generator=noise)
            logs = [measure.log_mass(path_point("so3", a, b, t_part))
                    for a, b, t_part in zip(source.split(32), target.split(32), t.split(32))]
            median = torch.cat(logs).median(0).values
            save_atomic({"median_log_mass": median.cpu()}, self.ctx.ckpt_dir / "density_calibration.pt")
            return DensityEnergy(measure, median, m["density_bounds"], m["density_power"]).to(self.device)

        return lambda q, dq: measure.quad_form(q, dq).sum(-1)

    def _load_generator(self, file, warm=False):
        self.generator_signature["correction_sha256"] = self.sources.get("correction", {}).get("sha256")
        state = read_checkpoint(file, self.generator_signature, self._legacy_ok("generator"))
        self._validate_legacy_source(file, state)
        self.gen.load_state_dict(state["model"])
        self._record_source("warm_start" if warm else "generator", file)

    def fit(self, train, val):
        self._setup()
        hp = self.hparams
        initial = hp["init"]
        g = hp["generator"]
        c = hp["correction"]
        frozen = absolute(initial["generator"])
        warm = absolute(initial["warm_start"])
        resumed = absolute(g["resume_from"])
        if sum(p is not None for p in (frozen, warm, resumed)) > 1:
            raise ValueError("Select one of frozen generator, weights-only warm start or exact resume")
        corr_init = absolute(initial["correction"])
        corr_resume = absolute(c["resume_from"])
        if not self.corrected and (corr_init or corr_resume or initial["prototypes"]):
            raise ValueError("The selected path does not use correction checkpoints")
        if corr_init and (corr_resume or initial["prototypes"]):
            raise ValueError("A frozen correction does not train or use prototypes")
        if frozen and self.corrected and not corr_init:
            raise ValueError("A frozen generator needs the correction it trained on")
        if frozen:
            if corr_init:
                self._load_correction(corr_init)
            self._load_generator(frozen)
        else:
            data, holdout = self._data(train), self._data(val)
            self.manifest["train_cache"], self.manifest["val_cache"] = data.key, holdout.key
            if self.corrected:
                if corr_init:
                    self._load_correction(corr_init)
                else:
                    energy = self._fit_energy(data, absolute(initial["prototypes"]))
                    self.correction = self._make_correction()
                    signature = self._correction_signature()
                    self.correction = train_correction(self.correction, energy, self.spec.path, self.spec, data,
                                                       self.codec, c, self.device, self.ctx.seed, self.ctx,
                                                       signature, corr_resume)
                    path = self.ctx.ckpt_dir / "correction_final.pt"
                    save_atomic({"model": self.correction.state_dict(), "signature": signature}, path)
                    self._record_source("correction", path)
                    if c["diagnostic_frames"] > 0:
                        self.ctx.log_metrics(correction_diagnostics(
                            self.correction, energy, self.spec, holdout, self.codec,
                            self.device, self.ctx.seed, c["diagnostic_frames"]))
                    del energy
            upstream = self.sources.get("correction", {}).get("sha256")
            self.generator_signature["correction_sha256"] = upstream
            if warm:
                self._load_generator(warm, warm=True)
            if self.spec.uses_fk:
                self.metric_weights = group_weights(hp["supervision"]["groups"]).to(self.device)
                self._get_fk()
            self.gen.to(self.device)
            train_generator(self.gen, self.correction, self.spec, data, holdout, self.codec, g, self.device,
                            self.ctx.seed, self.ctx, self.generator_signature, self.fk, self.metric_weights,
                            resumed, upstream)
            trained = self.ctx.ckpt_dir / "generator.pt"
            self._record_source("generator", trained if trained.is_file() else resumed)
        self.gen.to(self.device).eval()
        self.manifest["parameter_count"] = sum(p.numel() for p in self.gen.parameters())

    def save(self):
        write_json(self.ctx.ckpt_dir / "experiment.json", self.manifest)

    def load(self, ckpt_dir):
        self._setup()
        directory = Path(ckpt_dir)
        manifest = json.loads((directory / "experiment.json").read_text())
        for component in (("correction",) if self.corrected else ()) + ("generator",):
            path = Path(manifest["sources"][component]["path"])
            local = directory / ("correction_final.pt" if component == "correction" else "generator.pt")
            path = local if local.is_file() else path
            if manifest["sources"][component]["sha256"] != fingerprint(path):
                raise ValueError(f"Checkpoint content changed for {component}")
            (self._load_correction if component == "correction" else self._load_generator)(path)
        self.gen.to(self.device).eval()

    @torch.no_grad()
    def infer(self, inputs, *, lengths=None, sampling_seed=None):
        if len(inputs) == 0:
            return [], {}
        s = self.hparams["sampler"]
        n = len(inputs)
        seed = s["seed"] if sampling_seed is None else sampling_seed
        seed = self.ctx.seed if seed is None else int(seed)
        batch = s["batch_size"]
        ids = [item["name"] for item in inputs]
        if len(set(ids)) != n or len({name.replace("/", "__") for name in ids}) != n:
            raise ValueError("Clip IDs must be unique before and after archive encoding")
        texts = [item["text"] for item in inputs]
        sources = [item["src"] for item in inputs]
        if lengths is None:
            lengths = []
            for i in range(0, n, batch):
                text, pad = self.gen.encode_text(texts[i:i + batch], sources[i:i + batch], self.device)
                loglen = self.gen.predict_log_length(text, pad)
                lengths.extend(loglen.exp().clamp(s["min_len"], s["max_len"]).round().long().tolist())
        if len(lengths) != n or any(not 0 < length <= self.gen.max_frames for length in lengths):
            raise ValueError("Invalid input or locked lengths")
        predictions = [None] * n
        diagnostics = {}
        legacy = s["noise"] == "legacy"
        order = np.argsort(lengths, kind="quicksort" if legacy else "stable").tolist()
        generator = rng(seed, self.device)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            batch_lengths = [lengths[j] for j in idx]
            frames = max(batch_lengths)
            text, pad = self.gen.encode_text([texts[j] for j in idx], [sources[j] for j in idx], self.device)
            mask = torch.arange(frames, device=self.device)[None] < torch.tensor(batch_lengths, device=self.device)[:, None]
            if legacy:
                prior = sample_source(self.spec, (len(idx), frames, N_JOINTS), self.device, generator)
                expr = torch.randn(len(idx), frames, EXPR_DIM, device=self.device, generator=generator)
            else:
                keys = [f"{sources[j]}/{ids[j]}" for j in idx]
                prior, expr = per_clip_sources(self.spec, keys, seed, frames, self.gen.max_frames, self.device)
            rot, expr, drift = sample(self.gen, self.correction, self.spec, text, pad, mask, prior, expr,
                                      self.grid, s["expression_update"], s["diagnostics"])
            normalized = self.codec.decode(rot, expr).float().cpu().numpy()
            if not np.isfinite(normalized).all():
                raise FloatingPointError("Non-finite sampled output")
            steps = torch.stack(drift, dim=-1).cpu().tolist() if drift else [[] for _ in idx]
            for k, j in enumerate(idx):
                predictions[j] = normalized[k, :batch_lengths[k]].copy()
                if s["diagnostics"]:
                    diagnostics[ids[j]] = {"length": int(batch_lengths[k]), "endpoint_drift_deg": steps[k]}
        return predictions, diagnostics

    def predict(self, inputs):
        predictions, diagnostics = self.infer(inputs)
        path = self.ctx.run_dir / "predictions.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **{
            x["name"].replace("/", "__"): p for x, p in zip(inputs, predictions)
        })
        os.replace(temporary, path)
        if diagnostics:
            write_json(self.ctx.run_dir / "sampling_diagnostics.json", diagnostics)
        return predictions

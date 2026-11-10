"""SLP registry bridge; cached records are the explicit data contract, not old experiments."""

from pathlib import Path
from slp.method import BaseMethod
from .config import RaceConfig
from .geometry import SMPLXGeometry
from .data import InferenceBatch, RecordDataset
from .pipeline import RacePipeline
from .preparation import cache_candidates
from .training import train_decoder, train_selector, load_checkpoint, save_checkpoint


class Method(BaseMethod):
    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.pipeline = None
        self.provenance = {}
        self.device = self.hparams.get("device", "cpu")

    def fit(self, train, val):
        # Shared Split may point at our audited cache manifests; never infer split membership.
        train_path, val_path = train.meta.get("race_manifest"), val.meta.get("race_manifest")
        if train_path is None or val_path is None:
            raise ValueError("Provide explicit race_manifest in Split.meta; see race_sign README")
        paths = self.hparams.get("selector_manifests")
        policy = self.hparams.get("policy", "coupled")
        if policy in ("coupled", "sentence") and not paths:
            raise ValueError(
                "Coupled/sentence fit requires three selector_manifests; use flow for decoder-only runs"
            )
        cfg = RaceConfig.from_dict(self.hparams.get("model", {}))
        geometry = SMPLXGeometry(self.hparams["smplx_path"]).to(self.device)
        directory = self.ctx.ckpt_dir
        decoder, norm, _ = train_decoder(
            cfg,
            RecordDataset(train_path, "decoder_train"),
            RecordDataset(val_path, "decoder_val"),
            geometry,
            directory,
            device=self.device,
            seed=self.ctx.seed,
            epochs=self.hparams.get("epochs", 100),
            batch_size=self.hparams.get("batch_size", 8),
            log=self.ctx.log_metrics,
        )
        selector = None
        decoder, norm, _, saved = load_checkpoint(directory / "best.pt", self.device)
        self.provenance = saved["provenance"]
        if paths:
            candidate_paths = {}
            for key, role in (
                ("train", "selector_train"),
                ("val", "selector_val"),
                ("calibration", "selector_calibration"),
            ):
                candidate_paths[key] = cache_candidates(
                    directory / "best.pt",
                    paths[key],
                    role,
                    geometry,
                    self.ctx.run_dir / "candidate_cache" / role,
                    device=self.device,
                    seed=self.ctx.seed,
                )
            selector, _ = train_selector(
                directory / "best.pt",
                RecordDataset(candidate_paths["train"], "selector_train"),
                RecordDataset(candidate_paths["val"], "selector_val"),
                RecordDataset(candidate_paths["calibration"], "selector_calibration"),
                geometry,
                directory / "selector.pt",
                device=self.device,
                seed=self.ctx.seed,
                epochs=self.hparams.get("selector_epochs", 30),
                whole_sentence=policy == "sentence",
            )
            decoder, norm, selector, saved = load_checkpoint(directory / "selector.pt", self.device)
            self.provenance = saved["provenance"]
        self.pipeline = RacePipeline(decoder, norm, geometry, selector).eval()

    def predict(self, inputs):
        if self.pipeline is None:
            raise RuntimeError("Fit or load a checkpoint first")
        if not isinstance(inputs, InferenceBatch):
            raise TypeError("SLP predict inputs must be a target-free InferenceBatch")
        return self.pipeline.predict(
            inputs.to(self.device), policy=self.hparams.get("policy", "coupled"), seed=self.ctx.seed
        )

    def load(self, ckpt_dir):
        directory = Path(ckpt_dir)
        if directory.is_file():
            path = directory
        else:
            candidates = [directory / name for name in ("selector.pt", "best.pt", "model.pt")]
            path = next((p for p in candidates if p.exists()), directory / "best.pt")
        decoder, norm, selector, saved = load_checkpoint(path, self.device)
        self.provenance = saved["provenance"]
        geometry = SMPLXGeometry(self.hparams["smplx_path"]).to(self.device)
        self.pipeline = RacePipeline(decoder, norm, geometry, selector).eval()

    def save(self):
        if self.pipeline is None:
            raise RuntimeError("Nothing to save")
        save_checkpoint(
            self.ctx.ckpt_dir / "model.pt",
            self.pipeline.decoder,
            self.pipeline.normalizer,
            selector=self.pipeline.selector,
            stage="selector" if self.pipeline.selector else "decoder",
            provenance=self.provenance,
        )

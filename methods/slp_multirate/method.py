"""Repository method interface for the canonical multi-rate signing representation."""

from pathlib import Path

import torch

from slp.method import BaseMethod

from .cli import create_bundle, create_cascade
from .config import STAGES, resolve
from .data import CanonicalDataset
from .geometry import Rig, matrix_to_rot6d
from .pipeline import inference_dependencies, required_dependencies
from .training import split_hash, train_stage


class Method(BaseMethod):
    """Fit canonical train/dev_tune and accept only strings or strict text records."""

    def _settings(self):
        hp = self.hparams
        return resolve(hp.get("pilot"), hp.get("variant"), hp.get("config"),
                       [*hp.get("overrides", []), f"experiment.seed={self.ctx.seed}"])

    def fit(self, train, val):
        cfg = self._settings()
        if isinstance(train, CanonicalDataset) and isinstance(val, CanonicalDataset):
            rig, manifest = train.rig, train.manifest
            cohort = train.cohort
            if cohort != val.cohort:
                raise ValueError("Train and validation cohorts must match")
            rig_file = manifest.parent / "rig.json"
            if train.manifest != val.manifest or train.rig.sha256 != val.rig.sha256:
                raise ValueError("Train and validation canonical manifests must match")
        else:
            required = ("canonical_manifest", "canonical_rig")
            if (not all(train.meta.get(k) and val.meta.get(k) for k in required) or
                    any(train.meta.get(k) != val.meta.get(k) for k in (*required, "cohort"))):
                raise ValueError("Provide canonical train/dev_tune datasets or Split metadata "
                                 "naming the same canonical manifest, rig and optional cohort")
            manifest, rig_file = Path(train.meta[required[0]]), Path(train.meta[required[1]])
            rig = Rig.load(rig_file)
            cohort = train.meta.get("cohort")
            train = CanonicalDataset(manifest, rig, ("train",), cohort)
            val = CanonicalDataset(manifest, rig, ("dev_tune",), cohort)
        if cfg["anchors"]["confidence"] != "uniform":
            raise ValueError("Use the staged CLI for held-out confidence calibration")
        if Rig.load(rig_file).sha256 != rig.sha256:
            raise ValueError("Canonical manifest must be accompanied by its matching rig")
        root = self.ctx.ckpt_dir
        key, device = split_hash(manifest, cohort), torch.device(self.hparams.get("device", "cpu"))
        paths = {}
        for stage in STAGES:
            if stage == "anchors" and cfg["anchors"]["policy"] == "none":
                continue
            needed = {d: paths[d] for d in required_dependencies(stage, cfg)}
            cascade = create_cascade(cfg, rig, device, needed, key, root / "text", root / "conditions")
            paths[stage] = train_stage(stage, cascade, train, val, root / stage, cohort)
        self._bundle = create_bundle(paths, root / "bundle.json")
        self._cascade = create_cascade(cfg, rig, device, paths, key)
        from .data import atomic_bytes

        atomic_bytes(root / "rig.json", rig_file.read_bytes())

    def predict(self, inputs):
        rows = []
        for i, item in enumerate(inputs):
            if isinstance(item, str):
                rows.append((f"prompt-{i}", item))
            elif isinstance(item, dict) and set(item) == {"clip_id", "text"}:
                rows.append((item["clip_id"], item["text"]))
            else:
                raise ValueError("Prediction inputs must contain only clip_id and text")
        output = []
        for cid, text in rows:
            result = self._cascade.generate([text], [cid])
            pose = result["pose"]
            output.append({"clip_id": cid, "timestamps_s": result["times"][0].cpu().numpy(),
                           "local_rot6d": matrix_to_rot6d(pose.local[0]).cpu().numpy(),
                           "joints_m": pose.joints[0].cpu().numpy(),
                           "tips_m": pose.tips[0].cpu().numpy()})
        return output

    def load(self, ckpt_dir):
        from .cli import _bundle

        root = Path(ckpt_dir)
        cfg = self._settings()
        rig = Rig.load(root / "rig.json")
        device = torch.device(self.hparams.get("device", "cpu"))
        files = _bundle(root / "bundle.json")
        paths = {k: files[k] for k in inference_dependencies("text_only", cfg)}
        self._cascade = create_cascade(cfg, rig, device, paths, require_fine_dependencies=True)

"""SLP registry bridge. Reported metrics remain in the external evaluation runner."""

from slp.method import BaseMethod


class Method(BaseMethod):
    def fit(self, train, val):
        # The canonical manifest trainer owns preprocessing/provenance, rather than converting
        # arbitrary Split targets into an undocumented profile.
        from cprflow.cli.train import train as train_cpr
        from cprflow.config import load_config

        config_path = self.hparams.get(
            "config", "methods/cpr_flow/configs/runs/baseline_joint3.yaml"
        )
        config = load_config(config_path, self.hparams.get("overrides", []))
        train_path = getattr(train, "manifest_path", None)
        val_path = getattr(val, "manifest_path", None)
        if train_path is None or val_path is None:
            raise ValueError(
                "CPR-Flow requires canonical manifest splits; use python -m cprflow.cli.train"
            )
        from pathlib import Path

        if (
            Path(train_path).resolve()
            != Path(config["paths"]["manifests"], "train.jsonl").resolve()
            or Path(val_path).resolve() != Path(config["paths"]["manifests"], "dev.jsonl").resolve()
        ):
            raise ValueError("SLP split manifests must match the locked CPR-Flow train/dev paths")
        train_cpr(config)
        self.checkpoint = (
            Path(config["run"]["out_dir"]) / config["run"]["id"] / "checkpoints/best_ema.pt"
        )

    def predict(self, inputs):
        from cprflow.sampling.generator import MotionGenerator
        from cprflow.types import GenerationRequest

        if not hasattr(self, "generator"):
            if not hasattr(self, "checkpoint"):
                raise ValueError("Load a generator checkpoint before predict")
            self.generator = MotionGenerator.from_checkpoint(
                self.checkpoint, self.hparams.get("device", "cpu")
            )
        requests = inputs if isinstance(inputs, (tuple, list)) else [inputs]
        return [
            self.generator.generate(
                r if isinstance(r, GenerationRequest) else GenerationRequest(**r)
            )
            for r in requests
        ]

    def load(self, ckpt_dir):
        from pathlib import Path

        self.checkpoint = Path(ckpt_dir) / "best_ema.pt"

"""Repository registry integration for proposal 10."""

from pathlib import Path

from slp.method import BaseMethod

from .data import LANGUAGES, Request
from .pipeline import Pipeline
from .runtime import config_load, merge, validate_config


class Method(BaseMethod):
    """Fit split manifests and predict from text-only requests."""

    def _pipeline(self, output):
        opts = self.hparams
        config = config_load(opts.get("config"), opts.get("profile", "full"))
        merge(config, opts.get("overrides", {}))
        validate_config(config)
        return Pipeline(config, Path(output), self.ctx.seed, opts.get("resume", False))

    def fit(self, train, val):
        manifest = train.meta.get("manifest")
        if not manifest or val.meta.get("manifest") != manifest:
            raise ValueError("Supply the same proposal 10 manifest in train.meta and val.meta")
        pipe = self._pipeline(self.ctx.ckpt_dir)
        pipe.prepare(manifest)
        pipe.open()
        for split, name in ((train, "train"), (val, "dev")):
            ids = [self._request(item).sample_id for item in split.inputs]
            if sorted(ids) != sorted(c.id for c in pipe.data["subsets"][name]):
                raise ValueError(f"Split differs from prepared manifest: {name}")
        arm, folds = pipe.cfg["arm"], pipe.cfg["planner"]["oof_folds"]
        pipe.fit_planners(folds, True)
        if arm == "D":
            pipe.fit_head(folds, True)
        pipe.build_sources([arm])
        if pipe.cfg["bridge"]["mode"] == "naturalness":
            pipe.fit_bank()
            pipe.correction_stage(arm)
        pipe.generator_stage(arm)
        self.load(self.ctx.ckpt_dir)

    def load(self, ckpt_dir):
        pipe = self._pipeline(ckpt_dir).open(need_train=False)
        self.sampler = pipe.sampler(pipe.cfg["arm"])

    def _request(self, item):
        if isinstance(item, Request):
            return item
        if "source_language" in item and "target_sign_language" in item:
            source, target = item["source_language"], item["target_sign_language"]
        else:
            corpus = {"phoenix": "phoenix_2014t", "csl": "csl_daily", "how2sign": "how2sign"}[
                item["src"]
            ]
            source, target = LANGUAGES[corpus]
        return Request(
            item["text"],
            source,
            target,
            item.get("sample_id") or item.get("clip_id") or item["name"],
            item.get("seed", self.ctx.seed),
        )

    def predict(self, inputs):
        return [self.sampler.generate(self._request(item))["pose133"].numpy() for item in inputs]

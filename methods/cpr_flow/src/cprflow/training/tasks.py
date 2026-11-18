"""Executable AE, duration, CFM/diffusion and teacher objectives."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from cprflow.data.cache import digest, file_hash
from cprflow.data.collate import collate_motion, collate_text
from cprflow.geometry.temporal import lengths_mask, pool2
from cprflow.models.ae import MotionAE
from cprflow.models.dit import TypedDiT
from cprflow.models.duration import DurationModel
from cprflow.models.teacher import SignTeacher, corrupt_spans
from cprflow.objectives.ae import ae_objective
from cprflow.objectives.diffusion import CosineSchedule
from cprflow.objectives.flow import auxiliary_ramp, endpoint, linear_path, motion_fm
from cprflow.objectives.geometry import activities, fk_loss, velocity_loss
from cprflow.objectives.lag import lag_loss
from cprflow.objectives.reductions import per_clip_huber, per_clip_mse
from cprflow.objectives.semantic import multi_positive_nce, positive_masks
from cprflow.training.artifacts import (
    CachedFeatures,
    load_dependency,
    normalizer,
    scales,
    stats_path,
    text_revisions,
)
from cprflow.training.distributed import gather_grad, gather_objects


def make_model(config, motion_norm=None):
    task = config["run"]["task"]
    if task == "ae":
        return MotionAE(motion_norm, execution=config["ae"].get("execution", "per_clip"))
    if task == "duration":
        d = config["duration"]
        return DurationModel(d["hidden_dim"], d["sigma_floor"], d["sigma_max"])
    if task == "teacher":
        t = config["teacher"]
        return SignTeacher(
            t["hidden_dim"],
            t["blocks_per_part"],
            t["heads"],
            t["ffn_dim"],
            t["dropout"],
            t["embedding_dim"],
        )
    m = config["model"]
    return TypedDiT(
        m["hidden_dim"],
        m["blocks"],
        m["heads"],
        m["ffn_dim"],
        m["local_cross_part_radius_latent"],
        m["global_part_context_every"],
        config["relation"]["enabled"],
        condition_dropout=config["text"]["condition_dropout"],
    )


def training_rows(config, rows):
    rows = [
        r
        for r in rows
        if r["dataset_id"] in config["data"]["datasets"]
        and r["frames"] <= config["data"]["max_train_frames"]
    ]
    if config["run"]["id"].startswith("smoke_real48"):
        rows = [
            r
            for d in config["data"]["datasets"]
            for r in sorted(
                (r for r in rows if r["dataset_id"] == d), key=lambda r: (r["frames"], r["uid"])
            )[:16]
        ]
        if len(rows) != 48:
            raise ValueError("smoke_real48 requires 16 complete utterances per corpus")
    return rows


def provenance(config):
    paths = config["paths"]
    result = {
        "spec": config["spec_version"],
        "profile": config["data"]["profile"],
        "config": digest(config),
        "manifest": file_hash(Path(paths["manifests"]) / "train.jsonl"),
        "dev_manifest": file_hash(Path(paths["manifests"]) / "dev.jsonl"),
        "source_lock": file_hash(paths["sources_lock"]),
        "asset": file_hash(paths["smplx_model"]),
        "motion_stats": file_hash(stats_path(config, "motion")),
    }
    from cprflow.data.manifest import read_manifest

    included = [
        r["uid"]
        for r in training_rows(
            config, read_manifest(Path(paths["manifests"]) / "train.jsonl", "train")
        )
    ]
    result["included_train_hash"] = digest(included)
    if config["run"]["task"] != "ae":
        result["text_revision"], result["tokenizer_revision"] = text_revisions(config)
    if config["run"]["task"] == "flow":
        result.update(
            ae=file_hash(paths["ae_checkpoint"]),
            duration=file_hash(paths["duration_checkpoint"]),
            latent_stats=file_hash(stats_path(config, "latent")),
            relation_stats=file_hash(stats_path(config, "relation")),
        )
    if paths["teacher_checkpoint"]:
        result["teacher"] = file_hash(paths["teacher_checkpoint"])
    return result


class TaskObjective:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.task = config["run"]["task"]
        self.motion_norm = normalizer(config, "motion", device)
        self.text_cache = CachedFeatures(config, "text") if self.task != "ae" else None
        self.body = None
        self.geometry_cache = None
        self.teacher = None
        if (
            self.task == "ae"
            or self.task == "flow"
            and any(config["loss"][k] for k in ("geom", "cycle", "lag", "semantic"))
        ):
            from cprflow.geometry.body_model import BodyModel

            self.body = BodyModel(
                config["paths"]["smplx_model"], config["geometry"]["fk_chunk_frames"]
            ).to(device)
            self.scales = scales(config)
            from cprflow.data.geometry_cache import GeometryCache

            try:
                self.geometry_cache = GeometryCache(config)
            except FileNotFoundError:
                if not config["run"]["synthetic"]:
                    raise FileNotFoundError(
                        "Missing offline ground-truth geometry cache; run cprflow.cli.cache --kind geometry"
                    )
        if self.task == "flow":
            self.ae = load_dependency(
                MotionAE(), config["paths"]["ae_checkpoint"], config, "ae", device
            )
            self.latent_norm = normalizer(config, "latent", device)
            self.relation_norm = normalizer(config, "relation", device)
            self.latent_cache = CachedFeatures(config, "latent")
            self.relation_cache = (
                CachedFeatures(config, "relations") if config["relation"]["enabled"] else None
            )
            diff = config["diffusion"]
            self.diffusion = CosineSchedule(
                diff["steps"], diff["cosine_offset"], diff["beta_min"], diff["beta_max"], device
            )
            if config["loss"]["semantic"] or (
                config["guard"]["enabled"]
                and config["guard"].get("mode", "semantic") != "unrestricted"
            ):
                self.teacher = load_dependency(
                    make_teacher(config),
                    config["paths"]["teacher_checkpoint"],
                    config,
                    "teacher",
                    device,
                )
                from cprflow.sampling.guard import load_calibration

                self.calibration = load_calibration(
                    config["paths"]["teacher_calibration"],
                    {
                        "teacher_sha256": file_hash(config["paths"]["teacher_checkpoint"]),
                        "asset_sha256": file_hash(config["paths"]["smplx_model"]),
                        "preprocess_sha256": self.preprocessing_hash(),
                    },
                )
        self._prepared_geometry = {}
        self._prepared_features = {}
        self.validation = False

    def preprocessing_hash(self):
        import json

        return json.loads(
            (Path(self.config["paths"]["manifests"]) / "preprocessing_report.json").read_text()
        )["preprocessing_hash"]

    def cached(self, kind, row):
        key = (kind, row["uid"])
        if key not in self._prepared_features:
            cache = {
                "text": self.text_cache,
                "latent": getattr(self, "latent_cache", None),
                "relations": getattr(self, "relation_cache", None),
            }[kind]
            self._prepared_features[key] = cache.get(row)
        return self._prepared_features[key]

    def geometry(self, row):
        if row["uid"] not in self._prepared_geometry:
            self._prepared_geometry[row["uid"]] = self.geometry_cache.get(row)
        return {
            key: (
                value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in self._prepared_geometry[row["uid"]].items()
        }

    def text(self, rows):
        return collate_text([self.cached("text", row)["hidden"] for row in rows], self.device)

    def preflight(self, rows, logical_rows):
        # Bounded to one logical batch; do not decompress the same cache again per microstep.
        self._prepared_geometry = {}
        self._prepared_features = {}
        for row in rows:
            if row.get("synthetic", False) != self.config["run"]["synthetic"]:
                raise ValueError("Synthetic/real manifest mixing is forbidden")
            if self.geometry_cache is not None:
                if row["uid"] not in self._prepared_geometry:
                    self._prepared_geometry[row["uid"]] = self.geometry_cache.get(row)
            if self.text_cache:
                self.cached("text", row)
            if self.task == "flow":
                self.cached("latent", row)
                if self.relation_cache and self.config["run"]["variant"] != "capacity_control":
                    self.cached("relations", row)
        if self.config["loss"]["semantic"] and self.task == "flow":
            for row in logical_rows:
                self.cached("text", row)

    def __call__(self, model, rows, step, logical_rows):
        batch = collate_motion(rows, self.motion_norm, self.device)
        if self.task == "ae":
            pred, _ = model(batch.x, batch.lengths)
            vector, logs = ae_objective(
                pred,
                batch,
                self.motion_norm,
                self.body,
                self.scales,
                step,
                dense=self.validation,
                weights=self.config["ae"]["weights"],
                auxiliary_stride=self.config["ae"]["auxiliary_frame_stride"],
                batched_fk=self.config["ae"].get("execution", "per_clip") == "masked_batch",
                target_geometry=[self.geometry(row) for row in rows]
                if self.geometry_cache
                else None,
            )
            return vector, logs
        text = self.text(rows)
        if self.task == "duration":
            mu, sigma = model(text.hidden, text.mask, batch.language_id)
            target = batch.lengths.float().log()
            nll = (
                target
                + sigma.float().log()
                + 0.5 * ((target - mu.float()) / sigma.float()).square()
                + 0.5 * math.log(2 * math.pi)
            )
            return nll, {
                "absolute_log_length_error": (mu - target).abs(),
                "mae_frames": (mu.exp() - batch.lengths).abs(),
            }
        if self.task == "teacher":
            return self.teacher_objective(model, rows, batch, text)
        return self.flow_objective(model, rows, batch, text, step, logical_rows)

    def teacher_objective(self, model, rows, batch, text):
        raw = model.module if hasattr(model, "module") else model
        clean = model(batch.x, batch.lengths, batch.timestamps)
        text_features = raw.encode_text(text.hidden, text.mask, batch.language_id)
        # Each rank retains its own query rows; gradient-aware keys span the logical pool.
        motions = gather_grad(clean["global"])
        texts = gather_grad(text_features)
        all_rows = [
            r
            for group in gather_objects(
                [{"text": r["text"], "sign_language_id": r["sign_language_id"]} for r in rows]
            )
            for r in group
        ]
        all_languages = torch.tensor([r["sign_language_id"] for r in all_rows], device=self.device)
        positive, allowed = positive_masks(
            batch.text, batch.language_id, [r["text"] for r in all_rows], all_languages
        )
        a, valid = multi_positive_nce(
            clean["global"], texts, positive, allowed, self.config["teacher"]["temperature"]
        )
        b, _ = multi_positive_nce(
            text_features, motions, positive, allowed, self.config["teacher"]["temperature"]
        )
        tc = self.config["teacher"]
        corrupted, mask = corrupt_spans(
            batch.x, batch.lengths, tc["mask_probability"], tc["span_min"], tc["span_max"]
        )
        reconstruction = model(corrupted, batch.lengths, batch.timestamps, mask)["reconstruction"]
        terms = [
            per_clip_huber(p, t, mask & batch.frame_mask)
            for p, t in zip(
                reconstruction.split((60, 90, 90, 56), -1), batch.x.split((60, 90, 90, 56), -1)
            )
        ]
        rec = torch.stack(terms).mean(0)
        vector = (a + b) * 0.5 + tc["reconstruction_weight"] * rec
        return vector, {
            "contrastive": (a + b) * 0.5,
            "masked_reconstruction": rec,
            "teacher_coverage": valid.float(),
            "teacher_negative_pool": torch.full_like(vector, len(all_rows)),
        }

    def flow_objective(self, model, rows, batch, text, step, logical_rows):
        cfg = self.config
        z = (
            pad_sequence(
                [
                    torch.from_numpy(self.cached("latent", row)["z"]).permute(1, 0, 2)
                    for row in rows
                ],
                batch_first=True,
            )
            .permute(0, 2, 1, 3)
            .to(self.device)
        )
        latent_lengths = (batch.lengths + 1) // 2
        mask = lengths_mask(latent_lengths, z.shape[2])
        z = self.latent_norm(z) * mask[:, None, :, None]
        noise = torch.randn_like(z) * mask[:, None, :, None]
        times = pool2(batch.timestamps[..., None], batch.lengths)[..., 0]
        b = len(z)
        t = torch.rand(b, device=self.device)
        r, rt, rvel = None, None, None
        if cfg["run"]["variant"] == "diffusion":
            k = torch.randint(1, self.diffusion.steps + 1, (b,), device=self.device)
            state = self.diffusion.corrupt(z, noise, k)
            target = noise
            t = k.float() / self.diffusion.steps
        else:
            state, target = linear_path(z, noise, t)
        if cfg["relation"]["enabled"]:
            if cfg["run"]["variant"] == "capacity_control":
                # Stable per-clip control signal, independent of motion/geometry.
                signals = []
                for row, n in zip(rows, latent_lengths):
                    rng = torch.Generator(device=self.device).manual_seed(
                        int(digest(row["uid"])[:16], 16) % (2**63)
                    )
                    signals.append(torch.randn(int(n), 33, generator=rng, device=self.device))
                r = pad_sequence(signals, batch_first=True)
            else:
                r = pad_sequence(
                    [torch.from_numpy(self.cached("relations", row)["r"]) for row in rows],
                    batch_first=True,
                ).to(self.device)
                r = self.relation_norm(r) * mask[..., None]
            rnoise = torch.randn_like(r) * mask[..., None]
            rt, rvel = linear_path(r, rnoise, t)
        drop = (
            torch.rand(b, device=self.device) < cfg["text"]["condition_dropout"]
            if not self.validation
            else torch.zeros(b, device=self.device, dtype=torch.bool)
        )
        pred = model(
            state,
            rt,
            t,
            text.hidden,
            text.mask,
            batch.language_id,
            batch.lengths,
            mask,
            times,
            drop_text=drop,
            relation_alpha=min(1.0, step / max(1, cfg["relation"]["warmup_key_gate_updates"]))
            if cfg["relation"]["enabled"] and cfg["relation"]["warmup_required"]
            else 1.0,
        )
        fm = motion_fm(pred.z, target, mask) * cfg["flow"]["fm_motion_weight"]
        logs = {
            "fm_motion": fm.detach(),
            "time": t.detach(),
            "cfg_drop": drop.float(),
            "padding_fraction": 1 - mask.float().mean(1),
        }
        for bin_index in range(10):
            logs[f"t_bin_{bin_index}"] = (
                (t >= bin_index / 10)
                & (t <= (bin_index + 1) / 10 if bin_index == 9 else t < (bin_index + 1) / 10)
            ).float()
        for key in ("geom", "cycle", "lag", "semantic", "lag_coverage", "degenerate6d_rate"):
            logs[key] = torch.zeros_like(t)
        for part in range(4):
            logs[f"fm_motion_part_{part}"] = per_clip_mse(
                pred.z[:, part], target[:, part], mask
            ).detach()
        vector = fm
        if rt is not None:
            rf = per_clip_mse(pred.r, rvel, mask)
            logs["fm_relation"] = rf.detach()
            vector = vector + cfg["flow"]["fm_relation_weight"] * rf
        loss = cfg["loss"]
        every = loss["auxiliary_every_updates"]
        ramp = auxiliary_ramp(step, loss["auxiliary_ramp_start"], loss["auxiliary_ramp_end"])
        enabled = any(loss[k] for k in ("geom", "cycle", "lag", "semantic"))
        eligible = t >= loss["auxiliary_time_min"]
        logs["aux_coverage"] = eligible.float() * float(enabled and step % every == 0 and ramp > 0)
        if enabled and step % every == 0 and ramp > 0 and eligible.any():
            zhat = endpoint(state, pred.z, t)
            rhat = endpoint(rt, pred.r, t) if rt is not None else None
            aux, auxlogs = self.auxiliary(zhat, rhat, batch, t, drop, eligible, logical_rows, rows)
            vector = vector + loss["auxiliary_frequency_compensation"] * ramp * aux
            logs.update(auxlogs)
        return vector, logs

    def auxiliary(self, zhat, rhat, batch, t, drop, eligible, logical_rows, rows):
        from cprflow.geometry.relations import relation33

        cfg = self.config
        losses = []
        rawlogs = {
            k: [] for k in ("geom", "cycle", "lag", "semantic", "lag_coverage", "degenerate6d_rate")
        }
        logical_text = None
        text_keys = None
        if cfg["loss"]["semantic"]:
            logical_text = self.text(logical_rows)
            langs = torch.tensor([r["sign_language_id"] for r in logical_rows], device=self.device)
            with torch.no_grad():
                text_keys = self.teacher.encode_text(logical_text.hidden, logical_text.mask, langs)
        frame_lengths = batch.lengths.detach().cpu().tolist()
        eligibility = eligible.detach().cpu().tolist()
        packed = {}
        selected = [i for i, value in enumerate(eligibility) if value]
        if cfg["ae"].get("execution", "per_clip") == "masked_batch" and len(selected) > 1:
            from cprflow.objectives.ae_batch import split_geometry

            select_tensor = torch.tensor(selected, device=self.device)
            normalized = self.ae.decode(
                self.latent_norm.inverse(zhat[select_tensor]), batch.lengths[select_tensor]
            )
            physical = [
                self.motion_norm.inverse(normalized[j, : frame_lengths[i]])
                for j, i in enumerate(selected)
            ]
            geometries = split_geometry(
                self.body(torch.cat(physical)), [frame_lengths[i] for i in selected]
            )
            packed = {
                i: (normalized[j, : frame_lengths[i]], physical[j], geometries[j])
                for j, i in enumerate(selected)
            }
        for i, n in enumerate(frame_lengths):
            zero = zhat[i].sum() * 0
            terms = {k: zero for k in rawlogs}
            if eligibility[i]:
                n = int(n)
                if i in packed:
                    xhat_norm, xhat, pg = packed[i]
                else:
                    xhat_norm = self.ae.decode(
                        self.latent_norm.inverse(zhat[i : i + 1]), batch.lengths[i : i + 1]
                    )[0, :n]
                    xhat = self.motion_norm.inverse(xhat_norm)
                    pg = self.body(xhat)
                xgt = self.motion_norm.inverse(batch.x[i, :n])
                terms["degenerate6d_rate"] = pg["degenerate"].float().mean()
                if self.geometry_cache:
                    tg = self.geometry(rows[i])
                else:
                    with torch.no_grad():
                        tg = self.body(xgt)
                rp = relation33(
                    pg["joints"],
                    pg["names"],
                    cfg["relation"]["threshold"],
                    cfg["relation"]["softness"],
                )
                rt = relation33(
                    tg["joints"],
                    tg["names"],
                    cfg["relation"]["threshold"],
                    cfg["relation"]["softness"],
                )
                ap = self.relation_norm(pool2(rp[None], batch.lengths[i : i + 1])[0])
                at = self.relation_norm(pool2(rt[None], batch.lengths[i : i + 1])[0])
                if cfg["loss"]["geom"]:
                    terms["geom"] = (
                        F.huber_loss(ap, at)
                        + fk_loss(pg, tg)
                        + 0.1
                        * velocity_loss(
                            pg,
                            tg,
                            xhat[:, 246:],
                            xgt[:, 246:],
                            batch.timestamps[i, :n],
                            self.scales,
                        )
                    )
                if cfg["loss"]["cycle"]:
                    terms["cycle"] = F.huber_loss(rhat[i, : len(ap)], ap)
                if cfg["loss"]["lag"]:
                    pa = activities(
                        xhat, pg, batch.timestamps[i, :n], self.motion_norm, self.scales
                    )
                    with torch.no_grad():
                        ta = activities(
                            xgt, tg, batch.timestamps[i, :n], self.motion_norm, self.scales
                        )
                    lc = cfg["loss"]
                    terms["lag"], coverage = lag_loss(
                        pa,
                        ta,
                        lc["lag_window"],
                        lc["lag_hop"],
                        lc["lag_max_offset"],
                        lc["lag_temperature"],
                        lc["lag_variance_floor"],
                    )
                    terms["lag_coverage"] = zero + coverage
                if cfg["loss"]["semantic"] and not drop[i]:
                    q = self.teacher(
                        xhat_norm[None], batch.lengths[i : i + 1], batch.timestamps[i : i + 1, :n]
                    )["global"]
                    positive, allowed = positive_masks(
                        [batch.text[i]],
                        batch.language_id[i : i + 1],
                        [r["text"] for r in logical_rows],
                        langs,
                    )
                    semantic, _ = multi_positive_nce(
                        q, text_keys, positive, allowed, cfg["teacher"]["temperature"]
                    )
                    terms["semantic"] = semantic[0]
            losses.append(
                sum(cfg["loss"][key] * terms[key] for key in ("geom", "cycle", "lag", "semantic"))
            )
            for key, value in terms.items():
                rawlogs[key].append(value.detach())
        return torch.stack(losses), {key: torch.stack(values) for key, values in rawlogs.items()}


def make_teacher(config):
    t = config["teacher"]
    return SignTeacher(
        t["hidden_dim"],
        t["blocks_per_part"],
        t["heads"],
        t["ffn_dim"],
        t["dropout"],
        t["embedding_dim"],
    )

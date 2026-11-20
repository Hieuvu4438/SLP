"""Shared training and inference assembly, checkpoint contracts and condition cache keys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .anchors import (
    build_oracle_anchors, clip_pose, empty_anchors, jitter_anchors, score_scales,
    stack_anchors,
)
from .data import digest, file_hash
from .flow import Moments, Stats, sample_cfm, seeded_noise
from .geometry import (
    build_body_grid, build_fine_target, decode_pose, grid_timestamps, matrix_to_rot6d,
    rot6d_to_matrix, upsample_body,
)
from .models import Condition, TextBatch, TextEncoder, make_model, plan_anchors


@dataclass
class Statistics:
    body: Stats
    anchor: Stats
    fine: Stats | None
    event_scales: list[float]

    def state(self):
        out = {key: getattr(self, key).state() if getattr(self, key) is not None else None
               for key in ("body", "anchor", "fine")}
        return dict(out, event_scales=self.event_scales)

    @classmethod
    def load(cls, value):
        return cls(*(Stats(**value[k]) if value[k] is not None else None
                     for k in ("body", "anchor", "fine")), value["event_scales"])


def tensor_state_hash(state):
    def encode(x):
        if isinstance(x, dict):
            return {key: encode(v) for key, v in x.items()}
        if isinstance(x, torch.Tensor):
            return [str(x.dtype), list(x.shape), x.detach().cpu().reshape(-1).tolist()]
        return x
    return digest(encode(state))


def atomic_torch(path, value):
    # Write the tensor archive to disk first to avoid duplicating large checkpoints in memory.
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def load_torch(path):
    return torch.load(path, weights_only=True, map_location="cpu")


class TextStore:
    def __init__(self, cfg, device, root=None):
        self.device = torch.device(device)
        self.encoder = TextEncoder(cfg).to(device).eval()
        self.dim = self.encoder.dim
        self.contract = dict(cfg, hidden_dim=self.dim, schema=1)
        self.root = Path(root) if root is not None and cfg["cache_hidden"] else None

    def __call__(self, texts, split=None):
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._fetch(texts, split)

    def _fetch(self, texts, split):
        if self.root is None or split is None:
            return self.encoder(texts)
        root = self.root / split
        keys = [digest([self.contract, text]) for text in texts]
        rows = {}
        missing = {}
        for key, text in zip(keys, texts):
            if (root / f"{key}.pt").exists():
                value = load_torch(root / f"{key}.pt")
                if value["key"] != key or value["hidden"].shape[-1] != self.dim:
                    raise ValueError("Text cache contract mismatch")
                rows[key] = value
            else:
                missing[key] = text
        if missing:
            output = self.encoder(list(missing.values()))
            for i, key in enumerate(missing):
                rows[key] = {"hidden": output.hidden[i, output.mask[i]].cpu(),
                             "truncated": bool(output.truncated[i]), "key": key}
                atomic_torch(root / f"{key}.pt", rows[key])
        lengths = [len(rows[k]["hidden"]) for k in keys]
        width = max(lengths)
        hidden = torch.stack([F.pad(rows[key]["hidden"], (0, 0, 0, width - n))
                              for key, n in zip(keys, lengths)]).to(self.device)
        mask = torch.arange(width, device=self.device) < torch.tensor(
            lengths, device=self.device)[:, None]
        return TextBatch(hidden, mask, torch.tensor([rows[k]["truncated"] for k in keys],
                                                     device=self.device))


def _architecture(stage, cfg):
    if stage == "length":
        return {}
    if stage == "anchors":
        a = cfg["anchors"]
        return {key: a[key] for key in (
            "policy", "count", "width", "blocks", "heads", "nms_min_gap_frames",
            "turn_min_speed_shoulder_widths", "score_weights", "contact_mining")}
    if stage == "body":
        return cfg["body"]
    return {"fine": cfg["fine"], "stride": cfg["body"]["stride"], "anchors": {
        key: cfg["anchors"][key] for key in ("count", "policy", "sigma_frames", "use_time_bias")}}


def stage_contract(stage, cfg, rig, text):
    data = cfg["data"]
    return dict(schema=1, stage=stage, rig_sha256=rig.sha256, text=text.contract,
                architecture=_architecture(stage, cfg), fps=data["fps"],
                length_range=[data["min_frames"], data["max_frames"]],
                std_floor=data["normalize_std_floor"])


@dataclass
class FrozenStage:
    model: torch.nn.Module
    state: dict
    sha256: str


def load_stage(path, stage, cfg, rig, text, device, split_hash=None):
    path = Path(path)
    payload = load_torch(path)
    if payload["stage_contract"] != stage_contract(stage, cfg, rig, text):
        raise ValueError(f"{stage} checkpoint incompatible with rig, text or architecture")
    if split_hash is not None and payload["split_hash"] != split_hash:
        raise ValueError(f"{stage} checkpoint split/cohort hash mismatch")
    model = make_model(stage, text.dim, cfg).to(device)
    model.load_state_dict(payload["ema"])
    return FrozenStage(model.requires_grad_(False).eval(), payload, file_hash(path))


PREDICTED_PROTOCOLS = {
    "predicted_body_oracle_anchors", "predicted_gt_length", "oracle_body_predicted_anchors",
    "text_only",
}
PROTOCOLS = ("oracle", *sorted(PREDICTED_PROTOCOLS))


def required_dependencies(stage, cfg):
    if stage != "fine":
        return ()
    mix = cfg["train"]["condition_prob"]
    if not mix["predicted"] and not mix["no_anchor"]:
        return ()
    deps = ["body"]
    if mix["predicted"] and cfg["anchors"]["policy"] != "none":
        deps.append("anchors")
    if "anchors" in deps and cfg["anchors"]["confidence"] == "learned":
        deps.append("confidence")
    return tuple(deps)


def inference_dependencies(protocol, cfg):
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown inference protocol")
    deps = ["fine"]
    if protocol == "text_only":
        deps.append("length")
    if protocol in ("predicted_body_oracle_anchors", "predicted_gt_length", "text_only"):
        deps.append("body")
    if (protocol in ("predicted_gt_length", "oracle_body_predicted_anchors", "text_only") and
            cfg["anchors"]["policy"] != "none"):
        deps.append("anchors")
        if cfg["anchors"]["confidence"] == "learned":
            deps.append("confidence")
    return tuple(deps)


@dataclass
class Cascade:
    cfg: dict
    rig: object
    text: TextStore
    device: torch.device
    frozen: dict
    stats: Statistics | None = None
    condition_root: Path | None = None
    confidence: torch.nn.Module | None = None
    confidence_hash: str | None = None

    @property
    def seed(self):
        return self.cfg["experiment"]["seed"]

    def move_batch(self, batch):
        return {key: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for key, v in batch.items()}

    def oracle_grid(self, batch):
        return build_body_grid(batch["body_rot6d"], batch["timestamps_s"], batch["frame_mask"],
                               batch["body_obs"], self.cfg["body"]["stride"])

    def oracle(self, batch):
        return stack_anchors([build_oracle_anchors(c, self.rig, self.cfg["anchors"],
                              self.stats.event_scales, self.stats.anchor.normalize)
                              for c in batch["clips"]], self.device)

    def predicted(self, text, lengths):
        settings = self.cfg["anchors"]
        if settings["policy"] == "none":
            return empty_anchors(len(lengths), settings["count"], self.device)
        return plan_anchors(self.frozen["anchors"].model, text, lengths, self.confidence)

    def body_norm(self, matrices, mask):
        return self.stats.body.normalize(matrix_to_rot6d(matrices).flatten(-2), mask[..., None])

    @torch.no_grad()
    def sampled_body(self, text, ids, time, mask, seed, sample_index):
        ts, cm = grid_timestamps(time, mask, self.cfg["body"]["stride"])
        x = seeded_noise(ids, seed, "body", sample_index, cm, 72)
        clean = sample_cfm(self.frozen["body"].model, x, Condition(text, ts, cm), cm,
                           self.cfg["sample"]["body_nfe"])
        grid = rot6d_to_matrix(self.stats.body.denormalize(clean).unflatten(-1, (12, 6)))
        full, obs = upsample_body(grid, ts, cm, time, mask)
        return grid, ts, cm, full, obs

    def oracle_body(self, batch):
        grid = self.oracle_grid(batch)
        full, seen = upsample_body(grid.rotations, grid.timestamps, grid.mask, batch["timestamps_s"],
                                   batch["frame_mask"], grid.observation)
        return grid.rotations, grid.timestamps, grid.mask, full, seen

    def _condition_key(self, clip, need_anchor):
        return digest(dict(schema=1, clip=clip["row"]["clip_id"], length=len(clip["timestamps_s"]),
                           split=clip["row"]["split"], clip_sha256=clip["row"]["sha256"],
                           text_sha256=digest(clip["row"]["text"]), backend=self.device.type,
                           policy=self.cfg["anchors"]["policy"], seed=self.seed,
                           stride=self.cfg["body"]["stride"], nfe=self.cfg["sample"]["body_nfe"],
                           rig=self.rig.sha256, text=self.text.contract,
                           stats=tensor_state_hash({"body": self.stats.body.state(),
                                                    "anchor": self.stats.anchor.state()}),
                           checkpoints={k: v.sha256 for k, v in self.frozen.items()},
                           confidence=self.confidence_hash, with_anchors=need_anchor))

    @torch.no_grad()
    def cached_prediction(self, clip, need_anchor=True):
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._cached_prediction(clip, need_anchor)

    def _cached_prediction(self, clip, need_anchor):
        key = self._condition_key(clip, need_anchor)
        path = None if self.condition_root is None else (
            self.condition_root / clip["row"]["split"] / f"{key}.pt"
        )
        if path is not None and path.exists():
            state = load_torch(path)
            if state["key"] != key:
                raise ValueError("Condition cache key mismatch")
        else:
            text = self.text([clip["row"]["text"]], clip["row"]["split"])
            t = clip["timestamps_s"][None].to(self.device)
            frame = torch.ones_like(t, dtype=torch.bool)
            ids = [clip["row"]["clip_id"]]
            grid, ct, _, body, _ = self.sampled_body(text, ids, t, frame, self.seed, 0)
            anchor = self.predicted(text, frame.sum(-1)) if need_anchor else empty_anchors(
                1, self.cfg["anchors"]["count"], self.device
            )
            state = {"key": key, "grid": grid[0].cpu(), "ct": ct[0].cpu(), "body": body[0].cpu(),
                     "anchors": {k: getattr(anchor, k).cpu() for k in (
                         "time", "payload", "slot", "feature", "confidence")}}
            if path is not None:
                atomic_torch(path, state)
        return state

    @torch.no_grad()
    def fine_training_conditions(self, batch, generator):
        p = self.cfg["train"]["condition_prob"]
        need_anchor = bool(p["predicted"])
        choose = torch.rand(len(batch["clips"]), generator=generator, device=self.device)
        grid, ct, cm, body, seen = self.oracle_body(batch)
        anchors = self.oracle(batch)
        anchors = jitter_anchors(anchors, batch["length"], self.cfg["train"]["anchor_jitter_frames"],
                                 self.cfg["train"]["anchor_payload_noise_std"], generator)
        for i, (clip, value) in enumerate(zip(batch["clips"], choose.tolist())):
            if p["predicted"] <= value < p["predicted"] + p["oracle_jitter"]:
                continue
            pred = self.cached_prediction(clip, need_anchor)
            fm, coarse = batch["frame_mask"][i], cm[i]
            grid[i, coarse], ct[i, coarse] = pred["grid"].to(self.device), pred["ct"].to(self.device)
            body[i, fm], seen[i, fm] = pred["body"].to(self.device), True
            for key in ("time", "payload", "slot", "feature", "confidence"):
                value_t = getattr(anchors, key)
                if value < p["predicted"]:
                    value_t[i] = pred["anchors"][key][0].to(self.device)
                else:
                    value_t[i] = 0
        return grid, ct, cm, body, seen, anchors

    def fine_condition(self, text, time, mask, parts):
        grid, ct, cm, body, _, anchors = parts
        return Condition(text, time, mask, self.body_norm(grid, cm), ct, cm,
                         self.body_norm(body, mask), anchors)

    @torch.no_grad()
    def generate(self, texts, ids, protocol="text_only", batch=None, seed=None, sample_index=0):
        """Generate text-only requests with no reference-bearing argument."""
        if protocol == "text_only":
            if batch is not None:
                raise ValueError("Text-only inference cannot receive reference inputs")
        elif protocol not in PROTOCOLS or batch is None:
            raise ValueError("Reference protocols require an explicitly supplied canonical batch")
        if len(texts) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Each prompt needs a unique clip ID within the batch")
        seed = self.seed if seed is None else seed
        gpu = self.device.type == "cuda"
        marks = {}

        def tic():
            if gpu:
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        start = t0 = tic()
        text = self.text(texts, batch["clips"][0]["row"]["split"] if batch else None)
        if batch is None:
            minimum, maximum = (self.cfg["data"][name] for name in ("min_frames", "max_frames"))
            lengths, clamped = self.frozen["length"].model.predict(text, minimum, maximum)
            index = torch.arange(int(lengths.max()), device=self.device)
            mask = index < lengths[:, None]
            times = (index.float() / self.cfg["data"]["fps"]).expand(len(ids), -1)
            times = times.masked_fill(~mask, 0)
        else:
            batch = self.move_batch(batch)
            lengths, mask, times = (batch[k] for k in ("length", "frame_mask", "timestamps_s"))
            clamped = torch.zeros_like(lengths, dtype=torch.bool)
        use_pred_anchors = protocol in ("text_only", "predicted_gt_length",
                                        "oracle_body_predicted_anchors")
        anchors = self.predicted(text, lengths) if use_pred_anchors else self.oracle(batch)
        marks["text_length_planner"] = (tic() - t0) * 1000
        t0 = tic()
        pred_body = protocol in ("text_only", "predicted_gt_length", "predicted_body_oracle_anchors")
        parts = (self.sampled_body(text, ids, times, mask, seed, sample_index)
                 if pred_body else self.oracle_body(batch))
        marks["body"] = (tic() - t0) * 1000
        t0 = tic()
        body = parts[3]
        cond = self.fine_condition(text, times, mask, (*parts, anchors))
        x = seeded_noise(ids, seed, "fine", sample_index, mask, 204)
        output = sample_cfm(self.frozen["fine"].model, x, cond, mask,
                            self.cfg["sample"]["fine_nfe"])
        marks["fine"] = (tic() - t0) * 1000
        t0 = tic()
        physical = self.stats.fine.denormalize(output)
        pose = decode_pose(body, physical, self.rig, self.cfg["fine"]["correction_max_rad"],
                           self.cfg["fine"]["correction_enabled"])
        marks["fk"] = (tic() - t0) * 1000
        marks["total"] = (tic() - start) * 1000
        return {"pose": pose, "times": times, "lengths": lengths, "anchors": anchors,
                "length_clamped": clamped, "nfe": {"body": self.cfg["sample"]["body_nfe"]
                if pred_body else 0, "fine": self.cfg["sample"]["fine_nfe"]}, "latency_ms": marks,
                "truncated": text.truncated}


def training_statistics(cascade, dataset, need_fine=False):
    cfg = cascade.cfg
    floor = cfg["data"]["normalize_std_floor"]
    bm, am = Moments((72,)), Moments((2, 24))
    clips = list(dataset)
    if not clips:
        raise ValueError("Normalization requires a nonempty training split")
    if any(c["row"]["split"] != "train" for c in clips):
        raise ValueError("Statistics may only be fitted on the training split")
    from .data import collate_clips

    for c in clips:
        batch = collate_clips([c])
        grid = cascade.oracle_grid(batch)
        bm.add(matrix_to_rot6d(grid.rotations).flatten(-2),
               grid.observation.repeat_interleave(6, -1))
        pose = clip_pose(c, cascade.rig)
        payload = torch.cat((pose.wrist, matrix_to_rot6d(pose.palm), pose.local_tips.flatten(-2)), -1)
        valid = torch.cat((pose.hand_mask[..., None].expand(-1, -1, 9),
                           pose.tip_mask.repeat_interleave(3, -1)), -1)
        am.add(payload, valid)
    stats = Statistics(bm.finish(floor), am.finish(floor), None,
                       score_scales(clips, cascade.rig, cfg["anchors"]))
    for stage, key in (("body", "body"), ("anchors", "anchor")):
        if stage in cascade.frozen:
            expected = Statistics.load(cascade.frozen[stage].state["statistics"])
            if tensor_state_hash(getattr(stats, key).state()) != tensor_state_hash(
                    getattr(expected, key).state()):
                raise ValueError(f"Fresh {stage} statistics differ from the checkpoint")
    cascade.stats = stats
    if need_fine:
        # Estimate a fixed training-mixture population, weighted per clip rather
        # than hoping one small random branch assignment covers each condition.
        fm = Moments((204,))
        mix = cfg["train"]["condition_prob"]
        for c in clips:
            batch = cascade.move_batch(collate_clips([c]))
            gt = rot6d_to_matrix(batch["body_rot6d"])
            choices = []
            p_oracle = mix["oracle_jitter"]
            if p_oracle:
                _, _, _, base, seen = cascade.oracle_body(batch)
                choices.append((base, seen, p_oracle))
            if p_oracle < 1:
                cached = cascade.cached_prediction(c, bool(mix["predicted"]))
                base = cached["body"][None].to(cascade.device)
                choices.append((base, torch.ones_like(batch["body_obs"]), 1 - p_oracle))
            for base, seen, weight in choices:
                target, obs, _, _ = build_fine_target(
                    gt, batch["hands_rot6d"], base, batch["body_obs"], batch["hands_obs"], seen,
                    cfg["fine"]["correction_max_rad"], cfg["fine"]["correction_enabled"])
                fm.add(target, obs, weight)
        stats.fine = fm.finish(floor)
    return stats

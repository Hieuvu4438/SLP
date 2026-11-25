"""Manifest preparation, cross-fit caches, and the staged training pipeline."""

import itertools
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import pad

from .bridge import Correction, bridge, build_bank, correction_loss
from .data import collate, digest, load_clip, phase_grid, read_manifest, resample, sha256, unpack133
from .fk import FixtureFK, SokeFK
from .models import DirectHead, Generator, Planner, TextEncoder, length_term
from .relations import (
    GROUPS,
    dense_relations,
    extract_relations,
    kinematic_mask,
    relation_terms,
    scaler,
    wrist_tips,
)
from .rotations import identity_like, physical_angle
from .runtime import (
    checkpointed_records,
    git_revision,
    json_read,
    json_write,
    lock_assets,
    npz_write,
    seed_all,
    torch_write,
    versions,
)
from .sampling import Sampler, build_center, conditioning, plan_for
from .source import UNIFORM, choose_rest, generator, sample_source
from .train import Process, run_stage


def label_at(clip, grid, fk, rest, *, plan_mask=False):
    q, valid = resample(clip.q.to(grid), grid, quaternion=True, valid=clip.q_valid.to(grid.device))
    r, mask = extract_relations(q, valid, fk, rest.to(q))
    confidence = mask.to(q)
    if plan_mask and hasattr(clip, "relation_valid"):
        _, additional = resample(clip.q_valid.to(q), grid, valid=clip.relation_valid.to(q.device))
        confidence *= additional
    if clip.confidence is not None:
        confidence *= resample(clip.confidence.to(grid), grid)
    return r, confidence


def settings(cfg):
    return {
        "epochs": cfg["max_epochs"],
        "batch": cfg["batch_effective"],
        "lr": cfg["lr"],
        "decay": cfg["weight_decay"],
        "betas": cfg.get("betas", (0.9, 0.999)),
        "warmup": cfg.get("warmup_fraction", 0),
        "clip": cfg.get("grad_clip_norm", 0),
    }


def fold_assignment(clips, folds, seed):
    groups = {}
    for clip in clips:
        key = (clip.row["corpus"], clip.row.get("oof_group") or clip.row.get("video_id") or clip.id)
        groups.setdefault(key, []).append(clip.id)
    keys = sorted(groups, key=lambda k: (k[0], -len(groups[k]), digest([seed, k])))
    if len(keys) < folds or folds < 2:
        raise ValueError("Too few independent videos for OOF")
    total, strata = [0] * folds, {}
    output = {}
    for key in keys:
        counts = strata.setdefault(key[0], [0] * folds)
        fold = min(range(folds), key=lambda f: (counts[f], total[f], f))
        counts[fold] += len(groups[key])
        total[fold] += len(groups[key])
        output.update({clip_id: fold for clip_id in groups[key]})
    if not all(total):
        raise ValueError("Empty OOF fold")
    return output


def velocity_length(clip):
    return physical_angle(clip.q[1:], clip.q[:-1]) * clip.fps


def mark_outlier_relations(train, quantile=0.999):
    groups = [list(range(8)), [8, 9], list(range(10, 40)), [40]]
    observations = [[] for _ in groups]
    for clip in train:
        if clip.confidence is not None:
            continue
        speeds = velocity_length(clip)
        label_valid = clip.q_valid[1:] & clip.q_valid[:-1]
        for i, group in enumerate(groups):
            observations[i].append(speeds[:, group][label_valid[:, group]])
    limits = []
    for speeds in observations:
        speeds = torch.cat(speeds) if speeds else torch.empty(0)
        # Full corpora can exceed torch.quantile's 2**24-element limit. The
        # concatenated buffer is disposable, so NumPy can partition it in place.
        limits.append(
            float(np.quantile(speeds.cpu().numpy(), quantile, overwrite_input=True))
            if len(speeds)
            else None
        )
    for clip in train:
        good = torch.ones_like(clip.q_valid)
        if clip.confidence is None:
            speeds = velocity_length(clip)
            valid = clip.q_valid[1:] & clip.q_valid[:-1]
            for group, limit in zip(groups, limits):
                if limit is not None:
                    outlier = (speeds[:, group] > limit) & valid[:, group]
                    good[1:, group] &= ~outlier
                    good[:-1, group] &= ~outlier
        clip.relation_valid = kinematic_mask(good)
    return limits


class Pipeline:
    def __init__(self, cfg, output=None, seed=None, resume=False):
        self.cfg = cfg
        self.root = Path(output or cfg["runtime"]["output"])
        self.seed = cfg["runtime"]["seed"] if seed is None else seed
        device = cfg["runtime"]["device"]
        if device == "auto":
            device = (
                f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
                if torch.cuda.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.process = Process(self.device)
        self.resume = resume
        self.profile = cfg["profile"]
        self.fk = None
        self.data = None
        self.state = None

    def geometry(self):
        if self.fk is None:
            if self.profile == "smoke":
                self.fk = FixtureFK().to(self.device)
            else:
                path = self.cfg["assets"].get("smplx")
                if not path or not Path(path).exists():
                    raise ValueError("assets.smplx is required by FK, IK and geometry evaluation")
                self.fk = SokeFK(path).to(self.device)
        return self.fk

    def prepare(self, manifest):
        if self.process.rank != 0:
            self.process.barrier()
            return
        rows = read_manifest(manifest)
        track = self.cfg["data"]["corpus"]
        rows = [r for r in rows if track == "multi" or r["corpus"] == track]
        if not rows:
            raise ValueError("Manifest has no clips on this track")
        data_cfg = self.cfg["data"]
        all_train_rows = [r for r in rows if r["split"] == "train"]
        fps = {}
        for corpus in sorted({r["corpus"] for r in rows}):
            observed = {r["fps"] for r in all_train_rows if r["corpus"] == corpus}
            fixed = data_cfg.get("fps", {}).get(corpus)
            if not observed or fixed is None and len(observed) != 1:
                raise ValueError(
                    f"Set data.fps.{corpus} for mixed timebases and include train rows"
                )
            fps[corpus] = float(fixed if fixed is not None else next(iter(observed)))
        assets = lock_assets(self.cfg, rows)
        if self.profile != "smoke":
            unavailable = [
                a["name"]
                for a in assets
                if a["name"] in ("smplx", "SOKE") and a["availability"] != "available"
            ]
            if unavailable:
                raise ValueError(f"Missing geometry assets: {unavailable}")
            soke = next(a for a in assets if a["name"] == "SOKE")
            if soke["resolved_revision"] != soke["revision"]:
                raise ValueError("SOKE reference revision differs from the configured revision")
        # Reference paths for test are declared here but their files are left for the evaluator.
        cooked_rows = []
        for original in rows:
            row = dict(original)
            if row["split"] != "test" and row.get("pose_path"):
                checksum = sha256(row["pose_path"])
                if row.get("sha256") and row["sha256"] != checksum:
                    raise ValueError(f"Input checksum mismatch: {row['clip_id']}")
                row["sha256"] = checksum
            cooked_rows.append(row)
        inputs = {
            "rows": cooked_rows,
            "fps": fps,
            "assets": assets,
            "cfg": self.cfg,
            "seed": self.seed,
        }
        run_id = digest(inputs)
        if (self.root / "run.json").exists():
            if json_read(self.root / "run.json")["signature"] != run_id:
                raise ValueError("Prepared run differs. Use a new output directory.")
            self.process.barrier()
            return
        subsets = {split: [] for split in ("train", "dev")}
        for row in cooked_rows:
            if row["split"] in subsets:
                subsets[row["split"]].append(load_clip(row, {**data_cfg, "fps": fps}))
        if not subsets["train"] or not subsets["dev"]:
            raise ValueError("Train and dev both need references")
        print(
            f"Loaded {len(subsets['train'])} train and {len(subsets['dev'])} dev clips.", flush=True
        )
        self._prepare_clips(subsets, cooked_rows, fps, assets, run_id, manifest)
        self.process.barrier()

    def _prepare_clips(self, subsets, rows, fps, assets, run_id, manifest):
        train = subsets["train"]
        seed_all(self.seed)
        rest, rest_frame = choose_rest(train, self.seed, self.cfg["ik"]["rest_sample_limit"])
        fk = self.geometry()
        limits = mark_outlier_relations(train)
        print("Fitted per-group angular-speed thresholds. Building FK relation labels.", flush=True)
        cap = {corpus: max(len(c.q) for c in train if c.row["corpus"] == corpus) for corpus in fps}
        with torch.no_grad():
            labels = {
                c.id: tuple(
                    t.cpu()
                    for t in label_at(
                        c, self.grid(len(c.q)), fk, rest, plan_mask=(split_name == "train")
                    )
                )
                for split_name, clips in subsets.items()
                for c in clips
            }
            pairs = [labels[c.id] for c in train]
            r, weight = (torch.cat(items).cpu() for items in zip(*pairs))
            mean, scale = scaler(r, weight)
            expr = torch.cat([c.e for c in train])
            expr_mask = torch.cat([c.expr_valid for c in train])
            expr_count = expr_mask.sum(0).clamp_min(1)
            e_mean = (expr * expr_mask).sum(0) / expr_count
            e_scale = (
                ((expr_mask * (expr - e_mean).square()).sum(0) / expr_count).sqrt().clamp_min(0.05)
            )
            if not self.cfg["data"]["expression_standardize"]:
                e_mean, e_scale = torch.zeros_like(e_mean), torch.ones_like(e_scale)
            velocity_sum = torch.zeros((12, 3), device=self.device, dtype=torch.float64)
            velocity_square_sum = torch.zeros_like(velocity_sum)
            velocity_count = 0
            for c in train:
                if len(c.q) > 1:
                    tips, valid = wrist_tips(fk(c.q.to(self.device)))
                    good = valid & c.q_valid.all(-1).to(valid.device)
                    v = tips.diff(dim=0)[good[:-1] & good[1:]].double() * c.fps
                    velocity_sum += v.sum(0)
                    velocity_square_sum += v.square().sum(0)
                    velocity_count += len(v)
            velocity_scale = torch.ones(12, 3)
            if velocity_count:
                variance = (
                    velocity_square_sum / velocity_count - (velocity_sum / velocity_count).square()
                )
                velocity_scale = variance.clamp_min(0.05**2).sqrt().cpu().float()
        stats = {
            "mean": mean,
            "scale": scale,
            "rest": rest,
            "cap": cap,
            "fps": fps,
            "expr_mean": e_mean,
            "expr_scale": e_scale,
            "velocity_scale": velocity_scale,
        }
        audit = self.pose_audit(train)
        if self.profile != "smoke" and self.cfg["assets"].get("render_preview", True):
            from .diagnostics import render_preview

            render_preview(fk, train[0].q.to(self.device), self.root / "preview.png")
        torch_write(
            self.root / "prepared.pt",
            {"subsets": subsets, "stats": stats, "labels": labels, "signature": run_id},
        )
        npz_write(
            self.root / "stats.npz",
            **{k: v.numpy() for k, v in stats.items() if isinstance(v, torch.Tensor)},
        )
        json_write(self.root / "splits.json", rows)
        json_write(self.root / "inference.json", {"cap": cap, "fps": fps})
        json_write(self.root / "protocol.json", self.cfg["protocol"])
        json_write(self.root / "assets.lock.json", assets)
        json_write(
            self.root / "coverage.json",
            {
                "pose_audit": audit,
                "rest_frame": rest_frame,
                "rest_hash": digest(rest.tolist()),
                "outlier_limit_rad_sec": limits,
                "relation_mean_weight": weight.mean(0).tolist(),
                "train_median_length": {
                    corpus: float(
                        torch.tensor([len(c.q) for c in train if c.row["corpus"] == corpus])
                        .float()
                        .median()
                    )
                    for corpus in fps
                },
                "splits": {
                    name: [
                        {
                            "clip_id": c.id,
                            "frames_before": c.row["num_frames"],
                            "frames_after": len(c.q),
                            "q_missing": float((~c.q_valid).float().mean()),
                            "expr_missing": float((~c.expr_valid).float().mean()),
                            "nsa_unrepresented": c.unrepresented,
                        }
                        for c in clips
                    ]
                    for name, clips in subsets.items()
                },
            },
        )
        dependencies = versions()
        json_write(self.root / "dependencies.lock.json", dependencies)
        json_write(
            self.root / "run.json",
            {
                "signature": run_id,
                "config": self.cfg,
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "commit": git_revision(Path.cwd()),
                "dependencies": dependencies,
                "seed": self.seed,
                "parent_hashes": {
                    f: sha256(self.root / f)
                    for f in (
                        "splits.json",
                        "stats.npz",
                        "assets.lock.json",
                        "prepared.pt",
                        "inference.json",
                    )
                },
            },
        )

    @torch.no_grad()
    def pose_audit(self, clips):
        from slp.geometry import quat_to_matrix
        from .data import pack133

        selected = [c for c in clips if c.q_valid.all() and c.expr_valid.all()][:30]
        error = 0.0
        parity = []
        nsa_parity = []
        if self.profile != "smoke" and any(c.row["pose_layout"] != "nsa_named" for c in selected):
            from slp.evaluation.smplx_fk import SMPLXJoints

            reference = SMPLXJoints(
                str(self.device),
                self.cfg["assets"]["smplx"],
                self.cfg["data"].get("soke_stats"),
                load_stats=False,
            )
        for c in selected:
            ix = [0, len(c.q) // 2, len(c.q) - 1]
            q, e = c.q[ix].to(self.device), c.e[ix].to(self.device)
            recovered, _ = unpack133(pack133(q, e))
            error = max(
                error,
                float((quat_to_matrix(q) - quat_to_matrix(recovered)).norm(dim=(-1, -2)).max()),
            )
            if self.profile == "smoke":
                continue
            fk = self.geometry()
            layer_joints, _ = fk.decode(q, torch.zeros_like(e))
            expanded = layer_joints.index_select(1, fk.indices)
            if float((expanded - fk(q)).norm(dim=-1).max()) > 1e-4:
                raise ValueError("Compact relation FK failed SMPL-X landmark parity")
            raw_ix = [0, c.row["num_frames"] // 2, c.row["num_frames"] - 1]
            if c.row["pose_layout"] == "nsa_named":
                # Release-specific landmark arrays, already mapped into the codec
                # ordering and metres by the reference asset, can be audited here.
                path = c.row.get("nsa_reference_path")
                if path:
                    from .data import named_nsa

                    with np.load(c.row["pose_path"], allow_pickle=False) as original:
                        mapped = named_nsa(
                            original, c.row.get("nsa_mapping") or self.cfg["data"]["nsa_mapping"]
                        )
                    with np.load(path, allow_pickle=False) as original:
                        native = torch.as_tensor(original["landmarks"][raw_ix], device=self.device)
                    packed, _ = unpack133(
                        torch.as_tensor(mapped[raw_ix], device=self.device, dtype=q.dtype)
                    )
                    predicted = fk(packed)
                    if predicted.shape != native.shape:
                        raise ValueError(f"Invalid NSA parity fixture for {c.id}")
                    nsa_parity.append(float((predicted - native).norm(dim=-1).max()))
                continue
            # Use the stored parameters and native timebase at the scorer boundary.
            with np.load(c.row["pose_path"], allow_pickle=False) as original:
                raw = original["pose133"][raw_ix]
            packed_q, packed_e = unpack133(
                torch.as_tensor(raw, device=self.device, dtype=q.dtype),
                c.row["pose_layout"],
                self.cfg["data"].get("soke_stats"),
            )
            decoded = reference(raw, normalised=c.row["pose_layout"] == "soke133_norm")
            actual = fk.score_joints(packed_q, packed_e)
            parity.append(max(float((a - b).norm(dim=-1).max()) for a, b in zip(actual, decoded)))
        parity_error, nsa_error = max(parity, default=None), max(nsa_parity, default=None)
        if error > 1e-5 or any(v > 1e-4 for v in parity + nsa_parity):
            raise ValueError(f"Failed codec or FK parity: {error}, {parity_error}, {nsa_error}")
        return {
            "clips": len(selected),
            "frobenius": error,
            "soke_max_error_m": parity_error,
            "soke_clips": len(parity),
            "nsa_max_error_m": nsa_error,
            "nsa_clips": len(nsa_parity),
        }

    def open(self, *, need_train=True):
        run = json_read(self.root / "run.json")
        if run["config"] != self.cfg or run["seed"] != self.seed:
            raise ValueError("Resolved config or seed changed, use its prepared run")
        self.state = run
        if need_train:
            path = self.root / "prepared.pt"
            if sha256(path) != run["parent_hashes"]["prepared.pt"]:
                raise ValueError("Prepared cache checksum mismatch")
            self.data = torch.load(path, map_location="cpu", weights_only=False)
            self.stats = self.data["stats"]
        else:
            path = self.root / "stats.npz"
            if sha256(path) != run["parent_hashes"]["stats.npz"]:
                raise ValueError("Stats checksum mismatch")
            with np.load(path) as values:
                self.stats = {key: torch.as_tensor(values[key]) for key in values.files}
            # Runtime-only metadata is stored separately from the reference manifest.
            meta = self.root / "inference.json"
            if sha256(meta) != run["parent_hashes"]["inference.json"]:
                raise ValueError("Inference metadata checksum mismatch")
            self.stats.update(json_read(meta))
        return self

    def grid(self, length, stride=True):
        return phase_grid(
            length, self.cfg["planner"]["stride_frames"] if stride else 1, device=self.device
        )

    def fit(self, stage, model, clips, opt, weights, count, forward, val=None, parents=()):
        opt = {
            **opt,
            "checkpoint_every_steps": self.cfg["runtime"].get("checkpoint_every_steps", 1),
        }
        parent_hashes = {str(p.relative_to(self.root)): sha256(p) for p in parents}
        signature = {
            "run": self.state["signature"],
            "stage": stage,
            "options": opt,
            "ids_hash": digest([c.id for c in clips]),
            "parents": parent_hashes,
        }
        return run_stage(
            model,
            clips,
            opt,
            weights,
            count,
            forward,
            val,
            self.root / stage,
            signature,
            self.process,
            seed=self.seed,
            microbatch=self.cfg["runtime"]["microbatch"],
            resume=self.resume,
            multilingual=self.cfg["data"]["corpus"] == "multi",
            shuffle=self.cfg["data"]["shuffle_train"],
        )

    def make_planner(self, checkpoint=None):
        seed_all(self.seed)
        model = Planner(TextEncoder(self.cfg["text"], True), self.cfg["planner"]).to(self.device)
        return self.restore(model, checkpoint)

    def make_head(self, checkpoint=None):
        seed_all(self.seed)
        return self.restore(DirectHead(self.cfg["planner"]["width"]).to(self.device), checkpoint)

    def make_generator(self, checkpoint=None):
        seed_all(self.seed)
        model = Generator(TextEncoder(self.cfg["text"], False), self.cfg["generator"]).to(
            self.device
        )
        return self.restore(model, checkpoint)

    def make_correction(self, checkpoint=None):
        seed_all(self.seed)
        return self.restore(Correction(self.cfg["bridge"]).to(self.device), checkpoint)

    def restore(self, model, checkpoint):
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            lineage = saved["signature"]
            if lineage["run"] != self.state["signature"]:
                raise ValueError(f"Checkpoint is from another run: {checkpoint}")
            model.load_state_dict(saved["model"])
            model.eval()
        return model

    def planner_stage(self, clips, stage, dev=None, budget=None):
        model = self.make_planner()
        cfg = self.cfg["planner"]
        opt = settings(cfg)
        opt["patience"] = self.cfg["stopping"]["planner_patience_epochs"]
        if budget:
            opt["epochs"] = budget
        fk = self.geometry()
        local_cap = max(len(c.q) for c in clips)
        if stage == "planner":
            rest, scale = self.stats["rest"], self.stats["scale"]
            targets = {c.id: self.data["labels"][c.id] for c in clips}
        else:
            # Each OOF fold derives its thresholds, rest pose and scaler only
            # from the clips in its training partition.
            rest, _ = choose_rest(clips, self.seed, self.cfg["ik"]["rest_sample_limit"])
            with torch.no_grad():
                clip_copies = [replace(c) for c in clips]
                mark_outlier_relations(clip_copies)
                targets = {
                    c.id: tuple(
                        v.cpu() for v in label_at(c, self.grid(len(c.q)), fk, rest, plan_mask=True)
                    )
                    for c in clip_copies
                }
                values, masks = (torch.cat(parts) for parts in zip(*targets.values()))
                _, scale = scaler(values, masks)

        # Leave fold-dependent thresholds isolated from other runs.
        def weights_of(items):
            return torch.cat([targets[c.id][1] for c in items])

        def count(items):
            mask = weights_of(items)
            result = {
                name: (
                    mask[:, ids].reshape(-1, 4, 3).amin(-1).sum()
                    if name == "axes"
                    else mask[:, ids].sum()
                )
                for name, ids in GROUPS.items()
            }
            return {**result, "length": len(items)}

        def forward(items, epoch):
            lengths = torch.tensor([len(c.q) for c, _ in items], device=self.device)
            out = model([c.request(self.seed) for c, _ in items], local_cap, lengths)
            target = torch.cat([targets[c.id][0] for c, _ in items]).to(self.device)
            terms = relation_terms(
                out["relation"][out["valid"]],
                target,
                weights_of([c for c, _ in items]).to(self.device),
                scale.to(self.device),
            )
            terms["length"] = length_term(out, lengths)
            return {k: v[0] for k, v in terms.items()}, []

        def validate():
            totals = {k: [0.0, 0.0] for k in cfg["loss_weights"]}
            errors, rests, consistency, capped = [], [], [], []
            for c in dev:
                cap = self.stats["cap"][c.row["corpus"]]
                out = model([c.request(self.seed)], cap)
                n = int(out["length"][0])
                grid = self.grid(n)
                target, mask = label_at(c, grid, fk, rest)
                relations = out["relation"][0, out["valid"][0]]
                terms = relation_terms(relations, target, mask, scale.to(self.device))
                terms["length"] = length_term(out, torch.tensor([len(c.q)], device=self.device))
                for k, (num, count_value) in terms.items():
                    totals[k][0] += float(num)
                    totals[k][1] += float(count_value)
                errors.append(abs(n - len(c.q)) / c.fps)
                rest_r, _ = extract_relations(rest[None].to(self.device), None, fk)
                baseline = relation_terms(
                    rest_r.expand_as(target), target, mask, scale.to(self.device)
                )
                rests.append(
                    sum(
                        cfg["loss_weights"][k] * float(a / b)
                        for k, (a, b) in baseline.items()
                        if b > 0
                    )
                )
                consistency.append(
                    float((relations[:, 36:] - relations[:, 18:21] + relations[:, :3]).abs().mean())
                )
                capped.append(n == 1 or n == cap)
            per_group = {k: a / b for k, (a, b) in totals.items() if b > 0}
            score = sum(cfg["loss_weights"][k] * v for k, v in per_group.items())
            medians = {
                key: np.median([len(c.q) for c in clips if c.row["corpus"] == key])
                for key in self.stats["cap"]
            }
            return (score,), {
                "loss": score,
                "groups": per_group,
                "duration_mae": float(np.mean(errors)),
                "median_train_baseline_mae": float(
                    np.mean([abs(medians[c.row["corpus"]] - len(c.q)) / c.fps for c in dev])
                ),
                "rest_relation_loss": float(np.mean(rests)),
                "interwrist_consistency": float(np.mean(consistency)),
                "bound_rate": float(np.mean(capped)),
            }

        return self.fit(
            stage, model, clips, opt, cfg["loss_weights"], count, forward, validate if dev else None
        )

    def path(self, stage):
        return self.root / stage / "best.pt"

    def artifact(self, name, payload, parents=()):
        if self.process.rank == 0:
            path = self.root / name
            hashes = {str(p.relative_to(self.root)): sha256(p) for p in parents}
            previous = (
                torch.load(path, map_location="cpu", weights_only=True)
                if self.resume and path.is_file()
                else None
            )
            if (
                previous is None
                or previous["run"] != self.state["signature"]
                or previous["parents"] != hashes
            ):
                torch_write(
                    path, {"payload": payload, "run": self.state["signature"], "parents": hashes}
                )
            # Keep the published hash stable on resume so dependent checkpoints
            # keep working even when Torch would serialize tensor storage differently.
        self.process.barrier()

    def read_artifact(self, name):
        path = self.root / name
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry["run"] != self.state["signature"] or any(
            sha256(self.root / p) != h for p, h in entry["parents"].items()
        ):
            raise ValueError(f"Stale cache: {path}")
        return entry["payload"]

    def fit_planners(self, folds, fit_full):
        train, dev = (self.data["subsets"][key] for key in ("train", "dev"))
        if not fit_full:
            raise ValueError("OOF uses the epoch selected by a full-train planner; pass --fit-full")
        assignment = fold_assignment(train, folds, self.seed)
        if self.resume and (self.root / "plan_oof.pt").is_file():
            self.read_artifact("plan_oof.pt")
            if assignment != json_read(self.root / "fold_map.json")["folds"]:
                raise ValueError("The requested folds differ from the completed OOF artifact")
            print("Planner and held-out predictions already complete.", flush=True)
            return
        self.planner_stage(train, "planner", dev)
        best = torch.load(self.path("planner"), weights_only=True, map_location="cpu")
        epoch = best["epoch"]
        if self.process.rank == 0:
            json_write(
                self.root / "fold_map.json",
                {"folds": assignment, "epochs": epoch, "seed": self.seed},
            )
        parents = [self.path("planner"), self.root / "fold_map.json"]
        plans = {}
        for fold in range(folds):
            other = [c for c in train if assignment[c.id] != fold]
            stage = f"planner_fold_{fold}"
            model = self.planner_stage(other, stage, budget=epoch)
            fold_path = self.path(stage)
            parents.append(fold_path)
            caps = {
                k: max((len(c.q) for c in other if c.row["corpus"] == k), default=1)
                for k in self.stats["cap"]
            }
            held_out = {c.id: c for c in train if assignment[c.id] == fold}

            def predict(clip_id, planner=model):
                c = held_out[clip_id]
                pred = plan_for(planner, c.request(self.seed), caps[c.row["corpus"]])
                return {
                    **pred,
                    "relation": pred["relation"].cpu(),
                    "fold": fold,
                    "text_hash": digest(c.row["text"]),
                }

            with torch.no_grad():
                if self.process.world == 1:
                    binding = digest([self.state["signature"], fold, sha256(fold_path)])
                    plans.update(
                        checkpointed_records(
                            self.root / "work" / stage / binding,
                            list(held_out),
                            predict,
                            batch_size=32,
                            resume=self.resume,
                            label=f"{stage} held-out predictions",
                        )
                    )
                else:
                    plans.update({clip_id: predict(clip_id) for clip_id in held_out})
            # Release this encoder before the next fold allocates another.
            del predict, model
        self.artifact("plan_oof.pt", plans, parents)

    def fit_head(self, folds, fit_full):
        if not fit_full:
            raise ValueError("Direct-head cross fitting requires --fit-full")
        train, dev = (self.data["subsets"][key] for key in ("train", "dev"))
        assignment = json_read(self.root / "fold_map.json")["folds"]
        if len(set(assignment.values())) != folds:
            raise ValueError("Direct head must use the planner fold map")
        oof = self.read_artifact("plan_oof.pt")
        parents = [self.root / "plan_oof.pt"]

        def train_head(clipset, stage, planner_stage, valid=None, budget=None):
            planner = self.make_planner(self.path(planner_stage)).requires_grad_(False)
            head = self.make_head()
            opt = settings(self.cfg["direct_pose_head"])
            opt["patience"] = self.cfg["direct_pose_head"]["patience_epochs"]
            if budget:
                opt["epochs"] = budget
            cache = {}

            def target(c, n):
                return resample(
                    c.q.to(self.device),
                    self.grid(n),
                    quaternion=True,
                    valid=c.q_valid.to(self.device),
                )

            with torch.no_grad():
                for c in clipset:
                    n = len(c.q)
                    out = planner([c.request(self.seed)], n, torch.tensor([n], device=self.device))
                    cache[c.id] = tuple(
                        v.cpu() for v in (out["hidden"][0, out["valid"][0]], *target(c, n))
                    )

            def count(items):
                return {"pose": sum(int(cache[c.id][2].sum()) for c in items)}

            def forward(items, epoch):
                values = []
                for c, _ in items:
                    hidden, q, mask = (v.to(self.device) for v in cache[c.id])
                    values.append(((1 - (head(hidden) * q).sum(-1).square()) * mask).sum())
                return {"pose": torch.stack(values).sum()}, []

            def validate():
                nums = count_total = 0.0
                for c in valid:
                    out = planner([c.request(self.seed)], self.stats["cap"][c.row["corpus"]])
                    q, mask = target(c, int(out["length"][0]))
                    pred = head(out["hidden"][0, out["valid"][0]])
                    nums += float(((1 - (q * pred).sum(-1).square()) * mask).sum())
                    count_total += int(mask.sum())
                loss = nums / count_total
                return (loss,), {"pose": loss}

            self.fit(
                stage,
                head,
                clipset,
                opt,
                {"pose": 1.0},
                count,
                forward,
                validate if valid else None,
                [self.path(planner_stage)],
            )
            return planner, head

        train_head(train, "direct", "planner", dev)
        best = torch.load(self.path("direct"), map_location="cpu", weights_only=True)
        output = {}
        parents.append(self.path("direct"))
        for fold in range(folds):
            others = [c for c in train if assignment[c.id] != fold]
            stage = f"direct_fold_{fold}"
            planner, head = train_head(others, stage, f"planner_fold_{fold}", budget=best["epoch"])
            parents.append(self.path(stage))
            with torch.no_grad():
                for c in train:
                    if assignment[c.id] == fold:
                        pred = plan_for(
                            planner,
                            c.request(self.seed),
                            oof[c.id]["length"],
                            head,
                            length=oof[c.id]["length"],
                        )
                        output[c.id] = pred["direct"].cpu()
        self.artifact("direct_oof.pt", output, parents)

    def fit_bank(self):
        if self.process.rank == 0:
            bank = build_bank(
                self.data["subsets"]["train"],
                self.cfg["bridge"],
                self.seed,
                checkpoint_dir=self.root / "work" / "bank" / self.state["signature"],
                resume=self.resume,
            )
            npz_write(self.root / "bank.npz", **{k: v.numpy() for k, v in bank.items()})
            json_write(
                self.root / "bank.json",
                {
                    "sha256": sha256(self.root / "bank.npz"),
                    "observations": bank["counts"].sum(-1).tolist(),
                    "effective": (bank["counts"] > 0).sum(-1).tolist(),
                },
            )
        self.process.barrier()

    def plans(self, arm):
        train, dev = (self.data["subsets"][key] for key in ("train", "dev"))
        plans = self.read_artifact("plan_oof.pt")
        planner = self.make_planner(self.path("planner"))
        head = self.make_head(self.path("direct")) if arm == "D" else None
        if head is not None:
            direct = self.read_artifact("direct_oof.pt")
            for c in train:
                plans[c.id]["direct"] = direct[c.id]
        with torch.no_grad():
            for c in dev:
                plan = plan_for(
                    planner, c.request(self.seed), self.stats["cap"][c.row["corpus"]], head
                )
                plans[c.id] = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in plan.items()
                }
        return plans

    def build_sources(self, arms):
        if any(arm.startswith("F-") for arm in arms):
            raise ValueError("F is a dev-only sampling diagnostic")
        if "E-shuffle" in arms and "C" not in arms and not (self.root / "donors.pt").is_file():
            arms = ["C", *arms]
        if "C" in arms:
            arms = ["C", *(a for a in arms if a != "C")]
        clips = list(itertools.chain.from_iterable(self.data["subsets"].values()))
        train = self.data["subsets"]["train"]
        for arm in dict.fromkeys(arms):
            if self.process.rank != 0:
                self.process.barrier()
                if arm == "C":
                    self.process.barrier()
                continue
            plans = self.plans(arm)
            parents = [self.root / "plan_oof.pt", self.path("planner")]
            bank = None
            if arm == "D":
                parents += [self.root / "direct_oof.pt", self.path("direct")]
            if arm == "E-shuffle":
                bank = self.read_artifact("donors.pt")
                parents.append(self.root / "donors.pt")
            clips_by_id = {c.id: c for c in clips}

            def compute(clip_id):
                c = clips_by_id[clip_id]
                plan = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in plans[c.id].items()
                }
                center, status = build_center(
                    arm,
                    plan,
                    self.stats["rest"],
                    self.cfg,
                    self.geometry(),
                    self.stats["scale"],
                    donor_bank=bank,
                    request=c.request(self.seed),
                )
                return {
                    **plans[c.id],
                    "center": center.cpu() if center is not None else None,
                    "diagnostics": status,
                }

            if arm == "C":
                binding = digest([self.state["signature"], [sha256(path) for path in parents]])
                rows = checkpointed_records(
                    self.root / "work" / "sources_C" / binding,
                    list(clips_by_id),
                    compute,
                    batch_size=16,
                    resume=self.resume,
                    label="IK sources",
                )
            else:
                rows = {clip_id: compute(clip_id) for clip_id in clips_by_id}
            self.artifact(f"sources_{arm}.pt", rows, parents)
            if arm == "C":
                # Donors for diagnostics and retraining are sourced entirely from train OOF.
                donors = {
                    c.id: {
                        "center": rows[c.id]["center"],
                        "length": rows[c.id]["length"],
                        "sign": c.row["target_sign_language"],
                        "text_hash": digest(c.row["text"]),
                    }
                    for c in train
                }
                self.artifact("donors.pt", donors, [self.root / "sources_C.pt"])

    def motion_data(self, items, epoch, arm, sources):
        clips = [c for c, _ in items]
        batch = collate(clips, self.device)
        r, poses, expr, times = [], [], [], []
        for c, draw_id in items:
            record = sources[c.id]
            phase = self.grid(len(c.q), stride=False)
            r.append(dense_relations(record["relation"].to(self.device), phase))
            center = record["center"]
            if center is not None:
                center = resample(center.to(self.device), phase, quaternion=True)
            req = c.request(self.seed)
            draw = ("dev",) if epoch is None else ("train", epoch, draw_id)
            q0, e0 = sample_source(
                arm,
                center,
                len(c.q),
                self.cfg["source"],
                req,
                c.row["corpus"],
                device=self.device,
                dtype=batch["q"].dtype,
                draw=draw,
            )
            t_rng = generator(self.device, self.seed, c.row["corpus"], c.id, "time", *draw)
            times.append(
                torch.rand((1, 1, 1), device=self.device, generator=t_rng).clamp(1e-6, 1 - 1e-6)
            )
            poses.append(pad(q0, (0, 0, 0, 0, 0, batch["q"].shape[1] - len(c.q))))
            expr.append(e0)
        q0 = torch.stack(poses)
        q0 = torch.where(batch["frame_valid"][..., None, None], q0, identity_like(q0))
        e0, r = [torch.nn.utils.rnn.pad_sequence(x, batch_first=True) for x in (expr, r)]
        return batch, q0, e0, torch.stack(times), conditioning(arm, r, self.stats, self.cfg)

    def bank(self):
        path = self.root / "bank.npz"
        if sha256(path) != json_read(self.root / "bank.json")["sha256"]:
            raise ValueError("Prototype bank checksum mismatch")
        with np.load(path) as data:
            return {k: torch.from_numpy(data[k]).to(self.device) for k in data.files}

    def correction_stage(self, arm):
        if self.cfg["bridge"]["mode"] == "geodesic":
            return
        cfg = self.cfg["bridge"]
        bank = self.bank()
        sources = self.read_artifact(f"sources_{arm}.pt")
        stage = correction_stage_name(arm)
        train, dev = (self.data["subsets"][key] for key in ("train", "dev"))
        model = self.make_correction()
        opt = {
            "epochs": cfg["correction_epochs"],
            "batch": cfg["correction_batch_effective"],
            "lr": cfg["correction_lr"],
            "decay": cfg["correction_weight_decay"],
            "warmup": cfg["correction_warmup_fraction"],
        }

        def count(items):
            return {"energy": sum(int(c.q_valid.sum()) for c in items)}

        def forward(items, epoch):
            batch, q0, _, time, _ = self.motion_data(items, epoch, arm, sources)
            loss, logs = correction_loss(q0, batch["q"], time, model, bank, cfg, batch["q_valid"])
            return {"energy": loss}, [{"clips": [c.id for c, _ in items], **logs}]

        def validation():
            measured = {"learned": 0.0, "geodesic": 0.0}
            n = 0
            for c in dev:
                batch, q0, _, time, _ = self.motion_data([(c, 0)], None, arm, sources)
                for label, network in (("learned", model), ("geodesic", None)):
                    value, log = correction_loss(
                        q0, batch["q"], time, network, bank, cfg, batch["q_valid"]
                    )
                    measured[label] += float(value)
                n += log["valid"]
            results = {k: v / n for k, v in measured.items()}
            return (results["learned"],), results

        self.fit(
            stage,
            model,
            train,
            opt,
            {"energy": 1.0},
            count,
            forward,
            validation,
            [self.root / f"sources_{arm}.pt", self.root / "bank.npz"],
        )

    def correction(self, arm):
        if self.cfg["bridge"]["mode"] == "geodesic":
            return None
        return self.make_correction(self.path(correction_stage_name(arm))).requires_grad_(False)

    def generator_stage(self, arm):
        from .evaluation import locked_protocol, score_clip, semantics
        from slp.evaluation.flow_matching import corpus_bleu4

        train, dev = (self.data["subsets"][key] for key in ("train", "dev"))
        cfg = self.cfg["generator"]
        protocol = locked_protocol(self.cfg["protocol"])
        sources = self.read_artifact(f"sources_{arm}.pt")
        current = self.correction(arm)
        model = self.make_generator()
        planner = self.make_planner(self.path("planner"))
        parents = [self.path("planner"), self.root / f"sources_{arm}.pt"]
        if current is not None:
            parents.append(self.path(correction_stage_name(arm)))
        head = self.make_head(self.path("direct")) if arm == "D" else None
        donors = self.read_artifact("donors.pt") if arm == "E-shuffle" else None
        opt = settings(
            {
                **cfg,
                "batch_effective": cfg.get(
                    "batch_effective", cfg["batch_effective_reference_4gpu"]
                ),
            }
        )
        opt.update(
            clip=cfg["grad_clip_norm_matched"],
            patience=self.cfg["stopping"]["gen_matched_patience_evals"],
            val_period=self.cfg["evaluation"]["validation_period_epochs"],
        )
        weights = {"pose": cfg["pose_weight"], "expression": cfg["expression_weight"]}
        if cfg["aux_relation_weight"]:
            weights.update(
                {
                    name: cfg["aux_relation_weight"] * self.cfg["planner"]["loss_weights"][name]
                    for name in GROUPS
                }
            )
        if cfg["aux_velocity_weight"]:
            weights["velocity"] = cfg["aux_velocity_weight"]
        fk = self.geometry()
        e_mean = self.stats["expr_mean"].to(self.device)
        e_scale = self.stats["expr_scale"].to(self.device)
        aux = {}
        if cfg["aux_relation_weight"] or cfg["aux_velocity_weight"]:
            with torch.no_grad():
                for c in train:
                    q, mask = c.q.to(self.device), c.q_valid.to(self.device)
                    r, weight = label_at(
                        c, self.grid(len(q), False), fk, self.stats["rest"], plan_mask=True
                    )
                    tips, valid = wrist_tips(fk(q))
                    # The full pose mask for wrist/tip velocity intersects both frames.
                    valid &= mask.all(-1)
                    aux[c.id] = tuple(
                        v.cpu()
                        for v in (r, weight, tips.diff(dim=0) * c.fps, valid[1:] & valid[:-1])
                    )

        def count(clips):
            counts = {
                "pose": sum(4 * int(c.q_valid.sum()) for c in clips),
                "expression": sum(int(c.expr_valid.sum()) for c in clips),
            }
            if cfg["aux_relation_weight"]:
                w = torch.cat([aux[c.id][1] for c in clips])
                counts.update(
                    {
                        k: w[:, ids].reshape(-1, 4, 3).amin(-1).sum()
                        if k == "axes"
                        else w[:, ids].sum()
                        for k, ids in GROUPS.items()
                    }
                )
            if cfg["aux_velocity_weight"]:
                counts["velocity"] = sum(int(aux[c.id][3].sum()) * 36 for c in clips)
            return counts

        def forward(items, epoch):
            with torch.no_grad():
                batch, q0, e0, time, relation = self.motion_data(items, epoch, arm, sources)
                xt = bridge(q0, batch["q"], time, current)
                e1 = (batch["e"] - e_mean) / e_scale
                et = torch.lerp(e0, e1, time[..., 0])
            context = model.encode([c.request(self.seed) for c, _ in items])
            q, e = model.predict_target(
                xt, et, time, context, relation, batch["frame_valid"], sign_enabled=arm != "A-paper"
            )
            terms = {
                "pose": ((q - batch["q"]).square() * batch["q_valid"][..., None]).sum(),
                "expression": ((e - e1).square() * batch["expr_valid"]).sum(),
            }
            if aux:
                from .relations import describe
                from torch.nn.functional import smooth_l1_loss

                for i, (c, _) in enumerate(items):
                    r, weight, speed, valid = (v.to(self.device) for v in aux[c.id])
                    points = fk(q[i, : len(c.q)])
                    if cfg["aux_relation_weight"]:
                        pred, _ = describe(points)
                        rel = relation_terms(pred, r, weight, self.stats["scale"].to(q))
                        for name, (num, _) in rel.items():
                            terms[name] = terms.get(name, 0) + num
                    if cfg["aux_velocity_weight"]:
                        tips, _ = wrist_tips(points)
                        diff = (tips.diff(dim=0) * c.fps - speed) / self.stats["velocity_scale"].to(
                            q
                        )
                        terms["velocity"] = terms.get("velocity", 0) + smooth_l1_loss(
                            diff[valid], torch.zeros_like(diff[valid]), reduction="sum"
                        )
            return terms, [{"degenerate_quaternions": int(model.last_degenerate)}]

        def validation():
            samples = Sampler(
                self.cfg, planner, model, self.stats, current, fk, arm, head=head, donor_bank=donors
            )
            by_corpus = {}
            for c in dev:
                pred = samples.generate(c.request(self.seed))
                metrics = score_clip(pred, c, fk, protocol)
                metrics.update(semantics(pred, c, protocol, metric="bt", fk=fk))
                by_corpus.setdefault(c.row["corpus"], []).append(metrics)
            result = {}
            for corpus, rows in by_corpus.items():
                hand_pa = float(np.mean([r["hand_dtw_pa_jpe"] for r in rows]))
                hand_raw = float(np.mean([r["hand_dtw_jpe"] for r in rows]))
                bt = corpus_bleu4(rows) if all("bt_hypothesis_tokens" in r for r in rows) else None
                result[corpus] = {"hand_pa": hand_pa, "hand_raw": hand_raw, "bt_bleu4": bt}
            pa = float(np.mean([m["hand_pa"] for m in result.values()]))
            raw = float(np.mean([m["hand_raw"] for m in result.values()]))
            scored = all(m["bt_bleu4"] is not None for m in result.values())
            key = (
                (10 * pa + 0.1 * raw - np.mean([m["bt_bleu4"] for m in result.values()]),)
                if scored
                else (pa, raw)
            )
            return key, {
                "corpora": result,
                "selection": "matched" if scored else "geometry_exploratory",
            }

        self.fit(
            f"generator_{arm}", model, train, opt, weights, count, forward, validation, parents
        )

    def sampler(self, arm, weights_from=None):
        weights_from = weights_from or arm
        generator_path = self.path(f"generator_{weights_from}")
        planner_path = self.path("planner")
        model = self.make_generator(generator_path)
        planner = self.make_planner(planner_path)
        head = self.make_head(self.path("direct")) if arm == "D" else None
        donors = self.read_artifact("donors.pt") if arm == "E-shuffle" else None
        fk = self.geometry() if arm == "C" or arm.startswith("F-") else None
        version = digest(
            [
                self.state["signature"],
                sha256(planner_path),
                sha256(generator_path),
                arm,
                sha256(self.path(correction_stage_name(weights_from)))
                if self.cfg["bridge"]["mode"] == "naturalness"
                else None,
                sha256(self.path("direct")) if head is not None else None,
                sha256(self.root / "donors.pt") if donors is not None else None,
            ]
        )
        return Sampler(
            self.cfg,
            planner,
            model,
            self.stats,
            self.correction(weights_from),
            fk,
            arm,
            donor_bank=donors,
            head=head,
            version=version,
        )


def correction_stage_name(arm):
    return "correction_uniform" if arm in UNIFORM else f"correction_{arm}"

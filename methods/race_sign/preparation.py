"""Offline training preparation. Alignment is never reachable from inference."""

from pathlib import Path
import json
import numpy as np
import torch
from .adapters import file_hash
from .data import TRAIN_ROLES, RecordDataset, collate_inputs, validate_official_split
from .schema import manual_to_6d
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle


@torch.no_grad()
def align_training_target(target, baseline, geometry, *, role, max_ratio=2.0):
    if role not in TRAIN_ROLES:
        raise ValueError("Reference alignment is training-only")
    a, b = len(baseline), len(target)
    if min(a, b) == 0 or max(a / b, b / a) > max_ratio:
        raise ValueError("Excessive length mismatch; refusing to invent semantic correspondence")
    device = next(geometry.buffers()).device
    x = geometry(manual_to_6d(baseline.to(device), geometry.manual_mean)[None])
    y = geometry(manual_to_6d(target.to(device), geometry.manual_mean)[None])
    cost = torch.zeros(a, b, device=device)
    for p, q in ((x.body, y.body), (x.left, y.left), (x.right, y.right)):
        # Joint-wise distances in a single body coordinate system, equal weighting per part.
        cost += torch.cdist(p[0].flatten(1), q[0].flatten(1)) / np.sqrt(p.shape[-2])
    cost = cost.cpu().numpy()
    dp = np.full((a + 1, b + 1), np.inf)
    dp[0, 0] = 0
    steps = np.zeros((a, b), dtype=np.int8)
    for i in range(a):
        for j in range(b):
            choices = (dp[i, j], dp[i, j + 1], dp[i + 1, j])
            k = int(np.argmin(choices))
            dp[i + 1, j + 1] = cost[i, j] + choices[k]
            steps[i, j] = k
    aligned = [[] for _ in range(a)]
    i, j = a - 1, b - 1
    while True:
        aligned[i].append(j)
        if i == 0 and j == 0:
            break
        k = steps[i, j]
        if k == 0:
            i -= 1
            j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    pose_mean = torch.cat((geometry.manual_mean.to(target), target.new_zeros(3)))
    rotations = axis_angle_to_matrix((target[..., :123] + pose_mean).reshape(b, 41, 3))
    poses = []
    for indices in aligned:
        average = rotations[indices].mean(0)
        u, _, vh = torch.linalg.svd(average)
        sign = torch.linalg.det(u @ vh)
        fix = torch.ones(41, 3, device=target.device, dtype=target.dtype)
        fix[:, -1] = sign
        r = (u * fix[:, None, :]) @ vh
        poses.append(
            torch.cat(
                (matrix_to_axis_angle(r).flatten() - pose_mean, target[indices, 123:].mean(0))
            )
        )
    return torch.stack(poses)


@torch.no_grad()
def corrupt_tokens(streams, modules, probability, seed=0, *, repeat_probability=0.0, repeat_span=2):
    if not 0 <= probability <= 1 or not 0 <= repeat_probability <= 1 or repeat_span < 1:
        raise ValueError("Probability must be in [0,1]")
    rng = torch.Generator(device=streams[0].device).manual_seed(seed)
    out = []
    for ids, module in zip(streams, modules):
        book = module.quantizer.codebook
        distance = torch.cdist(book.float(), book.float())
        distance.fill_diagonal_(float("inf"))
        nearest = distance.argmin(-1)
        replace = torch.rand(ids.shape, device=ids.device, generator=rng) < probability
        corrupted = torch.where(replace, nearest[ids], ids)
        if repeat_probability and len(ids) > 1:
            original = corrupted.clone()
            repeat = torch.rand(ids.shape, device=ids.device, generator=rng) < repeat_probability
            for start in torch.nonzero(repeat, as_tuple=False).flatten().tolist():
                if start > 0:
                    corrupted[start : min(start + repeat_span, len(ids))] = original[start - 1]
        out.append(corrupted)
    return tuple(out)


def write_record(directory, record, index):
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    name = f"{index:08d}.pt"
    torch.save(record, p / name)
    return {
        "path": name,
        "sample_id": record["sample_id"],
        "source_group": record["source_group"],
        "role": record["role"],
        "official_split": record["official_split"],
    }


def prepare_manifest(
    raw_manifest,
    adapter,
    output,
    role,
    *,
    mode="teacher",
    geometry=None,
    align=False,
    corruption=0.0,
    repeat_probability=0.0,
    part_dropout=0.0,
    seed=0,
):
    if role not in TRAIN_ROLES:
        raise ValueError("Preparation requires an explicit training/development role")
    if mode not in ("teacher", "generated") or not all(
        0 <= p <= 1 for p in (corruption, repeat_probability, part_dropout)
    ):
        raise ValueError("Invalid preparation mode/corruption probability")
    if role.startswith("selector_") and mode != "generated":
        raise ValueError(
            "Selector development must use generated tokens, not oracle teacher tokens"
        )
    if role in ("decoder_train", "decoder_val") and mode != "teacher":
        raise ValueError("Stage A decoder data must use teacher tokens")
    if (corruption or repeat_probability or part_dropout) and role != "robustness_train":
        raise ValueError("Synthetic code corruption belongs to the explicit robustness stage")
    if (corruption or repeat_probability or part_dropout) and mode != "teacher":
        raise ValueError(
            "Synthetic corruption and generated-token tuning are separate preparations"
        )
    raw = Path(raw_manifest).resolve()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
    manifest = []
    seen = set()
    for i, row in enumerate(rows):
        if row.get("role") != role or not row.get("source_group") or row["sample_id"] in seen:
            raise ValueError("Missing/mismatched role, duplicate ID or absent video provenance")
        seen.add(row["sample_id"])
        validate_official_split(role, row.get("official_split"))
        if role == "robustness_train" and mode == "generated":
            known = getattr(adapter, "generator_train_groups", None)
            if known is None or row["source_group"] in known:
                raise ValueError(
                    "Generated-token tuning requires declared, held-out generator source groups"
                )
        target = torch.from_numpy(np.load(raw.parent / row["pose"], allow_pickle=False)).float()
        if target.ndim != 2 or target.shape[-1] != 133 or not torch.isfinite(target).all():
            raise ValueError("Expected finite raw physical [T,133] poses")
        if mode == "teacher":
            record = adapter.prepare(
                row["text"], row["language"], row["sample_id"], target=target, mode="teacher"
            )
            target = target[: len(record["baseline"])]
            if corruption or repeat_probability:
                streams = adapter.encode_pose(target)
                streams = corrupt_tokens(
                    streams,
                    (adapter.body, adapter.left, adapter.right),
                    corruption,
                    seed + i,
                    repeat_probability=repeat_probability,
                )
                base, codes = adapter.decode_tokens(streams)
                record.update(baseline=base.cpu(), codes=codes.cpu())
                record["provenance"]["corruption_probability"] = corruption
                record["provenance"]["repeat_probability"] = repeat_probability
            if part_dropout:
                rng = torch.Generator().manual_seed(seed + i + 300000)
                dropped = torch.rand(3, generator=rng) < part_dropout
                record["codes"][:, dropped] = 0
                record["provenance"]["dropped_code_parts"] = dropped.tolist()
                record["provenance"]["part_dropout_probability"] = part_dropout
        else:
            record = adapter.prepare(
                row["text"],
                row["language"],
                row["sample_id"],
                mode="generated",
                generator_seed=seed + i,
            )
            if not align:
                raise ValueError(
                    "Generated training pairs require explicit --align-target acknowledgement"
                )
            if geometry is None:
                raise ValueError("Actual geometry required for training correspondence")
            target = align_training_target(target, record["baseline"], geometry, role=role)
            record["provenance"]["training_alignment"] = "DTW_proxy_not_semantic_ground_truth"
        record.update(
            target=target.cpu(),
            role=role,
            source_group=row["source_group"],
            official_split=row["official_split"],
        )
        record["provenance"].update(
            raw_manifest_sha256=file_hash(raw),
            target_pose_sha256=file_hash(raw.parent / row["pose"]),
        )
        if role == "robustness_train" and mode == "generated":
            record["provenance"]["generator_holdout_groups_sha256"] = (
                adapter.generator_train_groups_sha256
            )
        manifest.append(write_record(out, record, i))
    path = out / "manifest.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    return path


@torch.no_grad()
def cache_candidates(
    checkpoint,
    manifest,
    role,
    geometry,
    output,
    *,
    seed=0,
    steps=None,
    device="cpu",
    use_text=None,
    coupled=None,
):
    from .training import load_checkpoint
    from .pipeline import RacePipeline

    dataset = RecordDataset(manifest, role)
    if role not in ("selector_train", "selector_val", "selector_calibration"):
        raise ValueError("Candidate cache is restricted to selector data roles")
    decoder, norm, _, ckpt = load_checkpoint(checkpoint, device)
    use_text = decoder.default_use_text if use_text is None else use_text
    coupled = decoder.default_coupled if coupled is None else coupled
    prior = set(ckpt["provenance"].get("decoder_train_groups", [])) | set(
        ckpt["provenance"].get("decoder_val_groups", [])
    )
    if prior & dataset.groups:
        raise ValueError("Selector candidates must be held out from decoder development")
    pipe = RacePipeline(decoder, norm, geometry.to(device)).eval()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(dataset)):
        record = dataset[i]
        # Target deliberately excluded from the inference dataclass.
        prediction = pipe.predict(
            collate_inputs([record]).to(device),
            policy="flow",
            seed=seed,
            steps=steps,
            use_text=use_text,
            coupled=coupled,
        )
        record["candidate"] = prediction.poses[0, : prediction.lengths[0]].cpu()
        record["provenance"] = dict(
            record["provenance"],
            candidate_decoder_sha256=file_hash(checkpoint),
            candidate_seed=seed,
            candidate_steps=steps or decoder.cfg.flow_steps,
            candidate_use_text=use_text,
            candidate_coupled=coupled,
        )
        rows.append(write_record(out, record, i))
    path = out / "manifest.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path

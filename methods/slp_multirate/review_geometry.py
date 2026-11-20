"""Export a development clip and a same-rig body-grid reconstruction for visual audit."""

from pathlib import Path

import torch

from .anchors import clip_pose
from .data import CanonicalDataset, collate_clips, save_npz, write_json
from .geometry import JOINTS, build_body_grid, fk, merge_local, rot6d_to_matrix, upsample_body


def review_geometry(manifest, rig, clip_id, out, stride, render=False):
    dataset = CanonicalDataset(manifest, rig, ("train", "dev_tune", "dev_cal"))
    index = next((i for i, row in enumerate(dataset.rows) if row["clip_id"] == clip_id), None)
    if index is None:
        raise ValueError("The requested clip is absent from train/dev_tune/dev_cal")
    clip = dataset[index]
    batch = collate_clips([clip])
    ref = clip_pose(clip, rig)
    grid = build_body_grid(batch["body_rot6d"], batch["timestamps_s"], batch["frame_mask"],
                           batch["body_obs"], stride)
    body, visible = upsample_body(grid.rotations, grid.timestamps, grid.mask, batch["timestamps_s"],
                                 batch["frame_mask"], grid.observation)
    hands = rot6d_to_matrix(clip["hands_rot6d"])
    seen = torch.cat((visible[0], clip["hands_obs"].flatten(1, 2)), -1)
    decoded = fk(merge_local(body[0], hands), rig, seen)
    out = Path(out)
    save_npz(out / "geometry.npz", joint_names=JOINTS,
             reference_joints_m=ref.joints.numpy(), reconstructed_joints_m=decoded.joints.numpy(),
             reference_tips_m=ref.tips.numpy(), reconstructed_tips_m=decoded.tips.numpy(),
             timestamps_s=clip["timestamps_s"].numpy(), joint_mask=ref.joint_mask.numpy())
    common = ref.joint_mask & decoded.joint_mask
    error = (ref.joints - decoded.joints).norm(dim=-1)[common]
    result = {"clip_id": clip_id, "frames": len(ref.joints), "stride": stride,
              "rig_sha256": rig.sha256, "source_joint_schema": clip["row"]["source_joint_schema"],
              "max_common_joint_mm": float(error.max() * 1000) if len(error) else None,
              "visible_joint_frames": int(common.sum()), "geometry": str(out / "geometry.npz")}
    if render:
        _render(ref, decoded, rig, out / "geometry.png")
        result["render"] = str(out / "geometry.png")
    write_json(out / "geometry.json", result)
    return result


def _render(reference, decoded, rig, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    combined = torch.cat((reference.joints, decoded.joints))
    lo, hi = combined.amin(dim=(0, 1)), combined.amax(dim=(0, 1))
    center = (lo + hi) / 2
    radius = max(.1, float((hi - lo).max() / 2))
    frames = [0, len(reference.joints) // 2, len(reference.joints) - 1]
    for i, frame in enumerate(frames, 1):
        axis = fig.add_subplot(1, 3, i, projection="3d")
        for data, color, style, label in (
            (reference, "#00768c", "-", "Canonical reference"),
            (decoded, "#ba5020", "--", "Coarse reconstruction"),
        ):
            drawn = False
            for j, p in enumerate(rig.parents):
                if p < 0 or not (data.joint_mask[frame, j] & data.joint_mask[frame, p]):
                    continue
                xyz = data.joints[frame, [j, p]].T.numpy()
                axis.plot(*xyz, color=color, ls=style, label=label if not drawn else None)
                drawn = True
        axis.set(xlim=(center[0]-radius, center[0]+radius),
                 ylim=(center[1]-radius, center[1]+radius),
                 zlim=(center[2]-radius, center[2]+radius), title=f"Frame {frame}")
        axis.set_box_aspect((1, 1, 1))
        if i == 1:
            axis.legend(loc="upper left", fontsize=6)
    fig.savefig(output, dpi=150)

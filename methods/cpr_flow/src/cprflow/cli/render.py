"""Asset-backed neutral skeleton preview; clearly labeled, no physical collision claims."""

import argparse
from pathlib import Path

import numpy as np
import torch


def render_motion(motion, body_model, output, fps=30.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from cprflow.geometry.joint_maps import FINGERS

    with torch.no_grad():
        geo = body_model(motion)
    joints = geo["joints"].detach().cpu().numpy()
    names = geo["names"]
    edges = []
    for side in ("left", "right"):
        edges.extend([(f"{side}_shoulder", f"{side}_elbow"), (f"{side}_elbow", f"{side}_wrist")])
        for finger in FINGERS:
            sequence = (
                [f"{side}_wrist"]
                + [f"{side}_{finger}{j}" for j in (1, 2, 3)]
                + [f"{side}_{finger}"]
            )
            edges.extend(zip(sequence[:-1], sequence[1:]))
    edges.extend([("left_shoulder", "right_shoulder"), ("neck", "head"), ("head", "jaw")])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    center = joints[:, names.index("neck")].mean(0)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    with imageio.get_writer(
        output, format="FFMPEG", codec="libx264", fps=fps, macro_block_size=1
    ) as writer:
        for frame in joints:
            ax.clear()
            for a, b in edges:
                points = frame[[names.index(a), names.index(b)]]
                ax.plot(
                    points[:, 0],
                    points[:, 2],
                    points[:, 1],
                    color="blue" if a.startswith("left") else "orange",
                    linewidth=1,
                )
            ax.set(
                xlim=(center[0] - 0.8, center[0] + 0.8),
                ylim=(center[2] - 0.8, center[2] + 0.8),
                zlim=(center[1] - 0.8, center[1] + 0.8),
                title="Native SMPL-X skeleton diagnostic",
            )
            ax.view_init(elev=5, azim=-90)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--smplx-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    from cprflow.geometry.body_model import BodyModel

    with np.load(args.motion, allow_pickle=False) as data:
        x = torch.from_numpy(data["motion"])
    print(render_motion(x, BodyModel(args.smplx_model), args.output, args.fps))


if __name__ == "__main__":
    main()

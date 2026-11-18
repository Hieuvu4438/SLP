"""No-training SMPL-X convention / differential audit and manifest creation.

python -m methods.coplan_so3.audit --write-manifest PATH --fps 25
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from slp.evaluation.smplx_fk import SOKE_SHAPE, SMPLXJoints, UPPER_BODY_IDX
from slp.skeleton_fk import FEAT_TO_SMPLX, SkeletonFK

from .data.manifest import MotionManifest, file_sha256
from .data.smplx_adapter import SOKECodec
from .geometry.fk import SOKEFK
from .geometry.physical_time_features import SequenceFeatures
from .geometry.pullback import feature_jvp
from .geometry import so3

ROOT = Path(__file__).resolve().parents[2]
MODEL = "third_party/SOKE/deps/smpl_models/smplx/SMPLX_NEUTRAL.npz"
STATS = "third_party/SOKE/deps/stats"


def soke_manifest(fps=25.0):
    from smplx.joint_names import JOINT_NAMES

    left = tuple(range(10, 25))
    right = tuple(range(25, 40))
    return MotionManifest(
        tuple(JOINT_NAMES[i] for i in FEAT_TO_SMPLX),
        tuple(FEAT_TO_SMPLX),
        tuple(range(10)) + (40,),
        left,
        right,
        tuple(tuple(range(i, i + 3)) for i in range(10, 25, 3)),
        tuple(tuple(range(i, i + 3)) for i in range(25, 40, 3)),
        10,
        fps,
        "soke_133_shared_evaluator",
        "SOKE_133_layout;actual_clip_extraction_record_required",
        "official_split;actual_annotation_record_required",
        "SOKE_SHAPE:" + json.dumps(SOKE_SHAPE),
        "physical=axis_angle+smplx_hand_mean",
        "root_lower_body_eyes_identity;no_translation",
        {p: file_sha256(ROOT / p) for p in (MODEL, STATS + "/mean.pt", STATS + "/std.pt")},
    )


def run_audit(manifest):
    torch.manual_seed(482)
    fk = SkeletonFK()
    codec = SOKECodec(manifest, fk.feat_mean, fk.feat_std, fk.hand_mean)
    adapter = SOKEFK(manifest, codec, fk=fk)
    ref = SMPLXJoints(device="cpu")
    f = torch.randn(4, 133) * 0.4
    # Physical neutral rotations and zero raw expression, including inverse hand means.
    neutral_r = torch.eye(3).expand(41, 3, 3)
    neutral_e = -codec.mean[123:] / codec.std[123:]
    f[0] = codec.decode(neutral_r, neutral_e)
    r, e = codec.encode(f)
    decoded = codec.decode(r, e)
    rr, ee = codec.encode(decoded)
    torch.testing.assert_close(rr, r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ee, e)
    joints, left, right = ref(f.numpy())
    p = adapter(r, e)
    tips = [0, 4, 8, 12, 16, 20]
    reference = {"body": joints[:, UPPER_BODY_IDX], "left": left[:, tips], "right": right[:, tips]}
    error = {k: float((p[k] - v).abs().max()) for k, v in reference.items()}
    if max(error.values()) > 1e-5:
        raise AssertionError(f"FK does not match official evaluator: {error}")
    clip_audit = None
    train_dir = ROOT / "data/processed/madis/Phoenix_2014T/poses_video_level/train"
    candidates = sorted(train_dir.glob("*.pkl"))
    if candidates:
        from slp.data.sign_smplx import feats133

        clip = candidates[0]
        with open(clip, "rb") as handle:
            raw = torch.from_numpy(feats133(pickle.load(handle))[:8])
        nf = (raw - codec.mean) / codec.std
        cr, ce = codec.encode(nf)
        cp = adapter(cr, ce)
        cj, cl, ch = ref(nf.numpy())
        cref = {"body": cj[:, UPPER_BODY_IDX], "left": cl[:, tips], "right": ch[:, tips]}
        clip_errors = {k: float((cp[k] - value).abs().max()) for k, value in cref.items()}
        if max(clip_errors.values()) > 1e-5:
            raise AssertionError("Training clip conventions differ from evaluator")
        clip_audit = {
            "clip": str(clip.relative_to(ROOT)),
            "sha256": file_sha256(clip),
            "frames": len(nf),
            "fk_max_absolute_error_m": clip_errors,
        }
    features = SequenceFeatures(
        adapter, {"joints": 36, "wrists": 6, "handshape": 30, "palms": 12, "relative": 18}
    )
    r, e = r[None], e[None]
    mask = torch.ones(1, 4, dtype=torch.bool)
    times = torch.arange(4)[None] / manifest.fps
    features.fit_batch(r, e, mask, times, split="train")
    features.finalize(split="train")
    # Validate real landmark JVP and tangent gradient in float64.
    features.double()
    r, e, times = r.double(), e.double(), times.double()
    dw = (torch.randn(*r.shape[:-2], 3, dtype=r.dtype) * 0.01).requires_grad_()
    de = (torch.randn_like(e) * 0.01).requires_grad_()
    jv = feature_jvp(features, r, e, dw, de, mask, times)

    def fn(a):
        return torch.cat(
            [v.flatten(1) for v in features(r @ so3.exp(a * dw), e + a * de, mask, times).values()],
            -1,
        )

    h = 1e-5
    discrepancy = float((jv - (fn(h) - fn(-h)) / (2 * h)).detach().abs().max())
    torch.testing.assert_close(jv, (fn(h) - fn(-h)) / (2 * h), atol=1e-5, rtol=1e-4)
    grad = torch.autograd.grad(jv.square().sum(), (dw, de))
    if not all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad):
        raise AssertionError("Real SMPL-X JVP lost its tangent gradient")
    return {
        "status": "passed",
        "training_executed": False,
        "fk_max_absolute_error_m": error,
        "real_smplx_jvp_fd_max_error": discrepancy,
        "manifest_sha256": manifest.digest,
        "train_clip_audit": clip_audit,
        "scope": "physical neutral and articulated poses plus available training clip; full evaluator",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--report")
    args = parser.parse_args()
    manifest = soke_manifest(args.fps)
    report = run_audit(manifest)
    if args.write_manifest:
        p = Path(args.write_manifest)
        p.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(p)
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(output)
    print(output)


if __name__ == "__main__":
    main()

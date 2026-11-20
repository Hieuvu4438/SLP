from pathlib import Path

import pytest
import torch

from methods.slp_multirate.config import resolve
from methods.slp_multirate.data import CanonicalDataset, assign_splits, create_synthetic, prepare_data
from methods.slp_multirate.geometry import (
    CORRECTIONS, Rig, build_body_grid, build_fine_target, decode_pose, identity6, matrix_to_rot6d,
    rot6d_to_matrix, so3_exp, so3_log, upsample_body,
)


@pytest.fixture
def fixture_data(tmp_path):
    path, rig_path = create_synthetic(tmp_path / "raw")
    rig = Rig.load(rig_path)
    manifest = prepare_data(path, rig_path, "synthetic", tmp_path / "canonical",
                            dict(resolve("p0")["data"], min_frames=1))
    return manifest, rig


def test_rotation_conventions_and_degeneracy():
    angle = torch.tensor([0., 0., torch.pi / 2])
    matrix = so3_exp(angle)
    point = torch.tensor([2., 1., 0.])
    expected = torch.tensor([-1., 2., 0.])
    row = matrix_to_rot6d(matrix)
    col = matrix_to_rot6d(matrix.T)
    torch.testing.assert_close(rot6d_to_matrix(row) @ point, expected)
    torch.testing.assert_close(rot6d_to_matrix(col, convention="columns") @ point, expected)
    assert not torch.allclose(rot6d_to_matrix(col) @ point, expected)
    random = torch.randn(180, 3, generator=torch.Generator().manual_seed(4))
    rotations = so3_exp(random)
    r, bad = rot6d_to_matrix(torch.cat((matrix_to_rot6d(rotations), torch.zeros(1, 6),
                                       torch.ones(1, 6), torch.full((1, 6), torch.nan))),
                            return_fallback=True)
    torch.testing.assert_close(r[:180], rotations, atol=1e-4, rtol=0)
    torch.testing.assert_close(r.transpose(-1, -2) @ r, torch.eye(3).expand_as(r),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(r.det(), torch.ones(len(r)), atol=1e-4, rtol=0)
    assert bad.sum() == 3
    for theta in (0., 1e-7, .2, torch.pi - 1e-6):
        m = so3_exp(torch.tensor([1., -2., 3.]) / 14**.5 * theta)
        torch.testing.assert_close(so3_exp(so3_log(m)), m, atol=1e-5, rtol=0)


def test_temporal_grid_partial_bin_and_padding():
    # 5 real frames followed by a NaN and a second clip with just one frame.
    times = torch.tensor([[0., .02, .07, .12, .17, float("nan")],
                          [0., float("nan"), 0., 0., 0., 0.]])
    mask = torch.tensor([[True] * 5 + [False], [True] + [False] * 5])
    aa = torch.zeros(2, 6, 12, 3)
    aa[0, :5, :, 2] = torch.linspace(0, .2, 5)[:, None]
    rotation = so3_exp(aa)
    x = matrix_to_rot6d(rotation).masked_fill(~mask[..., None, None], torch.nan)
    obs = mask[..., None].expand(-1, -1, 12)
    grid = build_body_grid(x, times, mask, obs, 1)
    same, available = upsample_body(grid.rotations, grid.timestamps, grid.mask, times, mask,
                                    grid.observation)
    torch.testing.assert_close(same[mask], rotation[mask])
    assert torch.equal(obs, available)
    coarse = build_body_grid(x, times, mask, obs, 4)
    torch.testing.assert_close(coarse.timestamps[0], torch.tensor([.0525, .17]))
    assert coarse.mask.tolist() == [[True, True], [True, False]]
    g = coarse.rotations.masked_fill(~coarse.mask[..., None, None, None], torch.nan)
    u, _ = upsample_body(g, coarse.timestamps, coarse.mask, times, mask)
    assert u.isfinite().all()
    torch.testing.assert_close(u[0, 4], rotation[0, 4])


def test_fk_corrections_contact_and_ancestor_mask(fixture_data):
    manifest, rig = fixture_data
    ds = CanonicalDataset(manifest, rig, ("train",))
    assert ds[8]["contact"].shape == (1, 1)
    c = ds[1]
    hand = c["hands_rot6d"]
    body = rot6d_to_matrix(c["body_rot6d"])
    zero_delta = torch.zeros(len(body), 12)
    raw = torch.cat((hand.flatten(1), zero_delta), -1)
    pose = decode_pose(body, raw, rig)
    # Known synthetic contact: two middle fingertips coincide in metres.
    tips = pose.tips.flatten(1, 2)
    positive = c["contact"][:, 0]
    torch.testing.assert_close(tips[positive, 2], tips[positive, 7], atol=1e-5, rtol=0)
    identity = rot6d_to_matrix(identity6(len(body), 12))
    zero, _, _, _ = build_fine_target(body, hand, body, c["body_obs"], c["hands_obs"],
                                      c["body_obs"])
    aa = torch.zeros(len(body), 12, 3)
    aa[:, list(CORRECTIONS), 1] = .12
    offset = body @ so3_exp(aa)
    target, _, clamp, _ = build_fine_target(body, hand, offset, c["body_obs"], c["hands_obs"],
                                           c["body_obs"])
    assert not clamp.any()
    assert (target[:, 192:] - zero[:, 192:]).abs().max() > .1
    torch.testing.assert_close(decode_pose(offset, target, rig).local, pose.local, atol=1e-5, rtol=0)
    _, _, clamped, _ = build_fine_target(body, hand, identity, c["body_obs"], c["hands_obs"],
                                        c["body_obs"])
    assert clamped.any()
    rot = so3_exp(torch.tensor([0., .6, 0.]))
    switched = raw.clone()
    switched[:, :6] = matrix_to_rot6d(rot)
    other = decode_pose(body, switched, rig)
    torch.testing.assert_close(other.joints[:, :13], pose.joints[:, :13])
    torch.testing.assert_close(other.joints[:, 28:], pose.joints[:, 28:])
    assert not torch.allclose(other.tips[:, 0], pose.tips[:, 0])
    obs = torch.ones(len(body), 44, dtype=torch.bool)
    obs[:, 12] = False
    assert not decode_pose(body, raw, rig, observation=obs).tip_mask[:, 0].any()


def test_split_group_leakage_and_unknown_config():
    rows = [{"split": s, "origin_video_id": "v"} for s in ("train", "dev")]
    with pytest.raises(ValueError, match="overlap"):
        assign_splits(rows)
    with pytest.raises(ValueError, match="Unknown"):
        resolve("p0", overrides=["fine.typo=4"])
    assert resolve("p1", "B")["anchors"]["policy"] == "uniform"
    assert resolve("p0", path=Path("methods/slp_multirate/configs/pilots/p0.yaml"))["text"][
        "encoder"] == "mock"


def test_lmdb_source_column_order_and_missing_pelvis(tmp_path):
    import io

    import numpy as np
    from methods.slp_multirate.data import SMPLX_BODY_21, write_jsonl

    lmdb = pytest.importorskip("lmdb")
    source, rig_file = create_synthetic(tmp_path)
    lmdb_dir = tmp_path / "toy.lmdb"
    stream = io.BytesIO()
    t = 8
    columns = identity6(t, 21).clone()
    target = so3_exp(torch.tensor([0., 0., torch.pi / 2]))
    columns[:, SMPLX_BODY_21.index("right_wrist")] = matrix_to_rot6d(target.T)
    np.savez(stream, body_features=columns.numpy().reshape(t, -1),
             left_features=identity6(t, 15).numpy().reshape(t, -1),
             right_features=identity6(t, 15).numpy().reshape(t, -1),
             translation=np.asarray(["fixture"]))
    with lmdb.open(str(lmdb_dir)) as db:
        with db.begin(write=True) as txn:
            txn.put(b"one", stream.getvalue())
    row = dict(clip_id="one", origin_video_id="one", split="train", target_language="ASL",
               path=lmdb_dir.name, text="fixture", key="one", fps=25, source_joint_schema=(
                   "smplx_body21_hand15_v1"), local_axes_id="toy-v1", rotation_convention="columns",
               unit="m", handedness="right", provenance="synthetic-lmdb-schema")
    write_jsonl(source, [row])
    with pytest.raises(ValueError, match="lacks pelvis"):
        prepare_data(source, rig_file, "signspark_lmdb", tmp_path / "bad", resolve("p0")["data"])
    write_jsonl(source, [dict(row, pelvis_source="verified_identity")])
    manifest = prepare_data(source, rig_file, "signspark_lmdb", tmp_path / "canonical",
                            resolve("p0")["data"])
    clip = CanonicalDataset(manifest, Rig.load(rig_file), ("train",), fps=25)[0]
    actual = rot6d_to_matrix(clip["hands_rot6d"][0, 1, 0])
    torch.testing.assert_close(actual, target)
    with pytest.raises(ValueError, match="FPS differs"):
        CanonicalDataset(manifest, Rig.load(rig_file), ("train",), fps=30)

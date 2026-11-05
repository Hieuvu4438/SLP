"""Real official SMPL-X engine on a tiny synthetic rig; no licensed human assets.

This validates packing, hand-mean convention and gradients, NOT human anatomy or
real-asset quality. Native joint locations have an independent analytic FK reference.
"""
import numpy as np
from dataclasses import replace
import pytest
import torch

from methods.temporal_plan.geometry import axis_angle_to_matrix, forward_kinematics, unpack_smplx
from methods.temporal_plan.kinematics import RETAINED_JOINTS, SMPLXForward
from methods.temporal_plan.losses import motion_reconstruction
from methods.temporal_plan.representation import MotionRepresentation


@pytest.fixture
def official_synthetic_smplx(tmp_path):
    smplx = pytest.importorskip("smplx")
    vertices, joints = 100, 55
    template = np.zeros((vertices, 3), dtype=np.float32)
    template[:, 0] = np.arange(vertices) * .02
    regressor = np.zeros((joints, vertices), dtype=np.float32)
    regressor[np.arange(joints), np.arange(joints)] = 1
    weights = np.zeros((vertices, joints), dtype=np.float32)
    weights[np.arange(vertices), np.minimum(np.arange(vertices), joints-1)] = 1
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez_compressed(
        path, v_template=template, f=np.array([[0, 1, 2]], dtype=np.int64),
        shapedirs=np.zeros((vertices, 3, 400), dtype=np.float32),
        posedirs=np.zeros((vertices, 3, (joints-1)*9), dtype=np.float32),
        J_regressor=regressor, weights=weights,
        kintree_table=np.stack((np.arange(joints)-1, np.arange(joints))),
        hands_componentsl=np.eye(45, dtype=np.float32),
        hands_componentsr=np.eye(45, dtype=np.float32),
        hands_meanl=np.tile(np.array([.5, .2, -.1], dtype=np.float32), 15),
        hands_meanr=np.tile(np.array([-.1, .1, .2], dtype=np.float32), 15),
        lmk_faces_idx=np.array([0], dtype=np.int64),
        lmk_bary_coords=np.array([[1., 0., 0.]], dtype=np.float32),
    )
    model = smplx.SMPLX(str(path), use_pca=False, flat_hand_mean=False,
                         num_betas=10, num_expression_coeffs=10, use_face_contour=False)
    # Synthetic topology has 100 vertices, so give auxiliary landmarks valid IDs.
    # Native 55-joint LBS/FK and pose means remain the actual official implementation.
    model.vertex_joint_selector.extra_joints_idxs = torch.arange(
        len(model.vertex_joint_selector.extra_joints_idxs)
    )
    return model


def test_official_smplx_adapter_matches_independent_fk_and_keeps_pose_gradients(
    official_synthetic_smplx,
):
    backend = SMPLXForward(official_synthetic_smplx)
    raw = torch.zeros(2, 133, requires_grad=True)
    with torch.no_grad():
        raw[0, 2] = 1.1
        raw[1, 30:33] = torch.tensor([3.4, .1, 0.])
    actual = backend(raw)
    packed = unpack_smplx(raw)
    full_pose = torch.cat((packed["global_orient"], packed["body_pose"], packed["jaw_pose"],
                           packed["leye_pose"], packed["reye_pose"], packed["left_hand_pose"],
                           packed["right_hand_pose"]), -1).reshape(2, 55, 3)
    effective = full_pose + backend.body_model.pose_mean.reshape(55, 3)
    offsets = torch.zeros(55, 3)
    offsets[1:, 0] = .02
    expected, _ = forward_kinematics(axis_angle_to_matrix(effective), offsets,
                                     tuple(range(-1, 54)))
    torch.testing.assert_close(actual, expected[:, list(RETAINED_JOINTS)],
                               atol=2e-6, rtol=2e-5)
    actual[..., 1].sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert all(p.grad is None for p in backend.body_model.parameters())


def test_effective_rotation_round_trip_preserves_official_fk_near_axis_angle_wrap(
    official_synthetic_smplx,
):
    backend = SMPLXForward(official_synthetic_smplx)
    representation = MotionRepresentation(backend.rotation_offsets)
    raw = torch.zeros(1, 2, 133)
    raw[0, 0, 30:33] = torch.tensor([3.4, .1, 0.])
    raw[0, 1, :3] = torch.tensor([.1, -.2, .3])
    lengths = torch.tensor([2])
    features = representation.encode(raw, lengths)
    restored = representation.decode(features, lengths)
    # Raw axis-angle coordinates may differ by a wrap; the rendered motion must not.
    torch.testing.assert_close(backend(restored[0]), backend(raw[0]), atol=2e-6, rtol=2e-5)
    predicted = features.clone().requires_grad_(True)
    loss = motion_reconstruction(predicted, features, lengths, fk=backend)
    torch.testing.assert_close(loss["total"], torch.tensor(0.), atol=1e-5, rtol=0)
    loss["total"].backward()
    assert torch.isfinite(predicted.grad).all()


def test_missing_assets_do_not_silently_disable_fk(tmp_path):
    with pytest.raises(FileNotFoundError, match="licensed"):
        SMPLXForward.from_model_dir(tmp_path / "missing")


def test_full_runner_with_positive_fk_and_official_engine_needs_no_assets_for_inference(
    official_synthetic_smplx, tiny_components, tmp_path,
):
    from methods.temporal_plan.checkpoint import load_checkpoint, read_checkpoint
    from methods.temporal_plan.data import MotionDataset
    from methods.temporal_plan.system import TemporalPlanSystem, STAGES
    from methods.temporal_plan.training import TrainingConfig, TrainingRunner

    tiny, tokenizer = tiny_components
    backend = SMPLXForward(official_synthetic_smplx)
    config = replace(tiny.config, losses=replace(tiny.config.losses, fk=1.0),
                     stop_threshold=.65)
    model = TemporalPlanSystem(config, tiny.text, backend.rotation_offsets)
    records = [{"name": f"train{i}", "source_id": f"video{i}", "lang": "en_ASL",
                "fps": 25., "text": "hello rain"} for i in range(2)]
    rng = np.random.default_rng(11)
    train = MotionDataset(records, [rng.normal(0, .05, (n, 133)).astype(np.float32)
                                   for n in [5, 8]], split="train")
    runner = TrainingRunner(model, tokenizer, TrainingConfig(
        batch_size=2, accumulation=1, epochs={s: 1 for s in STAGES},
    ), tmp_path / "official_fk_run", fk=backend)
    checkpoint = runner.fit(train)
    assert model.completed_stages.all()
    payload = read_checkpoint(checkpoint)
    assert not any("body_model" in key for key in payload["state_dict"])
    loaded, tokens, _ = load_checkpoint(checkpoint, require_completed=True)
    torch.testing.assert_close(loaded.representation.rotation_offsets, backend.rotation_offsets)
    assert loaded.planner.stop_threshold == .65
    batch = tokens(["hello rain"], return_tensors="pt")
    result = loaded.generate(batch["input_ids"], batch["attention_mask"], torch.tensor([0]),
                              generator=torch.Generator().manual_seed(9))
    assert torch.isfinite(result.raw133).all()
    assert torch.isfinite(backend(result.raw133[0])).all()

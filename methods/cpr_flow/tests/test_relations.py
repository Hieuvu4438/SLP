import torch

from cprflow.geometry.relations import relation33
from cprflow.geometry.temporal import pool2


def skeleton():
    names = ["pelvis", "neck", "left_shoulder", "right_shoulder", "jaw"]
    values = [[0, 0, 0], [0, 1, 0], [-0.3, 0.8, 0], [0.3, 0.8, 0], [0, 1.2, 0.1]]
    for side, s in [("left", -1), ("right", 1)]:
        names.extend([f"{side}_wrist", f"{side}_index1", f"{side}_pinky1"])
        values.extend([[s * 0.4, 0.5, 0.2], [s * 0.4, 0.6, 0.2], [s * 0.5, 0.5, 0.2]])
        for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            names.append(f"{side}_{finger}")
            values.append([s * (0.4 + 0.02 * i), 0.65, 0.2])
    return torch.tensor(values)[None], names


def test_order_rigid_invariance_and_handedness():
    joints, names = skeleton()
    r = relation33(joints, names)
    assert r.shape == (1, 33)
    rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = joints @ rotation.T + torch.tensor([2.0, 3, 4])
    torch.testing.assert_close(relation33(transformed, names), r, atol=2e-6, rtol=2e-6)
    swapped = joints.clone()
    for name in names:
        if name.startswith("left_"):
            other = name.replace("left_", "right_", 1)
            a, b = names.index(name), names.index(other)
            swapped[:, a], swapped[:, b] = joints[:, b], joints[:, a]
    assert not torch.allclose(relation33(swapped, names), r)
    torch.testing.assert_close(r[:, 6:9], r[:, 3:6] - r[:, :3])
    torch.testing.assert_close(r[:, 24:], torch.sigmoid((0.08 - r[:, 15:24]) / 0.02))


def test_odd_pool_and_masks():
    x = torch.tensor([[[1.0], [3.0], [9.0], [500.0]]])
    torch.testing.assert_close(pool2(x, torch.tensor([3])), torch.tensor([[[2.0], [9.0]]]))


def test_relation_plan_invalid_rate_preserves_averaged_normal_semantics():
    from cprflow.geometry.relations import relation_invalid_rates

    joints, names = skeleton()
    relation = relation33(joints, names)
    relation[:, 9:15] *= 0.5
    assert all(value == 0 for value in relation_invalid_rates(relation).values())
    relation[:, 24] = 2
    relation[:, 15] = -1
    rates = relation_invalid_rates(relation)
    assert abs(rates["relation_plan_invalid_proximity_rate"] - 1 / 9) < 1e-6
    assert abs(rates["relation_plan_negative_distance_rate"] - 1 / 9) < 1e-6

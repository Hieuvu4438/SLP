"""SMPL-X native names; tips are vertex-selected joints, never distal finger joints."""

BODY_NAMES = (
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
TIP_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def hand_names(side):
    return (
        (f"{side}_wrist",)
        + tuple(f"{side}_{f}{j}" for f in FINGERS for j in (1, 2, 3))
        + tuple(f"{side}_{f}" for f in TIP_ORDER)
    )


def select(joints, names, available):
    missing = set(names) - set(available)
    if missing:
        raise ValueError(f"Missing audited SMPL-X joints: {sorted(missing)}")
    return joints[..., [available.index(name) for name in names], :]

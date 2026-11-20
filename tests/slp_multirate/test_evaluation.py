import pytest
import torch

from methods.slp_multirate.config import resolve
from slp.evaluation.multirate import (
    gate_decision, hierarchical_effects, hysteresis, match_events, paired_effects,
)


def _e(time, kind="shape"):
    return {"time_s": time, "kind": kind}


def test_event_matching_maximum_hits_and_minimum_error():
    m = match_events([_e(.1), _e(.2), _e(.3, "turn")],
                     [_e(.05), _e(.12), _e(.24), _e(.8, "turn")], .08)
    assert (m["event_tp"], m["event_fp"], m["event_fn"]) == (2, 2, 1)
    assert m["timing_mae_ms"] == pytest.approx(30.)
    assert m["event_f1"] == pytest.approx(4 / 7)


def test_contact_hysteresis_keeps_first_confirmed_frame():
    d = torch.tensor([.1, .01, .01, .026, .04, .01, .05, .04, .01])[:, None]
    flags, events = hysteresis(d, torch.ones_like(d, dtype=torch.bool), .02)
    assert flags[:, 0].tolist() == [False, True, True, True, True, True, False, False, False]
    assert events == [(1, 0, "onset"), (6, 0, "offset")]


def _row(i, score, anchors):
    return dict(clip_id=str(i), origin_video_id=str(i // 2), event_tip_mm=score, wrist_mm=5.,
                outside_tip_mm=5., event_frames=list(range(7)),
                frame_tip_count=[1]*7, frame_tip_sum=[score]*7, oracle_anchor_frames=anchors)


def test_paired_common_mask_bootstrap_and_gate():
    cfg = dict(resolve("p0")["eval"], bootstrap_resamples=40)
    reference = [_row(i, 10., [1]) for i in range(10)]
    candidate = [_row(i, 8., [4]) for i in range(10)]
    effects = paired_effects(reference, candidate, cfg, 17)
    assert effects["event_tip_mm"]["effect"] == .2
    assert effects["event_tip_excluding_anchors_mm"]["n"] == 10
    assert effects["event_tip_excluding_anchors_mm"]["effect"] == .2
    assert gate_decision(effects, "p1", 10)[0] == "GO"
    assert gate_decision(effects, "p2", 10)[0] == "HOLD"
    altered = [dict(candidate[0], event_frames=[0]), *candidate[1:]]
    with pytest.raises(ValueError, match="event_frames"):
        paired_effects(reference, altered, cfg, 17)
    hierarchical = hierarchical_effects([(reference, candidate)] * 3, "event_tip_mm", cfg)
    assert hierarchical["mean"] == pytest.approx(.2)
    interaction = hierarchical_effects([(reference, candidate, reference, candidate)] * 3,
                                       "event_tip_mm", cfg, interaction=True)
    assert interaction["mean"] == 0

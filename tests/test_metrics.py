import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import match, per_kind, temporal_iou  # noqa: E402


def pred(eid, kind="congestion", cam="cam-1", t0=0.0, t1=10.0, conf=0.9):
    return {"event_id": eid, "camera_id": cam, "kind": kind, "t_start": t0, "t_end": t1, "confidence": conf}


def truth(kind="congestion", cam="cam-1", t0=0.0, t1=10.0):
    return {"camera_id": cam, "kind": kind, "t_start": t0, "t_end": t1}


def test_temporal_iou():
    assert temporal_iou(0, 10, 0, 10) == 1.0
    assert temporal_iou(0, 10, 20, 30) == 0.0
    assert abs(temporal_iou(0, 10, 5, 15) - (5 / 15)) < 1e-9


def test_perfect_match():
    m = match([pred("a")], [truth()])
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


def test_wrong_kind_is_a_false_positive():
    m = match([pred("a", kind="blocked_zone")], [truth()])
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_wrong_camera_is_a_false_positive():
    m = match([pred("a", cam="cam-2")], [truth()])
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_duplicate_predictions_count_against_precision():
    """Two alerts for one incident is two interruptions for the operator."""
    m = match([pred("a"), pred("b", t0=1, t1=9)], [truth()])
    assert m.tp == 1
    assert m.duplicates == 1
    assert m.fp == 1


def test_below_iou_threshold_does_not_match():
    m = match([pred("a", t0=0, t1=10)], [truth(t0=9.5, t1=40)], min_tiou=0.2)
    assert m.tp == 0


def test_scenario_scoping_prevents_cross_scenario_matches():
    p = pred("a")
    p["scenario_id"] = "s1"
    g = truth()
    g["scenario_id"] = "s2"
    assert match([p], [g]).tp == 0


def test_higher_confidence_prediction_claims_the_match():
    m = match([pred("low", conf=0.2), pred("high", conf=0.95)], [truth()])
    assert m.matched_pairs[0][0] == "high"


def test_per_kind_breakdown():
    out = per_kind([pred("a"), pred("b", kind="blocked_zone")], [truth(), truth(kind="blocked_zone")])
    assert out["congestion"]["tp"] == 1
    assert out["blocked_zone"]["tp"] == 1


def test_empty_inputs_are_zero_not_crash():
    m = match([], [])
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0

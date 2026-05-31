"""Scorer unit tests: detection P/R and zone-event matching on known cases."""
from __future__ import annotations

from vigil.eval.clips import ExpectedEvent
from vigil.eval.score import score_detections, score_zone_events
from vigil.types import Detection, Event, EventType


def _det(x1, y1, x2, y2, cls="person", conf=0.9):
    return Detection((x1, y1, x2, y2), 0, cls, conf)


def test_detection_perfect_match():
    gt = [[_det(0, 0, 10, 10)]]
    pred = [[_det(0, 0, 10, 10)]]
    s = score_detections(pred, gt)
    assert s["overall"]["precision"] == 1.0 and s["overall"]["recall"] == 1.0


def test_detection_false_positive_lowers_precision():
    gt = [[_det(0, 0, 10, 10)]]
    pred = [[_det(0, 0, 10, 10), _det(100, 100, 110, 110)]]  # one spurious
    s = score_detections(pred, gt)
    assert s["overall"]["recall"] == 1.0
    assert s["overall"]["precision"] == 0.5  # 1 tp, 1 fp


def test_detection_miss_lowers_recall():
    gt = [[_det(0, 0, 10, 10), _det(50, 50, 60, 60)]]
    pred = [[_det(0, 0, 10, 10)]]  # missed the second
    s = score_detections(pred, gt)
    assert s["overall"]["precision"] == 1.0
    assert s["overall"]["recall"] == 0.5


def test_detection_class_is_respected():
    gt = [[_det(0, 0, 10, 10, cls="person")]]
    pred = [[_det(0, 0, 10, 10, cls="car")]]  # right box, wrong class
    s = score_detections(pred, gt)
    assert s["overall"]["recall"] == 0.0  # person not found
    assert s["per_class"]["car"]["fp"] == 1


def _event(kind, zone, frame):
    return Event(
        event_type=kind, track_id=1, zone_id=zone, class_name="person",
        timestamp_utc="t", timestamp_monotonic=0.0, centroid=(0.0, 0.0),
        bbox=(0.0, 0.0, 1.0, 1.0), frame_index=frame,
    )


def test_zone_event_matches_within_tolerance():
    expected = [ExpectedEvent("ZONE_ENTRY", "A", "z1", 10, tolerance=3)]
    actual = [_event(EventType.ZONE_ENTRY, "z1", 12)]  # 2 frames off, within tol
    s = score_zone_events(expected, actual)
    assert s["matched"] == 1 and s["missed"] == 0 and s["false_events"] == 0


def test_zone_event_outside_tolerance_is_missed_and_false():
    expected = [ExpectedEvent("ZONE_ENTRY", "A", "z1", 10, tolerance=2)]
    actual = [_event(EventType.ZONE_ENTRY, "z1", 20)]  # too far -> not a match
    s = score_zone_events(expected, actual)
    assert s["matched"] == 0 and s["missed"] == 1 and s["false_events"] == 1


def test_zone_event_wrong_zone_does_not_match():
    expected = [ExpectedEvent("ZONE_ENTRY", "A", "z1", 10)]
    actual = [_event(EventType.ZONE_ENTRY, "other", 10)]
    s = score_zone_events(expected, actual)
    assert s["matched"] == 0 and s["false_events"] == 1


def test_zone_event_no_expected_no_actual_is_clean():
    s = score_zone_events([], [])
    assert s["matched"] == 0 and s["false_events"] == 0
    assert s["precision"] == 1.0 and s["recall"] == 1.0

"""Core typed records: geometry helpers and JSON round-trips."""
from __future__ import annotations

import json

from vigil.types import Detection, Event, EventType, Track


def test_detection_geometry():
    det = Detection(bbox=(10.0, 20.0, 30.0, 60.0), class_id=0, class_name="person", confidence=0.9)
    assert det.x1 == 10.0
    assert det.y2 == 60.0
    assert det.width == 20.0
    assert det.height == 40.0
    assert det.area == 800.0
    assert det.centroid == (20.0, 40.0)


def test_detection_roundtrip_is_serialisable():
    det = Detection(bbox=(1.0, 2.0, 3.0, 4.0), class_id=5, class_name="car", confidence=0.5)
    data = det.to_dict()
    json.dumps(data)  # must be JSON-serialisable (it crosses the log + web API)
    assert Detection.from_dict(data) == det


def test_track_roundtrip():
    det = Detection(bbox=(0.0, 0.0, 10.0, 10.0), class_id=2, class_name="dog", confidence=0.7)
    track = Track(track_id=42, detection=det, age=3, hits=4, time_since_update=1)
    data = track.to_dict()
    json.dumps(data)
    assert Track.from_dict(data) == track


def test_event_roundtrip_and_enum():
    assert EventType.ZONE_ENTRY.value == "ZONE_ENTRY"
    assert EventType.ZONE_EXIT.value == "ZONE_EXIT"
    event = Event(
        event_type=EventType.ZONE_ENTRY,
        timestamp=12.5,
        track_id=7,
        zone_id="dock-A",
        detail={"speed": 1.2},
    )
    data = event.to_dict()
    json.dumps(data)
    assert Event.from_dict(data) == event

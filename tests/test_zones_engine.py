"""Zone entry/exit event engine: debounce, exclude suppression, class filter."""
from __future__ import annotations

import json
from types import SimpleNamespace

from vigil.types import EventType, Track
from vigil.zones.engine import ZoneEventEngine
from vigil.zones.geometry import ZoneIndex
from vigil.zones.model import Zone, ZoneKind, ZoneSet

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]  # 100x100 square at origin


def _frame(i: int):
    return SimpleNamespace(index=i, timestamp=i / 30.0)


def _track(cx: float, cy: float, class_name: str = "person", tid: int = 1, size: float = 20.0):
    half = size / 2
    return Track(
        track_id=tid,
        class_id=0,
        class_name=class_name,
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=0.9,
    )


def _engine(zones, *, enter=2, exit=2, frame_size=(100, 100), resolution=(100, 100)):
    index = ZoneIndex(ZoneSet(resolution=resolution, zones=zones), frame_size)
    return ZoneEventEngine(
        index, enter_frames=enter, exit_frames=exit,
        utc_now=lambda: "2026-05-31T00:00:00+00:00",
    )


def _run(engine, positions, class_name="person"):
    events = []
    for i, cx in enumerate(positions):
        events.extend(engine.update([_track(cx, 50, class_name)], _frame(i)))
    return events


# --- crossing in then out fires exactly one ENTRY and one EXIT ------------- #


def test_crossing_fires_one_entry_and_one_exit():
    engine = _engine([Zone("z", "Z", SQUARE)], enter=2, exit=2)
    # outside x2, inside x4, outside x4   (inside x=50, outside x=150)
    events = _run(engine, [150, 150, 50, 50, 50, 50, 150, 150, 150, 150])
    assert [e.event_type for e in events] == [EventType.ZONE_ENTRY, EventType.ZONE_EXIT]
    entry, exit_ = events
    assert entry.zone_id == "z" and entry.track_id == 1 and entry.class_name == "person"
    assert exit_.zone_id == "z"
    # event carries full, JSON-serialisable context ready for logging
    data = entry.to_dict()
    json.dumps(data)
    assert data["centroid"] == [50.0, 50.0]
    assert data["bbox"] == [40.0, 40.0, 60.0, 60.0]
    assert data["timestamp_utc"] == "2026-05-31T00:00:00+00:00"
    assert isinstance(data["timestamp_monotonic"], float)
    assert data["frame_index"] == 3  # 2nd consecutive inside frame (enter=2)


def test_immediate_debounce_k1_fires_on_first_frame():
    engine = _engine([Zone("z", "Z", SQUARE)], enter=1, exit=1)
    events = _run(engine, [50, 50, 150])
    assert [e.event_type for e in events] == [EventType.ZONE_ENTRY, EventType.ZONE_EXIT]


# --- boundary jitter fires nothing ----------------------------------------- #


def test_boundary_jitter_fires_nothing():
    engine = _engine([Zone("z", "Z", SQUARE)], enter=2, exit=2)
    # alternate inside/outside every frame -> streak never reaches 2
    events = _run(engine, [50, 150, 50, 150, 50, 150, 50, 150])
    assert events == []


def test_brief_dip_below_threshold_does_not_fire():
    engine = _engine([Zone("z", "Z", SQUARE)], enter=3, exit=3)
    # inside for only 2 frames (< 3) then leaves -> no ENTRY ever
    events = _run(engine, [150, 50, 50, 150, 150, 150])
    assert events == []


# --- EXCLUDE zones suppress ------------------------------------------------ #


def test_exclude_zone_suppresses_events():
    include = Zone("inc", "Include", SQUARE, ZoneKind.INCLUDE)
    exclude = Zone("exc", "Exclude", SQUARE, ZoneKind.EXCLUDE)  # same area
    engine = _engine([include, exclude], enter=2, exit=2)
    events = _run(engine, [50, 50, 50, 50, 50])  # inside both -> excluded
    assert events == []


def test_without_exclude_the_same_path_fires_entry():
    engine = _engine([Zone("inc", "Include", SQUARE, ZoneKind.INCLUDE)], enter=2, exit=2)
    events = _run(engine, [50, 50, 50, 50, 50])
    assert [e.event_type for e in events] == [EventType.ZONE_ENTRY]


# --- class filter respected ------------------------------------------------ #


def test_class_filter_blocks_wrong_class():
    engine = _engine(
        [Zone("p", "People", SQUARE, ZoneKind.INCLUDE, classes=["person"])],
        enter=2, exit=2,
    )
    assert _run(engine, [50, 50, 50, 50], class_name="car") == []
    engine.reset()
    events = _run(engine, [50, 50, 50, 50], class_name="person")
    assert [e.event_type for e in events] == [EventType.ZONE_ENTRY]


# --- occlusion: a track that vanishes debounces to EXIT, not instantly ----- #


def test_absent_track_debounces_to_exit():
    engine = _engine([Zone("z", "Z", SQUARE)], enter=2, exit=3)
    # confirm inside
    events = []
    for i in range(4):
        events.extend(engine.update([_track(50, 50)], _frame(i)))
    assert [e.event_type for e in events] == [EventType.ZONE_ENTRY]

    # vanish for 2 frames (< exit window of 3) then return -> no spurious EXIT
    engine.update([], _frame(4))
    engine.update([], _frame(5))
    back = engine.update([_track(50, 50)], _frame(6))
    assert back == []

    # now vanish long enough to cross the exit window
    out = []
    for i in range(7, 11):
        out.extend(engine.update([], _frame(i)))
    assert [e.event_type for e in out] == [EventType.ZONE_EXIT]

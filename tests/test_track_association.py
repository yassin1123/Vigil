"""Detection->track association edge cases. Deterministic, no GPU.

This is the contract zones build on, so the corner cases are pinned down:
empty frames, a one-frame flicker, an object entering, an object leaving, and
two objects swapping positions.
"""
from __future__ import annotations

import json

from vigil.track import ByteTrackTracker
from vigil.types import Detection, Track


def _det(cx: float, cy: float, size: float = 40.0, class_id: int = 0, conf: float = 0.9):
    half = size / 2.0
    name = "person" if class_id == 0 else str(class_id)
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_id=class_id,
        class_name=name,
        confidence=conf,
    )


def test_empty_frames_age_then_remove_all_tracks():
    tracker = ByteTrackTracker(confirm_frames=2, lost_window=4)
    for i in range(5):
        tracker.update([_det(100 + i * 10, 100)])
    assert len(tracker.active_tracks()) == 1

    for _ in range(10):  # nothing detected -> age out
        tracker.update([])
    assert tracker.active_tracks() == []
    assert tracker.tracks == []


def test_flicker_single_missing_frame_keeps_id():
    tracker = ByteTrackTracker(confirm_frames=2, lost_window=5)
    tracks = []
    for i in range(4):
        tracks = tracker.update([_det(100 + i * 10, 100)])
    original_id = tracks[0].track_id

    tracker.update([])  # one-frame flicker
    tracks = tracker.update([_det(100 + 5 * 10, 100)])
    assert len(tracks) == 1
    assert tracks[0].track_id == original_id


def test_new_object_entering_gets_a_new_id():
    tracker = ByteTrackTracker(confirm_frames=2)
    a = []
    for i in range(4):
        a = tracker.update([_det(100 + i * 10, 100)])
    a_id = a[0].track_id

    tracks = []
    for i in range(4, 9):  # a second object enters far away
        tracks = tracker.update([_det(100 + i * 10, 100), _det(420, 220)])
    ids = {t.track_id for t in tracks}
    assert a_id in ids
    assert len(ids) == 2


def test_object_leaving_is_dropped():
    tracker = ByteTrackTracker(confirm_frames=2, lost_window=3)
    tracks = []
    for i in range(5):
        tracks = tracker.update([_det(100 + i * 10, 100)])
    left_id = tracks[0].track_id
    assert tracker.track_by_id(left_id) is not None

    for _ in range(6):  # object leaves the scene for good
        tracker.update([])
    assert tracker.track_by_id(left_id) is None
    assert tracker.active_tracks() == []


def test_two_objects_swapping_positions_keep_ids():
    tracker = ByteTrackTracker(confirm_frames=2)
    ids_seen: set[int] = set()
    for i in range(20):
        # A left->right at y=100; B right->left at y=132. They swap ends.
        tracks = tracker.update([_det(50 + i * 15, 100), _det(350 - i * 15, 132)])
        ids_seen |= {t.track_id for t in tracks}
    assert len(ids_seen) == 2


def test_track_api_is_serializable_and_queryable():
    tracker = ByteTrackTracker(confirm_frames=1)  # confirm immediately
    tracker.update([_det(100, 100)])
    active = tracker.active_tracks()
    assert len(active) == 1

    track = active[0]
    assert Track.from_dict(track.to_dict()) == track  # full round-trip
    json.dumps(track.to_dict())
    json.dumps(track.snapshot())  # compact form is also JSON-safe

    snap = tracker.snapshot()
    assert snap["count"] == 1 and snap["tracks"][0]["id"] == track.track_id
    json.dumps(snap)

    assert tracker.track_by_id(track.track_id) is not None
    assert tracker.track_by_id(999999) is None


def test_module_level_entry_point():
    from vigil import track as track_mod

    track_mod.reset()
    track_mod.configure()  # default tracker (confirm_frames=3)
    out = []
    for i in range(5):
        out = track_mod.update([_det(100 + i * 10, 100)])
    assert len(out) == 1
    assert track_mod.active_tracks()[0].track_id == out[0].track_id
    track_mod.reset()
    assert track_mod.active_tracks() == []

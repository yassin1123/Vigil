"""ByteTrack: stable IDs on synthetic detection sequences. No GPU, no network."""
from __future__ import annotations

import pytest

from vigil.config import VigilConfig
from vigil.track import ByteTrackTracker, Tracker, build_tracker
from vigil.types import Detection


def _det(cx: float, cy: float, size: float = 40.0, class_id: int = 0, conf: float = 0.9):
    half = size / 2.0
    name = "person" if class_id == 0 else str(class_id)
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_id=class_id,
        class_name=name,
        confidence=conf,
    )


def test_tracker_is_abstract():
    with pytest.raises(TypeError):
        Tracker()  # cannot instantiate the interface


def test_single_object_holds_one_stable_id():
    tracker = ByteTrackTracker()
    ids = []
    for i in range(20):
        cx = 100 + i * 10  # steady rightward motion
        tracks = tracker.update([_det(cx, 240)])
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1  # exactly one id for the whole trajectory
    # history accumulates centroids as the object moves
    final = tracker.update([_det(100 + 20 * 10, 240)])[0]
    assert len(final.history) > 1


def test_two_crossing_objects_keep_distinct_ids():
    tracker = ByteTrackTracker()
    ids_seen: set[int] = set()
    first_pair: set[int] | None = None
    last_ids: set[int] = set()

    # A moves left->right at y=220; B moves right->left at y=260 (a close pass,
    # trajectories crossing in x). Good motion tracking must not swap/merge ids.
    for i in range(24):
        a = _det(100 + i * 12, 220)
        b = _det(400 - i * 12, 260)
        tracks = tracker.update([a, b])
        ids = {t.track_id for t in tracks}
        ids_seen |= ids
        if first_pair is None and len(tracks) == 2:
            first_pair = ids
        last_ids = ids

    assert len(ids_seen) == 2, "no extra ids should be created through the crossing"
    assert last_ids == first_pair, "the original two ids survive to the end"
    assert len(last_ids) == 2


def test_lost_object_keeps_id_when_it_returns():
    tracker = ByteTrackTracker(track_buffer=30)
    for i in range(8):
        tracks = tracker.update([_det(100 + i * 10, 240)])
    original_id = tracks[0].track_id

    # Disappear for a few frames (within the buffer)...
    for _ in range(3):
        tracker.update([])

    # ...then reappear near the predicted location.
    tracks = tracker.update([_det(100 + 11 * 10, 240)])
    assert len(tracks) == 1
    assert tracks[0].track_id == original_id


def test_new_object_gets_new_id_after_first_frame():
    tracker = ByteTrackTracker()
    # frame 1: one object (id assigned, confirmed immediately on first frame)
    t1 = tracker.update([_det(100, 240)])
    assert len(t1) == 1
    first_id = t1[0].track_id
    # a second object appears later: it must get a different id
    for i in range(1, 6):
        tracks = tracker.update([_det(100 + i * 10, 240), _det(500, 100)])
    ids = {t.track_id for t in tracks}
    assert first_id in ids
    assert len(ids) == 2


def test_build_tracker_from_config():
    cfg = VigilConfig.from_dict({"tracker": {"track_thresh": 0.6, "track_buffer": 50}})
    tracker = build_tracker(cfg)
    assert isinstance(tracker, ByteTrackTracker)
    assert tracker.track_thresh == 0.6
    assert tracker.max_time_lost == 50

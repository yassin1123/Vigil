"""ByteTrack: stable IDs + lifecycle on synthetic sequences. No GPU, no network."""
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
    tracker = ByteTrackTracker(confirm_frames=3)
    ids = []
    for i in range(20):
        tracks = tracker.update([_det(100 + i * 10, 240)])
        if tracks:
            ids.append(tracks[0].track_id)
    # Confirmed from frame 3 onward -> output for ~17 frames, all one id.
    assert len(ids) >= 16
    assert len(set(ids)) == 1


def test_two_crossing_objects_keep_distinct_ids():
    tracker = ByteTrackTracker(confirm_frames=3)
    ids_seen: set[int] = set()
    first_pair: set[int] | None = None
    last_ids: set[int] = set()

    # A moves left->right at y=220; B moves right->left at y=260 (a close pass,
    # trajectories crossing in x). Good motion tracking must not swap/merge ids.
    for i in range(24):
        tracks = tracker.update([_det(100 + i * 12, 220), _det(400 - i * 12, 260)])
        ids = {t.track_id for t in tracks}
        ids_seen |= ids
        if first_pair is None and len(tracks) == 2:
            first_pair = ids
        last_ids = ids

    assert len(ids_seen) == 2, "no extra ids should be created through the crossing"
    assert last_ids == first_pair, "the original two ids survive to the end"
    assert len(last_ids) == 2


def test_tentative_track_not_emitted_before_confirmation():
    tracker = ByteTrackTracker(confirm_frames=3)
    # A single one-frame blip never reaches confirmation -> no output, then gone.
    assert tracker.update([_det(100, 100)]) == []  # frame 1: tentative
    assert tracker.update([]) == []  # frame 2: tentative miss -> removed
    assert tracker.update([]) == []
    assert len(tracker.tracks) == 0


def test_lost_object_keeps_id_when_it_returns():
    tracker = ByteTrackTracker(confirm_frames=2, lost_window=30)
    tracks = []
    for i in range(8):
        tracks = tracker.update([_det(100 + i * 10, 240)])
    original_id = tracks[0].track_id

    for _ in range(3):  # occluded, within the lost window
        tracker.update([])

    tracks = tracker.update([_det(100 + 11 * 10, 240)])
    assert len(tracks) == 1
    assert tracks[0].track_id == original_id


def test_late_arrival_gets_distinct_id():
    tracker = ByteTrackTracker(confirm_frames=2)
    tracks = []
    for i in range(4):  # confirm object A
        tracks = tracker.update([_det(100 + i * 10, 240)])
    a_ids = {t.track_id for t in tracks}
    assert len(a_ids) == 1

    for i in range(4, 9):  # B appears far away; confirm both
        tracks = tracker.update([_det(100 + i * 10, 240), _det(520, 90)])
    ids = {t.track_id for t in tracks}
    assert a_ids.issubset(ids)
    assert len(ids) == 2


def test_build_tracker_from_config():
    cfg = VigilConfig.from_dict({"tracker": {"confirm_frames": 4, "lost_window": 50}})
    tracker = build_tracker(cfg)
    assert isinstance(tracker, ByteTrackTracker)
    assert tracker.confirm_frames == 4
    assert tracker.lost_window == 50

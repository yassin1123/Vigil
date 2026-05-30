"""Track lifecycle windows + quality metrics. No GPU, no network."""
from __future__ import annotations

from vigil.track import ByteTrackTracker, TrackMetrics, associate_tracks_to_gt
from vigil.types import Detection


def _det(cx: float, cy: float, size: float = 40.0, conf: float = 0.9):
    half = size / 2.0
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_id=0,
        class_name="person",
        confidence=conf,
    )


# --- lifecycle windows ----------------------------------------------------- #


def test_occlusion_within_window_keeps_id():
    tracker = ByteTrackTracker(confirm_frames=2, lost_window=10)
    tracks = []
    for i in range(6):
        tracks = tracker.update([_det(100 + i * 12, 240)])
    original_id = tracks[0].track_id

    for _ in range(4):  # absent for 4 frames < lost_window
        tracker.update([])

    tracks = tracker.update([_det(100 + 10 * 12, 240)])  # near coasted prediction
    assert len(tracks) == 1
    assert tracks[0].track_id == original_id


def test_long_absence_gets_new_id():
    tracker = ByteTrackTracker(confirm_frames=2, lost_window=5)
    tracks = []
    for i in range(6):
        tracks = tracker.update([_det(100 + i * 12, 240)])
    original_id = tracks[0].track_id

    for _ in range(8):  # absent for 8 frames > lost_window -> removed
        tracker.update([])

    tracks = []
    for i in range(3):  # reappears; must re-confirm before output
        tracks = tracker.update([_det(300 + i * 12, 240)])
    assert len(tracks) == 1
    assert tracks[0].track_id != original_id


# --- metrics --------------------------------------------------------------- #


def test_id_switch_and_fragmentation_counter():
    metrics = TrackMetrics()
    # Ground-truth object A is covered by track 1, then switches to track 2.
    # Object B stays on track 7 throughout (no switch).
    sequence = [
        {"A": 1, "B": 7},
        {"A": 1, "B": 7},
        {"A": 1, "B": 7},
        {"A": 2, "B": 7},  # <- one ID switch on A
        {"A": 2, "B": 7},
    ]
    for gt_map in sequence:
        metrics.update(tracks=[], gt_map=gt_map)

    assert metrics.id_switches == 1
    assert metrics.fragmentation == {"A": 2, "B": 1}
    assert metrics.mean_fragmentation == 1.5


def test_metrics_on_clean_crossing_are_ideal():
    tracker = ByteTrackTracker(confirm_frames=2)
    metrics = TrackMetrics()
    for i in range(24):
        a = _det(100 + i * 12, 220)
        b = _det(400 - i * 12, 260)
        tracks = tracker.update([a, b])
        gt_map = associate_tracks_to_gt(tracks, {"A": a.bbox, "B": b.bbox})
        metrics.update(tracks, gt_map)

    assert metrics.id_switches == 0
    assert metrics.mean_fragmentation == 1.0  # one track per object
    assert metrics.active_track_count == 2


def test_mean_track_lifetime_and_totals():
    tracker = ByteTrackTracker(confirm_frames=2)
    metrics = TrackMetrics()
    output_frames = 0
    for i in range(10):
        tracks = tracker.update([_det(100 + i * 10, 240)])
        metrics.update(tracks)
        if tracks:
            output_frames += 1

    assert metrics.total_tracks == 1
    assert metrics.mean_track_lifetime == output_frames
    summary = metrics.summary()
    assert summary["id_switches"] == 0
    assert summary["total_tracks"] == 1

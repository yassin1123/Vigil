"""Vigil tracking: ByteTrack with persistent IDs, no ReID, fully offline.

The primary API is the class: `build_tracker(config)` -> `ByteTrackTracker`, then
`tracker.update(detections, frame_info)`. For quick/one-off use this module also
exposes a process-global default tracker so callers can just do:

    from vigil import track
    tracks = track.update(detections, frame_info)

`configure(config)` swaps the default tracker; `reset()` clears it. Production
code with more than one stream should instantiate its own ByteTrackTracker.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vigil.track.metrics import TrackMetrics, associate_tracks_to_gt
from vigil.track.tracker import (
    ByteTrackTracker,
    KalmanFilter,
    STrack,
    Tracker,
    TrackState,
    build_tracker,
)
from vigil.types import Detection, Track

if TYPE_CHECKING:
    from vigil.config import VigilConfig
    from vigil.frames import Frame

__all__ = [
    "ByteTrackTracker",
    "KalmanFilter",
    "STrack",
    "TrackMetrics",
    "TrackState",
    "Tracker",
    "active_tracks",
    "associate_tracks_to_gt",
    "build_tracker",
    "configure",
    "reset",
    "snapshot",
    "track_by_id",
    "update",
]

# Process-global default tracker (lazily created) for the convenience API.
_default: Optional[ByteTrackTracker] = None


def configure(config: "VigilConfig | None" = None) -> ByteTrackTracker:
    """Set (and return) the default tracker, from config or with defaults."""
    global _default
    _default = build_tracker(config) if config is not None else ByteTrackTracker()
    return _default


def _get() -> ByteTrackTracker:
    global _default
    if _default is None:
        _default = ByteTrackTracker()
    return _default


def update(
    detections: list[Detection], frame_info: "Frame | None" = None
) -> list[Track]:
    """The single clean entry point: associate detections, return active tracks."""
    return _get().update(detections, frame_info)


def reset() -> None:
    _get().reset()


def active_tracks() -> list[Track]:
    return _get().active_tracks()


def track_by_id(track_id: int) -> Optional[Track]:
    return _get().track_by_id(track_id)


def snapshot(frame_info: "Frame | None" = None) -> dict[str, Any]:
    return _get().snapshot(frame_info)

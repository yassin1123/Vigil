"""Vigil tracking: ByteTrack with persistent IDs, no ReID, fully offline."""
from __future__ import annotations

from vigil.track.tracker import ByteTrackTracker, KalmanFilter, Tracker, build_tracker

__all__ = ["ByteTrackTracker", "KalmanFilter", "Tracker", "build_tracker"]

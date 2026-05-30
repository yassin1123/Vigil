"""Vigil — on-device edge inference core (offline computer-vision pipeline)."""
from __future__ import annotations

from vigil.config import VigilConfig, load_config
from vigil.detect import Detector, MockDetector, TensorRTDetector
from vigil.frames import Frame, FrameSource, build_frame_source
from vigil.track import ByteTrackTracker, Tracker, build_tracker
from vigil.types import BBox, Detection, Event, EventType, Track

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "ByteTrackTracker",
    "Detection",
    "Detector",
    "Event",
    "EventType",
    "Frame",
    "FrameSource",
    "MockDetector",
    "TensorRTDetector",
    "Track",
    "Tracker",
    "VigilConfig",
    "__version__",
    "build_frame_source",
    "build_tracker",
    "load_config",
]

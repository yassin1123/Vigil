"""Vigil — on-device edge inference core (offline computer-vision pipeline)."""
from __future__ import annotations

from vigil.config import VigilConfig, load_config
from vigil.frames import Frame, FrameSource, build_frame_source
from vigil.types import BBox, Detection, Event, EventType, Track

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "Detection",
    "Event",
    "EventType",
    "Frame",
    "FrameSource",
    "Track",
    "VigilConfig",
    "__version__",
    "build_frame_source",
    "load_config",
]

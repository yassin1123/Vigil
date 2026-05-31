"""Core typed records used across Vigil.

These are the values that flow through the pipeline: a `Detection` is what the
detector emits, a `Track` is a detection followed across frames, and an `Event`
is something worth logging. All are JSON-serialisable via `to_dict()` /
`from_dict()` so they can cross the tamper-evident log and the web API
unchanged.

`Track` and `Event` are intentionally minimal here — they are placeholders that
grow on Day 3 (ByteTrack) and Days 4–5 (zones + hash-chained log). The fields
that already exist are stable; later days add to them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Axis-aligned bounding box in pixels, top-left origin: (x1, y1, x2, y2).
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection in one frame.

    Immutable and cheap to construct (created per object per frame). Geometry
    helpers are derived, not stored.
    """

    bbox: BBox
    class_id: int
    class_name: str
    confidence: float

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [float(v) for v in self.bbox],
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        b = data["bbox"]
        return cls(
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
        )


@dataclass(slots=True)
class Track:
    """A detection followed across frames with a persistent id.

    Produced by the tracker (Day 3). `bbox` is the current (possibly
    motion-predicted) box; `history` is recent centroids oldest->newest, used
    for drawing trails and reasoning about motion through zones (Day 4).
    """

    track_id: int
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float
    age: int = 0  # frames since the track was created (frames seen)
    time_since_update: int = 0  # frames since the last associated detection
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "bbox": [float(v) for v in self.bbox],
            "confidence": float(self.confidence),
            "centroid": list(self.centroid),
            "age": int(self.age),
            "time_since_update": int(self.time_since_update),
            "history": [[float(x), float(y)] for x, y in self.history],
        }

    def snapshot(self) -> dict[str, Any]:
        """A compact, JSON-friendly view for per-frame snapshots (no history)."""
        cx, cy = self.centroid
        return {
            "id": int(self.track_id),
            "class": str(self.class_name),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "centroid": [round(cx, 1), round(cy, 1)],
            "confidence": round(float(self.confidence), 3),
            "age": int(self.age),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        b = data["bbox"]
        return cls(
            track_id=int(data["track_id"]),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            confidence=float(data["confidence"]),
            age=int(data.get("age", 0)),
            time_since_update=int(data.get("time_since_update", 0)),
            history=[(float(x), float(y)) for x, y in data.get("history", [])],
        )


class EventType(str, Enum):
    """Kinds of loggable event. Grows as the pipeline grows."""

    ZONE_ENTRY = "ZONE_ENTRY"
    ZONE_EXIT = "ZONE_EXIT"


@dataclass(slots=True)
class Event:
    """A zone entry/exit occurrence with full context for the tamper-evident log.

    Carries both clocks: `timestamp_utc` (ISO-8601 wall-clock, for humans) and
    `timestamp_monotonic` (capture monotonic seconds, immune to clock changes).
    This is exactly the record the Day-5 hash-chained logger writes.
    """

    event_type: EventType
    track_id: int
    zone_id: str
    class_name: str
    timestamp_utc: str
    timestamp_monotonic: float
    centroid: tuple[float, float]
    bbox: BBox
    frame_index: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "track_id": int(self.track_id),
            "zone_id": str(self.zone_id),
            "class_name": str(self.class_name),
            "timestamp_utc": str(self.timestamp_utc),
            "timestamp_monotonic": float(self.timestamp_monotonic),
            "centroid": [float(self.centroid[0]), float(self.centroid[1])],
            "bbox": [float(v) for v in self.bbox],
            "frame_index": self.frame_index,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        c = data["centroid"]
        b = data["bbox"]
        return cls(
            event_type=EventType(data["event_type"]),
            track_id=int(data["track_id"]),
            zone_id=str(data["zone_id"]),
            class_name=str(data["class_name"]),
            timestamp_utc=str(data["timestamp_utc"]),
            timestamp_monotonic=float(data["timestamp_monotonic"]),
            centroid=(float(c[0]), float(c[1])),
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            frame_index=data.get("frame_index"),
            detail=dict(data.get("detail", {})),
        )


class SystemEventType(str, Enum):
    """System-level events (not tied to a track), e.g. config reloads, exports."""

    ZONES_RELOADED = "ZONES_RELOADED"
    ZONES_REJECTED = "ZONES_REJECTED"
    LOG_EXPORTED = "LOG_EXPORTED"
    # Records the roadmap export backends WOULD emit once built (rails only;
    # no real code emits these yet — see docs/LOG_EXPORT_ROADMAP.md).
    MESH_PEER_ACK = "MESH_PEER_ACK"
    RF_BURST_SENT = "RF_BURST_SENT"
    ARMORED_WRITE = "ARMORED_WRITE"


@dataclass(slots=True)
class SystemEvent:
    """A system-level occurrence for the tamper-evident log (no track context)."""

    event_type: SystemEventType
    timestamp_utc: str
    timestamp_monotonic: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp_utc": str(self.timestamp_utc),
            "timestamp_monotonic": float(self.timestamp_monotonic),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemEvent":
        return cls(
            event_type=SystemEventType(data["event_type"]),
            timestamp_utc=str(data["timestamp_utc"]),
            timestamp_monotonic=float(data["timestamp_monotonic"]),
            detail=dict(data.get("detail", {})),
        )

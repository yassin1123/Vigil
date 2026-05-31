"""Benchmark clip model: annotated ground-truth sequences.

A clip is an annotation-level sequence (no pixels): per-frame ground-truth object
boxes, a ZoneSet, and the expected ZONE_ENTRY/EXIT events with a timing tolerance.
This lets the benchmark measure tracking stability and zone-event accuracy
deterministically with no GPU. (Real-footage detection benchmarking on the Jetson
reuses the same scorers; see benchmark/README.md.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vigil.types import BBox
from vigil.zones.model import ZoneSet


@dataclass
class GTObject:
    gt_id: str
    cls: str
    boxes: dict[int, BBox]  # TRUE trajectory: every frame the object is present
    occluded: set[int] = field(default_factory=set)  # present but not detectable

    def visible(self, frame: int) -> BBox | None:
        """Box if the object is present AND detectable this frame, else None."""
        if frame in self.occluded:
            return None
        return self.boxes.get(frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_id": self.gt_id,
            "class": self.cls,
            "track": [
                {"frame": f, "bbox": [float(v) for v in self.boxes[f]]}
                for f in sorted(self.boxes)
            ],
            "occluded": sorted(self.occluded),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GTObject":
        boxes = {
            int(p["frame"]): (
                float(p["bbox"][0]), float(p["bbox"][1]),
                float(p["bbox"][2]), float(p["bbox"][3]),
            )
            for p in data["track"]
        }
        return cls(
            gt_id=str(data["gt_id"]),
            cls=str(data["class"]),
            boxes=boxes,
            occluded={int(f) for f in data.get("occluded", [])},
        )


@dataclass
class ExpectedEvent:
    event_type: str  # "ZONE_ENTRY" | "ZONE_EXIT"
    gt_id: str
    zone_id: str
    frame: int
    tolerance: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "gt_id": self.gt_id,
            "zone_id": self.zone_id,
            "frame": int(self.frame),
            "tolerance": int(self.tolerance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedEvent":
        return cls(
            event_type=str(data["event_type"]),
            gt_id=str(data["gt_id"]),
            zone_id=str(data["zone_id"]),
            frame=int(data["frame"]),
            tolerance=int(data.get("tolerance", 3)),
        )


@dataclass
class BenchmarkClip:
    name: str
    tier: int
    held_out: bool
    fps: float
    resolution: tuple[int, int]
    frames: int
    zones: ZoneSet
    objects: list[GTObject]
    expected_events: list[ExpectedEvent]
    annotation: str = "synthetic"  # provenance: synthetic | real
    notes: str = ""

    @property
    def group(self) -> str:
        return "heldout" if self.held_out else f"tier{self.tier}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "held_out": self.held_out,
            "fps": self.fps,
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "frames": self.frames,
            "annotation": self.annotation,
            "notes": self.notes,
            "zones": self.zones.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "expected_events": [e.to_dict() for e in self.expected_events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkClip":
        res = data["resolution"]
        return cls(
            name=str(data["name"]),
            tier=int(data["tier"]),
            held_out=bool(data["held_out"]),
            fps=float(data["fps"]),
            resolution=(int(res[0]), int(res[1])),
            frames=int(data["frames"]),
            zones=ZoneSet.from_dict(data["zones"]),
            objects=[GTObject.from_dict(o) for o in data["objects"]],
            expected_events=[ExpectedEvent.from_dict(e) for e in data["expected_events"]],
            annotation=str(data.get("annotation", "synthetic")),
            notes=str(data.get("notes", "")),
        )

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkClip":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_clips(clips_dir: str | Path) -> list[BenchmarkClip]:
    """Load every clip JSON under clips_dir (recursively), sorted by name."""
    clips_dir = Path(clips_dir)
    clips = [BenchmarkClip.from_file(p) for p in sorted(clips_dir.rglob("*.json"))]
    return clips

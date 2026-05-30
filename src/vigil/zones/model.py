"""Spatial zones: polygons in image space that gate detections and tracks.

Zones are plain local data — defined in a JSON file, never fetched over a
network. Configuration without connectivity is a core Vigil property.

A Zone is a polygon plus a kind:
  INCLUDE  trigger when a (matching) object's centroid is inside it.
  EXCLUDE  ignore detections whose centroid is inside it (e.g. a public road).
An optional class filter restricts a zone to certain classes (e.g. "person").

Polygons are stored in the resolution they were drawn in (`ZoneSet.resolution`,
the camera's full resolution); the geometry layer maps them to whatever frame
resolution tracks arrive in. This module has no third-party dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

Point = tuple[float, float]


class ZoneError(ValueError):
    """Raised when zone data is structurally malformed."""


class ZoneKind(str, Enum):
    INCLUDE = "include"  # trigger when a matching object is inside
    EXCLUDE = "exclude"  # ignore detections inside

    @classmethod
    def parse(cls, value: Any) -> "ZoneKind":
        if isinstance(value, ZoneKind):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ZoneError(
                f"invalid zone kind {value!r} (expected 'include' or 'exclude')"
            ) from exc


@dataclass
class Zone:
    """A named polygon with a kind and an optional class filter."""

    id: str
    name: str
    polygon: list[Point]
    kind: ZoneKind = ZoneKind.INCLUDE
    classes: list[str] = field(default_factory=list)  # empty = all classes

    def accepts(self, class_name: str) -> bool:
        """True if this zone applies to the given class (empty filter = all)."""
        return not self.classes or class_name in self.classes

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "classes": list(self.classes),
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Zone":
        if not isinstance(data, dict):
            raise ZoneError("zone must be a mapping")
        try:
            zone_id = str(data["id"])
            raw_polygon = data["polygon"]
        except KeyError as exc:
            raise ZoneError(f"zone missing required key: {exc}") from exc
        return cls(
            id=zone_id,
            name=str(data.get("name", zone_id)),
            polygon=_parse_polygon(raw_polygon, zone_id),
            kind=ZoneKind.parse(data.get("kind", "include")),
            classes=[str(c) for c in data.get("classes", [])],
        )


def _parse_polygon(raw: Any, zone_id: str) -> list[Point]:
    if not isinstance(raw, (list, tuple)):
        raise ZoneError(f"zone {zone_id!r}: polygon must be a list of [x, y] points")
    points: list[Point] = []
    for p in raw:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ZoneError(f"zone {zone_id!r}: each polygon point must be [x, y]")
        points.append((float(p[0]), float(p[1])))
    return points


@dataclass
class ZoneSet:
    """A collection of zones sharing a reference resolution."""

    resolution: tuple[int, int]  # (width, height) the polygons are defined in
    zones: list[Zone] = field(default_factory=list)

    def __iter__(self):
        return iter(self.zones)

    def __len__(self) -> int:
        return len(self.zones)

    def by_id(self, zone_id: str) -> Zone | None:
        for zone in self.zones:
            if zone.id == zone_id:
                return zone
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "zones": [z.to_dict() for z in self.zones],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ZoneSet":
        if not isinstance(data, dict):
            raise ZoneError("zone set must be a mapping")
        res = data.get("resolution")
        if not (isinstance(res, (list, tuple)) and len(res) == 2):
            raise ZoneError("zone set 'resolution' must be [width, height]")
        resolution = (int(res[0]), int(res[1]))
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ZoneError("zone set 'resolution' must be positive")
        zones_raw = data.get("zones", [])
        if not isinstance(zones_raw, list):
            raise ZoneError("'zones' must be a list")
        zones = [Zone.from_dict(z) for z in zones_raw]
        ids = [z.id for z in zones]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ZoneError(f"duplicate zone id(s): {duplicates}")
        return cls(resolution=resolution, zones=zones)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ZoneSet":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ZoneError(f"invalid zone JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "ZoneSet":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ZoneError(f"cannot read zone file {path}: {exc}") from exc
        return cls.from_json(text)


def load_zone_set(path: str | Path) -> ZoneSet:
    """Load a ZoneSet from a local JSON file (never fetched over a network)."""
    return ZoneSet.from_file(path)

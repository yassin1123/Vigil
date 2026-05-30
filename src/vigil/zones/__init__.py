"""Vigil zones: local polygon zones + image-space geometry (no network)."""
from __future__ import annotations

from vigil.zones.geometry import (
    InvalidPolygonError,
    ZoneIndex,
    bbox_overlap_fraction,
    point_in_polygon,
    scale_points,
    validate_polygon,
)
from vigil.zones.model import (
    Point,
    Zone,
    ZoneError,
    ZoneKind,
    ZoneSet,
    load_zone_set,
)

__all__ = [
    "InvalidPolygonError",
    "Point",
    "Zone",
    "ZoneError",
    "ZoneIndex",
    "ZoneKind",
    "ZoneSet",
    "bbox_overlap_fraction",
    "load_zone_set",
    "point_in_polygon",
    "scale_points",
    "validate_polygon",
]

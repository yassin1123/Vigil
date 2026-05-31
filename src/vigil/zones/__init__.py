"""Vigil zones: local polygon zones + image-space geometry (no network)."""
from __future__ import annotations

from vigil.zones.config import (
    ZoneReloader,
    ZoneValidationError,
    diff_zone_sets,
    load_zones,
    save_zones,
    validate_zone_set,
)
from vigil.zones.engine import ZoneEventEngine, build_zone_engine
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
    "ZoneEventEngine",
    "ZoneIndex",
    "ZoneKind",
    "ZoneReloader",
    "ZoneSet",
    "ZoneValidationError",
    "bbox_overlap_fraction",
    "build_zone_engine",
    "diff_zone_sets",
    "load_zone_set",
    "load_zones",
    "point_in_polygon",
    "save_zones",
    "scale_points",
    "validate_polygon",
    "validate_zone_set",
]

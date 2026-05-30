"""Zone geometry: point-in-polygon and bbox overlap, with coordinate mapping.

Built on shapely (CPU-only). Zones are defined in the camera's full resolution
(`ZoneSet.resolution`); a `ZoneIndex` scales every polygon once to the frame
resolution that tracks actually arrive in, so callers test track centroids and
boxes directly. Invalid polygons (too few points, self-intersecting, zero area)
are validated up front and reported via `InvalidPolygonError` rather than
silently misbehaving.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

from vigil.zones.model import Point, Zone, ZoneKind, ZoneSet

if TYPE_CHECKING:
    from vigil.types import BBox, Track


class InvalidPolygonError(ValueError):
    """A zone's polygon is not a usable simple polygon."""

    def __init__(self, zone_id: str, reason: str) -> None:
        self.zone_id = zone_id
        self.reason = reason
        super().__init__(f"zone {zone_id!r}: {reason}")


def validate_polygon(points: Sequence[Point]) -> tuple[bool, str]:
    """Return (ok, reason). A polygon must have >=3 points, be simple (no
    self-intersection), and enclose positive area."""
    if len(points) < 3:
        return False, f"polygon needs at least 3 points, got {len(points)}"
    try:
        poly = Polygon(points)
    except (ValueError, TypeError) as exc:
        return False, f"cannot build polygon: {exc}"
    if not poly.is_valid:
        return False, explain_validity(poly)
    if poly.area <= 0.0:
        return False, "polygon encloses zero area"
    return True, "ok"


def scale_points(
    points: Sequence[Point],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> list[Point]:
    """Map points from src resolution to dst resolution (independent x/y scale)."""
    sw, sh = src_size
    dw, dh = dst_size
    if sw <= 0 or sh <= 0:
        raise ValueError("source size must be positive")
    fx, fy = dw / sw, dh / sh
    return [(x * fx, y * fy) for x, y in points]


def point_in_polygon(polygon: Polygon, point: Point) -> bool:
    """True if the point is inside or on the boundary of the polygon."""
    return bool(polygon.covers(ShapelyPoint(point)))


def bbox_overlap_fraction(polygon: Polygon, bbox: "BBox") -> float:
    """Fraction of the bbox's area that lies inside the polygon, in [0, 1]."""
    x1, y1, x2, y2 = bbox
    rect = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if rect.area <= 0.0:
        return 0.0
    return float(polygon.intersection(rect).area / rect.area)


class ZoneIndex:
    """Prepared, frame-scaled polygons for fast repeated zone queries.

    Construction validates every polygon (in original *and* scaled space) and
    raises InvalidPolygonError on the first bad one.
    """

    def __init__(self, zone_set: ZoneSet, frame_size: tuple[int, int]) -> None:
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError("frame_size must be positive")
        self.zone_set = zone_set
        self.frame_size = frame_size
        self._polygons: dict[str, Polygon] = {}
        self._zones: dict[str, Zone] = {}
        for zone in zone_set.zones:
            ok, reason = validate_polygon(zone.polygon)
            if not ok:
                raise InvalidPolygonError(zone.id, reason)
            scaled = scale_points(zone.polygon, zone_set.resolution, frame_size)
            poly = Polygon(scaled)
            if not poly.is_valid:  # scaling shouldn't break validity, but verify
                raise InvalidPolygonError(zone.id, explain_validity(poly))
            self._polygons[zone.id] = poly
            self._zones[zone.id] = zone

    def __len__(self) -> int:
        return len(self._polygons)

    def polygon(self, zone_id: str) -> Polygon:
        return self._polygons[zone_id]

    def contains_point(self, zone_id: str, point: Point) -> bool:
        return point_in_polygon(self._polygons[zone_id], point)

    def overlap_fraction(self, zone_id: str, bbox: "BBox") -> float:
        return bbox_overlap_fraction(self._polygons[zone_id], bbox)

    def zones_containing_point(
        self,
        point: Point,
        *,
        kind: Optional[ZoneKind] = None,
        class_name: Optional[str] = None,
    ) -> list[Zone]:
        """Zones whose polygon covers the point, filtered by kind and class."""
        shp = ShapelyPoint(point)
        hits: list[Zone] = []
        for zone_id, poly in self._polygons.items():
            zone = self._zones[zone_id]
            if kind is not None and zone.kind != kind:
                continue
            if class_name is not None and not zone.accepts(class_name):
                continue
            if poly.covers(shp):
                hits.append(zone)
        return hits

    def zones_for_track(
        self, track: "Track", *, kind: ZoneKind = ZoneKind.INCLUDE
    ) -> list[Zone]:
        """INCLUDE (by default) zones a track's centroid is in, class-filtered."""
        return self.zones_containing_point(
            track.centroid, kind=kind, class_name=track.class_name
        )

    def is_excluded(self, point: Point, class_name: Optional[str] = None) -> bool:
        """True if the point falls in any EXCLUDE zone that applies to the class."""
        return bool(
            self.zones_containing_point(
                point, kind=ZoneKind.EXCLUDE, class_name=class_name
            )
        )

"""Zone geometry: point-in-polygon, bbox overlap, class filters, mapping, validation."""
from __future__ import annotations

import pytest

from vigil.types import Track
from vigil.zones.geometry import (
    InvalidPolygonError,
    ZoneIndex,
    bbox_overlap_fraction,
    point_in_polygon,
    scale_points,
    validate_polygon,
)
from vigil.zones.model import Zone, ZoneKind, ZoneSet

# A 100x100 square at the origin, defined at 100x100 resolution.
SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def _track(cx, cy, class_name="person", size=20.0):
    half = size / 2
    return Track(
        track_id=1,
        class_id=0,
        class_name=class_name,
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=0.9,
    )


def _index(zones, resolution=(100, 100), frame_size=(100, 100)):
    return ZoneIndex(ZoneSet(resolution=resolution, zones=zones), frame_size)


# --- point in polygon: inside / outside / on-edge -------------------------- #


def test_point_inside_outside_and_on_edge():
    idx = _index([Zone("s", "S", SQUARE)])
    assert idx.contains_point("s", (50, 50)) is True  # interior
    assert idx.contains_point("s", (150, 50)) is False  # outside
    assert idx.contains_point("s", (0, 50)) is True  # on an edge (inclusive)
    assert idx.contains_point("s", (100, 100)) is True  # on a corner (inclusive)


def test_polygon_on_image_border():
    # Polygon hugging the full frame border; corners and edges are inside.
    border = [(0, 0), (200, 0), (200, 200), (0, 200)]
    idx = _index([Zone("b", "B", border)], resolution=(200, 200), frame_size=(200, 200))
    assert idx.contains_point("b", (0, 0)) is True
    assert idx.contains_point("b", (200, 100)) is True
    assert idx.contains_point("b", (100, 100)) is True


# --- bbox overlap fraction ------------------------------------------------- #


def test_bbox_overlap_fraction():
    idx = _index([Zone("s", "S", SQUARE)])
    assert idx.overlap_fraction("s", (10, 10, 60, 60)) == pytest.approx(1.0)  # fully in
    assert idx.overlap_fraction("s", (50, 50, 150, 150)) == pytest.approx(0.25)  # quarter
    assert idx.overlap_fraction("s", (200, 200, 300, 300)) == pytest.approx(0.0)  # outside


# --- class filter ---------------------------------------------------------- #


def test_class_filter_ignores_wrong_class():
    idx = _index([Zone("p", "People", SQUARE, ZoneKind.INCLUDE, classes=["person"])])
    assert idx.zones_for_track(_track(50, 50, "car")) == []  # wrong class -> ignored
    hit = idx.zones_for_track(_track(50, 50, "person"))
    assert len(hit) == 1 and hit[0].id == "p"


def test_exclude_zone_membership():
    idx = _index([Zone("road", "Road", SQUARE, ZoneKind.EXCLUDE)])
    assert idx.is_excluded((50, 50)) is True
    assert idx.is_excluded((150, 150)) is False
    # an INCLUDE-only query finds nothing in an EXCLUDE zone
    assert idx.zones_containing_point((50, 50), kind=ZoneKind.INCLUDE) == []


# --- coordinate mapping (capture vs inference resolution) ------------------ #


def test_coordinate_mapping_between_resolutions():
    # Zone drawn at 1920x1080; tracks arrive at 960x540 (half scale).
    zone = Zone("z", "Z", [(200, 200), (600, 200), (600, 600), (200, 600)])
    idx = ZoneIndex(ZoneSet(resolution=(1920, 1080), zones=[zone]), frame_size=(960, 540))
    # Frame point (200, 200) maps to zone (400, 400) -> inside the 200..600 box.
    assert idx.contains_point("z", (200, 200)) is True
    # Frame point (50, 50) maps to zone (100, 100) -> outside.
    assert idx.contains_point("z", (50, 50)) is False
    assert idx.zones_for_track(_track(200, 200)) [0].id == "z"


def test_scale_points_helper():
    assert scale_points([(100, 100)], (200, 200), (100, 50)) == [(50.0, 25.0)]
    with pytest.raises(ValueError):
        scale_points([(1, 1)], (0, 10), (10, 10))


# --- invalid polygons reported cleanly ------------------------------------- #


def test_validate_polygon_rejects_bad_shapes():
    ok, _ = validate_polygon(SQUARE)
    assert ok
    too_few, reason = validate_polygon([(0, 0), (1, 1)])
    assert not too_few and "3 points" in reason
    # bowtie / self-intersection
    bad, reason = validate_polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bad and reason  # a non-empty explanation


def test_zone_index_raises_on_invalid_polygon():
    bowtie = Zone("x", "X", [(0, 0), (10, 10), (10, 0), (0, 10)])
    with pytest.raises(InvalidPolygonError) as exc:
        _index([bowtie])
    assert exc.value.zone_id == "x"
    assert exc.value.reason  # carries a human-readable reason

    with pytest.raises(InvalidPolygonError):
        _index([Zone("y", "Y", [(0, 0), (1, 1)])])  # too few points


def test_point_in_polygon_direct_helper():
    from shapely.geometry import Polygon

    poly = Polygon(SQUARE)
    assert point_in_polygon(poly, (50, 50)) is True
    assert point_in_polygon(poly, (-1, -1)) is False
    assert bbox_overlap_fraction(poly, (0, 0, 100, 100)) == pytest.approx(1.0)

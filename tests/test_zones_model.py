"""Zone model: serialization, class filter, validation. No third-party deps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigil.zones.model import Zone, ZoneError, ZoneKind, ZoneSet, load_zone_set

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "zones.example.json"


def test_zone_accepts_class_filter():
    z = Zone(id="d", name="Dock", polygon=[(0, 0), (10, 0), (10, 10)], classes=["person"])
    assert z.accepts("person")
    assert not z.accepts("car")
    # empty filter accepts everything
    assert Zone(id="a", name="All", polygon=[(0, 0), (1, 0), (1, 1)]).accepts("car")


def test_zone_set_roundtrip():
    zs = ZoneSet(
        resolution=(1920, 1080),
        zones=[
            Zone("a", "A", [(0, 0), (100, 0), (100, 100)], ZoneKind.INCLUDE, ["person"]),
            Zone("b", "B", [(0, 0), (50, 0), (50, 50)], ZoneKind.EXCLUDE),
        ],
    )
    data = zs.to_dict()
    json.dumps(data)  # JSON-serialisable
    back = ZoneSet.from_dict(data)
    assert back == zs
    assert ZoneSet.from_json(zs.to_json()) == zs


def test_zone_kind_parsing_is_case_insensitive():
    assert ZoneKind.parse("INCLUDE") == ZoneKind.INCLUDE
    assert ZoneKind.parse("Exclude") == ZoneKind.EXCLUDE
    with pytest.raises(ZoneError):
        ZoneKind.parse("maybe")


def test_by_id_and_len():
    zs = ZoneSet.from_file(EXAMPLE)
    assert len(zs) == 2
    assert zs.by_id("loading-dock").kind == ZoneKind.INCLUDE
    assert zs.by_id("public-road").kind == ZoneKind.EXCLUDE
    assert zs.by_id("nope") is None


def test_example_file_loads_and_roundtrips():
    zs = load_zone_set(EXAMPLE)
    assert zs.resolution == (1920, 1080)
    assert ZoneSet.from_dict(zs.to_dict()) == zs


def test_malformed_zone_data_reported():
    with pytest.raises(ZoneError):
        ZoneSet.from_dict({"zones": []})  # missing resolution
    with pytest.raises(ZoneError):
        Zone.from_dict({"id": "x"})  # missing polygon
    with pytest.raises(ZoneError):
        Zone.from_dict({"id": "x", "polygon": [[0, 0], [1]]})  # bad point
    with pytest.raises(ZoneError):
        ZoneSet.from_dict(
            {"resolution": [10, 10], "zones": [
                {"id": "dup", "polygon": [[0, 0], [1, 0], [1, 1]]},
                {"id": "dup", "polygon": [[0, 0], [1, 0], [1, 1]]},
            ]}
        )  # duplicate ids

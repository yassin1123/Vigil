"""Zone config loading, validation, and hot-reload. Local JSON, no network."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vigil.__main__ import main
from vigil.types import SystemEventType
from vigil.zones.config import (
    ZoneReloader,
    ZoneValidationError,
    diff_zone_sets,
    load_zones,
    validate_zone_set,
)
from vigil.zones.model import ZoneError, ZoneSet

SQUARE = [[0, 0], [50, 0], [50, 50], [0, 50]]
BOWTIE = [[0, 0], [10, 10], [10, 0], [0, 10]]


def _zone(zid, polygon=SQUARE, kind="include", classes=None):
    return {"id": zid, "name": zid.upper(), "kind": kind,
            "classes": classes or [], "polygon": polygon}


def _write(path: Path, zones, resolution=(100, 100), mtime: int | None = None):
    path.write_text(json.dumps({"resolution": list(resolution), "zones": zones}))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


# --- validation ------------------------------------------------------------ #


def test_load_valid_file(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a", classes=["person"])])
    zs = load_zones(p)
    assert len(zs) == 1 and zs.by_id("a").classes == ["person"]


def test_unknown_class_is_rejected(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a", classes=["unicorn"])])
    with pytest.raises(ZoneValidationError) as exc:
        load_zones(p)
    assert "unicorn" in str(exc.value)


def test_invalid_polygon_is_rejected(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a", polygon=BOWTIE)])
    with pytest.raises(ZoneError):
        load_zones(p)


def test_duplicate_ids_rejected_at_parse(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a"), _zone("a")])
    with pytest.raises(ZoneError):
        load_zones(p)


def test_validate_zone_set_lists_all_issues():
    zs = ZoneSet.from_dict(
        {"resolution": [100, 100], "zones": [_zone("a", classes=["nope"])]}
    )
    issues = validate_zone_set(zs)
    assert any("nope" in i for i in issues)


def test_diff_zone_sets():
    a = ZoneSet.from_dict({"resolution": [100, 100], "zones": [_zone("a"), _zone("c")]})
    b = ZoneSet.from_dict(
        {
            "resolution": [100, 100],
            "zones": [_zone("a", polygon=[[1, 1], [9, 1], [9, 9]]), _zone("b")],
        }
    )
    d = diff_zone_sets(a, b)
    assert d == {"added": ["b"], "removed": ["c"], "modified": ["a"], "count": 2}


# --- hot reload ------------------------------------------------------------ #


def test_reloader_no_change_returns_none(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a")], mtime=1_000)
    reloader = ZoneReloader(p, frame_size=(100, 100), utc_now=lambda: "T")
    assert reloader.poll() is None


def test_reloader_applies_valid_change_with_summary(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a"), _zone("c")], mtime=1_000)
    reloader = ZoneReloader(p, frame_size=(100, 100), utc_now=lambda: "T")

    _write(p, [_zone("a", polygon=[[1, 1], [9, 1], [9, 9]]), _zone("b")], mtime=2_000)
    event = reloader.poll()

    assert event is not None
    assert event.event_type == SystemEventType.ZONES_RELOADED
    assert event.detail["added"] == ["b"]
    assert event.detail["removed"] == ["c"]
    assert event.detail["modified"] == ["a"]
    assert event.detail["count"] == 2
    # the new set is applied live
    assert reloader.zone_set.by_id("b") is not None
    assert "b" in {z.id for z in reloader.index.zone_set}


def test_reloader_rejects_bad_edit_and_keeps_previous(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a")], mtime=1_000)
    reloader = ZoneReloader(p, frame_size=(100, 100), utc_now=lambda: "T")
    good = reloader.zone_set

    _write(p, [_zone("a", polygon=BOWTIE)], mtime=2_000)  # malformed edit
    event = reloader.poll()

    assert event is not None
    assert event.event_type == SystemEventType.ZONES_REJECTED
    assert "error" in event.detail
    # previous good set retained, no crash
    assert reloader.zone_set is good
    assert reloader.zone_set.by_id("a") is not None


def test_reloader_initial_invalid_raises(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a", polygon=BOWTIE)])
    with pytest.raises(ZoneError):
        ZoneReloader(p, frame_size=(100, 100))


# --- CLI ------------------------------------------------------------------- #


def test_cli_validate_ok_and_bad(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a", classes=["person"])])
    assert main(["zones", "validate", str(p)]) == 0
    _write(p, [_zone("a", polygon=BOWTIE)])
    assert main(["zones", "validate", str(p)]) == 1


def test_cli_show_returns_zero(tmp_path):
    p = tmp_path / "z.json"
    _write(p, [_zone("a"), _zone("b", kind="exclude")])
    assert main(["zones", "show", str(p)]) == 0


def test_cli_show_overlay(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    p = tmp_path / "z.json"
    _write(p, [_zone("a")], resolution=(100, 100))
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.zeros((50, 50, 3), dtype=np.uint8))
    out = tmp_path / "overlay.png"
    assert main(["zones", "show", str(p), "--overlay", str(still), "--out", str(out)]) == 0
    assert out.exists()

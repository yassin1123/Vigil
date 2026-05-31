"""Zone configuration: thorough validation + on-device hot-reload.

Zones live in a local JSON file (`zones.json`) — operator-editable, never
fetched over a network. `load_zones` parses and validates thoroughly (polygon
validity, unique ids, known class names) with clear, aggregated errors.

`ZoneReloader` watches the file by mtime (no external services) and applies edits
live: on a valid change it swaps in the new ZoneSet and returns a ZONES_RELOADED
system event summarising what changed; on a bad edit it keeps the last good set
and returns a ZONES_REJECTED event with the reason. It never raises mid-run.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from vigil.detect.coco import COCO_CLASSES
from vigil.types import SystemEvent, SystemEventType
from vigil.zones.geometry import ZoneIndex, validate_polygon
from vigil.zones.model import ZoneError, ZoneSet


class ZoneValidationError(ZoneError):
    """A zone set parsed but failed semantic validation."""

    def __init__(self, path: object, issues: list[str]) -> None:
        self.path = str(path)
        self.issues = issues
        joined = "\n  - ".join(issues)
        super().__init__(f"{len(issues)} problem(s) in {self.path}:\n  - {joined}")


def validate_zone_set(
    zone_set: ZoneSet, *, class_names: Sequence[str] = COCO_CLASSES
) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    issues: list[str] = []
    if zone_set.resolution[0] <= 0 or zone_set.resolution[1] <= 0:
        issues.append("resolution must be positive")

    ids = [z.id for z in zone_set]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        issues.append(f"duplicate zone id: {dup!r}")

    known = set(class_names)
    for zone in zone_set:
        ok, reason = validate_polygon(zone.polygon)
        if not ok:
            issues.append(f"zone {zone.id!r}: invalid polygon: {reason}")
        for cls in zone.classes:
            if cls not in known:
                issues.append(f"zone {zone.id!r}: unknown class {cls!r}")
    return issues


def load_zones(
    path: str | Path, *, class_names: Sequence[str] = COCO_CLASSES
) -> ZoneSet:
    """Load and fully validate a zones JSON file. Raises ZoneError on any problem."""
    zone_set = ZoneSet.from_file(path)  # structural parse (raises ZoneError)
    issues = validate_zone_set(zone_set, class_names=class_names)
    if issues:
        raise ZoneValidationError(path, issues)
    return zone_set


def save_zones(zone_set: ZoneSet, path: str | Path) -> None:
    """Write a zone set to a local JSON file."""
    zone_set.to_file(path)


def diff_zone_sets(old: ZoneSet, new: ZoneSet) -> dict[str, object]:
    """Summarise the change between two zone sets (added/removed/modified ids)."""
    old_map = {z.id: z.to_dict() for z in old}
    new_map = {z.id: z.to_dict() for z in new}
    old_ids, new_ids = set(old_map), set(new_map)
    return {
        "added": sorted(new_ids - old_ids),
        "removed": sorted(old_ids - new_ids),
        "modified": sorted(i for i in (old_ids & new_ids) if old_map[i] != new_map[i]),
        "count": len(new_map),
    }


def _default_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class ZoneReloader:
    """Holds the current good ZoneSet/ZoneIndex and hot-reloads on file change."""

    def __init__(
        self,
        path: str | Path,
        frame_size: tuple[int, int],
        *,
        class_names: Sequence[str] = COCO_CLASSES,
        utc_now: Optional[Callable[[], str]] = None,
    ) -> None:
        self.path = Path(path)
        self.frame_size = frame_size
        self.class_names = class_names
        self._utc_now = utc_now or _default_utc
        self.zone_set = load_zones(self.path, class_names=class_names)  # initial must be valid
        self.index = ZoneIndex(self.zone_set, frame_size)
        self._mtime = self._current_mtime()

    def _current_mtime(self) -> Optional[int]:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def poll(self, frame_info: object | None = None) -> Optional[SystemEvent]:
        """Check the file; on change, reload (or reject). Returns the event, if any."""
        mtime = self._current_mtime()
        if mtime is None or mtime == self._mtime:
            return None
        self._mtime = mtime

        ts_mono = getattr(frame_info, "timestamp", None)
        if ts_mono is None:
            ts_mono = time.monotonic()
        utc = self._utc_now()

        try:
            new_set = load_zones(self.path, class_names=self.class_names)
            new_index = ZoneIndex(new_set, self.frame_size)
        except (ZoneError, ValueError, OSError) as exc:
            return SystemEvent(
                SystemEventType.ZONES_REJECTED,
                utc,
                float(ts_mono),
                detail={"path": str(self.path), "error": str(exc)},
            )

        summary = diff_zone_sets(self.zone_set, new_set)
        self.zone_set = new_set
        self.index = new_index
        return SystemEvent(
            SystemEventType.ZONES_RELOADED,
            utc,
            float(ts_mono),
            detail={"path": str(self.path), **summary},
        )

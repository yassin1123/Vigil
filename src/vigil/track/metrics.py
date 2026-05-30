"""Track-quality metrics over a run.

Computed purely from the per-frame Track output (plus optional ground truth), so
the same code serves unit tests, the Day-8 benchmark, and a live run — with no
coupling to the tracker internals.

Metrics:
  * id_switches        — times a ground-truth object's assigned track id changed.
  * fragmentation      — distinct track ids assigned to each ground-truth object
                         (1 is ideal; >1 means the object was tracked in pieces).
  * mean_track_lifetime — mean number of frames each track id stayed active.
  * active_track_count — confirmed tracks reported in the most recent frame.
  * total_tracks       — distinct track ids ever reported.

ID switches and fragmentation need ground truth (which object is which); supply
it per frame via `associate_tracks_to_gt`. Lifetime/active/total need none.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Optional

from vigil.types import BBox, Track


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def associate_tracks_to_gt(
    tracks: list[Track],
    gt_boxes: dict[Hashable, BBox],
    iou_threshold: float = 0.3,
) -> dict[Hashable, Optional[int]]:
    """Greedy IoU assignment of tracks to ground-truth boxes for one frame.

    Returns {gt_id: track_id or None}. Each track is used at most once.
    """
    result: dict[Hashable, Optional[int]] = {gt_id: None for gt_id in gt_boxes}
    pairs = [
        (_iou(gbox, tr.bbox), gt_id, tr.track_id)
        for gt_id, gbox in gt_boxes.items()
        for tr in tracks
        if _iou(gbox, tr.bbox) >= iou_threshold
    ]
    pairs.sort(key=lambda p: p[0], reverse=True)
    used_gt: set[Hashable] = set()
    used_track: set[int] = set()
    for _, gt_id, track_id in pairs:
        if gt_id in used_gt or track_id in used_track:
            continue
        result[gt_id] = track_id
        used_gt.add(gt_id)
        used_track.add(track_id)
    return result


@dataclass
class TrackMetrics:
    """Accumulates track-quality metrics frame by frame."""

    frames: int = 0
    id_switches: int = 0
    _active_per_frame: list[int] = field(default_factory=list)
    _first_seen: dict[int, int] = field(default_factory=dict)
    _last_seen: dict[int, int] = field(default_factory=dict)
    _gt_last_track: dict[Hashable, int] = field(default_factory=dict)
    _gt_track_ids: dict[Hashable, set] = field(
        default_factory=lambda: defaultdict(set)
    )

    def update(
        self,
        tracks: list[Track],
        gt_map: Optional[dict[Hashable, Optional[int]]] = None,
    ) -> None:
        """Record one frame of tracker output (and optional gt association)."""
        self.frames += 1
        self._active_per_frame.append(len(tracks))
        for track in tracks:
            self._first_seen.setdefault(track.track_id, self.frames)
            self._last_seen[track.track_id] = self.frames

        if gt_map:
            for gt_id, track_id in gt_map.items():
                if track_id is None:
                    continue
                previous = self._gt_last_track.get(gt_id)
                if previous is not None and previous != track_id:
                    self.id_switches += 1
                self._gt_last_track[gt_id] = track_id
                self._gt_track_ids[gt_id].add(track_id)

    @property
    def active_track_count(self) -> int:
        return self._active_per_frame[-1] if self._active_per_frame else 0

    @property
    def total_tracks(self) -> int:
        return len(self._first_seen)

    @property
    def mean_track_lifetime(self) -> float:
        if not self._first_seen:
            return 0.0
        lifetimes = [
            self._last_seen[tid] - self._first_seen[tid] + 1 for tid in self._first_seen
        ]
        return sum(lifetimes) / len(lifetimes)

    @property
    def fragmentation(self) -> dict[Hashable, int]:
        """Per ground-truth object: how many distinct track ids covered it."""
        return {gt_id: len(ids) for gt_id, ids in self._gt_track_ids.items()}

    @property
    def mean_fragmentation(self) -> float:
        frag = self.fragmentation
        return sum(frag.values()) / len(frag) if frag else 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "frames": self.frames,
            "total_tracks": self.total_tracks,
            "active_track_count": self.active_track_count,
            "id_switches": self.id_switches,
            "mean_track_lifetime": round(self.mean_track_lifetime, 3),
            "mean_fragmentation": round(self.mean_fragmentation, 3),
        }

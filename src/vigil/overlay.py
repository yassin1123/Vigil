"""Frame annotation helpers for the operator feed.

Draws detections and tracks (boxes, ids, class labels, motion trails) onto a
BGR image. cv2 is imported lazily so importing this module needs no OpenCV.
Reused by `vigil run --show` now and the Day-6 operator UI later.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from vigil.types import Detection, Track

if TYPE_CHECKING:
    from vigil.zones.model import ZoneSet

# Distinct-ish BGR palette indexed by track id.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    color: tuple[int, int, int] = (180, 180, 180),
) -> np.ndarray:
    """Draw raw detection boxes (no ids) on a copy of `image`."""
    import cv2

    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            out, label, (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return out


def draw_tracks(
    image: np.ndarray,
    tracks: Sequence[Track],
    draw_trail: bool = True,
) -> np.ndarray:
    """Draw track boxes with stable-id colors, labels, and motion trails."""
    import cv2

    out = image.copy()
    for track in tracks:
        color = color_for_id(track.track_id)
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{track.track_id} {track.class_name}"
        cv2.putText(
            out, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
        if draw_trail and len(track.history) > 1:
            pts = np.asarray(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2)
    return out


def draw_zones(image: np.ndarray, zone_set: "ZoneSet") -> np.ndarray:
    """Draw zone polygons (scaled to the image) — green INCLUDE, red EXCLUDE."""
    import cv2

    from vigil.zones.geometry import scale_points

    out = image.copy()
    h, w = out.shape[:2]
    for zone in zone_set:
        pts = scale_points(zone.polygon, zone_set.resolution, (w, h))
        poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        color = (40, 200, 40) if zone.kind.value == "include" else (40, 40, 220)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2)
        x, y = int(pts[0][0]), int(pts[0][1])
        cv2.putText(
            out, f"{zone.id} [{zone.kind.value}]", (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out

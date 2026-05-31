#!/usr/bin/env python3
"""Generate Vigil's benchmark clips (committed for reproducibility).

Each clip is an annotation-level ground-truth sequence (no pixels): object
trajectories, a ZoneSet, and the EXPECTED zone events derived independently from
the true trajectories (point-in-polygon + a K-frame debounce) — NOT from Vigil's
own engine, so the benchmark measures the pipeline rather than restating it.

Run:  PYTHONPATH=src python3 benchmark/generate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vigil.eval.clips import BenchmarkClip, ExpectedEvent, GTObject  # noqa: E402
from vigil.types import BBox  # noqa: E402
from vigil.zones.geometry import ZoneIndex  # noqa: E402
from vigil.zones.model import Zone, ZoneKind, ZoneSet  # noqa: E402

RES = (640, 480)
FPS = 30.0
DEBOUNCE = 3
TOLERANCE = 3
CLIPS_DIR = Path(__file__).resolve().parent / "clips"


def _box(cx: float, cy: float, size: float) -> BBox:
    h = size / 2.0
    return (cx - h, cy - h, cx + h, cy + h)


def _centroid(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def line(gt_id, cls, p0, p1, f0, f1, size=40.0, occluded=()):
    boxes = {}
    span = max(f1 - f0, 1)
    for f in range(f0, f1 + 1):
        t = (f - f0) / span
        cx = p0[0] + (p1[0] - p0[0]) * t
        cy = p0[1] + (p1[1] - p0[1]) * t
        boxes[f] = _box(cx, cy, size)
    return GTObject(gt_id, cls, boxes, set(occluded))


def jitter_object(gt_id, cls, x_out, x_in, cy, f0, f1, size=40.0):
    """Straddle a vertical boundary: centroid alternates outside/inside each frame."""
    boxes = {}
    for f in range(f0, f1 + 1):
        cx = x_in if f % 2 else x_out
        boxes[f] = _box(cx, cy, size)
    return GTObject(gt_id, cls, boxes, set())


def derive_expected_events(zones, objects, frames, debounce=DEBOUNCE, tolerance=TOLERANCE):
    index = ZoneIndex(zones, RES)
    include_zones = [z for z in zones if z.kind == ZoneKind.INCLUDE]
    exclude_zones = [z for z in zones if z.kind == ZoneKind.EXCLUDE]
    expected: list[ExpectedEvent] = []
    for obj in objects:
        for zone in include_zones:
            reported = False
            streak = 0
            for f in range(frames):
                box = obj.boxes.get(f)  # true presence (occlusion keeps a box)
                if box is None:
                    inside = False
                else:
                    c = _centroid(box)
                    excluded = any(
                        z.accepts(obj.cls) and index.contains_point(z.id, c)
                        for z in exclude_zones
                    )
                    inside = (
                        not excluded
                        and zone.accepts(obj.cls)
                        and index.contains_point(zone.id, c)
                    )
                if inside != reported:
                    streak += 1
                    if streak >= debounce:
                        reported = inside
                        streak = 0
                        kind = "ZONE_ENTRY" if inside else "ZONE_EXIT"
                        expected.append(
                            ExpectedEvent(kind, obj.gt_id, zone.id, f, tolerance)
                        )
                else:
                    streak = 0
    return expected


def zone(zid, name, polygon, kind=ZoneKind.INCLUDE, classes=None):
    return Zone(zid, name, polygon, kind, classes or [])


def make_clip(name, tier, held_out, frames, zones, objects, notes):
    expected = derive_expected_events(zones, objects, frames)
    return BenchmarkClip(
        name=name, tier=tier, held_out=held_out, fps=FPS, resolution=RES,
        frames=frames, zones=zones, objects=objects,
        expected_events=expected, annotation="synthetic", notes=notes,
    )


CENTER = [(220, 140), (420, 140), (420, 340), (220, 340)]


def build_clips():
    clips = []

    # --- Tier 1: a single clear object crossing one zone --------------------
    z1 = ZoneSet(RES, [zone("z1", "Center", CENTER)])
    clips.append(make_clip(
        "t1_single_cross", 1, False, 44, z1,
        [line("A", "person", (100, 240), (560, 240), 0, 43)],
        "one person crosses one INCLUDE zone left-to-right",
    ))

    # --- Tier 2: occlusion, crossing paths, boundary jitter -----------------
    clips.append(make_clip(
        "t2_occlusion", 2, False, 44, ZoneSet(RES, [zone("z1", "Center", CENTER)]),
        [line("A", "person", (100, 240), (560, 240), 0, 43, occluded=(22, 23, 24, 25))],
        "person crosses the zone but is occluded for 4 frames mid-zone",
    ))
    clips.append(make_clip(
        "t2_crossing", 2, False, 44, ZoneSet(RES, [zone("z1", "Center", CENTER)]),
        [
            line("A", "person", (60, 200), (580, 200), 0, 43),
            line("B", "person", (580, 260), (60, 260), 0, 43),
        ],
        "two people cross paths while both traversing the zone",
    ))
    clips.append(make_clip(
        "t2_jitter", 2, False, 40, ZoneSet(RES, [zone("z1", "Center", CENTER)]),
        [jitter_object("A", "person", x_out=210, x_in=230, cy=240, f0=0, f1=39)],
        "person straddles the left zone edge; debounce must prevent event spam",
    ))

    # --- Tier 3: busy scene, multiple zones, mixed classes ------------------
    z3 = ZoneSet(RES, [
        zone("dock", "Dock (person only)", [(60, 60), (300, 60), (300, 300), (60, 300)],
             classes=["person"]),
        zone("road", "Road (ignore)", [(360, 200), (620, 200), (620, 440), (360, 440)],
             kind=ZoneKind.EXCLUDE),
    ])
    clips.append(make_clip(
        "t3_busy", 3, False, 50, z3,
        [
            line("P1", "person", (40, 120), (340, 120), 0, 49),       # enters dock
            line("C1", "car", (40, 180), (340, 180), 0, 49),          # car: ignored by dock
            line("P2", "person", (620, 320), (360, 320), 0, 49),      # person in road: excluded
        ],
        "person triggers dock; a car is ignored (class filter); a person in the "
        "road EXCLUDE zone is suppressed",
    ))

    # --- Held-out: novel geometry the params were never tuned against -------
    zH1 = ZoneSet(RES, [zone("gate", "Gate", [(150, 150), (490, 150), (490, 330), (150, 330)])])
    clips.append(make_clip(
        "h1_diag_occlusion", 2, True, 44, zH1,
        [line("A", "person", (80, 100), (560, 380), 0, 43, occluded=(24, 25))],
        "held-out: diagonal crossing of a different zone with brief occlusion",
    ))
    zH2 = ZoneSet(RES, [
        zone("yard", "Yard (person only)", [(200, 80), (440, 80), (440, 400), (200, 400)],
             classes=["person"]),
    ])
    clips.append(make_clip(
        "h2_two_persons_one_car", 2, True, 50, zH2,
        [
            line("P1", "person", (60, 160), (580, 160), 0, 49),
            line("P2", "person", (60, 320), (580, 320), 0, 49),
            line("C1", "car", (60, 240), (580, 240), 0, 49),  # ignored (person-only)
        ],
        "held-out: two people cross a person-only zone; a car is ignored",
    ))
    return clips


def main() -> int:
    layout = {1: "tier1", 2: "tier2", 3: "tier3"}
    written = 0
    for clip in build_clips():
        sub = "heldout" if clip.held_out else layout[clip.tier]
        out_dir = CLIPS_DIR / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{clip.name}.json"
        clip.to_file(path)
        print(f"  wrote {path.relative_to(CLIPS_DIR.parent)}  "
              f"(frames={clip.frames}, objects={len(clip.objects)}, "
              f"expected_events={len(clip.expected_events)})")
        written += 1
    print(f"generated {written} benchmark clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

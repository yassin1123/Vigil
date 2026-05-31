"""Reference detector for off-GPU benchmarking — a deterministic stand-in.

It replays a clip's ground-truth boxes with controlled, seeded imperfection:
missed detections, localization jitter, and occasional false positives. This is
NOT the YOLOv8n model — it is a fixed, reproducible imperfect detector so the
benchmark can stress Vigil's tracking + zone code off-GPU. The real model's
detection P/R is measured on the Jetson against real footage (same scorers).
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from vigil.detect.coco import COCO_CLASSES
from vigil.eval.clips import BenchmarkClip
from vigil.types import BBox, Detection

_FALLBACK_CLASS_ID = 0


def coco_id(name: str) -> int:
    try:
        return COCO_CLASSES.index(name)
    except ValueError:
        return _FALLBACK_CLASS_ID


@dataclass(frozen=True)
class NoiseModel:
    """Fixed, documented detector-imperfection levels (reproducible)."""

    miss_rate: float = 0.08  # fraction of true detections dropped per frame
    jitter_px: float = 4.0  # uniform localization noise on each box edge
    fp_rate: float = 0.03  # chance of a spurious detection per frame
    conf_lo: float = 0.65  # true-detection confidence range
    conf_hi: float = 0.95
    fp_conf_lo: float = 0.30  # false-positive confidence range (mostly sub-threshold)
    fp_conf_hi: float = 0.55
    fp_classes: tuple[str, ...] = ("person", "car")


def clip_seed(name: str) -> int:
    """Stable per-clip seed (CRC32 of the name) for reproducible noise."""
    return zlib.crc32(name.encode("utf-8"))


def _clamp_box(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    return (x1, y1, x2, y2)


def build_detection_script(
    clip: BenchmarkClip, noise: NoiseModel, seed: int | None = None
) -> list[list[Detection]]:
    """Per-frame detection lists the reference detector emits for this clip."""
    rng = random.Random(clip_seed(clip.name) if seed is None else seed)
    width, height = clip.resolution
    script: list[list[Detection]] = []

    for frame in range(clip.frames):
        dets: list[Detection] = []
        for obj in clip.objects:
            box = obj.visible(frame)
            if box is None:
                continue  # absent or occluded — even a perfect detector sees nothing
            if rng.random() < noise.miss_rate:
                continue  # detector miss
            jittered = tuple(c + rng.uniform(-noise.jitter_px, noise.jitter_px) for c in box)
            dets.append(
                Detection(
                    bbox=_clamp_box(jittered, width, height),  # type: ignore[arg-type]
                    class_id=coco_id(obj.cls),
                    class_name=obj.cls,
                    confidence=rng.uniform(noise.conf_lo, noise.conf_hi),
                )
            )
        if rng.random() < noise.fp_rate:
            cls = rng.choice(noise.fp_classes)
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            half = 18.0
            dets.append(
                Detection(
                    bbox=_clamp_box((cx - half, cy - half, cx + half, cy + half), width, height),
                    class_id=coco_id(cls),
                    class_name=cls,
                    confidence=rng.uniform(noise.fp_conf_lo, noise.fp_conf_hi),
                )
            )
        script.append(dets)
    return script


def ground_truth_detections(clip: BenchmarkClip) -> list[list[Detection]]:
    """The clean per-frame ground-truth detections (confidence 1.0)."""
    out: list[list[Detection]] = []
    for frame in range(clip.frames):
        out.append(
            [
                Detection(
                    bbox=obj.visible(frame),  # type: ignore[arg-type]
                    class_id=coco_id(obj.cls),
                    class_name=obj.cls,
                    confidence=1.0,
                )
                for obj in clip.objects
                if obj.visible(frame) is not None
            ]
        )
    return out

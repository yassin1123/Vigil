"""Scorers: detection P/R, tracking stability, zone-event accuracy.

Model-agnostic — they compare pipeline OUTPUT (detections, tracks, events) to
ground truth, so the same code scores the reference detector off-GPU and the real
TensorRT detector on the Jetson.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from vigil.eval.clips import BenchmarkClip, ExpectedEvent
from vigil.track.metrics import TrackMetrics, associate_tracks_to_gt
from vigil.types import BBox, Detection, Event, Track


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


def _pr(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def score_detections(
    pipeline_per_frame: list[list[Detection]],
    gt_per_frame: list[list[Detection]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-class + overall precision/recall by greedy IoU matching per frame."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for dets, gts in zip(pipeline_per_frame, gt_per_frame):
        by_class_det: dict[str, list[Detection]] = defaultdict(list)
        by_class_gt: dict[str, list[Detection]] = defaultdict(list)
        for d in dets:
            by_class_det[d.class_name].append(d)
        for g in gts:
            by_class_gt[g.class_name].append(g)

        for cls in set(by_class_det) | set(by_class_gt):
            preds = by_class_det.get(cls, [])
            truth = by_class_gt.get(cls, [])
            pairs = sorted(
                (
                    (_iou(p.bbox, t.bbox), pi, ti)
                    for pi, p in enumerate(preds)
                    for ti, t in enumerate(truth)
                ),
                reverse=True,
            )
            used_pred: set[int] = set()
            used_truth: set[int] = set()
            for iou_val, pi, ti in pairs:
                if iou_val < iou_threshold:
                    break
                if pi in used_pred or ti in used_truth:
                    continue
                used_pred.add(pi)
                used_truth.add(ti)
                tp[cls] += 1
            fp[cls] += len(preds) - len(used_pred)
            fn[cls] += len(truth) - len(used_truth)

    per_class = {c: _pr(tp[c], fp[c], fn[c]) for c in set(tp) | set(fp) | set(fn)}
    overall = _pr(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {"overall": overall, "per_class": per_class}


def score_tracking(
    clip: BenchmarkClip, tracks_per_frame: list[list[Track]]
) -> dict[str, Any]:
    """ID switches, fragmentation, mean lifetime via gt association per frame."""
    metrics = TrackMetrics()
    for frame, tracks in enumerate(tracks_per_frame):
        gt_boxes = {
            obj.gt_id: obj.visible(frame)
            for obj in clip.objects
            if obj.visible(frame) is not None
        }
        gt_map = associate_tracks_to_gt(tracks, gt_boxes)
        metrics.update(tracks, gt_map)
    return metrics.summary()


def score_zone_events(
    expected: list[ExpectedEvent], actual: list[Event]
) -> dict[str, Any]:
    """Match emitted events to expected by (type, zone, frame within tolerance)."""
    used: set[int] = set()
    matched = 0
    missed = 0
    for ex in expected:
        best_idx: int | None = None
        best_dist: int | None = None
        for i, ev in enumerate(actual):
            if i in used or ev.event_type.value != ex.event_type or ev.zone_id != ex.zone_id:
                continue
            frame = ev.frame_index if ev.frame_index is not None else -(10**9)
            dist = abs(frame - ex.frame)
            if dist <= ex.tolerance and (best_dist is None or dist < best_dist):
                best_idx, best_dist = i, dist
        if best_idx is not None:
            used.add(best_idx)
            matched += 1
        else:
            missed += 1
    false_events = len(actual) - len(used)
    precision = matched / (matched + false_events) if (matched + false_events) else 1.0
    recall = matched / (matched + missed) if (matched + missed) else 1.0
    return {
        "expected": len(expected),
        "actual": len(actual),
        "matched": matched,
        "missed": missed,
        "false_events": false_events,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_event_rate": round(false_events / max(1, len(actual)), 4),
    }

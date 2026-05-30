"""Detector post-processing: decode YOLOv8 output, NMS, map to Detection.

All pure numpy — no GPU, no cv2 — so the decode/NMS/rescale math is unit-tested
on any machine. Handles the ultralytics YOLOv8 export layout (1, 4+nc, anchors),
transposing as needed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vigil.detect.coco import COCO_CLASSES
from vigil.detect.preprocess import LetterboxMeta, scale_boxes
from vigil.types import Detection


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of a single box (4,) against many boxes (N,4) -> (N,)."""
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy single-class NMS. Returns kept indices, highest score first."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1].tolist()
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        ious = box_iou_xyxy(boxes_xyxy[i], boxes_xyxy[order])
        order = [j for k, j in enumerate(order) if ious[k] <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_predictions(
    output: np.ndarray, num_classes: int = 80
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw YOLOv8 output -> (boxes_xyxy, confidence, class_id) in input space."""
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]  # drop batch
    attrs = 4 + num_classes
    # ultralytics exports (4+nc, anchors); transpose to (anchors, 4+nc).
    if pred.shape[0] == attrs and pred.shape[1] != attrs:
        pred = pred.T
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4 : 4 + num_classes]
    class_ids = class_scores.argmax(axis=1)
    confidence = class_scores.max(axis=1)
    return xywh2xyxy(boxes_xywh), confidence, class_ids


def postprocess(
    output: np.ndarray,
    meta: LetterboxMeta,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 80,
    class_names: Sequence[str] = COCO_CLASSES,
    class_filter: Sequence[int] = (),
) -> list[Detection]:
    """Full pipeline: decode -> confidence/class filter -> rescale -> per-class
    NMS -> Detection list, sorted by confidence descending."""
    boxes, conf, cls = decode_predictions(output, num_classes)

    keep_mask = conf >= conf_threshold
    if class_filter:
        keep_mask &= np.isin(cls, np.asarray(list(class_filter)))
    boxes, conf, cls = boxes[keep_mask], conf[keep_mask], cls[keep_mask]
    if boxes.shape[0] == 0:
        return []

    boxes = scale_boxes(boxes, meta)  # IoU is invariant to the affine map

    kept: list[int] = []
    for class_id in np.unique(cls):
        idx = np.where(cls == class_id)[0]
        kept.extend(idx[nms(boxes[idx], conf[idx], iou_threshold)].tolist())

    detections = []
    for i in kept:
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        class_id = int(cls[i])
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=name,
                confidence=float(conf[i]),
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections

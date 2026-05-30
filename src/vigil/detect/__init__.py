"""Vigil detection: interface, mock + TensorRT detectors, pre/post-processing.

Import-safe everywhere — none of these imports pull in CUDA, TensorRT, or even
OpenCV (those are deferred to where they are actually used).
"""
from __future__ import annotations

from vigil.detect.coco import COCO_CLASSES, NUM_COCO_CLASSES
from vigil.detect.engine import Detector, MockDetector, TensorRTDetector
from vigil.detect.postprocess import (
    box_iou_xyxy,
    decode_predictions,
    nms,
    postprocess,
    xywh2xyxy,
)
from vigil.detect.preprocess import (
    LetterboxMeta,
    compute_letterbox_params,
    letterbox,
    preprocess,
    scale_boxes,
)

__all__ = [
    "COCO_CLASSES",
    "NUM_COCO_CLASSES",
    "Detector",
    "LetterboxMeta",
    "MockDetector",
    "TensorRTDetector",
    "box_iou_xyxy",
    "compute_letterbox_params",
    "decode_predictions",
    "letterbox",
    "nms",
    "postprocess",
    "preprocess",
    "scale_boxes",
    "xywh2xyxy",
]

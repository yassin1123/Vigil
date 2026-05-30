"""Detector pre-processing: letterbox resize + normalization.

The geometry (scale ratio + padding) is split out as a pure function so the
coordinate math can be unit-tested without OpenCV or a GPU. Only the actual
pixel resize touches cv2, and that import is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LetterboxMeta:
    """Everything needed to map detections back to the original image.

    Attributes:
        ratio: uniform scale applied to the original image.
        pad: (dw, dh) padding added on the left and top (half of the total).
        orig_shape: (height, width) of the original image.
        input_shape: (height, width) of the letterboxed network input.
    """

    ratio: float
    pad: tuple[float, float]
    orig_shape: tuple[int, int]
    input_shape: tuple[int, int]


def compute_letterbox_params(
    orig_h: int, orig_w: int, new_h: int, new_w: int
) -> tuple[float, tuple[int, int], tuple[float, float]]:
    """Return (ratio, (new_unpad_w, new_unpad_h), (dw, dh)) — pure math.

    Aspect ratio is preserved by scaling with a single ratio; the remaining
    space is split evenly as padding.
    """
    ratio = min(new_h / orig_h, new_w / orig_w)
    new_unpad_w = int(round(orig_w * ratio))
    new_unpad_h = int(round(orig_h * ratio))
    dw = (new_w - new_unpad_w) / 2.0
    dh = (new_h - new_unpad_h) / 2.0
    return ratio, (new_unpad_w, new_unpad_h), (dw, dh)


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize `image` into `new_shape` preserving aspect ratio, padding the rest."""
    import cv2

    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    new_h, new_w = new_shape
    ratio, (nuw, nuh), (dw, dh) = compute_letterbox_params(orig_h, orig_w, new_h, new_w)

    if (orig_w, orig_h) != (nuw, nuh):
        resized = cv2.resize(image, (nuw, nuh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    # Exact split so the padded image is precisely new_shape.
    top = int(np.floor(dh))
    bottom = new_h - nuh - top
    left = int(np.floor(dw))
    right = new_w - nuw - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    meta = LetterboxMeta(
        ratio=ratio,
        pad=(dw, dh),
        orig_shape=(orig_h, orig_w),
        input_shape=(new_h, new_w),
    )
    return padded, meta


def preprocess(
    image: np.ndarray, input_size: int = 640
) -> tuple[np.ndarray, LetterboxMeta]:
    """BGR image -> (NCHW float32 RGB blob in [0,1], LetterboxMeta).

    `image` is an OpenCV-convention BGR HxWx3 uint8 array (e.g. Frame.image).
    """
    padded, meta = letterbox(image, (input_size, input_size))
    rgb = padded[:, :, ::-1]  # BGR -> RGB
    chw = rgb.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    return blob[np.newaxis, ...], meta  # add batch dim -> (1,3,H,W)


def scale_boxes(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from letterboxed-input space back to original image pixels.

    Pure numpy — the inverse of `letterbox`'s scale+pad — and clipped to the
    original image bounds.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    dw, dh = meta.pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= meta.ratio
    h, w = meta.orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes

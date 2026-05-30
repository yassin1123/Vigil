"""Detection: pure-numpy pre/post-processing math + MockDetector contract.

No GPU, no TensorRT, and (except one explicit letterbox test) no OpenCV.
"""
from __future__ import annotations

import numpy as np
import pytest

from vigil.detect import (
    COCO_CLASSES,
    MockDetector,
    box_iou_xyxy,
    compute_letterbox_params,
    decode_predictions,
    nms,
    postprocess,
    scale_boxes,
    xywh2xyxy,
)
from vigil.detect.preprocess import LetterboxMeta
from vigil.types import Detection

# --- letterbox geometry (pure math) --------------------------------------- #


def test_letterbox_params_wide_image():
    # 480 tall x 640 wide into 640x640: scale 1.0, pad top/bottom only.
    ratio, (nuw, nuh), (dw, dh) = compute_letterbox_params(480, 640, 640, 640)
    assert ratio == 1.0
    assert (nuw, nuh) == (640, 480)
    assert (dw, dh) == (0.0, 80.0)


def test_letterbox_params_1080p():
    ratio, (nuw, nuh), (dw, dh) = compute_letterbox_params(1080, 1920, 640, 640)
    assert ratio == pytest.approx(640 / 1920)
    assert (nuw, nuh) == (640, 360)
    assert (dw, dh) == (0.0, pytest.approx((640 - 360) / 2))


def test_scale_boxes_inverts_letterbox():
    # Original 100h x 200w -> 640 input: ratio 3.2, dh 160, dw 0.
    meta = LetterboxMeta(ratio=3.2, pad=(0.0, 160.0), orig_shape=(100, 200), input_shape=(640, 640))
    # Full-image box in letterboxed space maps back to (0,0,200,100).
    boxed = np.array([[0.0, 160.0, 640.0, 480.0]], dtype=np.float32)
    out = scale_boxes(boxed, meta)
    assert out[0] == pytest.approx([0.0, 0.0, 200.0, 100.0])


def test_scale_boxes_clips_to_image():
    meta = LetterboxMeta(ratio=1.0, pad=(0.0, 0.0), orig_shape=(50, 50), input_shape=(640, 640))
    out = scale_boxes(np.array([[-10.0, -10.0, 999.0, 999.0]], dtype=np.float32), meta)
    assert out[0].tolist() == [0.0, 0.0, 50.0, 50.0]


# --- box ops --------------------------------------------------------------- #


def test_xywh2xyxy():
    out = xywh2xyxy(np.array([[10.0, 10.0, 4.0, 6.0]]))
    assert out[0].tolist() == [8.0, 7.0, 12.0, 13.0]


def test_box_iou():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
    iou = box_iou_xyxy(box, others)
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(25 / 175)
    assert iou[2] == pytest.approx(0.0)


def test_nms_suppresses_overlap_keeps_distinct():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]  # box 1 suppressed by box 0; box 2 distinct


# --- decode + full postprocess --------------------------------------------- #


def _fake_yolo_output(cx, cy, w, h, class_id, conf, num_classes=80, anchors=10):
    """One strong detection at anchor 0, layout (1, 4+nc, anchors)."""
    out = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = cx, cy, w, h
    out[0, 4 + class_id, 0] = conf
    return out


def test_decode_predictions_transposes_and_picks_class():
    out = _fake_yolo_output(320, 320, 40, 40, class_id=2, conf=0.95)
    boxes, conf, cls = decode_predictions(out, num_classes=80)
    assert boxes.shape == (10, 4)
    assert cls[0] == 2
    assert conf[0] == pytest.approx(0.95)
    assert boxes[0].tolist() == [300.0, 300.0, 340.0, 340.0]


def test_postprocess_produces_valid_detection_contract():
    out = _fake_yolo_output(320, 320, 40, 40, class_id=0, conf=0.95)
    meta = LetterboxMeta(ratio=1.0, pad=(0.0, 0.0), orig_shape=(640, 640), input_shape=(640, 640))
    dets = postprocess(out, meta, conf_threshold=0.25)
    assert len(dets) == 1
    det = dets[0]
    assert isinstance(det, Detection)
    assert det.class_name == "person"
    assert 0.0 <= det.confidence <= 1.0
    # bbox within image bounds and well-formed
    assert 0 <= det.x1 < det.x2 <= 640
    assert 0 <= det.y1 < det.y2 <= 640


def test_postprocess_confidence_and_class_filter():
    out = _fake_yolo_output(320, 320, 40, 40, class_id=5, conf=0.95)
    meta = LetterboxMeta(ratio=1.0, pad=(0.0, 0.0), orig_shape=(640, 640), input_shape=(640, 640))
    assert postprocess(out, meta, conf_threshold=0.99) == []  # below threshold
    assert postprocess(out, meta, conf_threshold=0.25, class_filter=(1, 2)) == []  # filtered out
    assert len(postprocess(out, meta, conf_threshold=0.25, class_filter=(5,))) == 1


# --- MockDetector ---------------------------------------------------------- #


def test_mock_detector_synthesizes_in_bounds():
    det = MockDetector()
    det.load()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.infer(image)
    assert len(results) == 1
    d = results[0]
    assert d.class_name in COCO_CLASSES
    assert 0.0 <= d.confidence <= 1.0
    assert 0 <= d.x1 < d.x2 <= 640
    assert 0 <= d.y1 < d.y2 <= 480


def test_mock_detector_scripted_sequence():
    a = Detection((0, 0, 1, 1), 0, "person", 0.5)
    b = Detection((1, 1, 2, 2), 2, "car", 0.6)
    det = MockDetector(script=[[a], [b], []])
    det.load()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert det.infer(image) == [a]
    assert det.infer(image) == [b]
    assert det.infer(image) == []
    assert det.infer(image) == []  # past the script -> default ([])


def test_mock_detector_as_context_manager():
    with MockDetector(default=[]) as det:
        det.load()
        assert det.infer(np.zeros((4, 4, 3), dtype=np.uint8)) == []


# --- letterbox actual pixels (needs cv2) ----------------------------------- #


def test_letterbox_pads_to_exact_shape():
    cv2 = pytest.importorskip("cv2")
    from vigil.detect.preprocess import letterbox, preprocess

    image = np.full((100, 200, 3), 50, dtype=np.uint8)
    padded, meta = letterbox(image, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert meta.orig_shape == (100, 200)
    # padded border rows are the fill color (114)
    assert (padded[0] == 114).all()

    blob, _ = preprocess(image, 640)
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() and blob.max() <= 1.0
    _ = cv2  # silence unused

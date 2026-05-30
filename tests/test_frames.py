"""Frame sources: deterministic mock, file decoding, and config-driven select.

These tests never touch the GPU, GStreamer, or a real camera. The CSI source is
constructed (to prove the factory wiring) but never opened.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vigil.config import VigilConfig
from vigil.frames import (
    CSIFrameSource,
    FileFrameSource,
    Frame,
    MockFrameSource,
    build_csi_pipeline,
    build_frame_source,
)

CLIP_DIR = Path(__file__).resolve().parent / "data" / "clip"


def test_mock_is_deterministic():
    a = list(MockFrameSource(width=64, height=48, num_frames=10, fps=20.0))
    b = list(MockFrameSource(width=64, height=48, num_frames=10, fps=20.0))
    assert len(a) == len(b) == 10
    for fa, fb in zip(a, b):
        assert fa.index == fb.index
        assert fa.timestamp == fb.timestamp
        assert np.array_equal(fa.image, fb.image)


def test_mock_shape_dtype_and_timestamps():
    frames = list(MockFrameSource(width=64, height=48, num_frames=5, fps=10.0))
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[0].image.dtype == np.uint8
    assert frames[0].shape == (48, 64)
    assert frames[2].timestamp == pytest.approx(0.2)


def test_mock_read_returns_none_at_end():
    src = MockFrameSource(num_frames=2)
    src.open()
    assert isinstance(src.read(), Frame)
    assert isinstance(src.read(), Frame)
    assert src.read() is None
    src.close()


def test_mock_context_manager_iterates():
    with MockFrameSource(num_frames=3) as src:
        assert sum(1 for _ in src) == 3


def test_mock_rejects_bad_params():
    with pytest.raises(ValueError):
        MockFrameSource(width=0)
    with pytest.raises(ValueError):
        MockFrameSource(fps=0)


def test_file_source_reads_committed_clip():
    pytest.importorskip("cv2")
    with FileFrameSource(CLIP_DIR, fps=10.0) as src:
        frames = list(src)
    assert len(frames) >= 3
    assert [f.index for f in frames] == list(range(len(frames)))
    assert frames[0].image.ndim == 3
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)  # monotonic non-decreasing


def test_file_source_missing_path_raises():
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError):
        FileFrameSource(CLIP_DIR.parent / "does-not-exist").open()


def test_build_frame_source_selects_by_kind():
    mock_cfg = VigilConfig.from_dict({"source": {"kind": "mock", "mock": {"num_frames": 4}}})
    assert isinstance(build_frame_source(mock_cfg), MockFrameSource)

    file_cfg = VigilConfig.from_dict(
        {"source": {"kind": "file", "file": {"path": str(CLIP_DIR)}}}
    )
    assert isinstance(build_frame_source(file_cfg), FileFrameSource)

    # CSI source is built (wiring check) but never opened — no hardware needed.
    csi_cfg = VigilConfig.from_dict({"source": {"kind": "csi"}})
    src = build_frame_source(csi_cfg)
    assert isinstance(src, CSIFrameSource)


def test_csi_pipeline_string_is_well_formed():
    pipeline = build_csi_pipeline(0, 1920, 1080, 30, 0)
    assert pipeline.startswith("nvarguscamerasrc sensor-id=0")
    assert "width=(int)1920" in pipeline
    assert "framerate=(fraction)30/1" in pipeline
    assert pipeline.rstrip().endswith("appsink drop=true max-buffers=2 sync=false")

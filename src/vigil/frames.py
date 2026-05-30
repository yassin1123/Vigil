"""Frame sources — the hardware boundary for Vigil's capture stage.

Everything downstream consumes `Frame` objects from a `FrameSource` and neither
knows nor cares whether they came from a live CSI camera, a video file, or a
synthetic generator. This is where Vigil's both-worlds rule starts:

  * CSIFrameSource  — real nvarguscamerasrc GStreamer pipeline (Jetson only).
  * FileFrameSource — a video file or a directory of images (CI + benchmark).
  * MockFrameSource — deterministic synthetic frames (unit tests, no I/O).

Only file decoding and the live CSI pipeline touch OpenCV, and that import is
deferred so the package — and MockFrameSource — import cleanly on a machine
without cv2. A unit test must never need the GPU, GStreamer, or even OpenCV.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

if TYPE_CHECKING:
    from vigil.config import VigilConfig

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(slots=True)
class Frame:
    """A captured frame plus its capture time and sequence index.

    Attributes:
        image: HxWx3 BGR uint8 ndarray (OpenCV convention).
        timestamp: capture time in monotonic seconds. Live sources use the real
            monotonic clock (relative to source open); File/Mock use a synthetic
            clock (index / fps) so replays and tests are deterministic.
        index: 0-based frame index within this source.
    """

    image: np.ndarray
    timestamp: float
    index: int

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


class FrameSource(abc.ABC):
    """Abstract, source-agnostic provider of timestamped frames.

    Lifecycle: `open()` acquires resources, `read()` returns the next frame (or
    None at end of stream), `close()` releases. Iterating the source yields
    frames until exhausted; it is also a context manager.
    """

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next Frame, or None when the source is exhausted."""

    @abc.abstractmethod
    def close(self) -> None: ...

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


class MockFrameSource(FrameSource):
    """Deterministic synthetic frames: a white square marching across a ramp.

    Each frame is a pure function of its index — two sources with identical
    parameters yield byte-identical frames, with no RNG and no I/O. This is the
    workhorse for unit tests.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        num_frames: int = 100,
        fps: float = 30.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self._index = 0

    def open(self) -> None:
        self._index = 0

    def _render(self, index: int) -> np.ndarray:
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        ramp = np.linspace(0, 255, self.width, dtype=np.uint8)
        img[:, :, 0] = ramp[np.newaxis, :]  # B: horizontal ramp
        img[:, :, 1] = (index * 7) % 256  # G: steps with frame index
        img[:, :, 2] = ramp[::-1][np.newaxis, :]  # R: reversed ramp
        size = min(40, self.width, self.height)
        span = max(self.width - size, 1)
        x = (index * 11) % span
        y = max((self.height - size) // 2, 0)
        img[y : y + size, x : x + size, :] = 255  # marching white square
        return img

    def read(self) -> Optional[Frame]:
        if self._index >= self.num_frames:
            return None
        frame = Frame(
            image=self._render(self._index),
            timestamp=self._index / self.fps,
            index=self._index,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._index = self.num_frames


class FileFrameSource(FrameSource):
    """Frames from a video file or a directory of images.

    Used by CI and the benchmark. Decoding uses OpenCV, imported lazily so the
    module loads without cv2. For a directory, images are read in sorted
    filename order; for a video file, the file's own FPS is used if available.
    """

    def __init__(self, path: str | Path, fps: float = 30.0, loop: bool = False) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.loop = bool(loop)
        self._index = 0
        self._cap = None  # cv2.VideoCapture, for video files
        self._images: list[Path] = []  # for image directories
        self._is_dir = False
        self._fps_effective = self.fps

    def open(self) -> None:
        import cv2

        if not self.path.exists():
            raise FileNotFoundError(f"frame source path not found: {self.path}")
        self._index = 0
        if self.path.is_dir():
            self._is_dir = True
            self._images = sorted(
                p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self._images:
                raise ValueError(
                    f"no images {IMAGE_SUFFIXES} found in directory {self.path}"
                )
            self._fps_effective = self.fps
        else:
            self._is_dir = False
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"could not open video file: {self.path}")
            self._cap = cap
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            self._fps_effective = file_fps if file_fps and file_fps > 0 else self.fps

    def read(self) -> Optional[Frame]:
        import cv2

        if self._is_dir:
            if self._index >= len(self._images):
                if self.loop and self._images:
                    self._index = 0
                else:
                    return None
            image_path = self._images[self._index]
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"failed to decode image: {image_path}")
        else:
            if self._cap is None:
                raise RuntimeError("FileFrameSource.read() before open()")
            ok, img = self._cap.read()
            if not ok or img is None:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._index = 0
                    ok, img = self._cap.read()
                    if not ok or img is None:
                        return None
                else:
                    return None
        frame = Frame(
            image=img,
            timestamp=self._index / self._fps_effective,
            index=self._index,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._images = []
        self._index = 0


class CSIFrameSource(FrameSource):
    """Live CSI camera via nvarguscamerasrc (Jetson only).

    Opening requires an OpenCV built with GStreamer, which the dev-machine pip
    wheel does NOT have — so this source is never exercised in tests. Downstream
    code treats it exactly like any other FrameSource.
    """

    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        framerate: int = 30,
        flip_method: int = 0,
        max_frames: Optional[int] = None,
    ) -> None:
        self.sensor_id = int(sensor_id)
        self.width = int(width)
        self.height = int(height)
        self.framerate = int(framerate)
        self.flip_method = int(flip_method)
        self.max_frames = max_frames
        self._cap = None
        self._index = 0
        self._t0 = 0.0

    @property
    def pipeline(self) -> str:
        return build_csi_pipeline(
            self.sensor_id, self.width, self.height, self.framerate, self.flip_method
        )

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                "could not open CSI camera. Needs a Jetson with a seated CSI "
                "module and an OpenCV built with GStreamer (the system OpenCV "
                "from JetPack — not the pip wheel)."
            )
        self._cap = cap
        self._index = 0
        self._t0 = time.monotonic()

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise RuntimeError("CSIFrameSource.read() before open()")
        if self.max_frames is not None and self._index >= self.max_frames:
            return None
        ok, img = self._cap.read()
        if not ok or img is None:
            return None
        frame = Frame(
            image=img,
            timestamp=time.monotonic() - self._t0,
            index=self._index,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build_csi_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    framerate: int,
    flip_method: int,
    display_width: Optional[int] = None,
    display_height: Optional[int] = None,
) -> str:
    """Construct the nvarguscamerasrc -> appsink GStreamer pipeline string."""
    dw = display_width or width
    dh = display_height or height
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){dw}, height=(int){dh}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=2 sync=false"
    )


def build_frame_source(config: "VigilConfig") -> FrameSource:
    """Create the FrameSource selected by `config.source.kind`.

    This is the only place that maps config -> concrete source; everything
    downstream takes a `FrameSource` and stays source-agnostic.
    """
    kind = config.source.kind
    if kind == "csi":
        cam = config.camera
        return CSIFrameSource(
            sensor_id=cam.sensor_id,
            width=cam.width,
            height=cam.height,
            framerate=cam.framerate,
            flip_method=cam.flip_method,
        )
    if kind == "file":
        f = config.source.file
        return FileFrameSource(path=f.path, fps=f.fps, loop=f.loop)
    if kind == "mock":
        m = config.source.mock
        return MockFrameSource(
            width=m.width, height=m.height, num_frames=m.num_frames, fps=m.fps
        )
    raise ValueError(f"unknown source.kind: {kind!r} (expected csi|file|mock)")

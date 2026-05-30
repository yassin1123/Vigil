"""Detector interface and implementations.

  * Detector         — the contract: load(engine_path) then infer(image).
  * MockDetector     — scripted/synthetic detections; zero GPU, for CI + pipeline.
  * TensorRTDetector — real GPU inference from a serialized .engine (Jetson only).

CUDA / TensorRT / pycuda are imported lazily *inside* TensorRTDetector methods,
so importing this module (and running every test) needs none of them.
"""
from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from vigil.detect.coco import COCO_CLASSES, NUM_COCO_CLASSES
from vigil.detect.postprocess import postprocess
from vigil.detect.preprocess import preprocess
from vigil.types import Detection


class Detector(abc.ABC):
    """Loads a model and turns a frame into a list of Detections."""

    @abc.abstractmethod
    def load(self, engine_path: str | Path | None = None) -> None:
        """Prepare the detector for inference (load weights / engine)."""

    @abc.abstractmethod
    def infer(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a BGR HxWx3 image and return detections."""

    def close(self) -> None:  # noqa: B027 - optional override, default no-op
        """Release any resources. Default: nothing."""

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


class MockDetector(Detector):
    """Deterministic detector for tests and off-device pipeline runs.

    Modes (first match wins):
      * `script`: a sequence of per-call detection lists, returned in order;
        once exhausted, `default` (or []) is returned.
      * `default`: a fixed detection list returned on every call.
      * neither: synthesises one centered 'person' box sized to the image.
    """

    def __init__(
        self,
        script: Optional[Sequence[Sequence[Detection]]] = None,
        default: Optional[Sequence[Detection]] = None,
        class_names: Sequence[str] = COCO_CLASSES,
        input_size: int = 640,
    ) -> None:
        self.script = [list(s) for s in script] if script is not None else None
        self.default = list(default) if default is not None else None
        self.class_names = class_names
        self.input_size = input_size
        self._calls = 0
        self._loaded = False

    def load(self, engine_path: str | Path | None = None) -> None:
        self._loaded = True
        self._calls = 0

    def infer(self, image: np.ndarray) -> list[Detection]:
        index = self._calls
        self._calls += 1
        if self.script is not None:
            if index < len(self.script):
                return list(self.script[index])
            return list(self.default) if self.default is not None else []
        if self.default is not None:
            return list(self.default)
        return self._synthesize(image)

    def _synthesize(self, image: np.ndarray) -> list[Detection]:
        h, w = int(image.shape[0]), int(image.shape[1])
        x1, y1 = w * 0.25, h * 0.25
        x2, y2 = w * 0.75, h * 0.75
        return [
            Detection(
                bbox=(x1, y1, x2, y2),
                class_id=0,
                class_name=self.class_names[0],
                confidence=0.9,
            )
        ]


class TensorRTDetector(Detector):
    """Real GPU inference from a serialized TensorRT .engine (Jetson only).

    Targets the TensorRT 8.5+ named-tensor execution API (execute_async_v3) and
    uses pycuda for device memory. The .engine is hardware-specific and built on
    the target by scripts/build_engine.py — it is never committed.
    """

    def __init__(
        self,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_filter: Sequence[int] = (),
        class_names: Sequence[str] = COCO_CLASSES,
        num_classes: int = NUM_COCO_CLASSES,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = tuple(class_filter)
        self.class_names = class_names
        self.num_classes = num_classes
        # Populated in load(); kept as Any to avoid importing trt/cuda at module load.
        self._trt = None
        self._cuda = None
        self._engine = None
        self._context = None
        self._stream = None
        self._inputs: list[dict] = []
        self._outputs: list[dict] = []

    def load(self, engine_path: str | Path | None = None) -> None:
        if engine_path is None:
            raise ValueError("TensorRTDetector.load requires an engine_path")
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"engine not found: {engine_path}. Build it on the Jetson with "
                f"scripts/build_engine.py (engines are not committed)."
            )

        import pycuda.autoinit  # noqa: F401  (creates the CUDA context on import)
        import pycuda.driver as cuda
        import tensorrt as trt

        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self._engine = engine
        self._context = engine.create_execution_context()
        self._stream = cuda.Stream()

        self._inputs.clear()
        self._outputs.clear()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(int(s) for s in engine.get_tensor_shape(name))
            if any(s < 0 for s in shape):  # dynamic — pin to our static input
                if is_input:
                    shape = (1, 3, self.input_size, self.input_size)
                    self._context.set_input_shape(name, shape)
                else:
                    shape = tuple(int(s) for s in self._context.get_tensor_shape(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self._context.set_tensor_address(name, int(device))
            binding = {"name": name, "host": host, "device": device, "shape": shape}
            (self._inputs if is_input else self._outputs).append(binding)

        if not self._inputs or not self._outputs:
            raise RuntimeError("engine has no input/output tensors")

    def infer(self, image: np.ndarray) -> list[Detection]:
        if self._context is None:
            raise RuntimeError("TensorRTDetector.infer() before load()")
        blob, meta = preprocess(image, self.input_size)

        inp = self._inputs[0]
        np.copyto(inp["host"], blob.ravel())
        self._cuda.memcpy_htod_async(inp["device"], inp["host"], self._stream)
        self._context.execute_async_v3(self._stream.handle)
        for out in self._outputs:
            self._cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)
        self._stream.synchronize()

        out0 = self._outputs[0]
        output = np.asarray(out0["host"]).reshape(out0["shape"])
        return postprocess(
            output,
            meta,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            num_classes=self.num_classes,
            class_names=self.class_names,
            class_filter=self.class_filter,
        )

    def close(self) -> None:
        # pycuda frees device allocations on GC; drop references explicitly.
        self._inputs.clear()
        self._outputs.clear()
        self._context = None
        self._engine = None
        self._stream = None

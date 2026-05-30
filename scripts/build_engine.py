#!/usr/bin/env python3
"""Build Vigil's detection engine: YOLOv8n -> ONNX -> INT8 TensorRT engine.

  RUN THIS ON THE JETSON. A TensorRT .engine is hardware- and version-specific:
  it is tuned for the exact GPU, TensorRT, and CUDA on the build machine and is
  NOT portable. Never commit a prebuilt engine — build it on the target. (The
  models/ directory is git-ignored for exactly this reason.)

Pipeline:
  1. Export YOLOv8n weights to ONNX (CPU; works off-device with ultralytics).
  2. Build a TensorRT engine from the ONNX:
       - INT8 with calibration if the platform supports fast INT8 and a
         calibration image set is present;
       - otherwise FP16 (with a clear note) if supported;
       - otherwise FP32.
  3. Write the .engine plus a sidecar JSON recording the build settings.

Off-device (no TensorRT) it still exports ONNX with --onnx-only; a full build
without TensorRT prints exactly what is missing and exits non-zero.

Examples:
  python3 scripts/build_engine.py --weights yolov8n.pt          # full INT8 build
  python3 scripts/build_engine.py --precision fp16              # skip INT8
  python3 scripts/build_engine.py --onnx-only                   # export ONNX only
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make `vigil` importable without an install (so calibration reuses the exact
# same preprocessing the detector uses).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_ONNX = "models/yolov8n.onnx"
DEFAULT_ENGINE = "models/yolov8n.engine"
DEFAULT_CALIB_DIR = "calibration/images"
DEFAULT_INPUT_SIZE = 640
DEFAULT_WORKSPACE_MB = 2048
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------- #
# ONNX export (CPU; no GPU needed)
# --------------------------------------------------------------------------- #


def export_onnx(weights: str, onnx_path: Path, input_size: int) -> Path:
    """Export YOLOv8 weights to ONNX via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required to export ONNX: pip install ultralytics"
        ) from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    exported = model.export(
        format="onnx", imgsz=input_size, opset=12, simplify=True, dynamic=False
    )
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        shutil.move(str(exported_path), str(onnx_path))
    print(f"  ONNX exported -> {onnx_path}")
    return onnx_path


# --------------------------------------------------------------------------- #
# INT8 calibration
# --------------------------------------------------------------------------- #


def _collect_calibration_images(calib_dir: Path) -> list[Path]:
    if not calib_dir.is_dir():
        return []
    return sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _make_calibrator(trt, cuda, images: list[Path], input_size: int, cache: Path):
    """Build an IInt8EntropyCalibrator2 that feeds preprocessed calibration
    images one at a time. Defined as a closure so trt is imported lazily."""
    import cv2

    from vigil.detect.preprocess import preprocess

    class _Int8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.input_size = input_size
            self.cache_path = cache
            self.images = images
            self.index = 0
            self.device_input = cuda.mem_alloc(3 * input_size * input_size * 4)

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names):  # noqa: ARG002 - TRT-supplied tensor names
            if self.index >= len(self.images):
                return None
            img = cv2.imread(str(self.images[self.index]), cv2.IMREAD_COLOR)
            self.index += 1
            if img is None:
                return None
            blob, _ = preprocess(img, self.input_size)
            blob = blob.ravel().copy()
            cuda.memcpy_htod(self.device_input, blob)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache_bytes) -> None:
            self.cache_path.write_bytes(cache_bytes)

    return _Int8Calibrator()


# --------------------------------------------------------------------------- #
# Engine build (Jetson / TensorRT)
# --------------------------------------------------------------------------- #


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_size: int,
    workspace_mb: int,
    precision_pref: str,
    calib_dir: Path,
) -> dict:
    """Build a TensorRT engine and return the build-settings record."""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    flag = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("failed to parse ONNX:\n  " + "\n  ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    images = _collect_calibration_images(calib_dir)
    precision = "fp32"
    note = ""
    if precision_pref == "int8" and builder.platform_has_fast_int8 and images:
        config.set_flag(trt.BuilderFlag.INT8)
        cache = engine_path.with_suffix(".calib.cache")
        config.int8_calibrator = _make_calibrator(trt, cuda, images, input_size, cache)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)  # mixed INT8/FP16 fallback layers
        precision = "int8"
    elif precision_pref in ("int8", "fp16") and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        precision = "fp16"
        if precision_pref == "int8":
            note = (
                "INT8 requested but "
                + ("no calibration images found" if not images else "INT8 unsupported")
                + f" -> fell back to FP16. Add images to {calib_dir} for INT8."
            )
    else:
        note = "Neither INT8 nor FP16 available -> FP32."

    print(f"  building {precision.upper()} engine (workspace {workspace_mb} MB)...")
    if note:
        print(f"  NOTE: {note}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"  engine written -> {engine_path}")

    return {
        "precision": precision,
        "note": note,
        "input_size": input_size,
        "workspace_mb": workspace_mb,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "calibration_images": len(images),
        "calibration_dir": str(calib_dir),
        "tensorrt_version": trt.__version__,
        "platform_has_fast_int8": bool(builder.platform_has_fast_int8),
        "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
        "device_model": _device_model(),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def _device_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_text(errors="replace").strip("\x00 ")
    except OSError:
        return platform.platform()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8 .pt weights")
    p.add_argument("--onnx", type=Path, default=Path(DEFAULT_ONNX))
    p.add_argument("--engine", type=Path, default=Path(DEFAULT_ENGINE))
    p.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument("--workspace-mb", type=int, default=DEFAULT_WORKSPACE_MB)
    p.add_argument("--calib-dir", type=Path, default=Path(DEFAULT_CALIB_DIR))
    p.add_argument(
        "--precision",
        choices=("int8", "fp16", "fp32"),
        default="int8",
        help="Preferred precision; falls back if unsupported or no calib data.",
    )
    p.add_argument("--onnx-only", action="store_true", help="Export ONNX and stop.")
    p.add_argument(
        "--skip-export", action="store_true", help="Use existing --onnx, skip export."
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    print("=" * 64)
    print("  Vigil engine build")
    print("=" * 64)

    if not args.skip_export:
        export_onnx(args.weights, args.onnx, args.input_size)
    elif not args.onnx.exists():
        print(f"  --skip-export set but {args.onnx} is missing")
        return 2

    if args.onnx_only:
        print("  --onnx-only: done.")
        return 0

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print(
            "  TensorRT is not available here. Engine building must run ON THE "
            "JETSON.\n  (ONNX export works anywhere with --onnx-only.)"
        )
        return 2

    record = build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        input_size=args.input_size,
        workspace_mb=args.workspace_mb,
        precision_pref=args.precision,
        calib_dir=args.calib_dir,
    )
    sidecar = args.engine.with_suffix(args.engine.suffix + ".json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"  build record -> {sidecar}")
    print("=" * 64)
    print(f"  DONE: {record['precision'].upper()} engine ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

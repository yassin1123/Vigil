"""`python -m vigil` — capture -> detect -> track run loop.

Loads the config, opens the configured FrameSource, runs detection and ByteTrack
tracking, and shows the tracks: a live OpenCV window with track ids (`--show`)
or a per-frame console summary (headless default). This is the seed of the full
pipeline runner (… -> zones -> log -> ui) wired up on Day 6.

Detector selection (`--detector`):
  auto      use the TensorRT engine if it exists, else the MockDetector
  mock      always the MockDetector (works anywhere, zero GPU)
  tensorrt  always the TensorRT engine (Jetson)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from vigil.config import VigilConfig, load_config
from vigil.detect import Detector, MockDetector, TensorRTDetector
from vigil.frames import build_frame_source
from vigil.track import build_tracker


def build_detector(config: VigilConfig, mode: str = "auto") -> tuple[Detector, str]:
    """Return (detector, kind). 'auto' uses TensorRT iff the engine file exists."""
    engine_exists = Path(config.model.engine_path).exists()
    use_trt = mode == "tensorrt" or (mode == "auto" and engine_exists)
    if use_trt:
        det = TensorRTDetector(
            input_size=config.model.input_size,
            conf_threshold=config.model.conf_threshold,
            iou_threshold=config.model.iou_threshold,
            class_filter=tuple(config.model.class_filter),
        )
        det.load(config.model.engine_path)
        return det, "tensorrt"
    det = MockDetector()
    det.load()
    return det, "mock"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vigil", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--detector", choices=("auto", "mock", "tensorrt"), default="auto"
    )
    parser.add_argument("--show", action="store_true", help="Live OpenCV window")
    parser.add_argument(
        "--max-frames", type=int, default=100, help="0 = run until source ends"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    source = build_frame_source(config)

    try:
        detector, det_kind = build_detector(config, args.detector)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"detector error: {exc}")
        return 1
    tracker = build_tracker(config)
    print(
        f"source={type(source).__name__}(kind={config.source.kind}) "
        f"detector={det_kind} tracker=ByteTrack"
    )

    count = 0
    start = time.monotonic()
    try:
        with source:
            for frame in source:
                count += 1
                detections = detector.infer(frame.image)
                tracks = tracker.update(detections, frame)

                if args.show:
                    import cv2

                    from vigil.overlay import draw_tracks

                    cv2.imshow("vigil", draw_tracks(frame.image, tracks))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif count <= 5 or count % 25 == 0:
                    ids = ",".join(str(t.track_id) for t in tracks)
                    print(
                        f"  frame {frame.index:>6} t={frame.timestamp:7.3f}s "
                        f"dets={len(detections)} tracks={len(tracks)} ids=[{ids}]"
                    )
                if args.max_frames and count >= args.max_frames:
                    break
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    finally:
        detector.close()

    elapsed = time.monotonic() - start
    fps = count / elapsed if elapsed > 0 else 0.0
    print(f"processed {count} frames in {elapsed:.3f}s  ({fps:.1f} FPS wall-clock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

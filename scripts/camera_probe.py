#!/usr/bin/env python3
"""Vigil CSI camera probe — opens the camera via GStreamer, measures capture FPS.

Part of Day-1 hardware bring-up. Runs *anywhere*:

  * On a Jetson with a CSI module it opens an nvarguscamerasrc GStreamer
    pipeline, warms up, captures N frames, measures the real capture FPS and
    resolution, saves a few stills, and writes docs/baseline/camera_report.json.
  * On any other machine (OpenCV without GStreamer, or no camera) it prints a
    clear "not on target hardware" message, writes a report saying so, exits 0.

It never crashes when the camera is absent — that path is expected off-device.

Usage:
    python3 scripts/camera_probe.py [--sensor-id 0] [--width 1920] [--height 1080]
        [--framerate 30] [--flip-method 0] [--num-frames 120] [--warmup 30]
        [--output-dir docs/baseline]

CSI camera nodes (IMX219-class) do NOT appear as a usable /dev/video device for
direct OpenCV capture — they come up through the Argus daemon, which is why we
go through the nvarguscamerasrc GStreamer pipeline rather than a device index.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_OUTPUT_DIR = Path("docs/baseline")
SCHEMA_VERSION = 1
STILL_PREFIX = "vigil_baseline"

# Sane defaults for an IMX219-class CSI module. Other native modes:
#   3280x2464@21, 1920x1080@30, 1640x1232@30, 1280x720@60, 1280x720@120
DEFAULTS = {
    "sensor_id": 0,
    "width": 1920,
    "height": 1080,
    "framerate": 30,
    "flip_method": 0,  # 0=none, 2=180deg, 1/3=90deg, 4/5/6/7=flips — see nvvidconv
    "num_frames": 120,
    "warmup": 30,
}


def is_jetson() -> bool:
    """True when running on Tegra/Jetson hardware."""
    try:
        model = Path("/proc/device-tree/model").read_text(errors="replace")
    except OSError:
        model = ""
    return "jetson" in model.lower() or Path("/etc/nv_tegra_release").exists()


def build_pipeline(
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


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "camera_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def base_report(args: argparse.Namespace, pipeline: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "on_target_hardware": is_jetson(),
        "captured": False,
        "reason": None,
        "pipeline": pipeline,
        "requested": {
            "sensor_id": args.sensor_id,
            "width": args.width,
            "height": args.height,
            "framerate": args.framerate,
            "flip_method": args.flip_method,
            "num_frames": args.num_frames,
            "warmup": args.warmup,
        },
        "measured": None,
        "stills": [],
    }


def finish_not_captured(
    output_dir: Path, report: dict[str, Any], reason: str, message: str
) -> int:
    """Write a 'not captured' report, print a friendly message, exit cleanly."""
    report["reason"] = reason
    path = write_report(output_dir, report)
    print(message)
    print(f"  report written: {path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    pipeline = build_pipeline(
        args.sensor_id, args.width, args.height, args.framerate, args.flip_method
    )
    report = base_report(args, pipeline)

    print("=" * 64)
    print("  VIGIL camera probe")
    print("=" * 64)
    print(f"  pipeline: {pipeline}")
    print()

    # --- OpenCV availability -------------------------------------------------
    try:
        import cv2  # noqa: PLC0415  (deferred import so the script loads anywhere)
    except ImportError:
        return finish_not_captured(
            output_dir,
            report,
            reason="opencv-not-installed",
            message=(
                "  opencv-python is not installed in this environment.\n"
                "  NOT ON TARGET HARDWARE path: nothing to capture. On the "
                "Jetson, OpenCV ships with JetPack (with GStreamer support)."
            ),
        )

    # --- GStreamer support in this OpenCV build ------------------------------
    build_info = cv2.getBuildInformation()
    gst_ok = False
    for line in build_info.splitlines():
        if "GStreamer" in line and "YES" in line:
            gst_ok = True
            break
    if not gst_ok:
        return finish_not_captured(
            output_dir,
            report,
            reason="opencv-without-gstreamer",
            message=(
                "  This OpenCV build has NO GStreamer support (the pip wheel "
                "doesn't).\n  NOT ON TARGET HARDWARE path: the CSI pipeline "
                "cannot be opened here.\n  On the Jetson use the system OpenCV "
                "from JetPack, which is built with GStreamer."
            ),
        )

    # --- Open the camera -----------------------------------------------------
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap.release()
        on_target = report["on_target_hardware"]
        where = (
            "on the Jetson but the CSI pipeline did not open"
            if on_target
            else "NOT ON TARGET HARDWARE"
        )
        return finish_not_captured(
            output_dir,
            report,
            reason="capture-not-opened",
            message=(
                f"  Could not open the camera ({where}).\n"
                "  Checks: CSI ribbon seated (contacts toward heatsink), correct "
                "--sensor-id,\n  and `ls /dev/video*` shows a node. Verify argus "
                "with:\n    gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! "
                "fakesink"
            ),
        )

    # --- Warm up (let auto-exposure / auto-gain settle) ----------------------
    print(f"  warming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        ok, _frame = cap.read()
        if not ok:
            cap.release()
            return finish_not_captured(
                output_dir,
                report,
                reason="no-frames-during-warmup",
                message=(
                    "  Camera opened but delivered no frames during warm-up.\n"
                    "  The sensor may be misconfigured for this mode - try a "
                    "documented\n  resolution/framerate (see BRINGUP.md)."
                ),
            )

    # --- Timed capture -------------------------------------------------------
    print(f"  capturing {args.num_frames} frames...")
    output_dir.mkdir(parents=True, exist_ok=True)
    still_indices = {0, args.num_frames // 2, args.num_frames - 1}
    stills: list[str] = []
    captured = 0
    dropped = 0
    last_shape: Optional[tuple[int, int]] = None

    start = time.perf_counter()
    for i in range(args.num_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            dropped += 1
            continue
        captured += 1
        last_shape = (frame.shape[1], frame.shape[0])  # (width, height)
        if i in still_indices:
            still_path = output_dir / f"{STILL_PREFIX}_{i:03d}.jpg"
            if cv2.imwrite(str(still_path), frame):
                stills.append(str(still_path))
    elapsed = time.perf_counter() - start
    cap.release()

    if captured == 0:
        return finish_not_captured(
            output_dir,
            report,
            reason="no-frames-captured",
            message="  Camera opened but no frames were captured during timing.",
        )

    fps = captured / elapsed if elapsed > 0 else 0.0
    resolution = f"{last_shape[0]}x{last_shape[1]}" if last_shape else "unknown"

    report["captured"] = True
    report["measured"] = {
        "resolution": resolution,
        "frames_requested": args.num_frames,
        "frames_captured": captured,
        "frames_dropped": dropped,
        "elapsed_s": round(elapsed, 4),
        "fps": round(fps, 2),
    }
    report["stills"] = stills
    path = write_report(output_dir, report)

    print()
    print(f"  resolution      : {resolution}")
    print(f"  frames captured : {captured}/{args.num_frames}  (dropped {dropped})")
    print(f"  elapsed         : {elapsed:.3f} s")
    print(f"  MEASURED FPS    : {fps:.2f}")
    print(f"  stills saved    : {len(stills)} -> {output_dir}")
    print(f"  report written  : {path}")
    print("=" * 64)
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a CSI camera via GStreamer and measure capture FPS."
    )
    parser.add_argument("--sensor-id", type=int, default=DEFAULTS["sensor_id"])
    parser.add_argument("--width", type=int, default=DEFAULTS["width"])
    parser.add_argument("--height", type=int, default=DEFAULTS["height"])
    parser.add_argument("--framerate", type=int, default=DEFAULTS["framerate"])
    parser.add_argument(
        "--flip-method",
        type=int,
        default=DEFAULTS["flip_method"],
        help="nvvidconv flip-method (0=none, 2=180deg). Default 0.",
    )
    parser.add_argument("--num-frames", type=int, default=DEFAULTS["num_frames"])
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULTS["warmup"],
        help="Frames to discard before timing (auto-exposure settling).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for stills + camera_report.json (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

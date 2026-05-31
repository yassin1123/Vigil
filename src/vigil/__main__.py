"""`vigil` / `python -m vigil` — the pipeline runner CLI.

Subcommands:
  run         capture -> detect -> track loop; live window (--show) or console.
  track-demo  run detection + tracking on a committed clip and print the track
              timeline. Deterministic, no GPU (uses scripted mock detections in
              CI; `--detector tensorrt` detects from real pixels on the Jetson).

`python -m vigil --max-frames 100` still works (defaults to the `run` command).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from vigil.config import VigilConfig, load_config
from vigil.detect import Detector, MockDetector, TensorRTDetector
from vigil.frames import FileFrameSource, MockFrameSource, build_frame_source
from vigil.log.verify import verify_file
from vigil.track import build_tracker
from vigil.types import Detection
from vigil.zones.config import load_zones, validate_zone_set
from vigil.zones.model import ZoneError, ZoneSet

_SUBCOMMANDS = {"run", "track-demo", "zones", "log", "bench"}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def build_detector(config: VigilConfig, mode: str = "auto") -> tuple[Detector, str]:
    """Return (detector, kind). 'auto' uses TensorRT iff the engine file exists."""
    engine_exists = Path(config.model.engine_path).exists()
    if mode == "tensorrt" or (mode == "auto" and engine_exists):
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


def _box(cx: float, cy: float, size: float, class_id: int, name: str, conf: float):
    half = size / 2.0
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_id=class_id,
        class_name=name,
        confidence=conf,
    )


def demo_scenario(num_frames: int, width: int, height: int) -> list[list[Detection]]:
    """A deterministic two-object scenario sized to `num_frames`.

    Object A (person) crosses left->right for the whole clip. Object B (car)
    appears in the middle quarter-to-three-quarters and moves right->left, then
    leaves — so the printed timeline shows a long-lived track and an enter/leave.
    """
    size = max(16, int(min(width, height) * 0.18))
    ay, by = height * 0.35, height * 0.70
    b_start, b_end = num_frames // 4, (num_frames * 3) // 4
    scenario: list[list[Detection]] = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        dets = [_box(width * 0.15 + width * 0.70 * t, ay, size, 0, "person", 0.90)]
        if b_start <= i <= b_end:
            tb = (i - b_start) / max(b_end - b_start, 1)
            dets.append(_box(width * 0.85 - width * 0.70 * tb, by, size, 2, "car", 0.85))
        scenario.append(dets)
    return scenario


def run_track_demo(frames, detector: Detector, tracker) -> dict:
    """Run detect+track over `frames`; return a timeline dict."""
    per_frame = []
    spans: dict[int, dict] = {}
    for frame in frames:
        detections = detector.infer(frame.image)
        tracks = tracker.update(detections, frame)
        per_frame.append({"frame": frame.index, "ids": [t.track_id for t in tracks]})
        for tr in tracks:
            span = spans.setdefault(
                tr.track_id,
                {"class": tr.class_name, "first": frame.index, "last": frame.index,
                 "count": 0, "conf_sum": 0.0},
            )
            span["last"] = frame.index
            span["count"] += 1
            span["conf_sum"] += tr.confidence
    return {"frames": len(frames), "per_frame": per_frame, "tracks": spans}


def print_timeline(timeline: dict) -> None:
    print("  per-frame track ids:")
    for row in timeline["per_frame"]:
        ids = ",".join(str(i) for i in row["ids"]) or "-"
        print(f"    frame {row['frame']:>3}: [{ids}]")
    print("  track timeline:")
    print(f"    {'id':>3}  {'class':<8} {'frames':>6}  {'span':>10}  {'mean_conf':>9}")
    for tid, span in sorted(timeline["tracks"].items()):
        mean_conf = span["conf_sum"] / span["count"] if span["count"] else 0.0
        print(
            f"    {tid:>3}  {span['class']:<8} {span['count']:>6}  "
            f"{span['first']:>3}->{span['last']:<5} {mean_conf:>9.2f}"
        )


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
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


# --------------------------------------------------------------------------- #
# track-demo
# --------------------------------------------------------------------------- #


def cmd_track_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    kind, path = args.source
    if kind == "file":
        source = FileFrameSource(path)
    elif kind == "mock":
        source = MockFrameSource(num_frames=24)
    else:
        print(f"track-demo supports --source file|mock (got {kind!r})")
        return 2

    try:
        with source:
            frames = list(source)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error reading source: {exc}")
        return 1
    if not frames:
        print("no frames read from source")
        return 1
    width, height = frames[0].width, frames[0].height

    engine_exists = Path(config.model.engine_path).exists()
    if args.detector == "tensorrt" or (args.detector == "auto" and engine_exists):
        detector: Detector = TensorRTDetector(
            input_size=config.model.input_size,
            conf_threshold=config.model.conf_threshold,
            iou_threshold=config.model.iou_threshold,
            class_filter=tuple(config.model.class_filter),
        )
        detector.load(config.model.engine_path)
        det_kind = "tensorrt"
    else:
        detector = MockDetector(script=demo_scenario(len(frames), width, height))
        detector.load()
        det_kind = "mock(scripted)"

    tracker = build_tracker(config)
    print(
        f"track-demo: source={kind}:{path} frames={len(frames)} "
        f"{width}x{height} detector={det_kind}"
    )
    timeline = run_track_demo(frames, detector, tracker)
    detector.close()
    print_timeline(timeline)
    return 0


# --------------------------------------------------------------------------- #
# zones validate / show
# --------------------------------------------------------------------------- #


def cmd_zones_validate(args: argparse.Namespace) -> int:
    try:
        zone_set = load_zones(args.file)
    except ZoneError as exc:
        print(f"INVALID: {exc}")
        return 1
    w, h = zone_set.resolution
    print(f"OK: {len(zone_set)} zone(s), resolution {w}x{h}")
    return 0


def cmd_zones_show(args: argparse.Namespace) -> int:
    try:
        zone_set = ZoneSet.from_file(args.file)
    except ZoneError as exc:
        print(f"error: {exc}")
        return 1

    w, h = zone_set.resolution
    print(f"zones file: {args.file}  resolution={w}x{h}  zones={len(zone_set)}")
    for zone in zone_set:
        classes = ",".join(zone.classes) or "all"
        print(
            f"  {zone.id:<16} {zone.kind.value:<8} "
            f"classes={classes:<14} points={len(zone.polygon)}"
        )
    issues = validate_zone_set(zone_set)
    if issues:
        print("  warnings:")
        for issue in issues:
            print(f"    - {issue}")

    if args.overlay:
        import cv2

        from vigil.overlay import draw_zones

        image = cv2.imread(args.overlay)
        if image is None:
            print(f"cannot read overlay image: {args.overlay}")
            return 1
        dest = args.out or "zones_overlay.png"
        cv2.imwrite(dest, draw_zones(image, zone_set))
        print(f"  overlay written -> {dest}")
    return 0


def cmd_zones(args: argparse.Namespace) -> int:
    print("usage: vigil zones {validate|show} <file> [--overlay IMG --out OUT]")
    return 2


# --------------------------------------------------------------------------- #
# log verify
# --------------------------------------------------------------------------- #


def cmd_log_verify(args: argparse.Namespace) -> int:
    result = verify_file(args.path)
    genesis = result.genesis_hash[:12]
    terminal = (result.terminal_hash or "-")[:12]
    if result.ok:
        print(
            f"PASS: {result.entry_count} entr{'y' if result.entry_count == 1 else 'ies'}; "
            f"chain intact from genesis {genesis}.. to terminal {terminal}.."
        )
        for warning in result.warnings:
            print(f"  warning: {warning}")
        return 0
    where = "<read error>" if result.error_index is None else f"entry {result.error_index}"
    print(f"FAIL at {where}: {result.error}")
    return 1


def cmd_log(args: argparse.Namespace) -> int:
    print("usage: vigil log verify <path>")
    return 2


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #


def cmd_bench(args: argparse.Namespace) -> int:
    from vigil.eval import run_benchmark_dir

    report = run_benchmark_dir(args.clips)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    cfg = report["config"]
    noise = cfg["noise"]
    print(f"Vigil behaviour benchmark - clips: {args.clips}")
    print(
        f"  reference detector (NOT the TensorRT model): miss={noise['miss_rate']} "
        f"jitter={noise['jitter_px']}px fp={noise['fp_rate']}"
    )
    for group, g in report["groups"].items():
        d, t, z = g["detection"], g["tracking"], g["zones"]
        print(f"\n[{group}]  clips: {', '.join(g['clips'])}")
        print(f"  detection (reference): P={d['precision']:.3f} R={d['recall']:.3f}")
        print(
            f"  tracking : id_switches={t['id_switches']} "
            f"frag={t['mean_fragmentation']:.2f} lifetime={t['mean_track_lifetime']:.1f}"
        )
        print(
            f"  zones    : P={z['precision']:.3f} R={z['recall']:.3f} "
            f"matched={z['matched']} missed={z['missed']} false_events={z['false_events']}"
        )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vigil", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="capture -> detect -> track loop")
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--detector", choices=("auto", "mock", "tensorrt"), default="auto")
    p_run.add_argument("--show", action="store_true", help="live OpenCV window")
    p_run.add_argument("--max-frames", type=int, default=100, help="0 = until source ends")
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("track-demo", help="print a track timeline for a clip")
    p_demo.add_argument(
        "--source",
        nargs=2,
        metavar=("KIND", "PATH"),
        default=["file", "tests/data/demo_clip"],
        help="source kind (file|mock) and path",
    )
    p_demo.add_argument("--detector", choices=("auto", "mock", "tensorrt"), default="mock")
    p_demo.add_argument("--config", type=Path, default=None)
    p_demo.set_defaults(func=cmd_track_demo)

    p_zones = sub.add_parser("zones", help="zone file utilities (validate/show)")
    p_zones.set_defaults(func=cmd_zones)
    zsub = p_zones.add_subparsers(dest="zones_command")
    p_val = zsub.add_parser("validate", help="validate a zones JSON file")
    p_val.add_argument("file")
    p_val.set_defaults(func=cmd_zones_validate)
    p_zshow = zsub.add_parser("show", help="print zones; optionally overlay on a still")
    p_zshow.add_argument("file")
    p_zshow.add_argument("--overlay", help="still image to draw the zones on")
    p_zshow.add_argument("--out", help="overlay output path (default: zones_overlay.png)")
    p_zshow.set_defaults(func=cmd_zones_show)

    p_log = sub.add_parser("log", help="event log utilities (verify)")
    p_log.set_defaults(func=cmd_log)
    logsub = p_log.add_subparsers(dest="log_command")
    p_lv = logsub.add_parser("verify", help="verify a hash-chained JSONL log")
    p_lv.add_argument("path")
    p_lv.set_defaults(func=cmd_log_verify)

    p_bench = sub.add_parser("bench", help="run the behaviour benchmark on committed clips")
    p_bench.add_argument("--clips", default="benchmark/clips", help="clips directory")
    p_bench.add_argument("--json", action="store_true", help="emit the full JSON report")
    p_bench.set_defaults(func=cmd_bench)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Default to `run` when no subcommand (so `vigil --max-frames N` still works).
    if not argv:
        argv = ["run"]
    elif argv[0] not in _SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv

    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

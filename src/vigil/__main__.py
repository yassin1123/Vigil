"""`python -m vigil` — Day-1 frame-source smoke runner.

Loads the config, opens the configured FrameSource, pumps frames, and prints
index, timestamp, shape, and a measured wall-clock FPS. This is the seed of the
full pipeline runner (capture -> detect -> track -> zones -> log -> ui) wired up
on Day 6; for now it exercises the capture abstraction end to end.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from vigil.config import load_config
from vigil.frames import build_frame_source


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vigil", description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to vigil.yaml (default: vigil.yaml, then vigil.example.yaml).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Stop after this many frames (0 = run until the source ends).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    source = build_frame_source(config)

    print(f"source: {type(source).__name__}  (kind={config.source.kind})")

    count = 0
    start = time.monotonic()
    try:
        with source:
            for frame in source:
                count += 1
                if count <= 5 or count % 25 == 0:
                    print(
                        f"  frame {frame.index:>6}  t={frame.timestamp:8.3f}s  "
                        f"{frame.width}x{frame.height}"
                    )
                if args.max_frames and count >= args.max_frames:
                    break
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    elapsed = time.monotonic() - start
    fps = count / elapsed if elapsed > 0 else 0.0
    print(f"pumped {count} frames in {elapsed:.3f}s  ({fps:.1f} FPS wall-clock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

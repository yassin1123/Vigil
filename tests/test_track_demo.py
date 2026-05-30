"""`vigil track-demo`: a track timeline on the committed clip, no GPU."""
from __future__ import annotations

from pathlib import Path

import pytest

from vigil.__main__ import demo_scenario, main, run_track_demo
from vigil.detect import MockDetector
from vigil.frames import FileFrameSource
from vigil.track import ByteTrackTracker

DEMO_CLIP = Path(__file__).resolve().parent / "data" / "demo_clip"


def test_demo_timeline_shows_long_lived_and_transient_tracks():
    pytest.importorskip("cv2")
    with FileFrameSource(DEMO_CLIP) as source:
        frames = list(source)
    assert len(frames) == 24

    detector = MockDetector(
        script=demo_scenario(len(frames), frames[0].width, frames[0].height)
    )
    detector.load()
    tracker = ByteTrackTracker(confirm_frames=3)
    timeline = run_track_demo(frames, detector, tracker)

    assert timeline["frames"] == 24
    assert len(timeline["tracks"]) == 2  # the person and the car

    by_class = {span["class"]: span for span in timeline["tracks"].values()}
    assert set(by_class) == {"person", "car"}
    # The person spans most of the clip; the car enters and leaves mid-clip.
    assert by_class["person"]["count"] >= 18
    assert by_class["car"]["first"] >= 5
    assert by_class["car"]["last"] <= 20


def test_track_demo_cli_returns_zero():
    pytest.importorskip("cv2")
    rc = main(["track-demo", "--source", "file", str(DEMO_CLIP)])
    assert rc == 0

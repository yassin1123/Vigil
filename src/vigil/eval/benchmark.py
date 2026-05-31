"""Run the behaviour benchmark: replay clips through the pipeline and score.

For each clip: a reference detector (deterministic imperfect ground-truth replay)
-> ByteTrack -> zone event engine, then detection/tracking/zone scoring. Results
are aggregated per tier (tier1/2/3) and for the held-out generalization set.

Pipeline parameters are FIXED at the production defaults (not tuned per clip):
tracker confirm_frames=3 / lost_window=30, zone debounce=3. Documented so runs
are reproducible.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vigil.detect import MockDetector
from vigil.eval.clips import BenchmarkClip, load_clips
from vigil.eval.reference import NoiseModel, build_detection_script, ground_truth_detections
from vigil.eval.score import score_detections, score_tracking, score_zone_events
from vigil.track import ByteTrackTracker
from vigil.zones.engine import ZoneEventEngine
from vigil.zones.geometry import ZoneIndex

# Fixed production-default pipeline parameters (not tuned to the clips).
TRACKER_CONFIRM_FRAMES = 3
TRACKER_LOST_WINDOW = 30
ZONE_DEBOUNCE_FRAMES = 3


def run_clip(clip: BenchmarkClip, noise: NoiseModel | None = None) -> dict[str, Any]:
    """Run one clip end-to-end and return its detection/tracking/zone scores."""
    noise = noise or NoiseModel()
    script = build_detection_script(clip, noise)
    detector = MockDetector(script=script)
    detector.load()
    tracker = ByteTrackTracker(
        confirm_frames=TRACKER_CONFIRM_FRAMES, lost_window=TRACKER_LOST_WINDOW
    )
    engine = ZoneEventEngine(
        ZoneIndex(clip.zones, clip.resolution),
        enter_frames=ZONE_DEBOUNCE_FRAMES,
        exit_frames=ZONE_DEBOUNCE_FRAMES,
        utc_now=lambda: "benchmark",
    )

    dets_per_frame = []
    tracks_per_frame = []
    events: list[Any] = []
    for frame in range(clip.frames):
        info = SimpleNamespace(index=frame, timestamp=frame / clip.fps)
        dets = detector.infer(None)  # script-driven; ignores the image
        tracks = tracker.update(dets, info)
        events.extend(engine.update(tracks, info))
        dets_per_frame.append(dets)
        tracks_per_frame.append(tracks)

    return {
        "name": clip.name,
        "group": clip.group,
        "detection": score_detections(dets_per_frame, ground_truth_detections(clip)),
        "tracking": score_tracking(clip, tracks_per_frame),
        "zones": score_zone_events(clip.expected_events, events),
    }


def _aggregate(clip_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-clip scores within one group (sum counts, mean rates)."""
    det_tp = det_fp = det_fn = 0
    zone_exp = zone_act = zone_matched = zone_missed = zone_false = 0
    id_switches = 0
    frag_vals: list[float] = []
    lifetime_vals: list[float] = []
    for r in clip_results:
        det = r["detection"]["overall"]
        det_tp += det["tp"]
        det_fp += det["fp"]
        det_fn += det["fn"]
        z = r["zones"]
        zone_exp += z["expected"]
        zone_act += z["actual"]
        zone_matched += z["matched"]
        zone_missed += z["missed"]
        zone_false += z["false_events"]
        t = r["tracking"]
        id_switches += t["id_switches"]
        frag_vals.append(t["mean_fragmentation"])
        lifetime_vals.append(t["mean_track_lifetime"])

    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 1.0

    def mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "clips": [r["name"] for r in clip_results],
        "detection": {
            "tp": det_tp, "fp": det_fp, "fn": det_fn,
            "precision": rate(det_tp, det_tp + det_fp),
            "recall": rate(det_tp, det_tp + det_fn),
        },
        "tracking": {
            "id_switches": id_switches,
            "mean_fragmentation": mean(frag_vals),
            "mean_track_lifetime": mean(lifetime_vals),
        },
        "zones": {
            "expected": zone_exp, "actual": zone_act,
            "matched": zone_matched, "missed": zone_missed, "false_events": zone_false,
            "precision": rate(zone_matched, zone_matched + zone_false),
            "recall": rate(zone_matched, zone_matched + zone_missed),
            "false_event_rate": rate(zone_false, zone_act),
        },
    }


def run_benchmark(
    clips: list[BenchmarkClip], noise: NoiseModel | None = None
) -> dict[str, Any]:
    """Run all clips and aggregate by group (tier1/2/3 + heldout)."""
    noise = noise or NoiseModel()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_clip = []
    for clip in clips:
        result = run_clip(clip, noise)
        per_clip.append(result)
        by_group[result["group"]].append(result)

    groups = {g: _aggregate(rs) for g, rs in sorted(by_group.items())}
    return {
        "config": {
            "noise": vars(noise),
            "iou_threshold": 0.5,
            "tracker_confirm_frames": TRACKER_CONFIRM_FRAMES,
            "tracker_lost_window": TRACKER_LOST_WINDOW,
            "zone_debounce_frames": ZONE_DEBOUNCE_FRAMES,
        },
        "groups": groups,
        "clips": per_clip,
    }


def run_benchmark_dir(clips_dir: str | Path, noise: NoiseModel | None = None) -> dict[str, Any]:
    return run_benchmark(load_clips(clips_dir), noise)

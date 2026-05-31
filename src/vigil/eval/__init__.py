"""Vigil evaluation: tiered behaviour benchmark + scorers (model-agnostic)."""
from __future__ import annotations

from vigil.eval.benchmark import run_benchmark, run_benchmark_dir, run_clip
from vigil.eval.clips import BenchmarkClip, ExpectedEvent, GTObject, load_clips
from vigil.eval.reference import (
    NoiseModel,
    build_detection_script,
    ground_truth_detections,
)
from vigil.eval.score import score_detections, score_tracking, score_zone_events

__all__ = [
    "BenchmarkClip",
    "ExpectedEvent",
    "GTObject",
    "NoiseModel",
    "build_detection_script",
    "ground_truth_detections",
    "load_clips",
    "run_benchmark",
    "run_benchmark_dir",
    "run_clip",
    "score_detections",
    "score_tracking",
    "score_zone_events",
]

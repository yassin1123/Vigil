"""The benchmark on committed clips: deterministic, with honest invariants."""
from __future__ import annotations

from pathlib import Path

from vigil.eval import load_clips, run_benchmark, run_clip
from vigil.eval.clips import BenchmarkClip

CLIPS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "clips"


def _clip(name) -> BenchmarkClip:
    return next(c for c in load_clips(CLIPS_DIR) if c.name == name)


def test_committed_clips_load_with_all_tiers_and_heldout():
    clips = load_clips(CLIPS_DIR)
    groups = {c.group for c in clips}
    assert {"tier1", "tier2", "tier3", "heldout"} <= groups
    assert any(c.held_out for c in clips)  # the generalization set exists


def test_benchmark_is_deterministic():
    clips = load_clips(CLIPS_DIR)
    assert run_benchmark(clips) == run_benchmark(clips)


def test_report_has_every_group_and_three_metric_families():
    report = run_benchmark(load_clips(CLIPS_DIR))
    for group in ("tier1", "tier2", "tier3", "heldout"):
        g = report["groups"][group]
        assert set(g) >= {"detection", "tracking", "zones", "clips"}


def test_tracking_is_clean_across_all_groups():
    # ByteTrack + lifecycle: no ID switches anywhere, fragmentation ideal.
    report = run_benchmark(load_clips(CLIPS_DIR))
    for group, g in report["groups"].items():
        assert g["tracking"]["id_switches"] == 0, group
        assert g["tracking"]["mean_fragmentation"] == 1.0, group


def test_zones_are_exact_except_the_known_occlusion_blink():
    report = run_benchmark(load_clips(CLIPS_DIR))
    for group in ("tier1", "tier3", "heldout"):
        z = report["groups"][group]["zones"]
        assert z["precision"] == 1.0 and z["recall"] == 1.0, group
    # tier2 carries exactly the documented occlusion false events
    assert report["groups"]["tier2"]["zones"]["recall"] == 1.0
    assert report["groups"]["tier2"]["zones"]["false_events"] == 2


def test_boundary_jitter_is_fully_suppressed():
    # debounce turns many raw boundary crossings into zero events
    result = run_clip(_clip("t2_jitter"))
    assert result["zones"]["actual"] == 0
    assert result["zones"]["false_events"] == 0


def test_long_occlusion_blink_is_localized_to_that_clip():
    occ = run_clip(_clip("t2_occlusion"))
    assert occ["zones"]["matched"] == 2  # the real ENTRY/EXIT are still found
    assert occ["zones"]["false_events"] == 2  # plus the documented blink
    # tracking still keeps the id through the occlusion
    assert occ["tracking"]["id_switches"] == 0


def test_brief_occlusion_is_handled_cleanly():
    held = run_clip(_clip("h1_diag_occlusion"))  # 2-frame gap <= debounce
    assert held["zones"]["false_events"] == 0
    assert held["zones"]["matched"] == 2


def test_detection_recall_reflects_reference_noise():
    # ~8% miss rate -> recall well below 1 but clearly high; precision high.
    report = run_benchmark(load_clips(CLIPS_DIR))
    overall = report["groups"]["tier1"]["detection"]
    assert 0.8 <= overall["recall"] < 1.0
    assert overall["precision"] >= 0.9

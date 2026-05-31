# Vigil Behaviour Benchmark

Measures what the pipeline **actually does** — detection, tracking stability, and
zone-event correctness — on committed, annotated clips across three difficulty
tiers plus a **held-out** generalization set. Honest measurement, not a flattering
demo: where Vigil falls short, the numbers say so (see *Findings*).

```bash
make bench                 # or: PYTHONPATH=src python3 -m vigil bench
python3 -m vigil bench --json > report.json
```

## What is real vs. what is a stand-in (read this first)

The three metric families have different hardware dependencies, and the benchmark
is explicit about which is which:

| Metric | Depends on | Measured here? |
|---|---|---|
| **Tracking stability** (ID switches, fragmentation, lifetime) | Vigil's ByteTrack code | ✅ **Real** — pure-CPU, fully measured |
| **Zone-event accuracy** (matched ENTRY/EXIT, false-event rate) | Vigil's zone engine | ✅ **Real** — pure-CPU, fully measured |
| **Detection P/R** | the YOLOv8n TensorRT **model** | ⚠️ **Reference only** off-GPU |

Off-GPU there is no YOLOv8n model, so the benchmark drives the pipeline with a
**reference detector** (`src/vigil/eval/reference.py`): it replays each clip's
ground-truth boxes with fixed, seeded imperfection — **8% missed detections, ±4px
localization jitter, 3% false-positive rate**. The detection P/R reported off-GPU
therefore reflects *that reference detector*, not the real model. The tracking and
zone numbers, however, are genuine measurements of Vigil's own code under
realistic imperfect input. The real model's detection P/R is measured on the
Jetson against real footage using the **same scorers** (`score.py`).

## Clips and annotation provenance

These are **synthetic annotation-level sequences** (object trajectories + zones +
expected events) — not pixel footage. They were generated deterministically by
[`generate.py`](generate.py) (committed for reproducibility), and the **expected
events are derived independently** of Vigil's engine: point-in-polygon over the
true trajectory + a K-frame debounce. So the benchmark measures the pipeline
rather than restating it. Real footage is the documented next step (drop annotated
video in, run the real detector on the Jetson; the scorers are unchanged).

The hard cases are genuinely hard: real multi-frame occlusion (the object is
present but undetectable), boundary straddling, crossing paths, mixed classes, and
EXCLUDE-zone suppression.

| Tier | Clip | What it stresses |
|---|---|---|
| 1 | `t1_single_cross` | one clear object crossing one zone |
| 2 | `t2_crossing` | two people crossing paths through a zone |
| 2 | `t2_occlusion` | a 4-frame mid-zone occlusion |
| 2 | `t2_jitter` | a centroid straddling the zone edge (debounce stress) |
| 3 | `t3_busy` | multiple zones + classes: dock (person-only), road (EXCLUDE), a car that must be ignored |
| held-out | `h1_diag_occlusion` | diagonal crossing of a *different* zone + brief occlusion |
| held-out | `h2_two_persons_one_car` | two people cross a person-only zone; a car is ignored |

The held-out clips use geometry the pipeline parameters were never adjusted for —
and the parameters are fixed at production defaults for every clip
(`tracker confirm_frames=3 / lost_window=30`, `zone debounce=3`), so nothing is
tuned to any clip. Runs are deterministic (per-clip CRC32 seed).

## Findings (current, honest)

- **Tracking: clean across every tier and the held-out set** — 0 ID switches,
  fragmentation 1.0. ByteTrack + the lifecycle handle crossing paths, occlusion,
  and busy scenes, and generalize to unseen geometry.
- **Zones: correct everywhere except long occlusion.** Class filtering and
  EXCLUDE suppression are exact (tier 3); boundary jitter is fully absorbed
  (`t2_jitter` emits **0** events vs. the many raw boundary crossings — the
  debounce working as intended); held-out zones are exact.
- **Known limitation — `t2_occlusion` emits 2 false events.** When an object is
  occluded for **longer than the zone exit-debounce (3 frames)**, the zone engine
  blinks: the tracker correctly keeps the object's ID through the gap (0 ID
  switches), but it only *emits* matched tracks, so the zone engine sees the
  object vanish and fires a spurious EXIT, then ENTRY on reappearance. Brief
  occlusion (≤ debounce, e.g. `h1_diag_occlusion`'s 2-frame gap) is handled
  cleanly. The fix — surfacing within-`lost_window` coasting tracks to the zone
  engine so it knows the object is still present — is tracked as future work and
  is *not* hidden by tuning the clips around it.

Detection recall (~0.89–0.93) and precision (~0.97–0.99) reflect the reference
detector's injected 8% miss / 3% FP rates, not the model.

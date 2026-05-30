#!/usr/bin/env python3
"""Vigil device probe — reports Jetson / CUDA / TensorRT / camera readiness.

Part of Day-1 hardware bring-up. This script runs *anywhere*:

  * On a Jetson it probes the platform and produces a real device report.
  * On any other machine it prints a clear "not on target hardware" notice,
    still writes a (mostly empty) report for the record, and exits 0.

It never crashes on a missing tool or file — every probe degrades to a
"not detected" entry that says exactly what is missing and how to fix it.

Usage:
    python3 scripts/check_device.py [--output docs/baseline/device_report.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_OUTPUT = Path("docs/baseline/device_report.json")
SCHEMA_VERSION = 1

# L4T (Linux for Tegra) major release -> JetPack family. Kept coarse on purpose:
# the L4T release does not uniquely pin a JetPack point release, so we do not
# pretend it does.
L4T_TO_JETPACK = {
    "36": "JetPack 6.x",
    "35": "JetPack 5.x",
    "34": "JetPack 5.0 (Developer Preview)",
    "32": "JetPack 4.x",
}


# --------------------------------------------------------------------------- #
# Probe result type
# --------------------------------------------------------------------------- #


@dataclass
class Probe:
    """The outcome of a single capability check."""

    name: str
    detected: bool
    value: Optional[str] = None
    source: Optional[str] = None
    hint: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def status_line(self) -> str:
        mark = "OK " if self.detected else "-- "
        body = self.value if self.detected and self.value else "not detected"
        line = f"  [{mark}] {self.name:<16} {body}"
        if not self.detected and self.hint:
            line += f"\n        fix: {self.hint}"
        return line


# --------------------------------------------------------------------------- #
# Low-level helpers (never raise)
# --------------------------------------------------------------------------- #


def _read_text(path: str) -> Optional[str]:
    """Read a text file, returning None on any error (and stripping NULs)."""
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return data.replace("\x00", "").strip()


def _run(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    """Run a command and return stripped stdout, or None on any failure."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _glob(pattern: str) -> list[str]:
    try:
        return sorted(str(p) for p in Path("/").glob(pattern.lstrip("/")))
    except OSError:
        return []


# --------------------------------------------------------------------------- #
# Individual probes
# --------------------------------------------------------------------------- #


def probe_model() -> Probe:
    """Jetson board model from the device tree."""
    model = _read_text("/proc/device-tree/model")
    if model:
        return Probe("model", True, value=model, source="/proc/device-tree/model")
    return Probe(
        "model",
        False,
        source="/proc/device-tree/model",
        hint="Not a Tegra device-tree platform - expected off-target.",
    )


def probe_l4t() -> Probe:
    """L4T release and the JetPack family it belongs to."""
    raw = _read_text("/etc/nv_tegra_release")
    if not raw:
        return Probe(
            "jetpack/l4t",
            False,
            source="/etc/nv_tegra_release",
            hint="File absent - JetPack not installed or not a Jetson.",
        )
    # Example: "# R36 (release), REVISION: 3.0, GCID: ..., BOARD: generic, ..."
    m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", raw)
    if not m:
        return Probe(
            "jetpack/l4t",
            True,
            value=raw.splitlines()[0],
            source="/etc/nv_tegra_release",
            extra={"raw": raw},
        )
    major, revision = m.group(1), m.group(2)
    l4t = f"R{major}.{revision}"
    jetpack = L4T_TO_JETPACK.get(major, f"JetPack (L4T R{major}, unknown mapping)")
    return Probe(
        "jetpack/l4t",
        True,
        value=f"{jetpack}  (L4T {l4t})",
        source="/etc/nv_tegra_release",
        extra={"l4t_major": major, "l4t_revision": revision, "jetpack": jetpack},
    )


def probe_cuda() -> Probe:
    """CUDA toolkit version, preferring the structured version.json."""
    txt = _read_text("/usr/local/cuda/version.json")
    if txt:
        try:
            data = json.loads(txt)
            version = data.get("cuda", {}).get("version")
            if version:
                return Probe(
                    "cuda",
                    True,
                    value=version,
                    source="/usr/local/cuda/version.json",
                )
        except (json.JSONDecodeError, AttributeError):
            pass

    txt = _read_text("/usr/local/cuda/version.txt")
    if txt:
        return Probe("cuda", True, value=txt, source="/usr/local/cuda/version.txt")

    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+([\d.]+)", out)
        if m:
            return Probe("cuda", True, value=m.group(1), source="nvcc --version")

    return Probe(
        "cuda",
        False,
        source="/usr/local/cuda, nvcc",
        hint=(
            "CUDA not found. On JetPack it ships preinstalled; ensure "
            "/usr/local/cuda is on PATH (add to ~/.bashrc) and reflash if absent."
        ),
    )


def probe_tensorrt() -> Probe:
    """TensorRT version via the Python runtime first, then dpkg."""
    out = _run([sys.executable, "-c", "import tensorrt; print(tensorrt.__version__)"])
    if out:
        return Probe("tensorrt", True, value=out, source="python -c import tensorrt")

    out = _run(["dpkg-query", "-W", "-f=${Version}", "nvidia-tensorrt"])
    if out:
        return Probe("tensorrt", True, value=out, source="dpkg nvidia-tensorrt")

    out = _run(["dpkg-query", "-W", "-f=${Version}", "libnvinfer-dev"])
    if out:
        return Probe("tensorrt", True, value=out, source="dpkg libnvinfer-dev")

    return Probe(
        "tensorrt",
        False,
        source="python import / dpkg",
        hint=(
            "TensorRT not found. Install with: "
            "sudo apt-get install nvidia-tensorrt  (it is part of JetPack)."
        ),
    )


def probe_gpu() -> Probe:
    """Integrated GPU presence; nvidia-smi if available, else sysfs/devnodes."""
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    if out:
        return Probe("gpu", True, value=out.replace("\n", "; "), source="nvidia-smi")

    for node in ("/dev/nvhost-gpu", "/dev/nvhost-ctrl-gpu", "/sys/devices/gpu.0"):
        if Path(node).exists():
            return Probe(
                "gpu",
                True,
                value="Tegra integrated GPU present",
                source=node,
            )

    return Probe(
        "gpu",
        False,
        source="nvidia-smi / sysfs",
        hint="No GPU device nodes - expected off-target; on Jetson reflash JetPack.",
    )


def probe_power_mode() -> Probe:
    """Active nvpmodel power mode (best-effort; may need sudo)."""
    out = _run(["nvpmodel", "-q"])
    if out:
        m = re.search(r"NV Power Mode:\s*(.+)", out)
        value = m.group(1).strip() if m else out.splitlines()[0]
        clocks = _run(["jetson_clocks", "--show"], timeout=5.0)
        extra: dict[str, Any] = {"raw": out}
        if clocks:
            extra["jetson_clocks"] = clocks.splitlines()[:4]
        return Probe("power_mode", True, value=value, source="nvpmodel -q", extra=extra)

    return Probe(
        "power_mode",
        False,
        source="nvpmodel -q",
        hint=(
            "Could not query power mode (nvpmodel missing or needs sudo). "
            "Try: sudo nvpmodel -q   and set max perf with: sudo nvpmodel -m 0"
        ),
    )


def probe_camera() -> Probe:
    """V4L2 video device nodes. CSI confirmation is done by camera_probe.py."""
    videos = _glob("/dev/video*")
    listing = _run(["v4l2-ctl", "--list-devices"], timeout=5.0)
    extra: dict[str, Any] = {"video_nodes": videos}
    if listing:
        extra["v4l2_list_devices"] = listing
    note = "Run scripts/camera_probe.py to confirm the CSI (argus) pipeline streams."
    if videos:
        return Probe(
            "camera",
            True,
            value=f"{len(videos)} video node(s): {', '.join(videos)}",
            source="/dev/video*",
            hint=note,
            extra=extra,
        )
    return Probe(
        "camera",
        False,
        source="/dev/video*",
        hint=(
            "No /dev/video* nodes. Seat the CSI ribbon (contacts toward the "
            "heatsink), reboot, then re-check. " + note
        ),
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def is_jetson() -> bool:
    """True when running on Tegra/Jetson hardware."""
    model = _read_text("/proc/device-tree/model") or ""
    return "jetson" in model.lower() or Path("/etc/nv_tegra_release").exists()


def build_report() -> dict[str, Any]:
    on_target = is_jetson()

    probes: dict[str, Probe] = {
        "model": probe_model(),
        "jetpack/l4t": probe_l4t(),
        "cuda": probe_cuda(),
        "tensorrt": probe_tensorrt(),
        "gpu": probe_gpu(),
        "power_mode": probe_power_mode(),
        "camera": probe_camera(),
    }

    # Inference-core readiness: what we need before building the pipeline.
    required = ("cuda", "tensorrt", "gpu", "camera")
    missing = [name for name in required if not probes[name].detected]
    ready = on_target and not missing

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "host_platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "on_target_hardware": on_target,
        "ready_for_inference_core": ready,
        "missing_for_readiness": missing,
        "probes": {name: asdict(p) for name, p in probes.items()},
        "_probe_objects": probes,  # popped before serialisation; used for printing
    }


def print_summary(report: dict[str, Any]) -> None:
    on_target = report["on_target_hardware"]
    print("=" * 64)
    print("  VIGIL device probe")
    print("=" * 64)
    host = report["host_platform"]
    print(
        f"  host: {report['hostname']}  "
        f"({host['system']} {host['release']} / {host['machine']}, "
        f"py{host['python']})"
    )

    if not on_target:
        print()
        print("  NOT ON TARGET HARDWARE - this is not a Jetson.")
        print("  Probes that need the device are reported as 'not detected'.")
        print("  The script ran cleanly and wrote the report anyway.")
        print()

    probes: dict[str, Probe] = report["_probe_objects"]
    print("  capabilities:")
    for probe in probes.values():
        print(probe.status_line())

    print()
    if report["ready_for_inference_core"]:
        print("  READY: CUDA, TensorRT, GPU and a camera node are all present.")
    elif on_target:
        missing = ", ".join(report["missing_for_readiness"])
        print(f"  NOT READY: missing {missing}. See per-item fix hints above.")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a Jetson for Vigil readiness and write a JSON report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON report (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report()
    print_summary(report)

    report.pop("_probe_objects", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  report written: {args.output}")

    # Always exit 0: this is a report, not a gate. Readiness lives in the JSON
    # (`ready_for_inference_core`) and the printed summary.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Log export sinks: verifiable copies, auto-export on insertion, corruption caught."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from vigil.config import VigilConfig
from vigil.log.logger import HashChainLogger
from vigil.log.sinks import (
    AutoExporter,
    LocalDirSink,
    NullSink,
    USBExportSink,
    build_export_sink,
)
from vigil.log.verify import verify_file

FIXED_UTC = "2026-05-31T00:00:00+00:00"


def _make_log(path, n=3):
    with HashChainLogger(path, utc_now=lambda: FIXED_UTC) as log:
        for i in range(n):
            log.append(f"EVENT_{i}", {"i": i})


def _read(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


# --- LocalDirSink produces a verifiable copy ------------------------------- #


def test_localdir_export_is_verifiable(tmp_path):
    log = tmp_path / "events.jsonl"
    _make_log(log, 3)
    dest = tmp_path / "dest"
    dest.mkdir()

    sink = LocalDirSink(dest)
    assert sink.available() == dest
    result = sink.export(log)

    assert result.ok and result.verified and result.entry_count == 3
    copy = dest / "vigil-export" / "events.jsonl"
    report = dest / "vigil-export" / "events.jsonl.verification.json"
    assert copy.exists() and report.exists()
    assert verify_file(copy).ok  # the copy independently verifies
    assert json.loads(report.read_text())["verified"] is True


def test_null_sink_exports_nothing(tmp_path):
    log = tmp_path / "events.jsonl"
    _make_log(log, 1)
    sink = NullSink()
    assert sink.available() is None
    assert not sink.export(log).ok


# --- simulated USB insertion triggers auto-export + logs the event --------- #


def test_drive_insertion_auto_exports_and_logs_event(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.mkdir()
    log = tmp_path / "events.jsonl"
    logger = HashChainLogger(log, utc_now=lambda: FIXED_UTC)
    logger.append("A", {"i": 0})
    logger.append("B", {"i": 1})

    sink = USBExportSink(candidate_roots=[mounts], require_mountpoint=False)
    auto = AutoExporter(logger, sink, enabled=True, utc_now=lambda: FIXED_UTC)

    assert auto.poll() is None  # no drive yet

    drive = mounts / "usb0"
    drive.mkdir()  # <-- simulate insertion
    result = auto.poll()
    assert result is not None and result.ok and result.verified

    # exported copy + report exist and the copy verifies
    copy = drive / "vigil-export" / "events.jsonl"
    assert copy.exists() and verify_file(copy).ok
    assert (drive / "vigil-export" / "events.jsonl.verification.json").exists()

    assert auto.poll() is None  # same drive still present -> no re-export

    shutil.rmtree(drive)  # drive removed
    assert auto.poll() is None
    drive.mkdir()  # re-inserted -> exports again
    assert auto.poll() is not None

    logger.close()
    entries = _read(log)
    exported = [e for e in entries if e["event_type"] == "LOG_EXPORTED"]
    assert len(exported) == 2  # one per insertion, recorded in the chain
    assert exported[0]["payload"]["detail"]["verified"] is True
    assert verify_file(log).ok  # the source chain (incl. export events) stays valid


def test_disabled_auto_export_does_nothing(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.mkdir()
    log = tmp_path / "events.jsonl"
    logger = HashChainLogger(log, utc_now=lambda: FIXED_UTC)
    logger.append("A", {"i": 0})
    sink = USBExportSink(candidate_roots=[mounts], require_mountpoint=False)
    auto = AutoExporter(logger, sink, enabled=False)
    (mounts / "usb0").mkdir()
    assert auto.poll() is None
    logger.close()
    assert all(e["event_type"] != "LOG_EXPORTED" for e in _read(log))


# --- post-copy verification catches a corrupted destination ---------------- #


def test_corrupted_copy_is_caught_by_post_copy_verification(tmp_path):
    log = tmp_path / "events.jsonl"
    _make_log(log, 3)
    dest = tmp_path / "dest"
    dest.mkdir()

    def corrupting_copier(src, dst):
        data = bytearray(Path(src).read_bytes())
        data[len(data) // 2] ^= 0x01  # flip a byte somewhere in the chain
        Path(dst).write_bytes(bytes(data))

    sink = LocalDirSink(dest, copier=corrupting_copier)
    result = sink.export(log)

    assert not result.ok and not result.verified
    assert result.error and "verification failed" in result.error
    report = dest / "vigil-export" / "events.jsonl.verification.json"
    assert json.loads(report.read_text())["verified"] is False


# --- config factory -------------------------------------------------------- #


def test_build_export_sink_from_config(tmp_path):
    null_cfg = VigilConfig.from_dict({"log": {"export_sink": "null"}})
    local_cfg = VigilConfig.from_dict(
        {"log": {"export_sink": "localdir", "export_dir": str(tmp_path)}}
    )
    usb_cfg = VigilConfig.from_dict({"log": {"export_sink": "usb"}})
    assert isinstance(build_export_sink(null_cfg), NullSink)
    assert isinstance(build_export_sink(local_cfg), LocalDirSink)
    assert isinstance(build_export_sink(usb_cfg), USBExportSink)

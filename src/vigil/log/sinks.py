"""Log export behind a pluggable sink — USB today, more backends later.

The logger does not know or care where its log goes. Export is an `ExportSink`:
`available()` says whether a destination is ready, `export(log_path)` copies the
log there and RE-VERIFIES the chain at the destination (a copy is never assumed
intact). Auto-export fires when a drive appears and records a LOG_EXPORTED system
event into the chain itself.

KNOWN LIMITATION (acknowledged, not hidden): USB export only protects the record
once a drive is inserted and the copy succeeds. If the device is physically
destroyed before any export, the on-device log is lost — USB alone is not a
black box. `ExportSink` is precisely the seam where the roadmap's resilient
backends — mesh streaming, RF burst, armoured-memory — plug in on Day 7 (which
lays those rails). Those backends are intentionally NOT built here; only the
interface they will implement, and a USB implementation, exist today.
"""
from __future__ import annotations

import abc
import json
import os
import platform
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from vigil.log.verify import verify_file
from vigil.types import SystemEvent, SystemEventType

if TYPE_CHECKING:
    from vigil.config import VigilConfig
    from vigil.log.logger import HashChainLogger

EXPORT_DIRNAME = "vigil-export"
Copier = Callable[[str, str], object]


@dataclass
class ExportResult:
    ok: bool
    sink: str
    destination: Optional[str]
    files: list[str] = field(default_factory=list)
    verified: bool = False
    entry_count: int = 0
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


def _fsync_path(path: Path) -> None:
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def _copy_and_verify(
    log_path: Path, dest_root: Path, sink_name: str, copier: Copier
) -> ExportResult:
    """Copy the log to dest, write a verification report, and verify the COPY."""
    log_path = Path(log_path)
    if not log_path.exists():
        return ExportResult(
            ok=False, sink=sink_name, destination=str(dest_root),
            error=f"log not found: {log_path}",
        )

    export_dir = Path(dest_root) / EXPORT_DIRNAME
    export_dir.mkdir(parents=True, exist_ok=True)
    dest_log = export_dir / log_path.name

    try:
        copier(str(log_path), str(dest_log))
    except OSError as exc:
        return ExportResult(
            ok=False, sink=sink_name, destination=str(export_dir),
            error=f"copy failed: {exc}",
        )
    _fsync_path(dest_log)

    # Never trust the copy — verify the chain at the destination.
    result = verify_file(dest_log)
    report = {
        "verified": result.ok,
        "entry_count": result.entry_count,
        "genesis_hash": result.genesis_hash,
        "terminal_hash": result.terminal_hash,
        "error_index": result.error_index,
        "error": result.error,
        "warnings": result.warnings,
        "source": str(log_path),
        "exported_file": dest_log.name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = export_dir / (log_path.name + ".verification.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _fsync_path(report_path)

    return ExportResult(
        ok=result.ok,
        sink=sink_name,
        destination=str(export_dir),
        files=[dest_log.name, report_path.name],
        verified=result.ok,
        entry_count=result.entry_count,
        error=None if result.ok else f"destination verification failed: {result.error}",
    )


class ExportSink(abc.ABC):
    """Where the log goes. Backends differ; the logger never knows which."""

    name: str = "sink"

    @abc.abstractmethod
    def available(self) -> Optional[Path]:
        """Return the destination root if an export target is ready, else None."""

    @abc.abstractmethod
    def export(self, log_path: str | Path) -> ExportResult:
        """Copy the log to the target and re-verify the chain there."""


class NullSink(ExportSink):
    """Exports nothing — disables export while keeping the interface."""

    name = "null"

    def available(self) -> Optional[Path]:
        return None

    def export(self, log_path: str | Path) -> ExportResult:
        return ExportResult(ok=False, sink=self.name, destination=None,
                            error="null sink exports nothing")


class LocalDirSink(ExportSink):
    """Exports to a fixed local directory — the CI/test stand-in for a drive."""

    name = "localdir"

    def __init__(self, dest_root: str | Path, copier: Copier = shutil.copy2) -> None:
        self.dest_root = Path(dest_root)
        self._copier = copier

    def available(self) -> Optional[Path]:
        return self.dest_root if self.dest_root.is_dir() else None

    def export(self, log_path: str | Path) -> ExportResult:
        root = self.available()
        if root is None:
            return ExportResult(ok=False, sink=self.name, destination=str(self.dest_root),
                                error="destination directory not available")
        return _copy_and_verify(Path(log_path), root, self.name, self._copier)


class USBExportSink(ExportSink):
    """Exports to a mounted removable drive (platform-guarded mount polling).

    Detection is a heuristic: writable mount points under the platform's removable
    roots (/media, /run/media, /mnt on Linux; /Volumes on macOS; removable drive
    letters on Windows). Tests inject `candidate_roots` (+ require_mountpoint=False)
    to simulate a drive appearing without a real mount.
    """

    name = "usb"

    def __init__(
        self,
        candidate_roots: Optional[list[str | Path]] = None,
        *,
        require_mountpoint: bool = True,
        copier: Copier = shutil.copy2,
    ) -> None:
        self._candidate_roots = (
            [Path(r) for r in candidate_roots] if candidate_roots is not None else None
        )
        self._require_mountpoint = require_mountpoint
        self._copier = copier

    def _iter_children(self, roots: list[Path]) -> list[Path]:
        mounts: list[Path] = []
        for base in roots:
            try:
                if not base.is_dir():
                    continue
                for child in sorted(base.iterdir()):
                    if not child.is_dir():
                        continue
                    if self._require_mountpoint and not os.path.ismount(child):
                        continue
                    if not os.access(child, os.W_OK):
                        continue
                    mounts.append(child)
            except OSError:
                continue
        return mounts

    def _mounts(self) -> list[Path]:
        if self._candidate_roots is not None:
            return self._iter_children(self._candidate_roots)
        system = platform.system()
        if system == "Windows":
            return _windows_removable_drives()
        return self._iter_children(_platform_mount_roots())

    def available(self) -> Optional[Path]:
        mounts = self._mounts()
        return mounts[0] if mounts else None

    def export(self, log_path: str | Path) -> ExportResult:
        root = self.available()
        if root is None:
            return ExportResult(ok=False, sink=self.name, destination=None,
                                error="no removable drive detected")
        return _copy_and_verify(Path(log_path), root, self.name, self._copier)


def _platform_mount_roots() -> list[Path]:
    system = platform.system()
    if system == "Linux":
        roots = [Path("/media"), Path("/run/media"), Path("/mnt")]
        user = os.environ.get("USER") or os.environ.get("LOGNAME")
        if user:
            roots += [Path("/media") / user, Path("/run/media") / user]
        return roots
    if system == "Darwin":
        return [Path("/Volumes")]
    return []


def _windows_removable_drives() -> list[Path]:
    try:
        import ctypes
        import string

        drive_removable = 2
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        bitmask = kernel32.GetLogicalDrives()
        out: list[Path] = []
        for i, letter in enumerate(string.ascii_uppercase):
            if not bitmask & (1 << i):
                continue
            root = f"{letter}:\\"
            if kernel32.GetDriveTypeW(root) == drive_removable and os.access(root, os.W_OK):
                out.append(Path(root))
        return out
    except (OSError, AttributeError):
        return []


class AutoExporter:
    """Exports the log when a sink's destination first appears, once per insertion.

    On a fresh destination it runs `sink.export()` and appends a LOG_EXPORTED
    system event to the chain (recording the outcome, verified or not). It does
    not re-export while the same destination stays present; removing and
    re-inserting a drive triggers a new export.
    """

    def __init__(
        self,
        logger: "HashChainLogger",
        sink: ExportSink,
        *,
        enabled: bool = True,
        utc_now: Optional[Callable[[], str]] = None,
    ) -> None:
        self.logger = logger
        self.sink = sink
        self.enabled = enabled
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat())
        self._last_dest: Optional[Path] = None

    def poll(self, frame_info: object | None = None) -> Optional[ExportResult]:
        if not self.enabled:
            return None
        dest = self.sink.available()
        if dest is None:
            self._last_dest = None
            return None
        if dest == self._last_dest:
            return None  # same drive still present; already exported
        self._last_dest = dest

        result = self.sink.export(self.logger.path)

        ts_mono = getattr(frame_info, "timestamp", None)
        if ts_mono is None:
            ts_mono = 0.0
        event = SystemEvent(
            SystemEventType.LOG_EXPORTED,
            self._utc_now(),
            float(ts_mono),
            detail={
                "sink": result.sink,
                "destination": result.destination,
                "files": result.files,
                "entry_count": result.entry_count,
                "verified": result.verified,
                "error": result.error,
            },
        )
        self.logger.log_system_event(event)
        return result


def build_export_sink(config: "VigilConfig") -> ExportSink:
    """Construct the export sink named by config.log.export_sink."""
    kind = config.log.export_sink
    if kind == "null":
        return NullSink()
    if kind == "localdir":
        return LocalDirSink(config.log.export_dir)
    if kind == "usb":
        return USBExportSink()
    raise ValueError(f"unknown export_sink: {kind!r} (expected usb|null|localdir)")

"""Typed configuration for the whole of Vigil, loaded from a single vigil.yaml.

One file configures every subsystem: camera, frame source, detection model,
zones, log, UI, and telemetry thresholds. Sections are plain dataclasses with
defaults, so a minimal (or empty) YAML still loads. Unknown keys raise a clear
`ConfigError` rather than being silently ignored — that turns a typo in the
config into an immediate, explained failure instead of a mystery at runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (Path("vigil.yaml"), Path("vigil.example.yaml"))
SOURCE_KINDS: frozenset[str] = frozenset({"csi", "file", "mock"})


class ConfigError(ValueError):
    """Raised when a config file or mapping is malformed."""


@dataclass
class CameraConfig:
    """CSI camera parameters (used when source.kind == 'csi')."""

    sensor_id: int = 0
    width: int = 1920
    height: int = 1080
    framerate: int = 30
    flip_method: int = 0  # nvvidconv: 0=none, 2=180deg


@dataclass
class FileSourceConfig:
    """A video file or a directory of images (CI / benchmark)."""

    path: str = "tests/data/clip"
    fps: float = 30.0
    loop: bool = False


@dataclass
class MockSourceConfig:
    """Deterministic synthetic frames (unit tests, no hardware)."""

    width: int = 640
    height: int = 480
    num_frames: int = 100
    fps: float = 30.0


@dataclass
class SourceConfig:
    """Which frame source to use, plus per-source parameters."""

    kind: str = "mock"  # csi | file | mock
    file: FileSourceConfig = field(default_factory=FileSourceConfig)
    mock: MockSourceConfig = field(default_factory=MockSourceConfig)

    @classmethod
    def from_dict(cls, data: Any) -> "SourceConfig":
        if data is None:
            return cls()
        _require_mapping(data, "source")
        _reject_unknown(cls, data, "source")
        kind = data.get("kind", "mock")
        if kind not in SOURCE_KINDS:
            raise ConfigError(
                f"source.kind must be one of {sorted(SOURCE_KINDS)}, got {kind!r}"
            )
        return cls(
            kind=kind,
            file=_build(FileSourceConfig, data.get("file"), "source.file"),
            mock=_build(MockSourceConfig, data.get("mock"), "source.mock"),
        )


@dataclass
class ModelConfig:
    """Detection model paths and thresholds (TensorRT INT8 engine; Day 1 S3)."""

    engine_path: str = "models/yolov8n.engine"
    onnx_path: str = "models/yolov8n.onnx"
    weights_path: str = "models/yolov8n.pt"
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_filter: list[int] = field(default_factory=list)  # empty = keep all


@dataclass
class TrackerConfig:
    """ByteTrack parameters + lifecycle thresholds (Day 3). No ReID — offline."""

    track_thresh: float = 0.5  # high-confidence threshold for the first stage
    low_thresh: float = 0.1  # floor for low-confidence (second stage)
    match_thresh: float = 0.8  # IoU-distance threshold (cost = 1 - IoU)
    confirm_frames: int = 3  # consecutive matches to confirm (TENTATIVE->CONFIRMED)
    lost_window: int = 30  # frames a lost track is kept for re-association
    min_box_area: float = 10.0  # ignore boxes smaller than this (pixels^2)


@dataclass
class ZonesConfig:
    """Polygon zone definitions (Day 4)."""

    path: str = "config/zones.yaml"


@dataclass
class LogConfig:
    """Tamper-evident hash-chained event log (Day 5)."""

    path: str = "logs/events.jsonl"
    export_dir: str = "/media/vigil-export"  # USB mount target for offline export


@dataclass
class UIConfig:
    """Operator UI (Day 6). The web view binds to localhost only — Vigil has
    no remote-access surface."""

    opencv_window: bool = True
    web_enabled: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    mjpeg_fps: int = 15


@dataclass
class TelemetryConfig:
    """Thermal / power sampling and safety thresholds (Day 2 / Day 6)."""

    poll_interval_s: float = 2.0
    max_temp_c: float = 80.0
    max_power_w: float = 15.0


@dataclass
class VigilConfig:
    """The complete, typed system configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, data: Any) -> "VigilConfig":
        data = data or {}
        _require_mapping(data, "<root>")
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(
                f"unknown top-level section(s): {sorted(unknown)} "
                f"(allowed: {sorted(allowed)})"
            )
        return cls(
            camera=_build(CameraConfig, data.get("camera"), "camera"),
            source=SourceConfig.from_dict(data.get("source")),
            model=_build(ModelConfig, data.get("model"), "model"),
            tracker=_build(TrackerConfig, data.get("tracker"), "tracker"),
            zones=_build(ZonesConfig, data.get("zones"), "zones"),
            log=_build(LogConfig, data.get("log"), "log"),
            ui=_build(UIConfig, data.get("ui"), "ui"),
            telemetry=_build(TelemetryConfig, data.get("telemetry"), "telemetry"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VigilConfig":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        return cls.from_dict(data)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_mapping(data: Any, section: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(
            f"section [{section}] must be a mapping, got {type(data).__name__}"
        )


def _reject_unknown(cls: type, data: dict[str, Any], section: str) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{section}]: {sorted(unknown)} "
            f"(allowed: {sorted(allowed)})"
        )


def _build(cls: type, data: Any, section: str):
    """Build a simple (non-nested) section dataclass from a mapping."""
    if data is None:
        return cls()
    _require_mapping(data, section)
    _reject_unknown(cls, data, section)
    return cls(**data)


def load_config(path: str | Path | None = None) -> VigilConfig:
    """Load config from `path`, else the first of DEFAULT_CONFIG_PATHS that
    exists, else all-defaults."""
    if path is not None:
        return VigilConfig.from_yaml(path)
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return VigilConfig.from_yaml(candidate)
    return VigilConfig.from_dict({})

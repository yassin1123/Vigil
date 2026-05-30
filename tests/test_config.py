"""Config model: defaults, YAML round-trip, and strict validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from vigil.config import ConfigError, VigilConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = REPO_ROOT / "vigil.example.yaml"


def test_empty_loads_to_defaults():
    cfg = VigilConfig.from_dict({})
    assert cfg.source.kind == "mock"
    assert cfg.camera.width == 1920
    assert cfg.ui.web_host == "127.0.0.1"  # localhost-only by default


def test_dict_roundtrip_is_identity():
    cfg = VigilConfig.from_dict({})
    assert VigilConfig.from_dict(cfg.to_dict()) == cfg


def test_example_yaml_loads():
    cfg = VigilConfig.from_yaml(EXAMPLE_YAML)
    assert cfg.source.kind in {"mock", "file", "csi"}
    assert cfg.model.input_size == 640
    # the committed example must itself survive a round-trip
    assert VigilConfig.from_dict(cfg.to_dict()) == cfg


def test_load_config_explicit_path():
    cfg = load_config(EXAMPLE_YAML)
    assert isinstance(cfg, VigilConfig)


def test_nested_override():
    cfg = VigilConfig.from_dict(
        {"camera": {"width": 1280, "height": 720}, "source": {"kind": "file"}}
    )
    assert (cfg.camera.width, cfg.camera.height) == (1280, 720)
    assert cfg.source.kind == "file"


def test_unknown_top_level_section_raises():
    with pytest.raises(ConfigError):
        VigilConfig.from_dict({"nonsense": {}})


def test_unknown_nested_key_raises():
    with pytest.raises(ConfigError):
        VigilConfig.from_dict({"camera": {"typo_here": 1}})


def test_bad_source_kind_raises():
    with pytest.raises(ConfigError):
        VigilConfig.from_dict({"source": {"kind": "satellite"}})


def test_non_mapping_section_raises():
    with pytest.raises(ConfigError):
        VigilConfig.from_dict({"camera": [1, 2, 3]})

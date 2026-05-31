"""Export-sink rails: routing/capability model + honest roadmap stubs."""
from __future__ import annotations

import pytest

from vigil.config import VigilConfig
from vigil.log.logger import HashChainLogger
from vigil.log.sinks import (
    ArmoredModuleSink,
    DispatchContext,
    ExportRouter,
    ExportTrigger,
    LocalDirSink,
    MeshStreamSink,
    RFBurstSink,
    build_export_router,
)

FIXED_UTC = "2026-05-31T00:00:00+00:00"


def _make_log(path, n=2):
    with HashChainLogger(path, utc_now=lambda: FIXED_UTC) as log:
        for i in range(n):
            log.append(f"EVENT_{i}", {"i": i})


def _names(results):
    return sorted(r.sink for r in results)


# --- routing dispatches to the sinks that declare the trigger -------------- #


def test_router_routes_by_declared_trigger(tmp_path):
    log = tmp_path / "events.jsonl"
    _make_log(log)
    dest = tmp_path / "dest"
    dest.mkdir()

    local = LocalDirSink(dest)
    mesh = MeshStreamSink()
    rf = RFBurstSink()
    armored = ArmoredModuleSink()
    router = ExportRouter([local, mesh, rf, armored])

    # DRIVE_INSERTED -> only the built local/USB-style sink
    drive = router.dispatch(DispatchContext(ExportTrigger.DRIVE_INSERTED, log_path=log))
    assert _names(drive) == ["localdir"]
    assert drive[0].ok and drive[0].implemented

    # SIGNAL_LOSS -> only the RF burst stub
    signal = router.dispatch(
        DispatchContext(ExportTrigger.SIGNAL_LOSS, recent_records=[{"i": 0}])
    )
    assert _names(signal) == ["rf-burst"]

    # PEER_AVAILABLE -> only the mesh stub
    peer = router.dispatch(DispatchContext(ExportTrigger.PEER_AVAILABLE, record={"i": 1}))
    assert _names(peer) == ["mesh-stream"]

    # EVENT_APPENDED -> both per-event streamers (mesh + armoured)
    appended = router.dispatch(
        DispatchContext(ExportTrigger.EVENT_APPENDED, record={"i": 1})
    )
    assert _names(appended) == ["armored-module", "mesh-stream"]


# --- stubs honestly report not-implemented (never silent success) ---------- #


def test_roadmap_stubs_declare_unimplemented():
    for sink in (MeshStreamSink(), RFBurstSink(), ArmoredModuleSink()):
        caps = sink.capabilities()
        assert caps.implemented is False
        assert sink.available() is None
        result = sink.handle(DispatchContext(ExportTrigger.EVENT_APPENDED, record={}))
        assert result.ok is False
        assert result.implemented is False
        assert "not implemented" in result.error
        assert result.implemented is not True  # never masquerades as success


def test_stub_transport_methods_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        MeshStreamSink().stream_event({"i": 0})
    with pytest.raises(NotImplementedError):
        RFBurstSink().burst([{"i": 0}])
    with pytest.raises(NotImplementedError):
        ArmoredModuleSink().write({"i": 0})


def test_built_sinks_declare_implemented(tmp_path):
    local = LocalDirSink(tmp_path)
    caps = local.capabilities()
    assert caps.implemented is True
    assert ExportTrigger.DRIVE_INSERTED in caps.triggers


# --- config wiring --------------------------------------------------------- #


def test_build_export_router_default_has_only_primary_sink():
    router = build_export_router(VigilConfig.from_dict({"log": {"export_sink": "null"}}))
    assert [s.name for s in router.sinks] == ["null"]


def test_build_export_router_includes_enabled_roadmap_stubs(tmp_path):
    cfg = VigilConfig.from_dict(
        {
            "log": {
                "export_sink": "localdir",
                "export_dir": str(tmp_path),
                "mesh": {"enabled": True, "peers": ["unit-2"]},
                "rf": {"enabled": True},
                "armored": {"enabled": True},
            }
        }
    )
    router = build_export_router(cfg)
    names = {s.name for s in router.sinks}
    assert names == {"localdir", "mesh-stream", "rf-burst", "armored-module"}
    # the enabled stubs still honestly report themselves unimplemented
    stub_caps = [c for c in router.capabilities() if not c.implemented]
    assert {c.name for c in stub_caps} == {"mesh-stream", "rf-burst", "armored-module"}


def test_log_config_roundtrips_with_roadmap_sections():
    cfg = VigilConfig.from_dict(
        {"log": {"mesh": {"enabled": True, "link": "serial"}}}
    )
    assert cfg.log.mesh.enabled is True
    assert cfg.log.mesh.link == "serial"
    assert VigilConfig.from_dict(cfg.to_dict()) == cfg


def test_log_config_rejects_unknown_roadmap_key():
    from vigil.config import ConfigError

    with pytest.raises(ConfigError):
        VigilConfig.from_dict({"log": {"mesh": {"nope": 1}}})


def test_export_event_types_exist():
    # the records the roadmap backends WOULD emit are defined (rails)
    from vigil.types import SystemEventType

    assert SystemEventType.MESH_PEER_ACK.value == "MESH_PEER_ACK"
    assert SystemEventType.RF_BURST_SENT.value == "RF_BURST_SENT"
    assert SystemEventType.ARMORED_WRITE.value == "ARMORED_WRITE"

"""Hash-chained logger: valid chain, durability, resume, tamper detection."""
from __future__ import annotations

import json

import pytest

from vigil.log.canonical import entry_hash, genesis_hash
from vigil.log.logger import HashChainLogger, LogError
from vigil.types import Detection, Event, EventType, SystemEvent, SystemEventType

FIXED_UTC = "2026-05-31T00:00:00+00:00"


def _read(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _verify_chain(entries) -> bool:
    """Independent re-derivation of the chain from genesis."""
    prev = genesis_hash()
    for i, e in enumerate(entries):
        if e["seq"] != i:
            return False
        content = {k: v for k, v in e.items() if k != "hash"}
        if entry_hash(content, prev) != e["hash"]:
            return False
        prev = e["hash"]
    return True


def _logger(path):
    return HashChainLogger(path, utc_now=lambda: FIXED_UTC)


def test_sequence_produces_a_valid_chain(tmp_path):
    p = tmp_path / "events.jsonl"
    with _logger(p) as log:
        log.log_system_event(
            SystemEvent(SystemEventType.ZONES_RELOADED, FIXED_UTC, 1.0, {"added": ["a"]})
        )
        log.log_event(
            Event(
                EventType.ZONE_ENTRY, track_id=5, zone_id="dock", class_name="person",
                timestamp_utc=FIXED_UTC, timestamp_monotonic=2.0,
                centroid=(10.0, 20.0), bbox=(0.0, 0.0, 4.0, 8.0), frame_index=3,
            )
        )
        log.log_detection(Detection((1.0, 2.0, 3.0, 4.0), 0, "person", 0.9))

    entries = _read(p)
    assert [e["seq"] for e in entries] == [0, 1, 2]
    assert [e["event_type"] for e in entries] == ["ZONES_RELOADED", "ZONE_ENTRY", "DETECTION"]
    assert _verify_chain(entries)


def test_first_entry_chains_from_genesis(tmp_path):
    p = tmp_path / "e.jsonl"
    with _logger(p) as log:
        entry = log.append("X", {"k": "v"})
    content = {k: v for k, v in entry.items() if k != "hash"}
    assert entry["hash"] == entry_hash(content, genesis_hash())


def test_entry_is_durable_before_append_returns(tmp_path):
    # Default fsync=True. After append() the line must be readable from a fresh
    # handle (flushed to disk) before the logger is closed.
    p = tmp_path / "e.jsonl"
    log = HashChainLogger(p, utc_now=lambda: FIXED_UTC)
    log.append("A", {"x": 1})
    assert len(_read(p)) == 1  # present on disk before close
    log.close()


def test_tampering_breaks_the_chain(tmp_path):
    p = tmp_path / "e.jsonl"
    with _logger(p) as log:
        log.append("A", {"x": 1})
        log.append("B", {"x": 2})
    entries = _read(p)
    assert _verify_chain(entries)

    entries[0]["payload"]["x"] = 999  # alter the first entry's payload
    assert not _verify_chain(entries)  # its stored hash no longer matches


def test_resume_continues_seq_and_chain(tmp_path):
    p = tmp_path / "e.jsonl"
    with _logger(p) as log:
        log.append("A", {"x": 1})
        log.append("B", {"x": 2})
        head = log.head_hash
    with _logger(p) as log2:
        assert log2.count == 2
        assert log2.head_hash == head
        third = log2.append("C", {"x": 3})
    assert third["seq"] == 2
    entries = _read(p)
    assert len(entries) == 3 and _verify_chain(entries)


def test_partial_trailing_line_truncated_on_resume(tmp_path):
    p = tmp_path / "e.jsonl"
    with _logger(p) as log:
        log.append("A", {"x": 1})
    # simulate a power cut mid-write: a partial line with no terminating newline
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"seq":1,"event_type":"B","partial"')
    with _logger(p) as log2:  # must truncate the partial line, not crash
        log2.append("B", {"x": 2})
    entries = _read(p)
    assert [e["seq"] for e in entries] == [0, 1]  # partial dropped; B is seq 1
    assert _verify_chain(entries)


def test_corrupt_last_complete_entry_refuses_resume(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"not":"an entry"}\n', encoding="utf-8")  # complete line, no seq/hash
    with pytest.raises(LogError):
        HashChainLogger(p, utc_now=lambda: FIXED_UTC)


def test_unicode_and_floats_roundtrip_through_the_chain(tmp_path):
    p = tmp_path / "e.jsonl"
    with _logger(p) as log:
        log.append("U", {"name": "café→", "score": 1 / 3, "box": [0.1 + 0.2, 2.0]})
    entries = _read(p)
    assert entries[0]["payload"]["name"] == "café→"
    assert _verify_chain(entries)  # re-canonicalising the stored values reproduces the hash

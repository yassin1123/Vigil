"""Tamper-detection matrix — the credibility of the evidentiary log.

A clean log verifies; every tamper type (edit, delete, reorder, partial re-hash)
fails at exactly the right entry index.
"""
from __future__ import annotations

import copy
import json

from vigil.__main__ import main
from vigil.log.canonical import entry_hash, genesis_hash
from vigil.log.logger import HashChainLogger
from vigil.log.verify import verify_entries, verify_file

FIXED_UTC = "2026-05-31T00:00:00+00:00"


def _write_log(path, n=5):
    """Write n entries and return the list of entry dicts as stored."""
    with HashChainLogger(path, utc_now=lambda: FIXED_UTC) as log:
        for i in range(n):
            log.append(f"EVENT_{i}", {"i": i, "note": f"payload-{i}"})
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _rewrite(path, entries):
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n" for e in entries),
        encoding="utf-8",
    )


# --- clean log passes ------------------------------------------------------ #


def test_clean_log_verifies(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 5)
    result = verify_file(p)
    assert result.ok
    assert result.entry_count == 5
    assert result.genesis_hash == genesis_hash()
    assert result.terminal_hash == entries[-1]["hash"]
    assert result.error_index is None


def test_empty_log_verifies(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("", encoding="utf-8")
    result = verify_file(p)
    assert result.ok and result.entry_count == 0
    assert result.terminal_hash == genesis_hash()


# --- (a) edit a payload field ---------------------------------------------- #


def test_edit_payload_fails_at_that_entry(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 5)
    entries[2]["payload"]["note"] = "tampered"
    _rewrite(p, entries)
    result = verify_file(p)
    assert not result.ok
    assert result.error_index == 2
    assert "hash mismatch" in result.error


# --- (b) delete an entry --------------------------------------------------- #


def test_delete_entry_fails_at_the_gap(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 5)
    del entries[2]  # remove the middle entry
    _rewrite(p, entries)
    result = verify_file(p)
    assert not result.ok
    # the entry now occupying index 2 chained off the deleted entry's hash
    assert result.error_index == 2


# --- (c) reorder two entries ----------------------------------------------- #


def test_reorder_entries_fails_at_first_swapped(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 5)
    entries[1], entries[2] = entries[2], entries[1]
    _rewrite(p, entries)
    result = verify_file(p)
    assert not result.ok
    assert result.error_index == 1


# --- (d) partial re-hash: fix the tampered entry but not its successors ----- #


def test_partial_rehash_fails_at_successor(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 5)
    k = 2
    # tamper entry k AND recompute its hash correctly from the real prior hash...
    prev = entries[k - 1]["hash"] if k > 0 else genesis_hash()
    entries[k]["payload"]["note"] = "tampered-but-rehashed"
    content = {kk: vv for kk, vv in entries[k].items() if kk != "hash"}
    entries[k]["hash"] = entry_hash(content, prev)
    # ...but leave entry k+1's stored hash (which chained off the OLD hash_k).
    _rewrite(p, entries)
    result = verify_file(p)
    assert not result.ok
    assert result.error_index == k + 1  # the successor is where it breaks


# --- renumbering seq does not save a deletion ------------------------------ #


def test_delete_then_renumber_still_caught(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 5)
    del entries[2]
    for i, e in enumerate(entries):  # attacker fixes seq to look contiguous
        e["seq"] = i
    _rewrite(p, entries)
    result = verify_file(p)
    assert not result.ok
    assert result.error_index == 2


# --- structural / parse failures ------------------------------------------- #


def test_unparseable_line_reported(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("not valid json\n", encoding="utf-8")
    result = verify_file(p)
    assert not result.ok and result.error_index == 0
    assert "unparseable" in result.error


# --- timestamp monotonicity is flagged, not failed ------------------------- #


def test_backwards_timestamp_is_a_warning_not_a_failure(tmp_path):
    p = tmp_path / "e.jsonl"
    stamps = iter(["2026-05-31T00:00:02+00:00", "2026-05-31T00:00:01+00:00"])
    with HashChainLogger(p, utc_now=lambda: next(stamps)) as log:
        log.append("A", {"i": 0})
        log.append("B", {"i": 1})
    result = verify_file(p)
    assert result.ok  # the chain is intact...
    assert any("stepped backwards" in w for w in result.warnings)  # ...but flagged


# --- in-memory verify mirrors file verify ---------------------------------- #


def test_verify_entries_matches_file(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 3)
    assert verify_entries(copy.deepcopy(entries)).ok
    entries[0]["seq"] = 99  # changes content -> breaks entry 0's own hash
    assert verify_entries(entries).error_index == 0


# --- CLI ------------------------------------------------------------------- #


def test_cli_verify_pass_and_fail(tmp_path):
    p = tmp_path / "e.jsonl"
    entries = _write_log(p, 4)
    assert main(["log", "verify", str(p)]) == 0
    entries[1]["payload"]["i"] = 999
    _rewrite(p, entries)
    assert main(["log", "verify", str(p)]) == 1

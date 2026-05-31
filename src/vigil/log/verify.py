"""Hash-chain verifier — proves a log is intact, and pinpoints the first break.

`verify_file(path)` walks the log from the genesis hash, recomputing each entry's
hash from its content + the prior hash using the SAME `vigil.log.canonical` rules
the writer used. The first entry whose recomputed hash differs from its stored
hash is reported precisely (index + what mismatched). It also checks
sequence-number contiguity (folded into the failure detail) and flags
timestamp_utc going backwards as a warning — clocks can legitimately step.

Detectable tampering: editing any field (the entry's own hash breaks),
inserting/deleting/reordering entries (the next entry's chained prev breaks), and
re-hashing a tampered entry without re-hashing its successors (the successor
breaks). Known limit of a bare hash chain: truncating the TAIL leaves a valid
prefix — detecting that needs an external anchor (a signed terminal hash), which
is out of scope here and noted for the airgap report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from vigil.log.canonical import entry_hash, genesis_hash


@dataclass
class VerificationResult:
    ok: bool
    entry_count: int
    genesis_hash: str
    terminal_hash: Optional[str]
    error_index: Optional[int] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def verify_entries(entries: list[dict[str, Any]]) -> VerificationResult:
    """Verify an in-memory list of entry dicts against the hash chain."""
    genesis = genesis_hash()
    prev = genesis
    warnings: list[str] = []
    last_dt: Optional[datetime] = None

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "hash" not in entry or "seq" not in entry:
            return VerificationResult(
                ok=False, entry_count=len(entries), genesis_hash=genesis,
                terminal_hash=prev, error_index=i,
                error="entry is missing required fields (seq/hash)",
            )

        content = {k: v for k, v in entry.items() if k != "hash"}
        try:
            recomputed = entry_hash(content, prev)
        except ValueError as exc:
            return VerificationResult(
                ok=False, entry_count=len(entries), genesis_hash=genesis,
                terminal_hash=prev, error_index=i,
                error=f"entry content cannot be canonicalised: {exc}",
            )

        if recomputed != entry["hash"]:
            detail = (
                f"hash mismatch: recomputed {recomputed}, stored {entry['hash']}"
            )
            if entry.get("seq") != i:
                detail += f"; sequence number {entry.get('seq')} != position {i}"
            return VerificationResult(
                ok=False, entry_count=len(entries), genesis_hash=genesis,
                terminal_hash=prev, error_index=i, error=detail, warnings=warnings,
            )

        # Chain link verified. Supplementary, non-fatal checks.
        if entry.get("seq") != i:
            warnings.append(
                f"entry {i}: sequence number {entry.get('seq')} != position {i}"
            )
        cur_dt = _ts(entry.get("timestamp_utc"))
        if last_dt is not None and cur_dt is not None and cur_dt < last_dt:
            warnings.append(
                f"entry {i}: timestamp_utc {entry.get('timestamp_utc')} is earlier "
                f"than the previous entry (clock stepped backwards)"
            )
        if cur_dt is not None:
            last_dt = cur_dt
        prev = entry["hash"]

    return VerificationResult(
        ok=True, entry_count=len(entries), genesis_hash=genesis,
        terminal_hash=prev, warnings=warnings,
    )


def verify_file(path: str | Path) -> VerificationResult:
    """Read a JSONL log and verify its hash chain end to end."""
    path = Path(path)
    genesis = genesis_hash()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return VerificationResult(
            ok=False, entry_count=0, genesis_hash=genesis, terminal_hash=None,
            error=f"cannot read log {path}: {exc}",
        )

    entries: list[dict[str, Any]] = []
    for i, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return VerificationResult(
                ok=False, entry_count=i, genesis_hash=genesis,
                terminal_hash=(entries[-1]["hash"] if entries else genesis),
                error_index=i, error=f"unparseable JSON on line {i}: {exc}",
            )
    return verify_entries(entries)

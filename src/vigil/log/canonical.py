"""Canonical serialization + hash-chain primitives for the evidentiary log.

The integrity of the whole log rests on this file: writing and verifying MUST
produce byte-identical canonical forms, on any machine, across runs. So the
rules here are fixed and deliberately boring.

Canonical form of a JSON-able value (`canonical_json` / `canonical_bytes`):
  * object keys are sorted (`sort_keys=True`);
  * no insignificant whitespace (separators ``(",", ":")``);
  * floats are rounded to FLOAT_PRECISION (6) decimal places, and ``-0.0`` is
    normalised to ``0.0`` — so platform/round-trip float jitter cannot change a
    hash. Integers and booleans are left exactly as-is;
  * non-finite floats (NaN, +/-inf) are rejected — they have no place in an
    evidentiary record;
  * text is emitted as UTF-8 (``ensure_ascii=False``) and encoded UTF-8, so
    unicode is handled identically everywhere.

Hash chain:
  * the genesis hash is ``SHA-256("VIGIL_GENESIS")``;
  * an entry's hash is ``SHA-256(canonical_bytes(content) + b"\\n" + prev_hash)``
    where ``content`` is the entry without its ``hash`` field and ``prev_hash``
    is the previous entry's hash (hex ascii). The ``b"\\n"`` separator keeps the
    boundary between content and the chained hash unambiguous.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

FLOAT_PRECISION = 6
GENESIS_SEED = "VIGIL_GENESIS"


def _normalize(value: Any) -> Any:
    """Round floats and reject non-finite values, recursively."""
    if isinstance(value, bool):  # bool is an int subclass — keep as-is
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float is not allowed in the log")
        return round(value, FLOAT_PRECISION) + 0.0  # +0.0 normalises -0.0 -> 0.0
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON string (see module docstring for the rules)."""
    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON encoded as UTF-8 — the exact bytes that get hashed."""
    return canonical_json(obj).encode("utf-8")


def genesis_hash() -> str:
    """The chain root: SHA-256 of the literal string 'VIGIL_GENESIS'."""
    return hashlib.sha256(GENESIS_SEED.encode("utf-8")).hexdigest()


def entry_hash(content: dict[str, Any], prev_hash: str) -> str:
    """SHA-256 of the canonical content chained to the previous entry's hash."""
    material = canonical_bytes(content) + b"\n" + prev_hash.encode("ascii")
    return hashlib.sha256(material).hexdigest()

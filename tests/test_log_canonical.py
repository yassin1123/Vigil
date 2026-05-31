"""Canonical serialization: deterministic, sorted, float/unicode-stable."""
from __future__ import annotations

import hashlib

import pytest

from vigil.log.canonical import (
    canonical_bytes,
    canonical_json,
    entry_hash,
    genesis_hash,
)


def test_keys_sorted_and_whitespace_free():
    out = canonical_json({"b": 2, "a": 1.0, "z": "x"})
    assert out == '{"a":1.0,"b":2,"z":"x"}'


def test_key_order_does_not_matter():
    assert canonical_json({"z": "x", "a": 1.0, "b": 2}) == canonical_json(
        {"a": 1.0, "b": 2, "z": "x"}
    )


def test_floats_rounded_deterministically():
    assert canonical_json({"v": 0.1 + 0.2}) == '{"v":0.3}'  # not 0.30000000000000004
    assert canonical_json({"v": 1 / 3}) == '{"v":0.333333}'
    assert canonical_json({"v": -0.0}) == '{"v":0.0}'  # -0.0 normalised


def test_integers_and_bools_unchanged():
    assert canonical_json({"i": 2, "t": True, "f": False}) == '{"f":false,"i":2,"t":true}'


def test_unicode_is_utf8():
    assert canonical_bytes({"n": "café→"}) == '{"n":"café→"}'.encode("utf-8")


def test_non_finite_floats_rejected():
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_canonical_is_stable_across_calls():
    obj = {"payload": {"bbox": [1.123456789, 2.0], "name": "café"}, "seq": 3}
    assert canonical_bytes(obj) == canonical_bytes(obj)


def test_genesis_is_sha256_of_literal_seed():
    assert genesis_hash() == hashlib.sha256(b"VIGIL_GENESIS").hexdigest()


def test_entry_hash_construction_is_explicit():
    content = {"seq": 0, "event_type": "X"}
    prev = genesis_hash()
    expected = hashlib.sha256(
        canonical_bytes(content) + b"\n" + prev.encode("ascii")
    ).hexdigest()
    assert entry_hash(content, prev) == expected

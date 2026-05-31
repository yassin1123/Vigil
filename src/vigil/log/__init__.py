"""Vigil tamper-evident logging: SHA-256 hash-chained JSONL event log."""
from __future__ import annotations

from vigil.log.canonical import (
    GENESIS_SEED,
    canonical_bytes,
    canonical_json,
    entry_hash,
    genesis_hash,
)
from vigil.log.logger import HashChainLogger, LogError, build_logger

__all__ = [
    "GENESIS_SEED",
    "HashChainLogger",
    "LogError",
    "build_logger",
    "canonical_bytes",
    "canonical_json",
    "entry_hash",
    "genesis_hash",
]

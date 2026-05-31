"""Append-only, SHA-256 hash-chained event log (JSONL).

Each line is one entry:
    {"seq", "timestamp_utc", "event_type", "payload", "hash"}
where `hash` chains the entry's content to the previous entry via
`vigil.log.canonical` (see that module for the exact, shared rules). The first
entry chains from the genesis hash, SHA-256("VIGIL_GENESIS"). Any later edit to
any field of any entry changes its canonical content, so its hash — and every
hash after it — no longer matches: tampering is visible end to end.

Durability guarantee:
  Every entry is written, flushed, and fsync'd to disk before `append()`
  returns. Vigil never holds events only in memory. An abrupt power loss can
  therefore lose at most the single event currently mid-write; on the next open
  a partial trailing line (one without a terminating newline) is truncated, and
  the log resumes cleanly from the last complete, fsync'd entry.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from vigil.log.canonical import entry_hash, genesis_hash
from vigil.types import Detection, Event, SystemEvent

if TYPE_CHECKING:
    from vigil.config import VigilConfig

DETECTION_EVENT_TYPE = "DETECTION"


class LogError(Exception):
    """Raised when an existing log cannot be safely resumed."""


def _default_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class HashChainLogger:
    """Writes a tamper-evident, fsync'd, hash-chained JSONL event log."""

    def __init__(
        self,
        path: str | Path,
        *,
        fsync: bool = True,
        utc_now: Optional[Callable[[], str]] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._utc_now = utc_now or _default_utc
        self._seq = 0
        self._prev_hash = genesis_hash()
        self._resume()
        self._file = open(self.path, "a", encoding="utf-8")

    # -- resume / durability ---------------------------------------------
    def _resume(self) -> None:
        """Continue an existing log: truncate any partial trailing line, then
        adopt the last complete entry's seq+hash as the chain head."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        raw = self.path.read_bytes()
        if not raw.endswith(b"\n"):  # partial in-flight line from a crash
            cut = raw.rfind(b"\n")
            raw = raw[: cut + 1] if cut != -1 else b""
            self.path.write_bytes(raw)
        lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        if not lines:
            return
        try:
            last = json.loads(lines[-1])
            self._seq = int(last["seq"]) + 1
            self._prev_hash = str(last["hash"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise LogError(
                f"cannot resume {self.path}: last entry is unreadable ({exc})"
            ) from exc

    # -- writing ----------------------------------------------------------
    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one entry; flush+fsync before returning. Returns the entry."""
        content = {
            "seq": self._seq,
            "timestamp_utc": self._utc_now(),
            "event_type": str(event_type),
            "payload": payload,
        }
        digest = entry_hash(content, self._prev_hash)
        entry = {**content, "hash": digest}

        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._file.write(line)
        self._file.flush()
        if self._fsync:
            os.fsync(self._file.fileno())

        self._seq += 1
        self._prev_hash = digest
        return entry

    def log_event(self, event: Event) -> dict[str, Any]:
        return self.append(event.event_type.value, event.to_dict())

    def log_system_event(self, event: SystemEvent) -> dict[str, Any]:
        return self.append(event.event_type.value, event.to_dict())

    def log_detection(
        self, detection: Detection, extra: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        payload = detection.to_dict()
        if extra:
            payload = {**payload, **extra}
        return self.append(DETECTION_EVENT_TYPE, payload)

    # -- state / lifecycle ------------------------------------------------
    @property
    def count(self) -> int:
        """Number of entries written so far (== seq of the next entry)."""
        return self._seq

    @property
    def head_hash(self) -> str:
        """Hash of the most recent entry (or the genesis hash if empty)."""
        return self._prev_hash

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            if self._fsync:
                os.fsync(self._file.fileno())
            self._file.close()
            self._file = None  # type: ignore[assignment]

    def __enter__(self) -> "HashChainLogger":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


def build_logger(config: "VigilConfig") -> HashChainLogger:
    """Create a HashChainLogger at the configured log path."""
    return HashChainLogger(config.log.path)

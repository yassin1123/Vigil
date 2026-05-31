"""Zone entry/exit event engine.

Turns per-frame zone membership into clean ZONE_ENTRY / ZONE_EXIT events per
tracked object, with debounce so an object jittering on a boundary doesn't spam
events.

Debounce model (per (track, INCLUDE-zone) pair):
  Each pair has a *reported* state (inside or outside) and a streak counter of
  consecutive frames the RAW membership has DISAGREED with the reported state.
  When the streak reaches the threshold the reported state flips and an event
  fires; any frame that agrees with the reported state resets the streak to 0.
  Entry needs `enter_frames` consecutive inside frames; exit needs `exit_frames`
  consecutive outside frames (symmetric hysteresis by default). A boundary
  flicker (in,out,in,out,...) never accumulates a streak, so it fires nothing.

  Raw membership for a (track, zone) pair is True iff the track's centroid is
  inside the zone, the zone's class filter accepts the track's class, AND the
  track is not inside any applicable EXCLUDE zone. A track absent this frame
  (e.g. occluded) is treated as outside, so it debounces to EXIT rather than
  firing instantly — and reappearing within the window cancels the pending exit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional

from vigil.types import Event, EventType, Track
from vigil.zones.geometry import ZoneIndex
from vigil.zones.model import ZoneKind, ZoneSet

if TYPE_CHECKING:
    from vigil.config import VigilConfig


def _default_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _PairState:
    reported_inside: bool = False
    streak: int = 0


class ZoneEventEngine:
    """Maintains per-(track, zone) membership and emits debounced zone events."""

    def __init__(
        self,
        index: ZoneIndex,
        enter_frames: int = 3,
        exit_frames: int = 3,
        utc_now: Optional[Callable[[], str]] = None,
    ) -> None:
        self.index = index
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self._utc_now = utc_now or _default_utc
        self._include_zones = [z for z in index.zone_set if z.kind == ZoneKind.INCLUDE]
        self._state: dict[tuple[int, str], _PairState] = {}
        self._last_track: dict[int, Track] = {}

    def reset(self) -> None:
        self._state.clear()
        self._last_track.clear()

    def update(
        self, tracks: list[Track], frame_info: object | None = None
    ) -> list[Event]:
        """Process one frame of tracks; return the events fired this frame."""
        ts_mono = getattr(frame_info, "timestamp", None)
        if ts_mono is None:
            ts_mono = time.monotonic()
        frame_index = getattr(frame_info, "index", None)
        utc = self._utc_now()

        by_id = {t.track_id: t for t in tracks}
        for track in tracks:
            self._last_track[track.track_id] = track

        # Raw membership for every present track against every INCLUDE zone.
        raw: dict[tuple[int, str], bool] = {}
        for track in tracks:
            excluded = self.index.is_excluded(track.centroid, track.class_name)
            for zone in self._include_zones:
                inside = (
                    not excluded
                    and zone.accepts(track.class_name)
                    and self.index.contains_point(zone.id, track.centroid)
                )
                raw[(track.track_id, zone.id)] = inside

        # Process existing states plus any newly-inside pairs. Pairs that are
        # outside with no history need no state.
        keys = set(self._state)
        keys.update(key for key, inside in raw.items() if inside)

        events: list[Event] = []
        for key in keys:
            inside = raw.get(key, False)  # absent/not-present -> treated as outside
            state = self._state.get(key, _PairState())
            threshold = self.enter_frames if not state.reported_inside else self.exit_frames

            if inside != state.reported_inside:
                state.streak += 1
                if state.streak >= threshold:
                    state.reported_inside = inside
                    state.streak = 0
                    track = by_id.get(key[0]) or self._last_track.get(key[0])
                    if track is not None:
                        events.append(
                            self._make_event(track, key[1], inside, ts_mono, utc, frame_index)
                        )
            else:
                state.streak = 0

            if state.reported_inside or state.streak > 0:
                self._state[key] = state
            else:
                self._state.pop(key, None)

        self._prune_last_tracks(by_id)
        return events

    def _make_event(
        self,
        track: Track,
        zone_id: str,
        entering: bool,
        ts_mono: float,
        utc: str,
        frame_index: Optional[int],
    ) -> Event:
        return Event(
            event_type=EventType.ZONE_ENTRY if entering else EventType.ZONE_EXIT,
            track_id=track.track_id,
            zone_id=zone_id,
            class_name=track.class_name,
            timestamp_utc=utc,
            timestamp_monotonic=float(ts_mono),
            centroid=track.centroid,
            bbox=track.bbox,
            frame_index=frame_index,
        )

    def _prune_last_tracks(self, by_id: dict[int, Track]) -> None:
        active_ids = {tid for (tid, _) in self._state}
        for tid in list(self._last_track):
            if tid not in by_id and tid not in active_ids:
                del self._last_track[tid]


def build_zone_engine(
    config: "VigilConfig",
    zone_set: ZoneSet,
    frame_size: tuple[int, int],
    utc_now: Optional[Callable[[], str]] = None,
) -> ZoneEventEngine:
    """Build a ZoneEventEngine from config (symmetric debounce) + a zone set."""
    index = ZoneIndex(zone_set, frame_size)
    k = config.zones.debounce_frames
    return ZoneEventEngine(index, enter_frames=k, exit_frames=k, utc_now=utc_now)

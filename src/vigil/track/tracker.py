"""Multi-object tracking with an explicit lifecycle.

  * Tracker          — the contract: update(detections, frame_info) -> list[Track].
  * ByteTrackTracker — ByteTrack association + a clean track lifecycle.

ByteTrack is chosen deliberately for Vigil's offline thesis: it associates by
**motion only** (a Kalman filter + IoU), with NO re-identification network. There
is no model to download and no embedding to compute, so the tracker keeps working
on an airgapped device. Association uses greedy IoU matching (not scipy/lap), so
it stays pure-numpy and runs with no GPU.

Lifecycle (thresholds configurable via vigil.yaml):
  TENTATIVE  newly seen; promoted once matched `confirm_frames` consecutive
             frames. A tentative track that misses a frame is removed at once.
  CONFIRMED  a real, reported object. Emitted in update()'s output.
  LOST       a confirmed track that missed its detection; coasted on the Kalman
             prediction and kept for re-association up to `lost_window` frames.
  REMOVED    terminal: a tentative miss, or a lost track past its window.
A LOST track that matches a detection again returns to CONFIRMED with the SAME
id (occlusion recovery). Past the window it is removed, and a fresh detection
earns a new id.
"""
from __future__ import annotations

import abc
from collections import deque
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import numpy as np

from vigil.detect.coco import COCO_CLASSES
from vigil.types import Detection, Track

if TYPE_CHECKING:
    from vigil.config import VigilConfig
    from vigil.frames import Frame


class Tracker(abc.ABC):
    """Turns per-frame detections into persistent Tracks with stable ids."""

    @abc.abstractmethod
    def update(
        self, detections: list[Detection], frame_info: "Frame | None" = None
    ) -> list[Track]:
        """Associate `detections` with existing tracks; return active tracks."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all tracks and id state."""


# --------------------------------------------------------------------------- #
# Kalman filter (constant-velocity, on [cx, cy, aspect, height]) — SORT/ByteTrack
# --------------------------------------------------------------------------- #


class KalmanFilter:
    """8-state (position + velocity) constant-velocity filter on xyah boxes."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return mean, np.diag(np.square(std))

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = (covariance @ self._update_mat.T) @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov


# --------------------------------------------------------------------------- #
# Single-object track state + lifecycle
# --------------------------------------------------------------------------- #


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """Internal track: Kalman state + lifecycle state machine."""

    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int, history_len: int = 30):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.score = float(score)
        self.class_id = int(class_id)
        self.track_id = 0
        self.state = TrackState.TENTATIVE
        self.hits = 0
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        self._history: deque[tuple[float, float]] = deque(maxlen=history_len)

    # -- id allocation ----------------------------------------------------
    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_count() -> None:
        STrack._count = 0

    # -- box conversions --------------------------------------------------
    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyxy_to_tlwh(xyxy: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # aspect * height -> width
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def centroid(self) -> tuple[float, float]:
        t = self.tlwh
        return float(t[0] + t[2] / 2), float(t[1] + t[3] / 2)

    @property
    def age(self) -> int:
        return self.frame_id - self.start_frame + 1

    def history(self) -> list[tuple[float, float]]:
        return list(self._history)

    # -- lifecycle transitions -------------------------------------------
    def activate(self, kalman_filter: KalmanFilter, frame_id: int, confirm_frames: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.hits = 1
        self.state = (
            TrackState.CONFIRMED if self.hits >= confirm_frames else TrackState.TENTATIVE
        )
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self._history.append(self.centroid)

    def update(self, new_track: "STrack", frame_id: int, confirm_frames: int) -> None:
        self.frame_id = frame_id
        self.hits += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.time_since_update = 0
        if self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED  # re-association keeps the id
        elif self.state == TrackState.TENTATIVE and self.hits >= confirm_frames:
            self.state = TrackState.CONFIRMED
        self._history.append(self.centroid)

    def predict(self) -> None:
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.CONFIRMED:
            mean_state[7] = 0  # do not drift height when not actively confirmed
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.time_since_update += 1

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


# --------------------------------------------------------------------------- #
# Matching helpers (pure numpy, no scipy)
# --------------------------------------------------------------------------- #


def _ious(atlbrs: list[np.ndarray], btlbrs: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if not atlbrs or not btlbrs:
        return out
    a = np.asarray(atlbrs, dtype=np.float64)
    b = np.asarray(btlbrs, dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(len(atlbrs)):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = area_a[i] + area_b - inter
        out[i] = np.where(union > 0, inter / union, 0.0)
    return out


def _iou_distance(tracks: list[STrack], dets: list[STrack]) -> np.ndarray:
    return 1.0 - _ious([t.tlbr for t in tracks], [d.tlbr for d in dets])


def _greedy_match(
    cost: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy assignment: take the lowest-cost free pair while cost <= thresh."""
    n_tracks, n_dets = cost.shape
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))
    pairs = [
        (cost[t, d], t, d)
        for t in range(n_tracks)
        for d in range(n_dets)
        if cost[t, d] <= thresh
    ]
    pairs.sort(key=lambda p: p[0])
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, t, d in pairs:
        if t in used_t or d in used_d:
            continue
        used_t.add(t)
        used_d.add(d)
        matches.append((t, d))
    u_track = [t for t in range(n_tracks) if t not in used_t]
    u_det = [d for d in range(n_dets) if d not in used_d]
    return matches, u_track, u_det


# --------------------------------------------------------------------------- #
# ByteTrack
# --------------------------------------------------------------------------- #


class ByteTrackTracker(Tracker):
    """ByteTrack two-stage association with an explicit track lifecycle."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        confirm_frames: int = 3,
        lost_window: int = 30,
        min_box_area: float = 10.0,
        history_len: int = 30,
        class_names: Sequence[str] = COCO_CLASSES,
    ) -> None:
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.confirm_frames = max(1, int(confirm_frames))
        self.lost_window = int(lost_window)
        self.min_box_area = min_box_area
        self.history_len = history_len
        self.class_names = class_names
        self.kalman_filter = KalmanFilter()
        self.tracks: list[STrack] = []  # TENTATIVE / CONFIRMED / LOST
        self.frame_id = 0
        STrack.reset_count()

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_id = 0
        STrack.reset_count()

    def update(
        self, detections: list[Detection], frame_info: "Frame | None" = None
    ) -> list[Track]:
        self.frame_id += 1
        cf = self.confirm_frames

        # Build detection STracks, dropping tiny boxes; split by confidence.
        dets: list[STrack] = []
        for det in detections:
            tlwh = STrack.xyxy_to_tlwh(det.bbox)
            if tlwh[2] * tlwh[3] < self.min_box_area:
                continue
            dets.append(STrack(tlwh, det.confidence, det.class_id, self.history_len))
        high = [s for s in dets if s.score >= self.track_thresh]
        low = [s for s in dets if self.low_thresh <= s.score < self.track_thresh]

        # Predict every live track to the current frame.
        for track in self.tracks:
            track.predict()

        confirmed = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        lost = [t for t in self.tracks if t.state == TrackState.LOST]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]
        matched: set[int] = set()

        # Stage 1: high-confidence detections vs confirmed + lost tracks.
        pool = confirmed + lost
        m1, u_t1, u_d1 = _greedy_match(_iou_distance(pool, high), self.match_thresh)
        for it, idet in m1:
            pool[it].update(high[idet], self.frame_id, cf)
            matched.add(pool[it].track_id)

        # Stage 2: low-confidence detections vs still-unmatched confirmed tracks.
        rem_confirmed = [
            pool[i] for i in u_t1 if pool[i].state == TrackState.CONFIRMED
        ]
        m2, _, _ = _greedy_match(_iou_distance(rem_confirmed, low), 0.5)
        for it, idet in m2:
            rem_confirmed[it].update(low[idet], self.frame_id, cf)
            matched.add(rem_confirmed[it].track_id)

        # Stage 3: tentative tracks vs remaining high-confidence detections.
        rem_high = [high[i] for i in u_d1]
        m3, u_t3, u_d3 = _greedy_match(_iou_distance(tentative, rem_high), 0.7)
        for it, idet in m3:
            tentative[it].update(rem_high[idet], self.frame_id, cf)
            matched.add(tentative[it].track_id)

        # Unmatched lifecycle transitions.
        for t in tentative:
            if t.track_id not in matched:
                t.mark_removed()  # a tentative miss dies immediately
        for t in pool:  # confirmed + lost
            if t.track_id not in matched and t.state == TrackState.CONFIRMED:
                t.mark_lost()

        # Spawn new tentative tracks from leftover high detections.
        for i in u_d3:
            det = rem_high[i]
            if det.score < self.track_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id, cf)
            self.tracks.append(det)

        # Expire lost tracks past their window.
        for t in self.tracks:
            if t.state == TrackState.LOST and t.time_since_update > self.lost_window:
                t.mark_removed()

        # Drop removed tracks.
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        return [
            self._to_track(t)
            for t in self.tracks
            if t.state == TrackState.CONFIRMED and t.time_since_update == 0
        ]

    def _to_track(self, s: STrack) -> Track:
        tlbr = s.tlbr
        name = (
            self.class_names[s.class_id]
            if 0 <= s.class_id < len(self.class_names)
            else str(s.class_id)
        )
        return Track(
            track_id=s.track_id,
            class_id=s.class_id,
            class_name=name,
            bbox=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
            confidence=s.score,
            age=s.age,
            time_since_update=s.time_since_update,
            history=s.history(),
        )


def build_tracker(config: "VigilConfig") -> ByteTrackTracker:
    """Create a ByteTrackTracker from the [tracker] config section."""
    t = config.tracker
    return ByteTrackTracker(
        track_thresh=t.track_thresh,
        low_thresh=t.low_thresh,
        match_thresh=t.match_thresh,
        confirm_frames=t.confirm_frames,
        lost_window=t.lost_window,
        min_box_area=t.min_box_area,
    )

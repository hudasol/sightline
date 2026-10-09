"""Track state: what a track knows, and the explicit lost/expired policy."""

from __future__ import annotations

from collections import deque

import numpy as np

from sightline.geometry import box_ref_point
from sightline.types import Box, TrackState, TrackView


class Track:
    """One anonymous identity. The id is an integer and is never reused."""

    def __init__(self, track_id: int, box: Box, score: float, frame_index: int, timestamp: float,
                 trajectory_len: int = 30) -> None:
        self.id = track_id
        self.box: Box = box  # best current estimate (Kalman-predicted when unmatched)
        self.last_det_box: Box = box
        self.score = score  # smoothed detection confidence
        self.state = TrackState.TENTATIVE
        self.age = 1  # processed frames since creation
        self.hits = 1
        self.time_since_update = 0
        self.last_seen_ts = timestamp
        self.last_seen_frame = frame_index
        self.trajectory: deque[tuple[float, float, int]] = deque(maxlen=trajectory_len)
        x, y = box_ref_point(box)
        self.trajectory.append((x, y, frame_index))
        # Kalman state, owned by the tracker that uses one
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    def mark_matched(self, box: Box, score: float, frame_index: int, timestamp: float, min_hits: int) -> None:
        self.last_det_box = box
        self.score = 0.7 * self.score + 0.3 * score
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_ts = timestamp
        self.last_seen_frame = frame_index
        x, y = box_ref_point(box)
        self.trajectory.append((x, y, frame_index))
        if self.state in (TrackState.TENTATIVE, TrackState.LOST) and self.hits >= min_hits:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self, max_age: int) -> None:
        """No detection this frame. Explicit policy:

        confirmed -> lost on the first miss; lost -> expired once the number of
        consecutive misses exceeds max_age. A tentative track that misses is
        dropped immediately by the tracker (it never earned an identity).
        """
        self.time_since_update += 1
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
        if self.time_since_update > max_age:
            self.state = TrackState.EXPIRED

    @property
    def observed(self) -> bool:
        """True if a detection was matched to this track on the current frame."""
        return self.time_since_update == 0

    def view(self) -> TrackView:
        return TrackView(
            id=self.id, box=self.box, score=self.score, state=self.state, age=self.age,
            hits=self.hits, time_since_update=self.time_since_update,
            last_seen_ts=self.last_seen_ts, trajectory=list(self.trajectory),
        )

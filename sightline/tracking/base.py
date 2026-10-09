"""Shared bookkeeping for both trackers."""

from __future__ import annotations

from dataclasses import dataclass, field

from sightline.config import TrackerConfig
from sightline.tracking.state import Track
from sightline.types import Detection, TrackState


@dataclass
class TrackerOutput:
    # every live track: tentative, confirmed or lost (not expired)
    tracks: list[Track] = field(default_factory=list)
    # confirmed tracks that expired on this frame (their identity is gone for good)
    expired: list[Track] = field(default_factory=list)

    @property
    def reported(self) -> list[Track]:
        """Confirmed tracks matched to a detection on this frame. This is what gets scored."""
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED and t.observed]


class BaseTracker:
    def __init__(self, cfg: TrackerConfig, high_thresh: float, low_thresh: float) -> None:
        self.cfg = cfg
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.tracks: list[Track] = []
        self._next_id = 1  # ids are never reused

    def update(self, detections: list[Detection], frame_index: int, timestamp: float) -> TrackerOutput:
        raise NotImplementedError

    # -- helpers -------------------------------------------------------------

    def _new_track(self, det: Detection, frame_index: int, timestamp: float) -> Track:
        t = Track(self._next_id, det.box, det.score, frame_index, timestamp, self.cfg.trajectory_len)
        self._next_id += 1
        if self.cfg.min_hits <= 1:
            t.state = TrackState.CONFIRMED
        return t

    def _retire(self) -> TrackerOutput:
        """Drop expired tracks and report which confirmed identities ended."""
        expired = [t for t in self.tracks if t.state == TrackState.EXPIRED]
        self.tracks = [t for t in self.tracks if t.state != TrackState.EXPIRED]
        return TrackerOutput(tracks=list(self.tracks), expired=expired)

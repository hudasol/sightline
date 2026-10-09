"""ByteTrack-style two-stage association, implemented from the published method.

Information used: geometry only. A Kalman filter predicts each track's box and
association is by IoU with Hungarian assignment. There is NO appearance model
and NO re-identification. Consequences:

  + low-confidence detections (often occluded or blurred people) are used to
    keep existing tracks alive, via the second stage;
  - two people who cross paths can swap ids, because nothing distinguishes them;
  - a person hidden for longer than max_age comes back with a new id;
  - camera motion shifts every box and can break IoU matching.
"""

from __future__ import annotations

from sightline.config import TrackerConfig
from sightline.tracking.base import BaseTracker, TrackerOutput
from sightline.tracking.kalman import KalmanBoxFilter, mean_to_box
from sightline.tracking.matching import associate
from sightline.types import Detection, TrackState


class ByteTracker(BaseTracker):
    def __init__(self, cfg: TrackerConfig, high_thresh: float = 0.5, low_thresh: float = 0.1) -> None:
        super().__init__(cfg, high_thresh, low_thresh)
        self.kf = KalmanBoxFilter()

    def update(self, detections: list[Detection], frame_index: int, timestamp: float) -> TrackerOutput:
        cfg = self.cfg
        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        # 1. predict every live track forward one step
        for t in self.tracks:
            t.age += 1
            assert t.mean is not None and t.cov is not None
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            t.box = mean_to_box(t.mean)

        pool = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]
        was_tracked = {t.id for t in pool if t.state == TrackState.CONFIRMED}

        # 2. first association: confirmed + lost tracks against HIGH-score detections
        m1, ut1, ud1 = associate([t.box for t in pool], [d.box for d in high], cfg.match_iou)
        for ti, di in m1:
            self._apply(pool[ti], high[di], frame_index, timestamp)

        # 2b. tentative tracks get one chance at the high detections that are left
        rest_high = [high[j] for j in ud1]
        m1b, ut1b, ud1b = associate([t.box for t in tentative], [d.box for d in rest_high], cfg.tentative_match_iou)
        for ti, di in m1b:
            self._apply(tentative[ti], rest_high[di], frame_index, timestamp)
        # an unmatched tentative track never earned an identity: drop it now
        dropped = {tentative[i].id for i in ut1b}

        # 3. second association: tracks that were tracked last frame and are still
        #    unmatched, against LOW-score detections. This is the part that keeps a
        #    partly-occluded or blurred person's identity alive.
        remaining = [pool[i] for i in ut1 if pool[i].id in was_tracked]
        m2, ut2, _ud2 = associate([t.box for t in remaining], [d.box for d in low], cfg.low_match_iou)
        matched2 = set()
        for ti, di in m2:
            self._apply(remaining[ti], low[di], frame_index, timestamp)
            matched2.add(remaining[ti].id)

        # 4. unmatched confirmed/lost tracks miss this frame
        for i in ut1:
            t = pool[i]
            if t.id not in matched2:
                t.mark_missed(cfg.max_age)

        # 5. new tracks only from high-score detections nobody claimed
        for j in ud1b:
            det = rest_high[j]
            if det.score >= cfg.new_track_score:
                t = self._new_track(det, frame_index, timestamp)
                t.mean, t.cov = self.kf.initiate(det.box)
                self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.id not in dropped]
        return self._retire()

    def _apply(self, track, det: Detection, frame_index: int, timestamp: float) -> None:
        assert track.mean is not None and track.cov is not None
        track.mean, track.cov = self.kf.update(track.mean, track.cov, det.box)
        track.box = det.box
        track.mark_matched(det.box, det.score, frame_index, timestamp, self.cfg.min_hits)

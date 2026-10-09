"""Simple geometric baseline: greedy IoU matching to the last seen box.

No motion model, no low-confidence recovery. The chosen tracker has to justify
replacing this on held-out data.
"""

from __future__ import annotations

import numpy as np

from sightline.config import TrackerConfig
from sightline.geometry import iou_matrix
from sightline.tracking.base import BaseTracker, TrackerOutput
from sightline.types import Detection, TrackState


class IoUTracker(BaseTracker):
    def __init__(self, cfg: TrackerConfig, high_thresh: float = 0.5, low_thresh: float = 0.1) -> None:
        super().__init__(cfg, high_thresh, low_thresh)

    def update(self, detections: list[Detection], frame_index: int, timestamp: float) -> TrackerOutput:
        cfg = self.cfg
        dets = [d for d in detections if d.score >= self.high_thresh]  # low scores are discarded
        for t in self.tracks:
            t.age += 1

        used_t: set[int] = set()
        used_d: set[int] = set()
        if self.tracks and dets:
            iou = iou_matrix(np.array([t.last_det_box for t in self.tracks]), np.array([d.box for d in dets]))
            order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            for ti, di in order:
                if iou[ti, di] < cfg.match_iou:
                    break
                if int(ti) in used_t or int(di) in used_d:
                    continue
                used_t.add(int(ti))
                used_d.add(int(di))
                t = self.tracks[int(ti)]
                t.box = dets[int(di)].box
                t.mark_matched(dets[int(di)].box, dets[int(di)].score, frame_index, timestamp, cfg.min_hits)

        survivors = []
        for i, t in enumerate(self.tracks):
            if i in used_t:
                survivors.append(t)
            elif t.state == TrackState.TENTATIVE:
                continue  # never earned an identity
            else:
                t.mark_missed(cfg.max_age)
                survivors.append(t)
        self.tracks = survivors

        for j, d in enumerate(dets):
            if j not in used_d and d.score >= cfg.new_track_score:
                self.tracks.append(self._new_track(d, frame_index, timestamp))
        return self._retire()

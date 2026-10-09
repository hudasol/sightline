"""The streaming pipeline: frame in, honest result out.

Stateful tracking and event logic live here and nowhere else. Presentation
(overlays, files) lives in runner.py / overlay.py and only reads FrameResult.

The corridor status rule: CLEAR is returned only when health is OK, nobody is
inside, and nobody might be inside (candidates and not-yet-confirmed tracks
count). Whenever the system cannot tell, the answer is UNKNOWN.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sightline.config import Config, CorridorConfig
from sightline.detector import Detector
from sightline.events import EventEngine, EventRecord
from sightline.geometry import box_ref_point
from sightline.health import HealthMonitor, HealthReport
from sightline.quality import measure_quality
from sightline.tracking import make_tracker
from sightline.types import CorridorStatus, Detection, FrameData, HealthState, TrackState, TrackView


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    health: HealthReport
    corridor: CorridorStatus
    detections: list[Detection] = field(default_factory=list)
    reported_tracks: list[TrackView] = field(default_factory=list)  # confirmed + matched this frame (scored)
    lost_tracks: list[TrackView] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    processed: bool = True
    latency_s: float = 0.0
    infer_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "ts": round(self.timestamp, 4),
            "processed": self.processed,
            "health": self.health.to_dict(),
            "corridor": self.corridor.value,
            "n_detections": len(self.detections),
            "tracks": [t.to_dict() for t in self.reported_tracks],
            "lost": [t.id for t in self.lost_tracks],
            "events": [e.to_dict() for e in self.events],
            "latency_ms": round(self.latency_s * 1000, 3),
            "infer_ms": round(self.infer_s * 1000, 3),
        }


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        corridor: CorridorConfig,
        detector: Detector | None,
        frame_size: tuple[int, int],
        nominal_fps: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.detector = detector
        self.corridor = corridor
        self.tracker = make_tracker(cfg.tracker, cfg.detector.score_threshold, cfg.detector.min_score)
        self.engine = EventEngine(cfg.events, corridor.polygon, frame_size)
        self.monitor = HealthMonitor(cfg.health, nominal_fps)
        self._last_index = 0
        self._last_ts = 0.0

    def process(self, frame: FrameData, detections: list[Detection] | None = None,
                arrival: float | None = None) -> FrameResult:
        """Process one frame.

        detections: pre-computed detections (cached evaluation). When given, the detector is
        not called and the runtime numbers are NOT valid evidence of real inference cost.
        arrival: wall-clock time the frame became available (live replay), so latency
        includes any queueing delay.
        """
        t0 = self.clock()
        arrival = t0 if arrival is None else arrival
        self._last_index, self._last_ts = frame.index, frame.timestamp

        # undecodable frame: do NOT update tracks, do NOT claim anything
        if not frame.valid or frame.image is None:
            health = self.monitor.observe(valid=False, ts=frame.timestamp, wall=t0)
            return FrameResult(frame.index, frame.timestamp, health,
                               CorridorStatus.OCCUPIED if self.engine.occupied else CorridorStatus.UNKNOWN,
                               processed=False, latency_s=self.clock() - arrival)

        quality = measure_quality(frame.image)
        infer_s = 0.0
        if detections is None:
            if self.detector is None:
                raise RuntimeError("no detector and no cached detections supplied")
            t_inf = self.clock()
            detections = self.detector.detect(frame.image)
            infer_s = self.clock() - t_inf

        out = self.tracker.update(detections, frame.index, frame.timestamp)
        records = self.engine.update(frame.index, frame.timestamp, out.tracks, out.expired)

        done = self.clock()
        health = self.monitor.observe(
            valid=True, ts=frame.timestamp, quality=quality, proc_time_s=done - t0,
            det_scores=[d.score for d in detections], wall=t0,
        )

        tentative_inside = any(
            t.state == TrackState.TENTATIVE and t.observed and self.corridor.polygon.contains(box_ref_point(t.box))
            for t in out.tracks
        )
        corridor = self._corridor_status(health, tentative_inside)

        return FrameResult(
            frame.index, frame.timestamp, health, corridor, detections,
            reported_tracks=[t.view() for t in out.reported],
            lost_tracks=[t.view() for t in out.tracks if t.state == TrackState.LOST],
            events=records, processed=True,
            latency_s=self.clock() - arrival, infer_s=infer_s,
        )

    def _corridor_status(self, health: HealthReport, tentative_inside: bool) -> CorridorStatus:
        if self.engine.occupied or tentative_inside:
            return CorridorStatus.OCCUPIED  # reporting "occupied" under doubt is the safe side
        if health.state == HealthState.OK:
            return CorridorStatus.CLEAR
        return CorridorStatus.UNKNOWN

    def finalize(self) -> list[EventRecord]:
        return self.engine.finalize(self._last_index, self._last_ts)

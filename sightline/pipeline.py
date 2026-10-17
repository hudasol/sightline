"""The streaming pipeline: frame in, honest result out.

Stateful tracking and event logic live here and nowhere else. Presentation
(overlays, files) lives in runner.py / overlay.py and only reads FrameResult.

The corridor status rule: CLEAR is returned only when health is OK, nobody is
inside, and nobody might be inside (candidates and not-yet-confirmed tracks
count). Whenever the system cannot tell, the answer is UNKNOWN.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sightline.async_detect import DetectionWorker
from sightline.config import Config, CorridorConfig, resolve_timing
from sightline.detector import Detector
from sightline.events import EventEngine, EventRecord
from sightline.geometry import box_ref_point
from sightline.health import HealthMonitor, HealthReport
from sightline.quality import measure_quality
from sightline.tracking import make_tracker
from sightline.types import CorridorStatus, Detection, FrameData, HealthState, TrackState, TrackView


def is_detector_frame(frame_index: int, detect_every: int) -> bool:
    """PLAN.md A2: the detector runs on source frames 1, 1+k, 1+2k, ... Evaluation uses the same rule."""
    return (frame_index - 1) % detect_every == 0


def detector_frames(n_frames: int, detect_every: int) -> list[int]:
    return list(range(1, n_frames + 1, detect_every))


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
    detector_ran: bool = True  # False on frames pipeline.detect_every skipped (tracks were predicted)
    # which frame's detections were applied on this frame (sync: this frame; async: an earlier one), and how late
    detection_applied_from: int | None = None
    detection_delay_frames: int | None = None
    detection_skipped_busy: bool = False  # async: a detector frame whose detection was skipped, worker busy

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "ts": round(self.timestamp, 4),
            "processed": self.processed,
            "detector_ran": self.detector_ran,
            "detection_applied_from": self.detection_applied_from,
            "detection_delay_frames": self.detection_delay_frames,
            "detection_skipped_busy": self.detection_skipped_busy,
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
        # time-based settings become frame counts for THIS clip's fps (and detection interval)
        cfg = resolve_timing(cfg, nominal_fps) if nominal_fps else cfg
        self.cfg = cfg
        self.clock = clock
        self.detector = detector
        self.corridor = corridor
        self.tracker = make_tracker(cfg.tracker, cfg.detector.score_threshold, cfg.detector.min_score)
        self.engine = EventEngine(cfg.events, corridor.polygon, frame_size)
        self.monitor = HealthMonitor(cfg.health, nominal_fps)
        self._last_index = 0
        self._last_ts = 0.0
        # async detection (live detector only): worker + tracker snapshots taken just before each submitted frame
        self.worker = DetectionWorker(detector, clock) if (cfg.pipeline.async_detection and detector is not None) else None
        self._snapshots: dict[int, object] = {}
        self._submitted_ts: dict[int, float] = {}
        self._last_det_ts: float | None = None  # timestamp of the frame the latest applied detection was run on
        self._first_ts: float | None = None

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def process(self, frame: FrameData, detections: list[Detection] | None = None,
                arrival: float | None = None) -> FrameResult:
        """Process one frame.

        detections: pre-computed detections (cached evaluation). When given, the detector is
        not called and the runtime numbers are NOT valid evidence of real inference cost.
        On frames pipeline.detect_every skips they are ignored and the tracker predicts.
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
        detector_ran = is_detector_frame(frame.index, self.cfg.pipeline.detect_every)
        applied_from = delay = None
        skipped_busy = False
        replay_expired = []
        if self.worker is not None and detections is None:
            # async: apply a finished detection by replay, submit this frame if it is a detector frame,
            # and predict this frame without waiting for anything
            done_det = self.worker.poll()
            if done_det is not None:
                f_det, detections, infer_s = done_det
                replay_expired = self._apply_late(f_det, detections, frame.index)
                applied_from, delay = f_det, frame.index - f_det
                self._last_det_ts = self._submitted_ts.pop(f_det, frame.timestamp)
            else:
                detections = []
            if detector_ran:
                if self.worker.idle:
                    self._snapshots[frame.index] = copy.deepcopy(self.tracker)
                    self._submitted_ts[frame.index] = frame.timestamp
                    self.worker.submit(frame.index, frame.image)
                else:
                    skipped_busy = True
            out = self.tracker.predict(frame.index, frame.timestamp)
            detector_ran = applied_from is not None  # "ran" = detections reached the tracker on this frame
        elif not detector_ran:
            detections = []
            out = self.tracker.predict(frame.index, frame.timestamp)
        else:
            if detections is None:
                if self.detector is None:
                    raise RuntimeError("no detector and no cached detections supplied")
                t_inf = self.clock()
                detections = self.detector.detect(frame.image)
                infer_s = self.clock() - t_inf
            out = self.tracker.update(detections, frame.index, frame.timestamp)
            applied_from, delay = frame.index, 0
            self._last_det_ts = frame.timestamp
        records = self.engine.update(frame.index, frame.timestamp, out.tracks, replay_expired + out.expired)

        done = self.clock()
        if self._first_ts is None:
            self._first_ts = frame.timestamp
        # source time since the image behind the tracker's latest detection (or since the stream began)
        age = frame.timestamp - (self._last_det_ts if self._last_det_ts is not None else self._first_ts)
        health = self.monitor.observe(
            valid=True, ts=frame.timestamp, quality=quality, proc_time_s=done - t0,
            det_scores=[d.score for d in detections] if detector_ran else None, wall=t0, detection_age_s=age,
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
            latency_s=self.clock() - arrival, infer_s=infer_s, detector_ran=detector_ran,
            detection_applied_from=applied_from, detection_delay_frames=delay, detection_skipped_busy=skipped_busy,
        )

    def _apply_late(self, f_det: int, detections: list[Detection], f_now: int) -> list:
        """Replay: the tracker as it was just before f_det, updated with f_det's detections, then
        predicted through f_det+1 .. f_now-1. Returns the tracks that expired during the replay."""
        tracker = self._snapshots.pop(f_det)
        out = tracker.update(detections, f_det, 0.0)
        for f in range(f_det + 1, f_now):
            tracker.predict(f, 0.0)
        self.tracker = tracker
        return list(out.expired)

    def _corridor_status(self, health: HealthReport, tentative_inside: bool) -> CorridorStatus:
        if self.engine.occupied or tentative_inside:
            return CorridorStatus.OCCUPIED  # reporting "occupied" under doubt is the safe side
        if health.state == HealthState.OK:
            return CorridorStatus.CLEAR
        return CorridorStatus.UNKNOWN

    def finalize(self) -> list[EventRecord]:
        return self.engine.finalize(self._last_index, self._last_ts)

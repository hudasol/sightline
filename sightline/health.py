"""System health: OK -> DEGRADED -> UNAVAILABLE.

The rule this module exists to enforce: the system never reports a clear
corridor when it cannot tell. Anything that makes the input untrustworthy
moves the state away from OK, and it only moves back after a run of clean
frames (hysteresis), so a single good frame cannot hide a bad stretch.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

from sightline.config import HealthConfig
from sightline.quality import Quality
from sightline.types import HealthState


@dataclass
class HealthReport:
    state: HealthState
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"state": self.state.value, "reasons": list(self.reasons)}


_LEVELS = [HealthState.OK, HealthState.DEGRADED, HealthState.UNAVAILABLE]


class HealthMonitor:
    def __init__(self, cfg: HealthConfig, nominal_fps: float | None = None) -> None:
        self.cfg = cfg
        self.nominal_interval = (1.0 / nominal_fps) if nominal_fps and nominal_fps > 0 else None
        self.state = HealthState.OK
        self._consec_corrupt = 0
        self._recent_valid: deque[bool] = deque(maxlen=cfg.corrupt_window)
        self._last_ts: float | None = None
        self._gaps: deque[float] = deque(maxlen=30)
        self._proc_times: deque[float] = deque(maxlen=cfg.fps_window)
        self._conf_window: deque[float] = deque(maxlen=cfg.low_conf_window)
        self._frames_seen = 0
        self._clean_streak = 0
        self._latched: str | None = None
        self._last_arrival_wall: float | None = None

    # -- hard failures ---------------------------------------------------------

    def fail(self, reason: str) -> HealthReport:
        """Unreadable source, invalid configuration, ...: UNAVAILABLE until reset."""
        self._latched = reason
        self.state = HealthState.UNAVAILABLE
        return HealthReport(self.state, [reason])

    def check_stall(self, now_wall: float) -> HealthReport | None:
        """For live input: call periodically. If no frame has arrived for
        stall_seconds the stream has stalled, even though observe() is not being called."""
        if self._last_arrival_wall is None:
            return None
        if now_wall - self._last_arrival_wall > self.cfg.stall_seconds:
            self.state = HealthState.UNAVAILABLE
            self._clean_streak = 0
            return HealthReport(self.state, ["no_frames_received"])
        return None

    # -- per frame ---------------------------------------------------------------

    def observe(
        self,
        *,
        valid: bool,
        ts: float,
        quality: Quality | None = None,
        proc_time_s: float | None = None,
        det_scores: list[float] | None = None,
        wall: float | None = None,
    ) -> HealthReport:
        cfg = self.cfg
        if self._latched:
            return HealthReport(HealthState.UNAVAILABLE, [self._latched])
        self._frames_seen += 1
        if wall is not None:
            self._last_arrival_wall = wall
        triggers: list[tuple[HealthState, str]] = []

        # corrupt frames
        self._recent_valid.append(valid)
        if not valid:
            self._consec_corrupt += 1
            if self._consec_corrupt >= cfg.k_corrupt:
                triggers.append((HealthState.UNAVAILABLE, f"corrupt_frames_x{self._consec_corrupt}"))
            else:
                triggers.append((HealthState.DEGRADED, "corrupt_frame"))
        else:
            self._consec_corrupt = 0
            if not all(self._recent_valid):
                triggers.append((HealthState.DEGRADED, "recent_corrupt_frame"))

        # timestamps
        if self._last_ts is not None:
            gap = ts - self._last_ts
            if gap <= 0:
                triggers.append((HealthState.DEGRADED, "non_monotonic_timestamp"))
            else:
                nominal = self.nominal_interval or (statistics.median(self._gaps) if len(self._gaps) >= 5 else None)
                if gap > cfg.stall_seconds:
                    triggers.append((HealthState.UNAVAILABLE, "stream_stalled"))
                elif nominal is not None and gap > cfg.gap_factor * nominal:
                    triggers.append((HealthState.DEGRADED, "timestamp_gap"))
                if gap <= cfg.stall_seconds:
                    self._gaps.append(gap)
        if valid or self._last_ts is None or ts > self._last_ts:
            self._last_ts = ts

        # throughput
        if proc_time_s is not None and valid:
            self._proc_times.append(proc_time_s)
            if self._frames_seen > cfg.warmup_frames and len(self._proc_times) >= min(cfg.fps_window, 10):
                mean_t = sum(self._proc_times) / len(self._proc_times)
                if mean_t > 0 and (1.0 / mean_t) < cfg.min_fps:
                    triggers.append((HealthState.DEGRADED, f"low_fps_{1.0 / mean_t:.1f}"))

        # image quality
        if quality is not None and valid:
            if quality.luminance < cfg.min_luminance:
                triggers.append((HealthState.DEGRADED, "low_light"))
            if quality.sharpness < cfg.min_sharpness:
                triggers.append((HealthState.DEGRADED, "blurry_or_low_contrast"))

        # detector confidence
        if det_scores and valid:
            self._conf_window.append(sum(det_scores) / len(det_scores))
            if len(self._conf_window) >= 10 and (sum(self._conf_window) / len(self._conf_window)) < cfg.low_conf_mean:
                triggers.append((HealthState.DEGRADED, "low_detection_confidence"))

        return self._resolve(triggers)

    def _resolve(self, triggers: list[tuple[HealthState, str]]) -> HealthReport:
        if triggers:
            worst = max(triggers, key=lambda t: t[0].severity)[0]
            self._clean_streak = 0
            if worst.severity >= self.state.severity:
                self.state = worst
            return HealthReport(self.state, [r for _s, r in triggers])
        # nothing wrong right now: step down one level only after a clean run
        if self.state != HealthState.OK:
            self._clean_streak += 1
            if self._clean_streak >= self.cfg.recover_frames:
                self.state = _LEVELS[self.state.severity - 1]
                self._clean_streak = 0
            return HealthReport(self.state, ["recovering"] if self.state != HealthState.OK else [])
        return HealthReport(HealthState.OK, [])

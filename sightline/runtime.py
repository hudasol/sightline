"""Runtime evidence: throughput, latency percentiles, inference time and every
frame that was dropped, skipped or late. Nothing is hidden to make fps look better."""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def machine_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        info["torch"] = None
    return info


def summarize_runs(parent: Path) -> dict:
    """Median and range over repeated benchmark runs (PLAN.md amendment A4).

    `parent` holds one folder per run, named <config>_run<N>, each with one benchmark report
    per clip. Per run: throughput = frames processed / wall time over all clips; latency is
    the WORST per-clip p95 and the overall maximum (the reports keep percentiles, not every
    latency, so no pooled p95 is invented). Per config: the median of the runs and min-max.
    """
    import json
    import re
    import statistics

    groups: dict[str, list[dict]] = {}
    for run_dir in sorted(d for d in Path(parent).iterdir() if d.is_dir()):
        m = re.fullmatch(r"(.+)_run(\d+)", run_dir.name)
        if not m:
            continue
        reports = [json.loads(f.read_text()) for f in sorted(run_dir.glob("*.json"))]
        clips = [c for r in reports for c in r["per_clip"]]
        if not clips:
            continue
        frames = sum(c["frames_processed"] for c in clips)
        wall = sum(c["wall_seconds"] for c in clips)
        groups.setdefault(m.group(1), []).append({
            "run": int(m.group(2)), "clips": len(clips), "frames_processed": frames,
            "frames_dropped": sum(c["frames_dropped_lagging"] for c in clips),
            "throughput_fps": frames / wall if wall else None,
            "worst_clip_latency_ms_p95": max(c["latency_ms_p95"] for c in clips),
            "latency_ms_max": max(c["latency_ms_max"] for c in clips),
            "detections_skipped_worker_busy": sum(c.get("detections_skipped_worker_busy", 0) for c in clips),
            "async_detection": reports[0].get("async_detection", False),
            "realtime_emulation": reports[0].get("realtime_emulation"),
        })
    out = {}
    keys = ("throughput_fps", "frames_dropped", "worst_clip_latency_ms_p95", "latency_ms_max",
            "detections_skipped_worker_busy")
    for name, runs in groups.items():
        out[name] = {"runs": len(runs), "per_run": sorted(runs, key=lambda r: r["run"]),
                     **{k: {"median": statistics.median(r[k] for r in runs), "min": min(r[k] for r in runs),
                            "max": max(r[k] for r in runs)} for k in keys}}
    return out


@dataclass
class RuntimeRecorder:
    warmup: int = 5
    budget_s: float = 0.1  # a frame slower than this is "late" (default: the 10 fps floor)
    latencies: list[float] = field(default_factory=list)  # arrival -> result, per processed frame
    infer: list[float] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)  # lagging live stream: frames not processed
    skipped_policy: list[int] = field(default_factory=list)  # frames the sampling policy skipped
    delays: list[int] = field(default_factory=list)  # frames between a detector frame and its detections' use
    skipped_busy: int = 0  # async: detector frames whose detection was skipped because the worker was busy
    invalid: int = 0
    _t0: float | None = None
    _t1: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._t1 = time.perf_counter()

    def record(self, latency_s: float, infer_s: float | None, delay_frames: int | None = None,
               skipped_busy: bool = False) -> None:
        """infer_s is None on frames where no detection reached the tracker (PLAN.md A2)."""
        self.latencies.append(latency_s)
        if infer_s is not None:
            self.infer.append(infer_s)
        if delay_frames is not None:
            self.delays.append(delay_frames)
        self.skipped_busy += int(skipped_busy)

    def summary(self) -> dict:
        wall = ((self._t1 or time.perf_counter()) - self._t0) if self._t0 is not None else float("nan")
        lat = np.array(self.latencies[self.warmup:]) if len(self.latencies) > self.warmup else np.array(self.latencies)
        inf = np.array(self.infer[self.warmup:]) if len(self.infer) > self.warmup else np.array(self.infer)

        def pct(a: np.ndarray, q: float) -> float | None:
            return float(np.percentile(a, q) * 1000.0) if len(a) else None

        processed = len(self.latencies)
        detector_frames = len(self.infer)
        late = int((lat > self.budget_s).sum()) if len(lat) else 0
        return {
            "frames_processed": processed,
            "detector_frames": detector_frames,
            "frames_dropped_lagging": len(self.dropped),
            "frames_skipped_by_policy": len(self.skipped_policy),
            "frames_invalid": self.invalid,
            "frames_late": late,
            "late_budget_ms": self.budget_s * 1000.0,
            "warmup_frames_excluded_from_latency": min(self.warmup, processed),
            "wall_seconds": wall,
            "throughput_fps": (processed / wall) if wall and wall > 0 else None,
            "latency_ms_p50": pct(lat, 50),
            "latency_ms_p95": pct(lat, 95),
            "latency_ms_max": float(lat.max() * 1000.0) if len(lat) else None,
            "inference_ms_p50": pct(inf, 50),
            "inference_ms_p95": pct(inf, 95),
            "dropped_frame_indices": self.dropped[:500],
            "detections_skipped_worker_busy": self.skipped_busy,
            "detection_delay_frames_p50": float(np.percentile(self.delays, 50)) if self.delays else None,
            "detection_delay_frames_max": max(self.delays) if self.delays else None,
        }

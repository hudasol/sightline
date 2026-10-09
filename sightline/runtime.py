"""Runtime evidence: throughput, latency percentiles, inference time and every
frame that was dropped, skipped or late. Nothing is hidden to make fps look better."""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field

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


@dataclass
class RuntimeRecorder:
    warmup: int = 5
    budget_s: float = 0.1  # a frame slower than this is "late" (default: the 10 fps floor)
    latencies: list[float] = field(default_factory=list)  # arrival -> result, per processed frame
    infer: list[float] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)  # lagging live stream: frames not processed
    skipped_policy: list[int] = field(default_factory=list)  # frames the sampling policy skipped
    invalid: int = 0
    _t0: float | None = None
    _t1: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._t1 = time.perf_counter()

    def record(self, latency_s: float, infer_s: float) -> None:
        self.latencies.append(latency_s)
        self.infer.append(infer_s)

    def summary(self) -> dict:
        wall = ((self._t1 or time.perf_counter()) - self._t0) if self._t0 is not None else float("nan")
        lat = np.array(self.latencies[self.warmup:]) if len(self.latencies) > self.warmup else np.array(self.latencies)
        inf = np.array(self.infer[self.warmup:]) if len(self.infer) > self.warmup else np.array(self.infer)

        def pct(a: np.ndarray, q: float) -> float | None:
            return float(np.percentile(a, q) * 1000.0) if len(a) else None

        processed = len(self.latencies)
        late = int((lat > self.budget_s).sum()) if len(lat) else 0
        return {
            "frames_processed": processed,
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
        }

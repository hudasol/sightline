"""Run one clip through the pipeline and write the evidence.

Failure policy: if the source cannot be read, or the pipeline raises, the run
ends UNAVAILABLE with a reason in health.jsonl and summary.json. It never ends
looking like "clear".
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sightline.config import Config, CorridorConfig
from sightline.detector import Detector
from sightline.events import Event
from sightline.ingest import IngestError, SourceInfo, iter_frames, probe
from sightline.overlay import OverlayWriter
from sightline.pipeline import FrameResult, Pipeline
from sightline.runtime import RuntimeRecorder, machine_info
from sightline.types import CorridorStatus, Detection, FrameData, HealthState


@dataclass
class ClipRun:
    clip: str
    info: SourceInfo | None
    events: list[Event] = field(default_factory=list)
    mot_rows: list[tuple] = field(default_factory=list)  # (frame, id, x1, y1, x2, y2, score)
    runtime: dict = field(default_factory=dict)
    health_counts: dict[str, int] = field(default_factory=dict)
    corridor_counts: dict[str, int] = field(default_factory=dict)
    final_health: str = HealthState.UNAVAILABLE.value
    unavailable_reason: str | None = None
    frames_seen: int = 0
    policy: dict = field(default_factory=dict)
    results: list[FrameResult] = field(default_factory=list)
    duration_s: float = 0.0
    corridor_by_frame: dict[int, str] = field(default_factory=dict)  # frame -> CLEAR | OCCUPIED | UNKNOWN

    def summary(self) -> dict:
        return {
            "clip": self.clip,
            "final_health": self.final_health,
            "unavailable_reason": self.unavailable_reason,
            "frames_seen": self.frames_seen,
            "duration_s": round(self.duration_s, 3),
            "health_frame_counts": self.health_counts,
            "corridor_frame_counts": self.corridor_counts,
            "events": [e.to_dict() for e in self.events],
            "events_unresolved": sum(1 for e in self.events if e.status.value == "UNRESOLVED"),
            "policy": self.policy,
            "runtime": self.runtime,
            "source": self.info.to_dict() if self.info else None,
        }


def run_clip(
    source: str | Path,
    cfg: Config,
    corridor: CorridorConfig,
    detector: Detector | None = None,
    cached: dict[int, list[Detection]] | None = None,
    fps_override: float | None = None,
    out_dir: str | Path | None = None,
    overlay: bool = False,
    realtime: bool = False,
    frames: Iterable[FrameData] | None = None,
    info: SourceInfo | None = None,
    keep_results: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> ClipRun:
    """frames/info let a caller inject a stream (tests, stress suite); otherwise `source` is read."""
    name = Path(str(source)).name
    run = ClipRun(clip=name, info=None)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # 1. open the source. Failure here is UNAVAILABLE, not a crash and not "clear".
    if info is None:
        try:
            info = probe(source, fps_override)
        except IngestError as exc:
            return _unavailable(run, out, f"source_unreadable: {exc}")
    run.info = info
    stride = cfg.pipeline.frame_stride
    run.policy = {"frame_stride": stride, "realtime_emulation": realtime, "tracker": cfg.tracker.kind,
                  "detections": "cache" if cached is not None else "live",
                  "note": "stride > 1 means frames were skipped by policy; judge it by event recall/latency"}

    if frames is None:
        frames = iter_frames(source, fps_override, stride=stride, info=info)

    pipe = Pipeline(cfg, corridor, detector, (info.width, info.height), nominal_fps=info.fps / stride, clock=clock)
    rec = RuntimeRecorder(warmup=cfg.health.warmup_frames, budget_s=1.0 / cfg.health.min_fps)
    health_log = (out / "health.jsonl").open("w") if out else None
    events_log = (out / "events.jsonl").open("w") if out else None
    tracks_log = (out / "tracks.jsonl").open("w") if out else None
    writer = None
    if out and overlay:
        writer = OverlayWriter(out / "overlay.mp4", info.fps / stride, (info.width, info.height), corridor.polygon)

    health_counts: Counter[str] = Counter()
    corridor_counts: Counter[str] = Counter()
    first_ts: float | None = None
    last_index = 0
    lag_budget = 1.0 / cfg.health.min_fps
    rec.start()
    t_start = clock()
    try:
        for frame in frames:
            last_index = frame.index
            run.frames_seen += 1
            if first_ts is None:
                first_ts = frame.timestamp
            arrival = None
            if realtime:
                arrival = t_start + (frame.timestamp - first_ts)
                wait = arrival - clock()
                if wait > 0:
                    time.sleep(wait)
                elif -wait > lag_budget:
                    rec.dropped.append(frame.index)  # stream is ahead of us: this frame is gone
                    continue
            det = cached.get(frame.index, []) if cached is not None else None
            try:
                res = pipe.process(frame, detections=det, arrival=arrival)
            except Exception as exc:  # the pipeline must fail loudly, not look clear
                reason = f"pipeline_error: {type(exc).__name__}: {exc}"
                report = pipe.monitor.fail(reason)
                run.unavailable_reason = reason
                res = FrameResult(frame.index, frame.timestamp, report, CorridorStatus.UNKNOWN, processed=False)
                _log(res, health_log, events_log, tracks_log)
                health_counts[res.health.state.value] += 1
                corridor_counts[res.corridor.value] += 1
                break

            run.corridor_by_frame[frame.index] = res.corridor.value
            if res.processed:
                rec.record(res.latency_s, res.infer_s)
            else:
                rec.invalid += 1
            health_counts[res.health.state.value] += 1
            corridor_counts[res.corridor.value] += 1
            for t in res.reported_tracks:
                x1, y1, x2, y2 = t.box
                run.mot_rows.append((frame.index, t.id, x1, y1, x2, y2, t.score))
            _log(res, health_log, events_log, tracks_log)
            if writer is not None and frame.image is not None:
                writer.write(frame.image, res)
            if keep_results:
                run.results.append(res)
    finally:
        rec.stop()
        if writer is not None:
            writer.close()

    final_records = pipe.finalize()
    if events_log:
        for r in final_records:
            events_log.write(json.dumps(r.to_dict()) + "\n")
    for f in (health_log, events_log, tracks_log):
        if f:
            f.close()

    rec.skipped_policy = [i for i in range(1, last_index + 1) if (i - 1) % stride]
    run.events = pipe.engine.all_events()
    run.runtime = {**rec.summary(), "machine": machine_info()}
    run.health_counts = dict(health_counts)
    run.corridor_counts = dict(corridor_counts)
    run.final_health = pipe.monitor.state.value
    run.duration_s = (last_index / info.fps) if info.fps else 0.0
    if run.frames_seen == 0:
        run.unavailable_reason = run.unavailable_reason or "no_frames_decoded"
        run.final_health = HealthState.UNAVAILABLE.value
    if out:
        _write_outputs(run, out)
    return run


def _log(res: FrameResult, health_log, events_log, tracks_log) -> None:
    if health_log:
        health_log.write(json.dumps({"frame": res.frame_index, "ts": round(res.timestamp, 4),
                                     "state": res.health.state.value, "reasons": res.health.reasons,
                                     "corridor": res.corridor.value}) + "\n")
    if events_log:
        for r in res.events:
            events_log.write(json.dumps(r.to_dict()) + "\n")
    if tracks_log and res.reported_tracks:
        tracks_log.write(json.dumps({"frame": res.frame_index, "tracks": [t.to_dict() for t in res.reported_tracks]}) + "\n")


def _write_outputs(run: ClipRun, out: Path) -> None:
    (out / "summary.json").write_text(json.dumps(run.summary(), indent=1))
    (out / "events_final.json").write_text(json.dumps([e.to_dict() for e in run.events], indent=1))
    (out / "runtime.json").write_text(json.dumps(run.runtime, indent=1))
    with (out / "tracks.txt").open("w") as f:  # MOTChallenge format
        for fr, tid, x1, y1, x2, y2, s in run.mot_rows:
            f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},-1,-1,-1\n")


def _unavailable(run: ClipRun, out: Path | None, reason: str) -> ClipRun:
    run.unavailable_reason = reason
    run.final_health = HealthState.UNAVAILABLE.value
    run.health_counts = {HealthState.UNAVAILABLE.value: 1}
    run.corridor_counts = {CorridorStatus.UNKNOWN.value: 1}
    run.runtime = {"machine": machine_info()}
    if out:
        (out / "health.jsonl").write_text(json.dumps({"frame": 0, "ts": 0.0, "state": "UNAVAILABLE",
                                                      "reasons": [reason], "corridor": "UNKNOWN"}) + "\n")
        _write_outputs(run, out)
    return run

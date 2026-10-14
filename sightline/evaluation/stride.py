"""Frame sampling (pipeline.frame_stride) vs event recall and latency, on cached detections.

This measures what skipping frames costs; it tunes nothing. The tracker and event
parameters (max_age, min_hits, n_open, m_close, cooldown) count PROCESSED frames and are
deliberately left as configured, so at stride s each of them spans s times more video.
Latency is always in ORIGINAL frames (and ms at the clip's own fps): the event's open
frame minus the ground-truth first-inside frame.
"""

from __future__ import annotations

import copy
import statistics
from collections.abc import Callable
from pathlib import Path

from sightline.config import Config
from sightline.detector import Detector
from sightline.evaluation.events import _percentile, event_metrics
from sightline.evaluation.run import evaluate_clip, get_detections, prepare_clip
from sightline.manifest import Manifest


def _summary(values: list[float]) -> dict:
    return {"median": statistics.median(values) if values else None,
            "p95": _percentile(values, 95) if values else None, "n": len(values)}


def stride_sweep(man: Manifest, cfg: Config, eval_cfg: dict, root: Path, cache_dir: Path,
                 detector_factory: Callable[[], Detector] | None, strides: list[int],
                 tracker: str = "bytetrack", progress=print) -> dict:
    clips = [c for c in man.clips if c.scored_on("events")]
    cds = [prepare_clip(c, root, eval_cfg) for c in clips]
    dets = {cd.clip.id: get_detections(cd, cfg, root, cache_dir, detector_factory) for cd in cds}
    rows = []
    for s in strides:
        c = copy.deepcopy(cfg)
        c.pipeline.frame_stride = s
        res = [evaluate_clip(cd, c, dets[cd.clip.id], eval_cfg, root, trackers=(tracker,)) for cd in cds]
        m = event_metrics([r.matches[tracker] for r in res], [r.cd.clip.negative for r in res],
                          [r.cd.info.frame_count_reported / r.cd.info.fps for r in res])
        lat_ms = [x / r.cd.info.fps * 1000.0 for r in res for x in r.matches[tracker].latencies]
        per_clip = [{"clip": r.cd.clip.id, "source_fps": r.cd.info.fps, "required_processing_fps": r.cd.info.fps / s,
                     "gt_events": len(r.cd.gt_events), "tp": r.matches[tracker].tp, "fp": r.matches[tracker].fp,
                     "fn": r.matches[tracker].fn, "latencies_frames": r.matches[tracker].latencies} for r in res]
        row = {"frame_stride": s,
               "event_recall": m["event_recall"], "event_precision": m["event_precision"],
               "events_tp": m["events_tp"], "events_fp": m["events_fp"], "events_fn": m["events_fn"],
               "events_unresolved": m["events_unresolved"],
               "latency_frames": _summary(m["latency_frames_all"]), "latency_ms": _summary(lat_ms),
               "per_clip": per_clip}
        rows.append(row)
        lf, lm = row["latency_frames"], row["latency_ms"]
        progress(f"stride={s}: event R={_fmt(m['event_recall'], 3)} P={_fmt(m['event_precision'], 3)} "
                 f"latency median={_fmt(lf['median'], 1)} fr / {_fmt(lm['median'], 0)} ms  "
                 f"p95={_fmt(lf['p95'], 1)} fr / {_fmt(lm['p95'], 0)} ms")
    return {
        "what": "event recall/precision and latency vs pipeline.frame_stride, on cached detections; nothing tuned",
        "manifest": man.path, "event_clips": [c.id for c in clips],
        "excluded_not_scored_on_events": [c.id for c in man.clips if not c.scored_on("events")],
        "detector": cfg.detector.name, "input_size": list(cfg.detector.input_size),
        "score_threshold": cfg.detector.score_threshold, "tracker": tracker,
        "tracker_params": copy.deepcopy(cfg.tracker.__dict__), "event_params": copy.deepcopy(cfg.events.__dict__),
        "note": "tracker/event parameters count processed frames and were NOT rescaled for stride",
        "rows": rows,
    }


def _fmt(v, decimals: int) -> str:
    return "n/a" if v is None else f"{v:.{decimals}f}"

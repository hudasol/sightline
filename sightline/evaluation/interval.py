"""Detection interval (pipeline.detect_every, PLAN.md A2) vs event recall and latency, on cached detections.

Every frame is processed; the detector runs on every k-th frame and the tracker predicts
in between. This measures what that costs; it tunes nothing. Latency is in ORIGINAL frames
(and ms at the clip's own fps): the event's open frame minus the ground-truth first-inside
frame. Events whose evidence rests on predicted boxes are counted separately.
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


def interval_sweep(man: Manifest, cfg: Config, eval_cfg: dict, root: Path, cache_dir: Path,
                   detector_factory: Callable[[], Detector] | None, intervals: list[int],
                   tracker: str = "bytetrack", progress: Callable[[str], object] = print) -> dict:
    clips = [c for c in man.clips if c.scored_on("events")]
    cds = [prepare_clip(c, root, eval_cfg) for c in clips]
    dets = {cd.clip.id: get_detections(cd, cfg, root, cache_dir, detector_factory) for cd in cds}
    rows = []
    for k in intervals:
        c = copy.deepcopy(cfg)
        c.pipeline.detect_every = k
        res = [evaluate_clip(cd, c, dets[cd.clip.id], eval_cfg, root, trackers=(tracker,)) for cd in cds]
        m = event_metrics([r.matches[tracker] for r in res], [r.cd.clip.negative for r in res],
                          [r.cd.info.frame_count_reported / r.cd.info.fps for r in res])
        lat_ms = [x / r.cd.info.fps * 1000.0 for r in res for x in r.matches[tracker].latencies]
        per_clip = [{"clip": r.cd.clip.id, "source_fps": r.cd.info.fps, "detector_runs_per_second": r.cd.info.fps / k,
                     "gt_events": len(r.cd.gt_events), "tp": r.matches[tracker].tp, "fp": r.matches[tracker].fp,
                     "fn": r.matches[tracker].fn, "latencies_frames": r.matches[tracker].latencies} for r in res]
        events = [e for r in res for e in r.runs[tracker].events]
        matched = {(r.cd.clip.id, pid) for r in res for _g, pid in r.matches[tracker].pairs}
        tp_events = [e for r in res for e in r.runs[tracker].events if (r.cd.clip.id, e.event_id) in matched]
        row = {"detect_every": k,
               "event_recall": m["event_recall"], "event_precision": m["event_precision"],
               "events_tp": m["events_tp"], "events_fp": m["events_fp"], "events_fn": m["events_fn"],
               "events_unresolved": m["events_unresolved"],
               # PLAN.md A2: how much of the event output rests on predicted boxes
               "events_total": len(events),
               "events_opened_on_prediction": sum(e.opened_on_prediction for e in events),
               "events_resting_on_prediction": sum(e.rests_on_prediction for e in events),
               "tp_events_resting_on_prediction": sum(e.rests_on_prediction for e in tp_events),
               "latency_frames": _summary(m["latency_frames_all"]), "latency_ms": _summary(lat_ms),
               "per_clip": per_clip}
        rows.append(row)
        lf, lm = row["latency_frames"], row["latency_ms"]
        progress(f"detect_every={k}: event R={_fmt(m['event_recall'], 3)} P={_fmt(m['event_precision'], 3)} "
                 f"latency median={_fmt(lf['median'], 1)} fr / {_fmt(lm['median'], 0)} ms  "
                 f"p95={_fmt(lf['p95'], 1)} fr / {_fmt(lm['p95'], 0)} ms  "
                 f"opened on prediction {row['events_opened_on_prediction']}/{row['events_total']}, "
                 f"resting on prediction {row['events_resting_on_prediction']}")
    return {
        "what": "event recall/precision and latency vs pipeline.detect_every (PLAN.md A2), on cached detections; "
                "nothing tuned",
        "manifest": man.path, "event_clips": [c.id for c in clips],
        "excluded_not_scored_on_events": [c.id for c in man.clips if not c.scored_on("events")],
        "detector": cfg.detector.name, "input_size": list(cfg.detector.input_size),
        "score_threshold": cfg.detector.score_threshold, "tracker": tracker,
        "tracker_params": copy.deepcopy(cfg.tracker.__dict__), "event_params": copy.deepcopy(cfg.events.__dict__),
        "rows": rows,
    }


def _fmt(v: float | None, decimals: int) -> str:
    return "n/a" if v is None else f"{v:.{decimals}f}"

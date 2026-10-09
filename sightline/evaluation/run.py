"""Evaluation orchestration: detection, tracking and event metrics over a manifest,
overall and per condition, for both the baseline and the chosen tracker.

Detections are computed once per clip (or read from a validated cache) and shared by
both trackers, so the tracker comparison isolates association, not the detector.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sightline.config import Config, CorridorConfig, load_corridor
from sightline.detector import Detector, read_detection_cache, write_detection_cache
from sightline.evaluation import detection as det_eval
from sightline.evaluation import tracking as trk_eval
from sightline.evaluation.events import (EventMatch, GTEvent, derive_gt_events, event_metrics, load_gt_events,
                                         match_events)
from sightline.evaluation.gt import GTFrames, IgnoreFrames, load_mot_gt
from sightline.ingest import SourceInfo, iter_frames, probe
from sightline.manifest import Clip
from sightline.runner import ClipRun, run_clip
from sightline.types import Detection

TRACKERS = ("iou", "bytetrack")


def load_eval_config(path: str | Path = "configs/eval.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass
class ClipData:
    clip: Clip
    info: SourceInfo
    corridor: CorridorConfig
    gt: GTFrames
    ignore: IgnoreFrames
    gt_events: list[GTEvent]
    n_frames: int
    tags: list[str]


def prepare_clip(clip: Clip, root: Path, eval_cfg: dict) -> ClipData:
    info = probe(root / clip.source, clip.fps)
    corridor = load_corridor(root / clip.corridor)
    gt, ignore = load_mot_gt(root / clip.gt)
    if clip.gt_events:
        gt_events = load_gt_events(root / clip.gt_events)
    else:
        rule = eval_cfg["gt_events"]
        gt_events = derive_gt_events(gt, corridor.polygon, rule["min_frames"], rule["merge_gap"])
    n = info.frame_count_reported or (max(gt) if gt else 0)
    tags = list(clip.conditions) + (["negative_corridor"] if clip.negative else [])
    return ClipData(clip, info, corridor, gt, ignore, gt_events, n, tags)


def get_detections(cd: ClipData, cfg: Config, root: Path, cache_dir: Path,
                   detector_factory: Callable[[], Detector] | None) -> dict[int, list[Detection]]:
    """Cached detections if the cache was made by the same model and floor; else run the detector."""
    path = cache_dir / cfg.detector.name / f"{cd.clip.id}.jsonl"
    want = {"detector": cfg.detector.name, "min_score": cfg.detector.min_score,
            "input_short_side": cfg.detector.input_short_side}
    if path.is_file():
        meta, frames = read_detection_cache(path)
        if all(meta.get(k) == v for k, v in want.items()):
            return frames
    if detector_factory is None:
        raise RuntimeError(f"no valid detection cache for {cd.clip.id} at {path} and no detector available")
    detector = detector_factory()
    frames: dict[int, list[Detection]] = {}
    for fr in iter_frames(root / cd.clip.source, cd.clip.fps, stride=1, info=cd.info):
        frames[fr.index] = detector.detect(fr.image) if fr.valid and fr.image is not None else []
    write_detection_cache(path, {**want, **detector.info()}, frames)
    return frames


@dataclass
class ClipResult:
    cd: ClipData
    det_counts: dict
    runs: dict[str, ClipRun] = field(default_factory=dict)
    accs: dict = field(default_factory=dict)
    matches: dict[str, EventMatch] = field(default_factory=dict)


def evaluate_clip(cd: ClipData, cfg: Config, dets: dict[int, list[Detection]], eval_cfg: dict, root: Path,
                  trackers: tuple[str, ...] = TRACKERS, out_root: Path | None = None) -> ClipResult:
    stride = cfg.pipeline.frame_stride
    frames = list(range(1, cd.n_frames + 1, stride))
    det_m = det_eval.evaluate_detections(cd.gt, cd.ignore, dets, cfg.detector.score_threshold, frames,
                                         eval_cfg["detection"]["iou"])
    res = ClipResult(cd, det_m)
    for kind in trackers:
        c = copy.deepcopy(cfg)
        c.tracker = copy.deepcopy(cfg.baseline_tracker if kind == "iou" else cfg.tracker)
        c.tracker.kind = kind
        run = run_clip(root / cd.clip.source, c, cd.corridor, cached=dets, fps_override=cd.clip.fps, info=cd.info,
                       out_dir=(out_root / kind / cd.clip.id) if out_root else None)
        res.runs[kind] = run
        res.accs[kind] = trk_eval.build_accumulator(cd.gt, cd.ignore, run.mot_rows, frames, eval_cfg["tracking"]["iou"])
        res.matches[kind] = match_events(cd.gt_events, run.events)
    return res


def aggregate(results: list[ClipResult], trackers: tuple[str, ...] = TRACKERS) -> dict:
    """Pool over clips: overall and per condition tag. Counts are pooled, never averaged ratios."""
    tags = sorted({t for r in results for t in r.cd.tags})
    groups = {"ALL": results, **{t: [r for r in results if t in r.cd.tags] for t in tags}}
    out: dict = {"n_clips": len(results), "clips": [r.cd.clip.id for r in results],
                 "detection": {}, "tracking": {k: {} for k in trackers}, "events": {k: {} for k in trackers},
                 "safety": {}}
    for gname, rs in groups.items():
        tp = sum(r.det_counts["tp"] for r in rs)
        fp = sum(r.det_counts["fp"] for r in rs)
        fn = sum(r.det_counts["fn"] for r in rs)
        out["detection"][gname] = {**det_eval.counts_to_metrics(tp, fp, fn), "n_clips": len(rs)}
        for k in trackers:
            summ = trk_eval.summarize({r.cd.clip.id: r.accs[k] for r in rs}, with_overall=True)
            out["tracking"][k][gname] = {**summ.get("OVERALL", {}), "n_clips": len(rs)}
            if k == trackers[-1]:
                out["safety"][gname] = safety_metrics(rs, k)
            out["events"][k][gname] = {
                **{kk: v for kk, v in event_metrics([r.matches[k] for r in rs], [r.cd.clip.negative for r in rs],
                                                    [r.cd.info.frame_count_reported / r.cd.info.fps if r.cd.info.fps else 0.0 for r in rs]).items()
                   if kk != "latency_frames_all"},
                "n_clips": len(rs),
            }
    return out


def per_clip_details(r: ClipResult, cfg: Config, dets: dict[int, list[Detection]], eval_cfg: dict, tracker: str) -> dict:
    """Everything visual error analysis needs, in machine-readable form."""
    run, m = r.runs[tracker], r.matches[tracker]
    acc = r.accs[tracker]
    switches = []
    ev = acc.mot_events
    if len(ev):
        sw = ev[ev["Type"] == "SWITCH"]
        switches = [{"frame": int(i[0]), "gt_id": int(row["OId"]), "track_id": int(row["HId"])} for i, row in sw.iterrows()]
    frames = list(range(1, r.cd.n_frames + 1, cfg.pipeline.frame_stride))
    errs = det_eval.detection_errors(r.cd.gt, r.cd.ignore, dets, cfg.detector.score_threshold, frames, eval_cfg["detection"]["iou"])
    return {
        "clip": r.cd.clip.id, "tags": r.cd.tags, "negative": r.cd.clip.negative, "tracker": tracker,
        "detection": r.det_counts,
        "id_switches": switches,
        "events_predicted": [e.to_dict() for e in run.events],
        "events_gt": [g.__dict__ for g in r.cd.gt_events],
        "event_pairs": m.pairs, "false_events": m.false_events, "missed_events": m.missed_events,
        "event_latencies_frames": m.latencies,
        "false_positives": errs["false_positives"][:400], "misses": errs["misses"][:400],
        "final_health": run.final_health, "health_frame_counts": run.health_counts,
        "corridor_frame_counts": run.corridor_counts,
    }


def safety_metrics(results: list[ClipResult], tracker: str) -> dict:
    """The number that matters most for an advisory system: how often did it say CLEAR
    while a ground-truth person was actually inside the corridor?

    occupied frame = a GT person's reference point is inside the polygon on that frame.
    """
    from sightline.geometry import box_ref_point

    occupied = false_clear = unknown = occupied_says_occupied = 0
    non_ok = total = 0
    for r in results:
        run = r.runs[tracker]
        poly = r.cd.corridor.polygon
        for f, status in run.corridor_by_frame.items():
            total += 1
            if any(poly.contains(box_ref_point(b)) for _i, b in r.cd.gt.get(f, [])):
                occupied += 1
                if status == "CLEAR":
                    false_clear += 1
                elif status == "UNKNOWN":
                    unknown += 1
                else:
                    occupied_says_occupied += 1
        non_ok += total_non_ok(run)
    return {
        "occupied_frames": occupied,
        "false_clear_frames": false_clear,
        "false_clear_rate": (false_clear / occupied) if occupied else None,
        "occupied_reported_unknown_frames": unknown,
        "occupied_reported_occupied_frames": occupied_says_occupied,
        "non_ok_frame_fraction": (non_ok / total) if total else 0.0,
        "note": "false clear = system said CLEAR while a GT person's feet were inside the corridor",
    }


def total_non_ok(run: ClipRun) -> int:
    return sum(v for k, v in run.health_counts.items() if k != "OK")


def bar_check(agg: dict, coverage: dict, eval_cfg: dict, tracker: str = "bytetrack", baseline: str = "iou",
              runtime: dict | None = None) -> dict:
    """The acceptance bar from the brief, evaluated mechanically from saved numbers."""
    b = eval_cfg["bar"]
    det, trk, evn = agg["detection"]["ALL"], agg["tracking"][tracker]["ALL"], agg["events"][tracker]["ALL"]
    base = agg["tracking"][baseline]["ALL"]
    idf1, base_idf1 = trk.get("idf1"), base.get("idf1")
    idsw, base_idsw = trk.get("num_switches"), base.get("num_switches")
    beats = ((idf1 is not None and base_idf1 is not None and idf1 > base_idf1)
             or (idsw is not None and base_idsw is not None and idsw < base_idsw))

    def ge(v, t):
        return v is not None and v >= t

    checks = {
        "heldout_clips>=12": coverage["meets_min_clips"],
        "three_difficult_conditions": coverage["meets_three_difficult"],
        "negative_corridor_set": coverage["has_negative_set"],
        "detection_precision>=0.80": ge(det["precision"], b["detection_precision"]),
        "detection_recall>=0.80": ge(det["recall"], b["detection_recall"]),
        "idf1>=0.65": ge(idf1, b["idf1"]),
        "tracker_beats_baseline": bool(beats),
        "event_precision>=0.85": ge(evn["event_precision"], b["event_precision"]),
        "event_recall>=0.85": ge(evn["event_recall"], b["event_recall"]),
        "false_events_per_5min_negative<=1": (evn["false_events_per_5min_negative"] is not None
                                              and evn["false_events_per_5min_negative"] <= b["max_false_events_per_5min_negative"]),
        "median_event_latency<=5_frames": (evn["latency_frames_median"] is not None
                                           and evn["latency_frames_median"] <= b["median_event_latency_frames"]),
    }
    if runtime is not None:
        fps = runtime.get("throughput_fps")
        checks["throughput>=10fps"] = fps is not None and fps >= b["min_fps"]
    else:
        checks["throughput>=10fps"] = None  # not measured yet: run `benchmark` on the demo machine
    return {"checks": checks, "all_pass": all(v is True for v in checks.values()),
            "not_yet_measured": [k for k, v in checks.items() if v is None],
            "note": "Mechanical check of the numeric bar only. Degraded-state behaviour, tests, "
                    "reproducibility and visual error analysis are reviewed separately."}


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))

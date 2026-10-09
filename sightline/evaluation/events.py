"""Event ground truth, matching, and metrics. All event metrics are per EVENT.

One person's frames are never counted as independent evidence: a person who is
inside the corridor for 200 frames is one event.

Ground-truth rule constants live in configs/eval.yaml. They define what counts
as an event in the labels and are NEVER tuned against the system's results.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from sightline.events import Event
from sightline.evaluation.gt import GTFrames
from sightline.geometry import Polygon, box_ref_point


@dataclass(frozen=True)
class GTEvent:
    id: int
    first_inside_frame: int
    last_inside_frame: int


def derive_gt_events(gt: GTFrames, polygon: Polygon, min_frames: int = 3, merge_gap: int = 5) -> list[GTEvent]:
    """Event labels from ground-truth boxes: a person's reference point inside the polygon.

    Runs separated by <= merge_gap frames are one event (boundary jitter); runs with
    fewer than min_frames inside frames are not events. Used for MOT17, where no
    corridor labels exist. Those labels are RULE-DERIVED, not independently human-labelled.
    """
    inside_frames: dict[int, list[int]] = {}
    for f in sorted(gt):
        for tid, box in gt[f]:
            if polygon.contains(box_ref_point(box)):
                inside_frames.setdefault(tid, []).append(f)
    events: list[GTEvent] = []
    next_id = 1
    for tid in sorted(inside_frames):
        frames = inside_frames[tid]
        run = [frames[0]]
        runs = []
        for f in frames[1:]:
            if f - run[-1] - 1 <= merge_gap:
                run.append(f)
            else:
                runs.append(run)
                run = [f]
        runs.append(run)
        for r in runs:
            if len(r) >= min_frames:
                events.append(GTEvent(next_id, r[0], r[-1]))
                next_id += 1
    return events


def load_gt_events(path: str | Path) -> list[GTEvent]:
    """Hand-labelled events: [{"first_inside_frame": int, "last_inside_frame": int}, ...]."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"event labels not found: {p}")
    rows = json.loads(p.read_text())
    return [GTEvent(int(r.get("id", i + 1)), int(r["first_inside_frame"]), int(r["last_inside_frame"]))
            for i, r in enumerate(rows)]


@dataclass
class EventMatch:
    tp: int
    fp: int
    fn: int
    latencies: list[int]  # frames, predicted open - GT first inside
    unresolved: int
    pairs: list[tuple[int, int]]  # (gt event id, predicted event id)
    false_events: list[int]  # predicted event ids with no GT match
    missed_events: list[int]  # GT event ids with no match


def match_events(gt_events: list[GTEvent], pred: list[Event]) -> EventMatch:
    """One-to-one greedy matching by temporal overlap (at least one shared frame).

    A second predicted event overlapping an already-matched GT event is a duplicate
    and counts as a false event, which is exactly the failure this metric must expose.
    """
    cands = []
    for g in gt_events:
        for p in pred:
            lo = max(g.first_inside_frame, p.first_inside_frame)
            hi = min(g.last_inside_frame, p.last_inside_frame)
            if hi >= lo:
                cands.append((hi - lo + 1, g.id, p.event_id))
    cands.sort(key=lambda c: -c[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs = []
    for _ov, gid, pid in cands:
        if gid in used_g or pid in used_p:
            continue
        used_g.add(gid)
        used_p.add(pid)
        pairs.append((gid, pid))
    by_g = {g.id: g for g in gt_events}
    by_p = {p.event_id: p for p in pred}
    lat = [by_p[pid].open_frame - by_g[gid].first_inside_frame for gid, pid in pairs]
    return EventMatch(
        tp=len(pairs), fp=len(pred) - len(pairs), fn=len(gt_events) - len(pairs), latencies=lat,
        unresolved=sum(1 for p in pred if p.status.value == "UNRESOLVED"), pairs=pairs,
        false_events=[p.event_id for p in pred if p.event_id not in used_p],
        missed_events=[g.id for g in gt_events if g.id not in used_g],
    )


def event_metrics(matches: list[EventMatch], negative_flags: list[bool], durations_s: list[float]) -> dict:
    """Pool matches across clips.

    false_events_per_5min is computed on NEGATIVE clips only (people visible, corridor never entered).
    """
    tp = sum(m.tp for m in matches)
    fp = sum(m.fp for m in matches)
    fn = sum(m.fn for m in matches)
    lat = [x for m in matches for x in m.latencies]
    neg_fp = sum(m.fp for m, neg in zip(matches, negative_flags) if neg)
    neg_seconds = sum(d for d, neg in zip(durations_s, negative_flags) if neg)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {
        "events_tp": tp, "events_fp": fp, "events_fn": fn,
        "event_precision": p, "event_recall": r,
        "latency_frames_median": statistics.median(lat) if lat else None,
        "latency_frames_p95": _percentile(lat, 95) if lat else None,
        "latency_frames_all": lat,
        "events_unresolved": sum(m.unresolved for m in matches),
        "negative_clip_seconds": neg_seconds,
        "negative_clip_false_events": neg_fp,
        "false_events_per_5min_negative": (neg_fp / (neg_seconds / 300.0)) if neg_seconds > 0 else None,
    }


def _percentile(values: list[int], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

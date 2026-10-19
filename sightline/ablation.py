"""Tracker ablation: switch off one part of the chosen tracker at a time, on the same cached detections.

What ByteTrack-style tracking adds over the IoU baseline is two things: the Kalman motion model
(where a track is expected on the next frame) and the second association on low-score detections
(keeping a partly hidden person's identity alive). Removing each one, and both, shows which of them
is responsible for any difference in IDF1, ID switches and the event metrics.

Every variant uses the FROZEN config and the same cached detections; only the named switch changes.
This is analysis after the fact: no setting is chosen from it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

from sightline.config import Config
from sightline.evaluation.run import aggregate, evaluate_clip, get_detections, prepare_clip
from sightline.manifest import load_manifest

VARIANTS: dict[str, dict] = {
    "bytetrack (shipped)": {},
    "no low-score stage": {"use_low_stage": False},
    "no motion model": {"use_motion": False},
    "neither (no motion, no low-score stage)": {"use_low_stage": False, "use_motion": False},
    "IoU baseline": {"baseline": True},
}


def _row(agg: dict, kind: str) -> dict:
    trk = agg["tracking"][kind].get("ALL", {})
    trk_in = agg["tracking_in_scope"][kind].get("ALL", {})
    ev = agg["events"][kind].get("ALL", {})
    return {"idf1": trk.get("idf1"), "id_switches": trk.get("num_switches"), "mota": trk.get("mota"),
            "idf1_in_scope": trk_in.get("idf1"), "id_switches_in_scope": trk_in.get("num_switches"),
            "event_precision": ev.get("event_precision"), "event_recall": ev.get("event_recall"),
            "events_tp": ev.get("events_tp"), "events_fp": ev.get("events_fp"), "events_fn": ev.get("events_fn"),
            "latency_frames_median": ev.get("latency_frames_median"), "events_unresolved": ev.get("events_unresolved"),
            "negative_clip_false_events": ev.get("negative_clip_false_events"),
            "n_clips_tracking": trk.get("n_clips"), "n_clips_events": ev.get("n_clips")}


def run_ablation(manifest_path: Path, cfg: Config, eval_cfg: dict, root: Path, cache_dir: Path,
                 progress: Callable[[str], object] = print) -> dict:
    man = load_manifest(manifest_path)
    cds = [prepare_clip(c, root, eval_cfg) for c in man.clips]
    by_source: dict[str, dict] = {}  # cached only, never a model; entries sharing footage share detections
    for cd in cds:
        if cd.clip.source not in by_source:
            by_source[cd.clip.source] = get_detections(cd, cfg, root, cache_dir, None)
    dets = {cd.clip.id: by_source[cd.clip.source] for cd in cds}
    rows = {}
    for name, sw in VARIANTS.items():
        c = copy.deepcopy(cfg)
        kind = "iou" if sw.get("baseline") else "bytetrack"
        for k, v in sw.items():
            if k != "baseline":
                setattr(c.tracker, k, v)
        res = [evaluate_clip(cd, c, dets[cd.clip.id], eval_cfg, root, trackers=(kind,)) for cd in cds]
        rows[name] = {"switches": sw, **_row(aggregate(res, trackers=(kind,)), kind)}
        r = rows[name]
        progress(f"{name}: IDF1 {r['idf1']}, ID switches {r['id_switches']}, event P {r['event_precision']} R {r['event_recall']}")
    return {"what": "tracker ablation on cached detections with the frozen config; analysis only, nothing chosen from it",
            "manifest": manifest_path.as_posix(), "clips": [c.id for c in man.clips], "variants": rows}

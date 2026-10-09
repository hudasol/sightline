"""Visual error analysis support.

This module SELECTS failures deterministically from saved results (worst first, not
hand-picked) and RENDERS them. It does not explain them: the "why it happened"
column is for a human who has looked at the frames. A script cannot honestly
write that.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from sightline.detector import read_detection_cache
from sightline.evaluation.gt import load_mot_gt
from sightline.ingest import probe, read_frame
from sightline.manifest import load_manifest


def _duplicate_frames(gt, tracks_txt: Path, iou_thr: float = 0.5) -> list[dict]:
    """Frames where two different track ids both overlap one ground-truth person."""
    import numpy as np

    from sightline.geometry import iou_matrix

    if not tracks_txt.is_file():
        return []
    hyp: dict[int, list] = {}
    for line in tracks_txt.read_text().splitlines():
        f, tid, x, y, w, h, *_ = line.split(",")
        hyp.setdefault(int(f), []).append((int(tid), (float(x), float(y), float(x) + float(w), float(y) + float(h))))
    out = []
    for f, g in gt.items():
        h = hyp.get(f, [])
        if len(h) < 2 or not g:
            continue
        iou = iou_matrix(np.array([b for _i, b in g]), np.array([b for _i, b in h]))
        for gi, (gid, gbox) in enumerate(g):
            ids = [h[j][0] for j in range(len(h)) if iou[gi, j] >= iou_thr]
            if len(set(ids)) >= 2:
                out.append({"frame": f, "gt_id": gid, "track_ids": sorted(set(ids)), "box": list(gbox)})
    return out


def select_failures(per_clip_dir: Path, results_dir: Path, per_type: int = 2) -> list[dict]:
    items: list[dict] = []
    fps, misses, switches, late, false_ev = [], [], [], [], []
    for f in sorted((per_clip_dir / "bytetrack").glob("*.json")):
        d = json.loads(f.read_text())
        clip = d["clip"]
        fps += [{"type": "false_detection", "clip": clip, **x} for x in d["false_positives"]]
        misses += [{"type": "missed_person", "clip": clip, **x} for x in d["misses"]]
        switches += [{"type": "id_switch", "clip": clip, **x} for x in d["id_switches"]]
        by_p = {e["event_id"]: e for e in d["events_predicted"]}
        gt_by = {g["id"]: g for g in d["events_gt"]}
        for gid, pid in d["event_pairs"]:
            lat = by_p[pid]["open_frame"] - gt_by[gid]["first_inside_frame"]
            if lat > 5:
                late.append({"type": "late_crossing", "clip": clip, "frame": by_p[pid]["open_frame"], "latency_frames": lat,
                             "gt_first_inside": gt_by[gid]["first_inside_frame"]})
        false_ev += [{"type": "false_event", "clip": clip, "frame": by_p[pid]["open_frame"], "event": by_p[pid]}
                     for pid in d["false_events"] if pid in by_p]
    fps.sort(key=lambda x: -x["score"])
    misses.sort(key=lambda x: -((x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1])))
    late.sort(key=lambda x: -x["latency_frames"])
    for group in (fps, misses, switches, late, false_ev):
        items += group[:per_type]
    return items


def render_failure(item: dict, root: Path, lock_path: Path, cache_dir: Path, detector_name: str,
                   threshold: float, tracks_root: Path, out_png: Path) -> bool:
    man = load_manifest(lock_path)
    clip = next(c for c in man.clips if c.id == item["clip"])
    src = root / clip.source
    info = probe(src, clip.fps)
    img = read_frame(src, item["frame"], info, clip.fps)
    if img is None:
        return False
    gt, _ig = load_mot_gt(root / clip.gt)
    for gid, (x1, y1, x2, y2) in gt.get(item["frame"], []):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (80, 190, 80), 2)
    cache = cache_dir / detector_name / f"{clip.id}.jsonl"
    if cache.is_file():
        _m, dets = read_detection_cache(cache)
        for d in dets.get(item["frame"], []):
            col = (170, 170, 170) if d.score >= threshold else (110, 110, 110)
            cv2.rectangle(img, (int(d.box[0]), int(d.box[1])), (int(d.box[2]), int(d.box[3])), col, 1)
    tracks = tracks_root / "bytetrack" / clip.id / "tracks.txt"
    if tracks.is_file():
        for line in tracks.read_text().splitlines():
            f, tid, x, y, w, h, *_ = line.split(",")
            if int(f) == item["frame"]:
                cv2.rectangle(img, (int(float(x)), int(float(y))), (int(float(x) + float(w)), int(float(y) + float(h))), (255, 170, 0), 2)
                cv2.putText(img, f"#{tid}", (int(float(x)), max(12, int(float(y)) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 0), 1)
    if "box" in item:
        b = [int(v) for v in item["box"]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
    cv2.putText(img, f"{item['type']} | {clip.id} | frame {item['frame']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return True


def analyze(results_dir: Path, root: Path, lock_path: Path, cache_dir: Path, detector_name: str, threshold: float,
            per_type: int = 2) -> Path:
    out_dir = results_dir / "error_analysis"
    items = select_failures(results_dir / "per_clip", results_dir, per_type)
    # duplicate-track candidates need GT and the saved tracks
    man = load_manifest(lock_path)
    dups = []
    for c in man.clips:
        gt, _ig = load_mot_gt(root / c.gt)
        for d in _duplicate_frames(gt, results_dir / "tracks" / "bytetrack" / c.id / "tracks.txt")[:per_type]:
            dups.append({"type": "duplicate_track", "clip": c.id, **d})
    items += dups[:per_type]
    rows = []
    for n, it in enumerate(items, start=1):
        png = out_dir / f"{n:02d}_{it['type']}_{it['clip']}_f{it['frame']}.png"
        rendered = render_failure(it, root, lock_path, cache_dir, detector_name, threshold, results_dir / "tracks", png)
        rows.append({**{k: v for k, v in it.items() if k not in ("event",)}, "image": png.name if rendered else None})
    (out_dir / "index.json").write_text(json.dumps(rows, indent=1, default=str))
    md = ["# Error analysis index (auto-selected, worst first)", "",
          "Legend: green = ground truth, grey = detections, orange = tracks with ids, red = the failure.",
          "The **Why it happened** column is for a human who has looked at the frame.", "",
          "| # | Type | Clip | Frame | Measured | Why it happened |", "|---|---|---|---|---|---|"]
    for n, r in enumerate(rows, start=1):
        measured = ", ".join(f"{k}={v}" for k, v in r.items() if k in ("score", "latency_frames", "gt_id", "track_id", "track_ids", "gt_first_inside"))
        md.append(f"| {n} | {r['type']} | {r['clip']} | {r['frame']} | {measured} | TODO(human): look at `{r['image']}` |")
    (out_dir / "INDEX.md").write_text("\n".join(md) + "\n")
    return out_dir

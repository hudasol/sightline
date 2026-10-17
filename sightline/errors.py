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


def select_failures(per_clip_dir: Path, results_dir: Path, per_type: int = 2, scored: dict | None = None) -> list[dict]:
    """scored: clip id -> the metric families it counts for. A failure is only taken from a clip scored on
    its family, so a second corridor on the same footage (a negative entry) adds no repeated detections."""
    items: list[dict] = []
    fps, misses, switches, late, false_ev = [], [], [], [], []
    for f in sorted((per_clip_dir / "bytetrack").glob("*.json")):
        d = json.loads(f.read_text())
        clip = d["clip"]
        fams = scored.get(clip, ()) if scored is not None else ("detection", "tracking", "events")
        if "detection" in fams:
            fps += [{"type": "false_detection", "clip": clip, **x} for x in d["false_positives"]]
            misses += [{"type": "missed_person", "clip": clip, **x} for x in d["misses"]]
        if "tracking" in fams:
            switches += [{"type": "id_switch", "clip": clip, **x} for x in d["id_switches"]]
        if "events" not in fams:
            continue
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


def render_failure(item: dict, root: Path, lock_path: Path, cache_dir: Path, detector_tag: str,
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
    cache = cache_dir / detector_tag / f"{clip.id}.jsonl"
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


def analyze(results_dir: Path, root: Path, lock_path: Path, cache_dir: Path, detector_tag: str, threshold: float,
            per_type: int = 2) -> Path:
    out_dir = results_dir / "error_analysis" / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = load_manifest(lock_path)
    items = select_failures(results_dir / "per_clip", results_dir, per_type, {c.id: c.metrics for c in man.clips})
    # duplicate-track candidates need GT and the saved tracks
    dups = []
    for c in man.clips:
        if not c.scored_on("tracking"):
            continue
        gt, _ig = load_mot_gt(root / c.gt)
        for d in _duplicate_frames(gt, results_dir / "tracks" / "bytetrack" / c.id / "tracks.txt")[:per_type]:
            dups.append({"type": "duplicate_track", "clip": c.id, **d})
    items += dups[:per_type]
    rows = []
    for n, it in enumerate(items, start=1):
        png = out_dir / f"{n:02d}_{it['type']}_{it['clip']}_f{it['frame']}.png"
        rendered = render_failure(it, root, lock_path, cache_dir, detector_tag, threshold, results_dir / "tracks", png)
        rows.append({**{k: v for k, v in it.items() if k not in ("event",)}, "image": png.name if rendered else None})
    (out_dir / "index.json").write_text(json.dumps(rows, indent=1, default=str))
    md = ["# Error analysis index (auto-selected, worst first)", "",
          "Legend: green = ground truth, grey = detections, light blue = tracks with ids, red = the failure.",
          "The **Why it happened** column is for a human who has looked at the frame.", "",
          "| # | Type | Clip | Frame | Measured | Why it happened |", "|---|---|---|---|---|---|"]
    for n, r in enumerate(rows, start=1):
        measured = ", ".join(f"{k}={v}" for k, v in r.items() if k in ("score", "latency_frames", "gt_id", "track_id", "track_ids", "gt_first_inside"))
        md.append(f"| {n} | {r['type']} | {r['clip']} | {r['frame']} | {measured} | TODO(human): look at `{r['image']}` |")
    (out_dir / "INDEX.md").write_text("\n".join(md) + "\n")
    return out_dir


# -- unresolved events as frame strips --------------------------------------------------------------

UNRESOLVED_RULE = ("UNRESOLVED events whose track was lost while the event was open (close_reason track_lost; "
                   "stream_end is not a failure). Worst first = longest open before the loss (close_frame - "
                   "open_frame), ties broken by clip id then frame. Fixed before any strip was looked at.")


def collect_details(manifest_path: Path, cfg, eval_cfg: dict, root: Path, cache_dir: Path, results_dir: Path) -> list[str]:
    """Per-clip details and track files for a manifest's event clips, from CACHED detections only
    (a missing cache is an error: this never runs a detector). Nothing is tuned or changed."""
    from sightline.evaluation.run import evaluate_clip, get_detections, per_clip_details, prepare_clip, write_json

    done = []
    for c in load_manifest(manifest_path).clips:
        if not c.scored_on("events"):
            continue
        cd = prepare_clip(c, root, eval_cfg)
        dets = get_detections(cd, cfg, root, cache_dir, None)
        r = evaluate_clip(cd, cfg, dets, eval_cfg, root, trackers=("bytetrack",), out_root=results_dir / "tracks")
        write_json(results_dir / "per_clip" / "bytetrack" / f"{c.id}.json", per_clip_details(r, cfg, dets, eval_cfg, "bytetrack"))
        done.append(c.id)
    return done


def _gt_match(d: dict, event_id: int) -> dict:
    """The ground-truth event this predicted event was matched to by the scorer (event_pairs), if any.
    Matching (evaluation/events.py) is the same for every status, UNRESOLVED included."""
    gid = next((g for g, p in d.get("event_pairs", []) if p == event_id), None)
    g = next((x for x in d.get("events_gt", []) if x["id"] == gid), None) if gid is not None else None
    return {"matched_gt_event": gid,
            "matched_gt_frames": [g["first_inside_frame"], g["last_inside_frame"]] if g else None}


def select_unresolved(per_clip_dir: Path, n: int = 10) -> list[dict]:
    items = []
    for f in sorted((per_clip_dir / "bytetrack").glob("*.json")):
        d = json.loads(f.read_text())
        for e in d["events_predicted"]:
            if e["status"] == "UNRESOLVED" and e["close_reason"] == "track_lost":
                items.append({"type": "unresolved_event", "clip": d["clip"], "frame": e["close_frame"],
                              "event_id": e["event_id"], "track_id": e["track_id"], "open_frame": e["open_frame"],
                              "first_inside_frame": e["first_inside_frame"], "last_inside_frame": e["last_inside_frame"],
                              "open_frames_before_loss": e["close_frame"] - e["open_frame"],
                              "evidence": e.get("evidence"), **_gt_match(d, e["event_id"])})
    items.sort(key=lambda x: (-x["open_frames_before_loss"], x["clip"], x["frame"]))
    return items[:n]


def render_strip(item: dict, root: Path, manifest_path: Path, cache_dir: Path, detector_tag: str, threshold: float,
                 tracks_root: Path, out_png: Path, n_frames: int = 6, tile_w: int = 480,
                 span: tuple[int, int] | None = None, title: str | None = None) -> bool:
    """n_frames evenly spaced from 10 frames before the person was last inside to the frame the event went
    UNRESOLVED (or over `span`, first to last frame). Green = GT, grey = detections >= threshold, light blue =
    reported tracks, red = the event's track."""
    import numpy as np

    from sightline.config import load_corridor

    clip = next(c for c in load_manifest(manifest_path).clips if c.id == item["clip"])
    src = root / clip.source
    info = probe(src, clip.fps)
    poly = np.array(load_corridor(root / clip.corridor).polygon.vertices, dtype=np.int32)
    gt, _ig = load_mot_gt(root / clip.gt)
    cache = cache_dir / detector_tag / f"{clip.id}.jsonl"
    dets = read_detection_cache(cache)[1] if cache.is_file() else {}
    tracks: dict[int, list] = {}
    tfile = tracks_root / "bytetrack" / clip.id / "tracks.txt"
    if tfile.is_file():
        for line in tfile.read_text().splitlines():
            f, tid, x, y, w, h, *_ = line.split(",")
            tracks.setdefault(int(f), []).append((int(tid), float(x), float(y), float(x) + float(w), float(y) + float(h)))
    start, end = span or (max(1, min(item["last_inside_frame"], item["frame"]) - 10), item["frame"])
    frames = sorted({int(round(v)) for v in np.linspace(start, end, n_frames)})
    tiles = []
    for fidx in frames:
        img = read_frame(src, fidx, info, clip.fps)
        if img is None:
            continue
        lw = max(2, img.shape[1] // 480)
        cv2.polylines(img, [poly], True, (0, 255, 0), lw)
        for _gid, (x1, y1, x2, y2) in gt.get(fidx, []):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (80, 190, 80), lw)
        for d in dets.get(fidx, []):
            if d.score >= threshold:
                cv2.rectangle(img, (int(d.box[0]), int(d.box[1])), (int(d.box[2]), int(d.box[3])), (170, 170, 170), 1)
        for tid, x1, y1, x2, y2 in tracks.get(fidx, []):
            col = (0, 0, 255) if tid == item["track_id"] else (255, 170, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, lw + (1 if tid == item["track_id"] else 0))
            cv2.putText(img, f"#{tid}", (int(x1), max(14, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        scale = tile_w / img.shape[1]
        tile = cv2.resize(img, (tile_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile_w, 24), (0, 0, 0), -1)
        cv2.putText(tile, f"frame {fidx}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return False
    strip = np.hstack(tiles)
    head = np.zeros((28, strip.shape[1], 3), np.uint8)
    title = title or (f"{item['clip']}  event {item['event_id']} (track #{item['track_id']})  opened {item['open_frame']}, "
                      f"last inside {item['last_inside_frame']}, UNRESOLVED at {item['frame']}")
    cv2.putText(head, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack([head, strip]))
    return True


def analyze_unresolved(results_dir: Path, root: Path, manifest_path: Path, cache_dir: Path, detector_tag: str,
                       threshold: float, n: int, run_note: str) -> Path:
    out_dir = results_dir / "error_analysis" / "unresolved"
    out_dir.mkdir(parents=True, exist_ok=True)  # an index is written even when nothing is unresolved
    items = select_unresolved(results_dir / "per_clip", n)
    rows = []
    for k, it in enumerate(items, start=1):
        png = out_dir / f"{k:02d}_unresolved_{it['clip']}_ev{it['event_id']}_f{it['frame']}.png"
        ok = render_strip(it, root, manifest_path, cache_dir, detector_tag, threshold, results_dir / "tracks", png)
        rows.append({**it, "image": png.name if ok else None})
    (out_dir / "index.json").write_text(json.dumps({"rule": UNRESOLVED_RULE, "run": run_note, "requested": n,
                                                    "selected": len(rows), "items": rows}, indent=1))
    md = ["# Unresolved events (auto-selected, worst first)", "", f"Selection: {UNRESOLVED_RULE}", "", f"Run: {run_note}", "",
          f"Requested {n}; {len(rows)} event(s) met the selection rule.", "",
          "Legend: green polygon = corridor, green boxes = ground truth, grey = detections >= threshold, "
          "light blue = reported tracks, red = the event's track. These frames are not to be published (they show faces).",
          "The **Why it happened** column is for a human who has looked at the strip. No setting was changed from it.", "",
          "| # | Clip | Event | Open frames before loss | Opened / last inside / unresolved | Matched GT event | Why it happened |",
          "|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, start=1):
        gt = (f"{r['matched_gt_event']} (frames {r['matched_gt_frames'][0]}-{r['matched_gt_frames'][1]})"
              if r.get("matched_gt_event") is not None else "none")
        md.append(f"| {k} | {r['clip']} | {r['event_id']} (track #{r['track_id']}) | {r['open_frames_before_loss']} | "
                  f"{r['open_frame']} / {r['last_inside_frame']} / {r['frame']} | {gt} | TODO(human): look at `{r['image']}` |")
    (out_dir / "INDEX.md").write_text("\n".join(md) + "\n")
    return out_dir


# -- false events (predicted events the scorer left unmatched) as frame strips -----------------------

FALSE_EVENT_RULE = ("False events: predicted events left unmatched by the event scorer (false_events in the per-clip "
                    "details; status plays no part, as in matching). Worst first = longest open (close_frame - "
                    "open_frame), ties broken by clip id then frame. Fixed before any strip was looked at.")


def _gt_overlaps(d: dict, e: dict) -> list[dict]:
    """GT events whose inside span overlaps this event's, and which predicted event (if any) the scorer gave each."""
    to_pred = {g: p for g, p in d.get("event_pairs", [])}
    return [{"gt_event": g["id"], "gt_frames": [g["first_inside_frame"], g["last_inside_frame"]],
             "matched_to_event": to_pred.get(g["id"])}
            for g in d.get("events_gt", [])
            if min(e["last_inside_frame"], g["last_inside_frame"]) >= max(e["first_inside_frame"], g["first_inside_frame"])]


def select_false_events(per_clip_dir: Path, n: int = 10) -> list[dict]:
    items = []
    for f in sorted((per_clip_dir / "bytetrack").glob("*.json")):
        d = json.loads(f.read_text())
        by_p = {e["event_id"]: e for e in d["events_predicted"]}
        for pid in d["false_events"]:
            e = by_p[pid]
            items.append({"type": "false_event", "clip": d["clip"], "frame": e["open_frame"], "event_id": pid,
                          "track_id": e["track_id"], "status": e["status"], "close_reason": e["close_reason"],
                          "open_frame": e["open_frame"], "close_frame": e["close_frame"],
                          "first_inside_frame": e["first_inside_frame"], "last_inside_frame": e["last_inside_frame"],
                          "open_frames": e["close_frame"] - e["open_frame"], "evidence": e.get("evidence"),
                          "overlapping_gt_events": _gt_overlaps(d, e)})
    items.sort(key=lambda x: (-x["open_frames"], x["clip"], x["frame"]))
    return items[:n]


def analyze_false_events(results_dir: Path, root: Path, manifest_path: Path, cache_dir: Path, detector_tag: str,
                         threshold: float, n: int, run_note: str) -> Path:
    """Same strip format as analyze_unresolved; each strip runs from 10 frames before the person was first
    inside to the frame the event closed."""
    out_dir = results_dir / "error_analysis" / "false_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = select_false_events(results_dir / "per_clip", n)
    rows = []
    for k, it in enumerate(items, start=1):
        png = out_dir / f"{k:02d}_false_event_{it['clip']}_ev{it['event_id']}_f{it['frame']}.png"
        title = (f"{it['clip']}  FALSE event {it['event_id']} (track #{it['track_id']})  first inside "
                 f"{it['first_inside_frame']}, opened {it['open_frame']}, {it['status']} at {it['close_frame']}")
        ok = render_strip(it, root, manifest_path, cache_dir, detector_tag, threshold, results_dir / "tracks", png,
                          span=(max(1, it["first_inside_frame"] - 10), it["close_frame"]), title=title)
        rows.append({**it, "image": png.name if ok else None})
    (out_dir / "index.json").write_text(json.dumps({"rule": FALSE_EVENT_RULE, "run": run_note, "requested": n,
                                                    "selected": len(rows), "items": rows}, indent=1))
    md = ["# False events (auto-selected, worst first)", "", f"Selection: {FALSE_EVENT_RULE}", "", f"Run: {run_note}", "",
          f"Requested {n}; {len(rows)} event(s) met the selection rule.", "",
          "Legend: green polygon = corridor, green boxes = ground truth, grey = detections >= threshold, "
          "light blue = reported tracks, red = the event's track. These frames are not to be published (they show faces).",
          "\"Overlapping GT events\" lists GT events whose inside span overlaps this one, and the predicted event the "
          "scorer matched each to (matching is one-to-one, so an overlap already taken leaves this one unmatched).",
          "The **Why it happened** column is for a human who has looked at the strip. No setting was changed from it.", "",
          "| # | Clip | Event | Open frames | First inside / opened / closed (status, reason) | Overlapping GT events | Why it happened |",
          "|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, start=1):
        ov = "; ".join(f"{o['gt_event']} (frames {o['gt_frames'][0]}-{o['gt_frames'][1]}, matched to "
                       f"{'event ' + str(o['matched_to_event']) if o['matched_to_event'] is not None else 'nothing'})"
                       for o in r["overlapping_gt_events"]) or "none"
        md.append(f"| {k} | {r['clip']} | {r['event_id']} (track #{r['track_id']}) | {r['open_frames']} | "
                  f"{r['first_inside_frame']} / {r['open_frame']} / {r['close_frame']} ({r['status']}, {r['close_reason']}) | "
                  f"{ov} | TODO(human): look at `{r['image']}` |")
    (out_dir / "INDEX.md").write_text("\n".join(md) + "\n")
    return out_dir

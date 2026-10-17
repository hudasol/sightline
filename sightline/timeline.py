"""One corridor event, end to end, from saved results only (`sightline timeline`).

From the evaluation's per-clip details and its saved tracks: when the event's track was first
reported, when the matched ground-truth person was first inside, when the event opened (the
latency), how long it stayed active and on what evidence, and how it ended (closed or UNRESOLVED).
"""

from __future__ import annotations

import json
from pathlib import Path


def _track_frames(tracks_txt: Path, track_id: int) -> list[int]:
    if not tracks_txt.is_file():
        return []
    return sorted(int(line.split(",")[0]) for line in tracks_txt.read_text().splitlines()
                  if line and int(line.split(",")[1]) == track_id)


def list_events(results: Path, clip: str, tracker: str = "bytetrack") -> list[dict]:
    d = json.loads((results / "per_clip" / tracker / f"{clip}.json").read_text())
    gt_of = {p: g for g, p in d.get("event_pairs", [])}
    gt_by = {g["id"]: g for g in d.get("events_gt", [])}
    out = []
    for e in d["events_predicted"]:
        g = gt_by.get(gt_of.get(e["event_id"]))
        out.append({"event_id": e["event_id"], "track_id": e["track_id"], "status": e["status"],
                    "matched_gt_event": g["id"] if g else None,
                    "latency_frames": (e["open_frame"] - g["first_inside_frame"]) if g else None})
    return out


def timeline(results: Path, clip: str, event_id: int, fps: float, tracker: str = "bytetrack") -> list[str]:
    d = json.loads((results / "per_clip" / tracker / f"{clip}.json").read_text())
    e = next((x for x in d["events_predicted"] if x["event_id"] == event_id), None)
    if e is None:
        raise KeyError(f"{clip} has no predicted event {event_id}")
    gid = next((g for g, p in d.get("event_pairs", []) if p == event_id), None)
    g = next((x for x in d.get("events_gt", []) if x["id"] == gid), None)
    frames = _track_frames(results / "tracks" / tracker / clip / "tracks.txt", e["track_id"])

    def t(f):
        return f"frame {f:>5} ({(f - 1) / fps:6.2f} s)"

    ev = e.get("evidence") or {}
    lines = [f"{clip}, event {event_id}, track #{e['track_id']}, {fps:g} fps", ""]
    if frames:
        lines.append(f"{t(frames[0])}  track #{e['track_id']} first reported (approach)")
    if g:
        lines.append(f"{t(g['first_inside_frame'])}  ground truth: the person's feet are first inside (GT event {g['id']})")
    lines.append(f"{t(e['first_inside_frame'])}  the track's reference point is first inside")
    lat = (e["open_frame"] - g["first_inside_frame"]) if g else None
    lines.append(f"{t(e['open_frame'])}  event OPENED (debounce passed)"
                 + (f"; latency {lat} frames = {lat / fps:.2f} s after the GT first-inside frame" if g else
                    "; no GT event matched: this is a false event"))
    lines.append(f"{'':27}  active: {ev.get('inside_detected', '?')} inside frames on a detection, "
                 f"{ev.get('inside_predicted', '?')} on a predicted box; opened on prediction: {ev.get('opened_on_prediction')}")
    lines.append(f"{t(e['last_inside_frame'])}  last frame the reference point was inside")
    if g:
        lines.append(f"{t(g['last_inside_frame'])}  ground truth: last frame inside")
    end = e.get("close_frame")
    if e["status"] == "UNRESOLVED":
        lines.append(f"{t(end)}  UNRESOLVED ({e['close_reason']}): the track was lost while the event was open, "
                     "so the system does not claim the person left")
    elif end is not None:
        lines.append(f"{t(end)}  CLOSED ({e['close_reason']}), type {e.get('type')}, exit edge {e.get('exit_edge')}")
    else:
        lines.append(f"{'':27}  still {e['status']} when the clip ended")
    if frames:
        lines.append(f"{t(frames[-1])}  track #{e['track_id']} last reported")
    return lines

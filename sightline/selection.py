"""Detector selection table, built only from saved results.

Reads results/detector_selection/detect_eval_val_<tag>.json (P/R/F1 vs threshold on val)
and results/runtime/candidates/<tag>/*.json (end-to-end timing on the same val scenes),
then applies PLAN.md section 5: pick the FASTEST candidate that reaches precision and
recall >= the bar on val at some threshold AND sustains >= min_fps. If none qualifies,
say so: the fallback is a documented detection interval (PLAN.md A2), not a lower bar.
"""

from __future__ import annotations

import json
from pathlib import Path


def _timing(run_dir: Path) -> dict | None:
    files = sorted(run_dir.glob("*.json"))
    if not files:
        return None
    frames = wall = 0.0
    infer = []
    for f in files:
        j = json.loads(f.read_text())
        for c in j["per_clip"]:
            frames += c["frames_processed"]
            wall += c["wall_seconds"]
            if c.get("inference_ms_p50") is not None:
                infer.append((c["inference_ms_p50"], c["frames_processed"]))
    tot = sum(n for _v, n in infer)
    return {"fps": frames / wall if wall else None, "frames_timed": int(frames),
            "inference_ms_p50_frame_weighted": sum(v * n for v, n in infer) / tot if tot else None,
            "max_frames_per_clip": json.loads(files[0].read_text()).get("max_frames_per_clip"),
            "files": [str(f.as_posix()) for f in files]}


VIEWS = {"in_scope": "overall_in_scope", "all_boxes": "overall"}


def _view(overall: list[dict], bar_p: float, bar_r: float, default_threshold: float) -> dict:
    return {"best_f1": max(overall, key=lambda r: r["f1"]),
            "at_default_threshold": next((r for r in overall if abs(r["threshold"] - default_threshold) < 1e-9), None),
            "max_recall": max(r["recall"] for r in overall),
            "thresholds_meeting_pr_bar": [r["threshold"] for r in overall if r["precision"] >= bar_p and r["recall"] >= bar_r]}


def build_table(results: Path, tags: list[str], bar_p: float, bar_r: float, min_fps: float,
                default_threshold: float) -> dict:
    rows = []
    for tag in tags:
        de = results / "detector_selection" / f"detect_eval_val_{tag}.json"
        row: dict = {"candidate": tag, "timing": _timing(results / "runtime" / "candidates" / tag)}
        if de.is_file():
            d = json.loads(de.read_text())
            row.update({"detector": d["detector"], "input_size": d["input_size"], "frames_evaluated": d["frames_evaluated"],
                        "detect_eval_file": str(de.as_posix()), "in_scope_rule": d.get("in_scope_rule")})
            for view, key in VIEWS.items():
                if key in d:
                    row[view] = _view(d[key], bar_p, bar_r, default_threshold)
            # the bar's population is in scope (PLAN.md amendment A1); older files only have all boxes
            judged = row.get("in_scope") or row.get("all_boxes")
            row["judged_view"] = "in_scope" if "in_scope" in row else "all_boxes"
            row["thresholds_meeting_pr_bar"] = judged["thresholds_meeting_pr_bar"]
        else:
            row["detect_eval_file"] = None
            row["pending"] = "no detect-eval result saved yet"
        fps = (row["timing"] or {}).get("fps")
        row["meets_pr_bar"] = bool(row.get("thresholds_meeting_pr_bar")) if "pending" not in row else None
        row["meets_fps_bar"] = (fps >= min_fps) if fps is not None else None
        rows.append(row)
    qualifying = [r for r in rows if r["meets_pr_bar"] and r["meets_fps_bar"]]
    choice = max(qualifying, key=lambda r: r["timing"]["fps"])["candidate"] if qualifying else None
    return {"what": "detector candidates on VAL: detection quality vs end-to-end speed; built from saved files only",
            "rule": (f"fastest candidate with P>={bar_p} and R>={bar_r} at some threshold on val (in-scope boxes) AND "
                     f">={min_fps} fps (PLAN.md s5, A1). Val holds only MOT17 so far: the bar itself is judged on held-out "
                     "own corridor clips."),
            "rows": rows, "choice": choice,
            "verdict": (f"chosen: {choice}" if choice else
                        "NO candidate meets the rule with the detector on every frame; per PLAN.md s5 and A2 choose a "
                        "detection interval and measure its effect on fps, event recall and latency instead of "
                        "lowering the bar")}


def to_markdown(t: dict) -> str:
    def f(v, nd=3):
        return "n/a" if v is None else f"{v:.{nd}f}"

    out = []
    for view, title in (("in_scope", "In scope (judged view)"), ("all_boxes", "All boxes")):
        lines = [f"### {title}", "",
                 "| Candidate | Frames | Thr @ best F1 | P | R | F1 | P / R @ 0.50 | Max R (any thr) | P,R >= bar? | fps (end to end) | Infer p50 ms |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in t["rows"]:
            tm = r["timing"] or {}
            v = r.get(view)
            if v is None:
                why = "pending" if "pending" in r else "not saved"
                lines.append(f"| {r['candidate']} | {why} | | | | | | | | {f(tm.get('fps'), 2)} | {f(tm.get('inference_ms_p50_frame_weighted'), 0)} |")
                continue
            b, d = v["best_f1"], v["at_default_threshold"] or {}
            lines.append(f"| {r['candidate']} | {r['frames_evaluated']} | {b['threshold']:.2f} | {f(b['precision'])} | {f(b['recall'])} | "
                         f"{f(b['f1'])} | {f(d.get('precision'))} / {f(d.get('recall'))} | {f(v['max_recall'])} | "
                         f"{'yes' if v['thresholds_meeting_pr_bar'] else 'no'} | {f(tm.get('fps'), 2)} | "
                         f"{f(tm.get('inference_ms_p50_frame_weighted'), 0)} |")
        out.append("\n".join(lines))
    return "\n\n".join(out) + f"\n\n{t['rule']}\n\n**{t['verdict']}**\n"


# -- PLAN.md amendment A4 (applied on MOT17 val under A6.8) -------------------------------------------

A4_CANDIDATES = ("mbv3_320fpn_640x360", "mbv3_320fpn_960x540", "ssdlite320_320x320", "yolox_nano_416x416",
                 "yolox_tiny_416x416", "yolox_s_640x640")  # fixed by A4.1; ResNet50-FPN-v2 is a reference only
A4_INTERVALS = (1, 2, 3, 4, 6)


def select_a4(results: Path, runtime_summary: Path, min_fps: float,
              candidates: tuple[str, ...] = A4_CANDIDATES, intervals: tuple[int, ...] = A4_INTERVALS) -> dict:
    """A4: the qualifying (detector, k) pair with the highest in-scope F1; for one detector the smallest k;
    an F1 tie at three decimals goes to the faster median. A pair qualifies when the median of its 3
    synchronous offline runs is >= min_fps. Built only from saved files.

    Not every pair has to be timed: a pair whose F1 is below the chosen one cannot win, measured or not.
    The result says which unmeasured pairs could still have changed the choice (it must be none)."""
    runtime = json.loads(runtime_summary.read_text()) if runtime_summary.is_file() else {}
    f1: dict[str, dict] = {}
    for tag in candidates:
        de = results / "detector_selection" / f"detect_eval_val_{tag}.json"
        if de.is_file():
            best = max(json.loads(de.read_text())["overall_in_scope"], key=lambda r: r["f1"])
            f1[tag] = {"f1": round(best["f1"], 3), "threshold": best["threshold"], "precision": best["precision"],
                       "recall": best["recall"], "file": de.as_posix()}
    pairs = []
    for tag in candidates:
        for k in intervals:
            r = runtime.get(f"{tag}_k{k}")
            fps = r["throughput_fps"] if r else None
            pairs.append({"detector": tag, "k": k, "in_scope_f1": (f1.get(tag) or {}).get("f1"),
                          "measured": r is not None, "runs": r.get("runs") if r else None,
                          "throughput_fps_median": fps["median"] if fps else None,
                          "throughput_fps_range": [fps["min"], fps["max"]] if fps else None,
                          "worst_clip_latency_ms_p95_median": (r.get("worst_clip_latency_ms_p95") or {}).get("median") if r else None,
                          "latency_ms_max_median": (r.get("latency_ms_max") or {}).get("median") if r else None,
                          "qualifies": (fps["median"] >= min_fps) if fps else None})
    qual = [p for p in pairs if p["qualifies"] and p["in_scope_f1"] is not None]
    choice = None
    if qual:
        choice = max(qual, key=lambda p: (p["in_scope_f1"], p["throughput_fps_median"]))  # F1, then the faster
        same = [p for p in qual if p["detector"] == choice["detector"]]
        choice = min(same, key=lambda p: p["k"])  # A4.4: smallest qualifying k for the chosen detector
    open_pairs = []
    if choice is not None:
        open_pairs = [f"{p['detector']} k={p['k']}" for p in pairs if not p["measured"] and p["in_scope_f1"] is not None
                      and ((p["detector"] != choice["detector"] and p["in_scope_f1"] >= choice["in_scope_f1"])
                           or (p["detector"] == choice["detector"] and p["k"] < choice["k"]))]
    return {"what": "PLAN.md A4 detector and detection-interval choice, applied on MOT17 val (A6.8); saved files only",
            "rule": f"highest in-scope F1 among pairs whose median of 3 sync offline runs is >= {min_fps} fps; "
                    "smallest k for that detector; F1 tie at 3 decimals -> faster median",
            "in_scope_f1": f1, "runtime_summary": runtime_summary.as_posix(), "pairs": pairs,
            "choice": ({k: choice[k] for k in ("detector", "k", "in_scope_f1", "throughput_fps_median",
                                               "throughput_fps_range", "worst_clip_latency_ms_p95_median",
                                               "latency_ms_max_median")} if choice else None),
            "operating_threshold": f1[choice["detector"]]["threshold"] if choice else None,
            "unmeasured_pairs_that_could_change_the_choice": open_pairs,
            "verdict": ("no pair qualifies: no detector is selected and the throughput bar fails (A4.5)" if choice is None
                        else "valid" if not open_pairs else "INCOMPLETE: time the pairs listed above first")}

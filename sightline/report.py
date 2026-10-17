"""Markdown tables generated from saved result files only (`sightline report`).

The point is that no metric in the docs is typed by hand: docs/RESULTS.md is this module's output,
and docs/EVALUATION.md interprets it. Every table names the file it was read from.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _f(v, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]


def _bar(final: dict) -> list[str]:
    b = final["bar"]
    det = final.get("detection_in_scope_by_dataset", {}).get("indoorcrowd", {})
    trk, base = final["tracking"]["bytetrack"].get("ALL", {}), final["tracking"]["iou"].get("ALL", {})
    ev = final["events"]["bytetrack"].get("ALL", {})
    cov = final["coverage"]
    measured = {
        "heldout_clips>=12": f"{cov['n_clips']} clips ({cov['n_entries']} manifest entries)",
        "three_difficult_conditions": ", ".join(cov["difficult_conditions_present"]),
        "negative_corridor_set": f"{cov['negative_clips']} negative entries",
        "detection_precision>=0.80": f"{_f(det.get('precision'))} (IndoorCrowd, in scope)",
        "detection_recall>=0.80": f"{_f(det.get('recall'))} (IndoorCrowd, in scope)",
        "idf1>=0.65": _f(trk.get("idf1")),
        "tracker_beats_baseline": f"IDF1 {_f(trk.get('idf1'))} vs {_f(base.get('idf1'))}; ID switches {trk.get('num_switches')} vs {base.get('num_switches')}",
        "event_precision>=0.85": _f(ev.get("event_precision")),
        "event_recall>=0.85": _f(ev.get("event_recall")),
        "false_events_per_5min_negative<=1": f"{_f(ev.get('false_events_per_5min_negative'), 2)} ({ev.get('negative_clip_false_events')} false events in {_f(ev.get('negative_clip_seconds'), 1)} s)",
        "median_event_latency<=5_frames": _f(ev.get("latency_frames_median"), 1),
        "throughput>=10fps": "see the runtime section",
    }
    rows = [[k, measured.get(k, ""), "PASS" if v else ("not measured" if v is None else "FAIL")] for k, v in b["checks"].items()]
    return ["## The bar (`results/bar_check.json`, `results/final_metrics.json`)", "",
            f"Detection judged on: {b['detection_judged_on']}. Detector `{final['detector']['name']}` at threshold "
            f"{final['detector']['score_threshold']}, detection interval k = {final['detect_every']}.", ""] + \
        _table(["Check", "Measured", "Result"], rows) + [""]


def _groups(section: dict) -> list[str]:
    return ["ALL"] + sorted(k for k in section if k != "ALL")


def _detection(final: dict) -> list[str]:
    rows = []
    for ds, m in sorted(final.get("detection_in_scope_by_dataset", {}).items()):
        rows.append([f"{ds} (in scope)", m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], m["n_clips"]])
    for g in _groups(final["detection"]):
        m = final["detection"][g]
        rows.append([f"{g} (all boxes)", m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], m["n_clips"]])
    for g in _groups(final["detection_in_scope"]):
        m = final["detection_in_scope"][g]
        rows.append([f"{g} (in scope)", m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], m["n_clips"]])
    return ["## Detection (IoU 0.5, frozen threshold, detector frames only)", ""] + \
        _table(["Group", "Precision", "Recall", "F1", "TP", "FP", "FN", "Clips"], rows) + [""]


def _tracking(final: dict) -> list[str]:
    out = ["## Tracking: IoU baseline against ByteTrack-style (every frame, predicted boxes included)", ""]
    for fam, label in (("tracking", "all boxes"), ("tracking_in_scope", "in scope")):
        rows = []
        for g in _groups(final[fam]["bytetrack"]):
            b, i = final[fam]["bytetrack"][g], final[fam]["iou"].get(g, {})
            rows.append([g, i.get("idf1"), b.get("idf1"), i.get("num_switches"), b.get("num_switches"),
                         i.get("mota"), b.get("mota"), b.get("n_clips")])
        out += [f"**{label}**", ""] + _table(["Group", "IDF1 baseline", "IDF1 ByteTrack", "ID sw baseline",
                                               "ID sw ByteTrack", "MOTA baseline", "MOTA ByteTrack", "Clips"], rows) + [""]
    return out


def _events(final: dict) -> list[str]:
    rows = []
    for g in _groups(final["events"]["bytetrack"]):
        e = final["events"]["bytetrack"][g]
        rows.append([g, e["event_precision"], e["event_recall"], e["events_tp"], e["events_fp"], e["events_fn"],
                     e["latency_frames_median"], e["latency_frames_p95"], e["events_unresolved"],
                     e["negative_clip_false_events"], e["n_clips"]])
    base = final["events"]["iou"].get("ALL", {})
    out = ["## Events (per event, one-to-one matching on [first inside, last inside])", ""] + \
        _table(["Group", "Precision", "Recall", "TP", "FP", "FN", "Median latency (frames)", "p95 latency",
                "Unresolved", "False events on negatives", "Entries"], rows)
    out += ["", f"IoU baseline, ALL: precision {_f(base.get('event_precision'))}, recall {_f(base.get('event_recall'))}, "
                f"median latency {_f(base.get('latency_frames_median'), 1)} frames.", ""]
    s = final.get("safety", {}).get("ALL")
    if s:
        out += [f"Safety, ALL: {s['false_clear_frames']} false-CLEAR frames of {s['occupied_frames']} occupied frames "
                f"(rate {_f(s['false_clear_rate'], 4)}); non-OK frame fraction {_f(s['non_ok_frame_fraction'])}.", ""]
    return out


def _runtime(rt: dict | None, label: str, path: str) -> list[str]:
    if not rt:
        return []
    s = rt["summary"]
    m = rt.get("machine", {})
    return [f"## Runtime: {label} (`{path}`)", "",
            f"Machine: {m.get('processor')} ({m.get('cpu_count_logical')} logical CPUs), {m.get('platform')}, Python "
            f"{m.get('python')}. Commit {(rt.get('code') or {}).get('git_commit')}. Detector {rt.get('detector', {}).get('detector')} "
            f"@{rt.get('detector_input_size')}, k = {rt.get('detect_every')}, real-time pacing: {_f(rt.get('realtime_emulation'))}, "
            f"async: {_f(rt.get('async_detection'))}.", ""] + \
        _table(["Frames", "Throughput fps", "p50 latency ms", "p95 latency ms (worst clip)", "p50 inference ms",
                "p95 inference ms", "Dropped (lagging)", "Skipped by policy", "Late (>100 ms)"],
               [[s["frames_processed"], s["throughput_fps"], s["latency_ms_p50"], s["latency_ms_p95"],
                 s["inference_ms_p50"], s["inference_ms_p95"], s["frames_dropped_lagging"], s["frames_skipped_by_policy"],
                 s["frames_late"]]]) + [""]


def _a4(a4: dict | None) -> list[str]:
    if not a4:
        return []
    rows = [[p["detector"], p["k"], p["in_scope_f1"], p["throughput_fps_median"],
             (f"{p['throughput_fps_range'][0]:.2f}-{p['throughput_fps_range'][1]:.2f}" if p["throughput_fps_range"] else "n/a"),
             p["worst_clip_latency_ms_p95_median"], p["latency_ms_max_median"], p["qualifies"]]
            for p in a4["pairs"] if p["measured"]]
    c = a4["choice"]
    return ["## Detector and interval choice, PLAN A4 on MOT17 val (`results/detector_selection/a4_choice.json`)", "",
            f"Rule: {a4['rule']}. Verdict: {a4['verdict']}. Choice: "
            + (f"{c['detector']} k = {c['k']} (operating threshold from val: {a4['operating_threshold']})." if c else "none."),
            "Only measured pairs are listed; an unmeasured pair has a lower in-scope F1 and cannot change the choice "
            f"(unmeasured pairs that could: {a4['unmeasured_pairs_that_could_change_the_choice'] or 'none'}).", ""] + \
        _table(["Detector", "k", "In-scope F1 (val)", "Median fps (3 runs)", "Range", "Worst p95 ms (median)",
                "Max ms (median)", "Qualifies (>= 10 fps)"], rows) + [""]


def _ablation(ab: dict | None) -> list[str]:
    if not ab:
        return []
    rows = [[n, v["idf1"], v["id_switches"], v["idf1_in_scope"], v["event_precision"], v["event_recall"],
             v["latency_frames_median"], v["events_unresolved"], v["negative_clip_false_events"]] for n, v in ab["variants"].items()]
    return ["## Ablation on the held-out set (`results/ablation/ablation_heldout.json`; analysis only)", ""] + \
        _table(["Variant", "IDF1", "ID switches", "IDF1 in scope", "Event precision", "Event recall",
                "Median latency", "Unresolved", "False events on negatives"], rows) + [""]


def _calibration(cal: dict | None, path: str) -> list[str]:
    if not cal:
        return []
    rows = [[f"{b['score_from']:.1f}-{b['score_to']:.1f}", b["detections"], b["mean_score"], b["precision"]]
            for b in cal["reliability_bins"]]
    q = [[f"{r['mean_confidence_from']:.3f}-{r['mean_confidence_to']:.3f}", r["frames"], r["mean_recall"]]
         for r in cal["recall_by_frame_confidence_quintile"]]
    return [f"## Confidence calibration (`{path}`; analysis only)", "",
            f"{cal['detections_counted']} detections at or above the {cal['detection_floor']} floor; ECE "
            f"{_f(cal['ece'])}. Spearman correlation of a frame's mean used confidence with its recall: "
            f"{_f(cal['spearman_frame_confidence_vs_recall'])} over {cal['frames_with_people_and_detections']} frames.", ""] + \
        _table(["Score bin", "Detections", "Mean score", "Precision"], rows) + [""] + \
        _table(["Frame mean confidence (quintile)", "Frames", "Mean recall"], q) + [""]


def _stress(rows: list | None) -> list[str]:
    if not rows:
        return []
    t = [[r["degradation"], r["level"], r["detection_precision"], r["detection_recall"], r["idf1"], r["event_precision"],
          r["event_recall"], r["latency_frames_median"], r["false_clear_rate"], r["non_ok_frame_fraction"]] for r in rows]
    return ["## Stress suite, frozen system (`results/stress/stress_results.json`; nothing retuned)", ""] + \
        _table(["Degradation", "Level", "Det P", "Det R", "IDF1", "Event P", "Event R", "Median latency",
                "False-CLEAR rate", "Non-OK fraction"], t) + [""]


def build(results_dir: Path) -> str:
    rd = Path(results_dir)
    final = _load(rd / "final_metrics.json")
    out = ["# Results (generated)", "",
           "Generated by `python -m sightline report` from the files named in each section. Do not edit by hand; "
           "re-run the command. Interpretation is in `docs/EVALUATION.md`.", ""]
    out += _a4(_load(rd / "detector_selection" / "a4_choice.json"))
    if final:
        out += _bar(final) + _detection(final) + _tracking(final) + _events(final)
    out += _runtime(_load(rd / "runtime_report.json"), "held-out clips, offline, every frame processed", "results/runtime_report.json")
    out += _runtime(_load(rd / "runtime" / "heldout_realtime.json"), "held-out clips paced in real time",
                    "results/runtime/heldout_realtime.json")
    out += _ablation(_load(rd / "ablation" / "ablation_heldout.json"))
    out += _calibration(_load(rd / "calibration" / "heldout.json"), "results/calibration/heldout.json")
    out += _stress(_load(rd / "stress" / "stress_results.json"))
    return "\n".join(out) + "\n"

"""Command line: python -m sightline <command>.

Exit codes: 0 ok, 1 usage/other error, 2 the system is UNAVAILABLE (bad input, bad
config, missing model). UNAVAILABLE is a normal, honest outcome, not a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

from sightline import __version__
from sightline.config import Config, ConfigError, load_config, load_corridor
from sightline.detector import DetectorUnavailable, make_detector, read_detection_cache
from sightline.ingest import IngestError, discover_sources, ingest
from sightline.manifest import ManifestError, coverage_check, load_manifest, lock_manifest, split_check, verify_lock
from sightline.runner import run_clip


def _p(msg: str) -> None:
    print(msg, flush=True)


def _unavailable(out: str | None, reason: str) -> int:
    """Stop with an explicit UNAVAILABLE. Never prints anything that reads as 'clear'."""
    _p(f"UNAVAILABLE: {reason}")
    if out:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "health.jsonl").write_text(json.dumps({"frame": 0, "state": "UNAVAILABLE", "reasons": [reason], "corridor": "UNKNOWN"}) + "\n")
        (o / "summary.json").write_text(json.dumps({"final_health": "UNAVAILABLE", "unavailable_reason": reason, "events": []}, indent=1))
    return 2


def _load_cfg(a) -> Config:
    """Load --config, then apply --detector / --input-size overrides (candidate comparison)."""
    cfg = load_config(a.config)
    if getattr(a, "detector", None):
        cfg.detector.name = a.detector
    if getattr(a, "input_size", None):
        cfg.detector.input_size = list(a.input_size)
    if getattr(a, "detect_every", None):
        cfg.pipeline.detect_every = a.detect_every
    if getattr(a, "async_detection", False):
        cfg.pipeline.async_detection = True
    return cfg.validate()


def _add_detector_overrides(sp) -> None:
    sp.add_argument("--detector", help="override detector.name (e.g. fasterrcnn_resnet50_fpn_v2)")
    sp.add_argument("--input-size", type=int, nargs=2, metavar=("W", "H"), help="override detector.input_size")


def _detector_factory(cfg: Config):
    def make():
        return make_detector(cfg.detector)
    return make


# -- commands --------------------------------------------------------------------------------------

def cmd_ingest(a) -> int:
    try:
        sources = discover_sources(a.path)
        for s in sources:
            rec = ingest(s, a.out, a.fps)
            _p(f"{rec['path']}: {rec['kind']} {rec['width']}x{rec['height']} fps={rec['fps']} ({rec['fps_source']}) "
               f"frames={rec['frames_total']} (container says {rec['frame_count_reported']}) invalid={rec['frames_invalid']} "
               f"sha256={str(rec['sha256'])[:12]}")
        return 0
    except IngestError as exc:
        return _unavailable(None, f"ingest: {exc}")


def cmd_draw_corridor(a) -> int:
    from sightline.footage import GuiUnavailable, draw_corridor

    try:
        res = draw_corridor(a.clip, a.out, a.name)
    except (IngestError, GuiUnavailable) as exc:
        return _unavailable(None, str(exc))
    if res is None:
        _p("cancelled: nothing written")
        return 1
    _p(f"wrote {res['corridor']}: {len(res['polygon'])} vertices on a {res['frame_size'][0]}x{res['frame_size'][1]} frame")
    return 0


def cmd_label_events(a) -> int:
    from sightline.config import ConfigError
    from sightline.footage import GuiUnavailable, label_events

    try:
        res = label_events(a.clip, a.corridor, a.out)
    except (IngestError, GuiUnavailable, ConfigError) as exc:
        return _unavailable(None, str(exc))
    if res is None:
        _p("quit without saving: nothing written")
        return 1
    still = sum(1 for e in res["events"] if e.get("still_inside_at_end"))
    _p(f"wrote {res['out']}: {len(res['events'])} events ({still} still inside at the end of the clip)")
    return 0


def cmd_check_clip(a) -> int:
    from sightline.footage import check_clip

    try:
        r = check_clip(a.file)
    except IngestError as exc:
        return _unavailable(None, f"check-clip: {exc}")
    iv = r["frame_interval_ms"]
    _p(f"{r['path']}: {r['kind']} {r['codec']} {r['resolution'][0]}x{r['resolution'][1]} {r['fps']:.3f} fps "
       f"({r['fps_source']}), {r['frames']} frames, {r['duration_s']:.1f} s, rotation {r['rotation_deg']}, "
       f"frame interval {iv['min']}-{iv['max']} ms" if iv else
       f"{r['path']}: {r['kind']} {r['resolution'][0]}x{r['resolution'][1]} {r['fps']:.3f} fps, {r['frames']} frames, "
       f"{r['duration_s']:.1f} s")
    for pr in r["problems"]:
        _p(f"PROBLEM: {pr}")
    for w in r["warnings"]:
        _p(f"warning: {w}")
    _p("ok: nothing here breaks ingest" if r["ok"] else "NOT OK: fix the problems above before using this clip")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


def cmd_import_cvat(a) -> int:
    from sightline.annotations import AnnotationError, import_cvat_mot

    try:
        rep = import_cvat_mot(a.export, a.out, a.frames, tuple(a.person_labels), tuple(a.ignore_labels))
    except (AnnotationError, FileNotFoundError) as exc:
        return _unavailable(None, f"import-cvat: {exc}")
    _p(f"imported {rep['person_boxes']} person boxes ({rep['ignore_boxes']} ignore) on {rep['frames_with_boxes']} of "
       f"{rep['annotated_frames']} annotated frames -> {a.out}; dropped labels: {rep['dropped_rows_by_label'] or 'none'}")
    for w in rep["warnings"]:
        _p(f"WARNING: {w}")
    _p('manifest entry: "gt": "<out>/gt.txt", "annotated_frames": "<out>/annotated_frames.json" (plus hand-labelled gt_events)')
    return 0


def cmd_replay(a) -> int:
    try:
        cfg = load_config(a.config)
        if a.tracker:
            cfg.tracker.kind = a.tracker
            cfg.validate()
        corridor = load_corridor(a.corridor)
    except ConfigError as exc:
        return _unavailable(a.out, f"invalid configuration: {exc}")

    cached = detector = None
    if a.detections_cache:
        try:
            _meta, cached = read_detection_cache(a.detections_cache)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            return _unavailable(a.out, f"detection cache unreadable: {exc}")
    else:
        try:
            detector = make_detector(cfg.detector)
        except DetectorUnavailable as exc:
            return _unavailable(a.out, f"detector unavailable: {exc}")

    run = run_clip(a.source, cfg, corridor, detector=detector, cached=cached, fps_override=a.fps, out_dir=a.out,
                   overlay=a.overlay, realtime=a.realtime)
    rt = run.runtime
    _p(f"clip={run.clip} health={run.final_health} frames={run.frames_seen} events={len(run.events)} "
       f"unresolved={sum(1 for e in run.events if e.status.value == 'UNRESOLVED')}")
    _p(f"corridor frames: {run.corridor_counts}   health frames: {run.health_counts}")
    if rt.get("throughput_fps"):
        _p(f"fps={rt['throughput_fps']:.1f} p50={rt['latency_ms_p50']:.1f}ms p95={rt['latency_ms_p95']:.1f}ms "
           f"dropped={rt['frames_dropped_lagging']} skipped={rt['frames_skipped_by_policy']} late={rt['frames_late']}")
    if run.unavailable_reason or run.final_health == "UNAVAILABLE":
        _p(f"UNAVAILABLE: {run.unavailable_reason or 'see health.jsonl'}")
        return 2
    return 0


def cmd_demo(a) -> int:
    """No downloads, no model: a generated clip through both trackers. NOT evidence of real performance."""
    from sightline.synth import make_synthetic_clip

    out = Path(a.out)
    clip = make_synthetic_clip(out / "data", "synthetic")
    cfg = load_config(a.config)
    corridor = load_corridor("configs/corridors/synthetic.yaml")
    _m, cached = read_detection_cache(clip.cache)
    _p("SYNTHETIC DEMO: generated video and detections. Not a measurement of real-world performance.")
    for kind in ("iou", "bytetrack"):
        cfg.tracker.kind = kind
        run = run_clip(clip.video, cfg, corridor, cached=cached, out_dir=out / kind, overlay=True)
        ids = sorted({r[1] for r in run.mot_rows})
        _p(f"{kind:10s} health={run.final_health} track_ids={ids} events={len(run.events)} -> {out / kind}")
    return 0


def cmd_split_check(a) -> int:
    try:
        val, test = load_manifest(a.val), load_manifest(a.test)
    except ManifestError as exc:
        return _unavailable(None, str(exc))
    problems = split_check(val, test)
    cov = coverage_check(test)
    _p(f"val clips={len(val.clips)} test clips={len(test.clips)}")
    _p(f"coverage: {json.dumps(cov)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "val": a.val, "test": a.test, "leakage_problems": problems,
            "val_clips_scored_on": coverage_check(val)["clips_scored_on"],
            "val_event_clips": [c.id for c in val.clips if c.scored_on("events")],
            "test_coverage": cov, "test_event_clips": [c.id for c in test.clips if c.scored_on("events")],
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    if problems:
        for pr in problems:
            _p(f"LEAK: {pr}")
        return 1
    _p("split is leakage-safe: no shared clip, source, scene, base scene, camera or recording session")
    return 0


def cmd_lock(a) -> int:
    try:
        man = load_manifest(a.manifest)
        cov = coverage_check(man)
        locked = lock_manifest(a.manifest, a.out, a.root, eval_config=a.eval_config)
    except (ManifestError, FileNotFoundError) as exc:
        return _unavailable(None, f"cannot lock: {exc}")
    _p(f"locked {len(locked['clips'])} clips -> {a.out}  status={locked['status'].upper()}  "
       f"sha256={locked['manifest_sha256'][:16]}  git={locked['git_commit']}")
    if locked["status"] != "final":
        _p("DRAFT lock: the held-out set does not meet the coverage bar yet; `tune` will refuse it")
    _p(f"coverage vs the bar: {json.dumps(cov)}")
    return 0


def cmd_verify_lock(a) -> int:
    try:
        problems = verify_lock(a.lock, a.root)
    except ManifestError as exc:
        return _unavailable(None, str(exc))
    for pr in problems:
        _p(f"CHANGED: {pr}")
    _p("held-out set unchanged since locking" if not problems else "HELD-OUT SET CHANGED AFTER LOCKING")
    return 1 if problems else 0


def cmd_detect_eval(a) -> int:
    from sightline.evaluation.detection import sweep_thresholds
    from sightline.evaluation.run import get_detections, in_scope_view, load_eval_config, prepare_clip, write_json

    cfg, eval_cfg, root = _load_cfg(a), load_eval_config(a.eval_config), Path(a.root)
    man = load_manifest(a.manifest)
    from sightline.evaluation.detection import THRESHOLDS, counts_to_metrics

    clips = [c for c in man.clips if c.scored_on("detection")]
    ths = THRESHOLDS
    views = {"all": {}, "in_scope": {}}  # view -> clip id -> sweep rows
    n_frames = 0
    for c in clips:
        cd = prepare_clip(c, root, eval_cfg)
        dets = get_detections(cd, cfg, root, Path(a.cache), _detector_factory(cfg))
        frames = cd.scored(range(1, cd.n_frames + 1))  # the detector itself, on every annotated frame
        n_frames += len(frames)
        views["all"][c.id] = sweep_thresholds(cd.gt, cd.ignore, dets, ths, frames)
        views["in_scope"][c.id] = sweep_thresholds(*in_scope_view(cd), dets, ths, frames)
    overall = {}
    for view, per_clip in views.items():
        tot = {t: [0, 0, 0] for t in ths}
        for rows in per_clip.values():
            for r in rows:
                for i, k in enumerate(("tp", "fp", "fn")):
                    tot[r["threshold"]][i] += r[k]
        overall[view] = [counts_to_metrics(*tot[t], threshold=t) for t in ths]
    for m_all, m_in in zip(overall["all"], overall["in_scope"]):
        _p(f"thr={m_all['threshold']:.2f}  all boxes P={m_all['precision']:.3f} R={m_all['recall']:.3f} F1={m_all['f1']:.3f}"
           f"  |  in scope P={m_in['precision']:.3f} R={m_in['recall']:.3f} F1={m_in['f1']:.3f}")
    if a.out:
        write_json(a.out, {
            "what": f"person detection P/R/F1 at IoU {eval_cfg['detection']['iou']} vs score threshold, pooled over clips",
            "manifest": a.manifest, "detector": cfg.detector.name, "input_size": list(cfg.detector.input_size),
            "min_score": cfg.detector.min_score,
            "clips": [c.id for c in clips], "frames_evaluated": n_frames,
            "ground_truth": "MOTChallenge rules as of 2026-10-14: class 1 conf 1 scored; classes 2,7,8,12 and conf-0 "
                            "pedestrians ignored; other classes dropped",
            "in_scope_rule": eval_cfg.get("in_scope"),
            "overall": overall["all"], "overall_in_scope": overall["in_scope"],
            "per_clip": views["all"], "per_clip_in_scope": views["in_scope"],
        })
    return 0



def cmd_interval_eval(a) -> int:
    from sightline.evaluation.run import load_eval_config, write_json
    from sightline.evaluation.interval import interval_sweep

    try:
        cfg = _load_cfg(a)
        man = load_manifest(a.manifest)
    except (ConfigError, ManifestError) as exc:
        return _unavailable(None, str(exc))
    out = interval_sweep(man, cfg, load_eval_config(a.eval_config), Path(a.root), Path(a.cache), _detector_factory(cfg),
                         a.every, progress=_p)
    if a.out:
        write_json(a.out, out)
        _p(f"wrote {a.out}")
    return 0


def cmd_tune(a) -> int:
    from sightline.evaluation.run import load_eval_config
    from sightline.tune import tune

    cfg, eval_cfg = load_config(a.config), load_eval_config(a.eval_config)
    try:
        tune(a.val, a.locked, cfg, eval_cfg, Path(a.root), Path(a.cache), _detector_factory(cfg), Path(a.out), Path(a.tuning_dir), _p)
    except ManifestError as exc:
        return _unavailable(None, str(exc))
    return 0


def cmd_evaluate(a) -> int:
    from sightline.evaluation.run import (aggregate, bar_check, evaluate_clip, get_detections, load_eval_config,
                                          per_clip_details, prepare_clip, write_json)

    root, results = Path(a.root), Path(a.results)
    final_path = results / "final_metrics.json"
    log_path = results / "final_run_log.jsonl"
    try:
        cfg_text = Path(a.config).read_text()
        raw = yaml.safe_load(cfg_text)
        cfg = load_config(a.config)
        prov = raw.get("provenance") or {}
        lock = json.loads(Path(a.lock).read_text())
        man = load_manifest(a.lock)
    except (ConfigError, ManifestError, FileNotFoundError, json.JSONDecodeError) as exc:
        return _unavailable(None, f"cannot start evaluation: {exc}")

    # the held-out set must be locked, complete, unchanged, and fixed BEFORE the config was tuned
    if lock.get("status") != "final":
        return _unavailable(None, f"{a.lock} is a draft lock; the held-out set is incomplete")
    if not prov.get("frozen") or prov.get("locked_test_manifest_sha256") != lock["manifest_sha256"]:
        return _unavailable(None, "frozen config was not tuned against THIS locked manifest (provenance mismatch)")
    problems = verify_lock(a.lock, root)
    if problems:
        for pr in problems:
            _p(f"CHANGED: {pr}")
        return _unavailable(None, "held-out set changed after locking")
    if final_path.exists() and not a.allow_rerun:
        return _unavailable(None, f"{final_path} already exists: the held-out test has been opened. Use --allow-rerun to repeat it (logged)")

    eval_cfg = load_eval_config(a.eval_config)
    cds = [prepare_clip(c, root, eval_cfg) for c in man.clips]
    _p(f"evaluating {len(cds)} held-out clips with frozen config {hashlib.sha256(cfg_text.encode()).hexdigest()[:12]}")
    all_results, dets_by_clip = [], {}
    for cd in cds:
        dets = get_detections(cd, cfg, root, Path(a.cache), _detector_factory(cfg))
        dets_by_clip[cd.clip.id] = dets
        r = evaluate_clip(cd, cfg, dets, eval_cfg, root, out_root=results / "tracks")
        all_results.append(r)
        for kind in ("iou", "bytetrack"):
            write_json(results / "per_clip" / kind / f"{cd.clip.id}.json", per_clip_details(r, cfg, dets, eval_cfg, kind))
        _p(f"  {cd.clip.id}: det P={r.det_counts['precision']:.2f} R={r.det_counts['recall']:.2f} "
           f"events={len(r.runs['bytetrack'].events)} gt_events={len(cd.gt_events)} health={r.runs['bytetrack'].final_health}")

    agg = aggregate(all_results)
    cov = coverage_check(man)
    runtime = None
    rp = results / "runtime_report.json"
    if rp.is_file():
        runtime = json.loads(rp.read_text()).get("summary")
    agg["coverage"] = cov
    agg["frozen_config_sha256"] = hashlib.sha256(cfg_text.encode()).hexdigest()
    agg["locked_manifest_sha256"] = lock["manifest_sha256"]
    agg["detector"] = {"name": cfg.detector.name, "score_threshold": cfg.detector.score_threshold}
    agg["detect_every"] = cfg.pipeline.detect_every
    agg["bar"] = bar_check(agg, cov, eval_cfg, runtime=runtime)
    write_json(final_path, agg)
    write_json(results / "bar_check.json", agg["bar"])
    with log_path.open("a") as f:
        f.write(json.dumps({"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "frozen_config_sha256": agg["frozen_config_sha256"],
                            "locked_manifest_sha256": agg["locked_manifest_sha256"], "allow_rerun": bool(a.allow_rerun),
                            "all_pass": agg["bar"]["all_pass"]}) + "\n")
    for k, v in agg["bar"]["checks"].items():
        _p(f"  [{'PASS' if v else 'n/a ' if v is None else 'FAIL'}] {k}")
    _p(f"wrote {final_path}")
    return 0


def cmd_benchmark(a) -> int:
    from sightline.runtime import machine_info

    try:
        cfg = _load_cfg(a)
        corridor = load_corridor(a.corridor)
        detector = make_detector(cfg.detector)
    except (ConfigError, DetectorUnavailable) as exc:
        return _unavailable(None, str(exc))
    di = detector.info()
    _p(f"detector={di['detector']} backend={di.get('backend')} "
       f"threads={di.get('torch_threads', di.get('intra_op_threads'))} interop={di.get('torch_interop_threads', 'n/a')}")
    all_lat, summaries = None, []
    for src in a.sources:
        run = run_clip(src, cfg, corridor, detector=detector, fps_override=a.fps, realtime=a.realtime,
                       max_frames=a.max_frames)
        summaries.append({"clip": run.clip, **run.runtime})
        rt = run.runtime
        _p(f"{run.clip}: fps={rt.get('throughput_fps')} p50={rt.get('latency_ms_p50')} p95={rt.get('latency_ms_p95')} "
           f"infer_p50={rt.get('inference_ms_p50')} dropped={rt.get('frames_dropped_lagging')} late={rt.get('frames_late')}")
    del all_lat
    total_frames = sum(s["frames_processed"] for s in summaries)
    wall = sum(s["wall_seconds"] for s in summaries)
    best = {"frames_processed": total_frames, "throughput_fps": (total_frames / wall) if wall else None,
            "latency_ms_p50": _wavg(summaries, "latency_ms_p50"), "latency_ms_p95": max((s["latency_ms_p95"] or 0) for s in summaries),
            "inference_ms_p50": _wavg(summaries, "inference_ms_p50"), "inference_ms_p95": max((s["inference_ms_p95"] or 0) for s in summaries),
            "frames_dropped_lagging": sum(s["frames_dropped_lagging"] for s in summaries),
            "frames_skipped_by_policy": sum(s["frames_skipped_by_policy"] for s in summaries),
            "frames_late": sum(s["frames_late"] for s in summaries),
            "note": "latency_ms_p95 is the worst p95 over clips; p50s are frame-weighted"}
    report = {"summary": best, "per_clip": summaries, "machine": machine_info(), "detector": detector.info(),
              "realtime_emulation": a.realtime, "declared_input": [cfg.pipeline.declared_width, cfg.pipeline.declared_height],
              "detector_input_size": list(cfg.detector.input_size), "decode_thread": cfg.pipeline.decode_thread,
              "detect_every": cfg.pipeline.detect_every, "async_detection": cfg.pipeline.async_detection,
              "max_frames_per_clip": a.max_frames}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    _p(f"wrote {a.out}")
    return 0


def cmd_threads(a) -> int:
    """Time detect() at several torch thread counts on real frames. Evidence for detector.torch_threads."""
    import statistics
    from itertools import islice

    import torch

    from sightline.ingest import iter_frames
    from sightline.runtime import machine_info

    try:
        cfg = _load_cfg(a)
        frames = [f.image for f in islice(iter_frames(a.source), a.frames) if f.valid]
    except (ConfigError, IngestError) as exc:
        return _unavailable(None, str(exc))
    _p(f"torch default: intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}; "
       f"{len(frames)} frames of {a.source}")
    rows = []
    for n in a.threads:
        cfg.detector.torch_threads = n  # rebuilt per count: applies to torch and to ONNX Runtime sessions
        try:
            detector = make_detector(cfg.detector)
        except DetectorUnavailable as exc:
            return _unavailable(None, str(exc))
        for img in frames[:3]:  # warm-up at this thread count, excluded
            detector.detect(img)
        ms = []
        for img in frames:
            t0 = time.perf_counter()
            detector.detect(img)
            ms.append((time.perf_counter() - t0) * 1000)
        row = {"torch_threads": n, "detect_ms_p50": statistics.median(ms), "detect_ms_mean": statistics.fmean(ms),
               "detect_fps_p50": 1000 / statistics.median(ms), "frames": len(ms)}
        rows.append(row)
        _p(f"threads={n:3d}  detect p50={row['detect_ms_p50']:.1f} ms  mean={row['detect_ms_mean']:.1f} ms")
    best = min(rows, key=lambda r: r["detect_ms_p50"])
    _p(f"fastest: torch_threads={best['torch_threads']} ({best['detect_ms_p50']:.1f} ms p50)")
    report = {"what": "detect() time per frame vs torch intra-op threads; warm-up excluded", "source": a.source,
              "rows": rows, "fastest": best["torch_threads"], "detector": detector.info(), "machine": machine_info()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    _p(f"wrote {a.out}")
    return 0


def cmd_detector_summary(a) -> int:
    from sightline.evaluation.run import load_eval_config, write_json
    from sightline.selection import build_table, to_markdown

    bar = load_eval_config(a.eval_config)["bar"]
    cfg = load_config(a.config)
    t = build_table(Path(a.results), a.candidates, bar["detection_precision"], bar["detection_recall"], bar["min_fps"],
                    cfg.detector.score_threshold)
    out = Path(a.results) / "detector_selection"
    write_json(out / "summary.json", t)
    (out / "summary.md").write_text(to_markdown(t))
    _p(to_markdown(t))
    _p(f"wrote {out / 'summary.json'} and summary.md")
    return 0


def cmd_runtime_summary(a) -> int:
    from sightline.runtime import summarize_runs

    s = summarize_runs(Path(a.parent))
    if not s:
        _p(f"no <config>_run<N> folders with benchmark reports in {a.parent}")
        return 1
    _p("| Config | Runs | Throughput fps, median (min-max) | Worst clip p95 ms | Max latency ms | Dropped | Det. skipped busy |")
    _p("|---|---|---|---|---|---|---|")
    for name, g in s.items():
        f = lambda k, nd=1: f"{g[k]['median']:.{nd}f} ({g[k]['min']:.{nd}f}-{g[k]['max']:.{nd}f})"  # noqa: E731
        _p(f"| {name} | {g['runs']} | {f('throughput_fps', 2)} | {f('worst_clip_latency_ms_p95', 0)} | "
           f"{f('latency_ms_max', 0)} | {f('frames_dropped', 0)} | {f('detections_skipped_worker_busy', 0)} |")
    out = Path(a.out or Path(a.parent) / "summary.json")
    out.write_text(json.dumps(s, indent=1))
    _p(f"wrote {out}")
    return 0


def _wavg(summaries: list[dict], key: str):
    w = [(s[key], s["frames_processed"]) for s in summaries if s.get(key) is not None]
    tot = sum(n for _v, n in w)
    return sum(v * n for v, n in w) / tot if tot else None


def cmd_stress(a) -> int:
    from sightline.evaluation.run import load_eval_config, prepare_clip, write_json
    from sightline.stress import LEVELS, run_stress

    try:
        cfg = load_config(a.config)
        man = load_manifest(a.lock)
        detector = make_detector(cfg.detector)
    except (ConfigError, ManifestError, DetectorUnavailable) as exc:
        return _unavailable(None, str(exc))
    problems = verify_lock(a.lock, a.root)
    if problems:
        return _unavailable(None, "held-out set changed after locking")
    eval_cfg = load_eval_config(a.eval_config)
    cds = [prepare_clip(c, Path(a.root), eval_cfg) for c in man.clips]
    kinds = tuple(a.kinds) if a.kinds else tuple(LEVELS)
    rows = run_stress(cds, cfg, eval_cfg, Path(a.root), detector, kinds, progress=_p)
    write_json(Path(a.results) / "stress" / "stress_results.json", rows)
    _p("stress results written; the frozen config was NOT changed")
    return 0


def cmd_errors(a) -> int:
    from sightline.errors import analyze
    from sightline.evaluation.run import detector_tag

    if a.unresolved or a.false_events:
        from sightline.errors import analyze_false_events, analyze_unresolved, collect_details
        from sightline.evaluation.run import load_eval_config

        cfg, manifest = _load_cfg(a), Path(a.manifest or a.lock)
        if a.manifest:  # val mode: build per-clip details from cached detections, nothing tuned
            try:
                clips = collect_details(manifest, cfg, load_eval_config(a.eval_config), Path(a.root), Path(a.cache),
                                        Path(a.results))
            except RuntimeError as exc:
                return _unavailable(None, f"errors: {exc}")
            _p(f"per-clip details for {clips} written to {a.results}")
        note = (f"{manifest.as_posix()} | detector {cfg.detector.name} @{cfg.detector.input_size} | detect_every "
                f"{cfg.pipeline.detect_every} | score threshold {cfg.detector.score_threshold} | config {a.config} (untuned)")
        for n, fn, what in ((a.unresolved, analyze_unresolved, "unresolved-event"),
                            (a.false_events, analyze_false_events, "false-event")):
            if n:
                out = fn(Path(a.results), Path(a.root), manifest, Path(a.cache), detector_tag(cfg),
                         cfg.detector.score_threshold, n, note)
                _p(f"{what} strips written to {out}: open INDEX.md and write the 'why' for each after looking at it")
        return 0
    cfg = load_config(a.config)
    out = analyze(Path(a.results), Path(a.root), Path(a.lock), Path(a.cache), detector_tag(cfg), cfg.detector.score_threshold, a.per_type)
    _p(f"error analysis written to {out}: open INDEX.md and write the 'why' for each failure after looking at it")
    return 0


def cmd_plots(a) -> int:
    from sightline.plots import make_all

    made = make_all(a.results)
    for p in made:
        _p(f"wrote {p}")
    if not made:
        _p("no saved results to plot yet")
    return 0


# -- parser ------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sightline", description="Advisory video perception: detect, track, corridor events.")
    ap.add_argument("--version", action="version", version=f"sightline {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("ingest", cmd_ingest, "record fps, frame count, timestamps and metadata for a clip or folder")
    sp.add_argument("path")
    sp.add_argument("--out", default="results/ingest")
    sp.add_argument("--fps", type=float)

    sp = add("draw-corridor", cmd_draw_corridor, "click a corridor polygon on frame 1 and write a validated YAML")
    sp.add_argument("clip")
    sp.add_argument("--out", required=True, help="e.g. configs/corridors/own-corridorA-tripod1.yaml")
    sp.add_argument("--name", help="corridor name (default: the file stem)")

    sp = add("label-events", cmd_label_events, "step through a clip and mark ground-truth corridor events")
    sp.add_argument("clip")
    sp.add_argument("--corridor", required=True)
    sp.add_argument("--out", required=True, help="e.g. annotations/<clip id>/events.json")

    sp = add("check-clip", cmd_check_clip, "fps, variable frame rate, resolution, duration; warn on what would break ingest")
    sp.add_argument("file")
    sp.add_argument("--out", help="also save the report as JSON")

    sp = add("import-cvat", cmd_import_cvat, "convert a CVAT 'MOT 1.1' export into annotations/<clip>/")
    sp.add_argument("export", help="the exported .zip, or its unpacked folder")
    sp.add_argument("--out", required=True, help="e.g. annotations/own-corridorA-20261014-03")
    sp.add_argument("--frames", required=True,
                    help="which frames were annotated: 1-600, 1-600:10, comma-separated parts, or a file of frame numbers")
    sp.add_argument("--person-labels", nargs="+", default=["person", "pedestrian"])
    sp.add_argument("--ignore-labels", nargs="*", default=[], help="labels to keep as ignore regions (never scored)")

    sp = add("replay", cmd_replay, "run one clip end to end")
    sp.add_argument("source")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--corridor", required=True)
    sp.add_argument("--tracker", choices=["iou", "bytetrack"])
    sp.add_argument("--detections-cache", help="replay cached detections instead of running the detector")
    sp.add_argument("--out", default="outputs/replay")
    sp.add_argument("--fps", type=float)
    sp.add_argument("--overlay", action="store_true")
    sp.add_argument("--realtime", action="store_true", help="emulate a live stream and count dropped frames")

    sp = add("demo", cmd_demo, "synthetic end-to-end demo (no downloads, no model)")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--out", default="outputs/demo")

    sp = add("split-check", cmd_split_check, "fail if val and test leak")
    sp.add_argument("--val", required=True)
    sp.add_argument("--test", required=True)
    sp.add_argument("--out", help="also save the leakage result and coverage as JSON")

    sp = add("lock-manifest", cmd_lock, "hash and lock the held-out manifest")
    sp.add_argument("manifest")
    sp.add_argument("--out", default="results/heldout_manifest.json")
    sp.add_argument("--root", default=".")
    sp.add_argument("--eval-config", default="configs/eval.yaml", help="label rules and bar; hashed into the lock")

    sp = add("verify-lock", cmd_verify_lock, "check the held-out set is unchanged since locking")
    sp.add_argument("lock", nargs="?", default="results/heldout_manifest.json")
    sp.add_argument("--root", default=".")

    common = lambda sp: (sp.add_argument("--config", default="configs/default.yaml"),  # noqa: E731
                         sp.add_argument("--eval-config", default="configs/eval.yaml"),
                         sp.add_argument("--root", default="."),
                         sp.add_argument("--cache", default="results/detections"))

    sp = add("detect-eval", cmd_detect_eval, "detector P/R/F1 vs threshold on a manifest (use val)")
    sp.add_argument("--manifest", default="configs/manifests/val.json")
    sp.add_argument("--out")
    _add_detector_overrides(sp)
    common(sp)

    sp = add("interval-eval", cmd_interval_eval,
             "event recall/latency vs detection interval on a manifest (use val); tunes nothing")
    sp.add_argument("--manifest", default="configs/manifests/val.json")
    sp.add_argument("--every", type=int, nargs="+", default=[1, 2, 3], help="detection intervals k to compare")
    sp.add_argument("--out")
    _add_detector_overrides(sp)
    common(sp)

    sp = add("tune", cmd_tune, "tune on VAL only, then freeze")
    sp.add_argument("--val", default="configs/manifests/val.json")
    sp.add_argument("--locked", default="results/heldout_manifest.json")
    sp.add_argument("--out", default="results/frozen_config.yaml")
    sp.add_argument("--tuning-dir", default="results/tuning")
    common(sp)

    sp = add("evaluate", cmd_evaluate, "held-out evaluation with the frozen config")
    sp.add_argument("--lock", default="results/heldout_manifest.json")
    sp.add_argument("--results", default="results")
    sp.add_argument("--allow-rerun", action="store_true")
    common(sp)
    sp.set_defaults(config="results/frozen_config.yaml")

    sp = add("benchmark", cmd_benchmark, "runtime evidence on THIS machine (live detector)")
    sp.add_argument("sources", nargs="+")
    sp.add_argument("--corridor", required=True)
    sp.add_argument("--config", default="results/frozen_config.yaml")
    sp.add_argument("--out", default="results/runtime_report.json")
    sp.add_argument("--fps", type=float)
    sp.add_argument("--realtime", action="store_true")
    sp.add_argument("--max-frames", type=int, help="time only the first N frames of each clip (recorded in the report)")
    sp.add_argument("--detect-every", type=int, help="override pipeline.detect_every (PLAN.md A2)")
    sp.add_argument("--async-detection", action="store_true", help="run the detector on a worker thread")
    _add_detector_overrides(sp)

    sp = add("runtime-summary", cmd_runtime_summary,
             "median and range over repeated benchmark runs (<config>_run<N> folders), PLAN.md A4")
    sp.add_argument("parent")
    sp.add_argument("--out", help="default: <parent>/summary.json")

    sp = add("detector-summary", cmd_detector_summary, "P/R/F1 vs fps table for detector candidates, from saved results")
    sp.add_argument("candidates", nargs="+", help="tags used in results/detector_selection/detect_eval_val_<tag>.json")
    sp.add_argument("--results", default="results")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--eval-config", default="configs/eval.yaml")

    sp = add("threads", cmd_threads, "time the detector at several torch thread counts on THIS machine")
    sp.add_argument("source", help="a val clip or image sequence")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--frames", type=int, default=40)
    sp.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16])
    sp.add_argument("--out", default="results/runtime/torch_threads.json")
    _add_detector_overrides(sp)

    sp = add("stress", cmd_stress, "degrade held-out clips and measure the frozen system")
    sp.add_argument("--lock", default="results/heldout_manifest.json")
    sp.add_argument("--results", default="results")
    sp.add_argument("--kinds", nargs="*", choices=["blur", "compression", "low_light", "shake"])
    common(sp)
    sp.set_defaults(config="results/frozen_config.yaml")

    sp = add("errors", cmd_errors, "select and render failures for visual error analysis")
    sp.add_argument("--lock", default="results/heldout_manifest.json")
    sp.add_argument("--results", default="results")
    sp.add_argument("--per-type", type=int, default=2)
    sp.add_argument("--unresolved", type=int, metavar="N",
                    help="instead: render the N worst UNRESOLVED events as frame strips")
    sp.add_argument("--false-events", type=int, metavar="N",
                    help="instead: render the N worst false (unmatched predicted) events as frame strips")
    sp.add_argument("--manifest", help="with --unresolved/--false-events: build details for this manifest (e.g. val) "
                    "from cached detections")
    sp.add_argument("--detect-every", type=int, help="override pipeline.detect_every")
    _add_detector_overrides(sp)
    common(sp)
    sp.set_defaults(config="results/frozen_config.yaml")

    sp = add("plots", cmd_plots, "regenerate every plot from saved results")
    sp.add_argument("--results", default="results")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

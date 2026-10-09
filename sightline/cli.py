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
from sightline.detector import DetectorUnavailable, TorchvisionPersonDetector, read_detection_cache
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


def _detector_factory(cfg: Config):
    def make():
        return TorchvisionPersonDetector(cfg.detector)
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
            detector = TorchvisionPersonDetector(cfg.detector)
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
        locked = lock_manifest(a.manifest, a.out, a.root)
    except (ManifestError, FileNotFoundError) as exc:
        return _unavailable(None, f"cannot lock: {exc}")
    _p(f"locked {len(locked['clips'])} clips -> {a.out}  sha256={locked['manifest_sha256'][:16]}  git={locked['git_commit']}")
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
    from sightline.evaluation.run import get_detections, load_eval_config, prepare_clip, write_json

    cfg, eval_cfg, root = load_config(a.config), load_eval_config(a.eval_config), Path(a.root)
    man = load_manifest(a.manifest)
    rows_all = []
    ths = [round(0.30 + 0.05 * i, 2) for i in range(13)]
    tot = {t: [0, 0, 0] for t in ths}
    for c in man.clips:
        cd = prepare_clip(c, root, eval_cfg)
        dets = get_detections(cd, cfg, root, Path(a.cache), _detector_factory(cfg))
        frames = list(range(1, cd.n_frames + 1, cfg.pipeline.frame_stride))
        for r in sweep_thresholds(cd.gt, cd.ignore, dets, ths, frames):
            for k, i in zip(("tp", "fp", "fn"), range(3)):
                tot[r["threshold"]][i] += r[k]
    from sightline.evaluation.detection import counts_to_metrics
    for t in ths:
        m = counts_to_metrics(*tot[t], threshold=t)
        rows_all.append(m)
        _p(f"thr={t:.2f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    if a.out:
        write_json(a.out, rows_all)
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

    # the held-out set must be locked, unchanged, and fixed BEFORE the config was tuned
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
    agg["frame_stride"] = cfg.pipeline.frame_stride
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
        cfg = load_config(a.config)
        corridor = load_corridor(a.corridor)
        detector = TorchvisionPersonDetector(cfg.detector)
    except (ConfigError, DetectorUnavailable) as exc:
        return _unavailable(None, str(exc))
    all_lat, summaries = None, []
    for src in a.sources:
        run = run_clip(src, cfg, corridor, detector=detector, fps_override=a.fps, realtime=a.realtime)
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
              "realtime_emulation": a.realtime, "declared_input": [cfg.pipeline.declared_width, cfg.pipeline.declared_height]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    _p(f"wrote {a.out}")
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
        detector = TorchvisionPersonDetector(cfg.detector)
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

    cfg = load_config(a.config)
    out = analyze(Path(a.results), Path(a.root), Path(a.lock), Path(a.cache), cfg.detector.name, cfg.detector.score_threshold, a.per_type)
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

    sp = add("lock-manifest", cmd_lock, "hash and lock the held-out manifest")
    sp.add_argument("manifest")
    sp.add_argument("--out", default="results/heldout_manifest.json")
    sp.add_argument("--root", default=".")

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

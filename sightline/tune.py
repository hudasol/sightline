"""Tuning on VALIDATION clips only, then freezing.

Hard rules enforced in code:
  * the locked held-out manifest must already exist (so tuning provably happens
    AFTER the held-out set was fixed),
  * val and the locked test manifest must share no scene / camera / session / clip,
  * only val clips are ever loaded here.

The chosen tracker and the baseline are tuned on the SAME grid, so beating the
baseline cannot be blamed on a hand-tuned straw man.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import yaml

from sightline.config import Config
from sightline.detector import Detector
from sightline.evaluation import detection as det_eval
from sightline.evaluation import tracking as trk_eval
from sightline.evaluation.events import event_metrics, match_events
from sightline.evaluation.run import ClipData, ClipResult, evaluate_clip, get_detections, prepare_clip, write_json
from sightline.ingest import iter_frames
from sightline.manifest import ManifestError, load_manifest, split_check
from sightline.quality import measure_quality


def _pooled_idf1(results: list[ClipResult], kind: str) -> tuple[float, int]:
    summ = trk_eval.summarize({r.cd.clip.id: r.accs[kind] for r in results}, with_overall=True)
    o = summ.get("OVERALL", {})
    return (o.get("idf1") or 0.0), int(o.get("num_switches") or 0)


def _event_f1(results: list[ClipResult], kind: str) -> tuple[float, float]:
    m = event_metrics([r.matches[kind] for r in results], [r.cd.clip.negative for r in results],
                      [r.cd.info.frame_count_reported / r.cd.info.fps for r in results])
    p, rc = m["event_precision"] or 0.0, m["event_recall"] or 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return f1, (m["latency_frames_median"] if m["latency_frames_median"] is not None else 99.0)


def tune(val_manifest: str | Path, locked_test: str | Path, base_cfg: Config, eval_cfg: dict, root: Path,
         cache_dir: Path, detector_factory, out_path: Path, tuning_dir: Path, progress=print) -> dict:
    root = Path(root)
    lock_path = Path(locked_test)
    if not lock_path.is_file():
        raise ManifestError(f"locked held-out manifest {lock_path} does not exist: run `lock-manifest` BEFORE tuning")
    locked = json.loads(lock_path.read_text())
    val = load_manifest(val_manifest)
    test = load_manifest(lock_path)
    problems = split_check(val, test)
    if problems:
        raise ManifestError("split leakage, refusing to tune:\n  " + "\n  ".join(problems))

    cfg = copy.deepcopy(base_cfg)
    progress(f"tuning on {len(val.clips)} val clips (held-out manifest {locked['manifest_sha256'][:12]} is locked)")
    cds: list[ClipData] = [prepare_clip(c, root, eval_cfg) for c in val.clips]
    dets = {cd.clip.id: get_detections(cd, cfg, root, cache_dir, detector_factory) for cd in cds}
    notes: list[str] = []

    # 1. detector operating threshold ---------------------------------------------------
    sweep = []
    for t in [round(0.30 + 0.05 * i, 2) for i in range(13)]:
        tp = fp = fn = 0
        for cd in cds:
            frames = list(range(1, cd.n_frames + 1, cfg.pipeline.frame_stride))
            m = det_eval.evaluate_detections(cd.gt, cd.ignore, dets[cd.clip.id], t, frames, eval_cfg["detection"]["iou"])
            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        sweep.append(det_eval.counts_to_metrics(tp, fp, fn, t))
    bar = eval_cfg["bar"]
    ok = [s for s in sweep if s["precision"] >= bar["detection_precision"] and s["recall"] >= bar["detection_recall"]]
    best = max(ok or sweep, key=lambda s: s["f1"])
    if not ok:
        notes.append(f"NO detector threshold reaches P>={bar['detection_precision']} and R>={bar['detection_recall']} on val; "
                     f"picked best F1 ({best['f1']:.3f}). Consider a different detector or documented sampling.")
    cfg.detector.score_threshold = best["threshold"]
    cfg.tracker.new_track_score = max(cfg.tracker.new_track_score, best["threshold"])
    cfg.baseline_tracker.new_track_score = max(cfg.baseline_tracker.new_track_score, best["threshold"])
    write_json(tuning_dir / "detector_threshold_sweep_val.json", sweep)
    progress(f"detector threshold -> {best['threshold']} (P={best['precision']:.3f} R={best['recall']:.3f} F1={best['f1']:.3f})")

    # 2. trackers, same grid ------------------------------------------------------------------
    grid = list(itertools.product((15, 30, 45), (1, 2, 3), (0.2, 0.3, 0.4)))
    tracker_grid = {"bytetrack": [], "iou": []}
    best_params = {}
    for kind in ("bytetrack", "iou"):
        best_key, best_row = None, None
        for max_age, min_hits, match_iou in grid:
            c = copy.deepcopy(cfg)
            tc = c.tracker if kind == "bytetrack" else c.baseline_tracker
            tc.max_age, tc.min_hits, tc.match_iou = max_age, min_hits, match_iou
            res = [evaluate_clip(cd, c, dets[cd.clip.id], eval_cfg, root, trackers=(kind,)) for cd in cds]
            idf1, sw = _pooled_idf1(res, kind)
            row = {"max_age": max_age, "min_hits": min_hits, "match_iou": match_iou, "idf1": idf1, "id_switches": sw}
            tracker_grid[kind].append(row)
            key = (idf1, -sw)
            if best_key is None or key > best_key:
                best_key, best_row = key, row
        best_params[kind] = best_row
        tc = cfg.tracker if kind == "bytetrack" else cfg.baseline_tracker
        tc.max_age, tc.min_hits, tc.match_iou = best_row["max_age"], best_row["min_hits"], best_row["match_iou"]
        progress(f"{kind}: {best_row}")
    write_json(tuning_dir / "tracker_grid_val.json", tracker_grid)

    # 3. event debounce ----------------------------------------------------------------------
    ev_grid, best_ev, best_ev_key = [], None, None
    for n_open, m_close, cooldown in itertools.product((2, 3, 4), (3, 5, 8), (10, 15, 30)):
        c = copy.deepcopy(cfg)
        c.events.n_open, c.events.m_close, c.events.cooldown = n_open, m_close, cooldown
        res = [evaluate_clip(cd, c, dets[cd.clip.id], eval_cfg, root, trackers=("bytetrack",)) for cd in cds]
        f1, lat = _event_f1(res, "bytetrack")
        row = {"n_open": n_open, "m_close": m_close, "cooldown": cooldown, "event_f1": f1, "median_latency_frames": lat}
        ev_grid.append(row)
        key = (f1 if lat <= bar["median_event_latency_frames"] else f1 - 1.0, -lat)
        if best_ev_key is None or key > best_ev_key:
            best_ev_key, best_ev = key, row
    cfg.events.n_open, cfg.events.m_close, cfg.events.cooldown = best_ev["n_open"], best_ev["m_close"], best_ev["cooldown"]
    write_json(tuning_dir / "event_grid_val.json", ev_grid)
    progress(f"events: {best_ev}")

    # 4. image-quality thresholds from where detection actually fails on val ----------------------
    qrows = _quality_vs_recall(cds, dets, cfg, root)
    write_json(tuning_dir / "quality_vs_recall_val.json", qrows)
    overall_recall = best["recall"]
    for key, field_name in (("luminance", "min_luminance"), ("sharpness", "min_sharpness")):
        bins = qrows[key]
        if bins and bins[0]["recall"] is not None and overall_recall and bins[0]["recall"] < 0.7 * overall_recall:
            setattr(cfg.health, field_name, round(bins[0]["upper_edge"], 2))
            notes.append(f"{field_name} set from val: lowest {key} quintile has recall {bins[0]['recall']:.2f} "
                         f"vs overall {overall_recall:.2f}")
        else:
            notes.append(f"{field_name} left at default: val shows no {key} range where detection clearly fails "
                         "(unvalidated; the stress suite is the evidence for this threshold)")
    notes.append("health.low_conf_mean left at default (unvalidated)")

    out = cfg.to_dict()
    out["provenance"] = {
        "tuned_on_split": "val",
        "val_manifest": str(val_manifest),
        "val_clip_ids": [c.id for c in val.clips],
        "locked_test_manifest_sha256": locked["manifest_sha256"],
        "locked_test_manifest": str(lock_path),
        "git_commit": _git_commit(root),
        "tuned_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector_sweep_best": best,
        "tracker_best": best_params,
        "event_best": best_ev,
        "notes": notes,
        "frozen": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False))
    progress(f"wrote {out_path}")
    return out


def _quality_vs_recall(cds: list[ClipData], dets: dict, cfg: Config, root: Path) -> dict:
    """Per-frame quality against per-frame detection recall, in quintile bins."""
    import numpy as np

    lum, sharp, hit, total = [], [], [], []
    for cd in cds:
        for fr in iter_frames(root / cd.clip.source, cd.clip.fps, stride=cfg.pipeline.frame_stride, info=cd.info):
            if not fr.valid or fr.image is None:
                continue
            gt = cd.gt.get(fr.index, [])
            if not gt:
                continue
            q = measure_quality(fr.image)
            ds = [d for d in dets[cd.clip.id].get(fr.index, []) if d.score >= cfg.detector.score_threshold]
            tp, _fp, fns = det_eval.match_frame(gt, ds, cd.ignore.get(fr.index, []), 0.5)
            lum.append(q.luminance)
            sharp.append(q.sharpness)
            hit.append(tp)
            total.append(len(gt))
    out: dict = {"luminance": [], "sharpness": []}
    for name, vals in (("luminance", lum), ("sharpness", sharp)):
        if len(vals) < 20:
            continue
        arr = np.array(vals)
        edges = np.quantile(arr, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            mask = (arr >= lo) & ((arr <= hi) if i == 4 else (arr < hi))
            t = int(np.array(total)[mask].sum())
            out[name].append({"lower_edge": float(lo), "upper_edge": float(hi), "frames": int(mask.sum()),
                              "recall": (float(np.array(hit)[mask].sum() / t) if t else None)})
    return out


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

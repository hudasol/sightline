"""Confidence calibration: does the detector's score mean what it says, and does low confidence
predict failure?

Two views, both from cached detections and in-scope ground truth (PLAN.md A1):

1. Detection reliability. Every detection at or above the detector floor is matched greedily, highest
   score first, to an unmatched in-scope GT box at IoU >= 0.5 (a hit) or else to an ignore region
   (left out: neither a hit nor a false positive) or else it is a false positive. Detections are binned
   by score; a calibrated detector's precision in a bin equals the bin's mean score. ECE is the
   detection-weighted mean gap.
2. Frame-level failure. For each frame with in-scope people, the mean score of the detections the
   system uses (score >= the frozen threshold) against the frame's recall. If low confidence predicts
   failure, the lowest-confidence frames have the lowest recall. This is what the health monitor's
   low-confidence trigger assumes.

Analysis only: no threshold is chosen from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sightline.evaluation.run import get_detections, in_scope_view, prepare_clip
from sightline.geometry import iou_matrix
from sightline.manifest import load_manifest

if TYPE_CHECKING:
    from sightline.config import Config
    from sightline.types import Box, Detection

EDGES = [round(0.1 * i, 1) for i in range(1, 11)]  # 0.1 .. 1.0


def _classify(gt_boxes: Sequence[Box], ignore_boxes: Sequence[Box], dets: Sequence[Detection],
              iou_thr: float) -> tuple[list[tuple[float, int]], int, int]:
    """Return [(score, label)] with label 1 = hit, 0 = false positive; detections on ignore regions dropped."""
    out = []
    dets = sorted(dets, key=lambda d: -d.score)
    g = np.array(gt_boxes, dtype=float).reshape(-1, 4)
    ig = np.array(ignore_boxes, dtype=float).reshape(-1, 4)
    used = np.zeros(len(g), bool)
    for d in dets:
        box = np.array([d.box], dtype=float)
        if len(g):
            ious = iou_matrix(box, g)[0]
            ious[used] = -1
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                used[j] = True
                out.append((d.score, 1))
                continue
        if len(ig) and float(iou_matrix(box, ig)[0].max()) >= iou_thr:
            continue
        out.append((d.score, 0))
    return out, int(used.sum()), len(g)


def calibration_report(manifest_path: Path, cfg: Config, eval_cfg: dict, root: Path, cache_dir: Path) -> dict:
    man = load_manifest(manifest_path)
    iou_thr = eval_cfg["detection"]["iou"]
    thr = cfg.detector.score_threshold
    scored: list[tuple[float, int]] = []
    frames: list[tuple[float, float]] = []  # (mean used score, recall)
    clips = []
    for c in man.clips:
        if not c.scored_on("detection"):
            continue
        cd = prepare_clip(c, root, eval_cfg)
        dets = get_detections(cd, cfg, root, cache_dir, None)
        gt, ig = in_scope_view(cd)
        clips.append(c.id)
        for f in cd.scored(sorted(dets)):
            g = [b for _i, b in gt.get(f, [])]
            all_d = [d for d in dets[f] if d.score >= cfg.detector.min_score]
            rows, _hits, _n = _classify(g, ig.get(f, []), all_d, iou_thr)
            scored += rows
            used = [d for d in all_d if d.score >= thr]
            if g and used:
                _r, hits, n = _classify(g, ig.get(f, []), used, iou_thr)
                frames.append((float(np.mean([d.score for d in used])), hits / n))
    s = np.array([x for x, _y in scored]) if scored else np.zeros(0)
    y = np.array([v for _x, v in scored]) if scored else np.zeros(0)
    bins, ece = [], 0.0
    lo = cfg.detector.min_score
    for hi in EDGES:
        if hi <= lo:
            continue
        m = (s >= lo) & ((s < hi) if hi < 1.0 else (s <= hi))
        n = int(m.sum())
        prec = float(y[m].mean()) if n else None
        mean_s = float(s[m].mean()) if n else None
        if n:
            ece += n / len(s) * abs(prec - mean_s)
        bins.append({"score_from": lo, "score_to": hi, "detections": n, "precision": prec, "mean_score": mean_s})
        lo = hi
    fr = np.array(frames) if frames else np.zeros((0, 2))
    quint = []
    if len(fr) >= 20:
        edges = np.quantile(fr[:, 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for i in range(5):
            m = (fr[:, 0] >= edges[i]) & ((fr[:, 0] <= edges[i + 1]) if i == 4 else (fr[:, 0] < edges[i + 1]))
            quint.append({"mean_confidence_from": float(edges[i]), "mean_confidence_to": float(edges[i + 1]),
                          "frames": int(m.sum()), "mean_recall": float(fr[m, 1].mean()) if m.any() else None})
    rho = None
    if len(fr) >= 3 and np.std(fr[:, 0]) > 0 and np.std(fr[:, 1]) > 0:
        rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        rho = float(np.corrcoef(rank(fr[:, 0]), rank(fr[:, 1]))[0, 1])
    return {"what": "detector confidence calibration on in-scope GT; analysis only, nothing chosen from it",
            "manifest": manifest_path.as_posix(), "clips": clips, "detector": cfg.detector.name,
            "operating_threshold": thr, "detection_floor": cfg.detector.min_score, "iou": iou_thr,
            "detections_counted": int(len(s)), "reliability_bins": bins, "ece": float(ece) if len(s) else None,
            "frames_with_people_and_detections": int(len(fr)), "recall_by_frame_confidence_quintile": quint,
            "spearman_frame_confidence_vs_recall": rho}

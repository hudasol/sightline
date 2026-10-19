"""Tracking metrics (IDF1, ID switches, ...) via py-motmetrics, with the standard
MOTChallenge treatment of ignore regions."""

from __future__ import annotations

from collections.abc import Iterable

import motmetrics as mm
import numpy as np

from sightline.evaluation.gt import GTFrames, IgnoreFrames
from sightline.geometry import iou_matrix

METRICS = ["idf1", "idp", "idr", "recall", "precision", "num_unique_objects", "mostly_tracked",
           "partially_tracked", "mostly_lost", "num_false_positives", "num_misses", "num_switches",
           "num_fragmentations", "mota", "num_objects"]


def iou_distance(gt_boxes: list, hyp_boxes: list, iou_thr: float) -> np.ndarray:
    """py-motmetrics' distance convention: 1 - IoU, and NaN (match not allowed) where IoU < iou_thr.

    Computed here instead of mm.distances.iou_matrix, which in motmetrics 1.4.0 calls
    np.asfarray, removed in NumPy 2.0.
    """
    d = 1.0 - iou_matrix(np.array(gt_boxes, dtype=float), np.array(hyp_boxes, dtype=float))
    d[d > 1.0 - iou_thr] = np.nan
    return d


def build_accumulator(gt: GTFrames, ignore: IgnoreFrames, hyp_rows: list[tuple], frames: Iterable[int],
                      iou_thr: float = 0.5) -> mm.MOTAccumulator:
    """hyp_rows: (frame, id, x1, y1, x2, y2, score)."""
    hyp: dict[int, list[tuple[int, tuple]]] = {}
    for fr, tid, x1, y1, x2, y2, _s in hyp_rows:
        hyp.setdefault(int(fr), []).append((int(tid), (x1, y1, x2, y2)))

    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g = gt.get(f, [])
        h = hyp.get(f, [])
        if h and ignore.get(f):
            # drop hypotheses that sit on an ignore region and match no real pedestrian
            ig = iou_matrix(np.array([b for _i, b in h]), np.array(ignore[f]))
            real = iou_matrix(np.array([b for _i, b in h]), np.array([b for _i, b in g])) if g else np.zeros((len(h), 0))
            keep = [k for k in range(len(h)) if not (ig[k].max() >= iou_thr and (real.shape[1] == 0 or real[k].max() < iou_thr))]
            h = [h[k] for k in keep]
        g_ids = [i for i, _b in g]
        h_ids = [i for i, _b in h]
        if g and h:
            dist = iou_distance([b for _i, b in g], [b for _i, b in h], iou_thr)
        else:
            dist = np.empty((len(g), len(h)))
        acc.update(g_ids, h_ids, dist, frameid=f)
    return acc


def summarize(accs: dict[str, mm.MOTAccumulator], with_overall: bool = True) -> dict[str, dict]:
    """Per-accumulator metrics, plus an OVERALL row computed over all of them pooled."""
    if not accs:
        return {}
    mh = mm.metrics.create()
    names = list(accs)
    df = mh.compute_many([accs[n] for n in names], metrics=METRICS, names=names, generate_overall=with_overall)
    out: dict[str, dict] = {}
    for name, row in df.iterrows():
        out[str(name)] = {k: _num(row[k]) for k in METRICS}
    return out


def _num(v: object) -> int | float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(f) if f.is_integer() and abs(f) > 1 else f

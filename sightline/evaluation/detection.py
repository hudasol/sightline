"""Detection precision / recall / F1 at a frozen threshold (IoU 0.5)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sightline.evaluation.gt import GTFrames, IgnoreFrames
from sightline.geometry import iou_matrix
from sightline.types import Detection


def match_frame(gt_boxes: list, dets: list[Detection], ignore: list, iou_thr: float) -> tuple[int, list[Detection], list]:
    """Returns (tp, false-positive detections, missed ground-truth (id, box) pairs)."""
    dets = sorted(dets, key=lambda d: -d.score)
    if not dets:
        return 0, [], list(gt_boxes)
    iou_g = iou_matrix(np.array([b for _i, b in gt_boxes]), np.array([d.box for d in dets])) if gt_boxes else np.zeros((0, len(dets)))
    iou_i = iou_matrix(np.array(ignore), np.array([d.box for d in dets])) if ignore else np.zeros((0, len(dets)))
    taken: set[int] = set()
    tp = 0
    fps: list[Detection] = []
    for j in range(len(dets)):
        best, best_i = iou_thr, -1
        for i in range(iou_g.shape[0]):
            if i not in taken and iou_g[i, j] >= best:
                best, best_i = iou_g[i, j], i
        if best_i >= 0:
            taken.add(best_i)
            tp += 1
        elif iou_i.shape[0] and iou_i[:, j].max() >= iou_thr:
            continue  # landed on an ignore region: not a false positive
        else:
            fps.append(dets[j])
    return tp, fps, [g for i, g in enumerate(gt_boxes) if i not in taken]


def evaluate_detections(gt: GTFrames, ignore: IgnoreFrames, dets_by_frame: dict[int, list[Detection]],
                        threshold: float, frames: Iterable[int], iou_thr: float = 0.5) -> dict:
    tp = fp = fn = 0
    for f in frames:
        dets = [d for d in dets_by_frame.get(f, []) if d.score >= threshold]
        a, fps, fns = match_frame(gt.get(f, []), dets, ignore.get(f, []), iou_thr)
        tp, fp, fn = tp + a, fp + len(fps), fn + len(fns)
    return counts_to_metrics(tp, fp, fn, threshold)


def detection_errors(gt: GTFrames, ignore: IgnoreFrames, dets_by_frame: dict[int, list[Detection]],
                     threshold: float, frames: Iterable[int], iou_thr: float = 0.5) -> dict:
    """Every false positive and miss, with frame and box, for visual error analysis."""
    false_pos, misses = [], []
    for f in frames:
        dets = [d for d in dets_by_frame.get(f, []) if d.score >= threshold]
        _tp, fps, fns = match_frame(gt.get(f, []), dets, ignore.get(f, []), iou_thr)
        false_pos += [{"frame": f, "box": list(d.box), "score": d.score} for d in fps]
        misses += [{"frame": f, "gt_id": gid, "box": list(b)} for gid, b in fns]
    return {"false_positives": false_pos, "misses": misses}


def counts_to_metrics(tp: int, fp: int, fn: int, threshold: float | None = None) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
    if threshold is not None:
        out["threshold"] = threshold
    return out


def sweep_thresholds(gt: GTFrames, ignore: IgnoreFrames, dets_by_frame: dict[int, list[Detection]],
                     thresholds: Iterable[float], frames: Iterable[int]) -> list[dict]:
    frames = list(frames)
    return [evaluate_detections(gt, ignore, dets_by_frame, t, frames) for t in thresholds]

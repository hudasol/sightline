"""Stress suite: degrade held-out clips in controlled ways and measure the FROZEN system.

Rules (from the brief): the frozen config is not retuned from these results, and
this is reported as a degradation curve, not used to pick thresholds.

Camera shake displaces everything in the image. Detection and tracking are scored
against ground truth shifted by the same displacement; event labels stay in the
unshifted world frame, so shake shows up honestly as a corridor-geometry failure.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from sightline.config import Config
from sightline.detector import Detector
from sightline.evaluation.run import ClipData, evaluate_clip
from sightline.ingest import iter_frames
from sightline.types import Detection, FrameData

LEVELS: dict[str, list[float]] = {
    "blur": [0, 1, 2, 4, 6],  # gaussian sigma, pixels
    "compression": [100, 50, 25, 10, 5],  # JPEG quality
    "low_light": [1.0, 0.6, 0.4, 0.25, 0.15],  # brightness gain (with sensor-like noise)
    "shake": [0, 3, 6, 12, 20],  # max per-frame displacement, pixels
}
CLEAN = {"blur": 0, "compression": 100, "low_light": 1.0, "shake": 0}


def shake_offsets(n: int, amplitude: float, seed: int) -> dict[int, tuple[int, int]]:
    out = {}
    for i in range(1, n + 1):
        r = np.random.default_rng(seed * 100003 + i)
        a = int(round(amplitude))
        out[i] = (int(r.integers(-a, a + 1)), int(r.integers(-a, a + 1))) if a > 0 else (0, 0)
    return out


def degrade(image: np.ndarray, kind: str, level: float, frame_index: int, offsets: dict | None = None, seed: int = 0) -> np.ndarray:
    if level == CLEAN[kind]:
        return image
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), float(level))
    if kind == "compression":
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image
    if kind == "low_light":
        r = np.random.default_rng(seed * 100003 + frame_index)
        noisy = image.astype(np.float32) * float(level) + r.normal(0, 2.0 / np.sqrt(level), image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "shake":
        dx, dy = (offsets or {}).get(frame_index, (0, 0))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"unknown degradation {kind!r}")


def degraded_frames(frames: Iterable[FrameData], kind: str, level: float, offsets: dict | None, seed: int) -> Iterator[FrameData]:
    for fr in frames:
        if fr.valid and fr.image is not None:
            yield FrameData(fr.index, fr.timestamp, degrade(fr.image, kind, level, fr.index, offsets, seed), True)
        else:
            yield fr


def _shift_gt(cd: ClipData, offsets: dict) -> ClipData:
    def sh(box, o):
        return (box[0] + o[0], box[1] + o[1], box[2] + o[0], box[3] + o[1])

    gt = {f: [(i, sh(b, offsets.get(f, (0, 0)))) for i, b in rows] for f, rows in cd.gt.items()}
    ig = {f: [sh(b, offsets.get(f, (0, 0))) for b in rows] for f, rows in cd.ignore.items()}
    return replace(cd, gt=gt, ignore=ig)  # gt_events deliberately unchanged


def stressed_detections(cd: ClipData, cfg: Config, root: Path, detector: Detector, kind: str, level: float,
                        offsets: dict, seed: int) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    src = iter_frames(root / cd.clip.source, cd.clip.fps, stride=1, info=cd.info)
    for fr in degraded_frames(src, kind, level, offsets, seed):
        out[fr.index] = detector.detect(fr.image) if fr.valid and fr.image is not None else []
    return out


def run_stress(cds: list[ClipData], cfg: Config, eval_cfg: dict, root: Path, detector: Detector,
               kinds: tuple[str, ...] = tuple(LEVELS), seed: int = 0, progress=print) -> list[dict]:
    """Frozen config in, degradation curves out. Nothing here changes cfg."""
    cfg = copy.deepcopy(cfg)
    rows = []
    for kind in kinds:
        for level in LEVELS[kind]:
            pooled = {"tp": 0, "fp": 0, "fn": 0}
            results = []
            for cd in cds:
                offsets = shake_offsets(cd.n_frames, level, seed) if kind == "shake" else {}
                cd_eval = _shift_gt(cd, offsets) if kind == "shake" and level else cd
                dets = stressed_detections(cd, cfg, root, detector, kind, level, offsets, seed)
                r = evaluate_clip(cd_eval, cfg, dets, eval_cfg, root, trackers=("bytetrack",))
                results.append(r)
                for k in pooled:
                    pooled[k] += r.det_counts[k]
            from sightline.evaluation import tracking as trk_eval
            from sightline.evaluation.events import event_metrics
            from sightline.evaluation.run import safety_metrics

            trk = trk_eval.summarize({r.cd.clip.id: r.accs["bytetrack"] for r in results}).get("OVERALL", {})
            ev = event_metrics([r.matches["bytetrack"] for r in results], [r.cd.clip.negative for r in results],
                               [r.cd.info.frame_count_reported / r.cd.info.fps for r in results])
            safety = safety_metrics(results, "bytetrack")
            p = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if pooled["tp"] + pooled["fp"] else 0.0
            rc = pooled["tp"] / (pooled["tp"] + pooled["fn"]) if pooled["tp"] + pooled["fn"] else 0.0
            row = {"degradation": kind, "level": level, "detection_precision": p, "detection_recall": rc,
                   "idf1": trk.get("idf1"), "id_switches": trk.get("num_switches"),
                   "event_precision": ev["event_precision"], "event_recall": ev["event_recall"],
                   "latency_frames_median": ev["latency_frames_median"],
                   "false_clear_rate": safety["false_clear_rate"],
                   "non_ok_frame_fraction": safety["non_ok_frame_fraction"]}
            rows.append(row)
            progress(f"{kind}={level}: IDF1={row['idf1']} event_recall={row['event_recall']} "
                     f"false_clear={row['false_clear_rate']} non_ok={row['non_ok_frame_fraction']:.2f}")
    return rows

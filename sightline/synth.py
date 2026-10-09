"""Synthetic clip generator.

Purpose: tests and a no-download end-to-end demo. It is NOT evidence about real
performance and nothing in results/ is computed from it. It writes a video, a
MOT-format ground truth, and a detection cache that mimics a detector with
jitter, a missed stretch behind a pillar, and a low-confidence partial occlusion,
so the tracker's low-confidence stage has something real to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sightline.detector import write_detection_cache
from sightline.evaluation.gt import write_mot_gt
from sightline.types import Detection

PERSON_W, PERSON_H = 40, 110


@dataclass
class SynthClip:
    video: Path
    cache: Path
    gt: Path
    size: tuple[int, int]
    fps: float
    n_frames: int
    corridor: list[tuple[float, float]]
    gt_boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = field(default_factory=dict)
    detections: dict[int, list[Detection]] = field(default_factory=dict)


def _person_box(cx: float, foot_y: float) -> tuple[float, float, float, float]:
    return (cx - PERSON_W / 2, foot_y - PERSON_H, cx + PERSON_W / 2, foot_y)


def _overlap_fraction(box, rect) -> float:
    x1, y1 = max(box[0], rect[0]), max(box[1], rect[1])
    x2, y2 = min(box[2], rect[2]), min(box[3], rect[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / area if area > 0 else 0.0


def make_synthetic_clip(
    out_dir: str | Path,
    name: str = "synthetic",
    n_frames: int = 150,
    fps: float = 15.0,
    size: tuple[int, int] = (640, 360),
    seed: int = 0,
    with_crosser: bool = True,
    with_pillar_walker: bool = True,
    write_video: bool = True,
) -> SynthClip:
    """Scene: a corridor box on the floor; a 'crosser' walks through it left to right;
    a 'bystander' walks past well above it (never enters); a 'pillar walker' disappears
    behind a pillar for a few frames, partly visible at its edges."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    corridor = [(250.0, 215.0), (400.0, 215.0), (400.0, 345.0), (250.0, 345.0)]
    pillar = (470, 90, 510, 300)

    bg = np.clip(rng.normal(110, 28, (h, w, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    colors = {1: (60, 60, 220), 2: (60, 200, 60), 3: (220, 120, 40)}

    def positions(f: int) -> dict[int, tuple[float, float]]:
        pos: dict[int, tuple[float, float]] = {}
        if with_crosser:
            pos[1] = (40 + 3.7 * (f - 1), 300.0)  # feet at y=300: inside the corridor band
        pos[2] = (600 - 2.4 * (f - 1), 150.0)  # bystander: feet far above the corridor
        if with_pillar_walker:
            pos[3] = (330 + 2.2 * (f - 1), 255.0 - 0.0)  # crosses the pillar region around frames 60-75
        return pos

    gt_rows, gt_boxes, dets = [], {}, {}
    writer = None
    video = out / f"{name}.mp4"
    if write_video:
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(1, n_frames + 1):
        img = bg.copy()
        frame_gt, frame_det = [], []
        for pid, (cx, fy) in positions(f).items():
            box = _person_box(cx, fy)
            if box[2] < 0 or box[0] > w:
                continue  # out of view
            occ = _overlap_fraction(box, pillar) if pid == 3 else 0.0
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[pid], -1)
            cv2.circle(img, ((x1 + x2) // 2, y1 + 12), 9, (200, 220, 240), -1)
            frame_gt.append((pid, box))
            gt_rows.append((f, pid, box, 1.0 - occ))
            if occ > 0.75:
                continue  # hidden: no detection at all
            score = 0.32 if occ > 0.2 else float(np.clip(rng.normal(0.88, 0.05), 0.55, 0.99))
            jit = rng.normal(0, 1.5, 4)
            frame_det.append(Detection(tuple(float(v) for v in (np.array(box) + jit)), score))
        if rng.random() < 0.05:  # an occasional low-confidence false detection
            fx, fy = rng.uniform(20, w - 80), rng.uniform(20, h - 140)
            frame_det.append(Detection((fx, fy, fx + 35, fy + 100), 0.22))
        cv2.rectangle(img, pillar[:2], pillar[2:], (40, 40, 40), -1)  # pillar drawn on top
        if writer is not None:
            writer.write(img)
        gt_boxes[f] = frame_gt
        dets[f] = frame_det
    if writer is not None:
        writer.release()

    gt_path = out / f"{name}.gt.txt"
    write_mot_gt(gt_path, gt_rows)
    cache = out / f"{name}.detections.jsonl"
    write_detection_cache(cache, {"detector": "synthetic", "min_score": 0.1, "note": "not a real detector"}, dets)
    return SynthClip(video, cache, gt_path, size, fps, n_frames, corridor, gt_boxes, dets)


class ColorBlobDetector:
    """A tiny image-dependent 'detector' for the synthetic scene only: finds saturated
    rectangles. Used by tests and the stress-suite smoke test, because degradations
    (blur, low light, compression) then genuinely change its output. Not a real detector."""

    name = "color_blob (synthetic only)"

    def detect(self, image: np.ndarray) -> list[Detection]:
        sat = (image.max(axis=2).astype(int) - image.min(axis=2).astype(int)) > 80
        n, _labels, stats, _c = cv2.connectedComponentsWithStats(sat.astype(np.uint8), connectivity=8)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < 300:
                continue
            score = float(min(0.95, 0.25 + 0.7 * area / (PERSON_W * PERSON_H)))
            dets.append(Detection((float(x), float(y), float(x + w), float(y + h)), score))
        return dets

    def info(self) -> dict:
        return {"detector": self.name, "min_score": 0.1, "note": "synthetic scene only"}

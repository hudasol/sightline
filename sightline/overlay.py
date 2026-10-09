"""Saved overlays. Deliberately plain: boxes, ids, the corridor, and the honest status."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sightline.geometry import Polygon, box_ref_point
from sightline.pipeline import FrameResult
from sightline.types import CorridorStatus, HealthState

_COLORS = {  # BGR
    CorridorStatus.CLEAR: (80, 170, 60),
    CorridorStatus.OCCUPIED: (50, 50, 220),
    CorridorStatus.UNKNOWN: (30, 170, 240),
}


def draw_frame(image: np.ndarray, result: FrameResult, polygon: Polygon) -> np.ndarray:
    img = image.copy()
    pts = np.array(polygon.vertices, dtype=np.int32).reshape(-1, 1, 2)
    color = _COLORS[result.corridor]
    layer = img.copy()
    cv2.fillPoly(layer, [pts], color)
    img = cv2.addWeighted(layer, 0.25, img, 0.75, 0)
    cv2.polylines(img, [pts], True, color, 2)

    for d in result.detections:  # raw detections, thin grey
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
    for t in result.reported_tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(img, f"#{t.id} {t.score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
        rx, ry = box_ref_point(t.box)
        cv2.circle(img, (int(rx), int(ry)), 4, (0, 0, 255), -1)

    banner = f"{result.corridor.value} | health {result.health.state.value}"
    if result.health.reasons:
        banner += " (" + ", ".join(result.health.reasons[:2]) + ")"
    bar_color = (30, 30, 30) if result.health.state == HealthState.OK else (0, 120, 220)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), bar_color, -1)
    cv2.putText(img, f"f{result.frame_index}  {banner}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, e in enumerate(result.events):
        cv2.putText(img, f"EVENT {e.kind} #{e.event['event_id']}", (6, 40 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return img


class OverlayWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], polygon: Polygon) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.polygon = polygon
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), size)
        self.ok = self.writer.isOpened()

    def write(self, image: np.ndarray | None, result: FrameResult) -> None:
        if self.ok and image is not None:
            self.writer.write(draw_frame(image, result, self.polygon))

    def close(self) -> None:
        self.writer.release()

"""Corridor geometry in image space, plus box utilities.

The corridor is a polygon in pixel coordinates. That assumes the camera keeps
the corridor in roughly the same place in the image. It is NOT valid for a
camera that moves a lot relative to the corridor, and it makes no metric claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sightline.types import Box


class GeometryError(ValueError):
    pass


def box_ref_point(box: Box) -> tuple[float, float]:
    """Bottom-centre of the box: where the person touches the floor."""
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy arrays -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Proper or touching intersection of segments p1p2 and p3p4."""

    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_seg(a, b, c) -> bool:
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and 0 not in (o1, o2, o3, o4):
        return True
    if o1 == 0 and on_seg(p1, p2, p3):
        return True
    if o2 == 0 and on_seg(p1, p2, p4):
        return True
    if o3 == 0 and on_seg(p3, p4, p1):
        return True
    if o4 == 0 and on_seg(p3, p4, p2):
        return True
    return False


@dataclass(frozen=True)
class Polygon:
    vertices: tuple[tuple[float, float], ...]

    @staticmethod
    def create(vertices, frame_size: tuple[int, int] | None = None) -> "Polygon":
        """Validate and build. frame_size is (width, height) if known."""
        try:
            pts = tuple((float(x), float(y)) for x, y in vertices)
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"polygon vertices must be (x, y) number pairs: {exc}") from exc
        if len(pts) < 3:
            raise GeometryError(f"polygon needs at least 3 vertices, got {len(pts)}")
        if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pts):
            raise GeometryError("polygon has non-finite coordinates")
        poly = Polygon(pts)
        if poly._self_intersects():
            raise GeometryError("polygon is self-intersecting")
        if poly.area() < 1e-6:
            raise GeometryError("polygon has zero area")
        if frame_size is not None:
            w, h = frame_size
            for x, y in pts:
                if not (0 <= x <= w and 0 <= y <= h):
                    raise GeometryError(f"vertex ({x}, {y}) is outside the {w}x{h} frame")
        return poly

    def area(self) -> float:
        pts = self.vertices
        s = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    def _self_intersects(self) -> bool:
        pts = self.vertices
        n = len(pts)
        for i in range(n):
            a1, a2 = pts[i], pts[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or (j + 1) % n == i or (i + 1) % n == j:
                    continue  # adjacent edges share a vertex by design
                b1, b2 = pts[j], pts[(j + 1) % n]
                if _segments_intersect(a1, a2, b1, b2):
                    return True
        return False

    def contains(self, point: tuple[float, float]) -> bool:
        """Ray casting. Points exactly on an edge count as inside."""
        x, y = point
        pts = self.vertices
        n = len(pts)
        inside = False
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if _point_on_segment((x, y), (x1, y1), (x2, y2)):
                return True
            if (y1 > y) != (y2 > y):
                x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_at:
                    inside = not inside
        return inside

    def nearest_edge(self, point: tuple[float, float]) -> int:
        """Index of the polygon edge closest to the point (edge i = v[i] -> v[i+1])."""
        pts = self.vertices
        best, best_d = 0, float("inf")
        for i in range(len(pts)):
            d = _dist_to_segment(point, pts[i], pts[(i + 1) % len(pts)])
            if d < best_d:
                best, best_d = i, d
        return best


def _point_on_segment(p, a, b, eps: float = 1e-9) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > eps * max(1.0, abs(b[0] - a[0]) + abs(b[1] - a[1])):
        return False
    return min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps


def _dist_to_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(p[0] - cx, p[1] - cy)

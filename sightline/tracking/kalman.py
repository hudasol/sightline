"""Constant-velocity Kalman filter over (cx, cy, aspect, height).

Same state parameterisation as SORT / ByteTrack. Geometry only: no appearance.
"""

from __future__ import annotations

import numpy as np

from sightline.types import Box

_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160


def box_to_z(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=float)


def mean_to_box(mean: np.ndarray) -> Box:
    cx, cy, a, h = mean[:4]
    h = max(float(h), 1e-3)
    w = float(a) * h
    return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


class KalmanBoxFilter:
    def __init__(self) -> None:
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, 4 + i] = 1.0
        self._H = np.eye(4, 8)

    def initiate(self, box: Box) -> tuple[np.ndarray, np.ndarray]:
        z = box_to_z(box)
        mean = np.concatenate([z, np.zeros(4)])
        h = z[3]
        std = [2 * _STD_POS * h, 2 * _STD_POS * h, 1e-2, 2 * _STD_POS * h,
               10 * _STD_VEL * h, 10 * _STD_VEL * h, 1e-5, 10 * _STD_VEL * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(mean[3], 1e-3)
        std_pos = [_STD_POS * h, _STD_POS * h, 1e-2, _STD_POS * h]
        std_vel = [_STD_VEL * h, _STD_VEL * h, 1e-5, _STD_VEL * h]
        q = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray]:
        z = box_to_z(box)
        h = max(mean[3], 1e-3)
        r = np.diag(np.square([_STD_POS * h, _STD_POS * h, 1e-1, _STD_POS * h]))
        s = self._H @ cov @ self._H.T + r
        k = cov @ self._H.T @ np.linalg.inv(s)
        mean = mean + k @ (z - self._H @ mean)
        cov = (np.eye(8) - k @ self._H) @ cov
        return mean, cov

"""Cheap image-quality measures, computed at a fixed resolution so thresholds
mean the same thing for a 640p and a 1080p clip."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_WIDTH = 320


@dataclass(frozen=True)
class Quality:
    luminance: float  # mean gray level, 0-255
    sharpness: float  # variance of the Laplacian


def measure_quality(image: np.ndarray) -> Quality:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if w != _WIDTH:
        scale = _WIDTH / w
        gray = cv2.resize(gray, (_WIDTH, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return Quality(luminance=float(gray.mean()), sharpness=float(lap.var()))

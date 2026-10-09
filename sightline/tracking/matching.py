"""Association helpers shared by both trackers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from sightline.geometry import iou_matrix
from sightline.types import Box


def associate(track_boxes: list[Box], det_boxes: list[Box], min_iou: float):
    """Hungarian assignment on 1 - IoU, rejecting pairs below min_iou.

    Returns (matches, unmatched_track_idx, unmatched_det_idx).
    """
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    iou = iou_matrix(np.array(track_boxes), np.array(det_boxes))
    cost = 1.0 - iou
    rows, cols = linear_sum_assignment(cost)
    matches, used_t, used_d = [], set(), set()
    for r, c in zip(rows, cols):
        if iou[r, c] >= min_iou:
            matches.append((int(r), int(c)))
            used_t.add(int(r))
            used_d.add(int(c))
    ut = [i for i in range(len(track_boxes)) if i not in used_t]
    ud = [j for j in range(len(det_boxes)) if j not in used_d]
    return matches, ut, ud

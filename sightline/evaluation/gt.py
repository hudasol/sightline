"""Ground-truth loading. MOTChallenge format: frame,id,x,y,w,h,flag,class,visibility.

Pedestrians (class 1, flag 1) are scored. Rows with flag 0, or in distractor
classes (person on vehicle, static person, reflection, ...), are IGNORE regions:
a detection that lands on one is neither a hit nor a false positive.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

Box = tuple[float, float, float, float]
GTFrames = dict[int, list[tuple[int, Box]]]  # frame -> [(gt id, xyxy)]
IgnoreFrames = dict[int, list[Box]]

PEDESTRIAN = 1
DEFAULT_IGNORE_CLASSES = (2, 7, 8, 12)  # MOT17: person on vehicle, static person, distractor, reflection


def load_mot_gt(path: str | Path, min_visibility: float = 0.0,
                ignore_classes: tuple[int, ...] = DEFAULT_IGNORE_CLASSES) -> tuple[GTFrames, IgnoreFrames]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ground truth not found: {p}")
    gt: GTFrames = defaultdict(list)
    ignore: IgnoreFrames = defaultdict(list)
    for ln, line in enumerate(p.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"{p}:{ln}: expected at least 6 comma-separated fields")
        frame, tid = int(float(parts[0])), int(float(parts[1]))
        x, y, w, h = (float(v) for v in parts[2:6])
        flag = int(float(parts[6])) if len(parts) > 6 else 1
        cls = int(float(parts[7])) if len(parts) > 7 else PEDESTRIAN
        vis = float(parts[8]) if len(parts) > 8 else 1.0
        box = (x, y, x + w, y + h)
        if flag == 0 or cls in ignore_classes:
            ignore[frame].append(box)
        elif cls == PEDESTRIAN and vis >= min_visibility:
            gt[frame].append((tid, box))
    return dict(gt), dict(ignore)


def write_mot_gt(path: str | Path, rows: list[tuple[int, int, Box, float]]) -> None:
    """rows: (frame, id, xyxy box, visibility)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for frame, tid, (x1, y1, x2, y2), vis in rows:
            f.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,{vis:.3f}\n")

"""Own-clip annotations: which frames were annotated, and import from CVAT.

Sparse annotation (docs/PLAN.md section 14, risk 1) is only honest if the scorer knows
which frames were looked at: an annotated frame with no boxes is a real negative, an
unannotated frame is unknown and must not be scored. A clip says so with an
`annotated_frames` file; without one every frame counts as annotated.
"""

from __future__ import annotations

import json
from pathlib import Path

class AnnotationError(ValueError):
    pass


def parse_frame_spec(spec: str) -> list[int]:
    """"1-600" (inclusive), "1-600:10" (every 10th from 1), comma-separated parts, or a file
    with one frame number per line. Frames are 1-based."""
    p = Path(spec)
    if p.is_file():
        parts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        frames = [int(x) for x in parts]
    else:
        frames = []
        for part in spec.split(","):
            part = part.strip()
            rng, _, step = part.partition(":")
            lo, _, hi = rng.partition("-")
            try:
                a, b = int(lo), int(hi or lo)
                s = int(step or 1)
            except ValueError as exc:
                raise AnnotationError(f"bad frame spec {part!r}; use e.g. 1-600 or 1-600:10") from exc
            if a < 1 or b < a or s < 1:
                raise AnnotationError(f"bad frame spec {part!r}: frames are 1-based and ranges ascending")
            frames.extend(range(a, b + 1, s))
    if not frames or min(frames) < 1:
        raise AnnotationError(f"frame spec {spec!r} selects no valid (1-based) frame")
    return sorted(set(frames))


def load_annotated_frames(path: str | Path) -> set[int]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"annotated-frames file not found: {p}")
    data = json.loads(p.read_text())
    frames = data["frames"] if isinstance(data, dict) else data
    if not isinstance(frames, list) or not frames or not all(isinstance(f, int) and f >= 1 for f in frames):
        raise AnnotationError(f"{p}: needs a non-empty list of 1-based frame numbers")
    return set(frames)

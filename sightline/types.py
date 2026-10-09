"""Small shared data types. Boxes are (x1, y1, x2, y2) in pixels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float


@dataclass
class FrameData:
    """One frame from a source.

    index is 1-based and refers to the ORIGINAL source, so evaluation stays
    aligned with ground truth even if frames are skipped or sampled.
    valid=False means the frame could not be decoded; image is then None.
    """

    index: int
    timestamp: float
    image: np.ndarray | None
    valid: bool = True


class HealthState(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def severity(self) -> int:
        return {"OK": 0, "DEGRADED": 1, "UNAVAILABLE": 2}[self.value]


class CorridorStatus(str, Enum):
    """What the operator is told about the corridor.

    CLEAR is only ever produced when health is OK and nobody is inside.
    UNKNOWN is the honest answer whenever the system cannot tell.
    """

    CLEAR = "CLEAR"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    EXPIRED = "expired"


@dataclass
class TrackView:
    """Read-only snapshot of a track for logging and events."""

    id: int
    box: Box
    score: float
    state: TrackState
    age: int
    hits: int
    time_since_update: int
    last_seen_ts: float
    trajectory: list[tuple[float, float, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "box": [round(v, 2) for v in self.box],
            "score": round(self.score, 4),
            "state": self.state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "last_seen_ts": round(self.last_seen_ts, 4),
        }

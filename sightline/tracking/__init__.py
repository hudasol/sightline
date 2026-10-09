from sightline.tracking.bytetrack import ByteTracker
from sightline.tracking.iou_tracker import IoUTracker
from sightline.tracking.state import Track

__all__ = ["ByteTracker", "IoUTracker", "Track", "make_tracker"]


def make_tracker(cfg, high_thresh: float, low_thresh: float):
    """Build the tracker named in a TrackerConfig.

    high_thresh is the detector's frozen operating threshold; low_thresh is its floor.
    """
    if cfg.kind == "bytetrack":
        return ByteTracker(cfg, high_thresh, low_thresh)
    if cfg.kind == "iou":
        return IoUTracker(cfg, high_thresh, low_thresh)
    raise ValueError(f"unknown tracker kind {cfg.kind!r}")

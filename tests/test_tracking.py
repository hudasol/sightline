import numpy as np

from sightline.config import TrackerConfig
from sightline.tracking import ByteTracker, IoUTracker
from sightline.tracking.kalman import KalmanBoxFilter, mean_to_box
from sightline.types import Detection, TrackState


def det(x, y=100, score=0.9, w=40, h=100):
    return Detection((x, y, x + w, y + h), score)


def run(tracker, frames):
    """frames: list of detection lists. Returns per-frame (reported ids, expired ids)."""
    out = []
    for i, dets in enumerate(frames, start=1):
        o = tracker.update(dets, i, i / 15.0)
        out.append(({t.id for t in o.reported}, {t.id for t in o.expired}))
    return out


def cfg(**kw):
    base = dict(max_age=5, min_hits=3, match_iou=0.3, new_track_score=0.6)
    base.update(kw)
    return TrackerConfig(**base)


def test_steady_motion_keeps_one_id():
    t = ByteTracker(cfg(), 0.5, 0.1)
    res = run(t, [[det(10 + 3 * i)] for i in range(40)])
    ids = set().union(*(r[0] for r in res))
    assert ids == {1}
    assert res[0][0] == set() and res[1][0] == set() and res[2][0] == {1}  # confirmed on the 3rd hit


def test_single_frame_false_detection_is_never_reported():
    t = ByteTracker(cfg(), 0.5, 0.1)
    frames = [[det(10)], [], [], [], []]
    res = run(t, frames)
    assert all(r[0] == set() for r in res)
    assert t.tracks == []  # tentative track dropped immediately


def test_low_confidence_stage_keeps_identity_that_the_baseline_loses():
    # 10 confident frames, 5 frames at score 0.3 (between low and high), then confident again.
    frames = [[det(10 + 3 * i)] for i in range(10)]
    frames += [[det(10 + 3 * i, score=0.3)] for i in range(10, 15)]
    frames += [[det(10 + 3 * i)] for i in range(15, 25)]
    c = cfg(max_age=2)  # shorter than the low-score stretch
    byte = set().union(*(r[0] for r in run(ByteTracker(c, 0.5, 0.1), frames)))
    base = set().union(*(r[0] for r in run(IoUTracker(c, 0.5, 0.1), frames)))
    assert byte == {1}, "second stage should bridge the low-confidence stretch"
    assert len(base) > 1, "baseline discards low scores, expires, and starts a new identity"


def test_lost_then_expired_then_new_id_never_reused():
    t = ByteTracker(cfg(max_age=3), 0.5, 0.1)
    frames = [[det(10 + 3 * i)] for i in range(6)] + [[] for _ in range(6)] + [[det(200 + 3 * i)] for i in range(6)]
    res = run(t, frames)
    expired_at = [i for i, r in enumerate(res) if r[1]]
    assert len(expired_at) == 1
    assert expired_at[0] == 6 + 3  # first miss at index 6, expired when misses exceed max_age=3 (4th miss)
    later_ids = set().union(*(r[0] for r in res[12:]))
    assert later_ids == {2}, "a returning person after expiry gets a NEW id; ids are never reused"


def test_state_machine_confirmed_lost_expired():
    t = ByteTracker(cfg(max_age=2), 0.5, 0.1)
    for i in range(4):
        t.update([det(10 + 3 * i)], i + 1, i / 15)
    tr = t.tracks[0]
    assert tr.state == TrackState.CONFIRMED
    t.update([], 5, 5 / 15)
    assert tr.state == TrackState.LOST and tr.time_since_update == 1
    t.update([], 6, 6 / 15)
    assert tr.state == TrackState.LOST
    out = t.update([], 7, 7 / 15)
    assert tr.state == TrackState.EXPIRED and out.expired == [tr] and t.tracks == []


def test_reentry_within_max_age_keeps_id():
    t = ByteTracker(cfg(max_age=6), 0.5, 0.1)
    frames = [[det(10 + 3 * i)] for i in range(8)] + [[] for _ in range(3)] + [[det(10 + 3 * i)] for i in range(11, 20)]
    ids = set().union(*(r[0] for r in run(t, frames)))
    assert ids == {1}


def test_track_state_exposes_required_fields():
    t = ByteTracker(cfg(), 0.5, 0.1)
    for i in range(5):
        t.update([det(10 + 3 * i, score=0.8)], i + 1, (i + 1) / 15)
    v = t.tracks[0].view()
    assert v.id == 1 and v.age == 5 and v.hits == 5 and v.state == TrackState.CONFIRMED
    assert v.last_seen_ts == 5 / 15 and 0.7 < v.score < 0.9
    assert len(v.trajectory) == 5 and v.trajectory[-1][2] == 5


def test_trajectory_is_bounded():
    t = ByteTracker(cfg(trajectory_len=10), 0.5, 0.1)
    for i in range(40):
        t.update([det(10 + 2 * i)], i + 1, i / 15)
    assert len(t.tracks[0].trajectory) == 10


def test_two_people_keep_separate_ids():
    t = ByteTracker(cfg(), 0.5, 0.1)
    res = run(t, [[det(10 + 3 * i), det(400 - 3 * i)] for i in range(20)])
    assert set().union(*(r[0] for r in res)) == {1, 2}


def test_kalman_predicts_along_velocity():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate((0, 100, 40, 200))
    for i in range(1, 15):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, (5 * i, 100, 5 * i + 40, 200))
    mean, cov = kf.predict(mean, cov)
    x1 = mean_to_box(mean)[0]
    assert 70 < x1 < 80  # last x1 = 70, moving +5 per frame

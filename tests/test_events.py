import pytest

from sightline.config import EventConfig
from sightline.events import EventEngine, EventStatus
from sightline.geometry import Polygon
from sightline.tracking.state import Track
from sightline.types import TrackState

POLY = Polygon.create([(0, 0), (100, 0), (100, 100), (0, 100)])  # edges: 0 top 1 right 2 bottom 3 left
SIZE = (400, 400)


def mk(n_open=3, m_close=3, cooldown=5, margin=0.05):
    return EventEngine(EventConfig(n_open=n_open, m_close=m_close, cooldown=cooldown, edge_margin_frac=margin), POLY, SIZE)


def box(x, y):
    return (x - 5, y - 20, x + 5, y)  # reference point (x, y)


def track(tid=1, x=50, y=50, observed=True, state=TrackState.CONFIRMED, frame=1, traj=None):
    t = Track(tid, box(x, y), 0.9, frame, frame / 15)
    t.state = state
    t.time_since_update = 0 if observed else 1
    if traj is not None:
        t.trajectory.clear()
        t.trajectory.extend(traj)
    return t


def feed(eng, points, start=1, tid=1):
    """points: list of (x, y) or None for lost. Returns all emitted records."""
    recs = []
    for i, p in enumerate(points, start=start):
        t = track(tid, *(p or (0, 0)), observed=p is not None, state=TrackState.CONFIRMED if p else TrackState.LOST, frame=i)
        if p is not None:
            t.trajectory.clear()
            t.trajectory.append((p[0], p[1], i))
        recs += eng.update(i, i / 15, [t], [])
    return recs


IN, OUT = (50, 50), (250, 250)


def test_sustained_occupancy_is_one_event_not_one_per_frame():
    eng = mk()
    recs = feed(eng, [IN] * 60)
    assert [r.kind for r in recs] == ["open"]
    assert len(eng.all_events()) == 1 and eng.occupied


def test_debounce_rejects_a_flicker_shorter_than_n_open():
    eng = mk(n_open=3)
    recs = feed(eng, [OUT, IN, IN, OUT, OUT, IN, IN, OUT])
    assert recs == [] and eng.all_events() == [] and not eng.occupied


def test_open_frame_and_latency_from_first_inside():
    eng = mk(n_open=3)
    feed(eng, [OUT] * 4 + [IN] * 5, start=1)
    ev = eng.all_events()[0]
    assert ev.first_inside_frame == 5 and ev.open_frame == 7
    assert ev.latency_frames_from_first_inside == 2  # n_open - 1


def test_close_after_m_close_and_crossing_type():
    eng = mk(m_close=3)
    # enter near the left edge (edge 3), leave near the right edge (edge 1)
    recs = feed(eng, [(5, 50), (30, 50), (60, 50), (95, 50), OUT, OUT, OUT])
    assert [r.kind for r in recs] == ["open", "close"]
    ev = eng.all_events()[0]
    assert ev.status == EventStatus.CLOSED and ev.close_reason == "left_corridor"
    assert ev.entry_edge == 3 and ev.exit_edge == 1 and ev.type == "crossing"


def test_enter_and_leave_by_the_same_edge_is_an_entry():
    eng = mk(m_close=2)
    feed(eng, [(5, 50), (10, 50), (15, 50), (20, 50), (10, 50), (3, 50), (-30, 50), (-40, 50)])
    ev = eng.all_events()[0]
    assert ev.status == EventStatus.CLOSED and ev.type == "entry"


def test_boundary_jitter_shorter_than_m_close_does_not_close_or_duplicate():
    eng = mk(m_close=3)
    recs = feed(eng, [IN] * 4 + [OUT, OUT] + [IN] * 4 + [OUT, OUT] + [IN] * 3)
    assert [r.kind for r in recs] == ["open"]
    assert len(eng.all_events()) == 1 and eng.all_events()[0].status == EventStatus.ACTIVE


def test_reentry_within_cooldown_reopens_the_same_event():
    eng = mk(m_close=2, cooldown=5)
    recs = feed(eng, [IN] * 4 + [OUT] * 2 + [OUT] * 2 + [IN] * 3)
    assert [r.kind for r in recs] == ["open", "close", "reopen"]
    evs = eng.all_events()
    assert len(evs) == 1 and evs[0].reopen_count == 1 and evs[0].status == EventStatus.ACTIVE


def test_reentry_after_cooldown_is_a_new_event():
    eng = mk(m_close=2, cooldown=3)
    feed(eng, [IN] * 4 + [OUT] * 10 + [IN] * 4)
    assert len(eng.all_events()) == 2


def test_lost_frames_do_not_advance_the_close_counter():
    eng = mk(m_close=3)
    feed(eng, [IN] * 4 + [None] * 20)  # track unobserved for 20 frames
    ev = eng.all_events()[0]
    assert ev.status == EventStatus.ACTIVE and eng.occupied  # absence of evidence is not evidence of leaving


def test_track_expiring_mid_event_away_from_border_is_unresolved_not_closed():
    eng = mk()
    feed(eng, [IN] * 4)
    t = track(1, *IN, observed=False, state=TrackState.EXPIRED)
    t.last_det_box = box(*IN)
    recs = eng.update(10, 10 / 15, [], [t])
    ev = eng.all_events()[0]
    assert [r.kind for r in recs] == ["unresolved"]
    assert ev.status == EventStatus.UNRESOLVED and ev.close_reason == "track_lost"


def test_track_expiring_near_frame_border_closes_as_left_view():
    eng = mk()
    feed(eng, [(50, 50), (50, 60), (50, 70), (50, 80)])
    t = track(1, 50, 395, observed=False, state=TrackState.EXPIRED)
    t.last_det_box = (45, 370, 55, 399)  # bottom edge of a 400px frame
    eng.update(12, 12 / 15, [], [t])
    ev = eng.all_events()[0]
    assert ev.status == EventStatus.CLOSED and ev.close_reason == "left_view"


def test_tentative_tracks_never_open_events():
    eng = mk()
    recs = []
    for i in range(1, 10):
        recs += eng.update(i, i / 15, [track(1, *IN, state=TrackState.TENTATIVE, frame=i)], [])
    assert recs == [] and eng.all_events() == []


def test_finalize_marks_open_events_unresolved():
    eng = mk()
    feed(eng, [IN] * 5)
    recs = eng.finalize(99, 99 / 15)
    assert [r.kind for r in recs] == ["unresolved"]
    assert eng.all_events()[0].close_reason == "stream_end"


def test_track_confirmed_while_already_inside_opens_without_extra_delay():
    eng = mk(n_open=3)
    traj = [(50, 50, 1), (51, 50, 2), (52, 50, 3)]  # three inside observations before confirmation
    t = track(1, 53, 50, frame=3, traj=traj + [(53, 50, 4)])
    recs = eng.update(4, 4 / 15, [t], [])
    assert [r.kind for r in recs] == ["open"]
    assert eng.all_events()[0].first_inside_frame == 1


def test_two_tracks_make_two_events():
    eng = mk()
    for i in range(1, 8):
        eng.update(i, i / 15, [track(1, 20, 20, frame=i), track(2, 80, 80, frame=i)], [])
    assert len(eng.all_events()) == 2


@pytest.mark.parametrize("n_open", [1, 2, 5])
def test_n_open_controls_the_open_frame(n_open):
    eng = mk(n_open=n_open)
    feed(eng, [IN] * 10)
    assert eng.all_events()[0].open_frame == n_open

"""Corridor event logic: one event per occupancy, not one per frame.

Semantics (see docs/PLAN.md section 1):

  open     the track's reference point (bottom-centre of the box) has been inside
           the corridor for n_open consecutive observed frames. The event keeps
           the first-inside frame so latency can be measured honestly.
  active   while the track stays inside, or has been outside for fewer than
           m_close consecutive frames (hysteresis against boundary jitter).
  close    m_close consecutive outside frames. Type is "crossing" if the exit
           edge differs from the entry edge, otherwise "entry".
  reopen   the same track re-enters within `cooldown` frames: SAME event again.
  lost     if the track expires while the event is active the event is either
           closed as "left_view" (last seen near the frame border) or marked
           UNRESOLVED. It is never silently closed as "left the corridor".

Frames where a track is lost (not observed) do not advance any counter: absence
of evidence is not evidence of leaving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sightline.config import EventConfig
from sightline.geometry import Polygon, box_ref_point
from sightline.tracking.state import Track
from sightline.types import Box, TrackState


class EventStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class Event:
    event_id: int
    track_id: int
    first_inside_frame: int
    first_inside_ts: float
    open_frame: int
    open_ts: float
    entry_edge: int
    last_inside_frame: int
    status: EventStatus = EventStatus.ACTIVE
    close_frame: int | None = None
    close_ts: float | None = None
    close_reason: str | None = None
    exit_edge: int | None = None
    reopen_count: int = 0

    @property
    def type(self) -> str | None:
        if self.status != EventStatus.CLOSED or self.close_reason != "left_corridor":
            return None
        return "crossing" if self.exit_edge != self.entry_edge else "entry"

    @property
    def latency_frames_from_first_inside(self) -> int:
        return self.open_frame - self.first_inside_frame

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "status": self.status.value,
            "type": self.type,
            "first_inside_frame": self.first_inside_frame,
            "open_frame": self.open_frame,
            "open_ts": round(self.open_ts, 4),
            "last_inside_frame": self.last_inside_frame,
            "close_frame": self.close_frame,
            "close_reason": self.close_reason,
            "entry_edge": self.entry_edge,
            "exit_edge": self.exit_edge,
            "reopen_count": self.reopen_count,
        }


@dataclass
class EventRecord:
    """A state transition, for the log. Per-frame state is never an event."""

    kind: str  # open | reopen | close | unresolved
    frame: int
    ts: float
    event: dict

    def to_dict(self) -> dict:
        return {"kind": self.kind, "frame": self.frame, "ts": round(self.ts, 4), "event": self.event}


@dataclass
class _TrackPhase:
    phase: str = "outside"  # outside | candidate | active | closing | cooldown
    inside_count: int = 0
    outside_count: int = 0
    first_inside_frame: int = 0
    first_inside_ts: float = 0.0
    first_inside_point: tuple[float, float] = (0.0, 0.0)
    last_inside_point: tuple[float, float] = (0.0, 0.0)
    event: Event | None = None
    closed_at_frame: int = -10**9


class EventEngine:
    def __init__(self, cfg: EventConfig, polygon: Polygon, frame_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.polygon = polygon
        self.frame_w, self.frame_h = frame_size
        self.events: dict[int, Event] = {}
        self._phases: dict[int, _TrackPhase] = {}
        self._next_event_id = 1

    # -- queries ---------------------------------------------------------------

    @property
    def occupied(self) -> bool:
        """True if anyone is, or might be, inside. Candidates count: be conservative."""
        return any(p.phase in ("candidate", "active", "closing") for p in self._phases.values())

    def all_events(self) -> list[Event]:
        return sorted(self.events.values(), key=lambda e: e.event_id)

    # -- update ----------------------------------------------------------------

    def update(self, frame_index: int, ts: float, tracks: list[Track], expired: list[Track]) -> list[EventRecord]:
        out: list[EventRecord] = []
        for t in tracks:
            if t.state not in (TrackState.CONFIRMED, TrackState.LOST):
                continue  # tentative tracks have no identity yet
            ph = self._phases.get(t.id)
            if ph is None:
                ph = self._phases[t.id] = _TrackPhase()
                self._seed(ph, t)
            if t.observed:
                inside = self.polygon.contains(box_ref_point(t.box))
                out.extend(self._observe(t.id, ph, inside, box_ref_point(t.box), frame_index, ts))
            # unobserved (lost): no counter moves
        for t in expired:
            out.extend(self._expire(t, frame_index, ts))
        return out

    def finalize(self, frame_index: int, ts: float) -> list[EventRecord]:
        """End of stream: anything still active cannot be resolved."""
        out = []
        for tid, ph in self._phases.items():
            if ph.event is not None and ph.phase in ("active", "closing"):
                ev = ph.event
                ev.status, ev.close_frame, ev.close_ts, ev.close_reason = EventStatus.UNRESOLVED, frame_index, ts, "stream_end"
                out.append(EventRecord("unresolved", frame_index, ts, ev.to_dict()))
                ph.phase = "outside"
        return out

    # -- internals -------------------------------------------------------------

    def _seed(self, ph: _TrackPhase, t: Track) -> None:
        """A track becomes visible to the engine when it is confirmed. Its earlier
        observed points are real evidence, so count the run of inside points that
        ends now. This keeps confirmation delay out of the event latency."""
        pts = list(t.trajectory)[:-1]  # the current observation is handled by _observe
        run: list[tuple[float, float, int]] = []
        for x, y, f in reversed(pts):
            if self.polygon.contains((x, y)):
                run.append((x, y, f))
            else:
                break
        if run:
            x, y, f = run[-1]  # earliest of the run
            ph.phase = "candidate"
            ph.inside_count = len(run)
            ph.first_inside_frame = f
            ph.first_inside_ts = t.last_seen_ts  # best available; refined by caller if known
            ph.first_inside_point = (x, y)
            ph.last_inside_point = (run[0][0], run[0][1])

    def _observe(self, tid: int, ph: _TrackPhase, inside: bool, point, frame: int, ts: float) -> list[EventRecord]:
        cfg = self.cfg
        out: list[EventRecord] = []

        if ph.phase == "cooldown" and frame - ph.closed_at_frame > cfg.cooldown:
            ph.phase, ph.event = "outside", None

        if ph.phase in ("outside",):
            if inside:
                ph.phase, ph.inside_count = "candidate", 1
                ph.first_inside_frame, ph.first_inside_ts = frame, ts
                ph.first_inside_point = ph.last_inside_point = point
                out.extend(self._maybe_open(tid, ph, frame, ts))
            return out

        if ph.phase == "candidate":
            if inside:
                ph.inside_count += 1
                ph.last_inside_point = point
                out.extend(self._maybe_open(tid, ph, frame, ts))
            else:
                ph.phase, ph.inside_count = "outside", 0  # debounce rejected a flicker
            return out

        if ph.phase == "active":
            assert ph.event is not None
            if inside:
                ph.event.last_inside_frame = frame
                ph.last_inside_point = point
            else:
                ph.phase, ph.outside_count = "closing", 1
                out.extend(self._maybe_close(tid, ph, frame, ts))
            return out

        if ph.phase == "closing":
            assert ph.event is not None
            if inside:
                ph.phase, ph.outside_count = "active", 0
                ph.event.last_inside_frame = frame
                ph.last_inside_point = point
            else:
                ph.outside_count += 1
                out.extend(self._maybe_close(tid, ph, frame, ts))
            return out

        if ph.phase == "cooldown":
            if inside:  # same crossing, jittering on the boundary: SAME event
                assert ph.event is not None
                ev = ph.event
                ev.status, ev.close_frame, ev.close_ts, ev.close_reason, ev.exit_edge = EventStatus.ACTIVE, None, None, None, None
                ev.reopen_count += 1
                ev.last_inside_frame = frame
                ph.phase, ph.outside_count = "active", 0
                ph.last_inside_point = point
                out.append(EventRecord("reopen", frame, ts, ev.to_dict()))
            return out
        return out

    def _maybe_open(self, tid: int, ph: _TrackPhase, frame: int, ts: float) -> list[EventRecord]:
        if ph.inside_count < self.cfg.n_open:
            return []
        ev = Event(
            event_id=self._next_event_id, track_id=tid,
            first_inside_frame=ph.first_inside_frame, first_inside_ts=ph.first_inside_ts,
            open_frame=frame, open_ts=ts,
            entry_edge=self.polygon.nearest_edge(ph.first_inside_point),
            last_inside_frame=frame,
        )
        self._next_event_id += 1
        self.events[ev.event_id] = ev
        ph.event, ph.phase = ev, "active"
        return [EventRecord("open", frame, ts, ev.to_dict())]

    def _maybe_close(self, tid: int, ph: _TrackPhase, frame: int, ts: float) -> list[EventRecord]:
        if ph.outside_count < self.cfg.m_close:
            return []
        assert ph.event is not None
        ev = ph.event
        ev.status, ev.close_frame, ev.close_ts, ev.close_reason = EventStatus.CLOSED, frame, ts, "left_corridor"
        ev.exit_edge = self.polygon.nearest_edge(ph.last_inside_point)
        ph.phase, ph.closed_at_frame = "cooldown", frame
        return [EventRecord("close", frame, ts, ev.to_dict())]

    def _expire(self, t: Track, frame: int, ts: float) -> list[EventRecord]:
        ph = self._phases.pop(t.id, None)
        if ph is None or ph.event is None:
            return []
        ev = ph.event
        if ph.phase == "closing":
            ev.status, ev.close_frame, ev.close_ts, ev.close_reason = EventStatus.CLOSED, frame, ts, "left_corridor"
            ev.exit_edge = self.polygon.nearest_edge(ph.last_inside_point)
            return [EventRecord("close", frame, ts, ev.to_dict())]
        if ph.phase == "active":
            if self._near_border(t.last_det_box):
                ev.status, ev.close_frame, ev.close_ts, ev.close_reason = EventStatus.CLOSED, frame, ts, "left_view"
                ev.exit_edge = self.polygon.nearest_edge(ph.last_inside_point)
                return [EventRecord("close", frame, ts, ev.to_dict())]
            ev.status, ev.close_frame, ev.close_ts, ev.close_reason = EventStatus.UNRESOLVED, frame, ts, "track_lost"
            return [EventRecord("unresolved", frame, ts, ev.to_dict())]
        return []  # outside / candidate / cooldown: nothing is open

    def _near_border(self, box: Box) -> bool:
        mx, my = self.cfg.edge_margin_frac * self.frame_w, self.cfg.edge_margin_frac * self.frame_h
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return cx <= mx or cx >= self.frame_w - mx or cy <= my or cy >= self.frame_h - my or x1 <= 0 or x2 >= self.frame_w

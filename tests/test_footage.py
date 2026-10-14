"""Own-footage helpers, driven by a scripted UI so no display is needed."""
from pathlib import Path

import pytest

from sightline.config import load_corridor
from sightline.footage import draw_corridor


class FakeUI:
    """Replays ("click", x, y) and ("key", code) steps; clicks are in frame pixels, like CvUI's."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.shown = 0
        self.closed = False
        self._cb = None

    def on_click(self, cb):
        self._cb = cb

    def show(self, img):
        self.shown += 1

    def wait_key(self, ms=30):
        if not self.steps:
            return 27  # Esc: never loop forever in a test
        kind, *args = self.steps.pop(0)
        if kind == "click":
            self._cb(*args)
            return -1
        return args[0]

    def close(self):
        self.closed = True


def key(c):
    return ("key", ord(c) if isinstance(c, str) else c)


def test_draw_corridor_writes_a_polygon_that_the_pipeline_accepts(clip, tmp_path):
    out = tmp_path / "corr.yaml"
    ui = FakeUI([("click", 100, 100), ("click", 300, 100), ("click", 300, 300), ("click", 900, 900),
                 key("z"),  # undo the stray click outside the frame
                 ("click", 100, 300), key(13)])
    res = draw_corridor(clip.video, out, "own-test", ui=ui)
    assert res["polygon"] == [[100, 100], [300, 100], [300, 300], [100, 300]]
    c = load_corridor(out)
    assert c.name == "own-test" and c.frame_size == tuple(res["frame_size"]) and c.polygon.area() == 200 * 200
    assert b"\r\n" not in out.read_bytes() and ui.closed


def test_draw_corridor_refuses_an_invalid_polygon_and_esc_writes_nothing(clip, tmp_path):
    out = tmp_path / "corr.yaml"
    ui = FakeUI([("click", 10, 10), ("click", 50, 50), key(13), key(27)])  # 2 vertices: not a polygon
    assert draw_corridor(clip.video, out, ui=ui) is None
    assert not out.exists()


def test_clicks_outside_the_frame_are_clamped(clip, tmp_path):
    out = tmp_path / "corr.yaml"
    ui = FakeUI([("click", -20, 10), ("click", 5000, 10), ("click", 5000, 5000), key("s")])
    res = draw_corridor(clip.video, out, ui=ui)
    w, h = res["frame_size"]
    assert res["polygon"] == [[0, 10], [w, 10], [w, h]]


CORR = str(Path(__file__).resolve().parent.parent / "configs" / "corridors" / "synthetic.yaml")


def test_label_events_records_first_inside_and_left_frames(clip, tmp_path):
    import json

    from sightline.evaluation.events import load_gt_events
    from sightline.footage import label_events

    out = tmp_path / "ann" / "events.json"
    steps = [key("d"), key("d"), key("i"),          # opens at frame 3
             key("d"), key("d"), key("d"), key("d"), key("l"),  # left on 7 -> last inside 6
             key("i"), key("u"),                     # an accidental open, undone
             key("D"), key("i"),                     # second person from frame 17
             key("D"), key("D"), key("i"),           # a third from frame 37, still open...
             key("d"), key("2"),                     # ...left on 38: closes open #2 (from 37), not the oldest
             key(13)]
    res = label_events(clip.video, CORR, out, ui=FakeUI(steps))
    n = clip.n_frames
    assert res["events"] == [{"first_inside_frame": 3, "last_inside_frame": 6},
                             {"first_inside_frame": 17, "last_inside_frame": n, "still_inside_at_end": True},
                             {"first_inside_frame": 37, "last_inside_frame": 37}]
    assert [(e.first_inside_frame, e.last_inside_frame) for e in load_gt_events(out)] == [(3, 6), (17, n), (37, 37)]
    prov = json.loads((tmp_path / "ann" / "events.provenance.json").read_text())
    assert prov["events"] == 3 and len(prov["clip_sha256"]) == 64 and b"\r\n" not in out.read_bytes()


def test_label_events_rejects_left_before_inside_and_esc_saves_nothing(clip, tmp_path):
    from sightline.footage import label_events

    out = tmp_path / "events.json"
    ui = FakeUI([key("l"), key("i"), key("l"), key(27)])  # nothing open; then left on the opening frame
    assert label_events(clip.video, CORR, out, ui=ui) is None
    assert not out.exists()


def test_label_events_refuses_a_corridor_drawn_on_another_frame_size(clip, tmp_path):
    from sightline.footage import label_events
    from sightline.ingest import IngestError

    other = tmp_path / "c.yaml"
    other.write_text("name: x\nframe_size: [1920, 1080]\npolygon: [[0, 0], [100, 0], [100, 100]]\n")
    with pytest.raises(IngestError, match="drawn on"):
        label_events(clip.video, other, tmp_path / "e.json", ui=FakeUI([]))


def test_check_clip_passes_a_clean_constant_rate_video(clip):
    from sightline.footage import check_clip

    r = check_clip(clip.video)
    assert r["ok"] and not r["problems"] and r["frames"] == clip.n_frames
    assert any("1920x1080" in w for w in r["warnings"])  # the synthetic clip is 640x360


def test_clip_findings_flag_what_breaks_ingest():
    from sightline.footage import clip_findings

    good = {"kind": "video", "frames_decoded": 300, "frames_invalid": 0, "invalid_frame_indices": [],
            "variable_frame_rate_suspected": False, "container_timestamps_non_monotonic": 0,
            "frame_count_matches_container": True, "fps_source": "container", "width": 1920, "height": 1080,
            "fps": 30.0, "codec": "avc1", "duration_s": 10.0, "frames_total": 300, "frame_count_reported": 300}
    assert clip_findings(good) == ([], [])
    phone = {**good, "variable_frame_rate_suspected": True, "fps": 29.42, "codec": "hvc1",
             "frame_count_matches_container": False, "frames_total": 297}
    problems, warnings = clip_findings(phone, rotation=90)
    assert any("variable frame rate" in p for p in problems) and any("300 frames" in p for p in problems)
    assert any("rotation" in w for w in warnings) and any("hvc1" in w for w in warnings)
    assert any("unusual fps" in w for w in warnings)


def test_missing_clip_is_reported_not_crashed(tmp_path):
    from sightline.ingest import IngestError

    with pytest.raises(IngestError):
        draw_corridor(tmp_path / "nope.mp4", tmp_path / "c.yaml", ui=FakeUI([]))
    assert not Path(tmp_path / "c.yaml").exists()

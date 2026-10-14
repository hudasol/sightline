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


def test_missing_clip_is_reported_not_crashed(tmp_path):
    from sightline.ingest import IngestError

    with pytest.raises(IngestError):
        draw_corridor(tmp_path / "nope.mp4", tmp_path / "c.yaml", ui=FakeUI([]))
    assert not Path(tmp_path / "c.yaml").exists()

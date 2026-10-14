"""Helpers for recording and labelling own corridor footage.

The interactive tools talk to a small UI object. CvUI is the real OpenCV window (it maps
clicks back to ORIGINAL frame pixels, whatever the window scale); tests drive the same
logic with a scripted fake, so nothing here needs a display to be tested.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from sightline.geometry import GeometryError, Polygon
from sightline.ingest import IngestError, probe, read_frame

KEY_ENTER, KEY_ESC, KEY_BACKSPACE = (13, 10), 27, 8

GUI_HELP = ("this OpenCV build has no GUI (opencv-python-headless, the project default). For the interactive "
            "tools install the GUI build of the same version in this venv:\n"
            "  pip uninstall -y opencv-python-headless\n"
            "  pip install opencv-python==<the same version>")


class GuiUnavailable(RuntimeError):
    pass


class CvUI:
    """An OpenCV window that fits the frame on screen and reports clicks in frame pixels."""

    def __init__(self, title: str, max_w: int = 1280, max_h: int = 720) -> None:
        self.title, self.max_w, self.max_h = title, max_w, max_h
        self.scale = 1.0
        self._on_click: Callable[[int, int], None] | None = None
        try:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(title, self._mouse)
        except cv2.error as exc:
            raise GuiUnavailable(GUI_HELP) from exc

    def _mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._on_click is not None:
            self._on_click(int(round(x / self.scale)), int(round(y / self.scale)))

    def on_click(self, cb: Callable[[int, int], None]) -> None:
        self._on_click = cb

    def show(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        self.scale = min(self.max_w / w, self.max_h / h, 1.0)
        shown = img if self.scale == 1.0 else cv2.resize(img, (round(w * self.scale), round(h * self.scale)),
                                                         interpolation=cv2.INTER_AREA)
        cv2.imshow(self.title, shown)

    def wait_key(self, ms: int = 30) -> int:
        return cv2.waitKey(ms)

    def close(self) -> None:
        cv2.destroyWindow(self.title)


def _text(img: np.ndarray, lines: list[str], color=(255, 255, 255)) -> None:
    scale = max(0.5, img.shape[0] / 1080 * 0.9)
    y = int(28 * scale)
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale) + 1)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(2 * scale)))
        y += int(32 * scale)


# -- (a) draw a corridor polygon ------------------------------------------------------------------

def corridor_yaml(name: str, frame_size: tuple[int, int], vertices: list[tuple[int, int]], source: str) -> str:
    """Validated corridor YAML text; raises GeometryError for a polygon the pipeline would reject."""
    Polygon.create(vertices, frame_size=frame_size)  # same validation as load_corridor
    w, h = frame_size
    pts = ", ".join(f"[{x}, {y}]" for x, y in vertices)
    return (f"# Drawn with `sightline draw-corridor` on frame 1 of {source} ({time.strftime('%Y-%m-%d')}).\n"
            "# Image-space floor polygon in original pixels; it is hashed into the held-out lock.\n"
            f"name: {name}\nframe_size: [{w}, {h}]\npolygon: [{pts}]\n")


def draw_corridor(source: str | Path, out: str | Path, name: str | None = None, ui=None) -> dict | None:
    """Click vertices on frame 1; Backspace/z undo, Enter/s save, Esc/q cancel. Returns None if cancelled."""
    try:
        info = probe(source)
    except IngestError as exc:
        raise IngestError(f"draw-corridor: {exc}") from exc
    frame = read_frame(source, 1, info)
    if frame is None:
        raise IngestError(f"draw-corridor: frame 1 of {source} is unreadable")
    h, w = frame.shape[:2]
    name = name or Path(out).stem
    ui = ui or CvUI(f"draw-corridor: {Path(source).name}")
    verts: list[tuple[int, int]] = []
    message = "click floor-polygon vertices"
    ui.on_click(lambda x, y: verts.append((min(max(x, 0), w), min(max(y, 0), h))))
    try:
        while True:
            img = frame.copy()
            if verts:
                pts = np.array(verts, dtype=np.int32)
                cv2.polylines(img, [pts], len(verts) >= 3, (0, 255, 0), max(2, w // 640))
                for x, y in verts:
                    cv2.circle(img, (x, y), max(4, w // 320), (0, 0, 255), -1)
            _text(img, [f"{len(verts)} vertices | Backspace/z undo | Enter/s save | Esc/q cancel", message])
            ui.show(img)
            key = ui.wait_key(30)
            if key in (KEY_BACKSPACE, ord("z")) and verts:
                verts.pop()
            elif key in (*KEY_ENTER, ord("s")):
                try:
                    text = corridor_yaml(name, (w, h), verts, Path(source).as_posix())
                except GeometryError as exc:
                    message = f"not saved: {exc}"
                    continue
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(text, newline="\n")  # LF: the lock hashes these bytes
                return {"corridor": Path(out).as_posix(), "frame_size": [w, h], "polygon": [list(v) for v in verts]}
            elif key in (KEY_ESC, ord("q")):
                return None
    finally:
        ui.close()

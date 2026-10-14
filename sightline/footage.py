"""Helpers for recording and labelling own corridor footage.

The interactive tools talk to a small UI object. CvUI is the real OpenCV window (it maps
clicks back to ORIGINAL frame pixels, whatever the window scale); tests drive the same
logic with a scripted fake, so nothing here needs a display to be tested.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from sightline.config import load_corridor
from sightline.geometry import GeometryError, Polygon
from sightline.ingest import IngestError, SourceInfo, _sequence_files, probe, read_frame, sha256_file

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


# -- (b) label corridor events --------------------------------------------------------------------

LABEL_HELP = "d/a frame +-1 | D/A +-10 | i first inside | l left (1-9: which open event) | u undo | Enter save | Esc quit"


class FrameReader:
    """Random access that stays cheap when stepping forward: one capture kept open, seek only on jumps."""

    def __init__(self, source: str | Path, info: SourceInfo, cache: int = 64) -> None:
        self.source, self.info, self.limit = Path(source), info, cache
        self._cache: dict[int, np.ndarray | None] = {}
        self._files = _sequence_files(self.source) if info.kind == "image_sequence" else None
        self._cap = cv2.VideoCapture(str(self.source)) if self._files is None else None
        self._next = 1  # index the capture will return on its next read

    def get(self, index: int) -> np.ndarray | None:
        if index in self._cache:
            return self._cache[index]
        if self._files is not None:
            img = cv2.imread(str(self._files[index - 1])) if 1 <= index <= len(self._files) else None
        else:
            if index != self._next:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            ok, img = self._cap.read()
            img = img if ok else None
            self._next = index + 1
        if len(self._cache) >= self.limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = img
        return img

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def label_events(source: str | Path, corridor_path: str | Path, out: str | Path, ui=None) -> dict | None:
    """Step through the clip with the corridor drawn and mark each event's first-inside and left frames.

    "i" marks the first frame on which the person's feet are inside the polygon. "l" marks the
    first frame on which they are no longer inside, so last_inside_frame = that frame - 1.
    Several people can be open at once; "l" closes the oldest, 1-9 close that numbered one.
    Writes the events.json list read by evaluation (first_inside_frame / last_inside_frame) and
    a provenance file next to it. Returns None if quit without saving.
    """
    info = probe(source)
    corridor = load_corridor(corridor_path)
    if corridor.frame_size and tuple(corridor.frame_size) != (info.width, info.height):
        raise IngestError(f"corridor {corridor_path} was drawn on {corridor.frame_size}, the clip is "
                          f"{info.width}x{info.height}")
    n = info.frame_count_reported
    reader = FrameReader(source, info)
    ui = ui or CvUI(f"label-events: {Path(source).name}")
    ui.on_click(lambda _x, _y: None)
    poly = np.array(corridor.polygon.vertices, dtype=np.int32)
    cur, opened, done, history = 1, [], [], []
    message = "mark each person's first inside frame (i) and the frame they have left (l)"

    def close(k: int) -> str:
        first = opened[k]
        if cur - 1 < first:
            return f"not closed: 'left' on frame {cur} is not after first-inside frame {first}"
        ev = {"first_inside_frame": first, "last_inside_frame": cur - 1}
        done.append(ev)
        history.append(("close", ev, k))
        opened.pop(k)
        return f"event {first}-{cur - 1} recorded"

    try:
        while True:
            img = reader.get(cur)
            img = np.zeros((info.height, info.width, 3), np.uint8) if img is None else img.copy()
            cv2.polylines(img, [poly], True, (0, 255, 0), max(2, info.width // 640))
            open_txt = ", ".join(f"#{i + 1} from {f}" for i, f in enumerate(opened)) or "none"
            _text(img, [f"frame {cur}/{n}   events {len(done)}   open: {open_txt}", LABEL_HELP, message])
            ui.show(img)
            key = ui.wait_key(30)
            step = {ord("d"): 1, ord("a"): -1, ord("D"): 10, ord("A"): -10}.get(key)
            if step:
                cur = min(max(cur + step, 1), n)
            elif key == ord("i"):
                if cur in opened:
                    message = f"frame {cur} is already an open first-inside frame"
                else:
                    opened.append(cur)
                    history.append(("open", cur, len(opened) - 1))
                    message = f"opened at frame {cur}"
            elif key == ord("l") or (ord("1") <= key <= ord("9")):
                k = 0 if key == ord("l") else key - ord("1")
                message = close(k) if k < len(opened) else "no such open event"
            elif key == ord("u") and history:
                kind, item, k = history.pop()
                if kind == "open":
                    opened.pop(k)
                else:
                    done.remove(item)
                    opened.insert(k, item["first_inside_frame"])
                message = "undone"
            elif key in KEY_ENTER:
                rows = sorted(done + [{"first_inside_frame": f, "last_inside_frame": n, "still_inside_at_end": True}
                                      for f in opened], key=lambda e: (e["first_inside_frame"], e["last_inside_frame"]))
                out_p = Path(out)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(rows, indent=1) + "\n", newline="\n")  # LF: the lock hashes it
                prov = {"tool": "sightline label-events", "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "clip": Path(source).as_posix(),
                        "clip_sha256": sha256_file(Path(source)) if Path(source).is_file() else None,
                        "corridor": Path(corridor_path).as_posix(), "corridor_sha256": sha256_file(Path(corridor_path)),
                        "rule": "first_inside_frame = first frame with the feet inside; last_inside_frame = the "
                                "'left' frame minus 1; still_inside_at_end = closed at the last frame",
                        "events": len(rows)}
                (out_p.parent / f"{out_p.stem}.provenance.json").write_text(json.dumps(prov, indent=1) + "\n", newline="\n")
                return {"events": rows, "out": out_p.as_posix()}
            elif key == KEY_ESC:
                return None
    finally:
        reader.close()
        ui.close()


# -- (c) check a clip before using it -------------------------------------------------------------

STANDARD_FPS = (15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)
FRAGILE_CODECS = ("hevc", "hev1", "hvc1", "h265", "av01", "vp09")


def clip_findings(rec: dict, rotation: float | None = None) -> tuple[list[str], list[str]]:
    """(problems that would break ingest or evaluation, warnings worth fixing) for an ingest record."""
    problems, warnings = [], []
    if rec["frames_decoded"] == 0:
        problems.append("no frame could be decoded")
    if rec["frames_invalid"]:
        problems.append(f"{rec['frames_invalid']} undecodable frame(s), e.g. {rec['invalid_frame_indices'][:5]}")
    if rec.get("variable_frame_rate_suspected"):
        problems.append("variable frame rate: container timestamps do not follow 1/fps (phones do this); "
                        "convert to constant 30 fps with the ffmpeg command in the README")
    if rec.get("container_timestamps_non_monotonic"):
        problems.append(f"{rec['container_timestamps_non_monotonic']} non-monotonic container timestamp(s)")
    if rec.get("frame_count_matches_container") is False:
        problems.append(f"container says {rec['frame_count_reported']} frames, {rec['frames_total']} were read")
    if rec.get("fps_source") != "container" and rec.get("kind") == "video":
        problems.append(f"fps did not come from the container ({rec.get('fps_source')})")
    if rotation:
        warnings.append(f"rotation metadata {rotation:g} deg: OpenCV may show the frame rotated; re-encode with "
                        "ffmpeg (it applies the rotation) before drawing a corridor")
    if (rec["width"], rec["height"]) != (1920, 1080):
        warnings.append(f"resolution {rec['width']}x{rec['height']}, not 1920x1080 (corridors and timings are "
                        "per resolution; keep one camera setting per session)")
    if not any(abs(rec["fps"] - f) < 0.05 for f in STANDARD_FPS):
        warnings.append(f"unusual fps {rec['fps']:.3f}")
    elif abs(rec["fps"] - 30.0) > 0.05:
        warnings.append(f"fps {rec['fps']:.2f}, not 30 (fine for evaluation; settings are in seconds, PLAN.md A3)")
    if str(rec.get("codec", "")).lower() in FRAGILE_CODECS:
        warnings.append(f"codec {rec['codec']}: decoding depends on the OpenCV/FFmpeg build; H.264 is the safe choice")
    if rec["duration_s"] < 10:
        warnings.append(f"only {rec['duration_s']:.1f} s long")
    return problems, warnings


def check_clip(path: str | Path) -> dict:
    """Read the whole clip once (sightline.ingest) and report what would break or mislead evaluation."""
    from sightline.ingest import ingest

    rec = ingest(path)
    rotation = None
    if rec["kind"] == "video" and hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        cap = cv2.VideoCapture(str(path))
        try:
            rotation = float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
        finally:
            cap.release()
    problems, warnings = clip_findings(rec, rotation)
    deltas = [b - a for a, b in zip(rec["container_timestamps_s"], rec["container_timestamps_s"][1:])]
    return {
        "path": Path(path).as_posix(), "kind": rec["kind"], "codec": rec.get("codec"),
        "resolution": [rec["width"], rec["height"]], "fps": rec["fps"], "fps_source": rec["fps_source"],
        "frames": rec["frames_total"], "duration_s": rec["duration_s"], "rotation_deg": rotation,
        "frame_interval_ms": ({"min": round(min(deltas) * 1000, 2), "max": round(max(deltas) * 1000, 2)} if deltas else None),
        "variable_frame_rate_suspected": rec.get("variable_frame_rate_suspected"),
        "sha256": rec["sha256"], "problems": problems, "warnings": warnings, "ok": not problems,
    }

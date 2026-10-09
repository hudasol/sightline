"""Reproducible video ingest: one command records fps, frame count, timestamps
and source metadata, and says plainly when the file is not what it claims to be.

Two source kinds:
  video           a file OpenCV can decode (.mp4, .avi, .mov, .mkv)
  image_sequence  a MOTChallenge-style folder: seqinfo.ini + img1/000001.jpg ...

Frame indices are 1-based and refer to the original source.
Timestamps are NOMINAL ((index - 1) / fps) by default, which keeps evaluation
deterministic. Container timestamps are recorded alongside and can be used
instead with use_container_ts=True, which makes variable-frame-rate gaps visible
to the health monitor.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from sightline.types import FrameData

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class IngestError(RuntimeError):
    pass


@dataclass
class SourceInfo:
    path: str
    kind: str
    width: int
    height: int
    fps: float
    frame_count_reported: int
    codec: str = ""
    size_bytes: int = 0
    fps_source: str = "container"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path, "kind": self.kind, "width": self.width, "height": self.height,
            "fps": self.fps, "fps_source": self.fps_source,
            "frame_count_reported": self.frame_count_reported, "codec": self.codec,
            "size_bytes": self.size_bytes, **self.extra,
        }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _is_sequence_dir(p: Path) -> bool:
    return p.is_dir() and ((p / "seqinfo.ini").is_file() or (p / "img1").is_dir())


def _sequence_files(p: Path) -> list[Path]:
    img_dir = p / "img1" if (p / "img1").is_dir() else p
    return sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def probe(path: str | Path, fps_override: float | None = None) -> SourceInfo:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"source not found: {p}")
    if _is_sequence_dir(p):
        return _probe_sequence(p, fps_override)
    if p.is_dir():
        raise IngestError(f"{p} is a directory but not an image sequence (no seqinfo.ini or img1/)")
    return _probe_video(p, fps_override)


def _probe_video(p: Path, fps_override: float | None) -> SourceInfo:
    if p.stat().st_size == 0:
        raise IngestError(f"empty file: {p}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise IngestError(f"cannot open video (corrupt or unsupported): {p}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_source = "container"
        if not math.isfinite(fps) or fps <= 0:
            if fps_override is None:
                raise IngestError(f"container reports no usable fps for {p}; pass --fps")
            fps, fps_source = float(fps_override), "override"
        elif fps_override is not None and abs(fps_override - fps) > 1e-6:
            fps, fps_source = float(fps_override), "override"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        if w <= 0 or h <= 0:
            raise IngestError(f"video reports no frame size: {p}")
        return SourceInfo(str(p), "video", w, h, fps, n, codec, p.stat().st_size, fps_source)
    finally:
        cap.release()


def _probe_sequence(p: Path, fps_override: float | None) -> SourceInfo:
    files = _sequence_files(p)
    if not files:
        raise IngestError(f"no images found in {p}")
    fps, fps_source, w, h = None, "seqinfo.ini", 0, 0
    ini = p / "seqinfo.ini"
    if ini.is_file():
        cp = configparser.ConfigParser()
        cp.read(ini)
        if "Sequence" in cp:
            seq = cp["Sequence"]
            fps = float(seq.get("frameRate", 0) or 0) or None
            w, h = int(seq.get("imWidth", 0) or 0), int(seq.get("imHeight", 0) or 0)
    if fps_override is not None:
        fps, fps_source = float(fps_override), "override"
    if not fps or fps <= 0:
        raise IngestError(f"no frame rate for {p}: add seqinfo.ini or pass --fps")
    first = cv2.imread(str(files[0]))
    if first is None:
        raise IngestError(f"first image is unreadable: {files[0]}")
    if not (w and h):
        h, w = first.shape[:2]
    listing = hashlib.sha256("\n".join(f"{f.name}:{f.stat().st_size}" for f in files).encode()).hexdigest()
    return SourceInfo(str(p), "image_sequence", w, h, float(fps), len(files), "images",
                      sum(f.stat().st_size for f in files), fps_source, {"listing_sha256": listing})


def iter_frames(
    path: str | Path,
    fps_override: float | None = None,
    stride: int = 1,
    use_container_ts: bool = False,
    info: SourceInfo | None = None,
) -> Iterator[FrameData]:
    """Yield frames in order. Undecodable frames are yielded as valid=False, never skipped silently.

    stride > 1 is a sampling policy: it yields every stride-th frame. The caller is
    responsible for recording that policy and evaluating its effect.
    """
    info = info or probe(path, fps_override)
    p = Path(path)
    if info.kind == "image_sequence":
        files = _sequence_files(p)
        for i, f in enumerate(files, start=1):
            if (i - 1) % stride:
                continue
            img = cv2.imread(str(f))
            yield FrameData(i, (i - 1) / info.fps, img, valid=img is not None)
        return

    cap = cv2.VideoCapture(str(p))
    try:
        reported = info.frame_count_reported
        index, last_ts = 0, -1.0
        while True:
            ok, frame = cap.read()
            container_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            index += 1
            if not ok or frame is None:
                if reported > 0 and index <= reported:
                    # the container promised this frame: it is corrupt, not end of file.
                    # Report it; the loop ends once the promised count is used up.
                    if (index - 1) % stride == 0:
                        yield FrameData(index, (index - 1) / info.fps, None, valid=False)
                    continue
                return
            ts = (index - 1) / info.fps
            if use_container_ts and math.isfinite(container_ts) and container_ts > last_ts:
                ts = container_ts
            last_ts = ts
            if (index - 1) % stride == 0:
                yield FrameData(index, ts, frame, valid=True)
    finally:
        cap.release()


def ingest(path: str | Path, out_dir: str | Path | None = None, fps_override: float | None = None) -> dict:
    """Read the whole source once and record what is really in it."""
    info = probe(path, fps_override)
    p = Path(path)
    timestamps: list[float] = []
    container_ts: list[float] = []
    invalid: list[int] = []
    decoded = 0
    cap = cv2.VideoCapture(str(p)) if info.kind == "video" else None
    try:
        for fr in iter_frames(path, fps_override, info=info):
            timestamps.append(round(fr.timestamp, 6))
            if fr.valid:
                decoded += 1
            else:
                invalid.append(fr.index)
    finally:
        if cap is not None:
            cap.release()
    if info.kind == "video":
        cap = cv2.VideoCapture(str(p))
        try:
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                container_ts.append(round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 6))
        finally:
            cap.release()

    total = len(timestamps)
    nonmono = sum(1 for a, b in zip(container_ts, container_ts[1:]) if b <= a)
    interval = 1.0 / info.fps
    vfr = bool(container_ts) and any(abs((b - a) - interval) > 0.5 * interval for a, b in zip(container_ts, container_ts[1:]))
    record = {
        **info.to_dict(),
        "sha256": sha256_file(p) if info.kind == "video" else info.extra.get("listing_sha256"),
        "frames_total": total,
        "frames_decoded": decoded,
        "frames_invalid": len(invalid),
        "invalid_frame_indices": invalid[:200],
        "frame_count_matches_container": (info.frame_count_reported == total) if info.frame_count_reported else None,
        "duration_s": round(total / info.fps, 4),
        "timestamp_source": "nominal ((index-1)/fps)",
        "timestamps_s": timestamps,
        "container_timestamps_s": container_ts,
        "container_timestamps_non_monotonic": nonmono,
        "variable_frame_rate_suspected": vfr,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{p.name}.ingest.json").write_text(json.dumps(record, indent=1))
    return record


def discover_sources(path: str | Path) -> list[Path]:
    """A single source, or every video / image-sequence folder directly inside a folder."""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"source not found: {p}")
    if p.is_file() or _is_sequence_dir(p):
        return [p]
    found = sorted(c for c in p.iterdir() if (c.is_file() and c.suffix.lower() in VIDEO_EXTS) or _is_sequence_dir(c))
    if not found:
        raise IngestError(f"no videos or image sequences in {p}")
    return found


def read_frame(path: str | Path, index: int, info: SourceInfo | None = None, fps_override: float | None = None):
    """Random access to one frame (1-based), for error-analysis renders. None if unreadable."""
    info = info or probe(path, fps_override)
    p = Path(path)
    if info.kind == "image_sequence":
        files = _sequence_files(p)
        return cv2.imread(str(files[index - 1])) if 1 <= index <= len(files) else None
    cap = cv2.VideoCapture(str(p))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()

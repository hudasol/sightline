"""Frame-rate-matched copies of image sequences (PLAN.md amendment A6.3).

The held-out IndoorCrowd clips run at 5 fps and the val clips at 14-30 fps. A decimated
copy keeps every `step`-th frame of a val sequence (frames 1, 1+step, ...), renumbers the
kept frames from 1 and rescales the frame rate, so tuning can see what the tracker and the
event rules do at the test frame rate. It is the same scene in the same split: no frame
crosses the split.

Detections are not recomputed. The copy's frames are the original images, so the original
detection cache is renumbered for each detector tag that has one: the same detections, on
the same pixels, at the new frame numbers.
"""

from __future__ import annotations

import configparser
import math
import os
import shutil
from pathlib import Path

from sightline.detector import read_detection_cache, write_detection_cache
from sightline.ingest import IngestError, probe


def step_for(fps: float, target_fps: float) -> int:
    """Keep every step-th frame; round half up so 30 -> 5 fps is a step of 6 and 25 -> 5 fps a step of 5."""
    return max(1, int(math.floor(fps / target_fps + 0.5)))


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # same bytes, no extra disk
    except OSError:
        shutil.copy2(src, dst)


def decimate_sequence(src: Path, out: Path, target_fps: float, name: str | None = None) -> dict:
    info = probe(src)
    if info.kind != "image_sequence":
        raise IngestError(f"{src}: decimation is for image sequences (seqinfo.ini + img1/)")
    step = step_for(info.fps, target_fps)
    cp = configparser.ConfigParser()
    cp.read(src / "seqinfo.ini")
    ext = cp["Sequence"].get("imExt", ".jpg") if "Sequence" in cp else ".jpg"
    n_src = info.frame_count_reported
    kept = list(range(1, n_src + 1, step))  # original frame numbers, in order
    new_of = {old: new for new, old in enumerate(kept, start=1)}
    (out / "img1").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    for old, new in new_of.items():
        dst = out / "img1" / f"{new:06d}{ext}"
        if not dst.exists():
            _link_or_copy(src / "img1" / f"{old:06d}{ext}", dst)
    rows = []
    gt_src = src / "gt" / "gt.txt"
    if gt_src.is_file():
        for line in gt_src.read_text().splitlines():
            parts = line.split(",")
            if line.strip() and int(float(parts[0])) in new_of:
                rows.append(",".join([str(new_of[int(float(parts[0]))]), *parts[1:]]))
        (out / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", newline="\n")
    fps = info.fps / step
    name = name or f"{src.name}-{target_fps:g}fps"
    (out / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate={fps:g}\nseqLength={len(kept)}\n"
        f"imWidth={info.width}\nimHeight={info.height}\nimExt={ext}\n", newline="\n")
    return {"source": src.as_posix(), "out": out.as_posix(), "name": name, "source_fps": info.fps, "step": step,
            "fps": fps, "frames": len(kept), "gt_rows": len(rows), "kept_source_frames": [kept[0], kept[-1]]}


def decimate_cache(cache_file: Path, out_file: Path, step: int, n_src: int, meta_note: str) -> int:
    """Renumber one detection cache to the decimated frames. Returns the number of frames written."""
    meta, per_frame = read_detection_cache(cache_file)
    kept = range(1, n_src + 1, step)
    new = {i: per_frame[old] for i, old in enumerate(kept, start=1) if old in per_frame}
    write_detection_cache(out_file, {**meta, "decimated_from": cache_file.as_posix(), "step": step,
                                     "note": meta_note}, new)
    return len(new)

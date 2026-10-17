"""IndoorCrowd import: the `tracking` config only, written as MOTChallenge sequences.

IndoorCrowd (Nae et al. 2026, arXiv:2604.02032) is gated on Hugging Face under a
noncommercial research licence that also prohibits surveillance and re-identification
(docs/licences/INDOORCROWD_LICENSE.txt, docs/DATA_CARD.md). This module only converts
files the user has already downloaded; it never downloads anything itself.

Layout of the `tracking` parquet files (one per site): one row per frame with the JPEG
bytes, `recording_id`, `file_name` (".../frames/frame_00042.jpg") and `objects` with
COCO `bbox` (x, y, w, h in pixels) and `track_id`. Each recording becomes one clip:

    <out>/<site>/<recording>/seqinfo.ini
    <out>/<site>/<recording>/img1/000001.jpg ...   (renumbered from 1, original order)
    <out>/<site>/<recording>/gt/gt.txt             (frame,id,x,y,w,h,1,1,1.0)
    <out>/<site>/<recording>/source_frames.json    (new frame -> original file name)

IndoorCrowd gives no visibility, so every box is written with visibility 1.0. Under the
in-scope rule (PLAN.md A1) that keeps every tall-enough person, occluded or not: the
stricter reading for the detector. The split is by SITE (one fixed camera each), never
by frame or recording; the manifest's scene and camera fields carry the site.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

SITES = ("acs_ec", "acs_eg", "ie_central", "r_central")
FPS = 5.0  # the dataset card and project page: videos sampled at 5 fps
_FRAME_RE = re.compile(r"frame_(\d+)\.\w+$")


class IndoorCrowdError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_id(site: str, recording: str) -> str:
    return f"IC-{site}-{recording}"


def tracking_files(raw_dir: Path, sites: tuple[str, ...] = SITES) -> dict[str, Path]:
    """The tracking parquet for each site. Only the `tracking` config is accepted."""
    out = {}
    for s in sites:
        if s not in SITES:
            raise IndoorCrowdError(f"unknown IndoorCrowd site {s!r}; expected one of {SITES}")
        found = sorted((raw_dir / "tracking").glob(f"{s}-*.parquet"))
        if not found:
            raise IndoorCrowdError(f"no tracking parquet for site {s!r} under {raw_dir / 'tracking'} "
                                   "(download the `tracking` config first, see README)")
        if len(found) > 1:
            raise IndoorCrowdError(f"expected one tracking parquet for {s!r}, found {[f.name for f in found]}")
        out[s] = found[0]
    return out


def _rows(parquet: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise IndoorCrowdError('pyarrow is needed to read IndoorCrowd: pip install -e ".[indoorcrowd]"') from exc
    t = pq.read_table(parquet)
    need = {"image", "recording_id", "file_name", "objects", "width", "height"}
    missing = need - set(t.schema.names)
    if missing:
        raise IndoorCrowdError(f"{parquet.name}: missing columns {sorted(missing)} (is this the tracking config?)")
    return t.to_pylist()


def import_site(parquet: Path, site: str, out_dir: Path) -> list[dict]:
    """Write every recording of one site as a MOT sequence. Returns one summary per clip."""
    by_rec: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for row in _rows(parquet):
        m = _FRAME_RE.search(row["file_name"])
        if not m:
            raise IndoorCrowdError(f"{parquet.name}: cannot read a frame number from {row['file_name']!r}")
        by_rec[row["recording_id"]].append((int(m.group(1)), row))
    clips = []
    for rec in sorted(by_rec):
        frames = sorted(by_rec[rec], key=lambda x: x[0])
        nums = [n for n, _r in frames]
        if len(set(nums)) != len(nums):
            raise IndoorCrowdError(f"{site}/{rec}: duplicate frame numbers")
        gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b != a + 1]
        seq = out_dir / site / rec
        (seq / "img1").mkdir(parents=True, exist_ok=True)
        (seq / "gt").mkdir(parents=True, exist_ok=True)
        w, h = frames[0][1]["width"], frames[0][1]["height"]
        lines, mapping, ids = [], {}, set()
        for new, (_orig, row) in enumerate(frames, start=1):
            if (row["width"], row["height"]) != (w, h):
                raise IndoorCrowdError(f"{site}/{rec}: frame size changes within the recording")
            (seq / "img1" / f"{new:06d}.jpg").write_bytes(row["image"]["bytes"])
            mapping[new] = row["file_name"]
            o = row["objects"]
            for (x, y, bw, bh), tid in zip(o["bbox"], o["track_id"]):
                lines.append(f"{new},{int(tid)},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1,1,1.0")
                ids.add(int(tid))
        (seq / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", newline="\n")
        (seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={clip_id(site, rec)}\nimDir=img1\nframeRate={FPS:g}\nseqLength={len(frames)}\n"
            f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n", newline="\n")
        (seq / "source_frames.json").write_text(json.dumps(mapping, indent=0), newline="\n")
        clips.append({"id": clip_id(site, rec), "site": site, "recording": rec, "path": seq.as_posix(),
                      "frames": len(frames), "first_source_frame": nums[0], "last_source_frame": nums[-1],
                      "source_frame_gaps": gaps, "width": w, "height": h, "fps": FPS,
                      "boxes": len(lines), "track_ids": len(ids)})
    return clips


def import_tracking(raw_dir: Path, out_dir: Path, sites: tuple[str, ...] = SITES, revision: str | None = None) -> dict:
    files = tracking_files(raw_dir, sites)
    clips = []
    for site, pq_path in files.items():
        clips += import_site(pq_path, site, out_dir)
    return {"dataset": "sebnae/IndoorCrowd", "config": "tracking", "revision": revision, "fps": FPS,
            "parquet": {s: {"file": f"tracking/{p.name}", "sha256": _sha256(p), "bytes": p.stat().st_size}
                        for s, p in files.items()},
            "clips": clips}

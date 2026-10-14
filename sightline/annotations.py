"""Own-clip annotations: which frames were annotated, and import from CVAT.

Sparse annotation (docs/PLAN.md section 14, risk 1) is only honest if the scorer knows
which frames were looked at: an annotated frame with no boxes is a real negative, an
unannotated frame is unknown and must not be scored. A clip says so with an
`annotated_frames` file; without one every frame counts as annotated.

CVAT's "MOT 1.1" export (gt/gt.txt + gt/labels.txt) cannot tell an empty annotated frame
from an unannotated one, so the importer requires the annotated frames to be stated.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from sightline.ingest import sha256_file

PEDESTRIAN, DISTRACTOR = 1, 8


class AnnotationError(ValueError):
    pass


def parse_frame_spec(spec: str) -> list[int]:
    """"1-600" (inclusive), "1-600:10" (every 10th from 1), comma-separated parts, or a file
    with one frame number per line. Frames are 1-based."""
    p = Path(spec)
    if p.is_file():
        parts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        frames = [int(x) for x in parts]
    else:
        frames = []
        for part in spec.split(","):
            part = part.strip()
            rng, _, step = part.partition(":")
            lo, _, hi = rng.partition("-")
            try:
                a, b = int(lo), int(hi or lo)
                s = int(step or 1)
            except ValueError as exc:
                raise AnnotationError(f"bad frame spec {part!r}; use e.g. 1-600 or 1-600:10") from exc
            if a < 1 or b < a or s < 1:
                raise AnnotationError(f"bad frame spec {part!r}: frames are 1-based and ranges ascending")
            frames.extend(range(a, b + 1, s))
    if not frames or min(frames) < 1:
        raise AnnotationError(f"frame spec {spec!r} selects no valid (1-based) frame")
    return sorted(set(frames))


def load_annotated_frames(path: str | Path) -> set[int]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"annotated-frames file not found: {p}")
    data = json.loads(p.read_text())
    frames = data["frames"] if isinstance(data, dict) else data
    if not isinstance(frames, list) or not frames or not all(isinstance(f, int) and f >= 1 for f in frames):
        raise AnnotationError(f"{p}: needs a non-empty list of 1-based frame numbers")
    return set(frames)


def _read_export(src: Path) -> tuple[str, str]:
    """(gt.txt text, labels.txt text) from a CVAT MOT 1.1 export, zipped or unpacked."""
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            gt = next((n for n in names if n.replace("\\", "/").endswith("gt/gt.txt")), None)
            labels = next((n for n in names if n.replace("\\", "/").endswith("gt/labels.txt")), None)
            if gt is None or labels is None:
                raise AnnotationError(f"{src}: not a CVAT MOT 1.1 export (needs gt/gt.txt and gt/labels.txt)")
            return (io.TextIOWrapper(z.open(gt), encoding="utf-8").read(),
                    io.TextIOWrapper(z.open(labels), encoding="utf-8").read())
    if src.is_dir():
        gt, labels = src / "gt" / "gt.txt", src / "gt" / "labels.txt"
        if not (gt.is_file() and labels.is_file()):
            raise AnnotationError(f"{src}: not a CVAT MOT 1.1 export (needs gt/gt.txt and gt/labels.txt)")
        return gt.read_text(encoding="utf-8"), labels.read_text(encoding="utf-8")
    raise AnnotationError(f"{src}: no such export (zip or folder)")


def import_cvat_mot(src: str | Path, out_dir: str | Path, frames_spec: str,
                    person_labels: tuple[str, ...] = ("person", "pedestrian"),
                    ignore_labels: tuple[str, ...] = ()) -> dict:
    """Convert a CVAT MOT 1.1 export into annotations/<clip>/{gt.txt, annotated_frames.json}.

    Person labels become class 1; ignore labels (people who must not be scored, e.g. a
    reflection) become class 8, an ignore region. Every other label is dropped. Visibility
    is kept as exported; an export whose boxes all say 1.0 is flagged, because the in-scope
    rule (PLAN.md amendment A1) needs real visibility estimates.
    """
    src, out = Path(src), Path(out_dir)
    gt_text, labels_text = _read_export(src)
    labels = [ln.strip() for ln in labels_text.splitlines() if ln.strip()]
    by_index = {i + 1: name for i, name in enumerate(labels)}  # CVAT writes 1-based class ids
    wanted = {n.lower() for n in person_labels}
    ignored = {n.lower() for n in ignore_labels}
    annotated = parse_frame_spec(frames_spec)
    annotated_set = set(annotated)

    rows, dropped, vis_values = [], {}, []
    for ln, line in enumerate(gt_text.splitlines(), start=1):
        if not line.strip():
            continue
        f = line.split(",")
        if len(f) < 9:
            raise AnnotationError(f"gt.txt:{ln}: expected 9 fields (frame,id,x,y,w,h,not_ignored,class,visibility)")
        frame, tid = int(float(f[0])), int(float(f[1]))
        x, y, w, h = (float(v) for v in f[2:6])
        flag, cls, vis = int(float(f[6])), int(float(f[7])), float(f[8])
        name = by_index.get(cls)
        if name is None:
            raise AnnotationError(f"gt.txt:{ln}: class id {cls} is not in labels.txt {labels}")
        if name.lower() in wanted:
            out_cls = PEDESTRIAN
        elif name.lower() in ignored:
            out_cls = DISTRACTOR
        else:
            dropped[name] = dropped.get(name, 0) + 1
            continue
        if frame not in annotated_set:
            raise AnnotationError(f"gt.txt:{ln}: a box on frame {frame}, which --frames says was not annotated")
        if not 0.0 <= vis <= 1.0:
            raise AnnotationError(f"gt.txt:{ln}: visibility {vis} is outside [0, 1]")
        if w <= 0 or h <= 0:
            raise AnnotationError(f"gt.txt:{ln}: empty box")
        rows.append((frame, tid, x, y, w, h, flag, out_cls, vis))
        if out_cls == PEDESTRIAN:
            vis_values.append(vis)

    warnings = []
    if vis_values and all(v == 1.0 for v in vis_values):
        warnings.append("every person box has visibility 1.0: the export probably has no real visibility "
                        "estimates, and the in-scope rule (PLAN.md A1) needs them")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gt.txt").open("w", newline="\n") as fh:
        for frame, tid, x, y, w, h, flag, cls, vis in sorted(rows):
            fh.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{flag},{cls},{vis:.3f}\n")
    # LF everywhere: the lock hashes these bytes, and .gitattributes keeps annotations/ LF on checkout
    (out / "annotated_frames.json").write_text(json.dumps({"frames": annotated, "spec": frames_spec}) + "\n", newline="\n")
    report = {
        "source": src.as_posix(), "source_sha256": sha256_file(src) if src.is_file() else None,
        "format": "CVAT MOT 1.1", "labels": labels, "person_labels": sorted(wanted), "ignore_labels": sorted(ignored),
        "dropped_rows_by_label": dropped, "person_boxes": sum(1 for r in rows if r[7] == PEDESTRIAN),
        "ignore_boxes": sum(1 for r in rows if r[7] == DISTRACTOR), "annotated_frames": len(annotated),
        "frames_with_boxes": len({r[0] for r in rows}), "warnings": warnings,
    }
    (out / "import_provenance.json").write_text(json.dumps(report, indent=1) + "\n", newline="\n")
    return report

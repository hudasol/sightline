import json

import cv2
import numpy as np
import pytest

from sightline.evaluation.gt import load_mot_gt
from sightline.indoorcrowd import IndoorCrowdError, import_tracking, tracking_files
from sightline.ingest import probe

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _jpeg(v: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((36, 64, 3), v, np.uint8))
    assert ok
    return buf.tobytes()


def _write_site(raw, site, rows):
    (raw / "tracking").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), raw / "tracking" / f"{site}-00000-of-00001.parquet")


def _row(rec, n, boxes):
    return {"image": {"bytes": _jpeg(10 * n), "path": f"frame_{n:05d}.jpg"}, "type": "tracking", "scene": "acs_eg",
            "objects": {"bbox": [b for b, _t in boxes], "track_id": [t for _b, t in boxes]},
            "image_id": n, "width": 64, "height": 36, "recording_id": rec,
            "file_name": f"acs_eg/{rec}/frames/frame_{n:05d}.jpg"}


def test_recordings_become_renumbered_mot_sequences_in_source_order(tmp_path):
    raw = tmp_path / "raw"
    # rows deliberately out of order; recording r2 starts at source frame 11
    _write_site(raw, "acs_eg", [_row("r2", 12, [([1, 2, 10, 20], 4)]), _row("r1", 2, [([5, 5, 8, 9], 1)]),
                                _row("r2", 11, [([0, 0, 10, 20], 4), ([30, 1, 5, 6], 7)]), _row("r1", 1, [])])
    rep = import_tracking(raw, tmp_path / "out", ("acs_eg",), "abc")
    assert [c["id"] for c in rep["clips"]] == ["IC-acs_eg-r1", "IC-acs_eg-r2"]
    r2 = rep["clips"][1]
    assert (r2["frames"], r2["first_source_frame"], r2["boxes"], r2["track_ids"]) == (2, 11, 3, 2)
    seq = tmp_path / "out" / "acs_eg" / "r2"
    info = probe(seq)
    assert (info.fps, info.width, info.height, info.frame_count_reported) == (5.0, 64, 36, 2)
    gt, ignore = load_mot_gt(seq / "gt" / "gt.txt")
    assert gt[1] == [(4, (0.0, 0.0, 10.0, 20.0)), (7, (30.0, 1.0, 35.0, 7.0))] and gt[2] == [(4, (1.0, 2.0, 11.0, 22.0))]
    assert not ignore
    assert (seq / "gt" / "gt.txt").read_text().splitlines()[0].endswith(",1,1,-1")  # visibility unknown, not 1.0
    assert json.loads((seq / "source_frames.json").read_text())["1"].endswith("frame_00011.jpg")
    assert rep["parquet"]["acs_eg"]["sha256"] and rep["revision"] == "abc"


def test_only_tracking_files_of_known_sites_are_accepted(tmp_path):
    with pytest.raises(IndoorCrowdError, match="unknown IndoorCrowd site"):
        tracking_files(tmp_path, ("lobby",))
    with pytest.raises(IndoorCrowdError, match="no tracking parquet"):
        tracking_files(tmp_path, ("acs_ec",))


def test_duplicate_frame_numbers_are_refused(tmp_path):
    raw = tmp_path / "raw"
    _write_site(raw, "acs_eg", [_row("r1", 1, []), _row("r1", 1, [])])
    with pytest.raises(IndoorCrowdError, match="duplicate frame numbers"):
        import_tracking(raw, tmp_path / "out", ("acs_eg",))


def test_ingest_names_a_sequence_by_its_seqinfo_name_not_its_folder(tmp_path):
    from sightline.ingest import ingest

    raw = tmp_path / "raw"
    _write_site(raw, "acs_eg", [_row("clip_1", 1, [])])
    _write_site(raw, "acs_ec", [_row("clip_1", 1, [])])
    import_tracking(raw, tmp_path / "out", ("acs_eg", "acs_ec"))
    for site in ("acs_eg", "acs_ec"):
        ingest(tmp_path / "out" / site / "clip_1", out_dir=tmp_path / "ingest")
    assert sorted(p.name for p in (tmp_path / "ingest").iterdir()) == [
        "IC-acs_ec-clip_1.ingest.json", "IC-acs_eg-clip_1.ingest.json"]

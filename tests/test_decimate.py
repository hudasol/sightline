import cv2
import numpy as np

from sightline.decimate import decimate_cache, decimate_sequence, step_for
from sightline.detector import Detection, read_detection_cache, write_detection_cache
from sightline.evaluation.gt import load_mot_gt
from sightline.ingest import probe


def _sequence(root, n=13, fps=30):
    (root / "img1").mkdir(parents=True)
    (root / "gt").mkdir()
    for i in range(1, n + 1):
        cv2.imwrite(str(root / "img1" / f"{i:06d}.jpg"), np.full((20, 30, 3), i * 10, np.uint8))
    (root / "gt" / "gt.txt").write_text("".join(f"{i},1,{i},0,5,10,1,1,1.0\n" for i in range(1, n + 1)))
    (root / "seqinfo.ini").write_text(f"[Sequence]\nname=S\nimDir=img1\nframeRate={fps}\nseqLength={n}\n"
                                      "imWidth=30\nimHeight=20\nimExt=.jpg\n")


def test_step_rounds_half_up():
    assert (step_for(30, 5), step_for(25, 5), step_for(14, 5), step_for(4, 5)) == (6, 5, 3, 1)


def test_decimated_copy_keeps_every_step_th_frame_and_renumbers_gt_and_cache(tmp_path):
    src, out = tmp_path / "S", tmp_path / "S-5fps"
    _sequence(src)
    r = decimate_sequence(src, out, 5.0, "S-5fps")
    assert (r["step"], r["frames"], r["fps"], r["kept_source_frames"]) == (6, 3, 5.0, [1, 13])  # frames 1, 7, 13
    info = probe(out)
    assert (info.fps, info.frame_count_reported, info.extra["name"]) == (5.0, 3, "S-5fps")
    gt, _ig = load_mot_gt(out / "gt" / "gt.txt")
    assert {f: b[0][1][0] for f, b in gt.items()} == {1: 1.0, 2: 7.0, 3: 13.0}  # x encodes the source frame
    assert (out / "img1" / "000002.jpg").read_bytes() == (src / "img1" / "000007.jpg").read_bytes()
    cache = tmp_path / "c.jsonl"
    write_detection_cache(cache, {"detector": "d"}, {i: [Detection((i, 0, 5, 10), 0.9)] for i in range(1, 14)})
    assert decimate_cache(cache, tmp_path / "c5.jsonl", 6, 13, "test") == 3
    meta, per = read_detection_cache(tmp_path / "c5.jsonl")
    assert {f: d[0].box[0] for f, d in per.items()} == {1: 1.0, 2: 7.0, 3: 13.0} and meta["step"] == 6

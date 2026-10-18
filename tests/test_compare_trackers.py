import json
from pathlib import Path

import pytest

from sightline.cli import main

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


@pytest.mark.skipif(not (R / "per_clip" / "iou" / "MOT17-09-FRCNN.json").is_file(), reason="no held out results saved")
def test_compare_trackers_prints_the_saved_id_switches_and_pooled_idf1(capsys):
    assert main(["compare-trackers", "MOT17-09-FRCNN", "--results", str(R)]) == 0
    out = capsys.readouterr().out
    iou = len(json.loads((R / "per_clip/iou/MOT17-09-FRCNN.json").read_text())["id_switches"])
    bt = len(json.loads((R / "per_clip/bytetrack/MOT17-09-FRCNN.json").read_text())["id_switches"])
    final = json.loads((R / "final_metrics.json").read_text())
    assert f"IoU baseline {iou}, ByteTrack-style {bt}" in out
    assert f"ALL: IoU baseline {final['tracking']['iou']['ALL']['idf1']:.3f}" in out
    assert "Per clip IDF1 is not saved" in out


def test_compare_trackers_on_an_unknown_clip_is_unavailable(tmp_path):
    assert main(["compare-trackers", "nope", "--results", str(tmp_path)]) == 2

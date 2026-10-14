"""IDF1 and ID switches against a case small enough to compute by hand."""
import numpy as np
import pytest

from sightline.evaluation.tracking import build_accumulator, iou_distance, summarize


def test_iou_distance_follows_motmetrics_convention():
    g = [(0, 0, 10, 10)]
    h = [(0, 0, 10, 10), (5, 0, 15, 10), (20, 20, 30, 30)]  # IoU 1, 1/3, 0
    d = iou_distance(g, h, iou_thr=0.3)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(2 / 3)
    assert np.isnan(d[0, 2])  # below the threshold: not allowed to match
    assert np.isnan(iou_distance(g, h, iou_thr=0.5)[0, 1])


def test_one_identity_switch_by_hand():
    # one person for 4 frames; the tracker calls them #1 for two frames, then #2
    box = (100.0, 100.0, 150.0, 250.0)
    gt = {f: [(7, box)] for f in range(1, 5)}
    hyp = [(f, 1 if f <= 2 else 2, *box, 0.9) for f in range(1, 5)]
    acc = build_accumulator(gt, {}, hyp, range(1, 5))
    m = summarize({"c": acc})["OVERALL"]
    assert m["num_switches"] == 1
    # best one-to-one identity mapping keeps 2 of 4 frames: IDTP=2, IDFP=2, IDFN=2 -> IDF1 = 4 / (4 + 2 + 2)
    assert m["idf1"] == pytest.approx(0.5)

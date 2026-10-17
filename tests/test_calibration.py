from sightline.calibration import _classify
from sightline.types import Detection


def test_detections_are_hits_false_positives_or_left_out_on_ignore_regions():
    gt = [(0, 0, 10, 20)]
    ignore = [(100, 0, 110, 20)]
    dets = [Detection((0, 0, 10, 20), 0.9), Detection((1, 0, 11, 20), 0.8),  # a duplicate of the same person
            Detection((100, 0, 110, 20), 0.7), Detection((50, 50, 60, 70), 0.6)]
    rows, hits, n = _classify(gt, ignore, dets, 0.5)
    assert rows == [(0.9, 1), (0.8, 0), (0.6, 0)]  # the duplicate is a false positive, the ignore one is dropped
    assert (hits, n) == (1, 1)

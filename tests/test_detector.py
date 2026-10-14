import numpy as np
import pytest

from sightline.config import config_from_dict
from sightline.detector import fit_size


@pytest.mark.parametrize("frame, box, want", [
    ((1920, 1080), (640, 360), (640, 360)),  # 16:9 into 16:9
    ((640, 480), (640, 360), (480, 360)),  # 4:3 limited by height
    ((640, 480), (960, 540), (640, 480)),  # never upscaled
    ((640, 360), (640, 360), (640, 360)),  # already the right size
])
def test_fit_size(frame, box, want):
    assert fit_size(*frame, *box) == want


def test_input_size_defaults_to_declared_size_and_is_validated():
    assert config_from_dict({}).detector.input_size == [640, 360]
    cfg = config_from_dict({"pipeline": {"declared_width": 960, "declared_height": 540}})
    assert cfg.detector.input_size == [960, 540]
    assert config_from_dict({"detector": {"input_size": [800, 450]}}).detector.input_size == [800, 450]
    for bad in ([640], [640, 0], "640x360"):
        with pytest.raises(ValueError):
            config_from_dict({"detector": {"input_size": bad}})


def test_unknown_detector_is_unavailable():
    from sightline.detector import DetectorUnavailable, make_detector

    with pytest.raises(DetectorUnavailable, match="unknown detector"):
        make_detector(config_from_dict({"detector": {"name": "magic_net"}}).detector)


def test_yolox_decode_matches_the_official_grid_formula():
    from sightline.detector import yolox_decode

    # 32x32 input: stride-8 grid is 4x4 (16 cells), stride-16 2x2 (4), stride-32 1x1 (1) = 21 rows
    raw = np.zeros((21, 6), dtype=np.float32)
    raw[5] = [0.5, 0.5, 0.0, np.log(2.0), 1.0, 1.0]  # stride-8 cell (x=1, y=1)
    out = yolox_decode(raw, (32, 32))
    assert out[5, :4] == pytest.approx([(1 + 0.5) * 8, (1 + 0.5) * 8, 8.0, 16.0])
    assert out[20, :2] == pytest.approx([0.0, 0.0])  # the single stride-32 cell at (0, 0)


def test_nms_keeps_the_best_of_overlapping_boxes():
    from sightline.detector import _nms

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert _nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45) == [0, 2]


def test_fixed_input_models_refuse_another_input_size():
    pytest.importorskip("torchvision")
    from sightline.detector import DetectorUnavailable, TorchvisionPersonDetector

    with pytest.raises(DetectorUnavailable, match="320x320"):
        TorchvisionPersonDetector(config_from_dict({"detector": {"name": "ssdlite320_mobilenet_v3_large"}}).detector,
                                  load_weights=False)
    ok = config_from_dict({"detector": {"name": "ssdlite320_mobilenet_v3_large", "input_size": [320, 320]}}).detector
    assert TorchvisionPersonDetector(ok, load_weights=False).family == "ssd"


def test_detector_sees_resized_frame_and_returns_original_pixels():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from sightline.detector import TorchvisionPersonDetector

    cfg = config_from_dict({}).detector  # input_size [640, 360]
    det = TorchvisionPersonDetector(cfg, load_weights=False)  # random init: no download
    seen = {}

    def fake_model(images):
        seen["shape"] = tuple(images[0].shape)
        seen["transform"] = (det.model.transform.min_size, det.model.transform.max_size)
        return [{"boxes": torch.tensor([[100.0, 50.0, 140.0, 150.0]]), "labels": torch.tensor([1]),
                 "scores": torch.tensor([0.9])}]

    det.model = type("M", (), {"__call__": lambda self, x: fake_model(x), "transform": det.model.transform})()
    out = det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert seen["shape"] == (3, 360, 640)
    assert seen["transform"] == ((360,), 640)  # torchvision's resize pinned to the same size: no second resize
    assert out[0].box == pytest.approx((300.0, 150.0, 420.0, 450.0))  # x3 back to 1920x1080

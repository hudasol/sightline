import pytest
import yaml

from sightline.config import ConfigError, config_from_dict, load_config, load_corridor


def test_default_config_loads_and_is_stable(cfg):
    assert cfg.tracker.kind == "bytetrack"
    assert cfg.baseline_tracker.kind == "iou"
    assert cfg.fingerprint() == load_config("configs/default.yaml").fingerprint()


@pytest.mark.parametrize("patch", [
    {"detector": {"score_threshold": 0.05}},  # not above min_score
    {"detector": {"score_threshold": 1.5}},
    {"tracker": {"kind": "magic"}},
    {"tracker": {"max_age": 0}},
    {"tracker": {"match_iou": 0}},
    {"events": {"n_open": 0}},
    {"events": {"m_close": 0}},
    {"events": {"edge_margin_frac": 0.7}},
    {"health": {"min_fps": 0}},
    {"health": {"gap_factor": 0.5}},
    {"pipeline": {"detect_every": 0}},
    {"pipeline": {"frame_stride": 2}},  # removed by PLAN.md A2: an old config must fail loudly
    {"detector": {"torch_threads": 0}},
    {"events": {"open_s": 0}},
    {"tracker": {"max_age_s": -1}},
])
def test_invalid_config_is_rejected(patch):
    with pytest.raises(ConfigError):
        config_from_dict(patch)


def test_cli_detector_overrides_are_applied_and_validated():
    from argparse import Namespace

    from sightline.cli import _load_cfg

    cfg = _load_cfg(Namespace(config="configs/default.yaml", detector="fasterrcnn_resnet50_fpn_v2", input_size=[960, 540]))
    assert (cfg.detector.name, cfg.detector.input_size) == ("fasterrcnn_resnet50_fpn_v2", [960, 540])
    with pytest.raises(ConfigError):
        _load_cfg(Namespace(config="configs/default.yaml", detector=None, input_size=[8, 8]))
    k3 = _load_cfg(Namespace(config="configs/default.yaml", detector=None, input_size=None, detect_every=3))
    assert k3.pipeline.detect_every == 3


def test_time_based_settings_reproduce_the_frame_defaults_at_30fps(cfg):
    from sightline.config import resolve_timing, timing_summary

    r = timing_summary(resolve_timing(cfg, 30.0, detect_every=1))  # the old frame defaults assumed k = 1
    assert (r["n_open"], r["m_close"], r["cooldown"], r["max_age_detector_frames"]) == (3, 5, 15, 30)


def test_time_based_settings_follow_fps_and_detection_interval(cfg):
    from sightline.config import resolve_timing

    at14 = resolve_timing(cfg, 14.0, detect_every=1)  # MOT17-05 runs at 14 fps
    assert (at14.events.n_open, at14.events.m_close, at14.events.cooldown, at14.tracker.max_age) == (1, 2, 7, 14)
    k3 = resolve_timing(cfg, 30.0, detect_every=3)
    assert k3.tracker.max_age == 10  # 1.0 s of patience = 10 DETECTOR frames when the detector runs at 10 Hz
    assert k3.baseline_tracker.max_age == 10
    assert k3.events.n_open == 3  # events count every processed frame, detected or predicted
    assert cfg.tracker.max_age == 30  # the input config is not modified


def test_frame_counts_stay_when_no_seconds_are_given():
    from sightline.config import resolve_timing

    c = resolve_timing(config_from_dict({"events": {"n_open": 4}, "tracker": {"max_age": 7}}), 25.0)
    assert (c.events.n_open, c.tracker.max_age) == (4, 7)


def test_unknown_keys_are_rejected_not_ignored():
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"detector": {"scor_threshold": 0.4}})  # typo must not silently use defaults
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"detectr": {}})


def test_missing_and_garbage_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("detector: [unclosed")
    with pytest.raises(ConfigError, match="valid YAML"):
        load_config(bad)


def test_corridor_validation(tmp_path):
    def write(d):
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(d))
        return p

    ok = load_corridor(write({"polygon": [[0, 0], [100, 0], [100, 100]], "frame_size": [640, 360]}))
    assert ok.polygon.area() == 5000
    with pytest.raises(ConfigError, match="polygon"):
        load_corridor(write({"name": "x"}))
    with pytest.raises(ConfigError, match="self-intersecting"):
        load_corridor(write({"polygon": [[0, 0], [100, 100], [100, 0], [0, 100]]}))
    with pytest.raises(ConfigError, match="outside"):
        load_corridor(write({"polygon": [[0, 0], [900, 0], [100, 100]], "frame_size": [640, 360]}))

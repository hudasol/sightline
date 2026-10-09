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
    {"pipeline": {"frame_stride": 0}},
])
def test_invalid_config_is_rejected(patch):
    with pytest.raises(ConfigError):
        config_from_dict(patch)


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

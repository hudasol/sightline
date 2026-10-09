from pathlib import Path

import pytest

from sightline.config import load_config, load_corridor
from sightline.detector import read_detection_cache
from sightline.synth import make_synthetic_clip

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    return make_synthetic_clip(tmp_path_factory.mktemp("synth"), "synthetic")


@pytest.fixture(scope="session")
def cached(clip):
    return read_detection_cache(clip.cache)[1]


@pytest.fixture()
def cfg():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture()
def corridor():
    return load_corridor(ROOT / "configs" / "corridors" / "synthetic.yaml")

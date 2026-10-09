import numpy as np
import pytest

from sightline.ingest import IngestError, ingest, probe
from sightline.stress import LEVELS, degrade, shake_offsets


def test_ingest_records_metadata(clip, tmp_path):
    rec = ingest(clip.video, tmp_path)
    assert rec["frames_total"] == clip.n_frames
    assert rec["frames_invalid"] == 0
    assert rec["sha256"]


def test_probe_rejects_empty_and_missing(tmp_path):
    with pytest.raises(IngestError):
        probe(tmp_path / "x.mp4")
    e = tmp_path / "e.mp4"
    e.write_bytes(b"")
    with pytest.raises(IngestError):
        probe(e)


@pytest.mark.parametrize("kind", list(LEVELS))
def test_degradations_keep_shape_and_dtype(kind):
    img = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    off = shake_offsets(3, 6, 0)
    out = degrade(img, kind, LEVELS[kind][-1], 2, off)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_clean_level_is_identity():
    img = np.zeros((10, 10, 3), np.uint8)
    assert degrade(img, "blur", 0, 1) is img


def test_shake_is_deterministic():
    assert shake_offsets(5, 6, 1) == shake_offsets(5, 6, 1)

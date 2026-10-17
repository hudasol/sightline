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


def test_stress_runs_the_detector_only_where_the_pipeline_would_and_once_per_source(clip, cfg, cached):
    from pathlib import Path

    from sightline.evaluation.run import load_eval_config, prepare_clip
    from sightline.manifest import Clip
    from sightline.stress import run_stress

    root = Path(__file__).resolve().parent.parent
    eval_cfg = load_eval_config(root / "configs" / "eval.yaml")
    corridor = str(root / "configs" / "corridors" / "synthetic.yaml")
    base = dict(source=str(clip.video), scene="s", camera="c", session="x", conditions=["clear"],
                corridor=corridor, gt=str(clip.gt), license="t")
    cds = [prepare_clip(Clip(id="a", **base), Path("."), eval_cfg),
           prepare_clip(Clip(id="a-neg", negative=True, metrics=("events",), **base), Path("."), eval_cfg)]

    class Counting:
        name, calls = "count", 0

        def detect(self, image):
            Counting.calls += 1
            return []

    cfg.pipeline.detect_every = 3
    rows = run_stress(cds, cfg, eval_cfg, Path("."), Counting(), kinds=("blur",), progress=lambda m: None)
    per_level = len(range(1, cds[0].n_frames + 1, 3))
    assert Counting.calls == per_level * len(rows)  # every k-th frame, and the -neg entry reuses the source

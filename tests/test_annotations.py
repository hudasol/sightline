import json
from pathlib import Path

import pytest

from sightline.evaluation.run import evaluate_clip, load_eval_config, prepare_clip
from sightline.manifest import Clip, ManifestError, load_manifest, lock_manifest, verify_lock

ROOT = Path(__file__).resolve().parent.parent


def test_only_annotated_frames_are_scored(clip, cfg, cached, tmp_path):
    eval_cfg = load_eval_config(ROOT / "configs" / "eval.yaml")
    corridor = str(ROOT / "configs" / "corridors" / "synthetic.yaml")
    (tmp_path / "af.json").write_text(json.dumps({"frames": [1, 2, 3, 4, 5]}))
    base = dict(source=str(clip.video), scene="s", camera="c", session="x", conditions=["clear"], corridor=corridor,
                gt=str(clip.gt), license="synthetic", metrics=("detection", "tracking"))
    full = evaluate_clip(prepare_clip(Clip(id="full", **base), Path("."), eval_cfg), cfg, cached, eval_cfg, Path("."))
    sparse_clip = Clip(id="sparse", raw={"annotated_frames": str(tmp_path / "af.json")}, **base)
    sparse = evaluate_clip(prepare_clip(sparse_clip, Path("."), eval_cfg), cfg, cached, eval_cfg, Path("."))
    n_gt = lambda r: r.det_counts["tp"] + r.det_counts["fn"]  # noqa: E731
    assert 0 < n_gt(sparse) < n_gt(full)


def test_sparse_clip_needs_event_labels_and_its_frame_list_is_locked(tmp_path):
    for name, text in (("v.txt", "x"), ("g.txt", "y"), ("c.yaml", "polygon: []"), ("af.json", '{"frames": [1]}')):
        (tmp_path / name).write_text(text)
    clip = {"id": "a", "source": "v.txt", "scene": "s", "camera": "c", "session": "x", "conditions": ["clear"],
            "corridor": "c.yaml", "gt": "g.txt", "license": "t", "annotated_frames": "af.json"}
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"clips": [clip]}))
    with pytest.raises(ManifestError, match="gt_events"):
        load_manifest(m)
    m.write_text(json.dumps({"clips": [{**clip, "metrics": ["detection", "tracking"]}]}))
    lock_manifest(m, tmp_path / "lock.json", tmp_path)
    assert verify_lock(tmp_path / "lock.json", tmp_path) == []
    (tmp_path / "af.json").write_text('{"frames": [1, 2]}')
    assert any("annotated-frames" in p for p in verify_lock(tmp_path / "lock.json", tmp_path))

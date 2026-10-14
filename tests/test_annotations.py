import json
import zipfile
from pathlib import Path

import pytest

from sightline.annotations import AnnotationError, import_cvat_mot, load_annotated_frames, parse_frame_spec
from sightline.evaluation.gt import load_mot_gt
from sightline.evaluation.run import evaluate_clip, load_eval_config, prepare_clip
from sightline.manifest import Clip, ManifestError, load_manifest, lock_manifest, verify_lock

ROOT = Path(__file__).resolve().parent.parent


def _export(tmp_path, rows, labels=("person", "car", "reflection")):
    z = tmp_path / "cvat.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("gt/labels.txt", "\n".join(labels) + "\n")
        f.writestr("gt/gt.txt", "\n".join(rows) + "\n")
    return z


def test_frame_specs():
    assert parse_frame_spec("1-5") == [1, 2, 3, 4, 5]
    assert parse_frame_spec("1-21:10") == [1, 11, 21]
    assert parse_frame_spec("1-3, 10") == [1, 2, 3, 10]
    for bad in ("0-3", "5-1", "a-b", "1-5:0"):
        with pytest.raises(AnnotationError):
            parse_frame_spec(bad)


def test_cvat_import_keeps_people_ignores_and_drops_the_rest(tmp_path):
    rows = ["1,1,10,10,20,60,1,1,0.8",   # person
            "1,2,100,10,40,20,1,2,1.0",  # car: dropped
            "11,3,200,10,20,60,1,3,1.0", # reflection: ignore region
            "11,1,12,10,20,60,1,1,0.5"]  # person
    rep = import_cvat_mot(_export(tmp_path, rows), tmp_path / "ann", "1-21:10", ignore_labels=("reflection",))
    assert (rep["person_boxes"], rep["ignore_boxes"], rep["dropped_rows_by_label"]) == (2, 1, {"car": 1})
    assert rep["annotated_frames"] == 3 and not rep["warnings"]
    gt, ignore = load_mot_gt(tmp_path / "ann" / "gt.txt")
    assert sorted(gt) == [1, 11] and len(ignore[11]) == 1  # class 8 rows load as ignore regions
    assert load_annotated_frames(tmp_path / "ann" / "annotated_frames.json") == {1, 11, 21}
    assert b"\r\n" not in (tmp_path / "ann" / "gt.txt").read_bytes()  # LF: the lock hashes these bytes


def test_cvat_import_refuses_boxes_on_unannotated_frames_and_flags_missing_visibility(tmp_path):
    with pytest.raises(AnnotationError, match="not annotated"):
        import_cvat_mot(_export(tmp_path, ["5,1,10,10,20,60,1,1,1.0"]), tmp_path / "a", "1-3")
    rep = import_cvat_mot(_export(tmp_path, ["1,1,10,10,20,60,1,1,1.0"]), tmp_path / "b", "1-3")
    assert rep["warnings"] and "visibility" in rep["warnings"][0]


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

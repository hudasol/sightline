import json

import pytest

from sightline.manifest import ManifestError, load_manifest, lock_manifest, split_check, verify_lock


def _write(path, clips, split="test"):
    path.write_text(json.dumps({"split": split, "clips": clips}))
    return path


def _clip(i, scene, src, **kw):
    return {"id": i, "scene": scene, "source": src, "gt": f"{i}.txt", "fps": 30, "camera": f"cam-{scene}",
            "session": f"sess-{scene}", "conditions": ["clear"], "corridor": "c.yaml", "license": "test", **kw}


def test_split_check_detects_scene_leak(tmp_path):
    v = _write(tmp_path / "v.json", [_clip("a", "MOT17-02", "a")])
    t = _write(tmp_path / "t.json", [_clip("b", "MOT17-02", "b")])
    problems = split_check(load_manifest(v), load_manifest(t))
    assert problems


def test_split_check_clean(tmp_path):
    v = _write(tmp_path / "v.json", [_clip("a", "MOT17-02", "a")])
    t = _write(tmp_path / "t.json", [_clip("b", "MOT17-04", "b")])
    assert split_check(load_manifest(v), load_manifest(t)) == []


def test_lock_detects_tamper(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("x")
    gt = tmp_path / "a.gt"
    gt.write_text("y")
    (tmp_path / "c.yaml").write_text("polygon: []")
    m = _write(tmp_path / "m.json", [_clip("a", "s", "a.txt", gt="a.gt")])
    lock = tmp_path / "lock.json"
    lock_manifest(m, lock, tmp_path)
    assert verify_lock(lock, tmp_path) == []
    gt.write_text("tampered")
    assert verify_lock(lock, tmp_path)


def test_lock_detects_corridor_and_eval_rule_tamper(tmp_path):
    for name, text in (("a.txt", "x"), ("a.gt", "y"), ("c.yaml", "polygon: []"), ("eval.yaml", "gt_events: {}")):
        (tmp_path / name).write_text(text)
    m = _write(tmp_path / "m.json", [_clip("a", "s", "a.txt", gt="a.gt")])
    lock = tmp_path / "lock.json"
    lock_manifest(m, lock, tmp_path, eval_config="eval.yaml")
    assert verify_lock(lock, tmp_path) == []
    (tmp_path / "c.yaml").write_text("polygon: [[0, 0]]")
    assert any("corridor" in p for p in verify_lock(lock, tmp_path))
    (tmp_path / "c.yaml").write_text("polygon: []")
    (tmp_path / "eval.yaml").write_text("gt_events: {min_frames: 1}")
    assert any("evaluation rules" in p for p in verify_lock(lock, tmp_path))


def _complete_heldout(tmp_path):
    """12 tiny clips covering the three difficult conditions plus a negative clip."""
    (tmp_path / "c.yaml").write_text("polygon: []")
    conds = ["occlusion_crowding", "camera_motion_blur", "low_light_compression"]
    clips = []
    for i in range(12):
        (tmp_path / f"v{i}.txt").write_text(str(i))
        (tmp_path / f"g{i}.txt").write_text(str(i))
        clips.append(_clip(f"c{i}", f"s{i}", f"v{i}.txt", gt=f"g{i}.txt", conditions=[conds[i % 3]], negative=(i == 0)))
    return _write(tmp_path / "test.json", clips)


def test_lock_status_is_draft_until_coverage_is_met(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "a.gt").write_text("y")
    (tmp_path / "c.yaml").write_text("polygon: []")
    draft = lock_manifest(_write(tmp_path / "m.json", [_clip("a", "s", "a.txt", gt="a.gt")]), tmp_path / "d.json", tmp_path)
    assert draft["status"] == "draft"
    assert lock_manifest(_complete_heldout(tmp_path), tmp_path / "f.json", tmp_path)["status"] == "final"


def test_tune_refuses_a_draft_lock(tmp_path):
    from sightline.tune import tune

    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "a.gt").write_text("y")
    (tmp_path / "c.yaml").write_text("polygon: []")
    lock_manifest(_write(tmp_path / "m.json", [_clip("a", "s", "a.txt", gt="a.gt")]), tmp_path / "lock.json", tmp_path)
    with pytest.raises(ManifestError, match="DRAFT"):
        tune(tmp_path / "val.json", tmp_path / "lock.json", None, {}, tmp_path, tmp_path, None, tmp_path / "f.yaml", tmp_path)


def test_final_lock_is_refused_once_its_files_change(tmp_path):
    from sightline.manifest import require_final_lock

    lock_manifest(_complete_heldout(tmp_path), tmp_path / "lock.json", tmp_path)
    assert require_final_lock(tmp_path / "lock.json", tmp_path)["status"] == "final"
    (tmp_path / "g3.txt").write_text("relabelled")
    with pytest.raises(ManifestError, match="changed since locking"):
        require_final_lock(tmp_path / "lock.json", tmp_path)


def test_own_clip_template_entries_are_complete(tmp_path):
    """Copying an entry from the template must never produce an invalid manifest."""
    from pathlib import Path

    from sightline.manifest import REQUIRED_FIELDS

    root = Path(__file__).resolve().parent.parent
    t = json.loads((root / "configs" / "manifests" / "own_clip.template.json").read_text())
    for c in t["clips"]:
        assert all(k in c for k in REQUIRED_FIELDS) and c["dataset"] == "own"
    # with placeholders filled in (unique ids and a valid condition), the template loads
    clips = [{**c, "id": f"own-x-{i}", "conditions": ["clear"]} for i, c in enumerate(t["clips"])]
    loaded = load_manifest(_write(tmp_path / "m.json", clips))
    assert [c.negative for c in loaded.clips] == [False, True, False]
    assert all(c.dataset == "own" for c in loaded.clips)


def test_missing_file_rejected(tmp_path):
    with pytest.raises((ManifestError, FileNotFoundError)):
        load_manifest(tmp_path / "none.json")

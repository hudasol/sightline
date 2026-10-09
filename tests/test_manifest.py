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
    m = _write(tmp_path / "m.json", [_clip("a", "s", "a.txt", gt="a.gt")])
    lock = tmp_path / "lock.json"
    lock_manifest(m, lock, tmp_path)
    assert verify_lock(lock, tmp_path) == []
    gt.write_text("tampered")
    assert verify_lock(lock, tmp_path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises((ManifestError, FileNotFoundError)):
        load_manifest(tmp_path / "none.json")

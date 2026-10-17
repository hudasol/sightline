import json
from pathlib import Path

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


def test_a_second_corridor_on_the_same_footage_is_an_entry_not_a_clip(tmp_path):
    from sightline.manifest import coverage_check

    def entry(cid, src, neg=False):
        e = {"id": cid, "source": src, "scene": "s", "camera": "c", "session": "x", "conditions": ["clear"],
             "corridor": "c.yaml", "gt": "gt.txt", "license": "t"}
        return {**e, "negative": True, "metrics": ["events"]} if neg else e

    p = tmp_path / "test.json"
    p.write_text(json.dumps({"split": "test", "clips": [entry("a", "data/a"), entry("a-neg", "data/a", True),
                                                        entry("b", "data/b")]}))
    cov = coverage_check(load_manifest(p), min_clips=3)
    assert (cov["n_clips"], cov["n_entries"], cov["negative_clips"]) == (2, 3, 1)
    assert cov["meets_min_clips"] is False


@pytest.mark.parametrize("shared, label", [
    ({"camera": "cam-X"}, "camera"),
    ({"session": "sess-X"}, "recording session"),
    ({"source": "same.mp4"}, "source path"),
])
def test_split_check_catches_each_key_on_its_own(tmp_path, shared, label):
    """Scenes differ; exactly one other key is shared. Each must be caught by itself."""
    v = _write(tmp_path / "v.json", [_clip("a", "site-1", "a.mp4", **shared)])
    t = _write(tmp_path / "t.json", [_clip("b", "site-2", "b.mp4", **shared)])
    problems = split_check(load_manifest(v), load_manifest(t))
    assert len(problems) == 1 and problems[0].startswith(f"val and test share {label}")


def test_split_check_catches_two_detector_copies_of_one_mot_scene(tmp_path):
    # MOT17-02-DPM and MOT17-02-FRCNN are the same video under different names
    v = _write(tmp_path / "v.json", [_clip("MOT17-02-DPM", "MOT17-02-DPM", "a", camera="c1", session="s1")])
    t = _write(tmp_path / "t.json", [_clip("MOT17-02-FRCNN", "MOT17-02-FRCNN", "b", camera="c2", session="s2")])
    assert split_check(load_manifest(v), load_manifest(t)) == ["val and test share base scene: 'MOT-02'"]


def test_evaluate_refuses_without_a_final_lock(tmp_path):
    from sightline.cli import main

    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "a.gt").write_text("y")
    (tmp_path / "c.yaml").write_text("polygon: []")
    cfg = tmp_path / "frozen.yaml"  # a loadable config, so the refusal can only come from the lock
    root = Path(__file__).resolve().parent.parent
    cfg.write_text((root / "configs" / "default.yaml").read_text() + "\nprovenance: {frozen: true}\n")
    assert main(["evaluate", "--lock", str(tmp_path / "missing.json"), "--config", str(cfg), "--root", str(tmp_path),
                 "--results", str(tmp_path / "r")]) == 2
    lock_manifest(_write(tmp_path / "m.json", [_clip("a", "s", "a.txt", gt="a.gt")]), tmp_path / "lock.json", tmp_path)
    assert main(["evaluate", "--lock", str(tmp_path / "lock.json"), "--config", str(cfg), "--root", str(tmp_path),
                 "--results", str(tmp_path / "r")]) == 2  # a draft lock is refused
    assert not (tmp_path / "r" / "final_metrics.json").exists()

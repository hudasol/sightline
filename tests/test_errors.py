import json
from pathlib import Path

from sightline.errors import select_unresolved

ROOT = Path(__file__).resolve().parent.parent


def _per_clip(d, clip, events):
    (d / "bytetrack").mkdir(parents=True, exist_ok=True)
    (d / "bytetrack" / f"{clip}.json").write_text(json.dumps({"clip": clip, "events_predicted": events}))


def _ev(eid, open_f, close_f, status="UNRESOLVED", reason="track_lost"):
    return {"event_id": eid, "track_id": eid, "status": status, "close_reason": reason, "open_frame": open_f,
            "first_inside_frame": open_f - 2, "last_inside_frame": close_f - 5, "close_frame": close_f}


def test_unresolved_selection_is_worst_first_and_skips_non_failures(tmp_path):
    _per_clip(tmp_path, "b", [_ev(1, 10, 20), _ev(2, 10, 90), _ev(3, 5, 400, reason="stream_end")])
    _per_clip(tmp_path, "a", [_ev(7, 10, 20), _ev(8, 10, 50, status="CLOSED", reason="left_corridor")])
    got = select_unresolved(tmp_path, n=10)
    assert [(x["clip"], x["event_id"]) for x in got] == [("b", 2), ("a", 7), ("b", 1)]  # 80, then ties by clip
    assert select_unresolved(tmp_path, n=1)[0]["open_frames_before_loss"] == 80


def test_unresolved_strips_on_val_style_run(clip, cached, tmp_path):
    """End to end on the synthetic clip: details from cached detections, then strips and an index."""
    from sightline.config import load_config
    from sightline.detector import write_detection_cache
    from sightline.errors import analyze_unresolved, collect_details
    from sightline.evaluation.run import detector_tag, load_eval_config

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cache = tmp_path / "cache"
    write_detection_cache(cache / detector_tag(cfg) / "syn.jsonl",
                          {"detector": cfg.detector.name, "min_score": cfg.detector.min_score,
                           "input_size": list(cfg.detector.input_size)}, cached)
    man = tmp_path / "val.json"
    man.write_text(json.dumps({"split": "val", "clips": [{
        "id": "syn", "source": str(clip.video), "scene": "s", "camera": "c", "session": "x", "conditions": ["clear"],
        "corridor": str(ROOT / "configs" / "corridors" / "synthetic.yaml"), "gt": str(clip.gt), "license": "t"}]}))
    res = tmp_path / "res"
    assert collect_details(man, cfg, load_eval_config(ROOT / "configs" / "eval.yaml"), Path("."), cache, res) == ["syn"]
    out = analyze_unresolved(res, Path("."), man, cache, detector_tag(cfg), 0.5, 10, "test run")
    index = json.loads((out / "index.json").read_text())
    assert index["rule"].startswith("UNRESOLVED") and (out / "INDEX.md").is_file()
    assert all((out / it["image"]).is_file() for it in index["items"] if it["image"])

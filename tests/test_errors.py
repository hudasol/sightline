import json
from pathlib import Path

from sightline.errors import select_false_events, select_unresolved

ROOT = Path(__file__).resolve().parent.parent


def _per_clip(d, clip, events):
    (d / "bytetrack").mkdir(parents=True, exist_ok=True)
    (d / "bytetrack" / f"{clip}.json").write_text(json.dumps({"clip": clip, "events_predicted": events}))


def _ev(eid, open_f, close_f, status="UNRESOLVED", reason="track_lost"):
    return {"event_id": eid, "track_id": eid, "status": status, "close_reason": reason, "open_frame": open_f,
            "first_inside_frame": open_f - 2, "last_inside_frame": close_f - 5, "close_frame": close_f}


def test_unresolved_items_say_which_gt_event_they_matched(tmp_path):
    (tmp_path / "bytetrack").mkdir()
    (tmp_path / "bytetrack" / "c.json").write_text(json.dumps({
        "clip": "c", "events_predicted": [_ev(1, 10, 40), _ev(2, 10, 30)],
        "event_pairs": [[5, 1]], "events_gt": [{"id": 5, "first_inside_frame": 9, "last_inside_frame": 50}]}))
    got = {x["event_id"]: x for x in select_unresolved(tmp_path)}
    assert got[1]["matched_gt_event"] == 5 and got[1]["matched_gt_frames"] == [9, 50]
    assert got[2]["matched_gt_event"] is None and got[2]["matched_gt_frames"] is None


def test_unresolved_selection_is_worst_first_and_skips_non_failures(tmp_path):
    _per_clip(tmp_path, "b", [_ev(1, 10, 20), _ev(2, 10, 90), _ev(3, 5, 400, reason="stream_end")])
    _per_clip(tmp_path, "a", [_ev(7, 10, 20), _ev(8, 10, 50, status="CLOSED", reason="left_corridor")])
    got = select_unresolved(tmp_path, n=10)
    assert [(x["clip"], x["event_id"]) for x in got] == [("b", 2), ("a", 7), ("b", 1)]  # 80, then ties by clip
    assert select_unresolved(tmp_path, n=1)[0]["open_frames_before_loss"] == 80


def test_false_event_selection_is_longest_open_first_and_names_overlapping_gt(tmp_path):
    (tmp_path / "bytetrack").mkdir()
    (tmp_path / "bytetrack" / "b.json").write_text(json.dumps({
        "clip": "b", "events_predicted": [_ev(1, 10, 40, "CLOSED", "left_corridor"), _ev(2, 12, 90), _ev(3, 200, 260)],
        "event_pairs": [[5, 1]], "false_events": [2, 3],
        "events_gt": [{"id": 5, "first_inside_frame": 9, "last_inside_frame": 50}]}))
    (tmp_path / "bytetrack" / "a.json").write_text(json.dumps({
        "clip": "a", "events_predicted": [_ev(7, 10, 70, "CLOSED", "left_corridor")], "event_pairs": [],
        "false_events": [7], "events_gt": []}))
    got = select_false_events(tmp_path, n=10)
    assert [(x["clip"], x["event_id"]) for x in got] == [("b", 2), ("a", 7), ("b", 3)]  # 78, 60, 60 (tie by clip)
    assert got[0]["overlapping_gt_events"] == [{"gt_event": 5, "gt_frames": [9, 50], "matched_to_event": 1}]
    assert got[2]["overlapping_gt_events"] == [] and got[1]["status"] == "CLOSED"
    assert len(select_false_events(tmp_path, n=1)) == 1


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
    assert index["requested"] == 10 and index["selected"] == len(index["items"])
    assert f"Requested 10; {index['selected']} event(s)" in (out / "INDEX.md").read_text()
    assert all((out / it["image"]).is_file() for it in index["items"] if it["image"])
    from sightline.errors import analyze_false_events

    fe = json.loads((analyze_false_events(res, Path("."), man, cache, detector_tag(cfg), 0.5, 10, "test run")
                     / "index.json").read_text())
    assert fe["rule"].startswith("False events") and fe["selected"] == len(fe["items"])


def test_failures_come_only_from_clips_scored_on_their_family(tmp_path):
    from sightline.errors import select_failures

    (tmp_path / "bytetrack").mkdir()
    detail = {"false_positives": [{"frame": 3, "score": 0.9, "box": [0, 0, 1, 1]}],
              "misses": [{"frame": 4, "box": [0, 0, 5, 5]}], "id_switches": [{"frame": 6}],
              "events_predicted": [], "events_gt": [], "event_pairs": [], "false_events": []}
    for cid in ("a", "a-neg"):  # the same footage twice: a clip and a negative-corridor entry
        (tmp_path / "bytetrack" / f"{cid}.json").write_text(json.dumps({"clip": cid, **detail}))
    got = select_failures(tmp_path, tmp_path, per_type=5,
                          scored={"a": ("detection", "tracking", "events"), "a-neg": ("events",)})
    assert [(x["type"], x["clip"]) for x in got] == [("false_detection", "a"), ("missed_person", "a"), ("id_switch", "a")]

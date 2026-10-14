import json

from sightline.selection import build_table, to_markdown


def _write(root, tag, p, r, fps):
    d = root / "detector_selection"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"threshold": 0.5, "precision": p, "recall": r, "f1": 2 * p * r / (p + r), "tp": 1, "fp": 1, "fn": 1}]
    (d / f"detect_eval_val_{tag}.json").write_text(json.dumps(
        {"detector": tag, "input_size": [640, 360], "frames_evaluated": 10, "overall": rows}))
    t = root / "runtime" / "candidates" / tag
    t.mkdir(parents=True, exist_ok=True)
    (t / "a.json").write_text(json.dumps({"max_frames_per_clip": 150, "per_clip": [
        {"frames_processed": 100, "wall_seconds": 100 / fps, "inference_ms_p50": 50.0}]}))


def test_fastest_candidate_meeting_both_bars_is_chosen(tmp_path):
    _write(tmp_path, "slow_good", 0.9, 0.9, 11.0)
    _write(tmp_path, "fast_good", 0.85, 0.82, 20.0)
    _write(tmp_path, "fast_bad", 0.9, 0.5, 40.0)
    t = build_table(tmp_path, ["slow_good", "fast_good", "fast_bad"], 0.8, 0.8, 10.0, 0.5)
    assert t["choice"] == "fast_good"


def test_selection_is_judged_on_the_in_scope_view(tmp_path):
    _write(tmp_path, "a", 0.9, 0.9, 20.0)  # passes on all boxes...
    f = tmp_path / "detector_selection" / "detect_eval_val_a.json"
    d = json.loads(f.read_text())
    d["overall_in_scope"] = [{"threshold": 0.5, "precision": 0.9, "recall": 0.6, "f1": 0.72, "tp": 1, "fp": 1, "fn": 1}]
    f.write_text(json.dumps(d))  # ...but not in scope
    t = build_table(tmp_path, ["a"], 0.8, 0.8, 10.0, 0.5)
    assert t["rows"][0]["judged_view"] == "in_scope" and t["choice"] is None
    md = to_markdown(t)
    assert "### In scope" in md and "### All boxes" in md


def test_no_candidate_means_sampling_not_a_lower_bar(tmp_path):
    _write(tmp_path, "a", 0.9, 0.3, 5.0)
    t = build_table(tmp_path, ["a", "missing"], 0.8, 0.8, 10.0, 0.5)
    assert t["choice"] is None and "frame-sampling" in t["verdict"]
    assert t["rows"][1]["pending"] and t["rows"][1]["meets_pr_bar"] is None
    assert "pending" in to_markdown(t)

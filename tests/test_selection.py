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


def test_threshold_grid_stays_above_the_detection_floor(cfg):
    from sightline.evaluation.detection import THRESHOLDS

    assert min(THRESHOLDS) > cfg.detector.min_score and THRESHOLDS[0] == 0.15 and THRESHOLDS[-1] == 0.9


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
    assert t["choice"] is None and "detection interval" in t["verdict"]
    assert t["rows"][1]["pending"] and t["rows"][1]["meets_pr_bar"] is None
    assert "pending" in to_markdown(t)


def _a4_fixture(tmp_path, f1s, runtime):
    import json as _json

    (tmp_path / "detector_selection").mkdir()
    for tag, f1 in f1s.items():
        (tmp_path / "detector_selection" / f"detect_eval_val_{tag}.json").write_text(_json.dumps(
            {"overall_in_scope": [{"threshold": 0.4, "precision": f1, "recall": f1, "f1": f1}]}))
    rs = tmp_path / "summary.json"
    rs.write_text(_json.dumps({lab: {"runs": 3, "throughput_fps": {"median": v, "min": v - 1, "max": v + 1}}
                               for lab, v in runtime.items()}))
    return rs


def test_a4_picks_the_most_accurate_qualifying_pair_and_its_smallest_k(tmp_path):
    from sightline.selection import select_a4

    rs = _a4_fixture(tmp_path, {"a": 0.85, "b": 0.78},
                     {"a_k1": 6.0, "a_k2": 12.0, "a_k3": 18.0, "b_k1": 30.0})
    out = select_a4(tmp_path, rs, 10.0, candidates=("a", "b"), intervals=(1, 2, 3))
    assert (out["choice"]["detector"], out["choice"]["k"]) == ("a", 2)  # b is faster but less accurate
    assert out["verdict"] == "valid" and out["operating_threshold"] == 0.4


def test_a4_says_when_an_unmeasured_pair_could_still_win(tmp_path):
    from sightline.selection import select_a4

    rs = _a4_fixture(tmp_path, {"a": 0.85, "b": 0.90}, {"a_k2": 12.0})
    out = select_a4(tmp_path, rs, 10.0, candidates=("a", "b"), intervals=(1, 2))
    assert out["verdict"].startswith("INCOMPLETE") and "b k=1" in out["unmeasured_pairs_that_could_change_the_choice"]
    assert "a k=1" in out["unmeasured_pairs_that_could_change_the_choice"]  # a smaller k of the chosen detector


def test_a4_selects_nothing_when_no_pair_reaches_the_fps_bar(tmp_path):
    from sightline.selection import select_a4

    rs = _a4_fixture(tmp_path, {"a": 0.85}, {"a_k1": 6.0})
    out = select_a4(tmp_path, rs, 10.0, candidates=("a",), intervals=(1,))
    assert out["choice"] is None and out["verdict"].startswith("no pair qualifies")

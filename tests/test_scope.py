"""Clip.metrics: a moving-camera clip counts for detection and tracking, never for events."""
import json
from pathlib import Path

import pytest

from sightline.evaluation.run import aggregate, evaluate_clip, load_eval_config, prepare_clip
from sightline.manifest import Clip, ManifestError, coverage_check, load_manifest

ROOT = Path(__file__).resolve().parent.parent


def _manifest(tmp_path, **extra):
    clip = {"id": "a", "source": "a", "scene": "s", "camera": "c", "session": "x", "conditions": ["clear"],
            "corridor": "c.yaml", "gt": "g.txt", "license": "t", **extra}
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"split": "val", "clips": [clip]}))
    return p


def test_metrics_default_to_all_and_are_validated(tmp_path):
    assert load_manifest(_manifest(tmp_path)).clips[0].metrics == ("detection", "tracking", "events")
    assert load_manifest(_manifest(tmp_path, metrics=["tracking", "detection"])).clips[0].metrics == ("detection", "tracking")
    for bad in ([], ["events", "events"], ["counting"], "events"):
        with pytest.raises(ManifestError, match="metrics"):
            load_manifest(_manifest(tmp_path, metrics=bad))
    with pytest.raises(ManifestError, match="negative"):
        load_manifest(_manifest(tmp_path, metrics=["detection", "tracking"], negative=True))


def test_coverage_reports_what_events_rest_on(tmp_path):
    p = tmp_path / "t.json"
    base = {"source": "a", "scene": "s", "camera": "c", "corridor": "c.yaml", "gt": "g", "license": "t"}
    p.write_text(json.dumps({"clips": [
        {**base, "id": "static", "session": "1", "conditions": ["clear"]},
        {**base, "id": "moving", "session": "2", "conditions": ["camera_motion_blur", "low_light_compression"],
         "metrics": ["detection", "tracking"]},
    ]}))
    cov = coverage_check(load_manifest(p))
    assert cov["difficult_conditions_present"] == ["camera_motion_blur", "low_light_compression"]
    assert cov["clips_scored_on"] == {"detection": 2, "tracking": 2, "events": 1}
    assert cov["event_difficult_conditions_present"] == []


def test_detection_bar_is_judged_on_the_named_dataset_in_scope(clip, cfg, cached):
    from sightline.evaluation.run import bar_check

    eval_cfg = load_eval_config(ROOT / "configs" / "eval.yaml")
    corridor = str(ROOT / "configs" / "corridors" / "synthetic.yaml")
    c = Clip(id="pub", source=str(clip.video), scene="MOT17-02", camera="c", session="s", conditions=["clear"],
             corridor=corridor, gt=str(clip.gt), license="synthetic")
    assert c.dataset == "mot17"
    agg = aggregate([evaluate_clip(prepare_clip(c, Path("."), eval_cfg), cfg, cached, eval_cfg, Path("."))])
    assert set(agg["detection_in_scope_by_dataset"]) == {"mot17"}
    assert "ALL" in agg["detection_in_scope"] and "ALL" in agg["tracking_in_scope"]["bytetrack"]
    cov = {"meets_min_clips": True, "meets_three_difficult": True, "has_negative_set": True}
    bar = bar_check(agg, cov, eval_cfg)
    # MOT17 is reported, never judged: with no IndoorCrowd clips (A6) the detection checks cannot pass
    assert bar["detection_judged_on"] == "indoorcrowd clips, in scope"
    assert bar["checks"]["detection_precision>=0.80"] is False and bar["checks"]["detection_recall>=0.80"] is False


def test_aggregate_keeps_moving_camera_clips_out_of_events(clip, cfg, cached):
    eval_cfg = load_eval_config(ROOT / "configs" / "eval.yaml")
    corridor = str(ROOT / "configs" / "corridors" / "synthetic.yaml")
    results = []
    for cid, metrics in (("static", ("detection", "tracking", "events")), ("moving", ("detection", "tracking"))):
        c = Clip(id=cid, source=str(clip.video), scene=cid, camera=cid, session=cid, conditions=["clear"],
                 corridor=corridor, gt=str(clip.gt), license="synthetic", metrics=metrics)
        cd = prepare_clip(c, Path("."), eval_cfg)
        results.append(evaluate_clip(cd, cfg, cached, eval_cfg, Path(".")))
    agg = aggregate(results)
    assert agg["scope"]["events"] == ["static"]
    assert agg["detection"]["ALL"]["n_clips"] == 2
    assert agg["tracking"]["bytetrack"]["ALL"]["n_clips"] == 2
    assert agg["events"]["bytetrack"]["ALL"]["n_clips"] == 1
    # the static copy alone: its events are exactly half of what pooling both copies would give
    assert agg["events"]["bytetrack"]["ALL"]["events_tp"] == results[0].matches["bytetrack"].tp

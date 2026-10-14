import json
from pathlib import Path

from sightline.evaluation.interval import interval_sweep
from sightline.evaluation.run import load_eval_config
from sightline.manifest import load_manifest
from sightline.runner import run_clip

ROOT = Path(__file__).resolve().parent.parent


def test_interval_sweep_reports_events_on_event_clips_only(clip, cfg, cached, tmp_path, monkeypatch):
    monkeypatch.setattr("sightline.evaluation.interval.get_detections", lambda *a, **k: cached)
    base = {"source": str(clip.video), "gt": str(clip.gt), "corridor": str(ROOT / "configs/corridors/synthetic.yaml"),
            "conditions": ["clear"], "license": "synthetic"}
    p = tmp_path / "val.json"
    p.write_text(json.dumps({"split": "val", "clips": [
        {**base, "id": "static", "scene": "a", "camera": "a", "session": "a"},
        {**base, "id": "moving", "scene": "b", "camera": "b", "session": "b", "metrics": ["detection", "tracking"]},
    ]}))
    out = interval_sweep(load_manifest(p), cfg, load_eval_config(ROOT / "configs/eval.yaml"), Path("."), tmp_path,
                         None, [1, 3], progress=lambda *_: None)
    assert out["event_clips"] == ["static"] and out["excluded_not_scored_on_events"] == ["moving"]
    by_k = {r["detect_every"]: r for r in out["rows"]}
    assert by_k[1]["events_tp"] + by_k[1]["events_fn"] == by_k[3]["events_tp"] + by_k[3]["events_fn"]  # same GT
    assert by_k[1]["per_clip"][0]["detector_runs_per_second"] == 3 * by_k[3]["per_clip"][0]["detector_runs_per_second"]
    for r in out["rows"]:  # latency is reported in ORIGINAL frames whatever the interval
        lat = [x for c in r["per_clip"] for x in c["latencies_frames"]]
        assert r["latency_frames"]["n"] == len(lat) == r["events_tp"]


def test_every_frame_is_processed_and_the_detector_runs_every_kth(clip, cfg, corridor, cached):
    cfg.pipeline.detect_every = 3
    run = run_clip(clip.video, cfg, corridor, cached=cached, keep_results=True)
    ran = [r.frame_index for r in run.results if r.detector_ran]
    assert run.frames_seen == clip.n_frames == len(run.results)  # no frame skipped
    assert ran == list(range(1, clip.n_frames + 1, 3))
    assert run.policy["detect_every"] == 3
    # tracks are still reported on frames the detector skipped, flagged as predicted
    assert any(t.predicted for r in run.results if not r.detector_ran for t in r.reported_tracks)

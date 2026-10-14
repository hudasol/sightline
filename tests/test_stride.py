import json
from pathlib import Path

from sightline.evaluation.run import load_eval_config
from sightline.evaluation.stride import stride_sweep
from sightline.manifest import load_manifest

ROOT = Path(__file__).resolve().parent.parent


def test_stride_sweep_reports_events_on_event_clips_only(clip, cfg, cached, tmp_path, monkeypatch):
    monkeypatch.setattr("sightline.evaluation.stride.get_detections", lambda *a, **k: cached)
    base = {"source": str(clip.video), "gt": str(clip.gt), "corridor": str(ROOT / "configs/corridors/synthetic.yaml"),
            "conditions": ["clear"], "license": "synthetic"}
    p = tmp_path / "val.json"
    p.write_text(json.dumps({"split": "val", "clips": [
        {**base, "id": "static", "scene": "a", "camera": "a", "session": "a"},
        {**base, "id": "moving", "scene": "b", "camera": "b", "session": "b", "metrics": ["detection", "tracking"]},
    ]}))
    out = stride_sweep(load_manifest(p), cfg, load_eval_config(ROOT / "configs/eval.yaml"), Path("."), tmp_path,
                       None, [1, 2], progress=lambda *_: None)
    assert out["event_clips"] == ["static"] and out["excluded_not_scored_on_events"] == ["moving"]
    by_s = {r["frame_stride"]: r for r in out["rows"]}
    assert set(by_s) == {1, 2}
    assert by_s[1]["events_tp"] + by_s[1]["events_fn"] == by_s[2]["events_tp"] + by_s[2]["events_fn"]  # same GT
    assert by_s[1]["per_clip"][0]["required_processing_fps"] == 2 * by_s[2]["per_clip"][0]["required_processing_fps"]
    for r in out["rows"]:  # latency is reported in ORIGINAL frames whatever the stride
        lat = [x for c in r["per_clip"] for x in c["latencies_frames"]]
        assert r["latency_frames"]["n"] == len(lat) == r["events_tp"]
        assert all(isinstance(x, int) for x in lat)

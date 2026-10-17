import json
from pathlib import Path

from sightline.ablation import VARIANTS, run_ablation
from sightline.config import load_config
from sightline.detector import write_detection_cache
from sightline.evaluation.run import detector_tag, load_eval_config

ROOT = Path(__file__).resolve().parent.parent


def test_ablation_runs_every_variant_on_cached_detections(clip, cached, tmp_path):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cache = tmp_path / "cache"
    write_detection_cache(cache / detector_tag(cfg) / "syn.jsonl",
                          {"detector": cfg.detector.name, "min_score": cfg.detector.min_score,
                           "input_size": list(cfg.detector.input_size)}, cached)
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"split": "test", "clips": [{
        "id": "syn", "source": str(clip.video), "scene": "s", "camera": "c", "session": "x", "conditions": ["clear"],
        "corridor": str(ROOT / "configs" / "corridors" / "synthetic.yaml"), "gt": str(clip.gt), "license": "t"}]}))
    out = run_ablation(man, cfg, load_eval_config(ROOT / "configs" / "eval.yaml"), Path("."), cache, progress=lambda m: None)
    assert list(out["variants"]) == list(VARIANTS)
    for row in out["variants"].values():
        assert row["idf1"] is not None and row["id_switches"] is not None and row["n_clips_tracking"] == 1


def test_switching_off_the_low_score_stage_and_motion_changes_the_tracker(clip, cached):
    from sightline.tracking import make_tracker

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.tracker.use_motion = False
    t = make_tracker(cfg.tracker, 0.5, 0.1)
    from sightline.types import Detection

    t.update([Detection((0, 0, 10, 20), 0.9)], 1, 0.0)
    t.update([Detection((4, 0, 14, 20), 0.9)], 2, 0.1)  # moving right
    out = t.predict(3, 0.2)
    assert out.tracks[0].box == (4, 0, 14, 20)  # no motion model: the box is held, not extrapolated

import json

import pytest

from sightline.runtime import summarize_runs


def _run(parent, name, fps, p95, mx, dropped):
    d = parent / name
    d.mkdir()
    for clip in ("a", "b"):
        (d / f"{clip}.json").write_text(json.dumps({"async_detection": False, "realtime_emulation": True, "per_clip": [
            {"frames_processed": 100, "wall_seconds": 100 / fps, "frames_dropped_lagging": dropped,
             "latency_ms_p95": p95 if clip == "a" else p95 / 2, "latency_ms_max": mx}]}))


def test_median_and_range_over_runs(tmp_path):
    _run(tmp_path, "cfg_k2_run1", 10.0, 150, 300, 4)
    _run(tmp_path, "cfg_k2_run2", 12.0, 120, 250, 2)
    _run(tmp_path, "cfg_k2_run3", 11.0, 180, 400, 0)
    _run(tmp_path, "other_run1", 5.0, 90, 100, 0)
    s = summarize_runs(tmp_path)
    g = s["cfg_k2"]
    assert g["runs"] == 3
    assert g["throughput_fps"] == pytest.approx({"median": 11.0, "min": 10.0, "max": 12.0})
    assert g["worst_clip_latency_ms_p95"] == {"median": 150, "min": 120, "max": 180}  # worst clip, not pooled
    assert g["latency_ms_max"]["max"] == 400 and g["frames_dropped"]["median"] == 4  # 2 clips x 2 dropped
    assert s["other"]["runs"] == 1

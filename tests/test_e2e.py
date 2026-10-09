"""End-to-end replay through the CLI on the synthetic clip, and broken inputs."""
import json
from pathlib import Path

import cv2
import numpy as np

from sightline.cli import main
from sightline.runner import run_clip

ROOT = Path(__file__).resolve().parent.parent
CORR = str(ROOT / "configs" / "corridors" / "synthetic.yaml")
CFG = str(ROOT / "configs" / "default.yaml")


def test_cli_replay_produces_events_and_files(clip, tmp_path):
    out = tmp_path / "o"
    rc = main(["replay", str(clip.video), "--config", CFG, "--corridor", CORR,
               "--detections-cache", str(clip.cache), "--out", str(out)])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else None
    assert s is not None
    assert len(s["events"]) == 2
    assert s["final_health"] == "OK"


def test_bytetrack_beats_baseline_on_id_stability(clip, cfg, corridor, cached):
    ids = {}
    for kind in ("iou", "bytetrack"):
        cfg.tracker.kind = kind
        run = run_clip(clip.video, cfg, corridor, cached=cached)
        ids[kind] = len({r[1] for r in run.mot_rows})
    assert ids["bytetrack"] <= ids["iou"]


def test_missing_source_is_unavailable(tmp_path, clip):
    rc = main(["replay", str(tmp_path / "nope.mp4"), "--config", CFG, "--corridor", CORR,
               "--detections-cache", str(clip.cache), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_garbage_file_is_unavailable(tmp_path, clip):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all" * 10)
    rc = main(["replay", str(bad), "--config", CFG, "--corridor", CORR,
               "--detections-cache", str(clip.cache), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_invalid_config_is_unavailable(tmp_path, clip):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("tracker:\n  bogus_key: 1\n")
    rc = main(["replay", str(clip.video), "--config", str(cfgp), "--corridor", CORR,
               "--detections-cache", str(clip.cache), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_bad_corridor_is_unavailable(tmp_path, clip):
    cp = tmp_path / "c.yaml"
    cp.write_text("polygon: [[0,0],[10,10]]\n")
    rc = main(["replay", str(clip.video), "--config", CFG, "--corridor", str(cp),
               "--detections-cache", str(clip.cache), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_corrupt_frames_never_report_clear(clip, cfg, corridor, cached, tmp_path):
    """Write a copy of the clip as an image sequence with a run of unreadable frames."""
    seq = tmp_path / "seq" / "img1"
    seq.mkdir(parents=True)
    cap = cv2.VideoCapture(str(clip.video))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        p = seq / f"{i:06d}.jpg"
        if 20 <= i <= 60:
            p.write_bytes(b"garbage")
        else:
            cv2.imwrite(str(p), f)
    (seq.parent / "seqinfo.ini").write_text(
        f"[Sequence]\nname=seq\nimDir=img1\nframeRate={int(clip.fps)}\nseqLength={i}\nimWidth={clip.size[0]}\nimHeight={clip.size[1]}\n")
    run = run_clip(seq.parent, cfg, corridor, cached=cached)
    assert run.health_counts.get("OK", 0) < run.frames_seen
    for f in range(20, 61):
        assert run.corridor_by_frame.get(f) != "CLEAR"

import json

from sightline.timeline import list_events, timeline


def _results(tmp_path, status="CLOSED", reason="left_corridor"):
    (tmp_path / "per_clip" / "bytetrack").mkdir(parents=True)
    (tmp_path / "tracks" / "bytetrack" / "c").mkdir(parents=True)
    ev = {"event_id": 2, "track_id": 7, "status": status, "close_reason": reason, "type": "crossing", "exit_edge": 1,
          "first_inside_frame": 12, "open_frame": 13, "last_inside_frame": 20, "close_frame": 22,
          "evidence": {"inside_detected": 3, "inside_predicted": 6, "opened_on_prediction": False}}
    (tmp_path / "per_clip" / "bytetrack" / "c.json").write_text(json.dumps({
        "clip": "c", "events_predicted": [ev], "event_pairs": [[4, 2]],
        "events_gt": [{"id": 4, "first_inside_frame": 11, "last_inside_frame": 21}]}))
    (tmp_path / "tracks" / "bytetrack" / "c" / "tracks.txt").write_text(
        "".join(f"{f},7,0,0,1,1,1,-1,-1,-1\n" for f in range(5, 25)))
    return tmp_path


def test_timeline_tells_the_event_from_approach_to_close_with_its_latency(tmp_path):
    lines = "\n".join(timeline(_results(tmp_path), "c", 2, fps=5.0))
    assert "frame     5" in lines and "first reported" in lines
    assert "latency 2 frames = 0.40 s" in lines and "CLOSED (left_corridor)" in lines
    assert list_events(tmp_path, "c") == [{"event_id": 2, "track_id": 7, "status": "CLOSED", "matched_gt_event": 4,
                                           "latency_frames": 2}]


def test_an_unresolved_event_does_not_claim_the_person_left(tmp_path):
    lines = "\n".join(timeline(_results(tmp_path, "UNRESOLVED", "track_lost"), "c", 2, fps=5.0))
    assert "UNRESOLVED (track_lost)" in lines and "does not claim the person left" in lines

"""Detection on a worker thread: other frames never wait for it, and late results are replayed."""
import hashlib
import time

from sightline.ingest import iter_frames
from sightline.runner import run_clip


class SlowDetector:
    """Returns the true cached detections for a frame (found by image content) after a fixed delay."""

    name = "slow-fake"

    def __init__(self, clip, cached, delay_s):
        self.delay_s = delay_s
        self.by_image = {hashlib.sha1(f.image.tobytes()).hexdigest(): cached.get(f.index, [])
                         for f in iter_frames(clip.video) if f.valid}
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.by_image[hashlib.sha1(image.tobytes()).hexdigest()]

    def info(self):
        return {"detector": self.name}


def test_other_frames_do_not_wait_for_a_slow_detector(clip, cfg, corridor, cached):
    # a live stream at the clip's 15 fps (67 ms per frame): the 50 ms detector keeps up with k = 3
    det = SlowDetector(clip, cached, delay_s=0.05)
    cfg.pipeline.detect_every, cfg.pipeline.async_detection = 3, True
    run = run_clip(clip.video, cfg, corridor, detector=det, keep_results=True, realtime=True, max_frames=45)
    assert run.policy["async_detection"] is True
    assert run.frames_seen == 45 == len(run.results)  # every frame processed, none dropped
    applied = [r for r in run.results if r.detection_applied_from is not None]
    assert applied and all(r.detection_delay_frames >= 1 for r in applied)  # results arrive later...
    assert all(r.detection_applied_from < r.frame_index for r in applied)
    # ...and no frame waited for the 50 ms detector
    assert max(r.latency_s for r in run.results) < 0.05
    skipped = sum(r.detection_skipped_busy for r in run.results)
    assert det.calls + skipped == len(range(1, 46, 3))  # every detector frame accounted for
    assert run.runtime["detections_skipped_worker_busy"] == skipped
    assert run.mot_rows  # replayed detections still produce reported tracks


def test_an_overloaded_worker_skips_detections_instead_of_blocking(clip, cfg, corridor, cached):
    # offline, frames come far faster than the detector: detections are skipped and counted, frames are not
    det = SlowDetector(clip, cached, delay_s=0.03)
    cfg.pipeline.detect_every, cfg.pipeline.async_detection = 3, True
    run = run_clip(clip.video, cfg, corridor, detector=det, keep_results=True)
    assert run.frames_seen == clip.n_frames and run.runtime["detections_skipped_worker_busy"] > 0


def test_sync_mode_waits_for_the_detector(clip, cfg, corridor, cached):
    det = SlowDetector(clip, cached, delay_s=0.02)
    cfg.pipeline.detect_every = 3
    run = run_clip(clip.video, cfg, corridor, detector=det, keep_results=True)
    assert run.policy["async_detection"] is False
    waited = [r.latency_s for r in run.results if r.detector_ran]
    assert min(waited) >= 0.02  # the detector frame itself blocks
    assert all(r.detection_delay_frames == 0 for r in run.results if r.detection_applied_from is not None)


def test_cached_detections_ignore_async_and_stay_deterministic(clip, cfg, corridor, cached):
    cfg.pipeline.detect_every = 2
    a = run_clip(clip.video, cfg, corridor, cached=cached)
    cfg.pipeline.async_detection = True
    b = run_clip(clip.video, cfg, corridor, cached=cached)
    assert b.policy["async_detection"] is False and a.mot_rows == b.mot_rows

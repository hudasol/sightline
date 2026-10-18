"""Detection on a worker thread: other frames never wait for it, and late results are replayed."""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import pytest

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


class GatedDetector:
    """Returns the true cached detections for a frame (found by image content), but only once the test opens
    the gate. While the gate is closed the detector is blocked inside detect(), indefinitely: a stand-in for a
    detector of any slowness, with no wall-clock threshold anywhere."""

    name = "gated-fake"

    def __init__(self, clip, cached):
        self.by_image = {hashlib.sha1(f.image.tobytes()).hexdigest(): cached.get(f.index, [])
                         for f in iter_frames(clip.video) if f.valid}
        self.gate = threading.Event()
        self.inside = threading.Event()  # set while a call is blocked in detect()
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        self.inside.set()
        if not self.gate.wait(timeout=60):  # a deadlock guard for a broken pipeline, not a performance limit
            raise RuntimeError("gate never opened")
        self.inside.clear()
        return self.by_image[hashlib.sha1(image.tobytes()).hexdigest()]

    def info(self):
        return {"detector": self.name}


def _process_in_thread(fn, timeout_s=60):
    """Run fn on a thread. If it does not finish, a frame waited for the blocked detector: fail, never hang."""
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=timeout_s)
    except FuturesTimeout:
        pytest.fail("frames waited for the detector: processing did not finish while it was blocked")
    finally:
        ex.shutdown(wait=False)


def test_other_frames_do_not_wait_for_a_slow_detector(clip, cfg, corridor, cached):
    """Deterministic: the detector is blocked for as long as the test likes. Every frame must still be processed
    while it is blocked, and its result must be applied later by replay. No wall-clock time is measured."""
    from sightline.ingest import probe
    from sightline.pipeline import Pipeline

    det = GatedDetector(clip, cached)
    cfg.pipeline.detect_every, cfg.pipeline.async_detection = 3, True
    info = probe(clip.video)
    pipe = Pipeline(cfg, corridor, det, (info.width, info.height), nominal_fps=info.fps)
    frames = [f for f in iter_frames(clip.video) if f.valid]
    first, rest = frames[:45], frames[45:]
    try:
        # 1. frame 1 goes to the worker, which blocks in detect(); wait for that event, then process 2-45
        results = [pipe.process(first[0])]
        assert det.inside.wait(timeout=60)
        results += _process_in_thread(lambda: [pipe.process(f) for f in first[1:]])
        assert det.inside.is_set() and not det.gate.is_set()  # still blocked inside detect() when the frames finished
        assert len(results) == 45 and all(r.processed for r in results)
        assert det.calls == 1  # only frame 1 was submitted...
        skipped = sum(r.detection_skipped_busy for r in results)
        assert skipped == len(range(1, 46, 3)) - 1  # ...every later detector frame was skipped, counted, not waited for
        assert all(r.detection_applied_from is None for r in results)  # nothing arrived while blocked
        # 2. open the gate; each later detection is collected before the next frame, so replay is deterministic
        det.gate.set()
        late = []
        for f in rest:
            pending = pipe.worker._pending
            if pending is not None:
                pending[1].result(timeout=60)  # wait for the worker, not for the clock
            late.append(pipe.process(f))
        applied = [r for r in late if r.detection_applied_from is not None]
        assert applied[0].detection_applied_from == 1  # frame 1's detections, applied late...
        assert applied[0].detection_delay_frames == applied[0].frame_index - 1 >= 1  # ...by replay, frames later
        assert all(r.detection_applied_from < r.frame_index for r in applied)
        assert any(r.reported_tracks for r in late)  # replayed detections still produce reported tracks
        if pipe.worker._pending is not None:
            pipe.worker._pending[1].result(timeout=60)
        every = results + late
        skipped_all = sum(r.detection_skipped_busy for r in every)
        assert det.calls + skipped_all == len(range(1, len(every) + 1, 3))  # every detector frame accounted for
    finally:
        det.gate.set()
        pipe.close()


def test_an_overloaded_worker_skips_detections_instead_of_blocking(clip, cfg, corridor, cached):
    """Deterministic, through run_clip: the gate opens only after the main thread has asked the clock for the
    time a fixed number of times (a counted event, not elapsed time). Until then the worker is blocked, so the
    detector frames in between are skipped, and the run still processes every frame."""
    det = GatedDetector(clip, cached)
    calls = {"n": 0}
    main = threading.main_thread()

    def counted_clock():
        if threading.current_thread() is main:
            calls["n"] += 1
            if calls["n"] == 120:  # well into the clip: dozens of frames processed while the worker is blocked
                det.gate.set()
        return calls["n"] / 1000.0  # a fake time, never compared to any threshold here

    cfg.pipeline.detect_every, cfg.pipeline.async_detection = 3, True
    run = run_clip(clip.video, cfg, corridor, detector=det, keep_results=True, clock=counted_clock)
    assert run.frames_seen == clip.n_frames == len(run.results)
    skipped = sum(r.detection_skipped_busy for r in run.results)
    assert skipped > 0 and run.runtime["detections_skipped_worker_busy"] == skipped
    # (exact calls + skipped accounting is in the test above: here the pool may cancel a last, unstarted submission)


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


def test_a_detector_slower_than_the_age_limit_turns_async_mode_degraded(clip, cfg, corridor, cached):
    # the per-frame time stays a few ms in async mode, so only the detection-age check can see this
    from sightline.types import CorridorStatus

    det = SlowDetector(clip, cached, delay_s=0.8)
    cfg.pipeline.detect_every, cfg.pipeline.async_detection = 1, True
    cfg.health.max_detection_age_s = 0.5
    run = run_clip(clip.video, cfg, corridor, detector=det, keep_results=True, realtime=True, max_frames=30)
    stale = [r for r in run.results if any(x.startswith("stale_detections") for x in r.health.reasons)]
    assert stale, "a 0.8 s detector must leave the corridor status resting on detections older than 0.5 s"
    assert all(r.corridor != CorridorStatus.CLEAR for r in stale)


def test_sync_mode_with_a_short_interval_never_reports_stale_detections(clip, cfg, corridor, cached):
    cfg.pipeline.detect_every = 3  # at 15 fps the oldest detection is 2/15 s old
    run = run_clip(clip.video, cfg, corridor, cached=cached, keep_results=True)
    assert not any(x.startswith("stale_detections") for r in run.results for x in r.health.reasons)

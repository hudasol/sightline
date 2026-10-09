import numpy as np

from sightline.config import HealthConfig
from sightline.health import HealthMonitor
from sightline.pipeline import Pipeline
from sightline.quality import measure_quality
from sightline.types import CorridorStatus, FrameData, HealthState

FPS = 15.0


def mon(**kw):
    base = dict(k_corrupt=3, corrupt_window=10, recover_frames=4, warmup_frames=2, fps_window=10, stall_seconds=2.0)
    base.update(kw)
    return HealthMonitor(HealthConfig(**base), nominal_fps=FPS)


def feed(m, n, start=0, **kw):
    rep = None
    for i in range(n):
        rep = m.observe(valid=kw.get("valid", True), ts=(start + i) / FPS, quality=kw.get("quality"),
                        proc_time_s=kw.get("proc", 0.01), det_scores=kw.get("scores"))
    return rep


def test_clean_stream_is_ok():
    assert feed(mon(), 30).state == HealthState.OK


def test_one_corrupt_frame_degrades_and_k_in_a_row_make_it_unavailable():
    m = mon()
    feed(m, 5)
    assert m.observe(valid=False, ts=5 / FPS).state == HealthState.DEGRADED
    m.observe(valid=False, ts=6 / FPS)
    rep = m.observe(valid=False, ts=7 / FPS)
    assert rep.state == HealthState.UNAVAILABLE and any("corrupt" in r for r in rep.reasons)


def test_recovery_needs_a_clean_run_and_steps_down_one_level_at_a_time():
    m = mon()
    feed(m, 3)
    for i in range(3, 6):
        m.observe(valid=False, ts=i / FPS)
    assert m.state == HealthState.UNAVAILABLE
    seen = []
    for i in range(6, 6 + 40):
        seen.append(m.observe(valid=True, ts=i / FPS, proc_time_s=0.01).state)
    assert seen[0] == HealthState.UNAVAILABLE  # one good frame is not enough
    assert HealthState.DEGRADED in seen  # steps down through DEGRADED
    assert seen.index(HealthState.DEGRADED) < seen.index(HealthState.OK)
    assert seen[-1] == HealthState.OK


def test_timestamp_gap_degrades_and_stall_is_unavailable():
    m = mon()
    feed(m, 10)
    assert m.observe(valid=True, ts=10 / FPS + 0.5, proc_time_s=0.01).state == HealthState.DEGRADED  # 7 frames missing
    m2 = mon()
    feed(m2, 10)
    rep = m2.observe(valid=True, ts=10 / FPS + 5.0, proc_time_s=0.01)
    assert rep.state == HealthState.UNAVAILABLE and "stream_stalled" in rep.reasons


def test_non_monotonic_timestamps_degrade():
    m = mon()
    feed(m, 10)
    rep = m.observe(valid=True, ts=2 / FPS, proc_time_s=0.01)
    assert rep.state == HealthState.DEGRADED and "non_monotonic_timestamp" in rep.reasons


def test_live_stall_with_no_frames_arriving_is_detected():
    m = mon()
    m.observe(valid=True, ts=0.0, proc_time_s=0.01, wall=100.0)
    assert m.check_stall(100.5) is None
    rep = m.check_stall(103.0)
    assert rep is not None and rep.state == HealthState.UNAVAILABLE


def test_low_fps_degrades_after_warmup():
    m = mon(min_fps=10.0)
    rep = feed(m, 20, proc=0.5)  # 2 fps
    assert rep.state == HealthState.DEGRADED and any(r.startswith("low_fps") for r in rep.reasons)


def test_dark_and_blurry_frames_degrade():
    dark = np.full((180, 320, 3), 5, np.uint8)
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    flat = np.full((180, 320, 3), 128, np.uint8)
    m = mon()
    assert m.observe(valid=True, ts=0, quality=measure_quality(dark), proc_time_s=0.01).state == HealthState.DEGRADED
    assert measure_quality(sharp).sharpness > 1000 and measure_quality(flat).sharpness == 0
    m2 = mon()
    rep = m2.observe(valid=True, ts=0, quality=measure_quality(flat), proc_time_s=0.01)
    assert rep.state == HealthState.DEGRADED and "blurry_or_low_contrast" in rep.reasons


def test_persistently_low_detector_confidence_degrades():
    m = mon(low_conf_mean=0.4, low_conf_window=10)
    rep = feed(m, 15, scores=[0.2, 0.25])
    assert rep.state == HealthState.DEGRADED and "low_detection_confidence" in rep.reasons


def test_fail_latches_unavailable():
    m = mon()
    m.fail("invalid configuration")
    assert feed(m, 5).state == HealthState.UNAVAILABLE


def test_quality_is_resolution_independent():
    rng = np.random.default_rng(1)
    small = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    import cv2
    big = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_AREA)
    assert abs(measure_quality(small).luminance - measure_quality(big).luminance) < 2


# -- the pipeline-level promise: never CLEAR unless health is OK ------------------------------

def textured(seed=0, w=320, h=180):
    return np.random.default_rng(seed).integers(60, 160, (h, w, 3), dtype=np.uint8)


def pipe(cfg, corridor, clock=None):
    kw = {"clock": clock} if clock else {}
    return Pipeline(cfg, corridor, None, (640, 360), nominal_fps=FPS, **kw)


def test_empty_healthy_scene_is_clear(cfg, corridor):
    p = pipe(cfg, corridor)
    res = [p.process(FrameData(i, i / FPS, textured(i), True), detections=[]) for i in range(1, 30)]
    assert res[-1].corridor == CorridorStatus.CLEAR


def test_corrupt_frames_never_report_clear(cfg, corridor):
    p = pipe(cfg, corridor)
    for i in range(1, 10):
        p.process(FrameData(i, i / FPS, textured(i), True), detections=[])
    for i in range(10, 20):
        r = p.process(FrameData(i, i / FPS, None, False), detections=[])
        assert r.corridor != CorridorStatus.CLEAR and r.health.state != HealthState.OK


def test_dark_input_never_reports_clear(cfg, corridor):
    p = pipe(cfg, corridor)
    dark = np.full((180, 320, 3), 3, np.uint8)
    for i in range(1, 20):
        r = p.process(FrameData(i, i / FPS, dark, True), detections=[])
        assert r.corridor == CorridorStatus.UNKNOWN


def test_slow_processing_never_reports_clear(cfg, corridor):
    t = [0.0]

    def slow_clock():
        t[0] += 0.3  # every clock read costs 300 ms: about 3 fps
        return t[0]

    p = pipe(cfg, corridor, slow_clock)
    statuses = []
    for i in range(1, 40):
        r = p.process(FrameData(i, i / FPS, textured(i), True), detections=[])
        statuses.append((r.health.state, r.corridor))
    degraded = [s for s in statuses if s[0] != HealthState.OK]
    assert degraded and all(c != CorridorStatus.CLEAR for _s, c in degraded)

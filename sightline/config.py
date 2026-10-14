"""Configuration: typed, validated, and loud about invalid input.

An invalid config must stop the system (UNAVAILABLE), never fall back to
silent defaults.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from sightline.geometry import GeometryError, Polygon


class ConfigError(ValueError):
    pass


@dataclass
class DetectorConfig:
    name: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    # FROZEN-ON-VAL: the operating point for detection P/R/F1; also the tracker's "high" split
    score_threshold: float = 0.5
    # floor: detections below this are never produced, so the tracker's low stage has something to use
    min_score: float = 0.1
    # [width, height] box every frame is resized to fit inside (aspect kept, never upscaled) before
    # detection; boxes are mapped back to original pixels. None = the pipeline's declared size.
    input_size: list[int] | None = None
    device: str = "cpu"
    # intra-op CPU threads (torch, or ONNX Runtime for ONNX models); None keeps the runtime's default.
    # Measured with `sightline threads`.
    torch_threads: int | None = None
    # ONNX models only: path to the .onnx file (default weights/yolox/<name>.onnx). Git-ignored.
    weights_path: str | None = None


@dataclass
class TrackerConfig:
    kind: str = "bytetrack"  # "bytetrack" | "iou"
    new_track_score: float = 0.6
    match_iou: float = 0.3  # minimum IoU for first-stage association
    low_match_iou: float = 0.5  # minimum IoU for the low-confidence stage
    tentative_match_iou: float = 0.5
    max_age: int = 30  # detector frames a track may go unmatched before it expires
    min_hits: int = 3  # detector hits before a tentative track is confirmed (evidence, so a count, not time)
    trajectory_len: int = 30
    # time-based alternative (seconds); when set it wins over max_age and is converted per clip into
    # DETECTOR frames: max_age = max_age_s * fps / detect_every, so patience in seconds is the same at any k
    max_age_s: float | None = None


@dataclass
class EventConfig:
    n_open: int = 3  # consecutive inside frames before an event opens
    m_close: int = 5  # consecutive outside frames before it closes
    cooldown: int = 15  # re-entry within this many frames reopens the SAME event
    edge_margin_frac: float = 0.05  # lost near the frame border counts as "left view"
    # time-based alternatives (seconds); when set they win over the frame counts above and are
    # converted per clip at its own fps (every frame is processed, detected or predicted)
    open_s: float | None = None
    close_s: float | None = None
    cooldown_s: float | None = None


@dataclass
class HealthConfig:
    min_fps: float = 10.0
    gap_factor: float = 3.0  # timestamp gap > gap_factor * nominal interval -> DEGRADED
    stall_seconds: float = 2.0  # gap longer than this -> UNAVAILABLE
    k_corrupt: int = 5  # consecutive undecodable frames -> UNAVAILABLE
    corrupt_window: int = 30  # any undecodable frame in this many frames -> DEGRADED
    recover_frames: int = 15  # clean frames needed to step health back down one level
    min_luminance: float = 35.0  # mean gray level (0-255); FROZEN-ON-VAL
    min_sharpness: float = 25.0  # Laplacian variance on a 320px-wide image; FROZEN-ON-VAL
    low_conf_window: int = 30
    low_conf_mean: float = 0.35  # mean detection score over the window; FROZEN-ON-VAL
    fps_window: int = 30
    warmup_frames: int = 5  # frames excluded from the fps check while the model warms up


@dataclass
class PipelineConfig:
    # PLAN.md A2: the detector runs on every k-th frame; the tracker predicts in between.
    # Every frame is still processed. k > 1 is a recorded sampling policy.
    detect_every: int = 1
    # run the detector on a worker thread so other frames never wait for it (sightline/async_detect.py).
    # Live detector only: evaluation from cached detections always runs synchronously and deterministically.
    async_detection: bool = False
    declared_width: int = 640
    declared_height: int = 360
    decode_thread: bool = True  # decode frames in a background thread so decoding overlaps inference
    decode_queue: int = 4  # frames decoded ahead; order kept, nothing dropped


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    # the simple geometric baseline, tuned on the same grid so the comparison is not a straw man
    baseline_tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(kind="iou"))
    events: EventConfig = field(default_factory=EventConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def validate(self) -> "Config":
        d, t, e, h, p = self.detector, self.tracker, self.events, self.health, self.pipeline
        _validate_tracker(self.baseline_tracker, "baseline_tracker")
        _check(0.0 < d.min_score < d.score_threshold <= 1.0, "detector: need 0 < min_score < score_threshold <= 1")
        _check(p.declared_width > 0 and p.declared_height > 0, "pipeline declared size must be positive")
        if d.input_size is None:  # resolved here so every consumer (detector, cache, reports) sees one value
            d.input_size = [p.declared_width, p.declared_height]
        _check(isinstance(d.input_size, (list, tuple)) and len(d.input_size) == 2
               and all(isinstance(v, int) and v >= 32 for v in d.input_size),
               "detector.input_size must be null or [width, height] with integers >= 32")
        d.input_size = [int(v) for v in d.input_size]
        _check(d.torch_threads is None or (isinstance(d.torch_threads, int) and d.torch_threads >= 1),
               "detector.torch_threads must be null or an integer >= 1")
        _validate_tracker(t, "tracker")
        _check(e.n_open >= 1, "events.n_open must be >= 1")
        _check(e.m_close >= 1, "events.m_close must be >= 1")
        _check(e.cooldown >= 0, "events.cooldown must be >= 0")
        _check(0.0 <= e.edge_margin_frac < 0.5, "events.edge_margin_frac must be in [0, 0.5)")
        for name in ("open_s", "close_s"):
            v = getattr(e, name)
            _check(v is None or (isinstance(v, (int, float)) and v > 0), f"events.{name} must be null or > 0 seconds")
        _check(e.cooldown_s is None or (isinstance(e.cooldown_s, (int, float)) and e.cooldown_s >= 0),
               "events.cooldown_s must be null or >= 0 seconds")
        _check(h.min_fps > 0, "health.min_fps must be > 0")
        _check(h.gap_factor > 1, "health.gap_factor must be > 1")
        _check(h.stall_seconds > 0, "health.stall_seconds must be > 0")
        _check(h.k_corrupt >= 1, "health.k_corrupt must be >= 1")
        _check(h.recover_frames >= 1, "health.recover_frames must be >= 1")
        _check(0 <= h.min_luminance <= 255, "health.min_luminance must be in [0, 255]")
        _check(h.min_sharpness >= 0, "health.min_sharpness must be >= 0")
        _check(0.0 <= h.low_conf_mean <= 1.0, "health.low_conf_mean must be in [0, 1]")
        _check(isinstance(p.detect_every, int) and p.detect_every >= 1, "pipeline.detect_every must be an integer >= 1")
        _check(p.decode_queue >= 1, "pipeline.decode_queue must be >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(yaml.safe_dump(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def _validate_tracker(t: "TrackerConfig", label: str) -> None:
    _check(t.kind in ("bytetrack", "iou"), f"{label}.kind must be 'bytetrack' or 'iou', got {t.kind!r}")
    for name in ("match_iou", "low_match_iou", "tentative_match_iou"):
        _check(0.0 < getattr(t, name) <= 1.0, f"{label}.{name} must be in (0, 1]")
    _check(0.0 < t.new_track_score <= 1.0, f"{label}.new_track_score must be in (0, 1]")
    _check(t.max_age >= 1, f"{label}.max_age must be >= 1")
    _check(t.min_hits >= 1, f"{label}.min_hits must be >= 1")
    _check(t.trajectory_len >= 2, f"{label}.trajectory_len must be >= 2")
    _check(t.max_age_s is None or (isinstance(t.max_age_s, (int, float)) and t.max_age_s > 0),
           f"{label}.max_age_s must be null or > 0 seconds")


def _frames(seconds: float, fps: float) -> int:
    """Seconds -> frames, rounding half UP (Python's round() would send 2.5 to 2)."""
    return int(math.floor(seconds * fps + 0.5))


def resolve_timing(cfg: "Config", fps: float, detect_every: int | None = None) -> "Config":
    """A copy of cfg whose frame counts come from its time-based settings at this clip's fps.

    Events count every processed frame (detected or predicted), so open/close/cooldown use fps.
    Tracker misses count DETECTOR frames, so max_age uses fps / detect_every. Settings left as
    null keep their frame counts. The result is what actually ran; runs record it.
    """
    out = copy.deepcopy(cfg)
    k = detect_every or cfg.pipeline.detect_every
    e = out.events
    if e.open_s is not None:
        e.n_open = max(1, _frames(e.open_s, fps))
    if e.close_s is not None:
        e.m_close = max(1, _frames(e.close_s, fps))
    if e.cooldown_s is not None:
        e.cooldown = max(0, _frames(e.cooldown_s, fps))
    for t in (out.tracker, out.baseline_tracker):
        if t.max_age_s is not None:
            t.max_age = max(1, _frames(t.max_age_s, fps / k))
    return out


def timing_summary(cfg: "Config") -> dict:
    return {"n_open": cfg.events.n_open, "m_close": cfg.events.m_close, "cooldown": cfg.events.cooldown,
            "max_age_detector_frames": cfg.tracker.max_age, "min_hits": cfg.tracker.min_hits,
            "baseline_max_age_detector_frames": cfg.baseline_tracker.max_age,
            "detect_every": cfg.pipeline.detect_every}


def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{path}: {exc}") from exc


def config_from_dict(data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    sections = {"detector": DetectorConfig, "tracker": TrackerConfig, "baseline_tracker": TrackerConfig,
                "events": EventConfig, "health": HealthConfig, "pipeline": PipelineConfig}
    # provenance blocks written by `tune` are allowed and ignored here
    extra = set(data) - set(sections) - {"provenance"}
    if extra:
        raise ConfigError(f"unknown top-level keys {sorted(extra)}")
    built = {name: _build(cls, data.get(name), name) for name, cls in sections.items()}
    if "baseline_tracker" not in data:
        built["baseline_tracker"] = TrackerConfig(kind="iou")
    return Config(**built).validate()


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"config is not valid YAML: {exc}") from exc
    return config_from_dict(data or {})


@dataclass
class CorridorConfig:
    name: str
    polygon: Polygon
    frame_size: tuple[int, int] | None = None


def load_corridor(path: str | Path) -> CorridorConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"corridor file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"corridor is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "polygon" not in data:
        raise ConfigError("corridor file needs a 'polygon' list")
    size = data.get("frame_size")
    if size is not None:
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise ConfigError("corridor frame_size must be [width, height]")
        size = (int(size[0]), int(size[1]))
    try:
        poly = Polygon.create(data["polygon"], frame_size=size)
    except GeometryError as exc:
        raise ConfigError(f"invalid corridor polygon: {exc}") from exc
    return CorridorConfig(name=str(data.get("name", p.stem)), polygon=poly, frame_size=size)

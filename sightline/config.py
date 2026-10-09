"""Configuration: typed, validated, and loud about invalid input.

An invalid config must stop the system (UNAVAILABLE), never fall back to
silent defaults.
"""

from __future__ import annotations

import hashlib
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
    input_short_side: int = 320
    device: str = "cpu"


@dataclass
class TrackerConfig:
    kind: str = "bytetrack"  # "bytetrack" | "iou"
    new_track_score: float = 0.6
    match_iou: float = 0.3  # minimum IoU for first-stage association
    low_match_iou: float = 0.5  # minimum IoU for the low-confidence stage
    tentative_match_iou: float = 0.5
    max_age: int = 30  # processed frames a track may go unmatched before it expires
    min_hits: int = 3  # hits before a tentative track is confirmed
    trajectory_len: int = 30


@dataclass
class EventConfig:
    n_open: int = 3  # consecutive inside frames before an event opens
    m_close: int = 5  # consecutive outside frames before it closes
    cooldown: int = 15  # re-entry within this many frames reopens the SAME event
    edge_margin_frac: float = 0.05  # lost near the frame border counts as "left view"


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
    frame_stride: int = 1  # 1 = process every frame. Anything else is a recorded sampling policy.
    declared_width: int = 640
    declared_height: int = 360


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
        _check(d.input_short_side > 0, "detector.input_short_side must be positive")
        _validate_tracker(t, "tracker")
        _check(e.n_open >= 1, "events.n_open must be >= 1")
        _check(e.m_close >= 1, "events.m_close must be >= 1")
        _check(e.cooldown >= 0, "events.cooldown must be >= 0")
        _check(0.0 <= e.edge_margin_frac < 0.5, "events.edge_margin_frac must be in [0, 0.5)")
        _check(h.min_fps > 0, "health.min_fps must be > 0")
        _check(h.gap_factor > 1, "health.gap_factor must be > 1")
        _check(h.stall_seconds > 0, "health.stall_seconds must be > 0")
        _check(h.k_corrupt >= 1, "health.k_corrupt must be >= 1")
        _check(h.recover_frames >= 1, "health.recover_frames must be >= 1")
        _check(0 <= h.min_luminance <= 255, "health.min_luminance must be in [0, 255]")
        _check(h.min_sharpness >= 0, "health.min_sharpness must be >= 0")
        _check(0.0 <= h.low_conf_mean <= 1.0, "health.low_conf_mean must be in [0, 1]")
        _check(p.frame_stride >= 1, "pipeline.frame_stride must be >= 1")
        _check(p.declared_width > 0 and p.declared_height > 0, "pipeline declared size must be positive")
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

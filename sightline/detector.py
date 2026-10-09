"""Person detection. Pretrained only: nothing here trains a model.

The detector returns every person box above a low floor (cfg.min_score). The
FROZEN operating threshold (cfg.score_threshold) is applied downstream: it is
the P/R/F1 operating point and the tracker's high/low split. Keeping the floor
low is what lets the tracker's second stage use low-confidence boxes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from sightline.config import DetectorConfig
from sightline.types import Detection


class DetectorUnavailable(RuntimeError):
    pass


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...

    def info(self) -> dict: ...


# model name -> (torchvision builder, weights enum name)
_TORCHVISION_MODELS = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": ("fasterrcnn_mobilenet_v3_large_320_fpn", "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights"),
    "fasterrcnn_mobilenet_v3_large_fpn": ("fasterrcnn_mobilenet_v3_large_fpn", "FasterRCNN_MobileNet_V3_Large_FPN_Weights"),
    "fasterrcnn_resnet50_fpn_v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
}


class TorchvisionPersonDetector:
    """COCO-pretrained torchvision detector, person class only (COCO id 1).

    Code: BSD-3-Clause (torchvision). Weights: trained on COCO train2017; record the
    exact weights name below in the model card.
    """

    PERSON_LABEL = 1

    def __init__(self, cfg: DetectorConfig, load_weights: bool = True) -> None:
        try:
            import torch
            import torchvision
        except ImportError as exc:
            raise DetectorUnavailable(
                "torch/torchvision are not installed: pip install -e '.[detector]'"
            ) from exc
        if cfg.name not in _TORCHVISION_MODELS:
            raise DetectorUnavailable(f"unknown detector {cfg.name!r}; choose from {sorted(_TORCHVISION_MODELS)}")
        self._torch = torch
        self.cfg = cfg
        self.name = cfg.name
        builder_name, weights_name = _TORCHVISION_MODELS[cfg.name]
        builder = getattr(torchvision.models.detection, builder_name)
        weights = getattr(torchvision.models.detection, weights_name).DEFAULT if load_weights else None
        self._weights_name = f"{weights_name}.DEFAULT" if load_weights else "none (random init)"
        self.model = builder(
            weights=weights, weights_backbone=None, box_score_thresh=cfg.min_score,
            min_size=cfg.input_short_side, max_size=int(cfg.input_short_side * 16 / 9) + 1,
        )
        self.model.eval().to(cfg.device)
        self._versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}

    def detect(self, image: np.ndarray) -> list[Detection]:
        torch = self._torch
        rgb = image[:, :, ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0).to(self.cfg.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        h, w = image.shape[:2]
        dets = []
        for box, label, score in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy()):
            if int(label) != self.PERSON_LABEL or float(score) < self.cfg.min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                dets.append(Detection((x1, y1, x2, y2), float(score)))
        return dets

    def info(self) -> dict:
        return {"detector": self.name, "weights": self._weights_name, "min_score": self.cfg.min_score,
                "input_short_side": self.cfg.input_short_side, "device": self.cfg.device, **self._versions}


# -- detection cache ---------------------------------------------------------------
#
# Running the detector once per clip and caching its raw output (above the floor)
# lets threshold sweeps and tracker tuning reuse it, and makes evaluation
# deterministic. The cache header records which model produced it; a cache from a
# different model/floor is refused.


def write_detection_cache(path: str | Path, meta: dict, per_frame: dict[int, list[Detection]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for frame in sorted(per_frame):
            dets = [[*[round(v, 2) for v in d.box], round(d.score, 4)] for d in per_frame[frame]]
            f.write(json.dumps({"frame": frame, "dets": dets}) + "\n")


def read_detection_cache(path: str | Path) -> tuple[dict, dict[int, list[Detection]]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    meta: dict = {}
    per_frame: dict[int, list[Detection]] = {}
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            per_frame[int(row["frame"])] = [Detection(tuple(d[:4]), float(d[4])) for d in row["dets"]]
    return meta, per_frame


class CachedDetector:
    """Replays cached detections by frame index. Not a model; used for tuning/eval speed."""

    def __init__(self, meta: dict, per_frame: dict[int, list[Detection]]) -> None:
        self.name = f"cache:{meta.get('detector', 'unknown')}"
        self._meta = meta
        self._frames = per_frame

    def detections_for(self, frame_index: int) -> list[Detection]:
        return self._frames.get(frame_index, [])

    def detect(self, image):  # pragma: no cover - cached detectors are queried by index
        raise RuntimeError("CachedDetector is queried by frame index via detections_for()")

    def info(self) -> dict:
        return {**self._meta, "source": "cache"}
